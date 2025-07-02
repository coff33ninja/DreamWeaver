import sys
import os
import signal

# Add the project root (SERVER directory) to the Python path
# This ensures that 'src' can be imported as a module and helps linters find it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import multiprocessing
from src.gradio_interface import launch_interface
from src.server_api import app as server_api_app
# Call this early, but it will be more effective if called inside each process's target function
# if they don't inherit the main process's logging config correctly due to 'spawn'.
from src.logging_config import setup_logging, get_logger


def run_gradio():
    """
    Starts the Gradio server by launching the Gradio interface in a separate process.
    """
    setup_logging()  # Setup logging for the Gradio process
    logger = get_logger()
    logger.info("Starting Gradio interface process...")
    launch_interface()


def run_fastapi() -> None:
    """
    Starts the FastAPI server using Uvicorn on host 0.0.0.0 and port 8000 with info-level logging.
    """
    setup_logging()  # Setup logging for the FastAPI process
    logger = get_logger()
    logger.info("Starting FastAPI server process...")
    uvicorn.run(server_api_app, host="0.0.0.0", port=8000, log_level="info")


def terminate_process(proc: multiprocessing.Process, name: str) -> None:
    """
    Attempt to gracefully terminate a multiprocessing process, forcefully killing it if necessary.

    Parameters:
        proc (multiprocessing.Process): The process to terminate.
        name (str): A human-readable name for the process, used in status messages.
    """
    logger = get_logger()
    if proc.is_alive():
        logger.info(f"Terminating {name} (PID: {proc.pid})...")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning(f"{name} did not terminate gracefully, killing...")
            proc.kill()
        else:
            logger.info(f"{name} terminated.")


def main():
    """
    Start and manage Gradio and FastAPI servers in separate processes with robust lifecycle and shutdown handling.

    This function initializes multiprocessing with the 'spawn' method for cross-platform compatibility, launches the Gradio and FastAPI servers in independent processes, and monitors their status. It registers signal handlers to ensure both servers are terminated gracefully on SIGINT or SIGTERM, and handles unexpected process exits or exceptions by shutting down both servers before exiting.
    """
    # Call setup_logging here for the main process
    setup_logging()
    logger = get_logger()

    # Use 'spawn' for Windows safety and cross-platform compatibility
    multiprocessing.set_start_method("spawn", force=True)  # Must be called only once
    logger.info("Starting Gradio and FastAPI servers in separate processes...")

    gradio_process = multiprocessing.Process(target=run_gradio, name="GradioInterface")
    fastapi_process = multiprocessing.Process(target=run_fastapi, name="FastAPIServer")

    gradio_process.start()
    fastapi_process.start()

    logger.info(f"Gradio process started with PID: {gradio_process.pid}")
    logger.info(f"FastAPI process started with PID: {fastapi_process.pid}")

    def shutdown_handler(signum, _frame):
        """
        Handles shutdown signals by terminating both the Gradio and FastAPI server processes and exiting the program.

        Parameters:
            signum (int): The received signal number.
            frame (FrameType): The current stack frame when the signal was received.
        """
        logger.info(f"\nReceived signal {signum}. Shutting down servers...")
        terminate_process(gradio_process, "GradioInterface")
        terminate_process(fastapi_process, "FastAPIServer")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown_handler)
    else:
        logger.warning(
            "SIGTERM signal not available on this platform. Graceful shutdown might be limited to Ctrl+C (SIGINT)."
        )

    try:
        while True:
            if not gradio_process.is_alive():
                logger.error(
                    "Gradio process exited unexpectedly. Shutting down FastAPI."
                )
                terminate_process(fastapi_process, "FastAPIServer")
                break
            if not fastapi_process.is_alive():
                logger.error(
                    "FastAPI process exited unexpectedly. Shutting down Gradio."
                )
                terminate_process(gradio_process, "GradioInterface")
                break
            # Using join with a timeout acts like a non-blocking check combined with sleep
            gradio_process.join(timeout=0.5)  # Check every 0.5 seconds
            fastapi_process.join(timeout=0.5)  # Check every 0.5 seconds

            # If both are still alive, loop will continue. If one exited, the other join will timeout
            # and the is_alive() check at the start of the loop will catch it.
            # This also makes the loop more responsive to signals than long joins.
            if not gradio_process.is_alive() or not fastapi_process.is_alive():
                break  # Exit if any process has died

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received. Shutting down servers.")
        # shutdown_handler will be called by the signal
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Attempting final shutdown of server processes...")
        terminate_process(
            gradio_process, "GradioInterface"
        )  # Ensure termination on any exit
        terminate_process(
            fastapi_process, "FastAPIServer"
        )  # Ensure termination on any exit
        logger.info("Servers shut down procedure complete.")


if __name__ == "__main__":
    main()
