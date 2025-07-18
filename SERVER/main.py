import sys
import signal
from src.gradio_interface import launch_interface
from src.server_api import app as server_api_app
import uvicorn
import multiprocessing

def run_gradio():
    """
    Starts the Gradio server by launching the Gradio interface in a separate process.
    """
    launch_interface()

<<<<<<< HEAD
def run_fastapi() -> None:
    """
    Starts the FastAPI server using Uvicorn on host 0.0.0.0 and port 8000 with info-level logging.
    """
    uvicorn.run(server_api_app, host="0.0.0.0", port=8000, log_level="info")

def terminate_process(proc: multiprocessing.Process, name: str) -> None:
    """
    Attempt to gracefully terminate a multiprocessing process, forcefully killing it if necessary.
    
    Parameters:
        proc (multiprocessing.Process): The process to terminate.
        name (str): A human-readable name for the process, used in status messages.
    """
=======
def run_fastapi():
    """
    Starts the FastAPI server using Uvicorn on host 0.0.0.0 and port 8000 with info-level logging.
    """
    uvicorn.run(server_api_app, host="0.0.0.0", port=8000, log_level="info")

def terminate_process(proc, name):
    """
    Attempt to gracefully terminate a multiprocessing process, forcefully killing it if necessary.
    
    Parameters:
        proc (multiprocessing.Process): The process to terminate.
        name (str): A human-readable name for the process, used in status messages.
    """
>>>>>>> 96fc77cc2cce22f1a9028bf0e9399df2f81b2e3d
    if proc.is_alive():
        print(f"Terminating {name} (PID: {proc.pid})...")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            print(f"{name} did not terminate gracefully, killing...")
            proc.kill()
        else:
            print(f"{name} terminated.")

def main():
    # Use 'spawn' for Windows safety and cross-platform compatibility
    """
    Start and manage Gradio and FastAPI servers in separate processes with robust lifecycle and shutdown handling.
    
    This function initializes multiprocessing with the 'spawn' method for cross-platform compatibility, launches the Gradio and FastAPI servers in independent processes, and monitors their status. It registers signal handlers to ensure both servers are terminated gracefully on SIGINT or SIGTERM, and handles unexpected process exits or exceptions by shutting down both servers before exiting.
    """
    multiprocessing.set_start_method("spawn", force=True)
    print("Starting Gradio and FastAPI servers in separate processes...")

    gradio_process = multiprocessing.Process(target=run_gradio, name="GradioInterface")
    fastapi_process = multiprocessing.Process(target=run_fastapi, name="FastAPIServer")

    gradio_process.start()
    fastapi_process.start()

    print(f"Gradio process started with PID: {gradio_process.pid}")
    print(f"FastAPI process started with PID: {fastapi_process.pid}")

    def shutdown_handler(signum, frame):
        """
        Handles shutdown signals by terminating both the Gradio and FastAPI server processes and exiting the program.
        
        Parameters:
            signum (int): The received signal number.
            frame (FrameType): The current stack frame when the signal was received.
        """
        print(f"\nReceived signal {signum}. Shutting down servers...")
        terminate_process(gradio_process, "GradioInterface")
        terminate_process(fastapi_process, "FastAPIServer")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        while True:
            if not gradio_process.is_alive():
                print("Gradio process exited unexpectedly. Shutting down FastAPI.")
                terminate_process(fastapi_process, "FastAPIServer")
                break
            if not fastapi_process.is_alive():
                print("FastAPI process exited unexpectedly. Shutting down Gradio.")
                terminate_process(gradio_process, "GradioInterface")
                break
            gradio_process.join(timeout=1)
            fastapi_process.join(timeout=1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down servers.")
        terminate_process(gradio_process, "GradioInterface")
        terminate_process(fastapi_process, "FastAPIServer")
    except Exception as e:
        print(f"Unexpected error: {e}. Shutting down servers.")
        terminate_process(gradio_process, "GradioInterface")
        terminate_process(fastapi_process, "FastAPIServer")
    finally:
        print("Servers shut down.")

if __name__ == "__main__":
    main()
