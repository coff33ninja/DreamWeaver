import sys
import signal
from src.gradio_interface import launch_interface
from src.server_api import app as server_api_app
import uvicorn
import multiprocessing

def run_gradio():
    """
    Starts the Gradio interface server in a separate process.
    """
    launch_interface()

def run_fastapi():
    """
    Starts the FastAPI server using Uvicorn on host 0.0.0.0 and port 8000 with info-level logging.
    """
    uvicorn.run(server_api_app, host="0.0.0.0", port=8000, log_level="info")

def terminate_process(proc, name):
    """
    Attempt to gracefully terminate a process, and forcibly kill it if it does not exit within 5 seconds.
    
    Parameters:
        proc: The process to terminate.
        name: A human-readable name for the process, used in status messages.
    """
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
    Starts and manages Gradio and FastAPI servers in separate processes, handling their lifecycle and graceful shutdown.
    
    This function launches both servers as independent OS-level processes, monitors their status, and ensures that if either server exits unexpectedly or a termination signal is received, both processes are properly terminated. It registers signal handlers for SIGINT and SIGTERM to support graceful shutdown and provides informative output throughout the server lifecycle.
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
        Handles termination signals by shutting down both the Gradio and FastAPI server processes and exiting the program.
        
        Parameters:
            signum (int): The received signal number.
            frame (FrameType): The current stack frame (unused).
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
