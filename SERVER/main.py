from src.gradio_interface import launch_interface
from src.server_api import app as server_api_app
import uvicorn
import multiprocessing

def run_gradio():
    """Target function for the Gradio process."""
    # This function should contain the logic to start the Gradio interface.
    # It's assumed launch_interface() is a blocking call.
    launch_interface()

def run_fastapi():
    """Target function for the FastAPI process."""
    # uvicorn.run is a blocking call that starts the server.
    uvicorn.run(server_api_app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    # The note in the original code suggested a more robust way to run multiple servers
    # for production, such as using a process manager or running them in separate processes.
    # The implementation below uses Python's `multiprocessing` module to run each
    # server in its own process. This is more robust than using threads due to the GIL
    # and provides better CPU utilization on multi-core systems.

    print("Starting Gradio and FastAPI servers in separate processes...")

    # Create a process for the Gradio interface
    gradio_process = multiprocessing.Process(target=run_gradio, name="GradioInterface")

    # Create a process for the FastAPI server
    fastapi_process = multiprocessing.Process(target=run_fastapi, name="FastAPIServer")

    # Start both processes
    gradio_process.start()
    fastapi_process.start()

    print(f"Gradio process started with PID: {gradio_process.pid}")
    print(f"FastAPI process started with PID: {fastapi_process.pid}")

    # Wait for both processes to complete and handle graceful shutdown.
    try:
        gradio_process.join()
        fastapi_process.join()
    except KeyboardInterrupt:
        print("\nShutting down servers.")
        gradio_process.terminate()
        fastapi_process.terminate()
        gradio_process.join()
        fastapi_process.join()
        print("Servers shut down.")
