from src.gradio_interface import launch_interface
from src.server_api import app as server_api_app
import uvicorn
import asyncio

if __name__ == "__main__":
    async def main():
        # Run Gradio in a separate task
        gradio_task = asyncio.create_task(asyncio.to_thread(launch_interface))

        # Run FastAPI server
        # Note: uvicorn.run is blocking, so we need to run it in a separate process or thread
        # For simplicity here, we'll use a direct call, but for production, consider
        # using a process manager or a more robust way to run multiple servers.
        uvicorn_config = uvicorn.Config(server_api_app, host="0.0.0.0", port=8000, log_level="info")
        server_task = asyncio.create_task(uvicorn.Server(uvicorn_config).serve())
        await asyncio.gather(gradio_task, server_task)
    asyncio.run(main())
