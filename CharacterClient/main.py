import argparse
import os
import uvicorn
from dotenv import load_dotenv
import logging # Import logging
from src.character_client import app, initialize_character_client, start_heartbeat_task
from src.logging_config import setup_client_logging, get_logger # Import client logging setup

def main():
    """
    Loads environment variables from .env, parses command-line arguments,
    initializes the DreamWeaver Character Client, and starts the FastAPI server.
    
    Configuration is resolved in the following order of precedence:
    1. Command-line arguments.
    2. Environment variables (which can be loaded from a .env file).
    3. Default values specified in argparse.

    This function validates required parameters (especially the token),
    initializes the character client instance, starts a heartbeat background task
    if initialization succeeds, and launches the FastAPI app using Uvicorn.
    If initialization fails or the authentication token is missing, the server does not start.
    """
    # Load environment variables from .env file if it exists.
    # This should be called before argparse reads os.getenv for defaults,
    # so .env values can act as defaults if not overridden by CLI args.
    load_dotenv()

    # Setup client-side logging
    setup_client_logging()
    logger = get_logger() # Get the configured logger

    logger.info("DreamWeaver Character Client starting...")

    parser = argparse.ArgumentParser(description="DreamWeaver Character Client")
    parser.add_argument(
        "--server_url",
        type=str,
        default=os.getenv("SERVER_URL", "http://127.0.0.1:8000"),
        help="URL of the DreamWeaver Server (default: env SERVER_URL or http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--Actor_id",
        type=str,
        default=os.getenv("CLIENT_Actor_ID", "Actor_Client"),
        help="Unique ID for this client (default: env CLIENT_Actor_ID or Actor_Client)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("CLIENT_TOKEN", None),
        help="Access token for server authentication (default: env CLIENT_TOKEN)"
    )
    parser.add_argument(
        "--client_host",
        type=str,
        default=os.getenv("CLIENT_HOST", "0.0.0.0"),
        help="Host for the client's own API server (default: env CLIENT_HOST or 0.0.0.0)"
    )
    parser.add_argument(
        "--client_port",
        type=int,
        default=int(os.getenv("CLIENT_PORT", 8001)), # Default to 8001 to avoid conflict with server
        help="Port for the client's own API server (default: env CLIENT_PORT or 8001)"
    )

    args = parser.parse_args()

    logger.info(f"Parsed arguments: Server URL='{args.server_url}', Actor_ID='{args.Actor_id}', Client Host='{args.client_host}', Client Port='{args.client_port}' Token Provided: {'Yes' if args.token else 'No'}")

    if args.token is None:
        logger.error("Client token not provided. Token is required via --token argument or CLIENT_TOKEN environment variable. Client will not start.")
        return

    # Store parsed args to be accessible by the FastAPI app instance if needed,
    # though initialize_character_client will pass them directly.
    app.state.config = args

    # Initialize the character client instance within the app state
    initialize_character_client(
        token=args.token,
        Actor_id=args.Actor_id,
        server_url=args.server_url,
        client_port=args.client_port # Pass client_port for registration
    )

    # Start the heartbeat task (if character_client_instance is now initialized)
    if hasattr(app.state, 'character_client_instance') and app.state.character_client_instance:
        start_heartbeat_task(app.state.character_client_instance) # This function should also use logging
        logger.info(f"Client '{args.Actor_id}' API server starting on {args.client_host}:{args.client_port}")
        # Uvicorn has its own logging. We can configure it if needed, or let it be.
        # Our application logs (FastAPI endpoints within character_client.py) will use our logger.
        uvicorn.run(app, host=args.client_host, port=args.client_port, log_level="info")
    else:
        logger.error(f"Failed to initialize CharacterClient for {args.Actor_id}. Client will not start.")

if __name__ == "__main__":
    main()
