import argparse
import os
import uvicorn
from src.character_client import app, initialize_character_client, start_heartbeat_task

def main():
    """
    Parses command-line arguments, initializes the DreamWeaver Character Client, and starts the FastAPI server.
    
    This function configures the client using command-line arguments or environment variables, validates required parameters, initializes the character client instance, starts a heartbeat background task if initialization succeeds, and launches the FastAPI app using Uvicorn. If initialization fails or the authentication token is missing, the server does not start.
    """
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

    if args.token is None:
        print("Error: Client token must be provided via --token argument or CLIENT_TOKEN environment variable.")
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
        start_heartbeat_task(app.state.character_client_instance)
        print(f"Client '{args.Actor_id}' API server starting on {args.client_host}:{args.client_port}")
        uvicorn.run(app, host=args.client_host, port=args.client_port, log_level="info")
    else:
        print(f"Failed to initialize CharacterClient for {args.Actor_id}. Client will not start.")

if __name__ == "__main__":
    main()
