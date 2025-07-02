from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional # For optional fields
import io
import logging

from .database import Database
from .client_manager import ClientManager
from .dashboard import router as dashboard_router
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH
import os
from datetime import datetime, timezone

logger = logging.getLogger("dreamweaver_server")

app = FastAPI(title="DreamWeaver Server API")

app.include_router(dashboard_router)

# Dependency to get a database instance
def get_db():
    """
    Yields a database connection for use within a request, ensuring it is closed afterward.
    """
    db = Database(DB_PATH)
    try:
        yield db
    finally:
        db.close() # Ensure DB connection is closed

# Dependency to get a ClientManager instance
def get_client_manager(db: Database = Depends(get_db)):
    """
    Provides a `ClientManager` instance initialized with the given database.
    
    Returns:
        ClientManager: An instance for managing client-related operations using the provided database.
    """
    return ClientManager(db) # ClientManager now also uses the new DB methods

# --- Pydantic Models ---
class SaveTrainingDataRequest(BaseModel):
    dataset: dict
    Actor_id: str
    token: str

    class Config:
        json_schema_extra = {
            "example": {
                "dataset": {"input": "Narrator: The wind howls.", "output": "I shiver."},
                "Actor_id": "Actor2",
                "token": "your_token_here"
            }
        }

class RegisterClientRequest(BaseModel):
    Actor_id: str
    token: str
    client_port: int

    class Config:
        json_schema_extra = {
            "example": {
                "Actor_id": "Actor2",
                "token": "your_token_here",
                "client_port": 8001
            }
        }

class HeartbeatRequest(BaseModel):
    Actor_id: str
    token: str
    status: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "Actor_id": "Actor2",
                "token": "your_token_here",
                "status": "Idle"
            }
        }


# --- API Endpoints ---
@app.get("/get_traits", summary="Fetch Character Traits")
async def get_traits(
    Actor_id: str,
    token: str,
    db: Database = Depends(get_db),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Retrieve the character traits associated with the specified Actor ID after validating authentication credentials.
    
    Returns:
        dict: The character traits for the given Actor ID if authentication succeeds and the character exists.
    
    Raises:
        HTTPException: If the token is invalid (401) or the character is not found (404).
    """
    if not client_manager.validate_token(Actor_id, token): # validate_token needs to be in ClientManager
        logger.warning(f"Invalid token for Actor_id: {Actor_id} in get_traits.")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    character = db.get_character(Actor_id) # Assumes Actor_id is the primary key for characters
    if not character:
        logger.warning(f"Character not found for Actor_id: {Actor_id} in get_traits.")
        raise HTTPException(status_code=404, detail=f"Character for Actor ID '{Actor_id}' not found")
    logger.info(f"Successfully retrieved traits for Actor_id: {Actor_id}.")
    return character


@app.post("/save_training_data", summary="Save Client LLM Training Data")
async def save_training_data(
    request_data: SaveTrainingDataRequest,
    db: Database = Depends(get_db),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Save training data for a specified actor after validating authentication.
    
    Validates the provided token for the given actor. If authentication succeeds, stores the supplied training dataset in the database. Returns a success message upon completion or raises an HTTP 500 error if saving fails.
    """
    if not client_manager.validate_token(request_data.Actor_id, request_data.token):
        logger.warning(f"Invalid token for Actor_id: {request_data.Actor_id} in save_training_data.")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        db.save_training_data(request_data.dataset, request_data.Actor_id)
        logger.info(f"Training data for {request_data.Actor_id} saved successfully.")
        return {"message": f"Training data for {request_data.Actor_id} saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save training data for {request_data.Actor_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while saving training data.")


@app.post("/register", summary="Register Client with Server")
async def register_client_endpoint( # Renamed to avoid conflict with db.register_client
    request_data: RegisterClientRequest,
    http_request: Request,
    db: Database = Depends(get_db),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Registers or updates a client in the database with the provided Actor ID, client IP address, and port after validating the authentication token.
    
    Returns:
        dict: A message confirming successful registration and client status.
    
    Raises:
        HTTPException: If the token is invalid (401) or registration fails due to a server error (500).
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    if not client_manager.validate_token(request_data.Actor_id, request_data.token):
        logger.warning(f"Invalid token for Actor_id: {request_data.Actor_id} during registration from IP: {client_ip}.")
        raise HTTPException(status_code=401, detail="Invalid token for registration")

    try:
        # Database method now handles inserting/updating client_port
        db.register_client(request_data.Actor_id, client_ip, request_data.client_port)
        logger.info(f"Client {request_data.Actor_id} registered from {client_ip}:{request_data.client_port}. Status: Online.")
        return {"message": f"Client {request_data.Actor_id} registered from {client_ip}:{request_data.client_port}. Status: Online."}
    except Exception as e:
        logger.error(f"Failed to register client {request_data.Actor_id} from IP {client_ip}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during client registration.")


@app.post("/heartbeat", summary="Client Heartbeat")
async def client_heartbeat(
    request_data: HeartbeatRequest,
    db: Database = Depends(get_db),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Handle client heartbeat requests by updating the client's status to "Online" and recording the current UTC timestamp.
    
    Returns:
        dict: A message confirming receipt of the heartbeat and the updated status.
    """
    if not client_manager.validate_token(request_data.Actor_id, request_data.token):
        logger.warning(f"Invalid token for Actor_id: {request_data.Actor_id} in heartbeat.")
        raise HTTPException(status_code=401, detail="Invalid token for heartbeat")

    try:
        timestamp_utc_iso = datetime.now(timezone.utc).isoformat()
        # db.update_client_status now handles setting status to 'Online' and updating last_seen
        db.update_client_status(request_data.Actor_id, "Online", timestamp_utc_iso)
        # logger.debug(f"Heartbeat received from {request_data.Actor_id}. Status: Online.") # Debug level for frequent logs
        return {"message": f"Heartbeat received from {request_data.Actor_id}. Status: Online."}
    except Exception as e:
        logger.error(f"Error processing heartbeat for {request_data.Actor_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing heartbeat.")


@app.get("/get_reference_audio/{filename}", summary="Download Reference Audio for Voice Cloning")
async def get_reference_audio(
    filename: str,
    Actor_id: str, # Added Actor_id as query param
    token: str, # Added token as query param
    client_manager: ClientManager = Depends(get_client_manager) # No db needed if just validating token
):
    """
    Serves a reference audio file to authenticated clients after validating their token and ensuring secure file access.
    
    Clients must provide a valid `Actor_id` and `token` as query parameters. The function checks for path traversal attempts and ensures the requested file exists within the allowed audio directory. Returns the audio file as a WAV response if all checks pass.
    """
    if not client_manager.validate_token(Actor_id, token):
        logger.warning(f"Invalid token for Actor_id: {Actor_id} in get_reference_audio for file: {filename}.")
        raise HTTPException(status_code=401, detail="Invalid or expired token for audio download")

    # Harden path sanitisation using os.path.basename
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        logger.error(f"Potential path traversal attempt for filename: {filename} by Actor_id: {Actor_id}.")
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    file_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, safe_name)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.warning(f"Reference audio file '{filename}' not found for Actor_id: {Actor_id}.")
        raise HTTPException(status_code=404, detail=f"Reference audio file '{filename}' not found.")

    # Ensure the file is within the allowed directory (important security check)
    if not os.path.abspath(file_path).startswith(os.path.abspath(REFERENCE_VOICES_AUDIO_PATH)):
        logger.error(f"Forbidden access attempt to file path: {file_path} by Actor_id: {Actor_id}.")
        raise HTTPException(status_code=403, detail="Access to this file path is forbidden.")

    logger.info(f"Serving reference audio '{filename}' to Actor_id: {Actor_id}.")
    return FileResponse(file_path, media_type="audio/wav", filename=safe_name)


@app.get("/download_client_config/{actor_id}", summary="Download Client Configuration File")
async def download_client_config(
    actor_id: str,
    server_url: str, # Provided by Gradio UI as a query parameter
    db: Database = Depends(get_db)
):
    """
    Generates and serves a .env-style configuration file for a given Actor ID.

    The file includes the Actor_ID, its token, and the server_url provided by the user
    via the Gradio interface. This helps in easy setup of new clients.
    """
    token_details = db.get_client_token_details(actor_id) # Fetch full details to ensure actor exists
    if not token_details or not token_details.get('token'):
        logger.warning(f"Token not found for Actor_id: {actor_id} in download_client_config.")
        raise HTTPException(status_code=404, detail=f"Token for Actor ID '{actor_id}' not found or actor does not exist.")

    token = token_details['token']
    logger.info(f"Generating client config for Actor_id: {actor_id} with server_url: {server_url}.")

    # Construct .env file content
    config_content = f"CLIENT_Actor_ID=\"{actor_id}\"\n"
    config_content += f"CLIENT_TOKEN=\"{token}\"\n"
    config_content += f"SERVER_URL=\"{server_url}\"\n"

    # Use io.BytesIO for binary content expected by StreamingResponse
    config_bytes = io.BytesIO(config_content.encode('utf-8'))

    filename = f"{actor_id}_client_config.env"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{filename}\""
    }
    return StreamingResponse(config_bytes, media_type="text/plain", headers=headers)
