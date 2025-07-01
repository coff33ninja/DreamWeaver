from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional # For optional fields

from .database import Database
from .client_manager import ClientManager
from .dashboard import router as dashboard_router
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH
import os
from datetime import datetime, timezone

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
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    character = db.get_character(Actor_id) # Assumes Actor_id is the primary key for characters
    if not character:
        raise HTTPException(status_code=404, detail=f"Character for Actor ID '{Actor_id}' not found")
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
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        db.save_training_data(request_data.dataset, request_data.Actor_id)
        return {"message": f"Training data for {request_data.Actor_id} saved successfully"}
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Failed to save training data: {str(e)}")


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
        raise HTTPException(status_code=401, detail="Invalid token for registration")

    try:
        # Database method now handles inserting/updating client_port
        db.register_client(request_data.Actor_id, client_ip, request_data.client_port)
        return {"message": f"Client {request_data.Actor_id} registered from {client_ip}:{request_data.client_port}. Status: Online."}
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Failed to register client {request_data.Actor_id}: {str(e)}")


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
        raise HTTPException(status_code=401, detail="Invalid token for heartbeat")

    try:
        timestamp_utc_iso = datetime.now(timezone.utc).isoformat()
        # db.update_client_status now handles setting status to 'Online' and updating last_seen
        db.update_client_status(request_data.Actor_id, "Online", timestamp_utc_iso)
        return {"message": f"Heartbeat received from {request_data.Actor_id}. Status: Online."}
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error processing heartbeat for {request_data.Actor_id}: {str(e)}")


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
        raise HTTPException(status_code=401, detail="Invalid or expired token for audio download")

    # Basic security check for filename (already in plan, good to have here)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    file_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, filename)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Reference audio file '{filename}' not found.")

    # Ensure the file is within the allowed directory (important security check)
    if not os.path.abspath(file_path).startswith(os.path.abspath(REFERENCE_VOICES_AUDIO_PATH)):
        raise HTTPException(status_code=403, detail="Access to this file path is forbidden.")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


# Example of how main.py might run this (if not using the existing main.py from project root)
# if __name__ == "__main__":
#     import uvicorn
#     # Ensure config.py creates directories if they don't exist
#     # from .config import इंश्योर_डिरेक्टरीज_एग्जिस्ट # (pseudo-code for ensure dirs)
#     # ensure_directories_exist()
#     uvicorn.run(app, host="0.0.0.0", port=8000)
