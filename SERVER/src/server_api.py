from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import FileResponse # Added FileResponse
from pydantic import BaseModel
from .database import Database
from .client_manager import ClientManager
from .dashboard import router as dashboard_router
from .config import DB_PATH, REFERENCE_VOICES_AUDIO_PATH # Import from config
import os
from datetime import datetime, timezone # For heartbeat

app = FastAPI()

app.include_router(dashboard_router)

# Dependency to get a database instance
def get_db():
    # Use DB_PATH from config
    db = Database(DB_PATH)
    try:
        yield db
    finally:
        # db.close() if your Database class has a close method
        pass

# Dependency to get a ClientManager instance
def get_client_manager(db: Database = Depends(get_db)):
    return ClientManager(db)

# Pydantic models for request bodies
class SaveTrainingDataRequest(BaseModel):
    dataset: dict # Expects {"input": "...", "output": "..."}
    pc_id: str # Changed from pc to pc_id for clarity
    token: str

class RegisterClientRequest(BaseModel):
    pc_id: str # Changed from pc to pc_id
    token: str
    # client_port: int # Optional: client could specify which port it's listening on

class HeartbeatRequest(BaseModel):
    pc_id: str
    token: str
    # status: str # Optional: client could report its own status e.g. "idle", "processing"

@app.get("/get_traits")
async def get_traits(pc_id: str, token: str, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    if not client_manager.validate_token(pc_id, token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    character = db.get_character(pc_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character for PC {pc_id} not found")
    return character

@app.post("/save_training_data")
async def save_training_data(request: SaveTrainingDataRequest, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    if not client_manager.validate_token(request.pc_id, request.token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        db.save_training_data(request.dataset, request.pc_id)
        return {"message": "Training data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save training data: {e}")

@app.post("/register")
async def register_client(request: RegisterClientRequest, http_request: Request, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    client_ip = http_request.client.host
    if not client_manager.validate_token(request.pc_id, request.token): # Use pc_id
        raise HTTPException(status_code=401, detail="Invalid token for registration")

    try:
        # The port the client is listening on might be fixed or sent by client
        # For now, assume client_manager or db.register_client handles this.
        db.register_client(request.pc_id, client_ip) # Use pc_id
        # Update last_seen on registration as well
        db.update_client_status(request.pc_id, "Online", datetime.now(timezone.utc).isoformat())
        return {"message": f"Client {request.pc_id} registered successfully from IP {client_ip}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register client: {e}")

@app.post("/heartbeat")
async def client_heartbeat(request: HeartbeatRequest, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    if not client_manager.validate_token(request.pc_id, request.token):
        raise HTTPException(status_code=401, detail="Invalid token for heartbeat")

    try:
        # Update the last_seen timestamp for the client
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        db.update_client_status(request.pc_id, "Online", timestamp_utc) # Assuming 'Online' status on heartbeat
        return {"message": f"Heartbeat received from {request.pc_id}"}
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Error processing heartbeat for {request.pc_id}: {e}")


@app.get("/get_reference_audio/{filename}")
async def get_reference_audio(filename: str, token: str, pc_id: str, client_manager: ClientManager = Depends(get_client_manager)): # Added token auth
    """
    Endpoint for clients to download reference audio files for voice cloning.
    Requires a valid token. The pc_id in the query is for logging/validation but not strictly needed if token is unique.
    """
    if not client_manager.validate_token(pc_id, token): # Validate token
        raise HTTPException(status_code=401, detail="Invalid or expired token for audio download")

    # Use path from config
    file_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, filename)

    # Security: Ensure the requested filename is simple and does not attempt path traversal.
    # A more robust check would be to ensure `filename` does not contain '..' or '/'
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.exists(file_path) or not os.path.isfile(file_path): # Check if it's a file
        raise HTTPException(status_code=404, detail="Reference audio file not found")

    # Ensure the file is within the allowed directory (double check, though join should handle it)
    if not os.path.abspath(file_path).startswith(os.path.abspath(REFERENCE_VOICES_AUDIO_PATH)):
        raise HTTPException(status_code=403, detail="Access to this file is forbidden.")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)
