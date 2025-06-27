from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from .database import Database
from .client_manager import ClientManager
from .dashboard import router as dashboard_router # Import the dashboard router
import os

app = FastAPI()

app.include_router(dashboard_router) # Include the dashboard routes in the main app

# Dependency to get a database instance
def get_db():
    db = Database("E:/DreamWeaver/data/dream_weaver.db")
    try:
        yield db
    finally:
        # In a real application, you might want to manage connection pooling
        # or ensure connections are closed if not using a context manager.
        # For SQLite, it's often fine to keep it open for the app's lifetime.
        pass

# Dependency to get a ClientManager instance
def get_client_manager(db: Database = Depends(get_db)):
    return ClientManager(db)

# Pydantic models for request bodies
class SaveTrainingDataRequest(BaseModel):
    dataset: dict
    pc: str
    token: str

class RegisterClientRequest(BaseModel):
    pc: str
    token: str

@app.get("/get_traits")
async def get_traits(pc: str, token: str, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    """
    Endpoint for clients to fetch their character traits.
    Requires a valid token for authentication.
    """
    if not client_manager.validate_token(pc, token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    character = db.get_character(pc)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character for PC {pc} not found")
    return character

@app.post("/save_training_data")
async def save_training_data(request: SaveTrainingDataRequest, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    """
    Endpoint for clients to save their LLM training data to the server.
    Requires a valid token for authentication.
    """
    if not client_manager.validate_token(request.pc, request.token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    try:
        db.save_training_data(request.dataset, request.pc)
        return {"message": "Training data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save training data: {e}")

@app.post("/register")
async def register_client(request: RegisterClientRequest, http_request: Request, db: Database = Depends(get_db), client_manager: ClientManager = Depends(get_client_manager)):
    """
    Endpoint for clients to register their IP address with the server.
    """
    client_ip = http_request.client.host
    if not client_manager.validate_token(request.pc, request.token):
        raise HTTPException(status_code=401, detail="Invalid token for registration")

    try:
        db.register_client(request.pc, client_ip)
        return {"message": f"Client {request.pc} registered successfully from IP {client_ip}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register client: {e}")

@app.get("/get_reference_audio/{filename}")
async def get_reference_audio(filename: str):
    """
    Endpoint for clients to download reference audio files for voice cloning.
    """
    file_path = os.path.join("E:/DreamWeaver/data/audio/reference_voices", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Reference audio file not found")

    # Ensure the file is within the allowed directory to prevent path traversal attacks
    return FileResponse(file_path, media_type="audio/wav", filename=filename)
