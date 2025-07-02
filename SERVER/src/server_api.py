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
from datetime import datetime, timezone, timedelta
import secrets
import hashlib
from fastapi import WebSocket, WebSocketDisconnect # For WebSocket support
from .websocket_manager import connection_manager # Import the global manager

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

class RequestChallengePayload(BaseModel): # Renamed for clarity
    Actor_id: str
    token: str # Primary token

class HandshakeResponseSubmit(BaseModel):
    Actor_id: str
    challenge_response: str


# --- API Endpoints ---

# TODO: Update all existing endpoints to use the new ClientManager.authenticate_request method

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
    if not client_manager.authenticate_request_token(Actor_id, token):
        logger.warning(f"Authentication failed for Actor_id: {Actor_id} in get_traits.")
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
    if not client_manager.authenticate_request_token(request_data.Actor_id, request_data.token):
        logger.warning(f"Authentication failed for Actor_id: {request_data.Actor_id} in save_training_data.")
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
    # Registration should always use the primary token.
    if not client_manager.validate_token(request_data.Actor_id, request_data.token):
        logger.warning(f"Invalid primary token for Actor_id: {request_data.Actor_id} during registration from IP: {client_ip}.")
        raise HTTPException(status_code=401, detail="Invalid primary token for registration.")

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
    if not client_manager.authenticate_request_token(request_data.Actor_id, request_data.token):
        logger.warning(f"Authentication failed for Actor_id: {request_data.Actor_id} in heartbeat.")
        raise HTTPException(status_code=401, detail="Invalid or expired token for heartbeat")

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
    if not client_manager.authenticate_request_token(Actor_id, token):
        logger.warning(f"Authentication failed for Actor_id: {Actor_id} in get_reference_audio for file: {filename}.")
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


# --- Handshake Endpoints ---

@app.post("/request_handshake_challenge", summary="Request a challenge for session handshake")
async def request_handshake_challenge(
    payload: RequestChallengePayload,
    db: Database = Depends(get_db),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Client provides its Actor_id and primary token in the request body.
    Server validates primary token. If valid, generates and returns a challenge.
    """
    Actor_id = payload.Actor_id
    client_primary_token = payload.token
    logger.info(f"Handshake: Received challenge request from Actor_id: {Actor_id}")

    # Validate primary token first
    # Using ClientManager.validate_token as it's designed for primary token validation
    if not client_manager.validate_token(Actor_id, client_primary_token):
        logger.warning(f"Handshake: Invalid primary token for Actor_id: {Actor_id} during challenge request.")
        raise HTTPException(status_code=401, detail="Invalid primary token for challenge request.")

    challenge = client_manager.generate_handshake_challenge(Actor_id)
    if not challenge:
        logger.error(f"Handshake: Failed to generate challenge for Actor_id: {Actor_id}")
        raise HTTPException(status_code=500, detail="Failed to generate handshake challenge.")

    logger.info(f"Handshake: Issued challenge to Actor_id: {Actor_id}")
    return {"challenge": challenge}


@app.post("/submit_handshake_response", summary="Submit challenge response to get a session token")
async def submit_handshake_response(
    submit_data: HandshakeResponseSubmit,
    db: Database = Depends(get_db),
    client_manager: ClientManager = Depends(get_client_manager)
):
    """
    Client submits its Actor_id and the computed challenge_response.
    Server verifies the response and, if valid, issues a session token.
    """
    Actor_id = submit_data.Actor_id
    client_challenge_response = submit_data.challenge_response
    logger.info(f"Handshake: Received challenge response from Actor_id: {Actor_id}")

    stored_challenge = client_manager.get_and_validate_challenge(Actor_id)
    if not stored_challenge:
        logger.warning(f"Handshake: No valid/unexpired challenge found for Actor_id: {Actor_id}. Response rejected.")
        raise HTTPException(status_code=400, detail="Invalid or expired challenge.")

    primary_token = db.get_primary_token(Actor_id)
    if not primary_token:
        # This should ideally not happen if challenge was issued based on a valid primary token
        logger.error(f"Handshake: Primary token for Actor_id: {Actor_id} not found in DB during response verification. This is unexpected.")
        client_manager.clear_challenge(Actor_id) # Clean up challenge
        raise HTTPException(status_code=500, detail="Internal server error during handshake.")

    import hashlib
    expected_message = primary_token + stored_challenge
    expected_response_hash = hashlib.sha256(expected_message.encode('utf-8')).hexdigest()

    if expected_response_hash == client_challenge_response:
        logger.info(f"Handshake: Challenge response verified for Actor_id: {Actor_id}.")
        client_manager.clear_challenge(Actor_id) # Clear the used challenge

        session_token = secrets.token_hex(32)
        # Define session duration (e.g., 1 hour)
        session_duration_hours = 1 # TODO: Make this configurable
        session_expiry = datetime.now(timezone.utc) + timedelta(hours=session_duration_hours)

        try:
            db.update_client_session_token(Actor_id, session_token, session_expiry)
            logger.info(f"Handshake: Issued session token for Actor_id: {Actor_id}, expires at {session_expiry.isoformat()}")
            return {"session_token": session_token, "expires_at": session_expiry.isoformat()}
        except Exception as e:
            logger.error(f"Handshake: Failed to save session token for Actor_id {Actor_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to finalize session.")
    else:
        logger.warning(f"Handshake: Invalid challenge response for Actor_id: {Actor_id}. Expected hash did not match.")
        client_manager.clear_challenge(Actor_id) # Clear the used challenge even on failure to prevent replay with same challenge
        raise HTTPException(status_code=401, detail="Invalid challenge response.")


# --- WebSocket Endpoint ---
@app.websocket("/ws/{actor_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    actor_id: str,
    session_token: str = "", # Expect session_token as a query parameter
    db: Database = Depends(get_db), # Required for client_manager auth
    client_manager_dep: ClientManager = Depends(get_client_manager) # Renamed to avoid conflict
):
    """
    Handles WebSocket connections for clients.
    Authenticates clients using their session token.
    Manages connection lifecycle and listens for messages (primarily for keep-alive/disconnect).
    """
    if not session_token:
        logger.warning(f"WebSocket: Connection attempt from Actor_id {actor_id} without a session token.")
        await websocket.close(code=4001, reason="Session token required") # Custom close code
        return

    # Authenticate using the session token (or primary, as per authenticate_request_token logic)
    if not client_manager_dep.authenticate_request_token(actor_id, session_token):
        logger.warning(f"WebSocket: Authentication failed for Actor_id {actor_id} with provided token.")
        await websocket.close(code=4003, reason="Invalid session token") # Custom close code
        return

    await connection_manager.connect(websocket, actor_id)
    try:
        while True:
            # For now, primarily server-to-client. Client might send pings or specific messages later.
            data = await websocket.receive_text()
            logger.debug(f"WebSocket: Received text from {actor_id}: {data}")
            # Example: await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket: Client {actor_id} disconnected (WebSocketDisconnect received).")
    except Exception as e:
        logger.error(f"WebSocket: Unexpected error with client {actor_id} in main loop: {e}", exc_info=True)
        if not websocket.client_state == websockets.protocol.State.CLOSED: # type: ignore
            try:
                await websocket.close(code=1011, reason="Internal server error") # type: ignore
                logger.info(f"WebSocket: Closed connection to {actor_id} due to server-side exception.")
            except Exception as e_close:
                logger.error(f"WebSocket: Error trying to close connection with {actor_id} after exception: {e_close}", exc_info=True)
    finally:
        connection_manager.disconnect(actor_id)
