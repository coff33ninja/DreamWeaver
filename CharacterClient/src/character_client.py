from fastapi import FastAPI, HTTPException, Request
import requests
import uuid
import os
import base64
import asyncio
import tempfile

# Import client-specific modules
# Assuming these are in the same directory or properly pathed
from .tts_manager import TTSManager  # Relative import
from .llm_engine import LLMEngine   # Relative import

app = FastAPI()

# Global to store CharacterClient instance, initialized via startup event
# app.state.character_client_instance will hold the client

HEARTBEAT_INTERVAL_SECONDS = 60

class CharacterClient:
    def __init__(self, token: str, pc_id: str, server_url: str, client_port: int):
        self.token = token
        self.pc_id = pc_id
        self.server_url = server_url
        self.client_port = client_port # Port this client is running on, for registration
        self.character = None
        self.tts = None
        self.llm = None
        self.local_reference_audio_path = None

        print(f"Initializing CharacterClient for PC_ID: {self.pc_id} to connect to SERVER: {self.server_url}")

        if not self._register_with_server():
            # If registration fails, character might not function properly.
            # Consider how to handle this - e.g., retry or exit.
            print(f"CRITICAL: Client {self.pc_id} failed to register with the server. Functionality may be impaired.")
            # Depending on desired robustness, you might raise an exception here or allow it to proceed without traits.
            # For now, we'll let it proceed, but fetch_traits will likely fail or return defaults.

        self.character = self.fetch_traits()
        if not self.character:
            print(f"CRITICAL: Failed to fetch character traits for {self.pc_id}. Client cannot operate fully.")
            # Fallback to default character dictionary to prevent crashes, but it won't be the configured one.
            self.character = {"name": self.pc_id, "personality": "default", "goals": "none", "backstory": "none", "tts": "piper", "tts_model": "en_US-ryan-high", "reference_audio_filename": None}


        # Initialize TTS and LLM engines after fetching character traits
        # The client-side TTSManager and LLMEngine need to be adapted to this project structure
        # For now, using placeholders or assuming they exist and are correctly configured.
        self.tts = TTSManager(
            tts_service_name=self.character.get("tts"),
            model_name=self.character.get("tts_model")
            # Client-side TTSManager might need a config for where its models are stored.
        )
        self.llm = LLMEngine(model_name="TinyLLaMA_client_variant") # Example model name for client

        # If XTTS-v2 and reference audio is specified, download it
        if self.character and self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            self._download_reference_audio()

    def _download_reference_audio(self):
        filename = self.character["reference_audio_filename"]
        # The server API expects pc_id and token for this endpoint now
        params = {"pc_id": self.pc_id, "token": self.token}
        try:
            # Corrected URL construction
            ref_audio_url = f"{self.server_url}/get_reference_audio/{filename}"
            print(f"Downloading reference audio from: {ref_audio_url} with params {params}")

            # Pass pc_id and token as query parameters
            response = requests.get(ref_audio_url, params=params, stream=True)
            response.raise_for_status()

            # Use a temporary directory for client-specific data if needed, or a defined data path
            client_data_dir = os.path.join(tempfile.gettempdir(), "dreamweaver_client_data", self.pc_id)
            os.makedirs(client_data_dir, exist_ok=True)

            self.local_reference_audio_path = os.path.join(client_data_dir, filename)

            with open(self.local_reference_audio_path, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            print(f"Reference audio '{filename}' downloaded to: {self.local_reference_audio_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading reference audio '{filename}': {e}")
            self.local_reference_audio_path = None
        except Exception as e:
            print(f"An unexpected error occurred during reference audio download: {e}")
            self.local_reference_audio_path = None


    def fetch_traits(self):
        try:
            response = requests.get(f"{self.server_url}/get_traits", params={"pc_id": self.pc_id, "token": self.token}, timeout=10)
            response.raise_for_status()
            print(f"Successfully fetched traits for {self.pc_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching traits for {self.pc_id} from {self.server_url}/get_traits: {e}")
        except Exception as e: # Catch any other exceptions like JSONDecodeError
            print(f"An unexpected error occurred fetching traits: {e}")
        return None # Return None on failure

    def _register_with_server(self):
        """Registers the client with the server, including its listening port."""
        print(f"Registering client {self.pc_id} running on port {self.client_port} with server {self.server_url}...")
        try:
            payload = {
                "pc_id": self.pc_id,
                "token": self.token,
                "client_port": self.client_port # Send client's listening port
            }
            response = requests.post(f"{self.server_url}/register", json=payload, timeout=10)
            response.raise_for_status()
            print(f"Registration successful: {response.json().get('message', 'OK')}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Could not register client {self.pc_id} with server: {e}")
            return False

    def send_heartbeat(self):
        """Sends a heartbeat to the server to indicate it's still alive."""
        print(f"Sending heartbeat from {self.pc_id}...")
        try:
            payload = {"pc_id": self.pc_id, "token": self.token}
            response = requests.post(f"{self.server_url}/heartbeat", json=payload, timeout=10)
            response.raise_for_status()
            # print(f"Heartbeat acknowledged: {response.json().get('message', 'OK')}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Heartbeat failed for {self.pc_id}: {e}")
            return False

    def generate_response(self, narration, character_texts):
        if not self.llm or not self.character:
            return "Error: LLM or character not initialized."

        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in character_texts.items()]) + f"\nCharacter: {self.character['name']} (as {self.character['personality']}):"
        text = self.llm.generate(prompt) # Assuming client's LLM has a generate method

        # Client-side fine-tuning and data saving
        training_data = {"input": prompt, "output": text}
        self.llm.fine_tune(training_data, self.pc_id) # Assuming client's LLM has fine_tune

        try:
            save_payload = {"dataset": training_data, "pc_id": self.pc_id, "token": self.token}
            requests.post(f"{self.server_url}/save_training_data", json=save_payload, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Error saving training data to server for {self.pc_id}: {e}")
        return text

    def synthesize_audio(self, text):
        if not self.tts or not self.character:
            return None # Or path to a default "error" audio

        speaker_wav_to_use = None
        if self.character.get("tts") == "xttsv2":
            if self.local_reference_audio_path and os.path.exists(self.local_reference_audio_path):
                speaker_wav_to_use = self.local_reference_audio_path
            else:
                print(f"Warning: XTTSv2 selected for {self.pc_id}, but local reference audio path is not valid: {self.local_reference_audio_path}. Using default voice.")

        # Define a directory for storing audio files, perhaps in a temporary location or client-specific data folder
        audio_output_dir = os.path.join(tempfile.gettempdir(), "dreamweaver_client_audio", self.pc_id)
        os.makedirs(audio_output_dir, exist_ok=True)

        audio_filename = f"character_audio_{self.character.get('name', 'unknown')}_{uuid.uuid4()}.wav"
        audio_path = os.path.join(audio_output_dir, audio_filename)

        success = self.tts.synthesize(text, audio_path, speaker_wav_for_synthesis=speaker_wav_to_use)
        if success:
            return audio_path
        else:
            print(f"Audio synthesis failed for {self.pc_id}")
            return None


@app.post("/character")
async def handle_character_generation_request(data: dict, request: Request):
    # Access the client instance from app.state, set by initialize_character_client
    client = request.app.state.character_client_instance
    if not client:
        raise HTTPException(status_code=503, detail="CharacterClient not available or not initialized.")

    # Validate token received in request against the client's own token
    request_token = data.get("token")
    if client.token != request_token:
        raise HTTPException(status_code=401, detail="Invalid token for this client instance.")

    narration = data.get("narration")
    character_texts = data.get("character_texts", {}) # Default to empty dict if not provided

    if narration is None:
        raise HTTPException(status_code=400, detail="Missing 'narration' in request.")

    response_text = client.generate_response(narration, character_texts)
    if not response_text: # Handle case where generation might fail or return empty
        response_text = f"[{client.character.get('name', client.pc_id)} is silent or unable to respond.]"

    audio_path = client.synthesize_audio(response_text)

    encoded_audio_data = None
    if audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f_audio:
            audio_data = f_audio.read()
        encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')
        try:
            os.remove(audio_path) # Clean up temporary audio file
        except OSError as e:
            print(f"Warning: Could not delete temporary audio file {audio_path}: {e}")
    else:
        print(f"Warning: No audio synthesized or audio file not found for {client.pc_id}")

    return {"text": response_text, "audio_data": encoded_audio_data}


async def _heartbeat_task_runner(client: CharacterClient):
    """Internal async task for sending heartbeats."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        if client: # Ensure client is still valid
            client.send_heartbeat()


def initialize_character_client(token: str, pc_id: str, server_url: str, client_port: int):
    """
    Initializes the CharacterClient instance and stores it in app.state.
    This function is called from main.py before uvicorn.run().
    """
    if not hasattr(app.state, 'character_client_instance') or app.state.character_client_instance is None:
        app.state.character_client_instance = CharacterClient(
            token=token,
            pc_id=pc_id,
            server_url=server_url,
            client_port=client_port
        )
        print(f"CharacterClient for {pc_id} initialized and attached to app.state.")
    else:
        print(f"CharacterClient for {pc_id} already initialized.")


def start_heartbeat_task(client_instance: CharacterClient):
    """Starts the background heartbeat task."""
    if client_instance:
        # Important: Ensure the event loop is running or get the current one
        # for asyncio.create_task. When called from main.py before uvicorn,
        # the loop might not be fully set up by uvicorn yet.
        # However, uvicorn itself will manage the loop for tasks started via on_event("startup")
        # For tasks started *before* uvicorn.run, it's simpler if uvicorn manages them.
        # Let's adjust this to be called via @app.on_event("startup")
        pass # Will be handled by FastAPI startup event below

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event.
    The CharacterClient should have been initialized by initialize_character_client()
    called from main.py before uvicorn.run().
    Here, we just ensure the heartbeat task is started for the initialized client.
    """
    if hasattr(app.state, 'character_client_instance') and app.state.character_client_instance:
        client = app.state.character_client_instance
        # Start the heartbeat task in the background
        # Ensure this is idempotent if startup_event can be called multiple times in some contexts
        if not hasattr(app.state, '_heartbeat_task_started') or not app.state._heartbeat_task_started:
            asyncio.create_task(_heartbeat_task_runner(client))
            app.state._heartbeat_task_started = True # Flag to prevent multiple task starts
            print(f"Heartbeat task initiated for client {client.pc_id}.")
    else:
        print("WARNING: CharacterClient instance not found in app.state at startup. Heartbeat will not start.")
        # This indicates an issue with the initialization flow from main.py

# Note: The if __name__ == "__main__": block is removed from here.
# uvicorn.run() will be called from CharacterClient/main.py directly.
