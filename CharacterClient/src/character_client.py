from fastapi import FastAPI, HTTPException, Request
import requests
import uuid
import os
import base64
import asyncio

from .tts_manager import TTSManager
from .llm_engine import LLMEngine # Now using the implemented LLMEngine
from .config import (
    CLIENT_TTS_REFERENCE_VOICES_PATH,
    ensure_client_directories
)

app = FastAPI()
ensure_client_directories()

HEARTBEAT_INTERVAL_SECONDS = 60

class CharacterClient:
    def __init__(self, token: str, pc_id: str, server_url: str, client_port: int):
        self.token = token
        self.pc_id = pc_id
        self.server_url = server_url
        self.client_port = client_port
        self.character = None
        self.tts = None
        self.llm = None
        self.local_reference_audio_path = None

        print(f"Initializing CharacterClient for PC_ID: {self.pc_id} to connect to SERVER: {self.server_url}")

        if not self._register_with_server():
            print(f"CRITICAL: Client {self.pc_id} failed to register. Functionality may be impaired.")

        self.character = self.fetch_traits()
        if not self.character:
            print(f"CRITICAL: Failed to fetch character traits for {self.pc_id}. Using defaults.")
            # Define a more complete default character, including llm_model if server doesn't provide
            self.character = {
                "name": self.pc_id,
                "personality": "default",
                "tts": "gtts", # A safe default that doesn't require model download
                "tts_model": "en",
                "reference_audio_filename": None,
                "language": "en",
                "llm_model": None # Will use LLMEngine's default if None
            }

        # Initialize the implemented TTSManager
        self.tts = TTSManager(
            tts_service_name=self.character.get("tts", "gtts"),
            model_name=self.character.get("tts_model"),
            language=self.character.get("language", "en")
        )

        # Initialize the implemented LLMEngine
        # Pass pc_id to LLMEngine for adapter path construction
        self.llm = LLMEngine(
            model_name=self.character.get("llm_model"), # LLMEngine handles default if None
            pc_id=self.pc_id
        )

        if self.character and self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            self._download_reference_audio()

    def _download_reference_audio(self):
        filename = self.character["reference_audio_filename"]
        os.makedirs(CLIENT_TTS_REFERENCE_VOICES_PATH, exist_ok=True)
        sane_pc_id = "".join(c if c.isalnum() else "_" for c in self.pc_id)
        sane_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else "_" for c in filename)
        self.local_reference_audio_path = os.path.join(CLIENT_TTS_REFERENCE_VOICES_PATH, f"{sane_pc_id}_{sane_filename}")

        if os.path.exists(self.local_reference_audio_path) and os.path.getsize(self.local_reference_audio_path) > 0:
            print(f"Ref audio '{sane_filename}' exists at {self.local_reference_audio_path}. Skipping download.")
            return

        params = {"pc_id": self.pc_id, "token": self.token}
        try:
            ref_audio_url = f"{self.server_url}/get_reference_audio/{sane_filename}"
            print(f"Downloading ref audio from: {ref_audio_url} for {self.pc_id}")
            response = requests.get(ref_audio_url, params=params, stream=True, timeout=30)
            response.raise_for_status()
            with open(self.local_reference_audio_path, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            print(f"Ref audio '{sane_filename}' downloaded to: {self.local_reference_audio_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading ref audio '{sane_filename}' for {self.pc_id}: {e}")
            self.local_reference_audio_path = None
        except Exception as e:
            print(f"Unexpected error downloading ref audio for {self.pc_id}: {e}")
            self.local_reference_audio_path = None

    def fetch_traits(self):
        try:
            response = requests.get(f"{self.server_url}/get_traits", params={"pc_id": self.pc_id, "token": self.token}, timeout=10)
            response.raise_for_status()
            print(f"Fetched traits for {self.pc_id}")
            return response.json()
        except Exception as e:
            print(f"Error fetching traits for {self.pc_id}: {e}")
        return None

    def _register_with_server(self):
        print(f"Registering client {self.pc_id} (port {self.client_port}) with {self.server_url}...")
        try:
            payload = {"pc_id": self.pc_id, "token": self.token, "client_port": self.client_port}
            response = requests.post(f"{self.server_url}/register", json=payload, timeout=10)
            response.raise_for_status()
            print(f"Registration for {self.pc_id}: {response.json().get('message', 'OK')}")
            return True
        except Exception as e:
            print(f"Could not register client {self.pc_id}: {e}")
            return False

    def send_heartbeat(self):
        try:
            payload = {"pc_id": self.pc_id, "token": self.token}
            requests.post(f"{self.server_url}/heartbeat", json=payload, timeout=5).raise_for_status()
            return True
        except Exception as e:
            print(f"Heartbeat failed for {self.pc_id}: {e}") # Less verbose for heartbeat
            return False

    def generate_response(self, narration, character_texts):
        if not self.llm or not self.llm.is_initialized or not self.character: # Check is_initialized
            error_msg = f"[{self.character.get('name', self.pc_id) if self.character else self.pc_id} LLM not ready]"
            print(f"LLM generate_response error for {self.pc_id}: {error_msg}")
            return error_msg

        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in character_texts.items()]) + f"\nCharacter: {self.character['name']} (as {self.character['personality']}):"
        text = self.llm.generate(prompt)

        # Call placeholder fine_tune on client's LLM
        self.llm.fine_tune({"input": prompt, "output": text}, self.pc_id)

        try:
            requests.post(f"{self.server_url}/save_training_data",
                          json={"dataset": {"input": prompt, "output": text}, "pc_id": self.pc_id, "token": self.token},
                          timeout=10)
        except Exception as e:
            print(f"Error saving training data to server for {self.pc_id}: {e}")
        return text

    def _synthesize_audio_internal(self, text_to_synthesize: str) -> str | None:
        if not self.tts or not self.tts.is_initialized:
            print(f"Error: TTS not initialized for {self.pc_id}.")
            return None

        speaker_wav_to_use = None
        if self.character.get("tts") == "xttsv2":
            if self.local_reference_audio_path and os.path.exists(self.local_reference_audio_path):
                speaker_wav_to_use = self.local_reference_audio_path
            else:
                print(f"Warning: XTTSv2 for {self.pc_id}, local ref audio not valid: {self.local_reference_audio_path}. Using default.")

        sane_char_name = "".join(c if c.isalnum() else "_" for c in self.character.get('name', self.pc_id))
        audio_filename_no_path = f"{sane_char_name}_{uuid.uuid4()}.wav"

        generated_audio_path = self.tts.synthesize(
            text_to_synthesize,
            audio_filename_no_path,
            speaker_wav_for_synthesis=speaker_wav_to_use
        )

        if generated_audio_path:
            # print(f"Audio for {self.pc_id} synthesized to: {generated_audio_path}") # Can be verbose
            return generated_audio_path
        else:
            print(f"Audio synthesis failed for {self.pc_id}.")
            return None


@app.post("/character")
async def handle_character_generation_request(data: dict, request: Request):
    client: CharacterClient = request.app.state.character_client_instance
    if not client:
        raise HTTPException(status_code=503, detail="CharacterClient not available.")

    if client.token != data.get("token"):
        raise HTTPException(status_code=401, detail="Invalid token.")

    narration = data.get("narration")
    if narration is None:
        raise HTTPException(status_code=400, detail="Missing 'narration'.")

    character_texts = data.get("character_texts", {})
    response_text = client.generate_response(narration, character_texts)
    audio_path = client._synthesize_audio_internal(response_text)

    encoded_audio_data = None
    if audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f_audio:
            audio_data = f_audio.read()
        encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')
        try:
            os.remove(audio_path)
        except OSError as e:
            print(f"Warning: Could not delete temp audio file {audio_path}: {e}")
    # else: # This can be very noisy if generation or TTS often produces no audio
        # print(f"Warning: No audio for {client.pc_id}. Path was: {audio_path}")

    return {"text": response_text, "audio_data": encoded_audio_data}

async def _heartbeat_task_runner(client: CharacterClient):
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        if client: client.send_heartbeat()

_heartbeat_task_instance = None

def initialize_character_client(token: str, pc_id: str, server_url: str, client_port: int):
    global _heartbeat_task_instance
    if not hasattr(app.state, 'character_client_instance') or app.state.character_client_instance is None:
        ensure_client_directories()

        # Critical: Instantiate the client and store it
        client_instance = CharacterClient(
            token=token, pc_id=pc_id, server_url=server_url, client_port=client_port
        )
        app.state.character_client_instance = client_instance # Assign to app.state

        print(f"CharacterClient for {pc_id} initialized (LLM Ready: {client_instance.llm.is_initialized if client_instance.llm else False}, TTS Ready: {client_instance.tts.is_initialized if client_instance.tts else False}).")

        if _heartbeat_task_instance is None or _heartbeat_task_instance.done():
            if client_instance: # Use the locally created instance for clarity
                # Schedule heartbeat task using the main event loop (managed by Uvicorn)
                try:
                    loop = asyncio.get_running_loop()
                    _heartbeat_task_instance = loop.create_task(_heartbeat_task_runner(client_instance))
                    print(f"Heartbeat task scheduled for client {pc_id}.")
                except RuntimeError: # If no loop is running yet (shouldn't happen if called before uvicorn.run)
                     print(f"WARNING: No running event loop for heartbeat task of {pc_id}. Will rely on Uvicorn startup.")
                     # Fallback: try to create task anyway, or let startup_event handle it if re-enabled
                     _heartbeat_task_instance = asyncio.create_task(_heartbeat_task_runner(client_instance))


            else: # Should not happen if instance was just created
                 print(f"Heartbeat task NOT scheduled for {pc_id} due to client init failure immediately after creation.")
    else:
        print(f"CharacterClient for {pc_id} already initialized.")
