import os
import uuid
import requests
import base64
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request

from .tts_manager import TTSManager # Now async
from .llm_engine import LLMEngine  # Now async
from .config import (
    CLIENT_TTS_REFERENCE_VOICES_PATH,
    ensure_client_directories
)

app = FastAPI()
ensure_client_directories()

HEARTBEAT_INTERVAL_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 2

class CharacterClient:
    def __init__(self, token: str, Actor_id: str, server_url: str, client_port: int):
        self.token = token
        self.Actor_id = Actor_id
        self.server_url = server_url
        self.client_port = client_port
        self.character = None
        self.tts = None
        self.llm = None
        self.local_reference_audio_path: Optional[str] = None

        # Note: __init__ itself cannot be async.
        # Blocking operations during init (like initial model loading in LLM/TTS) will still block here.
        # A more advanced pattern would involve an async factory or an explicit async `await self.async_init()` call.
        # For now, keeping init largely synchronous for LLM/TTS model loading.
        print(f"Initializing CharacterClient for Actor_ID: {self.Actor_id} to connect to SERVER: {self.server_url}")

        # _register_with_server and fetch_traits are blocking but involve network I/O.
        # These could be made async and awaited in an async factory or a separate startup method if desired.
        if not self._register_with_server_blocking():
            print(f"CRITICAL: Client {self.Actor_id} failed to register. Functionality may be impaired.")
        self.character = self._fetch_traits_blocking()
        if not self.character:
            print(f"CRITICAL: Failed to fetch character traits for {self.Actor_id}. Using defaults.")
            self.character = {
                "name": self.Actor_id,
                "personality": "default",
                "tts": "gtts",
                "tts_model": "en",
                "reference_audio_filename": None,
                "language": "en",
                "llm_model": None
            }

        if self.character and self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            self._download_reference_audio_blocking() # Blocking for init
        # Pass speaker_wav_path to TTSManager for XTTSv2
        tts_model = self.character.get("tts_model") or "en"
        tts_service = self.character.get("tts", "gtts")
        tts_language = self.character.get("language", "en")
        speaker_wav_path = self.local_reference_audio_path if tts_service == "xttsv2" else None
        self.tts = TTSManager(
            tts_service_name=tts_service,
            model_name=tts_model,
            language=tts_language,
            speaker_wav_path=speaker_wav_path
        )
        llm_model = self.character.get("llm_model") or ""
        self.llm = LLMEngine(
            model_name=llm_model,
            Actor_id=self.Actor_id
        )

    # --- Blocking I/O methods for use during __init__ or when async context isn't readily available ---
    def _download_reference_audio_blocking(self):
        # (Implementation is the same as original _download_reference_audio, just named for clarity)
        if not self.character:
            print("No character traits available for downloading reference audio.")
            return
        filename = self.character.get("reference_audio_filename")
        if not filename:
            print("No reference audio filename specified in character traits.")
            return
        os.makedirs(CLIENT_TTS_REFERENCE_VOICES_PATH, exist_ok=True)
        sane_Actor_id = "".join(c if c.isalnum() else "_" for c in self.Actor_id)
        sane_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else "_" for c in filename)
        self.local_reference_audio_path = os.path.join(CLIENT_TTS_REFERENCE_VOICES_PATH, f"{sane_Actor_id}_{sane_filename}")

        if os.path.exists(self.local_reference_audio_path) and os.path.getsize(self.local_reference_audio_path) > 0:
            return
        params = {"Actor_id": self.Actor_id, "token": self.token}
        try:
            ref_audio_url = f"{self.server_url}/get_reference_audio/{sane_filename}"
            response = requests.get(ref_audio_url, params=params, stream=True, timeout=30)
            response.raise_for_status()
            with open(self.local_reference_audio_path, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            print(f"Ref audio '{sane_filename}' downloaded to: {self.local_reference_audio_path}")
        except Exception as e:
            print(f"Error downloading ref audio '{sane_filename}' for {self.Actor_id}: {e}")
            self.local_reference_audio_path = None

    def _fetch_traits_blocking(self):
        # (Implementation is the same as original fetch_traits, just named for clarity)
        try:
            response = requests.get(f"{self.server_url}/get_traits", params={"Actor_id": self.Actor_id, "token": self.token}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching traits for {self.Actor_id} (blocking): {e}")
        return None

    def _register_with_server_blocking(self, max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS) -> bool:
        # (Implementation is the same as original _register_with_server with retries, just named for clarity)
        payload = {"Actor_id": self.Actor_id, "token": self.token, "client_port": self.client_port}
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(f"{self.server_url}/register", json=payload, timeout=10)
                response.raise_for_status()
                print(f"Registration for {self.Actor_id}: OK")
                return True
            except Exception as e:
                print(f"Reg attempt {attempt+1} for {self.Actor_id} failed: {e}")
                if attempt < max_retries:
                    time.sleep(base_delay * (2**attempt))
                else:
                    print(f"CRITICAL: Could not register {self.Actor_id} after retries.")
                    return False
        return False
    # --- End of blocking I/O methods ---

    async def send_heartbeat_async(self, max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS) -> bool:
        """Asynchronously sends a heartbeat with retries."""
        payload = {"Actor_id": self.Actor_id, "token": self.token}
        url = f"{self.server_url}/heartbeat"

        def _blocking_heartbeat_call():
            return requests.post(url, json=payload, timeout=5)

        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(_blocking_heartbeat_call)
                response.raise_for_status()
                return True
            except Exception as e:
                print(f"Async Heartbeat attempt {attempt+1} for {self.Actor_id} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2**attempt))
                else:
                    print(f"Async Heartbeat failed for {self.Actor_id} after retries.")
                    return False
        return False

    async def generate_response_async(self, narration: str, character_texts: dict) -> str:
        """Asynchronously generates LLM response and handles fine-tuning placeholder."""
        if not self.llm or not self.llm.is_initialized or not self.character:
            char_name = self.character["name"] if self.character and "name" in self.character else self.Actor_id
            return f"[{char_name} LLM not ready]"

        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in character_texts.items()]) + f"\nCharacter: {self.character['name']} (as {self.character['personality']}):"

        # LLMEngine.generate is now async
        text = await self.llm.generate(prompt)

        # LLMEngine.fine_tune_async is now async (placeholder still)
        await self.llm.fine_tune_async({"input": prompt, "output": text}, self.Actor_id)

        def _save_training_data_blocking():
            requests.post(f"{self.server_url}/save_training_data",
                          json={"dataset": {"input": prompt, "output": text}, "Actor_id": self.Actor_id, "token": self.token},
                          timeout=10)
        try:
            await asyncio.to_thread(_save_training_data_blocking)
        except Exception as e:
            print(f"Error saving training data (async) for {self.Actor_id}: {e}")
        return text

    async def synthesize_audio_async(self, text_to_synthesize: str) -> str | None:
        """Asynchronously synthesizes audio using the TTSManager."""
        if not self.tts or not self.tts.is_initialized:
            print(f"Error: TTS not initialized for {self.Actor_id} (async synthesize).")
            return None
        speaker_wav_to_use = None
        tts_type = self.character["tts"] if self.character and "tts" in self.character else None
        if tts_type == "xttsv2":
            if self.local_reference_audio_path and os.path.exists(self.local_reference_audio_path):
                speaker_wav_to_use = self.local_reference_audio_path
        sane_char_name = "".join(c if c.isalnum() else "_" for c in (self.character["name"] if self.character and "name" in self.character else self.Actor_id))
        audio_filename_no_path = f"{sane_char_name}_{uuid.uuid4()}.wav"
        generated_audio_path = await self.tts.synthesize(
            text_to_synthesize,
            audio_filename_no_path,
            speaker_wav_for_synthesis=speaker_wav_to_use
        )
        return generated_audio_path


@app.post("/character")
async def handle_character_generation_request(data: dict, request: Request):
    """FastAPI endpoint, now fully asynchronous."""
    client: CharacterClient = request.app.state.character_client_instance
    if not client:
        raise HTTPException(status_code=503, detail="CharacterClient not available.")

    if client.token != data.get("token"):
        raise HTTPException(status_code=401, detail="Invalid token.")

    narration = data.get("narration")
    if narration is None:
        raise HTTPException(status_code=400, detail="Missing 'narration'.")

    character_texts = data.get("character_texts", {})

    # Await the async methods
    response_text = await client.generate_response_async(narration, character_texts)
    audio_path = await client.synthesize_audio_async(response_text)

    encoded_audio_data = None
    if audio_path and os.path.exists(audio_path):
        # Reading file can be blocking, run in thread
        def _read_audio_file():
            with open(audio_path, "rb") as f:
                return f.read()
        audio_data = await asyncio.to_thread(_read_audio_file)
        encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')
        try:
            os.remove(audio_path)
        except OSError as e:
            print(f"Error deleting temp audio file {audio_path}: {e}")

    return {"text": response_text, "audio_data": encoded_audio_data}


@app.get("/health", status_code=200)
async def health_check(request: Request):
    client: CharacterClient = request.app.state.character_client_instance
    if not client:
        raise HTTPException(status_code=503, detail="Client service not initialized.")
    llm_status = client.llm.is_initialized if client.llm else False
    tts_status = client.tts.is_initialized if client.tts else False
    status = "ok" if llm_status and tts_status else "degraded"
    detail = "" if status == "ok" else "One or more sub-systems (LLM/TTS) are not fully ready."
    return {"status": status, "Actor_id": client.Actor_id, "llm_ready": llm_status, "tts_ready": tts_status, "detail": detail}


async def _heartbeat_task_runner(client: CharacterClient):
    """Runs send_heartbeat_async periodically."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS) # Use asyncio.sleep
        if client:
            await client.send_heartbeat_async() # Call the async version

# Add start_heartbeat_task for main.py compatibility
_heartbeat_task_instance = None

def start_heartbeat_task(client: CharacterClient):
    global _heartbeat_task_instance
    if _heartbeat_task_instance is None or _heartbeat_task_instance.done():
        loop = asyncio.get_running_loop()
        _heartbeat_task_instance = loop.create_task(_heartbeat_task_runner(client))

def initialize_character_client(token: str, Actor_id: str, server_url: str, client_port: int):
    global _heartbeat_task_instance
    if not hasattr(app.state, 'character_client_instance') or app.state.character_client_instance is None:
        ensure_client_directories()
        client_instance = CharacterClient(token=token, Actor_id=Actor_id, server_url=server_url, client_port=client_port)
        app.state.character_client_instance = client_instance

        llm_ready_msg = client_instance.llm.is_initialized if client_instance.llm else "N/A"
        tts_ready_msg = client_instance.tts.is_initialized if client_instance.tts else "N/A"
        print(f"CharacterClient for {Actor_id} initialized (LLM Ready: {llm_ready_msg}, TTS Ready: {tts_ready_msg}).")

        if _heartbeat_task_instance is None or _heartbeat_task_instance.done():
            if client_instance:
                try:
                    loop = asyncio.get_running_loop()
                    _heartbeat_task_instance = loop.create_task(_heartbeat_task_runner(client_instance))
                    print(f"Async heartbeat task scheduled for client {Actor_id}.")
                except RuntimeError:
                     print(f"WARNING: No running event loop for async heartbeat task of {Actor_id}.")
                     _heartbeat_task_instance = asyncio.create_task(_heartbeat_task_runner(client_instance))
            else:
                 print(f"Heartbeat task NOT scheduled for {Actor_id} due to client init failure.")
    else:
        print(f"CharacterClient for {Actor_id} already initialized.")
