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
        """
        Initialize a CharacterClient instance with authentication and connection details.
        
        This constructor sets up the basic attributes required for the client, but does not perform any blocking or network operations. Full initialization, including server registration and subsystem setup, must be completed by calling `async_init()` after construction.
        
        Parameters:
            token (str): Authentication token for server communication.
            Actor_id (str): Unique identifier for the character client.
            server_url (str): URL of the server to register and communicate with.
            client_port (int): Port number on which the client operates.
        """
        self.token = token
        self.Actor_id = Actor_id
        self.server_url = server_url
        self.client_port = client_port
        self.character = None
        self.tts = None
        self.llm = None
        self.local_reference_audio_path: Optional[str] = None
        # Initialization is now non-blocking. Use async_init() after construction.
        print(f"CharacterClient created for Actor_ID: {self.Actor_id} (call async_init() to finish setup)")

    @classmethod
    async def create(cls, token: str, Actor_id: str, server_url: str, client_port: int):
        """
        Asynchronously creates and initializes a CharacterClient instance.
        
        This async factory method constructs a CharacterClient and performs all necessary asynchronous initialization steps before returning the ready-to-use instance.
        
        Returns:
            CharacterClient: An initialized CharacterClient object.
        """
        self = cls(token, Actor_id, server_url, client_port)
        await self.async_init()
        return self

    async def async_init(self):
        # Perform all blocking/model loading operations here, using asyncio.to_thread for blocking I/O
        """
        Asynchronously initializes the CharacterClient by performing registration, fetching character traits, downloading reference audio if needed, and initializing TTS and LLM subsystems.
        
        This method offloads blocking I/O operations to background threads to avoid blocking the event loop. If registration or trait fetching fails, it logs critical warnings and applies default character settings. Initializes TTSManager and LLMEngine instances based on the retrieved or default traits.
        """
        def _register():
            return self._register_with_server_blocking()
        def _fetch():
            """
            Fetches character traits from the server using a blocking call.
            
            Returns:
                dict or None: The character traits if successfully fetched, otherwise None.
            """
            return self._fetch_traits_blocking()
        def _download():
            """
            Downloads the reference audio file for the character if required by the TTS model.
            
            Returns:
                The result of the blocking reference audio download operation, or None if not applicable.
            """
            return self._download_reference_audio_blocking()

        registered = await asyncio.to_thread(_register)
        if not registered:
            print(f"CRITICAL: Client {self.Actor_id} failed to register. Functionality may be impaired.")
        self.character = await asyncio.to_thread(_fetch)
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
            await asyncio.to_thread(_download)
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
        print(f"CharacterClient for {self.Actor_id} async-initialized (LLM Ready: {self.llm.is_initialized if self.llm else 'N/A'}, TTS Ready: {self.tts.is_initialized if self.tts else 'N/A'}).")

    # --- Blocking I/O methods for use during __init__ or when async context isn't readily available ---
    def _download_reference_audio_blocking(self):
        # (Implementation is the same as original _download_reference_audio, just named for clarity)
        """
        Downloads the reference audio file for the character from the server and saves it locally if not already present.
        
        If the character traits or reference audio filename are missing, the method exits without downloading. On failure, logs the error and sets the local reference audio path to None.
        """
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
        """
        Fetches character traits from the server in a blocking manner.
        
        Returns:
            dict or None: The character traits as a dictionary if successful, otherwise None.
        """
        try:
            response = requests.get(f"{self.server_url}/get_traits", params={"Actor_id": self.Actor_id, "token": self.token}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching traits for {self.Actor_id} (blocking): {e}")
        return None

    def _register_with_server_blocking(self, max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS) -> bool:
        # (Implementation is the same as original _register_with_server with retries, just named for clarity)
        """
        Attempt to register the client with the server using HTTP POST, retrying with exponential backoff on failure.
        
        Parameters:
        	max_retries (int): Maximum number of registration attempts before giving up.
        	base_delay (int): Base delay in seconds for exponential backoff between retries.
        
        Returns:
        	bool: True if registration succeeds, False if all attempts fail.
        """
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
        """
        Asynchronously sends a heartbeat signal to the server with retry logic and exponential backoff.
        
        Parameters:
        	max_retries (int): Maximum number of retry attempts if the heartbeat fails.
        	base_delay (int): Base delay in seconds for exponential backoff between retries.
        
        Returns:
        	bool: True if the heartbeat was successfully sent, False if all retries failed.
        """
        payload = {"Actor_id": self.Actor_id, "token": self.token}
        url = f"{self.server_url}/heartbeat"

        def _blocking_heartbeat_call():
            """
            Send a blocking HTTP POST request with the specified payload to the configured URL as part of the heartbeat mechanism.
            
            Returns:
                Response: The HTTP response object from the POST request.
            """
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
        """
        Asynchronously generates a character response using the LLM and saves training data for future fine-tuning.
        
        Parameters:
            narration (str): The narration text to include in the prompt.
            character_texts (dict): Dialogue or context from other characters to include in the prompt.
        
        Returns:
            str: The generated response text from the character.
        """
        if not self.llm or not self.llm.is_initialized or not self.character:
            char_name = self.character["name"] if self.character and "name" in self.character else self.Actor_id
            return f"[{char_name} LLM not ready]"

        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in character_texts.items()]) + f"\nCharacter: {self.character['name']} (as {self.character['personality']}):"

        # LLMEngine.generate is now async
        text = await self.llm.generate(prompt)

        # LLMEngine.fine_tune_async is now async (placeholder still)
        await self.llm.fine_tune_async({"input": prompt, "output": text}, self.Actor_id)

        def _save_training_data_blocking():
            """
            Sends a POST request to the server to save training data consisting of the input prompt and generated output text for the current actor.
            """
            requests.post(f"{self.server_url}/save_training_data",
                          json={"dataset": {"input": prompt, "output": text}, "Actor_id": self.Actor_id, "token": self.token},
                          timeout=10)
        try:
            await asyncio.to_thread(_save_training_data_blocking)
        except Exception as e:
            print(f"Error saving training data (async) for {self.Actor_id}: {e}")
        return text

    async def synthesize_audio_async(self, text_to_synthesize: str) -> str | None:
        """
        Asynchronously synthesizes speech audio from the provided text using the TTS engine.
        
        Parameters:
            text_to_synthesize (str): The text to convert to speech.
        
        Returns:
            str | None: The file path to the generated audio if successful, or None if TTS is not initialized.
        """
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
    """
    Handles POST requests to generate a character response and synthesize audio asynchronously.
    
    Validates the provided token and narration, generates a character response using the LLM, synthesizes audio for the response, encodes the audio as base64, and returns both the generated text and audio data.
    
    Returns:
        dict: A dictionary containing the generated response text and base64-encoded audio data.
    """
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
            """
            Read and return the binary contents of the specified audio file.
            
            Returns:
                bytes: The raw binary data of the audio file.
            """
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
    """
    Checks the readiness status of the character client's LLM and TTS subsystems.
    
    Returns:
        dict: A JSON-compatible dictionary containing the overall status ("ok" or "degraded"), the Actor ID, readiness flags for LLM and TTS, and a detail message if any subsystem is not ready.
    """
    client: CharacterClient = request.app.state.character_client_instance
    if not client:
        raise HTTPException(status_code=503, detail="Client service not initialized.")
    llm_status = client.llm.is_initialized if client.llm else False
    tts_status = client.tts.is_initialized if client.tts else False
    status = "ok" if llm_status and tts_status else "degraded"
    detail = "" if status == "ok" else "One or more sub-systems (LLM/TTS) are not fully ready."
    return {"status": status, "Actor_id": client.Actor_id, "llm_ready": llm_status, "tts_ready": tts_status, "detail": detail}


async def _heartbeat_task_runner(client: CharacterClient):
    """
    Periodically sends heartbeat messages for the given CharacterClient instance.
    
    Continuously sleeps for a configured interval and invokes the client's asynchronous heartbeat method to maintain server connectivity.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS) # Use asyncio.sleep
        if client:
            await client.send_heartbeat_async() # Call the async version

# Add start_heartbeat_task for main.py compatibility
_heartbeat_task_instance = None

def start_heartbeat_task(client: CharacterClient):
    """
    Starts the asynchronous heartbeat task for the given CharacterClient if it is not already running.
    
    This ensures that only one heartbeat task is active at a time for the client.
    """
    global _heartbeat_task_instance
    if _heartbeat_task_instance is None or _heartbeat_task_instance.done():
        loop = asyncio.get_running_loop()
        _heartbeat_task_instance = loop.create_task(_heartbeat_task_runner(client))

def initialize_character_client(token: str, Actor_id: str, server_url: str, client_port: int):
    """
    Initializes the CharacterClient instance asynchronously and schedules the heartbeat task if not already running.
    
    If the client is already initialized, this function does nothing. Otherwise, it creates the necessary directories, initializes the CharacterClient with the provided credentials and configuration, stores it in the FastAPI app state, and starts the background heartbeat task to maintain server connectivity.
    """
    global _heartbeat_task_instance
    if not hasattr(app.state, 'character_client_instance') or app.state.character_client_instance is None:
        ensure_client_directories()
        # Use the async factory to create and initialize the client
        async def _init():
            """
            Asynchronously initializes the CharacterClient instance and schedules the heartbeat task if not already running.
            
            This function creates and stores a fully initialized CharacterClient in the FastAPI app state, prints readiness status for LLM and TTS subsystems, and ensures the heartbeat background task is started for the client.
            """
            client_instance = await CharacterClient.create(token=token, Actor_id=Actor_id, server_url=server_url, client_port=client_port)
            app.state.character_client_instance = client_instance
            llm_ready_msg = client_instance.llm.is_initialized if client_instance.llm else "N/A"
            tts_ready_msg = client_instance.tts.is_initialized if client_instance.tts else "N/A"
            print(f"CharacterClient for {Actor_id} initialized (LLM Ready: {llm_ready_msg}, TTS Ready: {tts_ready_msg}).")
            if globals()["_heartbeat_task_instance"] is None or globals()["_heartbeat_task_instance"].done():
                try:
                    loop = asyncio.get_running_loop()
                    globals()["_heartbeat_task_instance"] = loop.create_task(_heartbeat_task_runner(client_instance))
                    print(f"Async heartbeat task scheduled for client {Actor_id}.")
                except RuntimeError:
                    print(f"WARNING: No running event loop for async heartbeat task of {Actor_id}.")
                    globals()["_heartbeat_task_instance"] = asyncio.create_task(_heartbeat_task_runner(client_instance))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_init())
        except RuntimeError:
            asyncio.run(_init())
    else:
        print(f"CharacterClient for {Actor_id} already initialized.")
