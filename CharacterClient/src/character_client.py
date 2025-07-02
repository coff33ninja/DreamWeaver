import os
import uuid
import requests
import base64
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
import logging
import hashlib
from datetime import datetime, timezone, timedelta

from .tts_manager import TTSManager # Now async
from .llm_engine import LLMEngine  # Now async
from .config import (
    CLIENT_TTS_REFERENCE_VOICES_PATH,
    ensure_client_directories
)

logger = logging.getLogger("dreamweaver_client") # Get client logger

app = FastAPI()
ensure_client_directories() # This function in config.py should also use logging if it prints

HEARTBEAT_INTERVAL_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 2

class CharacterClient:
    def __init__(self, token: str, Actor_id: str, server_url: str, client_port: int):
        """
        Initialize a CharacterClient instance with the provided authentication and connection details.
        
        This constructor sets up the basic attributes for the client but does not perform any blocking or network operations. To complete initialization, call the asynchronous `async_init()` method after construction.
        
        Parameters:
            token (str): Authentication token for server communication.
            Actor_id (str): Unique identifier for the character actor.
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
        self.session_token: Optional[str] = None
        self.session_token_expiry: Optional[datetime] = None
        # Initialization is now non-blocking. Use async_init() after construction.
        logger.info(f"CharacterClient instance created for Actor_ID: {self.Actor_id}. Call async_init() to complete setup.")

    @classmethod
    async def create(cls, token: str, Actor_id: str, server_url: str, client_port: int):
        """
        Asynchronously creates and initializes a CharacterClient instance.
        
        Returns:
            CharacterClient: An initialized CharacterClient object ready for use.
        """
        self = cls(token, Actor_id, server_url, client_port)
        await self.async_init()
        return self

    async def async_init(self):
        # Perform all blocking/model loading operations here, using asyncio.to_thread for blocking I/O
        """
        Asynchronously initializes the CharacterClient by performing registration, fetching character traits, downloading reference audio if needed, and initializing TTS and LLM subsystems.
        
        This method offloads blocking I/O operations to background threads and sets up the client for subsequent character generation and audio synthesis requests.
        """
        def _register():
            return self._register_with_server_blocking()
        def _fetch():
            """
            Fetches character traits by calling the blocking trait-fetching method.
            
            Returns:
                dict: The character traits retrieved from the server.
            """
            return self._fetch_traits_blocking()
        def _download():
            """
            Downloads the reference audio file for the character if required.
            
            Returns:
                str | None: The local file path to the downloaded reference audio, or None if not applicable.
            """
            return self._download_reference_audio_blocking()

        logger.info(f"CharacterClient ({self.Actor_id}): Starting asynchronous initialization...")
        registered = await asyncio.to_thread(_register)
        if not registered:
            logger.critical(f"CharacterClient ({self.Actor_id}): Failed to register with server after retries. Functionality may be severely impaired.")
        else:
            logger.info(f"CharacterClient ({self.Actor_id}): Successfully registered with server.")

        self.character = await asyncio.to_thread(_fetch)
        if not self.character:
            logger.critical(f"CharacterClient ({self.Actor_id}): Failed to fetch character traits. Using hardcoded defaults.")
            self.character = {
                "name": self.Actor_id, # Default name to Actor_id
                "personality": "default personality",
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
        llm_ready_status = self.llm.is_initialized if self.llm else False
        tts_ready_status = self.tts.is_initialized if self.tts else False
        logger.info(f"CharacterClient for {self.Actor_id} base asynchronous initialization complete. LLM Ready: {llm_ready_status}, TTS Ready: {tts_ready_status}.")

        if registered: # Only attempt handshake if registration was successful
            await self._perform_handshake_async()


    async def _perform_handshake_async(self):
        """
        Performs the handshake protocol with the server to obtain a session token.
        """
        logger.info(f"CharacterClient ({self.Actor_id}): Initiating handshake...")
        try:
            # Step 1: Request challenge
            challenge_url = f"{self.server_url}/request_handshake_challenge"
            # The server endpoint for challenge request is POST and expects Actor_id and token in body
            challenge_payload = {"Actor_id": self.Actor_id, "token": self.token}

            def _post_request_challenge():
                return requests.post(challenge_url, json=challenge_payload, timeout=10)

            response = await asyncio.to_thread(_post_request_challenge)
            response.raise_for_status()
            challenge_data = response.json()
            challenge = challenge_data.get("challenge")

            if not challenge:
                logger.error(f"CharacterClient ({self.Actor_id}): Handshake failed - server did not return a challenge.")
                return

            logger.info(f"CharacterClient ({self.Actor_id}): Received challenge from server.")

            # Step 2: Compute response and submit
            message_to_hash = self.token + challenge
            challenge_response_hash = hashlib.sha256(message_to_hash.encode('utf-8')).hexdigest()

            submit_url = f"{self.server_url}/submit_handshake_response"
            response_payload = {"Actor_id": self.Actor_id, "challenge_response": challenge_response_hash}

            def _post_submit_response():
                return requests.post(submit_url, json=response_payload, timeout=10)

            response = await asyncio.to_thread(_post_submit_response)
            response.raise_for_status()
            session_data = response.json()

            self.session_token = session_data.get("session_token")
            expires_at_str = session_data.get("expires_at")

            if self.session_token and expires_at_str:
                self.session_token_expiry = datetime.fromisoformat(expires_at_str)
                logger.info(f"CharacterClient ({self.Actor_id}): Handshake successful. Session token obtained, expires at {self.session_token_expiry.isoformat()}")
            else:
                logger.error(f"CharacterClient ({self.Actor_id}): Handshake failed - server did not return a valid session token or expiry. Response: {session_data}")
                self.session_token = None
                self.session_token_expiry = None

        except requests.exceptions.HTTPError as e_http:
            if e_http.response is not None:
                logger.error(f"CharacterClient ({self.Actor_id}): Handshake HTTP error {e_http.response.status_code} - {e_http.response.text}", exc_info=True)
            else:
                logger.error(f"CharacterClient ({self.Actor_id}): Handshake HTTP error: {e_http}", exc_info=True)
        except requests.exceptions.RequestException as e_req:
            logger.error(f"CharacterClient ({self.Actor_id}): Handshake network error: {e_req}", exc_info=True)
        except Exception as e:
            logger.error(f"CharacterClient ({self.Actor_id}): Unexpected error during handshake: {e}", exc_info=True)

        if not self.session_token:
             logger.warning(f"CharacterClient ({self.Actor_id}): Handshake did not result in a session token. Will use primary token.")


    def _get_active_token(self) -> str:
        """
        Returns the current active token: session token if valid and available, otherwise the primary token.
        """
        if self.session_token and self.session_token_expiry:
            if datetime.now(timezone.utc) < self.session_token_expiry:
                # logger.debug(f"CharacterClient ({self.Actor_id}): Using active session token.")
                return self.session_token
            else:
                logger.warning(f"CharacterClient ({self.Actor_id}): Session token expired at {self.session_token_expiry}. Invalidating and falling back to primary token. Re-handshake may be needed.")
                self.session_token = None
                self.session_token_expiry = None
                # TODO: Consider triggering re-handshake automatically here or on next API call failure.

        # logger.debug(f"CharacterClient ({self.Actor_id}): Using primary token.")
        return self.token


    # --- Blocking I/O methods for use during __init__ or when async context isn't readily available ---
    def _download_reference_audio_blocking(self):
        # (Implementation is the same as original _download_reference_audio, just named for clarity)
        """
        Downloads the reference audio file for the character and saves it locally if not already present.
        
        If the character traits or reference audio filename are missing, the method exits without downloading. The downloaded file is stored in a sanitized path based on the actor ID and filename. If an error occurs during download, the local reference audio path is set to None.
        """
        if not self.character:
            logger.warning(f"CharacterClient ({self.Actor_id}): No character traits available for downloading reference audio.")
            return
        filename = self.character.get("reference_audio_filename")
        if not filename:
            logger.info(f"CharacterClient ({self.Actor_id}): No reference audio filename specified in character traits. Skipping download.")
            return

        # Ensure CLIENT_TTS_REFERENCE_VOICES_PATH directory exists (config.py should handle this, but being safe)
        try:
            os.makedirs(CLIENT_TTS_REFERENCE_VOICES_PATH, exist_ok=True)
        except OSError as e:
            logger.error(f"CharacterClient ({self.Actor_id}): Could not create reference voices directory {CLIENT_TTS_REFERENCE_VOICES_PATH}: {e}", exc_info=True)
            self.local_reference_audio_path = None
            return

        sane_Actor_id = "".join(c if c.isalnum() else "_" for c in self.Actor_id)
        sane_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else "_" for c in filename)
        self.local_reference_audio_path = os.path.join(CLIENT_TTS_REFERENCE_VOICES_PATH, f"{sane_Actor_id}_{sane_filename}")

        if os.path.exists(self.local_reference_audio_path) and os.path.getsize(self.local_reference_audio_path) > 0:
            logger.debug(f"CharacterClient ({self.Actor_id}): Reference audio '{sane_filename}' already exists locally.")
            return

        active_token = self._get_active_token()
        params = {"Actor_id": self.Actor_id, "token": active_token}
        logger.debug(f"CharacterClient ({self.Actor_id}): Downloading reference audio '{sane_filename}' using {'session' if active_token == self.session_token else 'primary'} token.")
        try:
            ref_audio_url = f"{self.server_url}/get_reference_audio/{sane_filename}"
            response = requests.get(ref_audio_url, params=params, stream=True, timeout=30)
            response.raise_for_status()
            with open(self.local_reference_audio_path, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            logger.info(f"CharacterClient ({self.Actor_id}): Reference audio '{sane_filename}' downloaded to: {self.local_reference_audio_path}")
        except requests.exceptions.RequestException as e_req:
            logger.error(f"CharacterClient ({self.Actor_id}): HTTP Error downloading reference audio '{sane_filename}': {e_req}", exc_info=True)
            self.local_reference_audio_path = None
        except IOError as e_io:
            logger.error(f"CharacterClient ({self.Actor_id}): File I/O Error saving reference audio '{sane_filename}' to {self.local_reference_audio_path}: {e_io}", exc_info=True)
            self.local_reference_audio_path = None
        except Exception as e: # Catch any other unexpected error
            logger.error(f"CharacterClient ({self.Actor_id}): Unexpected error downloading reference audio '{sane_filename}': {e}", exc_info=True)
            self.local_reference_audio_path = None

    def _fetch_traits_blocking(self):
        # (Implementation is the same as original fetch_traits, just named for clarity)
        """
        Fetches character traits from the server in a blocking manner.
        
        Returns:
            dict | None: The character traits as a dictionary if the request succeeds, or None if an error occurs.
        """
        try:
            active_token = self._get_active_token()
            logger.debug(f"CharacterClient ({self.Actor_id}): Fetching traits from {self.server_url}/get_traits using {'session' if active_token == self.session_token else 'primary'} token.")
            response = requests.get(f"{self.server_url}/get_traits", params={"Actor_id": self.Actor_id, "token": active_token}, timeout=10)
            response.raise_for_status()
            traits = response.json()
            logger.info(f"CharacterClient ({self.Actor_id}): Successfully fetched traits: {str(traits)[:200]}...")
            return traits
        except requests.exceptions.RequestException as e_req:
            logger.error(f"CharacterClient ({self.Actor_id}): HTTP Error fetching traits: {e_req}", exc_info=True)
        except Exception as e: # Catch other errors like JSONDecodeError
            logger.error(f"CharacterClient ({self.Actor_id}): Unexpected error fetching traits: {e}", exc_info=True)
        return None

    def _register_with_server_blocking(self, max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS) -> bool:
        # (Implementation is the same as original _register_with_server with retries, just named for clarity)
        """
        Register the client with the server using a blocking HTTP POST request, retrying with exponential backoff on failure.
<<<<<<< HEAD

        Parameters:
        max_retries (int): Maximum number of retry attempts.
        base_delay (float): Initial delay in seconds before retrying, doubled after each failed attempt.

        Returns:
        bool: True if registration succeeds, False if all retries fail.
=======
        
        Parameters:
        	max_retries (int): Maximum number of retry attempts.
        	base_delay (float): Initial delay in seconds before retrying, doubled after each failed attempt.
        
        Returns:
        	bool: True if registration succeeds, False if all retries fail.
>>>>>>> 96fc77cc2cce22f1a9028bf0e9399df2f81b2e3d
        """
        payload = {"Actor_id": self.Actor_id, "token": self.token, "client_port": self.client_port}
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"CharacterClient ({self.Actor_id}): Attempting registration (attempt {attempt+1}/{max_retries+1}) to {self.server_url}/register")
                response = requests.post(f"{self.server_url}/register", json=payload, timeout=10)
                response.raise_for_status()
                logger.info(f"CharacterClient ({self.Actor_id}): Registration successful.")
                return True
            except requests.exceptions.RequestException as e_req:
                logger.warning(f"CharacterClient ({self.Actor_id}): Registration attempt {attempt+1} failed with HTTP error: {e_req}")
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.info(f"CharacterClient ({self.Actor_id}): Retrying registration in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.critical(f"CharacterClient ({self.Actor_id}): Could not register after {max_retries+1} attempts. Last error: {e_req}")
                    return False
            except Exception as e: # Catch any other unexpected error
                logger.error(f"CharacterClient ({self.Actor_id}): Unexpected error during registration attempt {attempt+1}: {e}", exc_info=True)
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.info(f"CharacterClient ({self.Actor_id}): Retrying registration in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.critical(f"CharacterClient ({self.Actor_id}): Could not register after {max_retries+1} attempts due to unexpected error. Last error: {e}")
                    return False
        return False # Should be unreachable if logic is correct, but as a fallback
    # --- End of blocking I/O methods ---

    async def send_heartbeat_async(self, max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS) -> bool:
        """
        Asynchronously sends a heartbeat signal to the server with retry logic and exponential backoff.
        
        Parameters:
        	max_retries (int): Maximum number of retry attempts if the request fails.
        	base_delay (float): Initial delay in seconds before retrying, doubled with each attempt.
        
        Returns:
        	bool: True if the heartbeat was successfully sent, False if all retries failed.
        """
        active_token = self._get_active_token()
        payload = {"Actor_id": self.Actor_id, "token": active_token}
        url = f"{self.server_url}/heartbeat"
        # logger.debug(f"CharacterClient ({self.Actor_id}): Preparing heartbeat with {'session' if active_token == self.session_token else 'primary'} token.")


        def _blocking_heartbeat_call():
            """
            Send a blocking POST request with a JSON payload to the specified URL as part of the heartbeat mechanism.
            
            Returns:
                Response: The HTTP response object from the POST request.
            """
            return requests.post(url, json=payload, timeout=5)

        for attempt in range(max_retries + 1):
            try:
                # logger.debug(f"CharacterClient ({self.Actor_id}): Sending heartbeat (attempt {attempt+1}/{max_retries+1})")
                response = await asyncio.to_thread(_blocking_heartbeat_call)
                response.raise_for_status()
                # logger.debug(f"CharacterClient ({self.Actor_id}): Heartbeat successful.")
                return True
            except requests.exceptions.RequestException as e_req:
                logger.warning(f"CharacterClient ({self.Actor_id}): Async Heartbeat attempt {attempt+1} failed with HTTP error: {e_req}")
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    # logger.debug(f"CharacterClient ({self.Actor_id}): Retrying heartbeat in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"CharacterClient ({self.Actor_id}): Async Heartbeat failed after {max_retries+1} attempts. Last error: {e_req}")
                    return False
            except Exception as e: # Catch any other unexpected error
                logger.error(f"CharacterClient ({self.Actor_id}): Unexpected error during heartbeat attempt {attempt+1}: {e}", exc_info=True)
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    # logger.debug(f"CharacterClient ({self.Actor_id}): Retrying heartbeat in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"CharacterClient ({self.Actor_id}): Async Heartbeat failed after {max_retries+1} attempts due to unexpected error. Last error: {e}")
                    return False
        return False # Should be unreachable

    async def generate_response_async(self, narration: str, character_texts: dict) -> str:
        """
        Asynchronously generates a character-specific response using the LLM and saves training data.
        
        Parameters:
            narration (str): The narration text to include in the prompt.
            character_texts (dict): Additional character dialogue or context to include.
        
        Returns:
            str: The generated response text, or a status message if the LLM or character data is not ready.
        """
        char_name = self.character.get("name", self.Actor_id) if self.character else self.Actor_id

        if not self.llm or not self.llm.is_initialized:
            logger.error(f"CharacterClient ({self.Actor_id}): LLM not initialized. Cannot generate response for narration: '{narration[:50]}...'.")
            return f"[{char_name}_LLM_ERROR:NOT_INITIALIZED]"
        if not self.character: # Should ideally be caught by earlier checks in init
            logger.error(f"CharacterClient ({self.Actor_id}): Character data not available. Cannot generate response.")
            return f"[{char_name}_ERROR:NO_CHARACTER_DATA]"

        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in character_texts.items()]) + f"\nCharacter: {char_name} (as {self.character.get('personality', 'default')}):"
        logger.info(f"CharacterClient ({self.Actor_id}): Generating LLM response for prompt: '{prompt[:100]}...'")

        text = await self.llm.generate(prompt)

        if not text or text.startswith("[LLM_ERROR"):
            logger.error(f"CharacterClient ({self.Actor_id}): LLM generation failed or returned error. Response: '{text}'")
            return text if text else f"[{char_name}_LLM_ERROR:EMPTY_RESPONSE]"

        logger.info(f"CharacterClient ({self.Actor_id}): LLM generated response: '{text[:100]}...'")

        # Fine-tuning (placeholder) and saving training data
        try:
            logger.debug(f"CharacterClient ({self.Actor_id}): Initiating fine-tuning (placeholder).")
            await self.llm.fine_tune_async({"input": prompt, "output": text}, self.Actor_id)
        except Exception as e_tune:
            logger.error(f"CharacterClient ({self.Actor_id}): Error during fine_tune_async: {e_tune}", exc_info=True)

        def _save_training_data_blocking():
            active_token = self._get_active_token()
            logger.debug(f"CharacterClient ({self.Actor_id}): Saving training data to server using {'session' if active_token == self.session_token else 'primary'} token...")
            requests.post(f"{self.server_url}/save_training_data",
                          json={"dataset": {"input": prompt, "output": text}, "Actor_id": self.Actor_id, "token": active_token},
                          timeout=10)
            logger.debug(f"CharacterClient ({self.Actor_id}): Training data save request sent.")
        try:
            await asyncio.to_thread(_save_training_data_blocking)
        except requests.exceptions.RequestException as e_req_save:
            logger.error(f"CharacterClient ({self.Actor_id}): HTTP Error saving training data: {e_req_save}", exc_info=True)
        except Exception as e_save:
            logger.error(f"CharacterClient ({self.Actor_id}): Unexpected error saving training data: {e_save}", exc_info=True)

        return text

    async def synthesize_audio_async(self, text_to_synthesize: str) -> str | None:
        """
        Asynchronously generates an audio file from the provided text using the TTSManager.
        
        Parameters:
            text_to_synthesize (str): The text to convert to speech.
        
        Returns:
            str | None: The file path to the generated audio if successful, or None if TTS is not initialized.
        """
        char_name = self.character.get("name", self.Actor_id) if self.character else self.Actor_id
        if not self.tts or not self.tts.is_initialized:
            logger.error(f"CharacterClient ({self.Actor_id}): TTS not initialized. Cannot synthesize audio for text: '{text_to_synthesize[:50]}...'.")
            return None
        if not text_to_synthesize: # Added check for empty text
            logger.warning(f"CharacterClient ({self.Actor_id}): Text is empty for audio synthesis. Skipping.")
            return None

        speaker_wav_to_use = None
        # Ensure self.character is not None before accessing its items
        tts_type = self.character.get("tts") if self.character else None
        if tts_type == "xttsv2":
            if self.local_reference_audio_path and os.path.exists(self.local_reference_audio_path):
                speaker_wav_to_use = self.local_reference_audio_path
                logger.debug(f"CharacterClient ({self.Actor_id}): Using local reference audio {speaker_wav_to_use} for XTTSv2 synthesis.")
            else:
                logger.warning(f"CharacterClient ({self.Actor_id}): XTTSv2 selected, but local reference audio not found or path invalid ('{self.local_reference_audio_path}'). TTS will use default voice.")

        # TTSManager.synthesize expects a base filename, it constructs the full path in its temp audio dir.
        sane_char_name_for_file = "".join(c if c.isalnum() else "_" for c in char_name)
        base_audio_filename = f"{sane_char_name_for_file}_{uuid.uuid4()}.wav"

        logger.info(f"CharacterClient ({self.Actor_id}): Synthesizing audio for text '{text_to_synthesize[:50]}...' (filename hint: {base_audio_filename})")

        generated_audio_path = await self.tts.synthesize( # This call was already correct based on tts_manager refactor
            text_to_synthesize,
            base_audio_filename,
            speaker_wav_for_synthesis=speaker_wav_to_use
        )

        if generated_audio_path:
            logger.info(f"CharacterClient ({self.Actor_id}): Audio synthesized successfully to {generated_audio_path}")
        else:
            logger.error(f"CharacterClient ({self.Actor_id}): Audio synthesis failed for text '{text_to_synthesize[:50]}...'.") # tts_manager logs details

        return generated_audio_path


@app.post("/character")
async def handle_character_generation_request(data: dict, request: Request):
    """
    Handles character generation requests by generating a character response and synthesizing corresponding audio.
    
    Validates the client instance and token, processes the provided narration and optional character texts, generates a response using the character's LLM, synthesizes audio for the response, encodes the audio in base64, and returns both the generated text and audio data.
    
    Returns:
        dict: A dictionary containing the generated text and base64-encoded audio data.
    """
    client: CharacterClient = request.app.state.character_client_instance # type: ignore
    if not client:
        logger.error("API /character: CharacterClient not available in app state.")
        raise HTTPException(status_code=503, detail="CharacterClient not available. Client may not have initialized correctly.")

    token_received = data.get("token")
    if client.token != token_received:
        logger.warning(f"API /character: Invalid token received for Actor_id {client.Actor_id}.")
        raise HTTPException(status_code=401, detail="Invalid token.")

    narration = data.get("narration")
    if narration is None: # Empty string for narration is allowed, but None is not.
        logger.error(f"API /character: Missing 'narration' in request for Actor_id {client.Actor_id}.")
        raise HTTPException(status_code=400, detail="Missing 'narration'.")

    character_texts = data.get("character_texts", {})
    logger.info(f"API /character: Request received for Actor_id {client.Actor_id}. Narration: '{str(narration)[:100]}...', Character Texts: {str(character_texts)[:100]}...")

    # Await the async methods
    response_text = await client.generate_response_async(narration, character_texts)
    audio_path = await client.synthesize_audio_async(response_text)

    encoded_audio_data = None
    if audio_path and os.path.exists(audio_path):
        # Reading file can be blocking, run in thread
        def _read_audio_file():
            """
            Read and return the binary contents of the audio file at the specified path.
            
            Returns:
                bytes: The raw audio data read from the file.
            """
            with open(audio_path, "rb") as f:
                return f.read()
        audio_data = await asyncio.to_thread(_read_audio_file)
        encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')
        try:
            os.remove(audio_path)
            logger.debug(f"API /character: Successfully deleted temporary audio file {audio_path} for Actor_id {client.Actor_id}.")
        except OSError as e:
            logger.error(f"API /character: Error deleting temporary audio file {audio_path} for Actor_id {client.Actor_id}: {e}", exc_info=True)
    elif audio_path:
        logger.warning(f"API /character: Audio path {audio_path} generated but file does not exist for Actor_id {client.Actor_id}.")
    else:
        logger.info(f"API /character: No audio path generated (synthesis likely failed or skipped) for Actor_id {client.Actor_id}.")


    logger.info(f"API /character: Response for Actor_id {client.Actor_id} - Text: '{response_text[:50]}...', Audio data present: {bool(encoded_audio_data)}")
    return {"text": response_text, "audio_data": encoded_audio_data}


@app.get("/health", status_code=200)
async def health_check(request: Request):
    """
    Check the readiness status of the character client and its LLM and TTS subsystems.
    
    Returns a JSON object indicating overall system status, the actor ID, and the readiness of the LLM and TTS components. If the client is not initialized, raises a 503 HTTPException.
    """
    client: CharacterClient = request.app.state.character_client_instance # type: ignore
    if not client:
        logger.error("API /health: CharacterClient not available in app state for health check.")
        raise HTTPException(status_code=503, detail="Client service not initialized.")

    llm_status = client.llm.is_initialized if client.llm else False
    tts_status = client.tts.is_initialized if client.tts else False
    overall_status = "ok" if llm_status and tts_status else "degraded"
    detail_message = ""
    if overall_status == "degraded":
        degraded_systems = []
        if not llm_status: degraded_systems.append("LLM")
        if not tts_status: degraded_systems.append("TTS")
        detail_message = f"{', '.join(degraded_systems)} not fully ready."

    logger.info(f"API /health for Actor_id {client.Actor_id}: Status: {overall_status}, LLM: {llm_status}, TTS: {tts_status}")
    return {"status": overall_status, "Actor_id": client.Actor_id, "llm_ready": llm_status, "tts_ready": tts_status, "detail": detail_message}


async def _heartbeat_task_runner(client: CharacterClient):
    """
    Periodically sends heartbeat signals for the given CharacterClient instance.
    
    Continuously waits for a configured interval and invokes the client's asynchronous heartbeat method to maintain server registration.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS) # Use asyncio.sleep
        if client:
            # logger.debug(f"Heartbeat task: Sending heartbeat for {client.Actor_id}") # Can be very verbose
            success = await client.send_heartbeat_async() # Call the async version
            if not success:
                logger.warning(f"Heartbeat task: Failed to send heartbeat for {client.Actor_id} after retries.")
        else: # Should not happen if task is managed properly
            logger.error("Heartbeat task: Client instance is None. Stopping task.")
            break


# Add start_heartbeat_task for main.py compatibility
_heartbeat_task_instance = None

def start_heartbeat_task(client: CharacterClient):
    """
    Starts the asynchronous heartbeat task for the given CharacterClient if it is not already running.
    """
    global _heartbeat_task_instance
    if _heartbeat_task_instance is None or _heartbeat_task_instance.done():
        try:
            loop = asyncio.get_running_loop()
            _heartbeat_task_instance = loop.create_task(_heartbeat_task_runner(client))
            logger.info(f"Asynchronous heartbeat task started for {client.Actor_id}.")
        except RuntimeError: # No running event loop
            logger.warning(f"No running asyncio event loop to schedule heartbeat task for {client.Actor_id}. Attempting asyncio.create_task.")
            try:
                # This might be problematic if called from a context where a new loop can't be effectively managed
                _heartbeat_task_instance = asyncio.create_task(_heartbeat_task_runner(client))
                logger.info(f"Asynchronous heartbeat task created (via asyncio.create_task) for {client.Actor_id}.")
            except RuntimeError as e_create_task:
                 logger.error(f"Failed to create heartbeat task for {client.Actor_id} even with asyncio.create_task: {e_create_task}", exc_info=True)
    else:
        logger.debug(f"Heartbeat task for {client.Actor_id} already running or scheduled.")


def initialize_character_client(token: str, Actor_id: str, server_url: str, client_port: int):
    """
    Asynchronously initializes the CharacterClient singleton and starts the heartbeat task if not already running.
    
    Ensures required client directories exist, creates and initializes a CharacterClient instance with the provided parameters, stores it in the FastAPI app state, and schedules the heartbeat background task. If the client is already initialized, no action is taken.
    """
    global _heartbeat_task_instance

    if not hasattr(app.state, 'character_client_instance') or app.state.character_client_instance is None:
        ensure_client_directories()
        # Use the async factory to create and initialize the client
        async def _init():
            """
            Asynchronously initializes the CharacterClient instance and schedules the heartbeat task if not already running.
            
            This function creates and stores a CharacterClient in the FastAPI app state, prints initialization status, and ensures the heartbeat task is started for the client.
            """
            client_instance = await CharacterClient.create(token=token, Actor_id=Actor_id, server_url=server_url, client_port=client_port) # create() logs its own completion
            app.state.character_client_instance = client_instance
            # Logging of LLM/TTS readiness is handled by CharacterClient.async_init itself.
            # Start heartbeat task is now called from main.py after this function, so no need to call it here again.
            # This function's primary role is to create and store the instance.
            logger.info(f"CharacterClient instance for {Actor_id} stored in app.state.")

        # This initialization needs to happen in a way that the event loop is managed correctly.
        # If called from main.py before uvicorn.run, there might not be a running loop.
        # If main.py is the entry point for uvicorn, then this function is likely called before loop starts.
        # A common pattern for FastAPI is to use startup events.
        # For now, assuming this is called from a context that can run or schedule async code.
        logger.info(f"Initializing CharacterClient for {Actor_id}...")
        try:
            # If an event loop is already running (e.g. if called from within an async function managed by uvicorn/FastAPI startup)
            loop = asyncio.get_running_loop()
            loop.create_task(_init()) # Schedule _init to run on the existing loop
            logger.info(f"Scheduled CharacterClient initialization for {Actor_id} on existing event loop.")
        except RuntimeError:
            # No event loop is running, so run _init to completion using asyncio.run()
            # This is suitable if initialize_character_client is called from a synchronous context
            # before the main FastAPI/Uvicorn event loop starts.
            logger.info(f"No running event loop found. Running CharacterClient initialization for {Actor_id} with asyncio.run().")
            asyncio.run(_init())
        # Heartbeat task is started in main.py after this function successfully sets up app.state.character_client_instance
    else:
        logger.info(f"CharacterClient for {Actor_id} already initialized and present in app.state.")
