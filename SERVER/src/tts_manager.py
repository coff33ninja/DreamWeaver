import gtts
import os
import torch
import asyncio # Added asyncio
from typing import Optional
import logging

from .config import MODELS_PATH

logger = logging.getLogger("dreamweaver_server")

try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    logger.info("Server TTSManager: Coqui TTS library not found. XTTSv2 will not be available.")

TTS_MODELS_PATH = os.path.join(MODELS_PATH, "tts")

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: Optional[str] = None, speaker_wav_path: Optional[str] = None, language: str = "en"):
        """
        Initialize a TTSManager instance for the specified text-to-speech service, model, and language.

        Parameters:
            tts_service_name (str): The name of the TTS service to use ("gtts" or "xttsv2").
            model_name (Optional[str]): The identifier or name of the TTS model to use, if applicable.
            speaker_wav_path (Optional[str]): Path to a speaker WAV file for voice cloning or speaker adaptation, if supported.
            language (str): Language code for synthesis (default is "en").

        Sets up environment variables and directories for model storage and prepares the TTS backend for synthesis.
        """
        self.service_name = tts_service_name
        self.model_name = model_name or ""  # Ensure string
        self.speaker_wav_path = speaker_wav_path or ""  # Ensure string
        # Always ensure language is a string, never None
        self.language = language or "en" # For gTTS and XTTS
        self.tts_instance = None
        self.is_initialized = False # Flag

        # Set TTS_HOME for Coqui models to be stored within our server's model directory
        os.environ['TTS_HOME'] = MODELS_PATH
        os.makedirs(os.path.join(MODELS_PATH, "tts_models"), exist_ok=True) # Common Coqui subfolder

        logger.info(f"Server TTSManager: Initializing for service '{self.service_name}' with model '{self.model_name}', lang '{self.language}', speaker_wav: '{self.speaker_wav_path}'")
        self._initialize_service()

    def _initialize_service(self):
        """
        Initializes the TTS service by loading the appropriate model or backend.

        Depending on the selected service, this method sets up the gTTS or Coqui XTTSv2 backend, loads the required model, and prepares the TTS instance for synthesis. Logs errors or warnings if initialization fails due to missing libraries, unsupported services, or unavailable models.
        """
        if self.service_name == "gtts":
            if gtts:
                self.tts_instance = self._gtts_synthesize_blocking # Store the blocking method
                self.is_initialized = True
                logger.info("Server TTSManager: gTTS initialized.")
            else:
                logger.error("Server TTSManager: Error - gTTS service selected but gtts library not found.")
            return

        if not self.model_name:
            logger.warning(f"Server TTSManager: No model name provided for TTS service '{self.service_name}'. Initialization may fail or use defaults.")
                self._handle_optional_model_name()

            def _handle_optional_model_name(self):
                """
                Handle cases where a model name might be optional for certain services.
                For Coqui, model_name is usually required.
                """
                pass

        model_path_or_name = self._get_or_download_model_blocking(self.service_name, self.model_name)
        if not model_path_or_name and self.service_name == "xttsv2": # Only critical for xttsv2 if model_name was expected
            logger.error(f"Server TTSManager: Error - Could not determine model path/name for '{self.model_name}' for service '{self.service_name}'.")
            return

        if self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    # Coqui TTS model_name is the identifier it uses for its internal cache/downloader
                    logger.info(f"Server TTSManager: Initializing Coqui XTTSv2 with model: {model_path_or_name or self.model_name}")
                    self.tts_instance = CoquiTTS(model_name=model_path_or_name or self.model_name, progress_bar=True)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.tts_instance.to(device)
                    self.is_initialized = True
                    logger.info(f"Server TTSManager: Coqui XTTSv2 initialized: {model_path_or_name or self.model_name}. Device: {device}")
                except Exception as e:
                    logger.error(f"Server TTSManager: Error initializing Coqui XTTSv2 model {model_path_or_name or self.model_name}: {e}", exc_info=True)
            else:
                logger.error("Server TTSManager: Error - Coqui TTS library not available for XTTSv2.")
        elif self.service_name != "gtts": # gtts is handled above
            logger.error(f"Server TTSManager: Unsupported TTS service '{self.service_name}'.")

    def _gtts_synthesize_blocking(self, text: str, output_path: str, lang: str):
        """
        Synchronously synthesizes speech from text using gTTS and saves the result to a file.

        Parameters:
            text (str): The text to be converted to speech.
            output_path (str): The file path where the synthesized audio will be saved.
            lang (str): The language code for the speech synthesis.
        """
        gtts.gTTS(text=text, lang=lang).save(output_path)

    def _xttsv2_synthesize_blocking(self, text: str, output_path: str, speaker_wav: Optional[str] = None, lang: str = "en"):
        # Only proceed if tts_instance is valid and not a method (i.e., not gTTS)
        """
        Generate speech audio from text using the XTTSv2 (Coqui TTS) backend and save it to a file.

        If a valid speaker WAV file is provided and exists, it is used for voice cloning; otherwise, the default voice is used. If the specified language is not supported by the model, the first available language is used instead.

        Parameters:
            text (str): The input text to synthesize.
            output_path (str): The file path where the synthesized audio will be saved.
            speaker_wav (Optional[str]): Path to a speaker WAV file for voice cloning. If not provided or invalid, the default voice is used.
            lang (str): Language code for synthesis. Defaults to "en".
        """
        if not self.tts_instance or callable(self.tts_instance) or not hasattr(self.tts_instance, 'tts_to_file'):
            logger.error("Server TTSManager: XTTSv2 instance is not initialized or invalid for _xttsv2_synthesize_blocking.")
            return
        # Ensure lang is always a string
        lang_to_use = lang or "en"
        languages = getattr(self.tts_instance, 'languages', None)
        if languages and lang_to_use not in languages:
            lang_to_use = languages[0]

        speaker_to_use = speaker_wav or self.speaker_wav_path
        if speaker_to_use and isinstance(speaker_to_use, str) and os.path.exists(speaker_to_use):
            logger.debug(f"Server TTSManager: Synthesizing XTTSv2 with speaker_wav: {speaker_to_use}, lang: {lang_to_use}")
            self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_to_use, language=lang_to_use, file_path=output_path)
        else:
            if speaker_to_use: # Only log warning if a speaker_wav was intended but not found
                logger.warning(f"Server TTSManager: XTTSv2 speaker_wav '{speaker_to_use}' not found or invalid. Using default voice for lang {lang_to_use}.")
            else:
                logger.debug(f"Server TTSManager: Synthesizing XTTSv2 with default voice for lang {lang_to_use}.")
            self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_path)

    async def synthesize(self, text: str, output_path: str, speaker_wav_for_synthesis: Optional[str] = None) -> bool:
        """
        Asynchronously synthesizes speech from text and saves the result to an audio file.

        Parameters:
            text (str): The input text to synthesize.
            output_path (str): The file path where the synthesized audio will be saved.
            speaker_wav_for_synthesis (Optional[str]): Optional path to a speaker WAV file for voice cloning (used with XTTSv2).

        Returns:
            bool: True if synthesis succeeds and the audio file is created, False otherwise.
        """
        if not self.is_initialized or not self.tts_instance:
            logger.error(f"Server TTSManager ({self.service_name}): Not initialized, cannot synthesize text: '{text[:50]}...'.")
            raise RuntimeError(f"TTSManager ({self.service_name}) is not initialized.")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Server TTSManager: Created output directory {output_dir}")
            except OSError as e:
                logger.error(f"Server TTSManager: Could not create output directory {output_dir}: {e}", exc_info=True)
                return False

        logger.info(f"Server TTSManager: Synthesizing text '{text[:50]}...' to {output_path} using {self.service_name}")
        try:
            if self.service_name == "gtts":
                await asyncio.to_thread(self._gtts_synthesize_blocking, text, output_path, self.language)
            elif self.service_name == "xttsv2":
                # Ensure speaker_wav_for_synthesis is a string
                speaker_wav = speaker_wav_for_synthesis or "" # Default to empty string if None
                await asyncio.to_thread(self._xttsv2_synthesize_blocking, text, output_path, speaker_wav, self.language)
            else:
                logger.error(f"Server TTSManager: No async synthesis method for unsupported service '{self.service_name}'.")
                raise ValueError(f"Unsupported TTS service: {self.service_name}")

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Server TTSManager: Successfully synthesized audio to {output_path}")
                return True
            else:
                logger.error(f"Server TTSManager: Synthesis completed but output file {output_path} is missing or empty.")
                return False
        except Exception as e:
            logger.error(f"Server TTSManager: Error during async TTS synthesis with {self.service_name} for text '{text[:50]}...': {e}", exc_info=True)
            if os.path.exists(output_path): # Attempt to clean up failed/partial file
                try:
                    os.remove(output_path)
                    logger.info(f"Server TTSManager: Removed partial/failed output file {output_path}")
                except OSError as e_remove:
                    logger.error(f"Server TTSManager: Error removing partial/failed output file {output_path}: {e_remove}", exc_info=True)
            return False

    def _get_or_download_model_blocking(self, service_name, model_identifier):
        """
        Return the model identifier for the specified TTS service, creating the service's model directory if needed.

        For the "xttsv2" service, returns the provided model identifier directly, as model management is handled internally by Coqui TTS. Returns None for unsupported services.
        """
        target_dir_base = os.path.join(TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)

        return model_identifier if service_name == "xttsv2" else None

    @staticmethod
    def list_services():
        """
        Return a list of available text-to-speech (TTS) services based on installed libraries.

        Returns:
            services (list): List of supported TTS service names, such as "gtts" and "xttsv2".
        """
        services = []
        if gtts:
            services.append("gtts")
        if CoquiTTS:
            services.append("xttsv2")
        return services

    @staticmethod
    def get_available_models(service_name: str): # This is mostly for UI hints
        """
        Return a list of available model identifiers or descriptors for the specified TTS service.

        Parameters:
            service_name (str): The name of the TTS service ("gtts" or "xttsv2").

        Returns:
            list: A list of model identifiers or UI hints relevant to the service, or an empty list if unsupported.
        """
        if service_name == "gtts":
            return ["N/A (uses language codes)"]
        if service_name == "xttsv2":
            return ["tts_models/multilingual/multi-dataset/xtts_v2"]
        return []

if __name__ == "__main__":
    async def test_tts_manager():
        """
        Asynchronously tests the TTSManager with available TTS services and saves synthesized audio outputs.

        This function creates a test output directory, initializes TTSManager instances for each supported service (gTTS and XTTSv2), and performs asynchronous synthesis of sample text. Synthesized audio files are saved to the test directory, and the function prints status messages for each test.
        """
        print("--- Server TTSManager Async Test ---")
        # Ensure MODELS_PATH (from server config) and subdirs are writable

        test_output_dir = os.path.join(TTS_MODELS_PATH, "test_outputs_server_async")
        os.makedirs(test_output_dir, exist_ok=True)

        if "gtts" in TTSManager.list_services():
            print("\nTesting gTTS (async)...")
            tts_g = TTSManager(tts_service_name="gtts", language="es")
            if tts_g.is_initialized:
                out_g = os.path.join(test_output_dir, "server_gtts_async_test.mp3")
                if await tts_g.synthesize("Hola mundo desde el servidor.", out_g):
                    print(f"gTTS async test audio saved to {out_g}")

        if "xttsv2" in TTSManager.list_services():
            print("\nTesting XTTSv2 (async)...")
            # XTTS model will be downloaded by Coqui library to server's MODELS_PATH/tts_models/
            # For speaker cloning, a speaker_wav would be needed. Test default voice.
            tts_x = TTSManager(tts_service_name="xttsv2", model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="en")
            if tts_x.is_initialized:
                out_x = os.path.join(test_output_dir, "server_xtts_async_test.wav")
                if await tts_x.synthesize("Hello from server-side Coqui XTTS, this is an asynchronous test.", out_x):
                logger.info(f"XTTSv2 async test audio saved to {out_x}")
            else:
                logger.error("XTTSv2 async test synthesis failed.")
        else:
            logger.warning("XTTSv2 (async test) was not initialized, skipping synthesis test.")


        logger.info("\n--- Server TTSManager Async Test Complete ---")

    # Setup basic logging for the test runner if this script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_tts_manager())
