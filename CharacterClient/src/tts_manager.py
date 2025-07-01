import os
import torch
import asyncio # Added asyncio
from typing import Optional

from .config import CLIENT_TTS_MODELS_PATH, CLIENT_TEMP_AUDIO_PATH, ensure_client_directories

ensure_client_directories()

try:
    import gtts
except ImportError:
    gtts = None
try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: Optional[str] = None, speaker_wav_path: Optional[str] = None, language: Optional[str] = 'en'):
        """
        Initialize a TTSManager instance for the specified text-to-speech service.
        
        Parameters:
            tts_service_name (str): The name of the TTS backend service to use (e.g., "gtts", "xttsv2").
            model_name (Optional[str]): The model identifier or path for services that require a model (e.g., Coqui XTTSv2).
            speaker_wav_path (Optional[str]): Path to a speaker WAV file for voice cloning, if supported by the service.
            language (Optional[str]): Language code for synthesis (default is "en").
        
        Initializes environment variables and directories required for the selected TTS service and model, and performs blocking service initialization.
        """
        self.service_name = tts_service_name
        self.model_name = model_name or ""
        self.speaker_wav_path = speaker_wav_path or ""
        self.language = language or "en"
        self.tts_instance = None
        self.is_initialized = False

        os.environ['TTS_HOME'] = CLIENT_TTS_MODELS_PATH # For Coqui TTS
        os.makedirs(os.path.join(CLIENT_TTS_MODELS_PATH, "tts_models"), exist_ok=True)

        # print(f"Client TTSManager: Initializing for service '{self.service_name}'...")
        self._initialize_service_blocking() # Keep init blocking for now

    def _initialize_service_blocking(self):
        """
        Synchronously initializes the TTS backend service and loads the required model.
        
        This method sets up the appropriate TTS instance for the selected service (e.g., gTTS or XTTSv2), loading models and configuring devices as needed. It updates the internal initialization state and prints status or error messages if dependencies or models are missing.
        """
        if self.service_name == "gtts":
            if gtts:
                self.tts_instance = self._gtts_synthesize_blocking
                self.is_initialized = True
                print("Client TTSManager: gTTS configured.")
            else:
                print("Client TTSManager: Error - gTTS library not found.")
            return

        if not self.model_name:
            print(f"Client TTSManager: Warning - No model name for '{self.service_name}'.")
            return

        model_path_or_name = self._get_or_download_model_blocking(self.service_name, self.model_name)
        if not model_path_or_name:
            print(f"Client TTSManager: Error - Could not get/download model '{self.model_name}'.")
            return

        if self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    self.tts_instance = CoquiTTS(model_name=model_path_or_name, progress_bar=True)
                    self.tts_instance.to("cuda" if torch.cuda.is_available() else "cpu")
                    self.is_initialized = True
                    print(f"Client TTSManager: XTTSv2 initialized with {model_path_or_name}. Device: {self.tts_instance.device}")
                except Exception as e:
                    print(f"Client TTSManager: Error initializing XTTSv2 model {model_path_or_name}: {e}")
            else:
                print("Client TTSManager: Error - Coqui TTS library not found for XTTSv2.")
        # else: print(f"Client TTSManager: Unsupported TTS service '{self.service_name}'.")

    def _gtts_synthesize_blocking(self, text: str, output_file_path: str, lang: str):
        """
        Synchronously synthesizes speech from text using the gTTS service and saves it to a file.
        
        Parameters:
            text (str): The text to be converted to speech.
            output_file_path (str): The file path where the synthesized audio will be saved.
            lang (str): The language code for the speech synthesis.
        """
        if gtts and hasattr(gtts, 'gTTS'):
            gtts.gTTS(text=text, lang=lang).save(output_file_path)
        else:
            print("gTTS is not available. Cannot synthesize.")

    def _xttsv2_synthesize_blocking(self, text: str, output_file_path: str, speaker_wav: Optional[str] = None, lang: str = "en"):
        """
        Synchronously synthesizes speech from text using the Coqui XTTSv2 backend and saves it to a file.
        
        If a valid speaker WAV file is provided and exists, it is used for voice cloning; otherwise, the default voice is used. If the requested language is not supported, the first available language is selected.
        """
        if self.tts_instance is None or not hasattr(self.tts_instance, 'tts_to_file') or not callable(getattr(self.tts_instance, 'tts_to_file')):
            print("XTTSv2 instance is not available or invalid. Cannot synthesize.")
            return
        lang_to_use = lang or "en"
        # Only check languages if attribute exists and is not a method
        languages = getattr(self.tts_instance, 'languages', None)
        if languages and isinstance(languages, (list, tuple)):
            if lang_to_use not in languages:
                lang_to_use = languages[0]
        speaker_to_use = speaker_wav or self.speaker_wav_path
        if speaker_to_use and isinstance(speaker_to_use, str) and os.path.exists(speaker_to_use):
            self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_to_use, language=lang_to_use, file_path=output_file_path)
        else:
            if speaker_to_use:
                print(f"Client TTSManager (XTTS): Warning - speaker_wav '{speaker_to_use}' not found. Using default voice.")
            self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_file_path)

    async def synthesize(self, text: str, output_filename_no_path: str, speaker_wav_for_synthesis: Optional[str] = None) -> str | None:
        """
        Asynchronously synthesizes speech from text and saves the audio to a file using the configured TTS backend.
        
        Parameters:
            text (str): The text to be converted to speech.
            output_filename_no_path (str): The output audio filename (without directory path).
            speaker_wav_for_synthesis (Optional[str]): Optional path to a speaker WAV file for voice cloning (used by supported services).
        
        Returns:
            str | None: The full path to the generated audio file on success, or None if synthesis fails.
        """
        if not self.is_initialized or not self.tts_instance:
            print("Client TTSManager: Not initialized, cannot synthesize.")
            return None
        os.makedirs(CLIENT_TEMP_AUDIO_PATH, exist_ok=True)
        output_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, output_filename_no_path)
        try:
            if self.service_name == "gtts":
                await asyncio.to_thread(self._gtts_synthesize_blocking, text, output_full_path, self.language)
            elif self.service_name == "xttsv2":
                speaker_wav = speaker_wav_for_synthesis or ""
                await asyncio.to_thread(self._xttsv2_synthesize_blocking, text, output_full_path, speaker_wav, self.language)
            else:
                print(f"Client TTSManager: No async synthesis method for '{self.service_name}'.")
                return None
            return output_full_path
        except Exception as e:
            print(f"Client TTSManager: Error during async TTS synthesis with {self.service_name} for '{text[:30]}...': {e}")
            if os.path.exists(output_full_path):
                try:
                    os.remove(output_full_path)
                except OSError:
                    pass
            return None

    def _get_or_download_model_blocking(self, service_name: str, model_identifier: str):
        """
        Return the model identifier for the specified service, creating the service's model directory if needed.
        
        For the "xttsv2" service, this method returns the provided model identifier directly. For other services, it returns None.
        """
        target_dir_base = os.path.join(CLIENT_TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)
        if service_name == "xttsv2":
            return model_identifier
        return None

    @staticmethod
    def list_services():
        """
        Return a list of available text-to-speech (TTS) services based on installed libraries.
        
        Returns:
            services (list of str): Names of supported TTS services available in the current environment.
        """
        services = []
        if gtts:
            services.append("gtts")
        if CoquiTTS:
            services.append("xttsv2")
        return services

    @staticmethod
    def get_available_models(service_name: str):
        """
        Return a list of available model identifiers for the specified TTS service.
        
        Parameters:
            service_name (str): The name of the TTS service ("gtts" or "xttsv2").
        
        Returns:
            list[str]: A list of available model names or identifiers for the given service. Returns an empty list if the service is unsupported.
        """
        if service_name == "gtts":
            return ["N/A (uses language codes)"]
        if service_name == "xttsv2":
            return ["tts_models/multilingual/multi-dataset/xtts_v2"]
        return []

if __name__ == "__main__":
    async def test_async_tts_manager():
        """
        Asynchronously tests the TTSManager with available TTS services and outputs synthesized audio files.
        
        This function initializes TTSManager instances for each supported backend (gTTS and XTTSv2), synthesizes sample text asynchronously, and prints the paths to the generated audio files. Output files are saved in the configured temporary audio directory.
        """
        print("--- Client TTSManager Async Test ---")
        test_output_dir = CLIENT_TEMP_AUDIO_PATH # Use configured temp path
        os.makedirs(test_output_dir, exist_ok=True)
        print(f"Test outputs will be in: {test_output_dir}")

        if "gtts" in TTSManager.list_services():
            print("\nTesting gTTS (async)...")
            tts_g = TTSManager(tts_service_name="gtts", language="fr")
            if tts_g.is_initialized:
                out_g_path = await tts_g.synthesize("Bonjour le monde, de manière asynchrone.", "client_gtts_async_test.mp3")
                if out_g_path:
                    print(f"gTTS async test audio saved to {out_g_path}")

        if "xttsv2" in TTSManager.list_services():
            print("\nTesting XTTSv2 (async)...")
            tts_x = TTSManager(tts_service_name="xttsv2", model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="en")
            if tts_x.is_initialized:
                out_x_path = await tts_x.synthesize("Hello from client Coqui XTTS, async default voice.", "client_xtts_async_default.wav")
                if out_x_path:
                    print(f"XTTSv2 async (default voice) test audio saved to {out_x_path}")
                # Add a dummy speaker wav test if one exists in configured path
                # dummy_speaker_wav = os.path.join(CLIENT_TTS_MODELS_PATH, "tts/reference_voices/client_dummy_speaker.wav")
                # if os.path.exists(dummy_speaker_wav):
                #    out_x_cloned_path = await tts_x.synthesize("This is a cloned voice, asynchronously.", "client_xtts_async_cloned.wav", speaker_wav_for_synthesis=dummy_speaker_wav)
                #    if out_x_cloned_path: print(f"XTTSv2 async (cloned voice) test audio saved to {out_x_cloned_path}")

        print("\n--- Client TTSManager Async Test Complete ---")

    asyncio.run(test_async_tts_manager())
