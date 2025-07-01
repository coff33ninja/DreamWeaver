import gtts
import os
import torch
import asyncio # Added asyncio
from typing import Optional

from .config import MODELS_PATH

try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    # print("Server TTSManager: Coqui TTS library not found. XTTSv2 will not be available.") # Less verbose

TTS_MODELS_PATH = os.path.join(MODELS_PATH, "tts")

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: Optional[str] = None, speaker_wav_path: Optional[str] = None, language: str = "en"):
        """
        Initialize a TTSManager instance for the specified text-to-speech service and model.
        
        Parameters:
            tts_service_name (str): The name of the TTS service to use ("gtts" or "xttsv2").
            model_name (Optional[str]): The identifier or name of the TTS model, if applicable.
            speaker_wav_path (Optional[str]): Path to a speaker WAV file for voice cloning (XTTSv2 only).
            language (str): Language code for synthesis (default is "en").
        
        Sets up environment variables, prepares model directories, and initializes the selected TTS backend.
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

        # print(f"Server TTSManager: Initializing for service '{self.service_name}' with model '{self.model_name}'")
        self._initialize_service()

    def _initialize_service(self):
        """
        Initializes the TTS service by loading the required model or backend.
        
        Sets up the TTS instance for the selected service ("gtts" or "xttsv2"), handling model loading, device assignment, and error reporting. Marks the manager as initialized if successful.
        """
        if self.service_name == "gtts":
            if gtts:
                self.tts_instance = self._gtts_synthesize_blocking # Store the blocking method
                self.is_initialized = True
                print("Server TTSManager: gTTS initialized.")
            else:
                print("Server TTSManager: Error - gTTS service selected but library not found.")
            return

        if not self.model_name:
            print(f"Server TTSManager: Warning - No model name provided for TTS service '{self.service_name}'.")
            return

        model_path_or_name = self._get_or_download_model_blocking(self.service_name, self.model_name)
        if not model_path_or_name:
            print(f"Server TTSManager: Error - Could not find/download model '{self.model_name}' for '{self.service_name}'.")
            return

        if self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    # Coqui TTS model_name is the identifier it uses for its internal cache/downloader
                    self.tts_instance = CoquiTTS(model_name=model_path_or_name, progress_bar=True)
                    self.tts_instance.to("cuda" if torch.cuda.is_available() else "cpu")
                    self.is_initialized = True
                    print(f"Server TTSManager: Coqui XTTSv2 initialized: {model_path_or_name}. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
                except Exception as e:
                    print(f"Server TTSManager: Error initializing Coqui XTTSv2 model {model_path_or_name}: {e}")
            else:
                print("Server TTSManager: Error - Coqui TTS library not available for XTTSv2.")
        else:
            print(f"Server TTSManager: Unsupported TTS service '{self.service_name}'.")

    def _gtts_synthesize_blocking(self, text: str, output_path: str, lang: str):
        """
        Synchronously synthesizes speech from text using Google Text-to-Speech and saves the result to a file.
        
        Parameters:
            text (str): The text to be converted to speech.
            output_path (str): The file path where the synthesized audio will be saved.
            lang (str): The language code for the speech synthesis.
        """
        gtts.gTTS(text=text, lang=lang).save(output_path)

    def _xttsv2_synthesize_blocking(self, text: str, output_path: str, speaker_wav: Optional[str] = None, lang: str = "en"):
        # Only proceed if tts_instance is valid and not a method (i.e., not gTTS)
        """
        Generate speech audio from text using the XTTSv2 TTS engine and save it to a file.
        
        If a valid speaker WAV file is provided or configured, it is used for voice cloning; otherwise, the default voice is used. If the specified language is not supported by the model, the first available language is selected.
        """
        if not self.tts_instance or callable(self.tts_instance) or not hasattr(self.tts_instance, 'tts_to_file'):
            print("Server TTSManager: XTTSv2 instance is not initialized or invalid.")
            return
        # Ensure lang is always a string
        lang_to_use = lang or "en"
        languages = getattr(self.tts_instance, 'languages', None)
        if languages and lang_to_use not in languages:
            lang_to_use = languages[0]

        speaker_to_use = speaker_wav or self.speaker_wav_path
        if speaker_to_use and isinstance(speaker_to_use, str) and os.path.exists(speaker_to_use):
            self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_to_use, language=lang_to_use, file_path=output_path)
        else:
            if speaker_to_use:
                print(f"Server TTSManager: Warning - XTTSv2 speaker_wav '{speaker_to_use}' not found. Using default voice.")
            self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_path)

    async def synthesize(self, text: str, output_path: str, speaker_wav_for_synthesis: Optional[str] = None) -> bool:
        """
        Asynchronously synthesizes speech from text and saves the audio to the specified output path.
        
        Parameters:
            text (str): The text to be converted to speech.
            output_path (str): The file path where the synthesized audio will be saved.
            speaker_wav_for_synthesis (Optional[str]): Optional path to a speaker WAV file for voice cloning (used with XTTSv2).
        
        Returns:
            bool: True if synthesis succeeds and the audio file is created; False otherwise.
        """
        if not self.is_initialized or not self.tts_instance:
            print("Server TTSManager: Not initialized, cannot synthesize.")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            if self.service_name == "gtts":
                await asyncio.to_thread(self._gtts_synthesize_blocking, text, output_path, self.language)
            elif self.service_name == "xttsv2":
                # Ensure speaker_wav_for_synthesis is a string
                speaker_wav = speaker_wav_for_synthesis or ""
                await asyncio.to_thread(self._xttsv2_synthesize_blocking, text, output_path, speaker_wav, self.language)
            else:
                print(f"Server TTSManager: No async synthesis method for '{self.service_name}'.")
                return False
            return True
        except Exception as e:
            print(f"Server TTSManager: Error during async TTS synthesis with {self.service_name}: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return False

    def _get_or_download_model_blocking(self, service_name, model_identifier):
        """
        Return the model identifier for the specified TTS service, creating the service's model directory if needed.
        
        For the "xttsv2" service, returns the provided model identifier directly, as model management is handled internally by Coqui TTS. Returns None for unsupported services.
        """
        target_dir_base = os.path.join(TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)

        if service_name == "xttsv2":
            # Coqui handles its own downloads; TTS_HOME is set.
            return model_identifier
        return None

    @staticmethod
    def list_services():
        """
        Return a list of available text-to-speech services based on installed libraries.
        
        Returns:
            services (list of str): Names of supported TTS services detected in the environment.
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
        Return a list of available model identifiers or hints for the specified TTS service.
        
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
        Asynchronously tests the TTSManager with both gTTS and XTTSv2 services, synthesizing sample audio files and reporting their output locations.
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
                    print(f"XTTSv2 async test audio saved to {out_x}")

        print("\n--- Server TTSManager Async Test Complete ---")

    asyncio.run(test_tts_manager())
