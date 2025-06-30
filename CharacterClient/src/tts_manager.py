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
        """Synchronous/blocking model loading for TTS, called during __init__."""
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
        if gtts and hasattr(gtts, 'gTTS'):
            gtts.gTTS(text=text, lang=lang).save(output_file_path)
        else:
            print("gTTS is not available. Cannot synthesize.")

    def _xttsv2_synthesize_blocking(self, text: str, output_file_path: str, speaker_wav: Optional[str] = None, lang: str = "en"):
        if not self.tts_instance or not hasattr(self.tts_instance, 'tts_to_file'):
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
            if hasattr(self.tts_instance, "tts_to_file") and callable(getattr(self.tts_instance, "tts_to_file")):
                self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_to_use, language=lang_to_use, file_path=output_file_path)
            else:
                print("XTTSv2 instance does not have a callable 'tts_to_file' method.")
        else:
            if speaker_to_use:
                print(f"Client TTSManager (XTTS): Warning - speaker_wav '{speaker_to_use}' not found. Using default voice.")
            if hasattr(self.tts_instance, "tts_to_file") and callable(getattr(self.tts_instance, "tts_to_file")):
                self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_file_path)
            else:
                print("XTTSv2 instance does not have a callable 'tts_to_file' method.")

    async def synthesize(self, text: str, output_filename_no_path: str, speaker_wav_for_synthesis: Optional[str] = None) -> str | None:
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
        target_dir_base = os.path.join(CLIENT_TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)
        if service_name == "xttsv2":
            return model_identifier
        return None

    @staticmethod
    def list_services():
        services = []
        if gtts:
            services.append("gtts")
        if CoquiTTS:
            services.append("xttsv2")
        return services

    @staticmethod
    def get_available_models(service_name: str):
        if service_name == "gtts":
            return ["N/A (uses language codes)"]
        if service_name == "xttsv2":
            return ["tts_models/multilingual/multi-dataset/xtts_v2"]
        return []

if __name__ == "__main__":
    async def test_async_tts_manager():
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
