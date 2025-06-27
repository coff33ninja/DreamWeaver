import os
import torch
from huggingface_hub import hf_hub_download
import asyncio # Added asyncio

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
try:
    import piper.voice as piper_voice_lib
except ImportError:
    piper_voice_lib = None

PIPER_VOICES_REPO = "rhasspy/piper-voices"

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: str = None, language: str = 'en'):
        self.service_name = tts_service_name
        self.model_name = model_name
        self.language = language
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
                self.is_initialized = True
                print(f"Client TTSManager: gTTS configured.")
            else: print("Client TTSManager: Error - gTTS library not found.")
            return

        if not self.model_name:
            print(f"Client TTSManager: Warning - No model name for '{self.service_name}'.")
            return

        model_path_or_name = self._get_or_download_model_blocking(self.service_name, self.model_name)
        if not model_path_or_name:
            print(f"Client TTSManager: Error - Could not get/download model '{self.model_name}'.")
            return

        if self.service_name == "piper":
            if piper_voice_lib:
                try:
                    self.tts_instance = piper_voice_lib.PiperVoice.load(model_path_or_name)
                    self.is_initialized = True
                    print(f"Client TTSManager: Piper initialized with {model_path_or_name}")
                except Exception as e: print(f"Client TTSManager: Error loading Piper model {model_path_or_name}: {e}")
            else: print("Client TTSManager: Error - Piper library not found.")

        elif self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    self.tts_instance = CoquiTTS(model_name=model_path_or_name, progress_bar=True)
                    self.tts_instance.to("cuda" if torch.cuda.is_available() else "cpu")
                    self.is_initialized = True
                    print(f"Client TTSManager: XTTSv2 initialized with {model_path_or_name}. Device: {self.tts_instance.device}")
                except Exception as e: print(f"Client TTSManager: Error initializing XTTSv2 model {model_path_or_name}: {e}")
            else: print("Client TTSManager: Error - Coqui TTS library not found for XTTSv2.")
        # else: print(f"Client TTSManager: Unsupported TTS service '{self.service_name}'.")


    # --- Blocking synthesis methods for use with asyncio.to_thread ---
    def _gtts_synthesize_blocking(self, text: str, output_file_path: str, lang: str):
        gtts.gTTS(text=text, lang=lang).save(output_file_path)

    def _piper_synthesize_blocking(self, text: str, output_file_path: str):
        with open(output_file_path, "wb") as f_out:
            self.tts_instance.synthesize_wav(text, f_out)

    def _xttsv2_synthesize_blocking(self, text: str, output_file_path: str, speaker_wav: str = None, lang: str = "en"):
        lang_to_use = lang
        if hasattr(self.tts_instance, 'languages') and self.tts_instance.languages:
            if lang not in self.tts_instance.languages:
                lang_to_use = self.tts_instance.languages[0]

        if speaker_wav and os.path.exists(speaker_wav):
            self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_wav, language=lang_to_use, file_path=output_file_path)
        else:
            if speaker_wav: print(f"Client TTSManager (XTTS): Warning - speaker_wav '{speaker_wav}' not found. Using default voice.")
            self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_file_path)
    # --- End of blocking synthesis methods ---

    async def synthesize(self, text: str, output_filename_no_path: str, speaker_wav_for_synthesis: str = None) -> str | None:
        if not self.is_initialized:
            print("Client TTSManager: Not initialized, cannot synthesize.")
            return None
        if self.service_name != "gtts" and not self.tts_instance: # gTTS uses a flag, others need instance
             print(f"Client TTSManager: TTS instance for {self.service_name} not available.")
             return None


        os.makedirs(CLIENT_TEMP_AUDIO_PATH, exist_ok=True)
        output_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, output_filename_no_path)
        # print(f"Client TTSManager: Async synthesizing to {output_full_path} using {self.service_name}")

        try:
            if self.service_name == "gtts":
                await asyncio.to_thread(self._gtts_synthesize_blocking, text, output_full_path, self.language)
            elif self.service_name == "piper":
                await asyncio.to_thread(self._piper_synthesize_blocking, text, output_full_path)
            elif self.service_name == "xttsv2":
                await asyncio.to_thread(self._xttsv2_synthesize_blocking, text, output_full_path, speaker_wav_for_synthesis, self.language)
            else:
                print(f"Client TTSManager: No async synthesis method for '{self.service_name}'.")
                return None

            # print(f"Client TTSManager: Async synthesis complete for {output_full_path}.")
            return output_full_path
        except Exception as e:
            print(f"Client TTSManager: Error during async TTS synthesis with {self.service_name} for '{text[:30]}...': {e}")
            if os.path.exists(output_full_path):
                try: os.remove(output_full_path)
                except OSError: pass
            return None

    def _get_or_download_model_blocking(self, service_name: str, model_identifier: str):
        """Synchronous/blocking model download logic."""
        target_dir_base = os.path.join(CLIENT_TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)

        if service_name == "xttsv2":
            # Coqui handles its own downloads to TTS_HOME (CLIENT_TTS_MODELS_PATH)
            return model_identifier

        if service_name == "piper":
            piper_model_dir = os.path.join(target_dir_base, model_identifier)
            onnx_filename = f"{model_identifier}.onnx"
            onnx_path = os.path.join(piper_model_dir, onnx_filename)
            config_path = os.path.join(piper_model_dir, f"{onnx_filename}.json")

            if os.path.exists(onnx_path) and os.path.exists(config_path):
                return onnx_path
            print(f"Client TTSManager: Downloading Piper model '{model_identifier}' to {piper_model_dir}")
            try:
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=onnx_filename, local_dir=piper_model_dir, local_dir_use_symlinks=False)
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=f"{onnx_filename}.json", local_dir=piper_model_dir, local_dir_use_symlinks=False)
                return onnx_path
            except Exception as e:
                print(f"Client TTSManager: Failed to download Piper model '{model_identifier}': {e}")
                return None
        return None

    @staticmethod
    def list_services():
        services = []
        if gtts: services.append("gtts")
        if piper_voice_lib: services.append("piper")
        if CoquiTTS: services.append("xttsv2")
        return services

    @staticmethod
    def get_available_models(service_name: str):
        # This is mostly for UI hints, actual model availability depends on downloads
        if service_name == "gtts": return ["N/A (uses language codes)"]
        if service_name == "piper": return ["en_US-ryan-high", "en_US-lessac-medium", "es_ES-sharvard-medium"]
        if service_name == "xttsv2": return ["tts_models/multilingual/multi-dataset/xtts_v2"]
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
                if out_g_path: print(f"gTTS async test audio saved to {out_g_path}")

        if "piper" in TTSManager.list_services():
            print("\nTesting Piper (async)...")
            tts_p = TTSManager(tts_service_name="piper", model_name="en_US-ryan-high")
            if tts_p.is_initialized:
                out_p_path = await tts_p.synthesize("Hello from client-side Piper, asynchronously.", "client_piper_async_test.wav")
                if out_p_path: print(f"Piper async test audio saved to {out_p_path}")

        if "xttsv2" in TTSManager.list_services():
            print("\nTesting XTTSv2 (async)...")
            tts_x = TTSManager(tts_service_name="xttsv2", model_name="tts_models/multilingual/multi-dataset/xtts_v2", language="en")
            if tts_x.is_initialized:
                # Test default voice (no speaker_wav)
                out_x_path = await tts_x.synthesize("Hello from client Coqui XTTS, async default voice.", "client_xtts_async_default.wav")
                if out_x_path: print(f"XTTSv2 async (default voice) test audio saved to {out_x_path}")
                # Add a dummy speaker wav test if one exists in configured path
                # dummy_speaker_wav = os.path.join(CLIENT_TTS_MODELS_PATH, "tts/reference_voices/client_dummy_speaker.wav")
                # if os.path.exists(dummy_speaker_wav):
                #    out_x_cloned_path = await tts_x.synthesize("This is a cloned voice, asynchronously.", "client_xtts_async_cloned.wav", speaker_wav_for_synthesis=dummy_speaker_wav)
                #    if out_x_cloned_path: print(f"XTTSv2 async (cloned voice) test audio saved to {out_x_cloned_path}")

        print("\n--- Client TTSManager Async Test Complete ---")

    asyncio.run(test_async_tts_manager())
