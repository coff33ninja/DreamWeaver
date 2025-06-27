import gtts
import os
import torch
from huggingface_hub import hf_hub_download
import asyncio # Added asyncio

from .config import MODELS_PATH

try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    # print("Server TTSManager: Coqui TTS library not found. XTTSv2 will not be available.") # Less verbose

try:
    import piper.voice as piper_voice_lib # Renamed to avoid conflict
except ImportError:
    piper_voice_lib = None
    # print("Server TTSManager: Piper TTS library not found. Piper will not be available.") # Less verbose

PIPER_VOICES_REPO = "rhasspy/piper-voices"
TTS_MODELS_PATH = os.path.join(MODELS_PATH, "tts")

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: str = None, speaker_wav_path: str = None, language: str = "en"):
        self.service_name = tts_service_name
        self.model_name = model_name
        self.speaker_wav_path = speaker_wav_path
        self.language = language # For gTTS and XTTS
        self.tts_instance = None
        self.is_initialized = False # Flag

        # Set TTS_HOME for Coqui models to be stored within our server's model directory
        os.environ['TTS_HOME'] = MODELS_PATH
        os.makedirs(os.path.join(MODELS_PATH, "tts_models"), exist_ok=True) # Common Coqui subfolder

        # print(f"Server TTSManager: Initializing for service '{self.service_name}' with model '{self.model_name}'")
        self._initialize_service()

    def _initialize_service(self):
        """Blocking part of initialization - loads models."""
        if self.service_name == "gtts":
            if gtts:
                self.tts_instance = self._gtts_synthesize_blocking # Store the blocking method
                self.is_initialized = True
                print(f"Server TTSManager: gTTS initialized.")
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

        if self.service_name == "piper":
            if piper_voice_lib:
                try:
                    self.tts_instance = piper_voice_lib.PiperVoice.load(model_path_or_name)
                    self.is_initialized = True
                    print(f"Server TTSManager: Piper TTS initialized with model: {model_path_or_name}")
                except Exception as e:
                    print(f"Server TTSManager: Error loading Piper model {model_path_or_name}: {e}")
            else:
                print("Server TTSManager: Error - Piper service selected but library not available.")

        elif self.service_name == "xttsv2":
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
        gtts.gTTS(text=text, lang=lang).save(output_path)

    def _piper_synthesize_blocking(self, text: str, output_path: str):
        with open(output_path, "wb") as f_out:
            self.tts_instance.synthesize_wav(text, f_out)

    def _xttsv2_synthesize_blocking(self, text: str, output_path: str, speaker_wav: str = None, lang: str = "en"):
        # Determine language for XTTS, defaulting if necessary
        lang_to_use = lang
        if hasattr(self.tts_instance, 'languages') and self.tts_instance.languages:
            if lang not in self.tts_instance.languages:
                # print(f"Server TTSManager (XTTS): Language '{lang}' not in model languages {self.tts_instance.languages}. Using first: '{self.tts_instance.languages[0]}'")
                lang_to_use = self.tts_instance.languages[0]

        speaker_to_use = speaker_wav or self.speaker_wav_path # Use per-call speaker_wav if provided, else instance default
        if speaker_to_use and os.path.exists(speaker_to_use):
            self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_to_use, language=lang_to_use, file_path=output_path)
            # print(f"Server TTSManager: XTTSv2 synthesized to {output_path} using speaker: {speaker_to_use}")
        else:
            if speaker_to_use: # If provided but not found
                 print(f"Server TTSManager: Warning - XTTSv2 speaker_wav '{speaker_to_use}' not found. Using default voice.")
            self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_path)
            # print(f"Server TTSManager: XTTSv2 synthesized to {output_path} using default voice for language {lang_to_use}.")


    async def synthesize(self, text: str, output_path: str, speaker_wav_for_synthesis: str = None) -> bool:
        if not self.is_initialized or not self.tts_instance:
            print("Server TTSManager: Not initialized, cannot synthesize.")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # print(f"Server TTSManager: Async synthesizing to {output_path} using {self.service_name}")

        try:
            if self.service_name == "gtts":
                await asyncio.to_thread(self._gtts_synthesize_blocking, text, output_path, self.language)
            elif self.service_name == "piper":
                await asyncio.to_thread(self._piper_synthesize_blocking, text, output_path)
            elif self.service_name == "xttsv2":
                await asyncio.to_thread(self._xttsv2_synthesize_blocking, text, output_path, speaker_wav_for_synthesis, self.language)
            else:
                print(f"Server TTSManager: No async synthesis method for '{self.service_name}'.")
                return False

            # print(f"Server TTSManager: Async synthesis complete for {output_path}.")
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
        target_dir_base = os.path.join(TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)

        if service_name == "xttsv2":
            # Coqui handles its own downloads; TTS_HOME is set.
            return model_identifier

        if service_name == "piper":
            piper_model_dir = os.path.join(target_dir_base, model_identifier)
            onnx_filename = f"{model_identifier}.onnx"
            onnx_path = os.path.join(piper_model_dir, onnx_filename)
            config_path = os.path.join(piper_model_dir, f"{onnx_filename}.json")

            if os.path.exists(onnx_path) and os.path.exists(config_path):
                return onnx_path

            print(f"Server TTSManager: Downloading Piper model '{model_identifier}' to {piper_model_dir}")
            try:
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=onnx_filename, local_dir=piper_model_dir, local_dir_use_symlinks=False)
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=f"{onnx_filename}.json", local_dir=piper_model_dir, local_dir_use_symlinks=False)
                return onnx_path
            except Exception as e:
                print(f"Server TTSManager: Failed to download Piper model '{model_identifier}': {e}")
                return None
        return None

    @staticmethod
    def list_services():
        services = []
        if gtts:
            services.append("gtts")
        if piper_voice_lib:
            services.append("piper")
        if CoquiTTS:
            services.append("xttsv2")
        return services

    @staticmethod
    def get_available_models(service_name: str): # This is mostly for UI hints
        if service_name == "gtts":
            return ["N/A (uses language codes)"]
        if service_name == "piper":
            return ["en_US-ryan-high", "en_US-lessac-medium"]
        if service_name == "xttsv2":
            return ["tts_models/multilingual/multi-dataset/xtts_v2"]
        return []

if __name__ == "__main__":
    async def test_tts_manager():
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

        if "piper" in TTSManager.list_services():
            print("\nTesting Piper (async)...")
            # Make sure "en_US-ryan-high" model is downloaded or downloadable to server's MODELS_PATH/tts/piper/
            tts_p = TTSManager(tts_service_name="piper", model_name="en_US-ryan-high")
            if tts_p.is_initialized:
                out_p = os.path.join(test_output_dir, "server_piper_async_test.wav")
                if await tts_p.synthesize("Hello from server-side Piper, asynchronously.", out_p):
                    print(f"Piper async test audio saved to {out_p}")

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
