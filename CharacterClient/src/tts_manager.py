import os
import torch
from huggingface_hub import hf_hub_download

# Assuming client's config.py is in the same directory or accessible via PYTHONPATH
from .config import CLIENT_TTS_MODELS_PATH, CLIENT_TEMP_AUDIO_PATH, ensure_client_directories

# Ensure directories are created when this module is loaded
ensure_client_directories()

# Lazy import TTS libraries to avoid slow startup if not used or available
try:
    import gtts
except ImportError:
    gtts = None
    print("Client TTSManager: gTTS library not found. gTTS will not be available.")

try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    print("Client TTSManager: Coqui TTS library not found. XTTSv2 will not be available.")

try:
    import piper.voice as piper_voice # Renamed to avoid conflict with local piper var
except ImportError:
    piper_voice = None
    print("Client TTSManager: Piper TTS library not found. Piper will not be available.")


PIPER_VOICES_REPO = "rhasspy/piper-voices"

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: str = None, language: str = 'en'):
        self.service_name = tts_service_name
        self.model_name = model_name
        self.language = language # For gTTS and potentially XTTS
        self.tts_instance = None
        self.is_initialized = False

        # Base path for Coqui TTS models if it needs to be set explicitly for client
        # Coqui typically uses TTS_HOME env var, or its own cache.
        # Setting TTS_HOME to CLIENT_TTS_MODELS_PATH ensures models are client-specific.
        os.environ['TTS_HOME'] = CLIENT_TTS_MODELS_PATH
        os.makedirs(os.path.join(CLIENT_TTS_MODELS_PATH, "tts_models"), exist_ok=True) # Common Coqui subfolder

        print(f"Client TTSManager: Initializing for service '{self.service_name}' with model '{self.model_name}'")

        if self.service_name == "gtts":
            if gtts:
                self.tts_instance = self._init_gtts
                self.is_initialized = True
                print(f"Client TTSManager: gTTS initialized.")
            else:
                print("Client TTSManager: Error - gTTS service selected but library not found.")
            return

        if not model_name:
            print(f"Client TTSManager: Warning - No model name provided for TTS service '{tts_service_name}'. TTS will be disabled.")
            return

        model_path_or_name = self._get_or_download_model(self.service_name, self.model_name)
        if not model_path_or_name:
            print(f"Client TTSManager: Error - Could not find or download model '{self.model_name}' for service '{self.service_name}'.")
            return

        if self.service_name == "piper":
            if piper_voice:
                try:
                    self.tts_instance = piper_voice.PiperVoice.load(model_path_or_name)
                    self.is_initialized = True
                    print(f"Client TTSManager: Piper TTS initialized with model: {model_path_or_name}")
                except Exception as e:
                    print(f"Client TTSManager: Error loading Piper model {model_path_or_name}: {e}")
            else:
                print("Client TTSManager: Error - Piper service selected but library not available.")

        elif self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    self.tts_instance = CoquiTTS(model_name=model_path_or_name, progress_bar=True)
                    self.tts_instance.to("cuda" if torch.cuda.is_available() else "cpu")
                    self.is_initialized = True
                    print(f"Client TTSManager: Coqui XTTSv2 initialized with model: {model_path_or_name}. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
                except Exception as e:
                    print(f"Client TTSManager: Error initializing Coqui XTTSv2 model {model_path_or_name}: {e}")
            else:
                print("Client TTSManager: Error - Coqui TTS library not available, but XTTSv2 service was selected.")
        else:
            print(f"Client TTSManager: Unsupported TTS service '{self.service_name}' selected.")

    def _init_gtts(self, text: str, output_file_path: str, lang: str):
        """Wrapper for gTTS synthesis call."""
        try:
            gtts.gTTS(text=text, lang=lang).save(output_file_path)
            return True
        except Exception as e:
            print(f"Client TTSManager: gTTS synthesis error: {e}")
            return False

    def synthesize(self, text: str, output_filename_no_path: str, speaker_wav_for_synthesis: str = None) -> str | None:
        """
        Synthesizes audio to a file within CLIENT_TEMP_AUDIO_PATH.
        output_filename_no_path is just the filename, not the full path.
        Returns the full path to the synthesized audio file on success, None on failure.
        """
        if not self.is_initialized or not self.tts_instance:
            print("Client TTSManager: Service not initialized, cannot synthesize audio.")
            return None

        os.makedirs(CLIENT_TEMP_AUDIO_PATH, exist_ok=True)
        output_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, output_filename_no_path)

        print(f"Client TTSManager: Synthesizing to {output_full_path} using {self.service_name}")
        success_flag = False
        try:
            if self.service_name == "gtts":
                success_flag = self.tts_instance(text, output_full_path, lang=self.language)
                if success_flag: print(f"Client TTSManager: gTTS synthesized audio to {output_full_path}")

            elif self.service_name == "piper" and hasattr(self.tts_instance, 'synthesize_wav'):
                with open(output_full_path, "wb") as f_out:
                    self.tts_instance.synthesize_wav(text, f_out)
                print(f"Client TTSManager: Piper TTS synthesized audio to {output_full_path}")
                success_flag = True

            elif self.service_name == "xttsv2" and hasattr(self.tts_instance, 'tts_to_file'):
                lang_to_use = self.language
                if hasattr(self.tts_instance, 'languages') and self.tts_instance.languages:
                    if self.language not in self.tts_instance.languages:
                        print(f"Client TTSManager: Language '{self.language}' not in XTTS model languages {self.tts_instance.languages}. Using first available: '{self.tts_instance.languages[0]}'")
                        lang_to_use = self.tts_instance.languages[0]

                if speaker_wav_for_synthesis and os.path.exists(speaker_wav_for_synthesis):
                    self.tts_instance.tts_to_file(text=text, speaker_wav=speaker_wav_for_synthesis, language=lang_to_use, file_path=output_full_path)
                    print(f"Client TTSManager: XTTSv2 synthesized audio to {output_full_path} using speaker: {speaker_wav_for_synthesis}")
                else:
                    if speaker_wav_for_synthesis:
                         print(f"Client TTSManager: Warning - XTTSv2 speaker_wav '{speaker_wav_for_synthesis}' not found. Using default voice.")
                    self.tts_instance.tts_to_file(text=text, language=lang_to_use, file_path=output_full_path)
                    print(f"Client TTSManager: XTTSv2 synthesized audio to {output_full_path} using default voice for language {lang_to_use}.")
                success_flag = True
            else:
                print(f"Client TTSManager: Synthesis method not found or service '{self.service_name}' not properly initialized for synthesis call.")

            return output_full_path if success_flag else None

        except Exception as e:
            print(f"Client TTSManager: Error during TTS synthesis with {self.service_name}: {e}")
            if os.path.exists(output_full_path):
                try: os.remove(output_full_path)
                except OSError: pass
            return None

    def _get_or_download_model(self, service_name: str, model_identifier: str):
        target_dir_base = os.path.join(CLIENT_TTS_MODELS_PATH, service_name.lower())
        os.makedirs(target_dir_base, exist_ok=True)

        if service_name == "xttsv2":
            os.environ['TTS_HOME'] = CLIENT_TTS_MODELS_PATH
            os.makedirs(os.path.join(CLIENT_TTS_MODELS_PATH, "tts_models"), exist_ok=True)
            print(f"Client TTSManager: XTTSv2 model '{model_identifier}' will be handled by Coqui TTS library using TTS_HOME={os.environ.get('TTS_HOME')}.")
            return model_identifier

        if service_name == "piper":
            piper_model_dir = os.path.join(target_dir_base, model_identifier)
            onnx_filename = f"{model_identifier}.onnx"
            onnx_path = os.path.join(piper_model_dir, onnx_filename)
            config_path = os.path.join(piper_model_dir, f"{onnx_filename}.json")

            if os.path.exists(onnx_path) and os.path.exists(config_path):
                print(f"Client TTSManager: Found existing Piper model: {onnx_path}")
                return onnx_path

            print(f"Client TTSManager: Downloading Piper model '{model_identifier}' to {piper_model_dir}")
            os.makedirs(piper_model_dir, exist_ok=True)
            try:
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=onnx_filename, local_dir=piper_model_dir, local_dir_use_symlinks=False)
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=f"{onnx_filename}.json", local_dir=piper_model_dir, local_dir_use_symlinks=False)
                print(f"Client TTSManager: Successfully downloaded Piper model: {model_identifier}")
                return onnx_path
            except Exception as e:
                print(f"Client TTSManager: Failed to download Piper model '{model_identifier}': {e}")
                return None

        print(f"Client TTSManager: Model download/check logic not implemented for service: {service_name}")
        return None

    @staticmethod
    def list_services():
        services = []
        if gtts: services.append("gtts")
        if piper_voice: services.append("piper")
        if CoquiTTS: services.append("xttsv2")
        return services

    @staticmethod
    def get_available_models(service_name: str):
        if service_name == "gtts":
            return ["N/A (uses language codes, e.g., 'en', 'es')"]

        if service_name == "piper":
            return ["en_US-ryan-high", "en_US-lessac-medium", "es_ES-sharvard-medium"]

        if service_name == "xttsv2":
            return ["tts_models/multilingual/multi-dataset/xtts_v2"]

        return []

if __name__ == "__main__":
    print("--- Client TTSManager Test ---")
    print(f"Available TTS services: {TTSManager.list_services()}")

    if "piper" in TTSManager.list_services():
        print("\nTesting Piper TTS...")
        piper_model_id = "en_US-ryan-high"
        tts_piper = TTSManager(tts_service_name="piper", model_name=piper_model_id)
        if tts_piper.is_initialized:
            output_path = tts_piper.synthesize("Hello from client-side Piper TTS.", "client_piper_test.wav")
            if output_path: print(f"Piper test audio saved to {output_path}")
            else: print(f"Piper synthesis failed for {piper_model_id}.")
        else: print(f"Could not initialize Piper TTS with model {piper_model_id}.")

    if "gtts" in TTSManager.list_services():
        print("\nTesting gTTS...")
        tts_gtts = TTSManager(tts_service_name="gtts", language="en")
        if tts_gtts.is_initialized:
            output_path = tts_gtts.synthesize("Hello from client-side Google Text to Speech.", "client_gtts_test.mp3")
            if output_path: print(f"gTTS test audio saved to {output_path}")
            else: print("gTTS synthesis failed.")
        else: print("Could not initialize gTTS.")

    if "xttsv2" in TTSManager.list_services():
        print("\nTesting XTTSv2...")
        xtts_model_id = TTSManager.get_available_models('xttsv2')[0]
        dummy_speaker_dir = os.path.join(CLIENT_TTS_MODELS_PATH, "tts", "reference_voices")
        os.makedirs(dummy_speaker_dir, exist_ok=True)
        dummy_speaker_wav_path = os.path.join(dummy_speaker_dir, "client_dummy_speaker.wav")

        if not os.path.exists(dummy_speaker_wav_path):
            try:
                import wave as pywave
                import numpy as np
                sample_rate = 22050; duration = 1; n_samples = int(sample_rate * duration)
                silence = np.zeros(n_samples, dtype=np.int16)
                with pywave.open(dummy_speaker_wav_path, 'w') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
                    wf.writeframes(silence.tobytes())
                print(f"Created dummy speaker WAV for XTTS testing: {dummy_speaker_wav_path}")
            except Exception as e: print(f"Could not create dummy speaker WAV for XTTS: {e}")

        tts_xtts = TTSManager(tts_service_name="xttsv2", model_name=xtts_model_id, language="en")
        if tts_xtts.is_initialized:
            speaker_to_use = dummy_speaker_wav_path if os.path.exists(dummy_speaker_wav_path) else None
            output_path = tts_xtts.synthesize("Hello from client-side Coqui XTTS version 2.", "client_xtts_test.wav", speaker_wav_for_synthesis=speaker_to_use)
            if output_path: print(f"XTTSv2 test audio saved to {output_path}")
            else: print("XTTSv2 synthesis failed.")
        else: print(f"Could not initialize XTTSv2 with model {xtts_model_id}.")

    print("\n--- Client TTSManager Test Complete ---")
