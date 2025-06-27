import gtts
import os
from huggingface_hub import hf_hub_download
import torch # Added for torch.cuda.is_available()

# Lazy import TTS to avoid slow startup if not used
TTS = None

PIPER_VOICES_REPO = "rhasspy/piper-voices"
MODEL_BASE_PATH = os.getenv("DREAMWEAVER_MODEL_PATH", "E:/DreamWeaver/data/models")

class TTSManager:
    def __init__(self, tts_service, model_name=None):
        self.service = None
        if tts_service == "gtts":
            self.service = lambda text, output: gtts.gTTS(text).save(output)
            return

        if not model_name:
            print(f"Warning: No model name provided for TTS service '{tts_service}'. TTS will be disabled.")
            return

        model_path = self._get_or_download_model(tts_service, model_name)
        if not model_path:
            print(f"Error: Could not find or download model '{model_name}' for service '{tts_service}'.")
            return

        if tts_service == "piper":
            import piper.voice
            self.service = piper.voice.PiperVoice.load(model_path)
        elif tts_service == "xttsv2":
            global TTS
            if TTS is None:
                from TTS.api import TTS
            # For XTTS, the model_name is the path identifier, and the library handles the rest.
            self.service = TTS(model_name, progress_bar=True).to("cuda" if torch.cuda.is_available() else "cpu")

    def synthesize(self, text, output_path):
        if not self.service:
            print("TTS service not initialized, cannot synthesize audio.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if callable(self.service):
            self.service(text, output_path)
        else:
            # Piper has a different synthesis method
            if hasattr(self.service, 'synthesize_wav'):
                with open(output_path, "wb") as f:
                    self.service.synthesize_wav(text, f)
            # Coqui TTS
            else:
                self.service.tts_to_file(text=text, file_path=output_path)

    def _get_or_download_model(self, service, model_name):
        if service == "xttsv2":
            # Coqui TTS handles its own downloads, so we just return the model name.
            return model_name
        if service == "piper":
            model_dir = os.path.join(MODEL_BASE_PATH, "tts", "piper", model_name)
            onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
            if os.path.exists(onnx_path):
                return onnx_path

            print(f"Downloading Piper model: {model_name}")
            os.makedirs(model_dir, exist_ok=True)
            try:
                # Download both the model and the config file
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=f"{model_name}.onnx", local_dir=model_dir, local_dir_use_symlinks=False)
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=f"{model_name}.onnx.json", local_dir=model_dir, local_dir_use_symlinks=False)
                return onnx_path
            except Exception as e:
                print(f"Failed to download Piper model '{model_name}': {e}")
                return None

    @staticmethod
    def list_services():
        return ["piper", "xttsv2", "gtts"]
