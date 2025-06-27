import gtts
import os
import torch
from huggingface_hub import hf_hub_download
from .config import MODELS_PATH # Import from config

# Lazy import TTS to avoid slow startup if not used, and to handle conditional import
try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    print("Warning: Coqui TTS library not found. XTTSv2 will not be available.")

try:
    import piper.voice
except ImportError:
    piper = None
    print("Warning: Piper TTS library not found. Piper will not be available.")


PIPER_VOICES_REPO = "rhasspy/piper-voices"
# MODELS_PATH is already defined in config.py, imported at the top.
# Specific subdirectories for TTS models will be under MODELS_PATH/tts/
TTS_MODELS_PATH = os.path.join(MODELS_PATH, "tts")

class TTSManager:
    def __init__(self, tts_service_name: str, model_name: str = None, speaker_wav_path: str = None):
        self.service_name = tts_service_name
        self.model_name = model_name
        self.speaker_wav_path = speaker_wav_path # For XTTSv2 voice cloning
        self.tts_instance = None

        if self.service_name == "gtts":
            if not gtts:
                print("Error: gTTS library not installed.")
                return
            # For gTTS, the "service" is just a lambda that calls gTTS.
            # No specific model_name is needed for gTTS initialization here.
            self.tts_instance = lambda text, output_file: gtts.gTTS(text=text, lang='en').save(output_file)
            print("TTSManager initialized with gTTS.")
            return

        if not model_name:
            print(f"Warning: No model name provided for TTS service '{tts_service_name}'. TTS will be disabled for this instance.")
            return

        model_path_or_name = self._get_or_download_model(self.service_name, self.model_name)
        if not model_path_or_name:
            print(f"Error: Could not initialize or download model '{self.model_name}' for service '{self.service_name}'.")
            return

        if self.service_name == "piper":
            if piper and piper.voice:
                try:
                    self.tts_instance = piper.voice.PiperVoice.load(model_path_or_name)
                    print(f"Piper TTS initialized with model: {model_path_or_name}")
                except Exception as e:
                    print(f"Error loading Piper model {model_path_or_name}: {e}")
            else:
                print("Error: Piper TTS library not available, but Piper service was selected.")

        elif self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    # Coqui TTS uses the model_name directly (e.g., "tts_models/multilingual/multi-dataset/xtts_v2")
                    # It handles its own model downloading/caching internally based on this name.
                    # The `MODELS_PATH` from config is used by Coqui TTS if `TTS_HOME` env var is set to it.
                    # Ensure Coqui TTS can find/download models to the correct location.
                    # We can also set the env var here if needed, though library often handles defaults.
                    os.environ['TTS_HOME'] = MODELS_PATH # Ensure Coqui knows where to store models
                    self.tts_instance = CoquiTTS(model_name=model_path_or_name, progress_bar=True)
                    self.tts_instance.to("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"Coqui XTTSv2 initialized with model: {model_path_or_name}. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
                except Exception as e:
                    print(f"Error initializing Coqui XTTSv2 model {model_path_or_name}: {e}")
            else:
                print("Error: Coqui TTS library not available, but XTTSv2 service was selected.")
        else:
            print(f"Unsupported TTS service: {self.service_name}")


    def synthesize(self, text: str, output_path: str, speaker_wav_for_synthesis: str = None):
        if not self.tts_instance:
            print("TTS service not initialized, cannot synthesize audio.")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            if self.service_name == "gtts":
                self.tts_instance(text, output_path) # gTTS lambda
                print(f"gTTS synthesized audio to {output_path}")
                return True
            elif self.service_name == "piper" and hasattr(self.tts_instance, 'synthesize_wav'):
                with open(output_path, "wb") as f_out:
                    self.tts_instance.synthesize_wav(text, f_out)
                print(f"Piper TTS synthesized audio to {output_path}")
                return True
            elif self.service_name == "xttsv2" and hasattr(self.tts_instance, 'tts_to_file'):
                # Use the instance's speaker_wav_path if set during init,
                # otherwise use the one passed during synthesis (if any).
                current_speaker_wav = speaker_wav_for_synthesis or self.speaker_wav_path
                if not current_speaker_wav or not os.path.exists(current_speaker_wav):
                    print(f"XTTSv2 requires a speaker WAV. Provided: {current_speaker_wav}. Using default voice.")
                    # Synthesize with the default voice if no speaker_wav is available
                    self.tts_instance.tts_to_file(text=text, file_path=output_path, language=self.tts_instance.languages[0] if self.tts_instance.languages else "en")

                else:
                     self.tts_instance.tts_to_file(text=text, speaker_wav=current_speaker_wav, language=self.tts_instance.languages[0] if self.tts_instance.languages else "en", file_path=output_path)
                print(f"XTTSv2 synthesized audio to {output_path} using speaker: {current_speaker_wav or 'default'}")
                return True
            else:
                print(f"Synthesis method not found or service '{self.service_name}' not properly initialized.")
                return False
        except Exception as e:
            print(f"Error during TTS synthesis with {self.service_name}: {e}")
            return False

    def _get_or_download_model(self, service_name, model_identifier):
        """
        Ensures the specified model is available locally, downloading it if necessary.
        For Piper, model_identifier is like "en_US-ryan-high".
        For XTTSv2, model_identifier is like "tts_models/multilingual/multi-dataset/xtts_v2",
        and Coqui TTS handles its own download/cache, so we just return the identifier.
        """
        if service_name == "xttsv2":
            # Coqui TTS manages its own model downloads/caching.
            # The `model_identifier` is what Coqui uses.
            # We ensure `TTS_HOME` is set so it uses our configured `MODELS_PATH`.
            os.environ['TTS_HOME'] = MODELS_PATH
            os.makedirs(os.path.join(MODELS_PATH, "tts_models"), exist_ok=True) # Ensure base for Coqui models exists
            return model_identifier

        if service_name == "piper":
            # model_identifier for piper is e.g., "en_US-ryan-high"
            # Piper models are stored in MODELS_PATH/tts/piper/<model_identifier>/
            piper_model_dir = os.path.join(TTS_MODELS_PATH, "piper", model_identifier)
            onnx_filename = f"{model_identifier}.onnx"
            onnx_path = os.path.join(piper_model_dir, onnx_filename)
            config_path = os.path.join(piper_model_dir, f"{onnx_filename}.json")

            if os.path.exists(onnx_path) and os.path.exists(config_path):
                print(f"Found existing Piper model: {onnx_path}")
                return onnx_path # Return path to the .onnx file

            print(f"Downloading Piper model: {model_identifier} to {piper_model_dir}")
            os.makedirs(piper_model_dir, exist_ok=True)
            try:
                # Download both the .onnx model and its .json config file
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=onnx_filename, local_dir=piper_model_dir, local_dir_use_symlinks=False)
                hf_hub_download(repo_id=PIPER_VOICES_REPO, filename=f"{onnx_filename}.json", local_dir=piper_model_dir, local_dir_use_symlinks=False)
                print(f"Successfully downloaded Piper model: {model_identifier}")
                return onnx_path
            except Exception as e:
                print(f"Failed to download Piper model '{model_identifier}': {e}")
                return None

        print(f"Model download/check logic not implemented for service: {service_name}")
        return None

    @staticmethod
    def list_services():
        services = []
        if gtts: services.append("gtts")
        if piper: services.append("piper")
        if CoquiTTS: services.append("xttsv2")
        # Add other services like "google" if implemented
        return services

    @staticmethod
    def get_available_models(service_name: str):
        """
        Returns a list of available model names for a given TTS service.
        This might involve checking local directories or providing predefined lists.
        """
        if service_name == "gtts":
            return ["N/A (uses language codes, e.g., 'en')"]

        if service_name == "piper":
            # Example: A curated list. Could also scan local MODELS_PATH/tts/piper/
            # or fetch from Hugging Face Hub for rhasspy/piper-voices
            return [
                "en_US-ryan-high", "en_US-lessac-medium", "en_US-joe-medium",
                "en_GB-alan-low", "en_US-amy-low", "en_US-arctic-medium",
                "fr_FR-upmc-medium", "es_ES-sharvard-medium", "de_DE-thorsten-medium"
                # Add more as needed or implement dynamic discovery
            ]

        if service_name == "xttsv2":
            # XTTSv2 typically has one main model identifier for a version.
            # Coqui TTS library might offer ways to list its available models,
            # but for XTTSv2, it's usually a specific one.
            return ["tts_models/multilingual/multi-dataset/xtts_v2"] # The common identifier

        # if service_name == "google":
        #    # Placeholder: Google TTS has many voices, often identified by codes
        #    # like 'en-US-Standard-A', 'en-GB-Wavenet-F', etc.
        #    # This would require the google-cloud-texttospeech library and setup.
        #    return ["en-US-Standard-A", "en-US-Wavenet-D", "en-GB-Standard-B"]

        return []


if __name__ == "__main__":
    # Example Usage (for testing)
    # Ensure MODELS_PATH is correctly pointing to your desired data/models directory
    # You might need to set the DREAMWEAVER_MODEL_PATH environment variable or modify config.py for direct script run
    print(f"TTS Models will be stored/looked for in: {TTS_MODELS_PATH}")
    os.makedirs(TTS_MODELS_PATH, exist_ok=True)

    # Test Piper (if installed and model exists/can be downloaded)
    if "piper" in TTSManager.list_services():
        print("\nTesting Piper TTS...")
        piper_model_to_test = "en_US-ryan-high" # A common Piper model
        tts_piper = TTSManager(tts_service_name="piper", model_name=piper_model_to_test)
        if tts_piper.tts_instance:
            output_file_piper = os.path.join(TTS_MODELS_PATH, "test_piper_output.wav")
            success = tts_piper.synthesize("Hello from Piper Text to Speech.", output_file_piper)
            if success:
                print(f"Piper test audio saved to {output_file_piper}")
            else:
                print("Piper synthesis failed.")
        else:
            print(f"Could not initialize Piper with model {piper_model_to_test}")

    # Test gTTS (if installed)
    if "gtts" in TTSManager.list_services():
        print("\nTesting gTTS...")
        tts_gtts = TTSManager(tts_service_name="gtts")
        if tts_gtts.tts_instance:
            output_file_gtts = os.path.join(TTS_MODELS_PATH, "test_gtts_output.mp3") # gTTS often saves as mp3
            success = tts_gtts.synthesize("Hello from Google Text to Speech.", output_file_gtts)
            if success:
                print(f"gTTS test audio saved to {output_file_gtts}")
            else:
                print("gTTS synthesis failed.")
        else:
            print("Could not initialize gTTS.")

    # Test XTTSv2 (if Coqui TTS installed and model can be downloaded/found)
    # This requires a reference voice for cloning.
    if "xttsv2" in TTSManager.list_services() and CoquiTTS:
        print("\nTesting XTTSv2 TTS...")
        # Create a dummy reference voice file for testing if you don't have one
        # Ensure REFERENCE_VOICES_AUDIO_PATH is valid and writable from config.py
        from .config import REFERENCE_VOICES_AUDIO_PATH
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)
        dummy_speaker_wav = os.path.join(REFERENCE_VOICES_AUDIO_PATH, "dummy_speaker.wav")

        # Create a simple silent WAV file if one doesn't exist, for testing purposes
        if not os.path.exists(dummy_speaker_wav):
            try:
                import wave
                import numpy as np
                sample_rate = 22050 # XTTS expects 22050 or 24000 Hz
                duration = 1 # 1 second of silence
                n_samples = int(sample_rate * duration)
                silence = np.zeros(n_samples, dtype=np.int16)
                with wave.open(dummy_speaker_wav, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(silence.tobytes())
                print(f"Created dummy speaker WAV for testing: {dummy_speaker_wav}")
            except Exception as e:
                print(f"Could not create dummy speaker WAV: {e}. XTTSv2 test might fail if it needs a speaker file.")


        xtts_model_id = "tts_models/multilingual/multi-dataset/xtts_v2" # Standard XTTSv2 model
        tts_xtts = TTSManager(tts_service_name="xttsv2", model_name=xtts_model_id, speaker_wav_path=dummy_speaker_wav if os.path.exists(dummy_speaker_wav) else None)
        if tts_xtts.tts_instance:
            output_file_xtts = os.path.join(TTS_MODELS_PATH, "test_xtts_output.wav")

            # Test with the speaker_wav specified during synthesis
            success = tts_xtts.synthesize(
                "Hello from Coqui XTTS version 2, this is a voice clone test.",
                output_file_xtts,
                speaker_wav_for_synthesis=dummy_speaker_wav if os.path.exists(dummy_speaker_wav) else None
            )
            if success:
                print(f"XTTSv2 test audio saved to {output_file_xtts}")
            else:
                print("XTTSv2 synthesis failed.")
        else:
            print(f"Could not initialize XTTSv2 with model {xtts_model_id}")

    print("\nTTSManager tests complete.")
