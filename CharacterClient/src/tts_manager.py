import piper
import xttsv2
import gtts
import os

class TTSManager:
    def __init__(self, tts_service):
        # Simplified for client, assuming models are pre-downloaded or handled externally
        self.services = {
            "piper": piper.Piper("en_US-ryan-high.onnx"), # Placeholder, actual model path needed
            "xttsv2": xttsv2.XTTSv2("generic_model"), # Placeholder, actual model path needed
            "gtts": lambda text, output: gtts.gTTS(text).save(output)
        }
        self.service = self.services.get(tts_service, self.services["piper"])

    def synthesize(self, text, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            if callable(self.service):
                self.service(text, output_path)
            else:
                self.service.synthesize(text, output_path=output_path)
        except Exception as e:
            print(f"Error synthesizing audio with {self.service}: {e}")
