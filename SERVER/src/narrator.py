import whisper
from pyannote.audio import Pipeline

class Narrator:
    def __init__(self):
        self.stt_model = whisper.load_model("base")
        self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization")
        self.speaker = "Narrator"

    def process_narration(self, audio):
        transcription = self.stt_model.transcribe(audio)["text"]
        speakers = self.diarization(audio)
        speaker = speakers[0].speaker if speakers else self.speaker
        return transcription
