import os
import whisper
# from pyannote.audio import Pipeline # pyannote.audio can be heavy, let's simplify for now if not strictly needed or make it optional
import asyncio # Added asyncio

# Placeholder for narrator audio saving if not done by whisper itself
# from .config import NARRATOR_AUDIO_PATH # Example, if narrator saves its own audio
# import os
# import uuid

class Narrator:
    def __init__(self, model_size="base"):
        print(f"Narrator: Loading Whisper STT model '{model_size}'...")
        try:
            self.stt_model = whisper.load_model(model_size)
            print("Narrator: Whisper STT model loaded.")
        except Exception as e:
            print(f"Narrator: Error loading Whisper STT model '{model_size}': {e}")
            self.stt_model = None

        # Diarization can be complex to set up and run, making it optional or simplified
        self.diarization_pipeline = None
        # try:
        #     print("Narrator: Loading Pyannote Diarization pipeline...")
        #     self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1") # Or other version
        #     print("Narrator: Pyannote Diarization pipeline loaded.")
        # except Exception as e:
        #     print(f"Narrator: Warning - Pyannote Diarization pipeline failed to load: {e}. Diarization will be skipped.")
        #     self.diarization_pipeline = None

        self.default_speaker_name = "Narrator"

    async def process_narration(self, audio_filepath: str) -> dict:
        """
        Performs Speech-to-Text (STT) on the given audio file.
        Optionally performs diarization if configured.
        Returns a dictionary: {"text": "transcribed text", "audio_path": "path_to_input_audio", "speaker": "speaker_name"}
        """
        if not self.stt_model:
            print("Narrator: STT model not loaded. Cannot process narration.")
            return {"text": "", "audio_path": audio_filepath, "speaker": self.default_speaker_name}

        try:
            print(f"Narrator: Transcribing audio file: {audio_filepath}...")
            # Whisper's transcribe is CPU/GPU bound, run in a thread
            transcription_result = await asyncio.to_thread(self.stt_model.transcribe, audio_filepath, fp16=False) # fp16=False for wider CPU compat
            transcribed_text = transcription_result.get("text", "").strip()
            print(f"Narrator: Transcription complete. Text: '{transcribed_text[:50]}...'")

            # Speaker diarization (simplified/optional for now)
            speaker = self.default_speaker_name
            # if self.diarization_pipeline and transcribed_text:
            #     try:
            #         print(f"Narrator: Performing diarization on {audio_filepath}...")
            #         diarization_output = await asyncio.to_thread(self.diarization_pipeline, audio_filepath)
            #         # Process diarization_output to get speaker label for the main segment.
            #         # This can be complex; for simplicity, we might take the first speaker or most prominent.
            #         # For now, we'll stick to default narrator.
            #         # Example (very basic, needs actual logic from pyannote docs):
            #         # if diarization_output:
            #         #    first_turn = next(iter(diarization_output.itertracks(yield_label=True)), None)
            #         #    if first_turn: speaker = first_turn[2] # speaker label
            #         print(f"Narrator: Diarization complete. Determined speaker (placeholder): {speaker}")
            #     except Exception as e:
            #         print(f"Narrator: Error during diarization: {e}. Using default speaker.")

            return {"text": transcribed_text, "audio_path": audio_filepath, "speaker": speaker}

        except Exception as e:
            print(f"Narrator: Error processing narration for {audio_filepath}: {e}")
            return {"text": "", "audio_path": audio_filepath, "speaker": self.default_speaker_name}

if __name__ == '__main__':
    # This test requires a sample audio file.
    # Create a dummy audio file for testing if you don't have one.
    # E.g., using ffmpeg: ffmpeg -f lavfi -i "anoisesrc=d=5:c=1:r=16000:a=0.1" dummy_narrator_audio.wav

    async def test_narrator():
        print("Testing Narrator...")
        narrator_instance = Narrator(model_size="tiny") # Use tiny for faster test
        if not narrator_instance.stt_model:
            print("Skipping test as STT model failed to load.")
            return

        # Create a dummy WAV file for testing if it doesn't exist
        dummy_audio = "dummy_narrator_test_audio.wav"
        if not os.path.exists(dummy_audio):
            try:
                import wave, struct, math
                sample_rate = 16000.0
                duration = 1 # seconds
                frequency = 440.0 # A4
                num_samples = int(duration * sample_rate)

                with wave.open(dummy_audio, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(sample_rate)
                    for i in range(num_samples):
                        value = int(32767.0 * math.cos(frequency * math.pi * float(i) / float(sample_rate)))
                        data = struct.pack('<h', value)
                        wf.writeframesraw(data)
                print(f"Created dummy audio file: {dummy_audio}")
            except Exception as e:
                print(f"Could not create dummy audio file: {e}. Please create '{dummy_audio}' manually to test.")
                return

        if os.path.exists(dummy_audio):
            result = await narrator_instance.process_narration(dummy_audio)
            print(f"Test Result: {result}")
            # You might want to delete the dummy audio after test
            # os.remove(dummy_audio)
        else:
            print(f"Dummy audio file '{dummy_audio}' not found. Skipping STT test.")

    # Python 3.7+
    asyncio.run(test_narrator())
