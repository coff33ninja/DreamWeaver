import os
from whisper import load_model
from pyannote.audio import Pipeline
import asyncio # Added asyncio
import re
import webbrowser
from .config import NARRATOR_AUDIO_PATH, DEFAULT_WHISPER_MODEL_SIZE, DIARIZATION_ENABLED, DIARIZATION_MODEL, MAX_DIARIZATION_RETRIES
import uuid
class Narrator:
    def __init__(self, model_size=None):
        """
        Initialize the Narrator with a Whisper speech-to-text model and optionally a Pyannote diarization pipeline.
        
        Loads the specified or default Whisper model for speech transcription. If diarization is enabled, attempts to load the Pyannote diarization pipeline with retry logic and user prompts for required actions (such as accepting terms or logging in). Sets up internal state for default speaker naming and last transcription storage.
        """
        if model_size is None:
            model_size = DEFAULT_WHISPER_MODEL_SIZE
        print(f"Narrator: Loading Whisper STT model '{model_size}'...")
        try:
            self.stt_model = load_model(model_size)
            print("Narrator: Whisper STT model loaded.")
        except Exception as e:
            print(f"Narrator: Error loading Whisper STT model '{model_size}': {e}")
            self.stt_model = None

        # Diarization can be complex to set up and run, making it optional or simplified
        self.diarization_pipeline = None
        if DIARIZATION_ENABLED:
            # --- Advanced Diarization Pipeline Loader ---
            max_retries = MAX_DIARIZATION_RETRIES
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(f"Narrator: Loading Pyannote Diarization pipeline ({DIARIZATION_MODEL})...")
                    self.diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
                    print("Narrator: Pyannote Diarization pipeline loaded.")
                    break
                except Exception as e:
                    print(f"\n[Narrator] Pyannote Diarization pipeline failed to load (attempt {retry_count+1}/{max_retries}): {e}\n")
                    # --- Open any URLs in the error message (e.g., TOS, token, or gated model pages) ---
                    urls = re.findall(r'https?://[^\s]+', str(e))
                    if urls:
                        for url in urls:
                            print(f"[Narrator] Opening required page: {url}")
                            webbrowser.open(url)
                        print("[Narrator] Please complete any required actions in your browser (e.g., accept TOS, login, or generate a token).\n")
                    else:
                        print("[Narrator] No actionable URLs found in the error message.")
                    user_input = input("Type 'r' to retry, 's' to skip diarization, or just press Enter to retry: ").strip().lower()
                    if user_input == 's':
                        print("[Narrator] Skipping diarization pipeline setup.")
                        self.diarization_pipeline = None
                        break
                    retry_count += 1
            else:
                print("[Narrator] Maximum retries reached. Diarization will be skipped.")
                self.diarization_pipeline = None

        self.default_speaker_name = "Narrator"
        self.last_transcription = None  # Store last transcription for correction

    async def process_narration(self, audio_filepath: str) -> dict:
        """
        Transcribes speech from an audio file and optionally identifies the speaker.
        
        The method saves a uniquely named copy of the input audio file, performs speech-to-text transcription using the loaded Whisper model, and, if enabled and available, applies speaker diarization to determine the speaker label. Returns a dictionary containing the transcribed text, the path to the saved audio copy, and the identified speaker name. If transcription or diarization fails, returns empty text and the default speaker name.
        
        Parameters:
            audio_filepath (str): Path to the input audio file to be processed.
        
        Returns:
            dict: A dictionary with keys:
                - "text": The transcribed text from the audio.
                - "audio_path": Path to the saved copy of the audio file.
                - "speaker": The identified speaker label or default speaker name.
        """
        if not self.stt_model:
            print("Narrator: STT model not loaded. Cannot process narration.")
            return {"text": "", "audio_path": audio_filepath, "speaker": self.default_speaker_name}

        # Save a copy of the audio file to NARRATOR_AUDIO_PATH
        try:
            os.makedirs(NARRATOR_AUDIO_PATH, exist_ok=True)
            ext = os.path.splitext(audio_filepath)[1]
            unique_name = f"narration_{uuid.uuid4().hex}{ext}"
            dest_path = os.path.join(NARRATOR_AUDIO_PATH, unique_name)
            with open(audio_filepath, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            print(f"Narrator: Saved a copy of the audio to {dest_path}")
        except Exception as e:
            print(f"Narrator: Failed to save audio copy: {e}")
            dest_path = audio_filepath  # fallback

        try:
            print(f"Narrator: Transcribing audio file: {audio_filepath}...")
            # Whisper's transcribe is CPU/GPU bound, run in a thread
            transcription_result = await asyncio.to_thread(self.stt_model.transcribe, audio_filepath, fp16=False)
            text_field = transcription_result.get("text", "")
            if isinstance(text_field, list):
                transcribed_text = " ".join(str(s) for s in text_field).strip()
            else:
                transcribed_text = str(text_field).strip()
            print(f"Narrator: Transcription complete. Text: '{transcribed_text[:50]}...'")
            self.last_transcription = transcribed_text  # Store for correction

            # Speaker diarization
            speaker = self.default_speaker_name
            if self.diarization_pipeline and transcribed_text:
                try:
                    print(f"Narrator: Performing diarization on {audio_filepath}...")
                    diarization_output = await asyncio.to_thread(self.diarization_pipeline, audio_filepath)
                    first_turn = next(iter(diarization_output.itertracks(yield_label=True)), None)
                    if first_turn:
                        speaker = first_turn[2]  # speaker label
                    print(f"Narrator: Diarization complete. Determined speaker: {speaker}")
                except Exception as e:
                    print(f"Narrator: Error during diarization: {e}. Using default speaker.")

            return {"text": transcribed_text, "audio_path": dest_path, "speaker": speaker}

        except Exception as e:
            print(f"Narrator: Error processing narration for {audio_filepath}: {e}")
            return {"text": "", "audio_path": dest_path, "speaker": self.default_speaker_name}

    def correct_last_transcription(self, new_text: str):
        """
        Update the most recent transcription with corrected text provided by the user.
        
        Parameters:
            new_text (str): The corrected transcription text to replace the previous value.
        """
        self.last_transcription = new_text
        print(f"Narrator: Last transcription corrected to: {new_text}")

if __name__ == '__main__':
    # This test requires a sample audio file.
    # Create a dummy audio file for testing if you don't have one.
    # E.g., using ffmpeg: ffmpeg -f lavfi -i "anoisesrc=d=5:c=1:r=16000:a=0.1" dummy_narrator_audio.wav

    async def test_narrator():
        """
        Asynchronously tests the Narrator class by generating a dummy audio file and processing it for transcription.
        
        Creates a 1-second 440 Hz WAV file if it does not exist, initializes a Narrator instance with a small model for fast testing, and prints the transcription result. Skips the test if the STT model fails to load or if the dummy audio file cannot be created.
        """
        print("Testing Narrator...")
        narrator_instance = Narrator(model_size="tiny") # Use tiny for faster test
        if not narrator_instance.stt_model:
            print("Skipping test as STT model failed to load.")
            return

        # Create a dummy WAV file for testing if it doesn't exist
        dummy_audio = "dummy_narrator_test_audio.wav"
        if not os.path.exists(dummy_audio):
            try:
                import wave
                import struct
                import math
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
