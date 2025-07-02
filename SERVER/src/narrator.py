import os
from whisper import load_model
from pyannote.audio import Pipeline
import asyncio # Added asyncio
import re
import webbrowser
from .config import NARRATOR_AUDIO_PATH, DEFAULT_WHISPER_MODEL_SIZE, DIARIZATION_ENABLED, DIARIZATION_MODEL, MAX_DIARIZATION_RETRIES
import uuid
import logging

logger = logging.getLogger("dreamweaver_server")

class Narrator:
    def __init__(self, model_size=None):
        """
        Initialize the Narrator with a Whisper speech-to-text model and optionally a Pyannote diarization pipeline.
        
        Loads the specified or default Whisper model for speech transcription. If diarization is enabled, attempts to load the Pyannote diarization pipeline with retry logic and user prompts for required actions (such as accepting terms or logging in). Sets up internal state for default speaker naming and last transcription storage.
        """
        if model_size is None:
            model_size = DEFAULT_WHISPER_MODEL_SIZE
        logger.info(f"Narrator: Loading Whisper STT model '{model_size}'...")
        try:
            self.stt_model = load_model(model_size)
            logger.info("Narrator: Whisper STT model loaded successfully.")
        except Exception as e:
            logger.error(f"Narrator: Error loading Whisper STT model '{model_size}': {e}", exc_info=True)
            self.stt_model = None

        self.diarization_pipeline = None
        if DIARIZATION_ENABLED:
            max_retries = MAX_DIARIZATION_RETRIES
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.info(f"Narrator: Loading Pyannote Diarization pipeline ({DIARIZATION_MODEL})...")
                    self.diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
                    logger.info("Narrator: Pyannote Diarization pipeline loaded successfully.")
                    break
                except Exception as e:
                    logger.warning(f"\n[Narrator] Pyannote Diarization pipeline failed to load (attempt {retry_count+1}/{max_retries}): {e}\n", exc_info=True)
                    urls = re.findall(r'https?://[^\s]+', str(e))
                    if urls:
                        for url in urls:
                            logger.info(f"[Narrator] Opening required page for diarization setup: {url}")
                            webbrowser.open(url)
                        logger.info("[Narrator] Please complete any required actions in your browser (e.g., accept TOS, login, or generate a token).\n")
                    else:
                        logger.info("[Narrator] No actionable URLs found in the diarization error message.")

                    # Allowing auto-retry without input for server environment, or could make this configurable
                    logger.info(f"Retrying diarization pipeline load in a moment (attempt {retry_count+1})...")
                    # For a server, direct input isn't ideal. We might log and retry, or skip.
                    # For now, let's assume it might be run interactively during setup, or we rely on pre-configuration.
                    # If running headless, this input prompt should be removed or handled differently.
                    # For simplicity in this refactor, I'll keep the retry logic but note that interactive input
                    # is problematic for a server. A better approach might be to fail diarization setup if not pre-configured.
                    # user_input = input("Type 'r' to retry, 's' to skip diarization, or just press Enter to retry: ").strip().lower()
                    # if user_input == 's':
                    # logger.info("[Narrator] Skipping diarization pipeline setup based on configuration or error.")
                    # self.diarization_pipeline = None
                    # break
                    retry_count += 1
                    if retry_count < max_retries:
                         asyncio.run(asyncio.sleep(5)) # Non-blocking sleep if in async context, but __init__ is sync
                    else:
                        logger.error("[Narrator] Maximum retries reached for diarization pipeline. Diarization will be skipped.")
                        self.diarization_pipeline = None
                        break # Exit loop
            if retry_count >= max_retries and self.diarization_pipeline is None: # Ensure it's None if all retries failed
                 logger.error("[Narrator] Maximum retries reached. Diarization will be skipped.")
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
            logger.error("Narrator: STT model not loaded. Cannot process narration.")
            return {"text": "", "audio_path": audio_filepath, "speaker": self.default_speaker_name}

        dest_path = audio_filepath # Default if copy fails
        try:
            os.makedirs(NARRATOR_AUDIO_PATH, exist_ok=True)
            ext = os.path.splitext(audio_filepath)[1]
            unique_name = f"narration_{uuid.uuid4().hex}{ext}"
            dest_path = os.path.join(NARRATOR_AUDIO_PATH, unique_name)
            with open(audio_filepath, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            logger.info(f"Narrator: Saved a copy of the input audio to {dest_path}")
        except Exception as e:
            logger.error(f"Narrator: Failed to save audio copy from {audio_filepath} to {dest_path}: {e}", exc_info=True)
            # Fallback to original path is already handled by dest_path initialization

        try:
            logger.info(f"Narrator: Transcribing audio file: {dest_path} (original: {audio_filepath})...")
            transcription_result = await asyncio.to_thread(self.stt_model.transcribe, dest_path, fp16=False)
            text_field = transcription_result.get("text", "")
            if isinstance(text_field, list): # Should not happen with standard whisper output
                transcribed_text = " ".join(str(s) for s in text_field).strip()
                logger.warning(f"Narrator: Transcription result text was a list, joined: '{transcribed_text[:50]}...'")
            else:
                transcribed_text = str(text_field).strip()
            logger.info(f"Narrator: Transcription complete. Text: '{transcribed_text[:50]}...'")
            self.last_transcription = transcribed_text

            speaker = self.default_speaker_name
            if self.diarization_pipeline and transcribed_text:
                try:
                    logger.info(f"Narrator: Performing diarization on {dest_path}...")
                    diarization_output = await asyncio.to_thread(self.diarization_pipeline, dest_path)
                    speaker_durations = {}
                    for turn, _, label in diarization_output.itertracks(yield_label=True):
                        duration = turn.end - turn.start
                        speaker_durations[label] = speaker_durations.get(label, 0) + duration
                    if speaker_durations:
                        speaker = max(speaker_durations, key=speaker_durations.get)
                    logger.info(f"Narrator: Diarization complete. Determined speaker: {speaker}")
                except Exception as e_diar:
                    logger.error(f"Narrator: Error during diarization for {dest_path}: {e_diar}. Using default speaker.", exc_info=True)

            return {"text": transcribed_text, "audio_path": dest_path, "speaker": speaker}

        except Exception as e_transcribe:
            logger.error(f"Narrator: Error processing narration for {dest_path}: {e_transcribe}", exc_info=True)
            return {"text": "", "audio_path": dest_path, "speaker": self.default_speaker_name}

    def correct_last_transcription(self, new_text: str):
        """
        Update the most recent transcription with corrected text provided by the user.
        
        Parameters:
            new_text (str): The corrected transcription text to replace the previous value.
        """
        self.last_transcription = new_text
        logger.info(f"Narrator: Last transcription corrected to: {new_text[:100]}...")

if __name__ == '__main__':
    # This test requires a sample audio file.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')

    async def test_narrator():
        """
        Asynchronously tests the Narrator class by generating a dummy audio file and processing it for transcription.
        
        Creates a 1-second 440 Hz WAV file if it does not exist, initializes a Narrator instance with a small model for fast testing, and prints the transcription result. Skips the test if the STT model fails to load or if the dummy audio file cannot be created.
        """
        logger.info("Testing Narrator...")
        narrator_instance = Narrator(model_size="tiny")
        if not narrator_instance.stt_model:
            logger.warning("Skipping Narrator test as STT model failed to load.")
            return

        dummy_audio = "dummy_narrator_test_audio.wav"
        if not os.path.exists(dummy_audio):
            try:
                import wave
                import struct
                import math
                sample_rate = 16000.0
                duration = 1
                frequency = 440.0
                num_samples = int(duration * sample_rate)

                with wave.open(dummy_audio, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    for i in range(num_samples):
                        value = int(32767.0 * math.cos(frequency * math.pi * float(i) / float(sample_rate)))
                        data = struct.pack('<h', value)
                        wf.writeframesraw(data)
                logger.info(f"Created dummy audio file: {dummy_audio}")
            except Exception as e:
                logger.error(f"Could not create dummy audio file: {e}. Please create '{dummy_audio}' manually to test.", exc_info=True)
                return

        if os.path.exists(dummy_audio):
            result = await narrator_instance.process_narration(dummy_audio)
            logger.info(f"Narrator Test Result: {result}")
        else:
            logger.warning(f"Dummy audio file '{dummy_audio}' not found. Skipping STT test.")

    asyncio.run(test_narrator())
