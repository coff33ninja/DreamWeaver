from .llm_engine import LLMEngine
from .tts_manager import TTSManager
from .config import REFERENCE_VOICES_AUDIO_PATH, CHARACTERS_AUDIO_PATH
import pygame
import uuid
import os
import asyncio
import logging

logger = logging.getLogger("dreamweaver_server")

class CharacterServer: # This is for Actor1, the server's own character
    def __init__(self, db):
        """
        Initialize the CharacterServer for "Actor1", loading character data from the database or creating a default if missing.

        If the character data for "Actor1" is not found in the database, a default character profile is created and saved. LLM and TTS engines are not initialized here and must be set up asynchronously.
        """
        self.db = db
        self.character_Actor_id = "Actor1" # Explicitly for Actor1
        logger.info(f"Initializing CharacterServer for {self.character_Actor_id}...")
        self.character = self.db.get_character(self.character_Actor_id)
        if not self.character:
            logger.warning(f"CharacterServer: {self.character_Actor_id} character data not found in DB. Creating and using defaults.")
            self.character = {
                "name": "Actor1_Default", "personality": "server_default",
                "goals": "assist", "backstory": "embedded",
                "tts": "piper", "tts_model": "en_US-ryan-high", # Default TTS
                "reference_audio_filename": None, "Actor_id": self.character_Actor_id,
                "llm_model": None # LLMEngine will use its default
            }
            try:
                self.db.save_character(**self.character)
                logger.info(f"CharacterServer: Saved default character profile for {self.character_Actor_id} to DB.")
            except Exception as e:
                logger.error(f"CharacterServer: Failed to save default character profile for {self.character_Actor_id} to DB: {e}", exc_info=True)
        else:
            logger.info(f"CharacterServer: Loaded character data for {self.character_Actor_id} from DB.")

        self.llm = None
        self.tts = None
        logger.info(f"CharacterServer for {self.character_Actor_id} initialized. LLM and TTS to be initialized asynchronously.")


    async def async_init(self):
        """
        Asynchronously initializes the LLM and TTS engines for Actor1, preparing the character for text generation and audio synthesis.

        This method offloads blocking initialization tasks for the LLM and TTS engines to background threads to avoid blocking the event loop. It also ensures the pygame mixer is initialized for audio playback, handling errors if initialization fails.
        """
        logger.info(f"CharacterServer ({self.character_Actor_id}): Starting asynchronous initialization of LLM and TTS...")
        loop = asyncio.get_event_loop()

        try:
            llm_model_name = self.character.get("llm_model") or "" # Ensure string
            logger.info(f"CharacterServer ({self.character_Actor_id}): Initializing LLMEngine with model '{llm_model_name}'...")
            self.llm = await loop.run_in_executor(None, lambda: LLMEngine(model_name=llm_model_name, db=self.db))
            if self.llm and self.llm.is_initialized:
                logger.info(f"CharacterServer ({self.character_Actor_id}): LLMEngine initialized successfully.")
            else:
                logger.warning(f"CharacterServer ({self.character_Actor_id}): LLMEngine failed to initialize or is_initialized is false.")
        except Exception as e_llm:
            logger.error(f"CharacterServer ({self.character_Actor_id}): Error initializing LLMEngine: {e_llm}", exc_info=True)
            self.llm = None

        try:
            tts_service = self.character.get("tts") or ""
            tts_model = self.character.get("tts_model") or ""
            ref_audio_filename = self.character.get("reference_audio_filename")
            speaker_wav = None
            if tts_service == "xttsv2" and ref_audio_filename:
                speaker_wav = os.path.join(REFERENCE_VOICES_AUDIO_PATH, ref_audio_filename)
                if not os.path.exists(speaker_wav):
                    logger.warning(f"CharacterServer ({self.character_Actor_id}): Reference audio {speaker_wav} not found for XTTSv2. Will use default voice.")
                    speaker_wav = None # Fallback to default voice

            logger.info(f"CharacterServer ({self.character_Actor_id}): Initializing TTSManager with service '{tts_service}', model '{tts_model}'...")
            self.tts = await loop.run_in_executor(None, lambda: TTSManager(
                tts_service_name=tts_service,
                model_name=tts_model,
                speaker_wav_path=speaker_wav, # Already validated path or None
                language=self.character.get("language", "en")
            ))
            if self.tts and self.tts.is_initialized:
                logger.info(f"CharacterServer ({self.character_Actor_id}): TTSManager initialized successfully.")
            else:
                logger.warning(f"CharacterServer ({self.character_Actor_id}): TTSManager failed to initialize or is_initialized is false.")
        except Exception as e_tts:
            logger.error(f"CharacterServer ({self.character_Actor_id}): Error initializing TTSManager: {e_tts}", exc_info=True)
            self.tts = None

        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                logger.info(f"CharacterServer ({self.character_Actor_id}): Pygame mixer initialized for audio playback.")
            except pygame.error as e_pygame:
                logger.warning(f"CharacterServer ({self.character_Actor_id}): Pygame mixer could not be initialized: {e_pygame}. Audio playback will fail.", exc_info=True)
        logger.info(f"CharacterServer ({self.character_Actor_id}): Asynchronous initialization complete.")


    async def generate_response(self, narration: str, other_texts: dict) -> str:
        """
        Asynchronously generates a text response for the character based on narration and other character inputs, then synthesizes and plays the response as audio.

        Parameters:
            narration (str): The narration or prompt to which the character should respond.
            other_texts (dict): A mapping of other character names to their spoken text for context.

        Returns:
            str: The generated text response from the character, or an error string if generation fails.
        """
        char_name = self.character.get('name', self.character_Actor_id)
        if not self.character: # Should have been caught by __init__
            logger.error(f"CharacterServer ({char_name}): Character not loaded. Cannot generate response.")
            return ""
        if not self.llm or not self.llm.is_initialized:
            logger.error(f"CharacterServer ({char_name}): LLM not initialized. Cannot generate response for narration: '{narration[:50]}...'.")
            return f"[{char_name}_LLM_ERROR:NOT_INITIALIZED]"

        prompt_parts = [f"Narrator: {narration}"]
        for name, text_val in other_texts.items():
            prompt_parts.append(f"{name}: {text_val}")
        prompt_parts.append(f"Character: {char_name} responds as {self.character.get('personality', 'default')}:")
        prompt = "\n".join(prompt_parts)

        logger.info(f"CharacterServer ({char_name}): Generating LLM response for prompt: '{prompt[:100]}...'")
        text = await self.llm.generate(prompt, max_new_tokens=120)

        if text and not text.startswith(f"[{char_name}_LLM_ERROR") and not text.startswith("[LLM_ERROR"): # Check specific and generic LLM errors
            logger.info(f"CharacterServer ({char_name}): LLM generated response: '{text[:100]}...'")
            try:
                self.db.save_training_data({"input": prompt, "output": text}, self.character_Actor_id)
                logger.debug(f"CharacterServer ({char_name}): Saved training data.")
            except Exception as e_db_train:
                logger.error(f"CharacterServer ({char_name}): Error saving training data: {e_db_train}", exc_info=True)

            if self.llm: # LLM is confirmed initialized above
                logger.debug(f"CharacterServer ({char_name}): Initiating fine-tuning (placeholder in LLMEngine).")
                await self.llm.fine_tune({"input": prompt, "output": text}, self.character_Actor_id)

            logger.debug(f"CharacterServer ({char_name}): Synthesizing audio output for response.")
            await self.output_audio(text)
        else:
            logger.error(f"CharacterServer ({char_name}): LLM generation failed or returned error. Response: '{text}'")
            # text might already be an error string from LLMEngine like "[LLM_ERROR: GENERATION_FAILED]"
            if not text: text = f"[{char_name}_LLM_ERROR:EMPTY_RESPONSE]"


        return text

    async def output_audio(self, text: str):
        """
        Asynchronously synthesizes speech audio from the given text using the character's TTS engine and plays it back.

        If the character uses XTTSv2 and a reference audio file is available, it is used for voice cloning. The synthesized audio is saved to the character's audio directory and played using pygame's mixer if initialized. Logs warnings if TTS is not initialized, reference audio is missing, synthesis fails, or audio playback is unavailable.
        """
        char_name = self.character.get('name', self.character_Actor_id)
        if not self.tts or not self.tts.is_initialized:
            logger.warning(f"CharacterServer ({char_name}): TTS not initialized. Cannot output audio for text: '{text[:50]}...'.")
            return
        if not text:
            logger.warning(f"CharacterServer ({char_name}): Text is empty. No audio to output.")
            return


        sane_char_name = "".join(c if c.isalnum() else "_" for c in char_name)
        character_audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, sane_char_name)
        try:
            os.makedirs(character_audio_dir, exist_ok=True)
        except OSError as e_mkdir:
            logger.error(f"CharacterServer ({char_name}): Could not create audio directory {character_audio_dir}: {e_mkdir}", exc_info=True)
            return

        final_audio_filename = f"{uuid.uuid4()}.wav"
        final_audio_path = os.path.join(character_audio_dir, final_audio_filename)

        speaker_wav_to_use = None
        if self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            ref_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, self.character["reference_audio_filename"])
            if os.path.exists(ref_path):
                speaker_wav_to_use = ref_path
                logger.debug(f"CharacterServer ({char_name}): Using reference audio {ref_path} for XTTSv2.")
            else:
                logger.warning(f"CharacterServer ({char_name}): Reference audio {ref_path} not found for XTTSv2. TTS will use default voice.")

        logger.info(f"CharacterServer ({char_name}): Attempting to synthesize audio to {final_audio_path} for text: '{text[:50]}...'.")
        success = await self.tts.synthesize(text, final_audio_path, speaker_wav_for_synthesis=speaker_wav_to_use)

        if success and os.path.exists(final_audio_path):
            logger.info(f"CharacterServer ({char_name}): Audio successfully synthesized to {final_audio_path}")
            if pygame.mixer.get_init():
                try:
                    def play_sound_blocking():
                        sound = pygame.mixer.Sound(final_audio_path)
                        sound.play()
                        while pygame.mixer.get_busy(): # Ensure sound plays out in the thread
                            pygame.time.Clock().tick(10)

                    logger.debug(f"CharacterServer ({char_name}): Playing audio {final_audio_path}")
                    await asyncio.to_thread(play_sound_blocking)
                    logger.info(f"CharacterServer ({char_name}): Finished playing audio {final_audio_path}")
                except Exception as e_play:
                    logger.error(f"CharacterServer ({char_name}): Error playing audio {final_audio_path}: {e_play}", exc_info=True)
            else:
                logger.warning(f"CharacterServer ({char_name}): Pygame mixer not initialized. Cannot play audio {final_audio_path}.")
        else:
            logger.error(f"CharacterServer ({char_name}): Audio synthesis failed or output file not found/empty at {final_audio_path}.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')

    async def test_character_server():
        logger.info("Testing CharacterServer (Actor1)...")

        class DummyDB:
            _data = {}
            def get_character(self, Actor_id):
                logger.debug(f"DummyDB: get_character called for {Actor_id}")
                if Actor_id == "Actor1" and Actor_id in self._data:
                    return self._data[Actor_id]
                elif Actor_id == "Actor1": # Default if not saved by save_character
                    return {"name": "TestActor1", "personality": "tester", "tts": "gtts", "language":"en",
                            "reference_audio_filename": None, "Actor_id": "Actor1", "llm_model": None}
                return None
            def save_character(self, **kwargs):
                actor_id = kwargs.get("Actor_id")
                logger.debug(f"DummyDB: save_character called for {actor_id} with data: {kwargs}")
                if actor_id:
                    self._data[actor_id] = kwargs
            def save_training_data(self, data, Actor_id):
                logger.debug(f"DummyDB: Save training data for {Actor_id}: {data}")

        os.makedirs(CHARACTERS_AUDIO_PATH, exist_ok=True)
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)

        # Test with a DB that initially has no Actor1
        dummy_db_instance_empty = DummyDB()
        logger.info("--- Test Case 1: Initializing CharacterServer with empty DB (should create default Actor1) ---")
        cs_empty_db = CharacterServer(db=dummy_db_instance_empty)
        await cs_empty_db.async_init() # Let it try to init LLM/TTS

        # Test with a DB that has Actor1 pre-configured (mimicking a real scenario)
        dummy_db_instance_filled = DummyDB()
        actor1_config = {"name": "ConfiguredActor1", "personality": "configured_tester",
                         "tts": "gtts", "language":"en", "tts_model": "en", # gTTS uses lang code as model effectively
                         "reference_audio_filename": None, "Actor_id": "Actor1", "llm_model": "some_test_llm"}
        dummy_db_instance_filled.save_character(**actor1_config)

        logger.info("--- Test Case 2: Initializing CharacterServer with pre-configured Actor1 ---")
        cs = CharacterServer(db=dummy_db_instance_filled)
        await cs.async_init()

        if cs.llm and cs.llm.is_initialized and cs.tts and cs.tts.is_initialized:
            logger.info("CharacterServer (Actor1) initialized with LLM and TTS for Test Case 2.")
            narration = "A test narration for the configured Actor1."
            response = await cs.generate_response(narration, {})
            logger.info(f"Actor1 Response to '{narration}': '{response}'")
        else:
            logger.error("CharacterServer (Actor1) LLM or TTS failed to initialize for Test Case 2.")
            if cs.llm:
                logger.info(f"LLM initialized: {cs.llm.is_initialized}")
            else:
                logger.error("LLM object is None.")
            if cs.tts:
                logger.info(f"TTS initialized: {cs.tts.is_initialized}")
            else:
                logger.error("TTS object is None.")
        logger.info("CharacterServer test finished.")

    asyncio.run(test_character_server())
