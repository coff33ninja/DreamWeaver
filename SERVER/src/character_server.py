from .llm_engine import LLMEngine
from .tts_manager import TTSManager
from .config import REFERENCE_VOICES_AUDIO_PATH, CHARACTERS_AUDIO_PATH
import pygame
import uuid
import os
import asyncio # Added asyncio

class CharacterServer: # This is for PC1, the server's own character
    def __init__(self, db):
        self.db = db
        self.character_pc_id = "PC1" # Explicitly for PC1
        self.character = self.db.get_character(self.character_pc_id)
        if not self.character:
            print("CharacterServer: WARNING - PC1 character data not found in DB. Using defaults.")
            # Define a more complete default, including llm_model
            self.character = {
                "name": "PC1_Default", "personality": "server_default",
                "goals": "assist", "backstory": "embedded",
                "tts": "piper", "tts_model": "en_US-ryan-high",
                "reference_audio_filename": None, "pc_id": self.character_pc_id, # pc_id was missing in default
                "llm_model": None # LLMEngine will use its default
            }
            # Optionally save this default to DB if it's truly missing
            self.db.save_character(**self.character) # Careful with pc_id vs pc key

        # LLM and TTS are initialized asynchronously for non-blocking startup
        self.llm = None
        self.tts = None

    async def async_init(self):
        """
        Initialize LLM and TTS for PC1.
        LLMEngine init is blocking, TTSManager init is also blocking (model downloads).
        These should ideally be loaded asynchronously or in a background thread at server startup,
        not during CharacterServer init if it's on a critical path.
        For now, keeping it here as an explicit async method as per best practice.
        """
        loop = asyncio.get_event_loop()
        self.llm = await loop.run_in_executor(None, lambda: LLMEngine(model_name=self.character.get("llm_model") or "", db=self.db))
        self.tts = await loop.run_in_executor(None, lambda: TTSManager(
            tts_service_name=self.character.get("tts") or "",
            model_name=self.character.get("tts_model") or "",
            speaker_wav_path=os.path.join(REFERENCE_VOICES_AUDIO_PATH, self.character["reference_audio_filename"]) if self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename") else None,
            language=self.character.get("language", "en")
        ))

        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                print("CharacterServer (PC1): Pygame mixer initialized.")
            except pygame.error as e:
                print(f"CharacterServer (PC1): Pygame mixer could not be initialized: {e}. Audio playback will fail.")

    async def generate_response(self, narration: str, other_texts: dict) -> str:
        if not self.character:
            print("CharacterServer (PC1): Character not loaded. Cannot generate response.")
            return ""
        if not self.llm or not self.llm.is_initialized:
            print("CharacterServer (PC1): LLM not initialized. Cannot generate response.")
            return "[PC1_LLM_ERROR]"

        # Construct prompt
        prompt_parts = [f"Narrator: {narration}"]
        for name, text_val in other_texts.items():
            prompt_parts.append(f"{name}: {text_val}")
        prompt_parts.append(f"Character: {self.character['name']} responds as {self.character['personality']}:")
        prompt = "\n".join(prompt_parts)

        # LLM generate is now async
        # print(f"CharacterServer (PC1): Generating LLM response...")
        text = await self.llm.generate(prompt, max_new_tokens=120) if self.llm else "[PC1_LLM_ERROR]"

        if text and text != "[LLM_ERROR: NOT_INITIALIZED]" and text != "[LLM_ERROR: GENERATION_FAILED]":
            # Save training data (DB op, could be threaded but usually fast)
            # For now, keep it blocking as it's quick.
            self.db.save_training_data({"input": prompt, "output": text}, self.character_pc_id)

            if self.llm:
                # LLM fine_tune is now async
                # print(f"CharacterServer (PC1): Initiating fine-tuning...")
                await self.llm.fine_tune({"input": prompt, "output": text}, self.character_pc_id)

            # TTS output is now async
            # print(f"CharacterServer (PC1): Synthesizing audio output...")
            await self.output_audio(text) # Pass only text, speaker_wav is handled by TTSManager instance or this method

        return text

    async def output_audio(self, text: str): # speaker_wav removed, TTSManager instance holds it for XTTS
        if not self.tts or not self.tts.is_initialized or not text or not self.character:
            if not self.tts or not self.tts.is_initialized:
                print("CharacterServer (PC1): TTS not initialized or text empty. No audio.")
            return

        # TTSManager's synthesize is now async and handles its own speaker_wav logic
        # It saves to a temporary file and returns the path.
        # For PC1, we want to save it to its character directory and then play.

        sane_char_name = "".join(c if c.isalnum() else "_" for c in self.character.get('name', self.character_pc_id))
        character_audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, sane_char_name)
        os.makedirs(character_audio_dir, exist_ok=True)

        # Filename for the final audio in the character's directory
        final_audio_filename = f"{uuid.uuid4()}.wav"
        final_audio_path = os.path.join(character_audio_dir, final_audio_filename)

        # Use speaker_wav_for_synthesis if XTTS and ref audio exists
        speaker_wav_to_use = None
        if self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            # Construct full path to reference audio for this character
            ref_path = os.path.join(REFERENCE_VOICES_AUDIO_PATH, self.character["reference_audio_filename"])
            if os.path.exists(ref_path):
                speaker_wav_to_use = ref_path
            else:
                print(f"CharacterServer (PC1): Warning - Reference audio {ref_path} not found for XTTSv2.")

        # Synthesize audio (this will save to a temp path managed by TTSManager)
        # The `synthesize` method in TTSManager has been updated to be async.
        # It now returns a boolean success status. The actual file is saved to `output_path` passed to it.
        # We need to pass the final_audio_path here.
        success = await self.tts.synthesize(text, final_audio_path, speaker_wav_for_synthesis=speaker_wav_to_use)

        if success and os.path.exists(final_audio_path):
            # print(f"CharacterServer (PC1): Audio synthesized to {final_audio_path}")
            if pygame.mixer.get_init(): # Check if mixer is still initialized
                try:
                    # Pygame sound ops are blocking, run in thread
                    def play_sound():
                        sound = pygame.mixer.Sound(final_audio_path)
                        sound.play()
                        # Need to ensure the sound has time to play if script exits or mixer quits.
                        # For a server, this is less of an issue.
                        # while pygame.mixer.get_busy():
                        #     pygame.time.Clock().tick(10)

                    await asyncio.to_thread(play_sound)
                    # print(f"CharacterServer (PC1): Playing audio {final_audio_path}")
                except Exception as e:
                    print(f"CharacterServer (PC1): Error playing audio {final_audio_path}: {e}")
            else:
                print(f"CharacterServer (PC1): Pygame mixer not initialized. Cannot play audio {final_audio_path}.")
        else:
            print(f"CharacterServer (PC1): Audio synthesis failed or file not found at {final_audio_path}.")

if __name__ == '__main__':
    # This test requires a DB with PC1 configured, and models.
    # It's more of an integration test component.
    async def test_character_server():
        print("Testing CharacterServer (PC1)...")
        # Mock DB or ensure DB_PATH points to a test DB with PC1
        class DummyDB:
            def get_character(self, pc_id):
                if pc_id == "PC1":
                    return {"name": "TestPC1", "personality": "tester", "tts": "gtts", "language":"en",
                            "reference_audio_filename": None, "pc_id": "PC1", "llm_model": None} # Use gTTS for no model download
                return None
            def save_training_data(self, data, pc_id): print(f"DummyDB: Save training data for {pc_id}: {data}")

        # Ensure server config paths are valid for this test run
        # For example, CHARACTERS_AUDIO_PATH needs to be writable.
        os.makedirs(CHARACTERS_AUDIO_PATH, exist_ok=True)
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)


        cs = CharacterServer(db=DummyDB())
        await cs.async_init()
        if cs.llm and cs.llm.is_initialized and cs.tts and cs.tts.is_initialized:
            print("CharacterServer initialized with LLM and TTS.")
            narration = "A test narration for PC1."
            response = await cs.generate_response(narration, {})
            print(f"PC1 Response to '{narration}': '{response}'")
        else:
            print("CharacterServer LLM or TTS failed to initialize.")
            if cs.llm:
                print(f"LLM initialized: {cs.llm.is_initialized}")
            if cs.tts:
                print(f"TTS initialized: {cs.tts.is_initialized}")


    asyncio.run(test_character_server())
