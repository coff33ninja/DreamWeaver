from .llm_engine import LLMEngine
from .tts_manager import TTSManager
from .config import REFERENCE_VOICES_AUDIO_PATH, CHARACTERS_AUDIO_PATH
import pygame
import uuid
import os
import asyncio # Added asyncio

class CharacterServer: # This is for Actor1, the server's own character
    def __init__(self, db):
        """
        Initialize the CharacterServer with a database interface and load character data for "Actor1".
        
        If character data for "Actor1" is missing in the database, a default character profile is created and saved. LLM and TTS components are set to None for asynchronous initialization.
        """
        self.db = db
        self.character_Actor_id = "Actor1" # Explicitly for Actor1
        self.character = self.db.get_character(self.character_Actor_id)
        if not self.character:
            print("CharacterServer: WARNING - Actor1 character data not found in DB. Using defaults.")
            # Define a more complete default, including llm_model
            self.character = {
                "name": "Actor1_Default", "personality": "server_default",
                "goals": "assist", "backstory": "embedded",
                "tts": "piper", "tts_model": "en_US-ryan-high",
                "reference_audio_filename": None, "Actor_id": self.character_Actor_id, # Actor_id was missing in default
                "llm_model": None # LLMEngine will use its default
            }
            # Optionally save this default to DB if it's truly missing
            self.db.save_character(**self.character) # Careful with Actor_id vs Actor key

        # LLM and TTS are initialized asynchronously for non-blocking startup
        self.llm = None
        self.tts = None

    async def async_init(self):
        """
        Asynchronously initializes the LLM and TTS components for the character, and ensures the audio mixer is ready for playback.
        
        This method offloads blocking initializations of the language model and text-to-speech manager to executor threads, and initializes the pygame mixer if not already set up.
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
                print("CharacterServer (Actor1): Pygame mixer initialized.")
            except pygame.error as e:
                print(f"CharacterServer (Actor1): Pygame mixer could not be initialized: {e}. Audio playback will fail.")

    async def generate_response(self, narration: str, other_texts: dict) -> str:
        """
        Generate a character response using the LLM based on provided narration and other character texts, then synthesize and play the response as audio.
        
        Parameters:
            narration (str): The narration text to include in the prompt.
            other_texts (dict): A mapping of other character names to their respective dialogue or context.
        
        Returns:
            str: The generated response text, or an error string if generation fails.
        """
        if not self.character:
            print("CharacterServer (Actor1): Character not loaded. Cannot generate response.")
            return ""
        if not self.llm or not self.llm.is_initialized:
            print("CharacterServer (Actor1): LLM not initialized. Cannot generate response.")
            return "[Actor1_LLM_ERROR]"

        # Construct prompt
        prompt_parts = [f"Narrator: {narration}"]
        for name, text_val in other_texts.items():
            prompt_parts.append(f"{name}: {text_val}")
        prompt_parts.append(f"Character: {self.character['name']} responds as {self.character['personality']}:")
        prompt = "\n".join(prompt_parts)

        # LLM generate is now async
        # print(f"CharacterServer (Actor1): Generating LLM response...")
        text = await self.llm.generate(prompt, max_new_tokens=120) if self.llm else "[Actor1_LLM_ERROR]"

        if text and text != "[LLM_ERROR: NOT_INITIALIZED]" and text != "[LLM_ERROR: GENERATION_FAILED]":
            # Save training data (DB op, could be threaded but usually fast)
            # For now, keep it blocking as it's quick.
            self.db.save_training_data({"input": prompt, "output": text}, self.character_Actor_id)

            if self.llm:
                # LLM fine_tune is now async
                # print(f"CharacterServer (Actor1): Initiating fine-tuning...")
                await self.llm.fine_tune({"input": prompt, "output": text}, self.character_Actor_id)

            # TTS output is now async
            # print(f"CharacterServer (Actor1): Synthesizing audio output...")
            await self.output_audio(text) # Pass only text, speaker_wav is handled by TTSManager instance or this method

        return text

    async def output_audio(self, text: str): # speaker_wav removed, TTSManager instance holds it for XTTS
        """
        Asynchronously synthesizes speech from text using the character's TTS engine and plays the resulting audio.
        
        If the character uses XTTSv2 and a reference audio file is available, it is used for voice cloning. The synthesized audio is saved to the character's audio directory and played using pygame if the mixer is initialized. Logs warnings if TTS is uninitialized, synthesis fails, or playback encounters errors.
        """
        if not self.tts or not self.tts.is_initialized or not text or not self.character:
            if not self.tts or not self.tts.is_initialized:
                print("CharacterServer (Actor1): TTS not initialized or text empty. No audio.")
            return

        # TTSManager's synthesize is now async and handles its own speaker_wav logic
        # It saves to a temporary file and returns the path.
        # For Actor1, we want to save it to its character directory and then play.

        sane_char_name = "".join(c if c.isalnum() else "_" for c in self.character.get('name', self.character_Actor_id))
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
                print(f"CharacterServer (Actor1): Warning - Reference audio {ref_path} not found for XTTSv2.")

        # Synthesize audio (this will save to a temp path managed by TTSManager)
        # The `synthesize` method in TTSManager has been updated to be async.
        # It now returns a boolean success status. The actual file is saved to `output_path` passed to it.
        # We need to pass the final_audio_path here.
        success = await self.tts.synthesize(text, final_audio_path, speaker_wav_for_synthesis=speaker_wav_to_use)

        if success and os.path.exists(final_audio_path):
            # print(f"CharacterServer (Actor1): Audio synthesized to {final_audio_path}")
            if pygame.mixer.get_init(): # Check if mixer is still initialized
                try:
                    # Pygame sound ops are blocking, run in thread
                    def play_sound():
                        """
                        Play an audio file located at `final_audio_path` using pygame's mixer.
                        
                        This function initiates playback of the specified audio file asynchronously. It does not block execution while the sound is playing.
                        """
                        sound = pygame.mixer.Sound(final_audio_path)
                        sound.play()
                        # Need to ensure the sound has time to play if script exits or mixer quits.
                        # For a server, this is less of an issue.
                        # while pygame.mixer.get_busy():
                        #     pygame.time.Clock().tick(10)

                    await asyncio.to_thread(play_sound)
                    # print(f"CharacterServer (Actor1): Playing audio {final_audio_path}")
                except Exception as e:
                    print(f"CharacterServer (Actor1): Error playing audio {final_audio_path}: {e}")
            else:
                print(f"CharacterServer (Actor1): Pygame mixer not initialized. Cannot play audio {final_audio_path}.")
        else:
            print(f"CharacterServer (Actor1): Audio synthesis failed or file not found at {final_audio_path}.")

if __name__ == '__main__':
    # This test requires a DB with Actor1 configured, and models.
    # It's more of an integration test component.
    async def test_character_server():
        """
        Asynchronously tests the CharacterServer class by initializing it with a dummy database and generating a sample response.
        
        This function sets up a minimal environment with mock database methods, ensures required audio directories exist, initializes the CharacterServer asynchronously, and verifies LLM and TTS readiness. If initialization succeeds, it generates and prints a response to a test narration.
        """
        print("Testing CharacterServer (Actor1)...")
        # Mock DB or ensure DB_PATH points to a test DB with Actor1
        class DummyDB:
            def get_character(self, Actor_id):
                """
                Retrieve character data for the specified Actor ID.
                
                Returns a dictionary containing character attributes if the Actor ID is "Actor1"; otherwise, returns None.
                """
                if Actor_id == "Actor1":
                    return {"name": "TestActor1", "personality": "tester", "tts": "gtts", "language":"en",
                            "reference_audio_filename": None, "Actor_id": "Actor1", "llm_model": None} # Use gTTS for no model download
                return None
            def save_training_data(self, data, Actor_id): """
Simulate saving training data for a given actor by printing the data and actor ID.
"""
print(f"DummyDB: Save training data for {Actor_id}: {data}")

        # Ensure server config paths are valid for this test run
        # For example, CHARACTERS_AUDIO_PATH needs to be writable.
        os.makedirs(CHARACTERS_AUDIO_PATH, exist_ok=True)
        os.makedirs(REFERENCE_VOICES_AUDIO_PATH, exist_ok=True)


        cs = CharacterServer(db=DummyDB())
        await cs.async_init()
        if cs.llm and cs.llm.is_initialized and cs.tts and cs.tts.is_initialized:
            print("CharacterServer initialized with LLM and TTS.")
            narration = "A test narration for Actor1."
            response = await cs.generate_response(narration, {})
            print(f"Actor1 Response to '{narration}': '{response}'")
        else:
            print("CharacterServer LLM or TTS failed to initialize.")
            if cs.llm:
                print(f"LLM initialized: {cs.llm.is_initialized}")
            if cs.tts:
                print(f"TTS initialized: {cs.tts.is_initialized}")


    asyncio.run(test_character_server())
