from .llm_engine import LLMEngine
from .tts_manager import TTSManager
from .config import REFERENCE_VOICES_AUDIO_PATH, CHARACTERS_AUDIO_PATH # Import necessary paths
import pygame
import uuid
import os

class CharacterServer:
    def __init__(self, db):
        self.db = db
        self.character = db.get_character("PC1") or {"name": "PC1_Default", "personality": "neutral", "goals": "survive", "backstory": "none", "tts": "piper", "tts_model": "en_US-ryan-high", "reference_audio_filename": None, "pc": "PC1"} # Ensure character is not None
        self.llm = LLMEngine(db=self.db) # Pass the database instance to LLMEngine
        self.tts = TTSManager(self.character.get("tts"), self.character.get("tts_model"))
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    def generate_response(self, narration, other_texts):
        if not self.character:
            return ""
        prompt = f"Narrator: {narration}\n" + "\n".join([f"{k}: {v}" for k, v in other_texts.items()]) + f"\nCharacter: {self.character['name']} responds as {self.character['personality']}:"
        text = self.llm.generate(prompt, max_length=100) # Use a more reasonable max_length for generation
        # Placeholder for actual fine-tuning logic
        self.db.save_training_data({"input": prompt, "output": text}, "PC1")
        self.llm.fine_tune({"input": prompt, "output": text}, "PC1") # Trigger fine-tuning after saving data

        speaker_wav = None
        if self.character.get("tts") == "xttsv2" and self.character.get("reference_audio_filename"):
            # Use path from config
            speaker_wav = os.path.join(REFERENCE_VOICES_AUDIO_PATH, self.character["reference_audio_filename"])

        if text: # Only output audio if there's text
            self.output_audio(text, speaker_wav=speaker_wav)
        return text

    def output_audio(self, text, speaker_wav=None):
        if self.tts and text and self.character:
            # Use path from config
            audio_dir = os.path.join(CHARACTERS_AUDIO_PATH, self.character['name'])
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"{uuid.uuid4()}.wav")
            try:
                self.tts.synthesize(text, audio_path, speaker_wav=speaker_wav)
                sound = pygame.mixer.Sound(audio_path)
                sound.play()
            except Exception as e:
                print(f"Error playing audio for {self.character['name']}: {e}")
