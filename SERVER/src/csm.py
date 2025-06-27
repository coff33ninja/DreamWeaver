from .database import Database
from .narrator import Narrator
from .character_server import CharacterServer
from .client_manager import ClientManager
from .hardware import Hardware
from .chaos_engine import ChaosEngine
from .config import DB_PATH # Import from config

import os

class CSM:
    def __init__(self):
        # Use DB_PATH from config
        self.db = Database(DB_PATH)
        self.narrator = Narrator()
        self.character_server = CharacterServer(self.db)
        self.client_manager = ClientManager(self.db)
        self.hardware = Hardware()
        self.chaos_engine = ChaosEngine()
        self.state = "idle"

    def process_story(self, audio, chaos_level):
        narration_data = self.narrator.process_narration(audio) # Returns dict with 'text' and 'audio_path'
        narration_text = narration_data.get("text", "") if narration_data else ""

        if not narration_text:
            # If narration is empty, perhaps return early or handle as appropriate
            print("CSM: Narration text is empty.")
            return "", {}

        character_texts = {}
        # Generate server character response
        server_character = self.db.get_character("PC1")
        if server_character:
            # Pass only narration_text to character_server, it doesn't need other_texts for the first character
            server_response_text = self.character_server.generate_response(narration_text, {})
            if server_response_text: # Ensure there's a response
                 character_texts[server_character["name"]] = server_response_text

        # Get responses from active clients
        active_clients = self.client_manager.get_active_clients()
        for client_info in active_clients:
            character = self.db.get_character(client_info["pc"])
            if character:
                # Pass current character_texts for context
                client_response_text = self.client_manager.send_to_client(client_info["pc"], client_info["ip_address"], narration_text, character_texts.copy())
                if client_response_text: # Ensure there's a response
                    character_texts[character["name"]] = client_response_text

        # Apply chaos
        if chaos_level > self.chaos_engine.random_factor():
            narration_text, character_texts = self.chaos_engine.apply_chaos(narration_text, character_texts)

        # Update LEDs based on narration text
        self.hardware.update_leds(narration_text)

        # Save to DB - ensure narration_text and character_texts are what you intend to save
        if narration_text or character_texts: # Save only if there's something to save
            self.db.save_story(narration_text, character_texts, narrator_audio_path=narration_data.get("audio_path"))

        return narration_text, character_texts
