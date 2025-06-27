from .database import Database
from .narrator import Narrator
from .character_server import CharacterServer
from .client_manager import ClientManager
from .hardware import Hardware
from .chaos_engine import ChaosEngine

import os

class CSM:
    def __init__(self):
        self.db = Database(os.getenv("DB_PATH", "E:/DreamWeaver/data/dream_weaver.db"))
        self.narrator = Narrator()
        self.character_server = CharacterServer(self.db)
        self.client_manager = ClientManager(self.db)
        self.hardware = Hardware()
        self.chaos_engine = ChaosEngine()
        self.state = "idle"

    def process_story(self, audio, chaos_level):
        narration = self.narrator.process_narration(audio)
        character_texts = {}
        # Generate server character response
        server_character = self.db.get_character("PC1")
        if server_character:
            character_texts[server_character["name"]] = self.character_server.generate_response(narration, "")
        # Get responses from active clients
        for client_info in self.client_manager.get_active_clients():
            character = self.db.get_character(client_info["pc"])
            if character:
                text = self.client_manager.send_to_client(client_info["pc"], client_info["ip_address"], narration, character_texts)
                character_texts[character["name"]] = text
        # Apply chaos
        if chaos_level > self.chaos_engine.random_factor():
            narration, character_texts = self.chaos_engine.apply_chaos(narration, character_texts)
        # Update LEDs
        self.hardware.update_leds(narration)
        # Save to DB
        self.db.save_story(narration, character_texts)
        return narration, character_texts
