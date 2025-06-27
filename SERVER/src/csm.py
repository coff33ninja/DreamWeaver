from .database import Database
from .narrator import Narrator
from .character_server import CharacterServer
from .client_manager import ClientManager
from .hardware import Hardware
from .chaos_engine import ChaosEngine
from .config import DB_PATH

# import os # No longer needed directly here

class CSM:
    def __init__(self):
        self.db = Database(DB_PATH)
        self.narrator = Narrator() # Assuming Narrator handles its own audio saving if needed
        self.character_server = CharacterServer(self.db) # For PC1
        self.client_manager = ClientManager(self.db)
        self.hardware = Hardware() # Assuming it's configured elsewhere or needs no specific path here
        self.chaos_engine = ChaosEngine()
        # self.state = "idle" # This state isn't used much, consider removing or integrating

        # Start periodic health checks when CSM is initialized, as it's a central component.
        # This assumes ClientManager is ready to start its thread.
        self.client_manager.start_periodic_health_checks()
        print("CSM Initialized and started client health checks.")


    def process_story(self, audio_filepath: str, chaos_level: float):
        """
        Processes narration audio, gets responses from server character (PC1)
        and all 'Online_Responsive' clients, applies chaos, and saves the story turn.
        """
        narration_data = self.narrator.process_narration(audio_filepath)
        narration_text = narration_data.get("text", "") if narration_data else ""
        narrator_audio_path_db = narration_data.get("audio_path") # Path to save in DB

        if not narration_text:
            print("CSM: Narration text is empty after STT. Skipping story processing for this turn.")
            return "", {} # Return empty narration and character texts

        character_texts = {} # Stores {character_name: response_text}

        # 1. Server Character (PC1) Response
        server_character_details = self.db.get_character("PC1")
        if server_character_details:
            # Server character doesn't need other_texts for its first response in a turn.
            print(f"CSM: Getting response from server character: {server_character_details.get('name', 'PC1')}")
            server_response_text = self.character_server.generate_response(narration_text, {})
            if server_response_text:
                character_texts[server_character_details.get('name', 'PC1')] = server_response_text
            else:
                print(f"CSM: Server character {server_character_details.get('name', 'PC1')} provided no response.")
        else:
            print("CSM: PC1 (server character) not configured. Skipping its response.")

        # 2. Get responses from 'Online_Responsive' clients
        # get_clients_for_story_progression now returns clients suitable for interaction
        responsive_clients = self.client_manager.get_clients_for_story_progression()
        print(f"CSM: Found {len(responsive_clients)} responsive clients for story progression.")

        for client_info in responsive_clients:
            client_pc_id = client_info.get("pc_id")
            client_ip = client_info.get("ip_address")
            client_port = client_info.get("client_port")

            # Fetch full character details for the client (name is needed for character_texts key)
            client_character_details = self.db.get_character(client_pc_id)
            if not client_character_details:
                print(f"CSM: Warning - Could not get character details for responsive client {client_pc_id}. Skipping.")
                continue

            client_char_name = client_character_details.get("name", client_pc_id)
            print(f"CSM: Getting response from client: {client_char_name} ({client_pc_id}) at {client_ip}:{client_port}")

            # Pass current character_texts (context from previous characters in this turn)
            # Make a copy so each client gets the context up to that point.
            context_for_client = character_texts.copy()

            client_response_text = self.client_manager.send_to_client(
                client_pc_id, client_ip, client_port, narration_text, context_for_client
            )

            if client_response_text:
                character_texts[client_char_name] = client_response_text
            else:
                print(f"CSM: Client {client_char_name} ({client_pc_id}) provided no response or failed.")
                # ClientManager's send_to_client already updates status on failure.

        # 3. Apply Chaos
        if chaos_level > 0 and self.chaos_engine.random_factor() < (chaos_level / 10.0): # Assuming chaos_level 0-10
            print(f"CSM: Applying chaos with level {chaos_level}...")
            narration_text, character_texts = self.chaos_engine.apply_chaos(narration_text, character_texts)

        # 4. Update Hardware (e.g., LEDs)
        self.hardware.update_leds(narration_text) # Or based on combined story elements

        # 5. Save to Database
        if narration_text or character_texts: # Save only if there's something to save
            self.db.save_story(narration_text, character_texts, narrator_audio_path=narrator_audio_path_db)
            print("CSM: Story turn processed and saved.")
        else:
            print("CSM: Nothing to save for this story turn (no narration or character responses).")

        return narration_text, character_texts

    def shutdown(self):
        """Properly shut down CSM resources, like the health check thread."""
        print("CSM: Shutting down...")
        self.client_manager.stop_periodic_health_checks()
        self.db.close() # Ensure database connection is closed.
        print("CSM: Shutdown complete.")

    def __del__(self):
        # Fallback if shutdown isn't explicitly called, though explicit is better.
        self.shutdown()
