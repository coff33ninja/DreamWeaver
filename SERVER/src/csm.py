from .database import Database
from .narrator import Narrator
from .character_server import CharacterServer # Should be async
from .client_manager import ClientManager   # Should be async
from .hardware import Hardware
from .chaos_engine import ChaosEngine
from .config import DB_PATH
import asyncio # Added asyncio
import os

class CSM:
    def __init__(self):
        """
        Initialize the CSM instance and its core components, including database, narrator, character server, client manager, hardware interface, and chaos engine. Starts periodic health checks for connected clients.
        """
        self.db = Database(DB_PATH)
        self.narrator = Narrator()
        self.character_server = CharacterServer(self.db)
        self.client_manager = ClientManager(self.db)
        self.hardware = Hardware()
        self.chaos_engine = ChaosEngine()

        self.client_manager.start_periodic_health_checks()
        print("CSM Initialized, client health checks started.")

    async def process_story(self, audio_filepath: str, chaos_level: float):
        """
        Asynchronously processes a story turn by generating narration, collecting character and client responses, applying chaos effects, updating hardware, and saving results to the database.
        
        Parameters:
            audio_filepath (str): Path to the audio file containing the narration input.
            chaos_level (float): The level of chaos to potentially apply to the narration and responses.
        
        Returns:
            narration_text (str): The processed narration text.
            character_texts (dict): A mapping of character names to their generated responses.
        """
        # Narrator process_narration is now async
        # print("CSM: Processing narration...")
        narration_data = await self.narrator.process_narration(audio_filepath)
        narration_text = narration_data.get("text", "") if narration_data else ""
        narrator_audio_path_db = narration_data.get("audio_path")

        if not narration_text:
            print("CSM: Narration text empty. Skipping story processing.")
            return "", {}

        character_texts = {}

        # 1. Server Character (Actor1) Response
        server_character_details = await asyncio.to_thread(self.db.get_character, "Actor1") # DB call in thread
        if server_character_details:
            # print(f"CSM: Getting response from server character: {server_character_details.get('name', 'Actor1')}")
            # CharacterServer.generate_response is now async
            server_response_text = await self.character_server.generate_response(narration_text, {})
            if server_response_text:
                character_texts[server_character_details.get('name', 'Actor1')] = server_response_text
            # else:
                # print(f"CSM: Server character {server_character_details.get('name', 'Actor1')} provided no response.")
        # else:
            # print("CSM: Actor1 (server character) not configured.")

        # 2. Get responses from 'Online_Responsive' clients
        # ClientManager.get_clients_for_story_progression uses DB, run in thread
        responsive_clients = await asyncio.to_thread(self.client_manager.get_clients_for_story_progression)
        # print(f"CSM: Found {len(responsive_clients)} responsive clients.")

        client_response_tasks = []
        for client_info in responsive_clients:
            client_actor_id = client_info.get("Actor_id")
            client_ip = client_info.get("ip_address")
            client_port = client_info.get("client_port")

            # Fetch character details (DB call) - can also be done concurrently if many clients
            # For now, keeping it sequential before dispatching the send_to_client task
            client_character_details = await asyncio.to_thread(self.db.get_character, client_actor_id)
            if not client_character_details:
                print(f"CSM: Warning - No character details for responsive client {client_actor_id}. Skipping.")
                continue

            client_char_name = client_character_details.get("name", client_actor_id)
            # print(f"CSM: Preparing to get response from client: {client_char_name} ({client_actor_id})")

            context_for_client = character_texts.copy() # Context up to this point

            # Defensive: Ensure no None is passed to send_to_client
            if client_actor_id is None or client_ip is None or client_port is None:
                print(f"CSM: Skipping client due to missing info: actor_id={client_actor_id}, ip={client_ip}, port={client_port}")
                continue
            # ClientManager.send_to_client is now async
            task = self.client_manager.send_to_client(
                str(client_actor_id), str(client_ip), int(client_port), narration_text, context_for_client
            )
            client_response_tasks.append((client_char_name, client_actor_id, task))

        # Gather responses from all clients concurrently
        # print(f"CSM: Gathering responses from {len(client_response_tasks)} clients...")
        for char_name, actor_id, task in client_response_tasks:
            try:
                client_response_text = await task # await the future
                if client_response_text:
                    character_texts[char_name] = client_response_text
                # else:
                    # print(f"CSM: Client {char_name} ({actor_id}) provided no/empty response.")
            except Exception as e:
                print(f"CSM: Error processing response from client {char_name} ({actor_id}): {e}")
                # ClientManager's send_to_client already handles DB status updates on failure

        # 3. Apply Chaos (assuming chaos_engine is fast and not I/O bound)
        if chaos_level > 0 and self.chaos_engine.random_factor() < (chaos_level / 10.0):
            # print(f"CSM: Applying chaos (level {chaos_level})...")
            # If apply_chaos becomes complex/slow, it could also be run in a thread
            narration_text, character_texts = self.chaos_engine.apply_chaos(narration_text, character_texts)

        # 4. Update Hardware (assuming hardware update is fast or non-critical path)
        # If self.hardware.update_leds is blocking and slow, use asyncio.to_thread
        await asyncio.to_thread(self.hardware.update_leds, narration_text)

        # 5. Save to Database (DB calls wrapped in to_thread)
        if narration_text or character_texts:
            # print("CSM: Saving story turn to database...")
            await asyncio.to_thread(self.db.save_story, narration_text, character_texts, narrator_audio_path_db)
            # print("CSM: Story turn processed and saved.")
        # else:
            # print("CSM: Nothing to save for this story turn.")

        return narration_text, character_texts

    def update_last_narration_text(self, new_text):
        """
        Update the most recent narrator entry in the story log with new text.
        
        Parameters:
        	new_text (str): The corrected narration text to update.
        
        Returns:
        	bool: True if a narrator entry was updated; False if no such entry exists.
        """
        # Get the last narrator entry
        history = self.db.get_story_history()
        narrator_entries = [entry for entry in history if entry["speaker"] == "Narrator"]
        if narrator_entries:
            last_entry = narrator_entries[-1]
            self.db.update_story_entry(last_entry["id"], new_text=new_text)
            print(f"CSM: Updated last narrator transcription in DB (id={last_entry['id']})")
            return True
        print("CSM: No narrator entry found to update.")
        return False

    async def shutdown_async(self): # Renamed for clarity
        """
        Asynchronously shuts down CSM resources by stopping client health checks and closing the database connection.
        """
        print("CSM: Async shutdown initiated...")
        await asyncio.to_thread(self.client_manager.stop_periodic_health_checks)
        await asyncio.to_thread(self.db.close)
        print("CSM: Async shutdown complete.")

    # __del__ is tricky with async, better to rely on explicit shutdown call from main app.
    # If using __del__, ensure it doesn't try to run async code directly without a loop.
    # For now, removing __del__ and relying on explicit shutdown.