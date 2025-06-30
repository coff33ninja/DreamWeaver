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
        Asynchronously processes narration, gets responses from server character (PC1)
        and all 'Online_Responsive' clients, applies chaos, and saves the story turn.
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

        # 1. Server Character (PC1) Response
        server_character_details = await asyncio.to_thread(self.db.get_character, "PC1") # DB call in thread
        if server_character_details:
            # print(f"CSM: Getting response from server character: {server_character_details.get('name', 'PC1')}")
            # CharacterServer.generate_response is now async
            server_response_text = await self.character_server.generate_response(narration_text, {})
            if server_response_text:
                character_texts[server_character_details.get('name', 'PC1')] = server_response_text
            # else:
                # print(f"CSM: Server character {server_character_details.get('name', 'PC1')} provided no response.")
        # else:
            # print("CSM: PC1 (server character) not configured.")

        # 2. Get responses from 'Online_Responsive' clients
        # ClientManager.get_clients_for_story_progression uses DB, run in thread
        responsive_clients = await asyncio.to_thread(self.client_manager.get_clients_for_story_progression)
        # print(f"CSM: Found {len(responsive_clients)} responsive clients.")

        client_response_tasks = []
        for client_info in responsive_clients:
            client_pc_id = client_info.get("pc_id")
            client_ip = client_info.get("ip_address")
            client_port = client_info.get("client_port")

            # Fetch character details (DB call) - can also be done concurrently if many clients
            # For now, keeping it sequential before dispatching the send_to_client task
            client_character_details = await asyncio.to_thread(self.db.get_character, client_pc_id)
            if not client_character_details:
                print(f"CSM: Warning - No character details for responsive client {client_pc_id}. Skipping.")
                continue

            client_char_name = client_character_details.get("name", client_pc_id)
            # print(f"CSM: Preparing to get response from client: {client_char_name} ({client_pc_id})")

            context_for_client = character_texts.copy() # Context up to this point

            # Defensive: Ensure no None is passed to send_to_client
            if client_pc_id is None or client_ip is None or client_port is None:
                print(f"CSM: Skipping client due to missing info: pc_id={client_pc_id}, ip={client_ip}, port={client_port}")
                continue
            # ClientManager.send_to_client is now async
            task = self.client_manager.send_to_client(
                str(client_pc_id), str(client_ip), int(client_port), narration_text, context_for_client
            )
            client_response_tasks.append((client_char_name, client_pc_id, task))

        # Gather responses from all clients concurrently
        # print(f"CSM: Gathering responses from {len(client_response_tasks)} clients...")
        for char_name, pc_id, task in client_response_tasks:
            try:
                client_response_text = await task # await the future
                if client_response_text:
                    character_texts[char_name] = client_response_text
                # else:
                    # print(f"CSM: Client {char_name} ({pc_id}) provided no/empty response.")
            except Exception as e:
                print(f"CSM: Error processing response from client {char_name} ({pc_id}): {e}")
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

    async def shutdown_async(self): # Renamed for clarity
        """Asynchronously shut down CSM resources."""
        print("CSM: Async shutdown initiated...")
        await asyncio.to_thread(self.client_manager.stop_periodic_health_checks)
        await asyncio.to_thread(self.db.close)
        print("CSM: Async shutdown complete.")

    # __del__ is tricky with async, better to rely on explicit shutdown call from main app.
    # If using __del__, ensure it doesn't try to run async code directly without a loop.
    # For now, removing __del__ and relying on explicit shutdown.

if __name__ == '__main__':
    # Basic test for CSM process_story
    # Requires dummy audio, and CharacterServer/ClientManager mocks or instances.
    # This is becoming more of an integration test.
    async def test_csm_process_story():
        print("Testing CSM process_story (async)...")

        # Create a dummy audio file
        dummy_audio_file = "csm_test_narration.wav"
        if not os.path.exists(dummy_audio_file):
            # (Code to create a dummy WAV file, similar to narrator_test)
            print(f"Please create a dummy audio file: {dummy_audio_file}")
            # return

        csm = CSM()

        # Mock some parts for isolated testing if full setup is too complex here
        # For example, mock narrator.process_narration to return fixed text
        original_narrator_process = csm.narrator.process_narration
        async def mock_narrator_process(audio_filepath):
            return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
        csm.narrator.process_narration = mock_narrator_process

        # Mock CharacterServer response
        original_cs_gen_response = csm.character_server.generate_response
        async def mock_cs_gen_response(narration, other_texts):
            return "PC1 says hello asynchronously!"
        csm.character_server.generate_response = mock_cs_gen_response

        # Mock ClientManager response
        original_cm_send_to_client = csm.client_manager.send_to_client
        async def mock_cm_send_to_client(client_pc_id, client_ip, client_port, narration, character_texts):
            return f"{client_pc_id} says hi via async mock!"
        csm.client_manager.send_to_client = mock_cm_send_to_client

        # Mock get_clients_for_story_progression
        original_cm_get_clients = csm.client_manager.get_clients_for_story_progression
        def mock_cm_get_clients(): # This is called via to_thread, so sync mock is fine
            return [{"pc_id": "PC_TestClient", "ip_address": "127.0.0.1", "client_port": 8001}]
        csm.client_manager.get_clients_for_story_progression = mock_cm_get_clients

        # Mock DB get_character for the test client
        original_db_get_char = csm.db.get_character
        def mock_db_get_char(pc_id):
            if pc_id == "PC1":
                return {"name": "ServerTestChar", "pc_id": "PC1"}
            if pc_id == "PC_TestClient":
                return {"name": "RemoteTestChar", "pc_id": "PC_TestClient"}
            return None
        csm.db.get_character = mock_db_get_char


        print("Processing story with CSM...")
        narration, characters = await csm.process_story(dummy_audio_file, chaos_level=0.0)

        print("\n--- CSM Test Results ---")
        print(f"Narrator: {narration}")
        print("Characters:")
        for char, text in characters.items():
            print(f"  {char}: {text}")

        # Restore mocks if needed or expect test to end
        csm.narrator.process_narration = original_narrator_process
        csm.character_server.generate_response = original_cs_gen_response
        csm.client_manager.send_to_client = original_cm_send_to_client
        csm.client_manager.get_clients_for_story_progression = original_cm_get_clients
        csm.db.get_character = original_db_get_char

        await csm.shutdown_async() # Test shutdown

    asyncio.run(test_csm_process_story())
