from .database import Database
from .narrator import Narrator
from .character_server import CharacterServer # Should be async
from .client_manager import ClientManager   # Should be async
from .hardware import Hardware
from .chaos_engine import ChaosEngine
from .config import DB_PATH
import asyncio # Added asyncio
import os
import logging

logger = logging.getLogger("dreamweaver_server")

class CSM:
    def __init__(self):
        """
        Initializes the CSM instance and its core components, including database access, narration, character server, client management, hardware control, and chaos engine. Starts periodic health checks for clients upon initialization.
        """
        self.db = Database(DB_PATH)
        self.narrator = Narrator()
        self.character_server = CharacterServer(self.db)
        self.client_manager = ClientManager(self.db)
        self.hardware = Hardware()
        self.chaos_engine = ChaosEngine()

        self.client_manager.start_periodic_health_checks() # This will log its own startup
        logger.info("CSM Initialized.")

    async def process_story(self, audio_filepath: str, chaos_level: float):
        """
        Asynchronously processes a story turn by generating narration, collecting character responses from the server and online clients, applying chaos effects, updating hardware, and saving the results to the database.
        
        Parameters:
            audio_filepath (str): Path to the audio file containing the narration.
            chaos_level (float): The level of chaos to potentially apply to the narration and character responses.
        
        Returns:
            tuple: A tuple containing the final narration text (str) and a dictionary mapping character names to their response texts.
        """
        logger.info(f"CSM: Starting to process story for audio_filepath: {audio_filepath}, chaos_level: {chaos_level}")
        # Narrator process_narration is now async
        # logger.debug("CSM: Processing narration...")
        narration_data = await self.narrator.process_narration(audio_filepath)
        narration_text = narration_data.get("text", "") if narration_data else ""
        narrator_audio_path_db = narration_data.get("audio_path")

        if not narration_text:
            logger.warning("CSM: Narration text empty after processing. Skipping further story processing for this turn.")
            return "", {}

        character_texts = {}

        # 1. Server Character (Actor1) Response
        server_character_details = await asyncio.to_thread(self.db.get_character, "Actor1") # DB call in thread
        if server_character_details:
            logger.debug(f"CSM: Getting response from server character: {server_character_details.get('name', 'Actor1')}")
            # CharacterServer.generate_response is now async
            server_response_text = await self.character_server.generate_response(narration_text, {})
            if server_response_text:
                character_texts[server_character_details.get('name', 'Actor1')] = server_response_text
                logger.debug(f"CSM: Server character {server_character_details.get('name', 'Actor1')} responded.")
            else:
                logger.debug(f"CSM: Server character {server_character_details.get('name', 'Actor1')} provided no response.")
        else:
            logger.debug("CSM: Actor1 (server character) not configured or found.")

        # 2. Get responses from 'Online_Responsive' clients
        # ClientManager.get_clients_for_story_progression uses DB, run in thread
        responsive_clients = await asyncio.to_thread(self.client_manager.get_clients_for_story_progression)
        logger.info(f"CSM: Found {len(responsive_clients)} responsive clients for story progression.")

        client_response_tasks = []
        for client_info in responsive_clients:
            client_actor_id = client_info.get("Actor_id")
            client_ip = client_info.get("ip_address")
            client_port = client_info.get("client_port")

            # Fetch character details (DB call) - can also be done concurrently if many clients
            # For now, keeping it sequential before dispatching the send_to_client task
            client_character_details = await asyncio.to_thread(self.db.get_character, client_actor_id)
            if not client_character_details:
                logger.warning(f"CSM: No character details found for responsive client {client_actor_id}. Skipping.")
                continue

            client_char_name = client_character_details.get("name", client_actor_id)
            logger.debug(f"CSM: Preparing to get response from client: {client_char_name} ({client_actor_id}) at {client_ip}:{client_port}")

            context_for_client = character_texts.copy() # Context up to this point

            # Defensive: Ensure no None is passed to send_to_client
            if client_actor_id is None or client_ip is None or client_port is None: # Should ideally not happen if from DB
                logger.error(f"CSM: Skipping client due to missing info from DB record: actor_id={client_actor_id}, ip={client_ip}, port={client_port}")
                continue
            # ClientManager.send_to_client is now async
            task = self.client_manager.send_to_client(
                str(client_actor_id), str(client_ip), int(client_port), narration_text, context_for_client
            )
            client_response_tasks.append((client_char_name, client_actor_id, task))

        # Gather responses from all clients concurrently
        logger.debug(f"CSM: Gathering responses from {len(client_response_tasks)} clients...")
        for char_name, actor_id, task in client_response_tasks:
            try:
                client_response_text = await task # await the future
                if client_response_text:
                    character_texts[char_name] = client_response_text
                    logger.debug(f"CSM: Received response from client {char_name} ({actor_id}).")
                else:
                    logger.debug(f"CSM: Client {char_name} ({actor_id}) provided no/empty response.")
            except Exception as e:
                logger.error(f"CSM: Error processing/gathering response from client {char_name} ({actor_id}): {e}", exc_info=True)
                # ClientManager's send_to_client already handles DB status updates on failure

        # 3. Apply Chaos (assuming chaos_engine is fast and not I/O bound)
        if chaos_level > 0 and self.chaos_engine.random_factor() < (chaos_level / 10.0):
            logger.info(f"CSM: Applying chaos (level {chaos_level})...")
            # If apply_chaos becomes complex/slow, it could also be run in a thread
            narration_text, character_texts = self.chaos_engine.apply_chaos(narration_text, character_texts)
            logger.debug(f"CSM: Chaos applied. Narration: '{narration_text[:50]}...', Char texts: { {k: v[:50] for k,v in character_texts.items()} }")


        # 4. Update Hardware (assuming hardware update is fast or non-critical path)
        # If self.hardware.update_leds is blocking and slow, use asyncio.to_thread
        logger.debug("CSM: Updating hardware LEDs based on narration.")
        await asyncio.to_thread(self.hardware.update_leds, narration_text)

        # 5. Save to Database (DB calls wrapped in to_thread)
        if narration_text or character_texts:
            logger.info("CSM: Saving story turn to database...")
            await asyncio.to_thread(self.db.save_story, narration_text, character_texts, narrator_audio_path_db)
            logger.info("CSM: Story turn processed and saved.")
        else:
            logger.info("CSM: Nothing to save for this story turn (empty narration and character texts).")

        return narration_text, character_texts

    def update_last_narration_text(self, new_text):
        """
        Update the most recent narrator entry in the story log with new text.
        
        Parameters:
        	new_text (str): The corrected text to replace the last narrator entry.
        
        Returns:
        	bool: True if the update was successful, False if no narrator entry was found.
        """
        # Get the last narrator entry
        history = self.db.get_story_history()
        narrator_entries = [entry for entry in history if entry["speaker"] == "Narrator"]
        if narrator_entries:
            last_entry = narrator_entries[-1]
            self.db.update_story_entry(last_entry["id"], new_text=new_text)
            logger.info(f"CSM: Updated last narrator transcription in DB (id={last_entry['id']}) with new text: {new_text[:100]}...")
            return True
        logger.warning("CSM: No narrator entry found to update.")
        return False

    async def shutdown_async(self): # Renamed for clarity
        """
        Asynchronously shuts down CSM resources by stopping client health checks and closing the database connection.
        """
        logger.info("CSM: Async shutdown initiated...")
        await asyncio.to_thread(self.client_manager.stop_periodic_health_checks) # This method now logs
        await asyncio.to_thread(self.db.close) # Database close method might also log
        logger.info("CSM: Async shutdown complete.")

    # __del__ is tricky with async, better to rely on explicit shutdown call from main app.
    # If using __del__, ensure it doesn't try to run async code directly without a loop.
    # For now, removing __del__ and relying on explicit shutdown.