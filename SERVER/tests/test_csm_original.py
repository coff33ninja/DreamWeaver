# Basic test for CSM process_story
    # Requires dummy audio, and CharacterServer/ClientManager mocks or instances.
    # This is becoming more of an integration test.
    async def test_csm_process_story():
        """
        Asynchronously tests the CSM process_story method by mocking dependencies and verifying narration and character responses.
        
        This integration-style test sets up a dummy audio file, mocks internal methods of the CSM instance to return controlled outputs, executes the process_story method, prints the results, restores the original methods, and shuts down the CSM instance.
        """
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
            """
            Simulate narration processing by returning a fixed narration text, the provided audio file path, and a speaker label.
            
            Parameters:
                audio_filepath (str): Path to the audio file to be processed.
            
            Returns:
                dict: Contains the narration text, audio file path, and speaker label.
            """
            return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
        csm.narrator.process_narration = mock_narrator_process

        # Mock CharacterServer response
        original_cs_gen_response = csm.character_server.generate_response
        async def mock_cs_gen_response(narration, other_texts):
            """
            Asynchronously returns a fixed character response string for testing purposes.
            
            Parameters:
                narration: The narration text input (unused).
                other_texts: Additional text inputs (unused).
            
            Returns:
                str: A fixed response string simulating a character's asynchronous reply.
            """
            return "Actor1 says hello asynchronously!"
        csm.character_server.generate_response = mock_cs_gen_response

        # Mock ClientManager response
        original_cm_send_to_client = csm.client_manager.send_to_client
        async def mock_cm_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
            """
            Asynchronously simulates sending narration and character texts to a client, returning a mock response string.
            
            Parameters:
            	client_actor_id: Identifier for the client actor.
            	client_ip: IP address of the client.
            	client_port: Port number of the client.
            	narration: Narration text to send.
            	character_texts: Character responses to send.
            
            Returns:
            	str: A mock response indicating the client actor sent a message.
            """
            return f"{client_actor_id} says hi via async mock!"
        csm.client_manager.send_to_client = mock_cm_send_to_client

        # Mock get_clients_for_story_progression
        original_cm_get_clients = csm.client_manager.get_clients_for_story_progression
        def mock_cm_get_clients(): # This is called via to_thread, so sync mock is fine
            """
            Return a list containing a single mock client dictionary for testing purposes.
            """
            return [{"Actor_id": "Actor_TestClient", "ip_address": "127.0.0.1", "client_port": 8001}]
        csm.client_manager.get_clients_for_story_progression = mock_cm_get_clients

        # Mock DB get_character for the test client
        original_db_get_char = csm.db.get_character
        def mock_db_get_char(Actor_id):
            """
            Return mock character data for specific actor IDs.
            
            Parameters:
                Actor_id (str): The actor ID to look up.
            
            Returns:
                dict or None: A dictionary with character information if the actor ID matches a test case; otherwise, None.
            """
            if Actor_id == "Actor1":
                return {"name": "ServerTestChar", "Actor_id": "Actor1"}
            if Actor_id == "Actor_TestClient":
                return {"name": "RemoteTestChar", "Actor_id": "Actor_TestClient"}
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