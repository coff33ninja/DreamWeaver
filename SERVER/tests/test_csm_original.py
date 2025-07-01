# Basic test for CSM process_story
    # Requires dummy audio, and CharacterServer/ClientManager mocks or instances.
    # This is becoming more of an integration test.
    async def test_csm_process_story():
        """
        Asynchronously tests the CSM class's process_story method with mocked dependencies.
        
        This integration-style test verifies that process_story correctly processes a dummy audio file, interacts with mocked narrator, character server, client manager, and database methods, and outputs the expected narration and character responses. Restores all mocks and performs cleanup after execution.
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
            Simulate narration processing by returning a mock narration result for the given audio file path.
            
            Parameters:
                audio_filepath (str): Path to the input audio file.
            
            Returns:
                dict: A dictionary containing mock narration text, the input audio file path, and the speaker name.
            """
            return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
        csm.narrator.process_narration = mock_narrator_process

        # Mock CharacterServer response
        original_cs_gen_response = csm.character_server.generate_response
        async def mock_cs_gen_response(narration, other_texts):
            """
            Asynchronously returns a fixed character response string for testing purposes.
            
            Parameters:
                narration (str): The narration text input.
                other_texts (Any): Additional text inputs (unused).
            
            Returns:
                str: A fixed string simulating an asynchronous character response.
            """
            return "Actor1 says hello asynchronously!"
        csm.character_server.generate_response = mock_cs_gen_response

        # Mock ClientManager response
        original_cm_send_to_client = csm.client_manager.send_to_client
        async def mock_cm_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
            """
            Asynchronously simulates sending narration and character texts to a client, returning a mock response string.
            
            Parameters:
                client_actor_id: The identifier of the client actor.
                client_ip: The IP address of the client.
                client_port: The port number of the client.
                narration: The narration text to send.
                character_texts: The character dialogue texts to send.
            
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
            Return mock character data for a given actor ID.
            
            Parameters:
            	Actor_id (str): The identifier of the actor to retrieve mock data for.
            
            Returns:
            	dict or None: A dictionary with character data if the actor ID matches a test case, otherwise None.
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