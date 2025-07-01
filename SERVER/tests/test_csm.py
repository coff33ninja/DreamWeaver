import pytest
import asyncio
import os
import tempfile
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch, call
import sys
import logging

# Add the SERVER/src directory to the path to import CSM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from csm import CSM
except ImportError:
    # Mock CSM for testing purposes if not available
    class MockCSM:
        def __init__(self):
            """
            Initialize the mock CSM instance with mocked dependencies for testing purposes.
            
            All major components (`narrator`, `character_server`, `client_manager`, `db`, `hardware`, `chaos_engine`) are set as `MagicMock` objects to allow isolated unit testing without real implementations.
            """
            self.narrator = MagicMock()
            self.character_server = MagicMock()
            self.client_manager = MagicMock() 
            self.db = MagicMock()
            self.hardware = MagicMock()
            self.chaos_engine = MagicMock()
            
        async def process_story(self, audio_file, chaos_level=0.0):
            """
            Processes a story audio file and returns a narration along with character responses.
            
            Parameters:
                audio_file: The path to the audio file to be processed.
                chaos_level (float, optional): The level of randomness or chaos to apply during processing. Defaults to 0.0.
            
            Returns:
                tuple: A tuple containing the narration text (str) and a dictionary mapping actor names to their responses (dict).
            """
            return "Mock narration", {"Actor1": "Mock response"}
            
        def update_last_narration_text(self, new_text):
            """
            Stub method for updating the last narration text.
            
            Parameters:
                new_text (str): The new narration text to update.
            
            Returns:
                bool: Always returns True as a placeholder.
            """
            return True
            
        async def shutdown_async(self):
            """
            Asynchronously shuts down the CSM instance, releasing any allocated resources.
            """
            pass
    
    CSM = MockCSM


class TestCSM:
    """Comprehensive unit tests for CSM (Character Story Manager)."""
    
    @pytest.fixture
    def csm_instance(self):
        """
        Provides a pytest fixture that returns a new `CSM` instance with all dependencies mocked for isolated testing.
        """
        with patch.multiple(
            'csm',
            Database=MagicMock,
            Narrator=MagicMock,
            CharacterServer=MagicMock,
            ClientManager=MagicMock,
            Hardware=MagicMock,
            ChaosEngine=MagicMock
        ):
            csm = CSM()
            # Setup basic mocks
            csm.narrator = MagicMock()
            csm.character_server = MagicMock()
            csm.client_manager = MagicMock()
            csm.db = MagicMock()
            csm.hardware = MagicMock()
            csm.chaos_engine = MagicMock()
            return csm
    
    @pytest.fixture
    def dummy_audio_file(self):
        """
        Yields the path to a temporary WAV audio file with a minimal valid header for use in tests.
        
        The file is automatically deleted after use.
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Write minimal WAV header
            f.write(b'RIFF\x24\x00\x00\x00WAVE')
            f.write(b'fmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00')
            f.write(b'data\x00\x00\x00\x00')
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    def test_csm_initialization(self):
        """
        Verify that the CSM instance initializes with all required dependencies as attributes.
        """
        with patch.multiple(
            'csm',
            Database=MagicMock,
            Narrator=MagicMock,
            CharacterServer=MagicMock,
            ClientManager=MagicMock,
            Hardware=MagicMock,
            ChaosEngine=MagicMock
        ):
            csm = CSM()
            
            # Verify all dependencies are initialized
            assert hasattr(csm, 'narrator')
            assert hasattr(csm, 'character_server')
            assert hasattr(csm, 'client_manager')
            assert hasattr(csm, 'db')
            assert hasattr(csm, 'hardware')
            assert hasattr(csm, 'chaos_engine')
    
    @pytest.mark.asyncio
    async def test_process_story_happy_path(self, csm_instance, dummy_audio_file):
        """
        Asynchronously tests the successful processing of a story by the CSM instance, ensuring all components interact correctly with mocked dependencies.
        
        This test simulates narration processing, character response generation, client communication, chaos application, hardware updates, and database saving. It verifies that the `process_story` method returns valid narration and character responses when all components function as expected.
        """
        # Setup mocks for async operations
        csm_instance.narrator.process_narration = AsyncMock(
            return_value={
                "text": "Test narration content",
                "audio_path": dummy_audio_file,
                "speaker": "Narrator"
            }
        )
        
        csm_instance.character_server.generate_response = AsyncMock(
            return_value="Character response"
        )
        
        csm_instance.client_manager.get_clients_for_story_progression = MagicMock(
            return_value=[{
                "Actor_id": "TestActor",
                "ip_address": "127.0.0.1",
                "client_port": 8001
            }]
        )
        
        csm_instance.client_manager.send_to_client = AsyncMock(
            return_value="Client response"
        )
        
        csm_instance.db.get_character = MagicMock(
            side_effect=lambda actor_id: {
                "Actor1": {"name": "ServerCharacter", "Actor_id": "Actor1"},
                "TestActor": {"name": "TestCharacter", "Actor_id": "TestActor"}
            }.get(actor_id)
        )
        
        csm_instance.chaos_engine.random_factor = MagicMock(return_value=0.5)
        csm_instance.chaos_engine.apply_chaos = MagicMock(
            return_value=("Modified narration", {"TestCharacter": "Chaotic response"})
        )
        
        csm_instance.hardware.update_leds = MagicMock()
        csm_instance.db.save_story = MagicMock()
        
        # Execute
        with patch('asyncio.to_thread') as mock_to_thread:
            # Mock asyncio.to_thread calls
            async def side_effect(func, *args):
                """
                Runs a synchronous function with the provided arguments in an asynchronous context.
                
                Parameters:
                	func (Callable): The function to execute.
                	*args: Arguments to pass to the function.
                
                Returns:
                	Result of the function call.
                """
                return func(*args)
            mock_to_thread.side_effect = side_effect
            
            narration, characters = await csm_instance.process_story(
                dummy_audio_file, chaos_level=0.5
            )
        
        # Verify results
        assert narration is not None
        assert isinstance(characters, dict)
        
        # Verify method calls
        csm_instance.narrator.process_narratio
        csm = CSM()

        # Mock some parts for isolated testing if full setup is too complex here
        # For example, mock narrator.process_narration to return fixed text
        original_narrator_process = csm.narrator.process_narration
        async def mock_narrator_process(audio_filepath):
            """
            Simulates narration processing by returning a mock narration result for the given audio file path.
            
            Parameters:
                audio_filepath (str): Path to the input audio file.
            
            Returns:
                dict: A dictionary containing mock narration text, the original audio file path, and the speaker name.
            """
            return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
        csm.narrator.process_narration = mock_narrator_process

        # Mock CharacterServer response
        original_cs_gen_response = csm.character_server.generate_response
        async def mock_cs_gen_response(narration, other_texts):
            """
            Asynchronously returns a mock character response for testing purposes.
            
            Parameters:
                narration (str): The narration text input.
                other_texts (Any): Additional context or texts for the response.
            
            Returns:
                str: A fixed mock response string.
            """
            return "Actor1 says hello asynchronously!"
        csm.character_server.generate_response = mock_cs_gen_response

        # Mock ClientManager response
        original_cm_send_to_client = csm.client_manager.send_to_client
        async def mock_cm_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
            """
            Asynchronously simulates sending narration and character texts to a client, returning a mock response message.
            
            Parameters:
                client_actor_id: The identifier of the client actor.
                client_ip: The IP address of the client.
                client_port: The port number of the client.
                narration: The narration text to send.
                character_texts: The character responses to send.
            
            Returns:
                str: A mock response message indicating the client actor has received the message.
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
            Return mock character information for a given actor ID.
            
            Parameters:
            	Actor_id (str): The identifier of the actor to look up.
            
            Returns:
            	dict or None: A dictionary with character information if the actor ID matches a known test value, otherwise None.
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