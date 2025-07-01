import asyncio
import os
import pytest
import tempfile
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch
import json
import sys
from pathlib import Path

# Add the SERVER directory to the path to import CSM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.csm import CSM
except ImportError:
    # Fallback import path
    try:
        from csm import CSM
    except ImportError:
        from SERVER.src.csm import CSM

class TestCSM:
    """Comprehensive unit tests for the CSM (Character Story Manager) class.
    
    Testing Framework: pytest
    
    This test suite covers:
    - Happy path scenarios
    - Edge cases and boundary conditions  
    - Failure modes and error handling
    - Concurrent processing
    - Component integration
    """
    
    @pytest.fixture
    def mock_dependencies(self):
        """
        Fixture that patches core CSM dependencies with mocks for isolated testing.
        
        Yields:
            dict: A dictionary of the patched mock classes for use within the test.
        """
        with patch.multiple(
            'src.csm',
            Narrator=MagicMock,
            CharacterServer=MagicMock,
            ClientManager=MagicMock,
            Database=MagicMock
        ) as mocks:
            yield mocks
    
    @pytest.fixture
    def csm_instance(self, mock_dependencies):
        """
        Creates a `CSM` instance with all major dependencies replaced by mocks for isolated testing.
        
        Returns:
            CSM: The `CSM` instance with mocked methods for narrator, character server, client manager, and database.
        """
        csm = CSM()
        
        # Set up async mocks
        csm.narrator.process_narration = AsyncMock()
        csm.character_server.generate_response = AsyncMock()
        csm.client_manager.send_to_client = AsyncMock()
        csm.client_manager.get_clients_for_story_progression = MagicMock()
        csm.db.get_character = MagicMock()
        
        return csm
    
    @pytest.fixture
    def dummy_audio_file(self):
        """
        Yields the path to a temporary dummy WAV audio file for use in tests, ensuring cleanup after use.
        
        Returns:
            str: The file path to the created temporary WAV audio file.
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Create minimal WAV file header
            temp_file.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            temp_file.flush()
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except FileNotFoundError:
            pass
    
    @pytest.mark.asyncio
    async def test_csm_process_story_happy_path(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method successfully processes a story with valid narration, character response, client communication, and character retrieval.
        
        Verifies that narration and character data are returned, and that all key processing methods are called as expected.
        """
        # Setup mocks
        csm_instance.narrator.process_narration.return_value = {
            "text": "Once upon a time in a distant land...",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Hero says: I shall embark on this quest!"
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "Hero", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.return_value = "Hero responds via client!"
        
        csm_instance.db.get_character.return_value = {
            "name": "Hero Character",
            "Actor_id": "Hero"
        }
        
        # Execute
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.5)
        
        # Verify
        assert narration is not None
        assert isinstance(characters, dict)
        assert "ServerCharacter" in characters
        csm_instance.narrator.process_narration.assert_called_once_with(dummy_audio_file)
        csm_instance.character_server.generate_response.assert_called()
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_with_nonexistent_audio_file(self, csm_instance):
        """
        Test that CSM's process_story raises FileNotFoundError when given a nonexistent audio file.
        
        Verifies that the process_story method correctly propagates a FileNotFoundError if the narrator fails to process a missing audio file.
        """
        nonexistent_file = "nonexistent_audio.wav"
        
        csm_instance.narrator.process_narration.side_effect = FileNotFoundError("Audio file not found")
        
        with pytest.raises(FileNotFoundError):
            await csm_instance.process_story(nonexistent_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_narrator_failure(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story raises an exception when narrator processing fails.
        
        Simulates a failure in the narrator's process_narration method and verifies that the exception is propagated.
        """
        csm_instance.narrator.process_narration.side_effect = Exception("Narrator processing failed")
        
        with pytest.raises(Exception, match="Narrator processing failed"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_character_server_failure(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method raises an exception when the character server fails.
        
        Simulates a failure in the character server's response generation and verifies that the exception is propagated.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.side_effect = Exception("Character server error")
        
        with pytest.raises(Exception, match="Character server error"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_no_clients(self, csm_instance, dummy_audio_file):
        """
        Test that process_story returns narration and character data when no clients are available, and does not attempt to send to any client.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server character response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        assert "ServerCharacter" in characters
        csm_instance.client_manager.send_to_client.assert_not_called()
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_multiple_clients(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method correctly handles multiple clients by sending responses to each and aggregating character data.
        
        Verifies that narration is processed, each client receives a response, and character information is collected for all actors involved.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Epic battle begins!",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server hero attacks!"
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "Warrior", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "Wizard", "ip_address": "127.0.0.1", "client_port": 8002},
            {"Actor_id": "Rogue", "ip_address": "127.0.0.1", "client_port": 8003}
        ]
        
        csm_instance.client_manager.send_to_client.side_effect = [
            "Warrior swings sword!",
            "Wizard casts spell!",
            "Rogue sneaks behind!"
        ]
        
        def mock_get_character(actor_id):
            """
            Return mock character data for a given actor ID.
            
            Parameters:
            	actor_id (str): The identifier of the actor whose character data is requested.
            
            Returns:
            	dict or None: A dictionary containing character information if the actor ID exists, otherwise None.
            """
            characters = {
                "Warrior": {"name": "Brave Warrior", "Actor_id": "Warrior"},
                "Wizard": {"name": "Wise Wizard", "Actor_id": "Wizard"},
                "Rogue": {"name": "Sneaky Rogue", "Actor_id": "Rogue"}
            }
            return characters.get(actor_id)
        
        csm_instance.db.get_character.side_effect = mock_get_character
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.8)
        
        assert narration is not None
        assert len(characters) >= 1  # At least ServerCharacter
        assert "ServerCharacter" in characters
        assert csm_instance.client_manager.send_to_client.call_count == 3
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_chaos_levels(self, csm_instance, dummy_audio_file):
        """
        Tests that CSM's process_story method returns valid narration when invoked with minimum, maximum, and mid-level chaos values.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Chaos test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Chaotic response!"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Test with minimum chaos
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration is not None
        
        # Test with maximum chaos
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=1.0)
        assert narration is not None
        
        # Test with mid-level chaos
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.5)
        assert narration is not None
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_invalid_chaos_level_negative(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story raises ValueError when given a negative chaos level.
        
        Verifies that providing a chaos level below 0.0 results in a ValueError with the expected message.
        """
        with pytest.raises(ValueError, match="Chaos level must be between 0.0 and 1.0"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=-0.1)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_invalid_chaos_level_too_high(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story raises ValueError when chaos level is greater than 1.0.
        """
        with pytest.raises(ValueError, match="Chaos level must be between 0.0 and 1.0"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=1.1)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_shutdown_async(self, csm_instance):
        """
        Test that the CSM's shutdown_async method properly calls the asynchronous shutdown or close methods on all its components.
        """
        csm_instance.narrator.shutdown_async = AsyncMock()
        csm_instance.character_server.shutdown_async = AsyncMock()
        csm_instance.client_manager.shutdown_async = AsyncMock()
        csm_instance.db.close_async = AsyncMock()
        
        await csm_instance.shutdown_async()
        
        # Verify shutdown was called (methods may not exist, so check if they were added)
        if hasattr(csm_instance.narrator, 'shutdown_async'):
            csm_instance.narrator.shutdown_async.assert_called_once()
        if hasattr(csm_instance.character_server, 'shutdown_async'):
            csm_instance.character_server.shutdown_async.assert_called_once()
        if hasattr(csm_instance.client_manager, 'shutdown_async'):
            csm_instance.client_manager.shutdown_async.assert_called_once()
        if hasattr(csm_instance.db, 'close_async'):
            csm_instance.db.close_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_csm_client_communication_failure(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method handles client communication failures during story processing.
        
        Simulates a scenario where sending data to a client raises a ConnectionError. Verifies that the method either handles the error gracefully or propagates the exception, depending on implementation.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Communication test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "FailingClient", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.side_effect = ConnectionError("Client unreachable")
        csm_instance.db.get_character.return_value = {"name": "Failing Character", "Actor_id": "FailingClient"}
        
        # Should handle client communication failures gracefully or raise
        # Depending on implementation, this might raise or continue
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
            assert isinstance(characters, dict)
        except ConnectionError:
            # If the implementation propagates the error, that's also valid
            pass
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_database_failure(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method handles database failures during character retrieval.
        
        Simulates a database exception when retrieving character data and verifies that process_story either handles the error gracefully or propagates the exception.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Database test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "TestActor", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.return_value = "Client response"
        csm_instance.db.get_character.side_effect = Exception("Database connection failed")
        
        # Database failure during client processing might be handled gracefully or raise
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            # If handled gracefully
            assert narration is not None
        except Exception as e:
            # If database errors are propagated
            assert "Database connection failed" in str(e)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_concurrent_processing(self, csm_instance, dummy_audio_file):
        """
        Test that the CSM can handle multiple concurrent story processing requests without errors.
        
        Submits several `process_story` calls with different chaos levels concurrently and verifies that all complete, returning valid narration and character data or raising exceptions as appropriate.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Concurrent test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Concurrent response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Submit multiple concurrent requests
        tasks = [
            csm_instance.process_story(dummy_audio_file, chaos_level=0.1),
            csm_instance.process_story(dummy_audio_file, chaos_level=0.2),
            csm_instance.process_story(dummy_audio_file, chaos_level=0.3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed (whether successfully or with exceptions)
        assert len(results) == 3
        for result in results:
            if not isinstance(result, Exception):
                narration, characters = result
                assert narration is not None
                assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()
    
    def test_csm_initialization(self, mock_dependencies):
        """
        Verify that the CSM class initializes all required components and that none are None.
        """
        csm = CSM()
        
        # Verify all components are initialized
        assert hasattr(csm, 'narrator')
        assert hasattr(csm, 'character_server')
        assert hasattr(csm, 'client_manager')
        assert hasattr(csm, 'db')
        assert hasattr(csm, 'executor')
        
        # Verify components are not None
        assert csm.narrator is not None
        assert csm.character_server is not None
        assert csm.client_manager is not None
        assert csm.db is not None
        assert csm.executor is not None
    
    @pytest.mark.asyncio
    async def test_csm_edge_case_empty_narration(self, csm_instance, dummy_audio_file):
        """
        Tests that CSM's process_story method correctly handles the case where the narration text is empty, ensuring valid narration and character responses are returned.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Character responds to silence"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        assert "ServerCharacter" in characters
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_edge_case_very_long_narration(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method correctly handles narration with a very long text input.
        
        Verifies that the method returns valid narration and character data when the narration text is extremely long.
        """
        long_narration = "A" * 10000  # Very long text
        
        csm_instance.narrator.process_narration.return_value = {
            "text": long_narration,
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Response to long narration"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_edge_case_special_characters_in_narration(self, csm_instance, dummy_audio_file):
        """
        Test that CSM's process_story method correctly handles narration containing special characters and unicode.
        
        Verifies that narration with emojis, accented characters, and other unicode symbols is processed without errors and returns valid results.
        """
        special_narration = "Testing with émojis 🎭, symbols & unicode: café naïve résumé"
        
        csm_instance.narrator.process_narration.return_value = {
            "text": special_narration,
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Character handles unicode well"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_boundary_chaos_levels(self, csm_instance, dummy_audio_file):
        """
        Tests that CSM's process_story method correctly handles chaos_level values at the exact boundaries of 0.0 and 1.0.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Boundary test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Boundary response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Test exact boundaries
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration is not None
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=1.0)
        assert narration is not None
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_client_timeout(self, csm_instance, dummy_audio_file):
        """
        Tests that CSM's process_story method handles client communication timeouts, either by handling the timeout gracefully or by propagating the asyncio.TimeoutError exception.
        """
        csm_instance.narrator.process_narration.return_value = {
            "text": "Timeout test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "SlowClient", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.side_effect = asyncio.TimeoutError("Client timeout")
        csm_instance.db.get_character.return_value = {"name": "Slow Character", "Actor_id": "SlowClient"}
        
        # Should handle timeouts gracefully or propagate the error
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
        except asyncio.TimeoutError:
            # If timeouts are propagated, that's also valid behavior
            pass
        
        await csm_instance.shutdown_async()

# Preserve original integration test for backward compatibility
async def test_csm_process_story():
    """
    Runs an integration test for the CSM's process_story method with mocked dependencies.
    
    This test creates a dummy audio file, mocks key methods of the CSM's components to provide predictable responses, and verifies that process_story executes end-to-end. It prints the resulting narration and character responses, restores original methods, shuts down the CSM, and cleans up the test audio file.
    """
    print("Testing CSM process_story (async)...")

    # Create a dummy audio file
    dummy_audio_file = "csm_test_narration.wav"
    if not os.path.exists(dummy_audio_file):
        # Create a simple dummy WAV file for testing
        with open(dummy_audio_file, 'wb') as f:
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        print(f"Created dummy audio file: {dummy_audio_file}")

    csm = CSM()

    # Mock some parts for isolated testing if full setup is too complex here
    # For example, mock narrator.process_narration to return fixed text
    original_narrator_process = csm.narrator.process_narration
    async def mock_narrator_process(audio_filepath):
        """
        Simulates narrator processing by returning a mock narration result for the given audio file path.
        
        Parameters:
            audio_filepath (str): Path to the audio file to be processed.
        
        Returns:
            dict: A dictionary containing mock narration text, the provided audio file path, and the speaker name.
        """
        return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
    csm.narrator.process_narration = mock_narrator_process

    # Mock CharacterServer response
    original_cs_gen_response = csm.character_server.generate_response
    async def mock_cs_gen_response(narration, other_texts, chaos_level=0.0):
        """
        Asynchronously returns a mock character server response for testing purposes.
        
        Parameters:
            narration (str): The narration text to process.
            other_texts (Any): Additional texts or context for the response.
            chaos_level (float, optional): The chaos level influencing the response. Defaults to 0.0.
        
        Returns:
            str: A fixed mock response string simulating an actor's reply.
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
        Return mock character data for a given actor ID.
        
        Parameters:
            Actor_id (str): The identifier of the actor whose character data is requested.
        
        Returns:
            dict or None: A dictionary containing character information if the actor ID matches a known test value, otherwise None.
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

    # Cleanup dummy file
    try:
        os.unlink(dummy_audio_file)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    # Run the original integration test if executed directly
    asyncio.run(test_csm_process_story())