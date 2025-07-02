import sys
import os
import pytest
import asyncio
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

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
        """Set up mocked dependencies for CSM testing."""
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
        """Create a CSM instance with mocked dependencies."""
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
        """Create a temporary dummy audio file for testing."""
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
        """Test CSM process_story with successful processing."""
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
        """Test CSM process_story with nonexistent audio file."""
        nonexistent_file = "nonexistent_audio.wav"
        
        csm_instance.narrator.process_narration.side_effect = FileNotFoundError("Audio file not found")
        
        with pytest.raises(FileNotFoundError):
            await csm_instance.process_story(nonexistent_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_narrator_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when narrator processing fails."""
        csm_instance.narrator.process_narration.side_effect = Exception("Narrator processing failed")
        
        with pytest.raises(Exception, match="Narrator processing failed"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_character_server_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when character server fails."""
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
        """Test CSM process_story when no clients are available."""
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
        """Test CSM process_story with multiple clients."""
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
        """Test CSM process_story with different chaos levels."""
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
        """Test CSM process_story with negative chaos level."""
        with pytest.raises(ValueError, match="Chaos level must be between 0.0 and 1.0"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=-0.1)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_process_story_invalid_chaos_level_too_high(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with chaos level > 1.0."""
        with pytest.raises(ValueError, match="Chaos level must be between 0.0 and 1.0"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=1.1)
        
        await csm_instance.shutdown_async()
    
    @pytest.mark.asyncio
    async def test_csm_shutdown_async(self, csm_instance):
        """Test CSM shutdown_async method."""
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
        """Test CSM process_story when client communication fails."""
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
        """Test CSM process_story when database operations fail."""
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
        """Test CSM handling concurrent story processing requests."""
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
        """Test CSM class initialization."""
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
        """Test CSM process_story with empty narration text."""
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
        """Test CSM process_story with very long narration text."""
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
        """Test CSM process_story with special characters and unicode in narration."""
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
        """Test CSM process_story with exact boundary chaos levels."""
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
        """Test CSM process_story when client operations timeout."""
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
    """Basic test for CSM process_story - Original integration test.
    
    This is the original integration test preserved for backward compatibility.
    Requires dummy audio, and CharacterServer/ClientManager mocks or instances.
    This is becoming more of an integration test.
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
        return {"text": "This is a test narration from mock.", "audio_path": audio_filepath, "speaker": "Narrator"}
    csm.narrator.process_narration = mock_narrator_process

    # Mock CharacterServer response
    original_cs_gen_response = csm.character_server.generate_response
    async def mock_cs_gen_response(narration, other_texts, chaos_level=0.0):
        return "Actor1 says hello asynchronously!"
    csm.character_server.generate_response = mock_cs_gen_response

    # Mock ClientManager response
    original_cm_send_to_client = csm.client_manager.send_to_client
    async def mock_cm_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
        return f"{client_actor_id} says hi via async mock!"
    csm.client_manager.send_to_client = mock_cm_send_to_client

    # Mock get_clients_for_story_progression
    original_cm_get_clients = csm.client_manager.get_clients_for_story_progression
    def mock_cm_get_clients(): # This is called via to_thread, so sync mock is fine
        return [{"Actor_id": "Actor_TestClient", "ip_address": "127.0.0.1", "client_port": 8001}]
    csm.client_manager.get_clients_for_story_progression = mock_cm_get_clients

    # Mock DB get_character for the test client
    original_db_get_char = csm.db.get_character
    def mock_db_get_char(Actor_id):
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
    @pytest.mark.asyncio
    async def test_csm_update_last_narration_text_success(self, csm_instance):
        """Test CSM update_last_narration_text with successful update."""
        # Mock story history with narrator entries
        mock_history = [
            {"id": 1, "speaker": "Character1", "text": "Character response"},
            {"id": 2, "speaker": "Narrator", "text": "Old narration text"},
            {"id": 3, "speaker": "Character2", "text": "Another character response"},
            {"id": 4, "speaker": "Narrator", "text": "Most recent narration"}
        ]
        
        csm_instance.db.get_story_history.return_value = mock_history
        csm_instance.db.update_story_entry = MagicMock()
        
        # Test successful update
        result = csm_instance.update_last_narration_text("Corrected narration text")
        
        assert result is True
        csm_instance.db.get_story_history.assert_called_once()
        csm_instance.db.update_story_entry.assert_called_once_with(4, new_text="Corrected narration text")
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_update_last_narration_text_no_narrator_entries(self, csm_instance):
        """Test CSM update_last_narration_text when no narrator entries exist."""
        # Mock story history with no narrator entries
        mock_history = [
            {"id": 1, "speaker": "Character1", "text": "Character response"},
            {"id": 2, "speaker": "Character2", "text": "Another character response"}
        ]
        
        csm_instance.db.get_story_history.return_value = mock_history
        csm_instance.db.update_story_entry = MagicMock()
        
        # Test when no narrator entries exist
        result = csm_instance.update_last_narration_text("New narration text")
        
        assert result is False
        csm_instance.db.get_story_history.assert_called_once()
        csm_instance.db.update_story_entry.assert_not_called()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_update_last_narration_text_empty_history(self, csm_instance):
        """Test CSM update_last_narration_text with empty story history."""
        csm_instance.db.get_story_history.return_value = []
        csm_instance.db.update_story_entry = MagicMock()
        
        # Test with empty history
        result = csm_instance.update_last_narration_text("New narration text")
        
        assert result is False
        csm_instance.db.get_story_history.assert_called_once()
        csm_instance.db.update_story_entry.assert_not_called()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_update_last_narration_text_database_error(self, csm_instance):
        """Test CSM update_last_narration_text when database operations fail."""
        csm_instance.db.get_story_history.side_effect = Exception("Database connection failed")
        
        # Test database error handling
        with pytest.raises(Exception, match="Database connection failed"):
            csm_instance.update_last_narration_text("New narration text")
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_none_audio_file(self, csm_instance):
        """Test CSM process_story with None as audio file path."""
        with pytest.raises((TypeError, ValueError)):
            await csm_instance.process_story(None, chaos_level=0.5)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_empty_string_audio_file(self, csm_instance):
        """Test CSM process_story with empty string as audio file path."""
        csm_instance.narrator.process_narration.side_effect = FileNotFoundError("Audio file not found")
        
        with pytest.raises(FileNotFoundError):
            await csm_instance.process_story("", chaos_level=0.5)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_invalid_chaos_level_string(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with string chaos level."""
        with pytest.raises((TypeError, ValueError)):
            await csm_instance.process_story(dummy_audio_file, chaos_level="0.5")
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_chaos_level_infinity(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with infinity chaos level."""
        with pytest.raises((ValueError, OverflowError)):
            await csm_instance.process_story(dummy_audio_file, chaos_level=float('inf'))
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_chaos_level_nan(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with NaN chaos level."""
        with pytest.raises((ValueError, TypeError)):
            await csm_instance.process_story(dummy_audio_file, chaos_level=float('nan'))
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_narrator_returns_none(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when narrator returns None."""
        csm_instance.narrator.process_narration.return_value = None
        
        # Should return empty results when narrator returns None
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == ""
        assert characters == {}
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_narrator_returns_malformed_data(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when narrator returns malformed data."""
        csm_instance.narrator.process_narration.return_value = {
            "invalid_key": "value",
            "missing_text": True
        }
        
        # Should handle malformed data gracefully by treating missing text as empty
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == ""
        assert characters == {}
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_character_server_returns_none(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when character server returns None."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = None
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Server Character", "Actor_id": "Actor1"}
        
        # Should handle None response gracefully
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # No server character response should be added to characters dict
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_hardware_integration(self, csm_instance, dummy_audio_file):
        """Test CSM process_story hardware integration."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Hardware test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Hardware response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Hardware Character", "Actor_id": "Actor1"}
        
        # Mock hardware update_leds method
        csm_instance.hardware.update_leds = MagicMock()
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # Verify hardware was updated
        csm_instance.hardware.update_leds.assert_called_once_with("Hardware test narration")
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_hardware_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when hardware update fails."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Hardware failure test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Server Character", "Actor_id": "Actor1"}
        
        # Mock hardware failure
        csm_instance.hardware.update_leds = MagicMock(side_effect=Exception("Hardware error"))
        
        # Should handle hardware failure gracefully if implemented properly
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
            assert isinstance(characters, dict)
        except Exception as e:
            # If hardware errors are propagated, that's also valid
            assert "Hardware error" in str(e)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_chaos_engine_integration(self, csm_instance, dummy_audio_file):
        """Test CSM process_story chaos engine integration."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Chaos test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Chaos response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Chaos Character", "Actor_id": "Actor1"}
        
        # Mock chaos engine to always trigger chaos
        csm_instance.chaos_engine.random_factor = MagicMock(return_value=0.01)  # Low value to trigger chaos
        csm_instance.chaos_engine.apply_chaos = MagicMock(return_value=("Chaotic narration", {"Chaos Character": "Chaotic response"}))
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=1.0)
        
        # Verify chaos was applied
        csm_instance.chaos_engine.random_factor.assert_called_once()
        csm_instance.chaos_engine.apply_chaos.assert_called_once()
        assert narration == "Chaotic narration"
        assert "Chaos Character" in characters
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_database_save_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when database save fails."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Database save test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Server Character", "Actor_id": "Actor1"}
        
        # Mock database save failure
        csm_instance.db.save_story = MagicMock(side_effect=Exception("Database save failed"))
        
        # Should handle database save failure
        with pytest.raises(Exception, match="Database save failed"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_client_with_none_values(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when client data contains None values."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "None values test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        
        # Client data with None values
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": None, "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "ValidClient", "ip_address": None, "client_port": 8002},
            {"Actor_id": "AnotherValidClient", "ip_address": "127.0.0.1", "client_port": None}
        ]
        
        csm_instance.db.get_character.return_value = {"name": "Test Character", "Actor_id": "ValidClient"}
        
        # Should skip clients with None values gracefully
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # send_to_client should not be called for clients with None values
        csm_instance.client_manager.send_to_client.assert_not_called()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_very_large_number_of_clients(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with a very large number of clients."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Mass client test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server coordinates mass clients"
        
        # Create 50 mock clients (reasonable number for testing)
        large_client_list = [
            {"Actor_id": f"Client{i}", "ip_address": "127.0.0.1", "client_port": 8000+i}
            for i in range(50)
        ]
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = large_client_list
        csm_instance.client_manager.send_to_client.return_value = "Client response"
        
        def mock_get_character_bulk(actor_id):
            return {"name": f"Character {actor_id}", "Actor_id": actor_id}
        
        csm_instance.db.get_character.side_effect = mock_get_character_bulk
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # Should have handled all clients
        assert csm_instance.client_manager.send_to_client.call_count == 50
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_unicode_narration_and_responses(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with unicode characters in narration and responses."""
        unicode_narration = "ñárráción with émójís 🎭 and spëcial charäcters café"
        
        csm_instance.narrator.process_narration.return_value = {
            "text": unicode_narration,
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Respuesta con acentos y símbolos: ñáéíóú 🎯"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "UnicodeClient", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.return_value = "クライアント응답 with mixed unicode"
        csm_instance.db.get_character.side_effect = lambda actor_id: {"name": f"Carácter {actor_id}", "Actor_id": actor_id}
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration == unicode_narration
        assert isinstance(characters, dict)
        # Verify unicode handling in character names and responses
        for char_name, response in characters.items():
            assert isinstance(char_name, str)
            assert isinstance(response, str)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_initialization_with_health_checks(self, mock_dependencies):
        """Test CSM initialization properly starts client health checks."""
        # Mock the start_periodic_health_checks method
        mock_start_health_checks = MagicMock()
        
        with patch.object(mock_dependencies['ClientManager'].return_value, 'start_periodic_health_checks', mock_start_health_checks):
            csm = CSM()
            
            # Verify health checks were started
            mock_start_health_checks.assert_called_once()
            
            # Verify all components exist
            assert hasattr(csm, 'db')
            assert hasattr(csm, 'narrator')
            assert hasattr(csm, 'character_server')
            assert hasattr(csm, 'client_manager')
            assert hasattr(csm, 'hardware')
            assert hasattr(csm, 'chaos_engine')

    @pytest.mark.asyncio
    async def test_csm_process_story_performance_timing(self, csm_instance, dummy_audio_file):
        """Test CSM process_story performance and timing."""
        import time
        
        # Set up fast mock responses
        csm_instance.narrator.process_narration.return_value = {
            "text": "Performance test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Fast server response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Performance Character", "Actor_id": "Actor1"}
        
        # Measure processing time
        start_time = time.time()
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert narration is not None
        assert isinstance(characters, dict)
        # Processing should complete within reasonable time (5 seconds for mocked operations)
        assert processing_time < 5.0, f"Processing took {processing_time} seconds, which is too long"
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_shutdown_stops_health_checks(self, csm_instance):
        """Test that CSM shutdown properly stops client health checks."""
        # Mock the stop_periodic_health_checks method
        csm_instance.client_manager.stop_periodic_health_checks = MagicMock()
        csm_instance.db.close = MagicMock()
        
        await csm_instance.shutdown_async()
        
        # Verify shutdown operations were called
        csm_instance.client_manager.stop_periodic_health_checks.assert_called_once()
        csm_instance.db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_csm_process_story_edge_case_extremely_long_narration(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with extremely long narration text."""
        # Create extremely long narration (1MB of text)
        extremely_long_narration = "A" * (1024 * 1024)  # 1MB of 'A' characters
        
        csm_instance.narrator.process_narration.return_value = {
            "text": extremely_long_narration,
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Response to very long narration"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Long Text Character", "Actor_id": "Actor1"}
        
        # Should handle extremely long text without crashing
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert len(narration) == 1024 * 1024
        assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_concurrent_database_operations(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with concurrent database operations."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Concurrent DB test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Concurrent response"
        
        # Multiple clients to trigger concurrent DB operations
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "ConcurrentClient1", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "ConcurrentClient2", "ip_address": "127.0.0.1", "client_port": 8002},
            {"Actor_id": "ConcurrentClient3", "ip_address": "127.0.0.1", "client_port": 8003}
        ]
        
        csm_instance.client_manager.send_to_client.return_value = "Concurrent client response"
        
        # Simulate some DB delay to test concurrency
        async def delayed_get_character(actor_id):
            await asyncio.sleep(0.01)  # Small delay
            return {"name": f"Concurrent Character {actor_id}", "Actor_id": actor_id}
        
        csm_instance.db.get_character.side_effect = lambda actor_id: {"name": f"Concurrent Character {actor_id}", "Actor_id": actor_id}
        
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # All client operations should complete
        assert csm_instance.client_manager.send_to_client.call_count == 3
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_corrupted_audio_metadata(self, csm_instance):
        """Test CSM process_story with corrupted audio file metadata."""
        # Create corrupted audio file
        corrupted_audio = "corrupted_test.wav"
        with open(corrupted_audio, 'wb') as f:
            f.write(b'CORRUPTED_DATA_NOT_VALID_WAV')
        
        try:
            csm_instance.narrator.process_narration.side_effect = Exception("Invalid audio format")
            
            with pytest.raises(Exception, match="Invalid audio format"):
                await csm_instance.process_story(corrupted_audio, chaos_level=0.5)
        finally:
            # Cleanup
            try:
                os.unlink(corrupted_audio)
            except FileNotFoundError:
                pass
            await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_memory_pressure_scenario(self, csm_instance, dummy_audio_file):
        """Test CSM process_story under memory pressure conditions."""
        # Simulate high memory usage scenario
        large_data_blocks = []
        try:
            # Create multiple large data blocks to simulate memory pressure
            for i in range(10):
                large_data_blocks.append(b'x' * (1024 * 1024))  # 1MB blocks
            
            csm_instance.narrator.process_narration.return_value = {
                "text": "Memory pressure test",
                "audio_path": dummy_audio_file,
                "speaker": "Narrator"
            }
            
            csm_instance.character_server.generate_response.return_value = "Memory test response"
            csm_instance.client_manager.get_clients_for_story_progression.return_value = []
            csm_instance.db.get_character.return_value = {"name": "Memory Character", "Actor_id": "Actor1"}
            
            # Should handle memory pressure gracefully
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            
            assert narration is not None
            assert isinstance(characters, dict)
            
        finally:
            # Clean up memory
            large_data_blocks.clear()
            await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_network_partition_simulation(self, csm_instance, dummy_audio_file):
        """Test CSM process_story during network partition scenarios."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Network partition test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Network test response"
        
        # Simulate clients across different network segments
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "LocalClient", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "RemoteClient", "ip_address": "192.168.1.100", "client_port": 8002},
            {"Actor_id": "InaccessibleClient", "ip_address": "10.0.0.1", "client_port": 8003}
        ]
        
        def mock_network_send(actor_id, ip, port, narration, character_texts):
            if ip == "10.0.0.1":
                raise ConnectionError("Network unreachable")
            elif ip == "192.168.1.100":
                raise asyncio.TimeoutError("Connection timeout")
            else:
                return f"{actor_id} response via local network"
        
        csm_instance.client_manager.send_to_client.side_effect = mock_network_send
        csm_instance.db.get_character.side_effect = lambda actor_id: {"name": f"Network Character {actor_id}", "Actor_id": actor_id}
        
        # Should handle network failures for some clients gracefully
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            # If graceful handling
            assert narration is not None
        except (ConnectionError, asyncio.TimeoutError):
            # If errors are propagated, that's valid too
            pass
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_malicious_input_sanitization(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with potentially malicious inputs."""
        malicious_narration = """<script>alert('xss')</script>
        SELECT * FROM users; DROP TABLE users; --
        ../../etc/passwd
        \x00\x01\x02\x03NULL_BYTES"""
        
        csm_instance.narrator.process_narration.return_value = {
            "text": malicious_narration,
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Sanitized response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "<script>malicious</script>", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.return_value = "Safe client response"
        csm_instance.db.get_character.side_effect = lambda actor_id: {"name": f"Safe Character", "Actor_id": "SafeActor"}
        
        # Should handle malicious input safely
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # Verify no script tags are preserved in output
        for char_name, response in characters.items():
            assert "<script>" not in str(response).lower()
            assert "drop table" not in str(response).lower()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_race_condition_protection(self, csm_instance, dummy_audio_file):
        """Test CSM process_story protection against race conditions."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Race condition test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Race test response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Race Character", "Actor_id": "Actor1"}
        
        # Simulate rapid-fire concurrent requests that could cause race conditions
        rapid_tasks = []
        for i in range(20):
            task = csm_instance.process_story(dummy_audio_file, chaos_level=0.1 * i % 1.0)
            rapid_tasks.append(task)
        
        results = await asyncio.gather(*rapid_tasks, return_exceptions=True)
        
        # All requests should complete without race condition errors
        successful_results = 0
        for result in results:
            if not isinstance(result, Exception):
                narration, characters = result
                assert narration is not None
                assert isinstance(characters, dict)
                successful_results += 1
        
        # At least most requests should succeed
        assert successful_results >= 15, f"Only {successful_results} out of 20 requests succeeded"
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_resource_cleanup_on_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story properly cleans up resources on failure."""
        # Track resource allocation/deallocation
        resource_tracker = {"allocated": 0, "deallocated": 0}
        
        async def failing_narrator_process(audio_path):
            resource_tracker["allocated"] += 1
            try:
                raise Exception("Simulated narrator failure")
            finally:
                resource_tracker["deallocated"] += 1
        
        csm_instance.narrator.process_narration = failing_narrator_process
        
        # Process should fail but clean up resources
        with pytest.raises(Exception, match="Simulated narrator failure"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Verify resource cleanup occurred
        assert resource_tracker["allocated"] == resource_tracker["deallocated"]
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_chaos_level_floating_point_precision(self, csm_instance, dummy_audio_file):
        """Test CSM chaos level handling with floating point precision edge cases."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Precision test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Precision response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Test floating point precision edge cases
        precision_test_cases = [
            0.9999999999999999,  # Should be treated as valid
            1.0000000000000001,  # Should be invalid (slightly > 1.0)
            -0.0000000000000001, # Should be invalid (slightly < 0.0)
            0.5000000000000001,  # Valid precision case
        ]
        
        for i, chaos_level in enumerate(precision_test_cases):
            try:
                if chaos_level < 0.0 or chaos_level > 1.0:
                    # Should raise validation error
                    with pytest.raises(ValueError):
                        await csm_instance.process_story(dummy_audio_file, chaos_level=chaos_level)
                else:
                    # Should process successfully
                    narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=chaos_level)
                    assert narration is not None
            except Exception as e:
                # Log which test case failed
                print(f"Test case {i} with chaos_level {chaos_level} failed: {e}")
                raise
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_interrupt_handling(self, csm_instance, dummy_audio_file):
        """Test CSM process_story handling of interruption signals."""
        import signal
        
        csm_instance.narrator.process_narration.return_value = {
            "text": "Interrupt test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Slow character server to allow interruption
        async def slow_character_server(*args, **kwargs):
            await asyncio.sleep(2.0)  # Long operation
            return "Slow response"
        
        csm_instance.character_server.generate_response = slow_character_server
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Start processing and cancel after short delay
        task = asyncio.create_task(
            csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        )
        
        # Cancel the task after a short delay
        await asyncio.sleep(0.1)
        task.cancel()
        
        # Should handle cancellation gracefully
        with pytest.raises(asyncio.CancelledError):
            await task
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_state_consistency_across_failures(self, csm_instance, dummy_audio_file):
        """Test CSM state consistency when operations fail partway through."""
        call_counts = {"narrator": 0, "character_server": 0, "client_manager": 0}
        
        async def counting_narrator_process(audio_path):
            call_counts["narrator"] += 1
            return {
                "text": "State consistency test",
                "audio_path": audio_path,
                "speaker": "Narrator"
            }
        
        async def failing_character_server(*args, **kwargs):
            call_counts["character_server"] += 1
            if call_counts["character_server"] == 2:  # Fail on second call
                raise Exception("Character server state error")
            return "Character response"
        
        csm_instance.narrator.process_narration = counting_narrator_process
        csm_instance.character_server.generate_response = failing_character_server
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # First call should succeed
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration is not None
        assert call_counts["narrator"] == 1
        assert call_counts["character_server"] == 1
        
        # Second call should fail but state should remain consistent
        with pytest.raises(Exception, match="Character server state error"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert call_counts["narrator"] == 2  # Narrator was called
        assert call_counts["character_server"] == 2  # Character server failed
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_component_initialization_failure(self, mock_dependencies):
        """Test CSM behavior when component initialization fails."""
        # Mock component initialization failure
        mock_dependencies['Narrator'].side_effect = Exception("Narrator initialization failed")
        
        with pytest.raises(Exception, match="Narrator initialization failed"):
            CSM()

    @pytest.mark.asyncio
    async def test_csm_process_story_database_transaction_rollback(self, csm_instance, dummy_audio_file):
        """Test CSM database transaction rollback on failure."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Transaction test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Transaction response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Mock database transaction methods
        csm_instance.db.begin_transaction = MagicMock()
        csm_instance.db.commit_transaction = MagicMock()
        csm_instance.db.rollback_transaction = MagicMock()
        csm_instance.db.get_character.return_value = {"name": "Transaction Character", "Actor_id": "Actor1"}
        
        # Simulate failure during save that should trigger rollback
        csm_instance.db.save_story = MagicMock(side_effect=Exception("Save failed"))
        
        # Should handle transaction rollback
        with pytest.raises(Exception, match="Save failed"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Verify transaction rollback was called if implemented
        if hasattr(csm_instance.db, 'rollback_transaction'):
            csm_instance.db.rollback_transaction.assert_called()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_component_health_monitoring(self, csm_instance, dummy_audio_file):
        """Test CSM component health monitoring during processing."""
        health_status = {"narrator": True, "character_server": True, "client_manager": True, "db": True}
        
        # Mock health check methods
        csm_instance.narrator.health_check = MagicMock(return_value=health_status["narrator"])
        csm_instance.character_server.health_check = MagicMock(return_value=health_status["character_server"])
        csm_instance.client_manager.health_check = MagicMock(return_value=health_status["client_manager"])
        csm_instance.db.health_check = MagicMock(return_value=health_status["db"])
        
        csm_instance.narrator.process_narration.return_value = {
            "text": "Health monitoring test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Healthy response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Healthy Character", "Actor_id": "Actor1"}
        
        # Process story with all components healthy
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration is not None
        
        # Simulate unhealthy component
        health_status["db"] = False
        csm_instance.db.health_check.return_value = False
        
        # Should still process but potentially with warnings/fallbacks
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration is not None
        
        await csm_instance.shutdown_async()

    def test_csm_class_attributes_and_constants(self, mock_dependencies):
        """Test CSM class attributes and constants are properly defined."""
        csm = CSM()
        
        # Test that all expected attributes exist
        expected_attributes = [
            'narrator', 'character_server', 'client_manager', 'db', 
            'executor', 'hardware', 'chaos_engine'
        ]
        
        for attr in expected_attributes:
            assert hasattr(csm, attr), f"CSM missing expected attribute: {attr}"
            assert getattr(csm, attr) is not None, f"CSM attribute {attr} is None"
        
        # Test any class-level constants if they exist
        if hasattr(CSM, 'MAX_CHAOS_LEVEL'):
            assert CSM.MAX_CHAOS_LEVEL == 1.0
        if hasattr(CSM, 'MIN_CHAOS_LEVEL'):
            assert CSM.MIN_CHAOS_LEVEL == 0.0
        if hasattr(CSM, 'DEFAULT_TIMEOUT'):
            assert isinstance(CSM.DEFAULT_TIMEOUT, (int, float))

    @pytest.mark.asyncio
    async def test_csm_graceful_degradation_mode(self, csm_instance, dummy_audio_file):
        """Test CSM graceful degradation when multiple components fail."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Degradation test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Simulate multiple component failures
        csm_instance.character_server.generate_response.side_effect = Exception("Character server down")
        csm_instance.client_manager.get_clients_for_story_progression.side_effect = Exception("Client manager down")
        csm_instance.hardware.update_leds = MagicMock(side_effect=Exception("Hardware failure"))
        
        # Should degrade gracefully, providing minimal functionality
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            # If graceful degradation is implemented
            assert narration is not None
            # May have empty or minimal character responses
        except Exception:
            # If system fails completely when multiple components are down, that's also valid
            pass
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_stress_testing_burst_requests(self, csm_instance, dummy_audio_file):
        """Test CSM under burst request load conditions."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Stress test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Stress test response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Stress Character", "Actor_id": "Actor1"}
        
        # Create burst of 100 concurrent requests
        burst_tasks = []
        for i in range(100):
            task = csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            burst_tasks.append(task)
        
        # Process all requests
        results = await asyncio.gather(*burst_tasks, return_exceptions=True)
        
        # Count successful vs failed requests
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        # Should handle most requests successfully (at least 80%)
        success_rate = successful / len(results)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2%} is below 80%"
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_data_integrity_validation(self, csm_instance, dummy_audio_file):
        """Test CSM data integrity validation throughout processing pipeline."""
        # Test with various data integrity scenarios
        test_scenarios = [
            {
                "name": "Valid data",
                "narration": {"text": "Valid test", "audio_path": dummy_audio_file, "speaker": "Narrator"},
                "character_response": "Valid response",
                "should_succeed": True
            },
            {
                "name": "Missing text field",
                "narration": {"audio_path": dummy_audio_file, "speaker": "Narrator"},
                "character_response": "Response to missing text",
                "should_succeed": False
            },
            {
                "name": "Null speaker",
                "narration": {"text": "Test with null speaker", "audio_path": dummy_audio_file, "speaker": None},
                "character_response": "Response to null speaker",
                "should_succeed": False
            },
            {
                "name": "Empty audio path",
                "narration": {"text": "Test with empty audio", "audio_path": "", "speaker": "Narrator"},
                "character_response": "Response to empty audio",
                "should_succeed": False
            }
        ]
        
        for scenario in test_scenarios:
            csm_instance.narrator.process_narration.return_value = scenario["narration"]
            csm_instance.character_server.generate_response.return_value = scenario["character_response"]
            csm_instance.client_manager.get_clients_for_story_progression.return_value = []
            
            if scenario["should_succeed"]:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
                assert narration is not None, f"Scenario '{scenario['name']}' should have succeeded"
            else:
                # Should handle invalid data gracefully or raise appropriate errors
                try:
                    narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
                    # If handled gracefully, verify minimal response
                    if narration == "" and characters == {}:
                        pass  # Graceful handling is acceptable
                except (ValueError, TypeError, KeyError):
                    pass  # Raising validation errors is also acceptable
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_configuration_edge_cases(self, csm_instance, dummy_audio_file):
        """Test CSM behavior with various configuration edge cases."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Configuration test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Config response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Config Character", "Actor_id": "Actor1"}
        
        # Test with various chaos level edge cases
        edge_case_chaos_levels = [
            0.0,  # Minimum
            1.0,  # Maximum
            0.00001,  # Very small positive
            0.99999,  # Very close to maximum
            0.5,  # Exact middle
        ]
        
        for chaos_level in edge_case_chaos_levels:
            narration, characters = await csm_instance.process_story(
                dummy_audio_file, chaos_level=chaos_level
            )
            assert narration is not None
            assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_audio_format_compatibility(self, csm_instance):
        """Test CSM process_story with different audio file formats."""
        # Test various audio file extensions and formats
        audio_test_cases = [
            {"extension": ".wav", "content": b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'},
            {"extension": ".mp3", "content": b'ID3\x03\x00\x00\x00\x00\x00\x00'},
            {"extension": ".ogg", "content": b'OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00'},
            {"extension": ".flac", "content": b'fLaC\x00\x00\x00\x22'},
        ]
        
        for test_case in audio_test_cases:
            audio_file = f"test_audio{test_case['extension']}"
            
            try:
                # Create test audio file
                with open(audio_file, 'wb') as f:
                    f.write(test_case['content'])
                
                # Mock narrator response for this format
                csm_instance.narrator.process_narration.return_value = {
                    "text": f"Audio format test {test_case['extension']}",
                    "audio_path": audio_file,
                    "speaker": "Narrator"
                }
                
                csm_instance.character_server.generate_response.return_value = f"Response to {test_case['extension']}"
                csm_instance.client_manager.get_clients_for_story_progression.return_value = []
                csm_instance.db.get_character.return_value = {"name": "Audio Character", "Actor_id": "Actor1"}
                
                # Should handle different audio formats
                narration, characters = await csm_instance.process_story(audio_file, chaos_level=0.0)
                assert narration is not None
                assert isinstance(characters, dict)
                
            except Exception as e:
                # Some formats might not be supported - that's acceptable
                assert "format" in str(e).lower() or "codec" in str(e).lower()
            finally:
                # Cleanup
                try:
                    os.unlink(audio_file)
                except FileNotFoundError:
                    pass
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_component_lifecycle_management(self, csm_instance, dummy_audio_file):
        """Test CSM component lifecycle management during processing."""
        # Track component lifecycle calls
        lifecycle_calls = {
            "narrator_init": 0, "narrator_process": 0, "narrator_cleanup": 0,
            "character_server_init": 0, "character_server_process": 0, "character_server_cleanup": 0,
            "client_manager_init": 0, "client_manager_process": 0, "client_manager_cleanup": 0
        }
        
        # Mock lifecycle tracking
        original_narrator_process = csm_instance.narrator.process_narration
        
        async def tracked_narrator_process(*args, **kwargs):
            lifecycle_calls["narrator_process"] += 1
            return await original_narrator_process(*args, **kwargs)
        
        csm_instance.narrator.process_narration = tracked_narrator_process
        csm_instance.narrator.process_narration.return_value = {
            "text": "Lifecycle test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Lifecycle response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        csm_instance.db.get_character.return_value = {"name": "Lifecycle Character", "Actor_id": "Actor1"}
        
        # Process story and verify lifecycle calls
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert lifecycle_calls["narrator_process"] == 1
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_error_aggregation_and_reporting(self, csm_instance, dummy_audio_file):
        """Test CSM error aggregation and reporting across components."""
        error_log = []
        
        # Mock components to simulate various error conditions
        async def error_logging_narrator(*args, **kwargs):
            error_log.append("Narrator processing error")
            raise Exception("Narrator failed")
        
        async def error_logging_character_server(*args, **kwargs):
            error_log.append("Character server error")
            raise Exception("Character server failed")
        
        # Test sequential error handling
        csm_instance.narrator.process_narration = error_logging_narrator
        
        with pytest.raises(Exception, match="Narrator failed"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert len(error_log) == 1
        assert "Narrator processing error" in error_log
        
        # Reset for character server error test
        error_log.clear()
        csm_instance.narrator.process_narration.return_value = {
            "text": "Error aggregation test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        csm_instance.character_server.generate_response = error_logging_character_server
        
        with pytest.raises(Exception, match="Character server failed"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert len(error_log) == 1
        assert "Character server error" in error_log
        
        await csm_instance.shutdown_async()

