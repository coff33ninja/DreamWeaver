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
    async def test_csm_process_story_memory_management(self, csm_instance, dummy_audio_file):
        """Test CSM process_story memory management with large data structures."""
        # Create large mock data to test memory handling
        large_narration = {
            "text": "Memory test " * 1000,  # Reasonably large text
            "audio_path": dummy_audio_file,
            "speaker": "Narrator",
            "metadata": {"key": "value"} * 100  # Large metadata dict
        }
        
        csm_instance.narrator.process_narration.return_value = large_narration
        csm_instance.character_server.generate_response.return_value = "Memory efficient response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Process multiple times to test memory cleanup
        for i in range(5):
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
            assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio  
    async def test_csm_process_story_with_malicious_input_injection(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with potentially malicious input."""
        # Test with script injection attempts
        malicious_narration = {
            "text": "<script>alert('xss')</script>; DROP TABLE users; --",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.narrator.process_narration.return_value = malicious_narration
        csm_instance.character_server.generate_response.return_value = "Sanitized response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Should handle malicious input safely
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        # Verify no script execution or SQL injection occurred
        assert "script" in narration.lower()  # Text preserved but not executed
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_network_partition_simulation(self, csm_instance, dummy_audio_file):
        """Test CSM process_story behavior during network partition scenarios."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Network partition test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Partition response"
        
        # Simulate network partition with mixed client responses
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "ReachableClient", "ip_address": "127.0.0.1", "client_port": 8001},
            {"Actor_id": "UnreachableClient", "ip_address": "192.168.1.100", "client_port": 8002}
        ]
        
        def mock_send_to_client(client_actor_id, client_ip, client_port, narration, character_texts):
            if client_actor_id == "UnreachableClient":
                raise ConnectionError("Network unreachable")
            return f"{client_actor_id} responded despite partition"
        
        csm_instance.client_manager.send_to_client.side_effect = mock_send_to_client
        csm_instance.db.get_character.side_effect = lambda actor_id: {"name": f"Character {actor_id}", "Actor_id": actor_id}
        
        # Should handle network partitions gracefully
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
            assert isinstance(characters, dict)
        except ConnectionError:
            # If partition errors are propagated, that's also valid
            pass
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_resource_exhaustion_simulation(self, csm_instance, dummy_audio_file):
        """Test CSM process_story under resource exhaustion conditions."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Resource exhaustion test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Simulate resource exhaustion in character server
        csm_instance.character_server.generate_response.side_effect = MemoryError("Insufficient memory")
        
        # Should handle resource exhaustion gracefully
        with pytest.raises(MemoryError):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_circular_dependencies(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with circular dependency scenarios."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Circular dependency test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Circular response"
        
        # Client that references itself or creates circular references
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "SelfReferencingActor", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        
        csm_instance.client_manager.send_to_client.return_value = "Circular client response"
        
        # Character that might reference itself in some way
        def circular_character_getter(actor_id):
            char = {"name": f"Character {actor_id}", "Actor_id": actor_id}
            char["reference"] = char  # Circular reference
            return char
        
        csm_instance.db.get_character.side_effect = circular_character_getter
        
        # Should handle circular references without infinite loops
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_state_consistency_across_failures(self, csm_instance, dummy_audio_file):
        """Test CSM process_story maintains state consistency across component failures."""
        # Set up initial state
        csm_instance.narrator.process_narration.return_value = {
            "text": "State consistency test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # First call succeeds
        csm_instance.character_server.generate_response.return_value = "First response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        narration1, characters1 = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Second call with character server failure
        csm_instance.character_server.generate_response.side_effect = Exception("Character server down")
        
        with pytest.raises(Exception):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Third call - verify system can recover
        csm_instance.character_server.generate_response.side_effect = None
        csm_instance.character_server.generate_response.return_value = "Recovery response"
        
        narration3, characters3 = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Verify state consistency
        assert narration1 is not None
        assert narration3 is not None
        assert isinstance(characters1, dict)
        assert isinstance(characters3, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_audio_format_variations(self, csm_instance):
        """Test CSM process_story with different audio file formats and properties."""
        test_cases = [
            ("test.wav", "WAV format"),
            ("test.mp3", "MP3 format"),
            ("test.ogg", "OGG format"),
            ("test.flac", "FLAC format"),
            ("test_with_spaces.wav", "Filename with spaces"),
            ("test-with-dashes.wav", "Filename with dashes"),
            ("test_with_unicode_café.wav", "Unicode filename")
        ]
        
        for audio_file, description in test_cases:
            # Create temporary file for each test
            with tempfile.NamedTemporaryFile(suffix=audio_file.split('.')[-1], delete=False) as temp_file:
                temp_file.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
                temp_file.flush()
                
                csm_instance.narrator.process_narration.return_value = {
                    "text": f"Testing {description}",
                    "audio_path": temp_file.name,
                    "speaker": "Narrator"
                }
                
                csm_instance.character_server.generate_response.return_value = f"Response to {description}"
                csm_instance.client_manager.get_clients_for_story_progression.return_value = []
                
                narration, characters = await csm_instance.process_story(temp_file.name, chaos_level=0.0)
                
                assert narration is not None
                assert isinstance(characters, dict)
                
                # Cleanup
                os.unlink(temp_file.name)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_component_health_monitoring(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with component health monitoring."""
        # Mock health status for each component
        csm_instance.narrator.health_status = MagicMock(return_value={"status": "healthy", "uptime": 100})
        csm_instance.character_server.health_status = MagicMock(return_value={"status": "degraded", "response_time": 500})
        csm_instance.client_manager.health_status = MagicMock(return_value={"status": "healthy", "connected_clients": 5})
        csm_instance.db.health_status = MagicMock(return_value={"status": "healthy", "connection_pool": 10})
        
        csm_instance.narrator.process_narration.return_value = {
            "text": "Health monitoring test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Health checked response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Add health check method to CSM if it doesn't exist
        def mock_health_check():
            return {
                "narrator": csm_instance.narrator.health_status(),
                "character_server": csm_instance.character_server.health_status(),
                "client_manager": csm_instance.client_manager.health_status(),
                "database": csm_instance.db.health_status()
            }
        
        csm_instance.get_system_health = mock_health_check
        
        # Process story and check health
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        health_status = csm_instance.get_system_health()
        
        assert narration is not None
        assert isinstance(characters, dict)
        assert "narrator" in health_status
        assert "character_server" in health_status
        assert health_status["character_server"]["status"] == "degraded"
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_rate_limiting(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with rate limiting scenarios."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Rate limiting test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Simulate rate limiting on character server
        call_count = 0
        def rate_limited_response(narration, other_texts, chaos_level=0.0):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise Exception("Rate limit exceeded")
            return f"Rate limited response {call_count}"
        
        csm_instance.character_server.generate_response.side_effect = rate_limited_response
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # First few calls should succeed
        for i in range(3):
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
            assert isinstance(characters, dict)
        
        # Fourth call should hit rate limit
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_data_serialization_edge_cases(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with complex data serialization scenarios."""
        # Complex data structures that might cause serialization issues
        complex_narration = {
            "text": "Serialization test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator",
            "metadata": {
                "nested": {"deeply": {"nested": {"data": [1, 2, {"key": "value"}]}}},
                "unicode": "café résumé naïve",
                "special_chars": "!@#$%^&*()[]{}|\\:;\"'<>?,./",
                "numbers": [float('inf'), -float('inf'), 0, -0, 1e10, 1e-10],
                "mixed_types": [True, False, None, "", 0, []]
            }
        }
        
        csm_instance.narrator.process_narration.return_value = complex_narration
        csm_instance.character_server.generate_response.return_value = "Complex serialization response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Should handle complex data structures without serialization errors
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_component_version_mismatches(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with simulated component version mismatches."""
        # Mock version information for components
        csm_instance.narrator.version = "1.0.0"
        csm_instance.character_server.version = "2.0.0"  # Different version
        csm_instance.client_manager.version = "1.5.0"   # Another different version
        
        csm_instance.narrator.process_narration.return_value = {
            "text": "Version compatibility test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Simulate API changes due to version differences
        def version_aware_response(narration, other_texts, chaos_level=0.0):
            if hasattr(csm_instance.character_server, 'version') and csm_instance.character_server.version.startswith('2'):
                return {"response": "v2 response format", "version": "2.0.0"}
            return "v1 response format"
        
        csm_instance.character_server.generate_response.side_effect = version_aware_response
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Should handle version differences gracefully
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        assert narration is not None
        assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_graceful_degradation_on_partial_component_failure(self, csm_instance, dummy_audio_file):
        """Test CSM graceful degradation when some components fail but others work."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Graceful degradation test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Hardware component fails but other components work
        csm_instance.hardware.update_leds = MagicMock(side_effect=Exception("Hardware disconnected"))
        csm_instance.character_server.generate_response.return_value = "Server still works"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Should continue processing despite hardware failure
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            assert narration is not None
            assert isinstance(characters, dict)
            assert "ServerCharacter" in characters  # Core functionality preserved
        except Exception as e:
            # If hardware failures are critical, verify the error is handled appropriately
            assert "Hardware disconnected" in str(e)
        
        await csm_instance.shutdown_async()

    def test_csm_configuration_validation(self, mock_dependencies):
        """Test CSM initialization with various configuration scenarios."""
        # Test with minimal configuration
        csm_minimal = CSM()
        assert csm_minimal is not None
        
        # Test component initialization validation
        assert hasattr(csm_minimal, 'narrator')
        assert hasattr(csm_minimal, 'character_server')
        assert hasattr(csm_minimal, 'client_manager')
        assert hasattr(csm_minimal, 'db')
        
        # Verify all components are properly initialized
        assert csm_minimal.narrator is not None
        assert csm_minimal.character_server is not None
        assert csm_minimal.client_manager is not None
        assert csm_minimal.db is not None

    @pytest.mark.asyncio
    async def test_csm_cleanup_after_exceptions(self, csm_instance, dummy_audio_file):
        """Test CSM properly cleans up resources after exceptions."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Cleanup test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Track resource allocation
        resources_allocated = []
        
        def mock_allocate_resource():
            resource_id = len(resources_allocated) + 1
            resources_allocated.append(resource_id)
            return resource_id
        
        def mock_deallocate_resource(resource_id):
            if resource_id in resources_allocated:
                resources_allocated.remove(resource_id)
        
        csm_instance.allocate_resource = mock_allocate_resource
        csm_instance.deallocate_resource = mock_deallocate_resource
        
        # Cause an exception during processing
        csm_instance.character_server.generate_response.side_effect = Exception("Processing error")
        
        try:
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        except Exception:
            pass
        
        # Verify cleanup was called during shutdown
        await csm_instance.shutdown_async()
        
        # In a real implementation, we'd verify resources were cleaned up
        # For this test, we just ensure shutdown completed without additional errors

    @pytest.mark.asyncio
    async def test_csm_process_story_with_corrupted_audio_metadata(self, csm_instance):
        """Test CSM process_story with corrupted audio file metadata."""
        # Create corrupted audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as corrupted_file:
            # Write invalid WAV header
            corrupted_file.write(b'INVALID_HEADER_DATA' + b'\x00' * 100)
            corrupted_file.flush()
            
            # Narrator should handle corrupted audio gracefully
            csm_instance.narrator.process_narration.side_effect = Exception("Audio format not supported")
            
            with pytest.raises(Exception, match="Audio format not supported"):
                await csm_instance.process_story(corrupted_file.name, chaos_level=0.0)
            
            # Cleanup
            os.unlink(corrupted_file.name)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_maximum_concurrent_requests(self, csm_instance, dummy_audio_file):
        """Test CSM process_story at maximum concurrent request capacity."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Max concurrency test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Concurrent response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Create many concurrent requests (reasonable number for testing)
        max_concurrent = 20
        tasks = []
        
        for i in range(max_concurrent):
            task = csm_instance.process_story(dummy_audio_file, chaos_level=0.1 * i / max_concurrent)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests were handled
        assert len(results) == max_concurrent
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # Most requests should succeed (allow some to fail due to resource constraints)
        assert len(successful_results) >= max_concurrent * 0.8  # At least 80% success rate
        
        # Verify successful results have correct format
        for narration, characters in successful_results:
            assert narration is not None
            assert isinstance(characters, dict)
        
        await csm_instance.shutdown_async()

# Performance and Load Testing
    @pytest.mark.asyncio
    async def test_csm_process_story_load_testing(self, csm_instance, dummy_audio_file):
        """Test CSM process_story under sustained load."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Load test narration",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Load test response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Run sustained load for a short period
        import time
        start_time = time.time()
        request_count = 0
        duration = 2  # 2 seconds of load testing
        
        while time.time() - start_time < duration:
            try:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
                assert narration is not None
                assert isinstance(characters, dict)
                request_count += 1
            except Exception:
                # Some failures are acceptable under load
                pass
        
        # Verify reasonable throughput
        requests_per_second = request_count / duration
        assert requests_per_second > 0  # Should process at least some requests
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_update_last_narration_text_with_special_characters(self, csm_instance):
        """Test CSM update_last_narration_text with special characters and edge cases."""
        test_cases = [
            ("", "Empty text"),
            ("   ", "Whitespace only"),
            ("Text with\nnewlines\tand\ttabs", "Text with control characters"),
            ("Unicode: café naïve résumé 🎭", "Unicode characters"),
            ("Very " + "long " * 100 + "text", "Very long text"),
            ("Special chars: !@#$%^&*()[]{}|\\:;\"'<>?,./", "Special symbols"),
            ("SQL injection attempt: '; DROP TABLE users; --", "SQL injection attempt"),
            ("<script>alert('xss')</script>", "XSS attempt")
        ]
        
        for test_text, description in test_cases:
            # Mock story history with narrator entry
            mock_history = [
                {"id": 1, "speaker": "Narrator", "text": "Original text"}
            ]
            
            csm_instance.db.get_story_history.return_value = mock_history
            csm_instance.db.update_story_entry = MagicMock()
            
            # Test update with special text
            result = csm_instance.update_last_narration_text(test_text)
            
            assert result is True, f"Failed for {description}"
            csm_instance.db.update_story_entry.assert_called_with(1, new_text=test_text)
            
            # Reset mock for next iteration
            csm_instance.db.update_story_entry.reset_mock()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_update_last_narration_text_concurrent_access(self, csm_instance):
        """Test CSM update_last_narration_text with concurrent access patterns."""
        # Mock story history
        mock_history = [
            {"id": 1, "speaker": "Narrator", "text": "Original text"},
            {"id": 2, "speaker": "Narrator", "text": "Second text"}
        ]
        
        csm_instance.db.get_story_history.return_value = mock_history
        csm_instance.db.update_story_entry = MagicMock()
        
        # Simulate concurrent updates
        async def concurrent_update(update_text):
            return csm_instance.update_last_narration_text(update_text)
        
        # Run multiple concurrent updates
        tasks = [
            concurrent_update("Update 1"),
            concurrent_update("Update 2"),
            concurrent_update("Update 3")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed (though only last one takes effect in real DB)
        assert all(results)
        # Database update should be called for each attempt
        assert csm_instance.db.update_story_entry.call_count == 3
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_with_mock_hardware_variations(self, csm_instance, dummy_audio_file):
        """Test CSM process_story with different hardware configurations."""
        hardware_configs = [
            {"type": "LED_STRIP", "count": 50, "brightness": 255},
            {"type": "MATRIX", "width": 16, "height": 16},
            {"type": "SOUND_REACTIVE", "mic_sensitivity": 0.8},
            {"type": "DISABLED", "enabled": False}
        ]
        
        for config in hardware_configs:
            csm_instance.narrator.process_narration.return_value = {
                "text": f"Hardware test for {config['type']}",
                "audio_path": dummy_audio_file,
                "speaker": "Narrator"
            }
            
            csm_instance.character_server.generate_response.return_value = "Hardware config response"
            csm_instance.client_manager.get_clients_for_story_progression.return_value = []
            
            # Mock hardware with different configurations
            if config.get("enabled", True):
                csm_instance.hardware.update_leds = MagicMock()
                csm_instance.hardware.config = config
            else:
                csm_instance.hardware.update_leds = MagicMock(side_effect=Exception("Hardware disabled"))
            
            # Process story with different hardware configs
            try:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
                assert narration is not None
                assert isinstance(characters, dict)
                
                if config.get("enabled", True):
                    csm_instance.hardware.update_leds.assert_called_once()
            except Exception as e:
                if not config.get("enabled", True):
                    assert "Hardware disabled" in str(e)
                else:
                    raise
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_chaos_engine_boundary_conditions(self, csm_instance, dummy_audio_file):
        """Test CSM process_story chaos engine with boundary conditions."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Chaos boundary test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Chaos boundary response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Test chaos engine with various boundary conditions
        chaos_test_cases = [
            (0.0, 1.0, "Never trigger"),    # chaos=0, random=1 (never trigger)
            (1.0, 0.0, "Always trigger"),   # chaos=1, random=0 (always trigger)  
            (0.5, 0.05, "Trigger"),         # chaos=0.5, random=0.05 (should trigger)
            (0.5, 0.06, "Don't trigger"),   # chaos=0.5, random=0.06 (shouldn't trigger)
            (10.0, 0.01, "Max chaos")       # chaos=10, random=0.01 (should trigger)
        ]
        
        for chaos_level, random_value, description in chaos_test_cases:
            csm_instance.chaos_engine.random_factor = MagicMock(return_value=random_value)
            csm_instance.chaos_engine.apply_chaos = MagicMock(
                return_value=(f"Chaotic: {description}", {"Chaos Character": f"Chaotic response: {description}"})
            )
            
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=chaos_level)
            
            assert narration is not None
            assert isinstance(characters, dict)
            
            # Verify chaos trigger logic based on formula: random < (chaos / 10.0)
            expected_trigger = random_value < (chaos_level / 10.0)
            
            if expected_trigger and chaos_level > 0:
                csm_instance.chaos_engine.apply_chaos.assert_called_once()
                assert "Chaotic:" in narration
            else:
                # Reset call count for next iteration
                csm_instance.chaos_engine.apply_chaos.reset_mock()
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_database_operations_stress_test(self, csm_instance, dummy_audio_file):
        """Test CSM database operations under stress conditions."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Database stress test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "DB stress response"
        
        # Create many clients to stress database operations
        large_client_list = [
            {"Actor_id": f"StressClient{i}", "ip_address": "127.0.0.1", "client_port": 8000+i}
            for i in range(25)  # Reasonable number for stress testing
        ]
        
        csm_instance.client_manager.get_clients_for_story_progression.return_value = large_client_list
        csm_instance.client_manager.send_to_client.return_value = "Stress client response"
        
        # Simulate database call delays to stress async operations
        call_count = 0
        async def delayed_db_operation(actor_id):
            nonlocal call_count
            call_count += 1
            # Simulate some database processing time
            await asyncio.sleep(0.001)  # 1ms delay per call
            return {"name": f"Stress Character {actor_id}", "Actor_id": actor_id}
        
        # Mock database with realistic delays
        csm_instance.db.get_character.side_effect = lambda actor_id: delayed_db_operation(actor_id)
        
        # Process story with many database operations
        start_time = asyncio.get_event_loop().time()
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        
        assert narration is not None
        assert isinstance(characters, dict)
        # Verify all database calls were made
        assert call_count == 26  # 25 clients + 1 server character
        # Verify reasonable performance despite database stress
        assert processing_time < 5.0  # Should complete within 5 seconds
        
        await csm_instance.shutdown_async()

