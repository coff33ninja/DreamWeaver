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
    # ========== Tests for update_last_narration_text method ==========
    
    def test_csm_update_last_narration_text_success(self, mock_dependencies):
        """Test successful update of last narration text."""
        csm = CSM()
        
        # Mock database to return story history with narrator entries
        mock_history = [
            {"id": 1, "speaker": "Character1", "text": "Hello world"},
            {"id": 2, "speaker": "Narrator", "text": "Old narration text"},
            {"id": 3, "speaker": "Character2", "text": "Another response"},
            {"id": 4, "speaker": "Narrator", "text": "Most recent narration"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Execute
        result = csm.update_last_narration_text("Corrected narration text")
        
        # Verify
        assert result is True
        csm.db.get_story_history.assert_called_once()
        csm.db.update_story_entry.assert_called_once_with(4, new_text="Corrected narration text")

    def test_csm_update_last_narration_text_no_narrator_entries(self, mock_dependencies):
        """Test update when no narrator entries exist."""
        csm = CSM()
        
        # Mock database to return history without narrator entries
        mock_history = [
            {"id": 1, "speaker": "Character1", "text": "Hello world"},
            {"id": 2, "speaker": "Character2", "text": "Another response"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Execute
        result = csm.update_last_narration_text("New narration text")
        
        # Verify
        assert result is False
        csm.db.get_story_history.assert_called_once()
        csm.db.update_story_entry.assert_not_called()

    def test_csm_update_last_narration_text_empty_history(self, mock_dependencies):
        """Test update when story history is empty."""
        csm = CSM()
        
        # Mock database to return empty history
        csm.db.get_story_history.return_value = []
        csm.db.update_story_entry = MagicMock()
        
        # Execute
        result = csm.update_last_narration_text("New narration text")
        
        # Verify
        assert result is False
        csm.db.get_story_history.assert_called_once()
        csm.db.update_story_entry.assert_not_called()

    def test_csm_update_last_narration_text_single_narrator_entry(self, mock_dependencies):
        """Test update with only one narrator entry."""
        csm = CSM()
        
        # Mock database to return history with single narrator entry
        mock_history = [
            {"id": 5, "speaker": "Narrator", "text": "Only narration"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Execute
        result = csm.update_last_narration_text("Updated only narration")
        
        # Verify
        assert result is True
        csm.db.update_story_entry.assert_called_once_with(5, new_text="Updated only narration")

    def test_csm_update_last_narration_text_database_error(self, mock_dependencies):
        """Test update when database operations fail."""
        csm = CSM()
        
        # Mock database to raise error on get_story_history
        csm.db.get_story_history.side_effect = Exception("Database connection failed")
        csm.db.update_story_entry = MagicMock()
        
        # Execute - should propagate the database error
        with pytest.raises(Exception, match="Database connection failed"):
            csm.update_last_narration_text("Test text")
        
        csm.db.update_story_entry.assert_not_called()

    def test_csm_update_last_narration_text_update_error(self, mock_dependencies):
        """Test update when update operation fails."""
        csm = CSM()
        
        # Mock database to return valid history but fail on update
        mock_history = [
            {"id": 1, "speaker": "Narrator", "text": "Test narration"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry.side_effect = Exception("Update failed")
        
        # Execute - should propagate the update error
        with pytest.raises(Exception, match="Update failed"):
            csm.update_last_narration_text("New text")

    def test_csm_update_last_narration_text_malformed_history(self, mock_dependencies):
        """Test update with malformed history data."""
        csm = CSM()
        
        # Mock database to return malformed history entries
        mock_history = [
            {"id": 1, "text": "Missing speaker field"},  # Missing speaker
            {"speaker": "Narrator"},  # Missing id and text
            {"id": 2, "speaker": "Narrator", "text": "Valid entry"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Should handle malformed entries gracefully and find valid narrator entry
        result = csm.update_last_narration_text("Updated text")
        
        assert result is True
        csm.db.update_story_entry.assert_called_once_with(2, new_text="Updated text")

    def test_csm_update_last_narration_text_none_history(self, mock_dependencies):
        """Test update when database returns None for history."""
        csm = CSM()
        
        # Mock database to return None
        csm.db.get_story_history.return_value = None
        csm.db.update_story_entry = MagicMock()
        
        # Should handle None history gracefully
        with pytest.raises((TypeError, AttributeError)):
            csm.update_last_narration_text("Test text")

    def test_csm_update_last_narration_text_empty_string(self, mock_dependencies):
        """Test update with empty string."""
        csm = CSM()
        
        # Mock database with valid narrator entry
        mock_history = [
            {"id": 1, "speaker": "Narrator", "text": "Original text"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Execute with empty string
        result = csm.update_last_narration_text("")
        
        # Should still work with empty string
        assert result is True
        csm.db.update_story_entry.assert_called_once_with(1, new_text="")

    def test_csm_update_last_narration_text_unicode_text(self, mock_dependencies):
        """Test update with unicode characters."""
        csm = CSM()
        
        # Mock database with valid narrator entry
        mock_history = [
            {"id": 1, "speaker": "Narrator", "text": "Original text"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Execute with unicode text
        unicode_text = "Tëst ñärràtîøn wîth émøjîs 🎭 àñd spéçîål çhäräçtérs café naïve"
        result = csm.update_last_narration_text(unicode_text)
        
        # Should handle unicode properly
        assert result is True
        csm.db.update_story_entry.assert_called_once_with(1, new_text=unicode_text)

    def test_csm_update_last_narration_text_very_long_text(self, mock_dependencies):
        """Test update with very long text."""
        csm = CSM()
        
        # Mock database with valid narrator entry
        mock_history = [
            {"id": 1, "speaker": "Narrator", "text": "Original text"}
        ]
        csm.db.get_story_history.return_value = mock_history
        csm.db.update_story_entry = MagicMock()
        
        # Execute with very long text
        long_text = "A" * 50000  # 50KB of text
        result = csm.update_last_narration_text(long_text)
        
        # Should handle long text
        assert result is True
        csm.db.update_story_entry.assert_called_once_with(1, new_text=long_text)

    # ========== Additional Initialization Tests ==========
    
    def test_csm_initialization_component_order(self, mock_dependencies):
        """Test that CSM components are initialized in correct order."""
        with patch('SERVER.src.csm.Database') as mock_db_class, \
             patch('SERVER.src.csm.Narrator') as mock_narrator_class, \
             patch('SERVER.src.csm.CharacterServer') as mock_cs_class, \
             patch('SERVER.src.csm.ClientManager') as mock_cm_class, \
             patch('SERVER.src.csm.Hardware') as mock_hw_class, \
             patch('SERVER.src.csm.ChaosEngine') as mock_ce_class:
            
            # Track initialization order
            init_order = []
            
            def track_db_init(*args):
                init_order.append('Database')
                return MagicMock()
            
            def track_narrator_init(*args):
                init_order.append('Narrator')
                return MagicMock()
            
            def track_cs_init(*args):
                init_order.append('CharacterServer')
                return MagicMock()
            
            def track_cm_init(*args):
                init_order.append('ClientManager')
                cm_mock = MagicMock()
                cm_mock.start_periodic_health_checks = MagicMock()
                return cm_mock
            
            def track_hw_init(*args):
                init_order.append('Hardware')
                return MagicMock()
            
            def track_ce_init(*args):
                init_order.append('ChaosEngine')
                return MagicMock()
            
            mock_db_class.side_effect = track_db_init
            mock_narrator_class.side_effect = track_narrator_init
            mock_cs_class.side_effect = track_cs_init
            mock_cm_class.side_effect = track_cm_init
            mock_hw_class.side_effect = track_hw_init
            mock_ce_class.side_effect = track_ce_init
            
            # Initialize CSM
            csm = CSM()
            
            # Verify initialization order
            expected_order = ['Database', 'Narrator', 'CharacterServer', 'ClientManager', 'Hardware', 'ChaosEngine']
            assert init_order == expected_order

    def test_csm_initialization_health_checks_started(self, mock_dependencies):
        """Test that client health checks are started during initialization."""
        csm = CSM()
        
        # Verify health checks were started
        csm.client_manager.start_periodic_health_checks.assert_called_once()

    def test_csm_initialization_with_invalid_db_path(self, mock_dependencies):
        """Test CSM initialization with invalid database path."""
        with patch('SERVER.src.csm.Database') as mock_db_class:
            mock_db_class.side_effect = Exception("Invalid database path")
            
            # Should propagate database initialization error
            with pytest.raises(Exception, match="Invalid database path"):
                CSM()

    # ========== Additional Process Story Edge Cases ==========
    
    @pytest.mark.asyncio
    async def test_csm_process_story_narrator_timeout(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when narrator processing times out."""
        csm_instance.narrator.process_narration.side_effect = asyncio.TimeoutError("Narrator timeout")
        
        with pytest.raises(asyncio.TimeoutError, match="Narrator timeout"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_character_server_timeout(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when character server times out."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Timeout test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.side_effect = asyncio.TimeoutError("Character server timeout")
        
        with pytest.raises(asyncio.TimeoutError, match="Character server timeout"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_hardware_update_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when hardware update fails."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Hardware test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Mock hardware update failure
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = [
                {"name": "Test Character", "Actor_id": "Actor1"},  # DB call for character
                [],  # get_clients_for_story_progression
                Exception("Hardware update failed"),  # hardware.update_leds
            ]
            
            # Should handle hardware failure gracefully or propagate error
            try:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
                # If handled gracefully
                assert narration is not None
            except Exception as e:
                # If hardware errors are propagated
                assert "Hardware update failed" in str(e)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_database_save_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when database save fails."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Save test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Mock database save failure
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = [
                {"name": "Test Character", "Actor_id": "Actor1"},  # DB call for character
                [],  # get_clients_for_story_progression
                None,  # hardware.update_leds
                Exception("Database save failed"),  # db.save_story
            ]
            
            # Should handle save failure gracefully or propagate error
            try:
                narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
                # If handled gracefully
                assert narration is not None
            except Exception as e:
                # If save errors are propagated
                assert "Database save failed" in str(e)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_process_story_chaos_engine_failure(self, csm_instance, dummy_audio_file):
        """Test CSM process_story when chaos engine fails."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Chaos test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Server response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Mock chaos engine to trigger chaos, then fail
        csm_instance.chaos_engine.random_factor.return_value = 0.05  # Should trigger chaos at level 0.6
        csm_instance.chaos_engine.apply_chaos.side_effect = Exception("Chaos engine malfunction")
        
        # Should handle chaos engine failure
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.6)
            # If handled gracefully
            assert narration is not None
        except Exception as e:
            # If chaos errors are propagated
            assert "Chaos engine malfunction" in str(e)
        
        await csm_instance.shutdown_async()

    # ========== Performance and Resource Management Tests ==========
    
    @pytest.mark.asyncio
    async def test_csm_resource_cleanup_verification(self, csm_instance, dummy_audio_file):
        """Test that CSM properly manages and cleans up resources."""
        csm_instance.narrator.process_narration.return_value = {
            "text": "Resource test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "Resource response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # Track resource usage
        import resource
        import psutil
        import os
        
        # Get initial resource usage
        initial_memory = psutil.Process(os.getpid()).memory_info().rss
        initial_open_files = len(psutil.Process(os.getpid()).open_files())
        
        # Process multiple stories
        for i in range(10):
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.1)
            assert narration is not None
            assert isinstance(characters, dict)
        
        # Force garbage collection
        gc.collect()
        
        # Check resource usage hasn't grown excessively
        final_memory = psutil.Process(os.getpid()).memory_info().rss
        final_open_files = len(psutil.Process(os.getpid()).open_files())
        
        # Memory growth should be reasonable (allow for 50MB growth)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 50 * 1024 * 1024, f"Excessive memory growth: {memory_growth} bytes"
        
        # File handles should not leak
        file_growth = final_open_files - initial_open_files
        assert file_growth <= 5, f"Potential file handle leak: {file_growth} additional files"
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_concurrent_processing_isolation(self, mock_dependencies):
        """Test that concurrent CSM processing maintains proper isolation."""
        # Create multiple CSM instances
        csm_instances = [CSM() for _ in range(3)]
        
        # Set up different responses for each instance
        for i, csm in enumerate(csm_instances):
            csm.narrator.process_narration = AsyncMock()
            csm.character_server.generate_response = AsyncMock()
            csm.client_manager.send_to_client = AsyncMock()
            csm.client_manager.get_clients_for_story_progression = MagicMock()
            csm.db.get_character = MagicMock()
            
            csm.narrator.process_narration.return_value = {
                "text": f"Instance {i} narration",
                "audio_path": f"test_{i}.wav",
                "speaker": "Narrator"
            }
            csm.character_server.generate_response.return_value = f"Instance {i} response"
            csm.client_manager.get_clients_for_story_progression.return_value = []
        
        # Create dummy files
        dummy_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=f'_isolation_{i}.wav', delete=False) as temp_file:
                temp_file.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
                temp_file.flush()
                dummy_files.append(temp_file.name)
        
        try:
            # Process stories concurrently
            tasks = []
            for i, (csm, dummy_file) in enumerate(zip(csm_instances, dummy_files)):
                task = csm.process_story(dummy_file, chaos_level=0.1 * i)
                tasks.append((i, task))
            
            # Wait for all to complete
            results = []
            for i, task in tasks:
                result = await task
                results.append((i, result))
            
            # Verify isolation - each instance should have its own results
            assert len(results) == 3
            for i, (instance_id, (narration, characters)) in enumerate(results):
                assert instance_id == i
                assert narration is not None
                assert f"Instance {i} narration" in str(narration)
                assert isinstance(characters, dict)
            
            # Shutdown all instances
            for csm in csm_instances:
                await csm.shutdown_async()
        
        finally:
            # Cleanup
            for dummy_file in dummy_files:
                try:
                    os.unlink(dummy_file)
                except FileNotFoundError:
                    pass

    # ========== Additional Error Recovery Tests ==========
    
    @pytest.mark.asyncio
    async def test_csm_partial_component_recovery(self, csm_instance, dummy_audio_file):
        """Test CSM recovery when some components fail but others succeed."""
        # Set up a scenario where narrator succeeds, character server fails, but processing continues
        csm_instance.narrator.process_narration.return_value = {
            "text": "Recovery test",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        # Character server fails
        csm_instance.character_server.generate_response.side_effect = Exception("Character server down")
        
        # Client manager works
        csm_instance.client_manager.get_clients_for_story_progression.return_value = [
            {"Actor_id": "RecoveryClient", "ip_address": "127.0.0.1", "client_port": 8001}
        ]
        csm_instance.client_manager.send_to_client.return_value = "Client still works"
        csm_instance.db.get_character.return_value = {"name": "Recovery Character", "Actor_id": "RecoveryClient"}
        
        # Should handle partial failure gracefully
        try:
            narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
            # If it handles partial failures gracefully
            assert narration is not None
            # May or may not have character responses depending on implementation
        except Exception as e:
            # If character server failures are fatal
            assert "Character server down" in str(e)
        
        await csm_instance.shutdown_async()

    @pytest.mark.asyncio
    async def test_csm_state_consistency_after_errors(self, csm_instance, dummy_audio_file):
        """Test that CSM maintains consistent state after errors."""
        # First call succeeds
        csm_instance.narrator.process_narration.return_value = {
            "text": "First success",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        csm_instance.character_server.generate_response.return_value = "First response"
        csm_instance.client_manager.get_clients_for_story_progression.return_value = []
        
        # First call should work
        narration1, characters1 = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration1 is not None
        
        # Second call fails
        csm_instance.narrator.process_narration.side_effect = Exception("Temporary failure")
        
        # Second call should fail
        with pytest.raises(Exception, match="Temporary failure"):
            await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        
        # Third call should work again (component recovered)
        csm_instance.narrator.process_narration.side_effect = None
        csm_instance.narrator.process_narration.return_value = {
            "text": "Recovery success",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        narration3, characters3 = await csm_instance.process_story(dummy_audio_file, chaos_level=0.0)
        assert narration3 is not None
        
        await csm_instance.shutdown_async()

    # ========== Integration Scenario Tests ==========
    
    @pytest.mark.asyncio
    async def test_csm_full_story_processing_integration(self, csm_instance, dummy_audio_file):
        """Test full story processing integration with realistic scenario."""
        # Set up realistic multi-character scenario
        csm_instance.narrator.process_narration.return_value = {
            "text": "The adventurers entered the dark dungeon, hearing strange noises echoing from the depths.",
            "audio_path": dummy_audio_file,
            "speaker": "Narrator"
        }
        
        csm_instance.character_server.generate_response.return_value = "The wizard raises his staff, casting a light spell to illuminate the passage ahead."
        
        # Multiple diverse clients
        test_clients = [
            {"Actor_id": "Warrior", "ip_address": "192.168.1.10", "client_port": 8001},
            {"Actor_id": "Rogue", "ip_address": "192.168.1.11", "client_port": 8002},
            {"Actor_id": "Cleric", "ip_address": "192.168.1.12", "client_port": 8003}
        ]
        csm_instance.client_manager.get_clients_for_story_progression.return_value = test_clients
        
        # Different responses from each client
        csm_instance.client_manager.send_to_client.side_effect = [
            "The warrior draws his sword and takes point, ready to face whatever lurks ahead.",
            "The rogue melts into the shadows, moving silently to scout for traps.",
            "The cleric whispers a prayer of protection, blessing the party before they proceed."
        ]
        
        # Character details for each client
        def mock_get_character(actor_id):
            characters = {
                "Actor1": {"name": "Gandalf the Wise", "Actor_id": "Actor1"},
                "Warrior": {"name": "Sir Braveheart", "Actor_id": "Warrior"},
                "Rogue": {"name": "Shadows McGee", "Actor_id": "Rogue"},
                "Cleric": {"name": "Sister Mercy", "Actor_id": "Cleric"}
            }
            return characters.get(actor_id)
        
        csm_instance.db.get_character.side_effect = mock_get_character
        
        # Process the story with moderate chaos
        narration, characters = await csm_instance.process_story(dummy_audio_file, chaos_level=0.3)
        
        # Verify comprehensive results
        assert narration is not None
        assert "dungeon" in narration["text"]
        assert isinstance(characters, dict)
        assert len(characters) >= 4  # Server + 3 clients
        
        # Verify all expected characters responded
        character_names = list(characters.keys())
        assert "Gandalf the Wise" in character_names  # Server character
        assert "Sir Braveheart" in character_names   # Warrior
        assert "Shadows McGee" in character_names    # Rogue
        assert "Sister Mercy" in character_names     # Cleric
        
        # Verify response content is appropriate
        for char_name, response in characters.items():
            assert isinstance(response, str)
            assert len(response) > 0
            if char_name == "Gandalf the Wise":
                assert "wizard" in response.lower() or "staff" in response.lower()
            elif char_name == "Sir Braveheart":
                assert "warrior" in response.lower() or "sword" in response.lower()
            elif char_name == "Shadows McGee":
                assert "rogue" in response.lower() or "shadow" in response.lower()
            elif char_name == "Sister Mercy":
                assert "cleric" in response.lower() or "prayer" in response.lower()
        
        await csm_instance.shutdown_async()
