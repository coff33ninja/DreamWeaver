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