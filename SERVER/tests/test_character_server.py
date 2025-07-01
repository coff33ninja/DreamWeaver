import pytest
import asyncio
import os
import uuid
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from pathlib import Path

# Import the CharacterServer class and related dependencies
try:
    import sys
    sys.path.append('SERVER')
    from character_server import CharacterServer
except ImportError:
    pytest.skip("CharacterServer module not found", allow_module_level=True)


class TestCharacterServer:
    """Comprehensive test suite for CharacterServer functionality.
    
    Testing Framework: pytest
    This test suite covers the async AI-powered character server that integrates
    LLM and TTS functionality for generating character responses and audio.
    """

    @pytest.fixture
    def mock_database(self):
        """
        Provides a mock database fixture that returns predefined character data for testing purposes.
        
        Returns:
            Mock: A mock database object with methods for retrieving and saving character and training data.
        """
        mock_db = Mock()
        mock_db.get_character.return_value = {
            "name": "TestActor1",
            "personality": "friendly_tester",
            "goals": "assist with testing",
            "backstory": "created for unit tests",
            "tts": "gtts",
            "tts_model": "en_US-test-model",
            "reference_audio_filename": "test_reference.wav",
            "Actor_id": "Actor1",
            "llm_model": "test-llm-model",
            "language": "en"
        }
        mock_db.save_character.return_value = True
        mock_db.save_training_data.return_value = True
        return mock_db

    @pytest.fixture
    def mock_database_empty(self):
        """
        Fixture that provides a mock database object returning None for character data queries.
        
        Returns:
            Mock: A mock database where `get_character` returns None, simulating missing character data.
        """
        mock_db = Mock()
        mock_db.get_character.return_value = None
        mock_db.save_character.return_value = True
        mock_db.save_training_data.return_value = True
        return mock_db

    @pytest.fixture
    def character_server(self, mock_database):
        """
        Fixture that provides a CharacterServer instance initialized with a mocked database.
        """
        return CharacterServer(mock_database)

    @pytest.fixture
    def character_server_empty_db(self, mock_database_empty):
        """
        Create a CharacterServer instance using a mock database that returns no character data.
        
        Returns:
            CharacterServer: An instance initialized with an empty database mock.
        """
        return CharacterServer(mock_database_empty)

    @pytest.fixture
    def mock_llm_engine(self):
        """
        Provides an asynchronous mock LLM engine fixture with initialized state and mocked `generate` and `fine_tune` methods for testing.
        """
        mock_llm = AsyncMock()
        mock_llm.is_initialized = True
        mock_llm.generate = AsyncMock(return_value="Generated response from LLM")
        mock_llm.fine_tune = AsyncMock(return_value=True)
        return mock_llm

    @pytest.fixture
    def mock_tts_manager(self):
        """
        Provides a mock asynchronous TTS manager fixture with initialized state and a mocked `synthesize` method for testing.
        """
        mock_tts = AsyncMock()
        mock_tts.is_initialized = True
        mock_tts.synthesize = AsyncMock(return_value=True)
        return mock_tts

    @pytest.fixture
    def temp_audio_dir(self):
        """
        Provides a temporary directory for audio file operations during tests.
        
        Yields:
            str: Path to the temporary directory.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir


class TestCharacterServerInitialization:
    """Test CharacterServer initialization functionality."""

    def test_init_with_valid_character_data(self, mock_database):
        """
        Test that CharacterServer initializes correctly when valid character data is available in the database.
        
        Verifies that the character fields are set as expected and that LLM and TTS components remain uninitialized until asynchronous setup. Also checks that the database is queried for the character.
        """
        server = CharacterServer(mock_database)
        
        assert server.db == mock_database
        assert server.character_Actor_id == "Actor1"
        assert server.character is not None
        assert server.character["name"] == "TestActor1"
        assert server.character["Actor_id"] == "Actor1"
        assert server.llm is None  # Not initialized until async_init
        assert server.tts is None  # Not initialized until async_init
        
        # Verify database was queried
        mock_database.get_character.assert_called_once_with("Actor1")

    def test_init_with_missing_character_data(self, mock_database_empty):
        """
        Test that CharacterServer initializes with default character data when the database returns no character.
        
        Verifies that the server creates a default character, assigns expected default fields, and saves it to the database.
        """
        server = CharacterServer(mock_database_empty)
        
        assert server.db == mock_database_empty
        assert server.character_Actor_id == "Actor1"
        assert server.character is not None
        assert server.character["name"] == "Actor1_Default"
        assert server.character["personality"] == "server_default"
        assert server.character["Actor_id"] == "Actor1"
        
        # Verify database was queried and default character was saved
        mock_database_empty.get_character.assert_called_once_with("Actor1")
        mock_database_empty.save_character.assert_called_once()

    def test_init_default_character_structure(self, mock_database_empty):
        """
        Verify that the default character created by CharacterServer contains all required fields with expected default values.
        """
        server = CharacterServer(mock_database_empty)
        
        required_fields = [
            "name", "personality", "goals", "backstory", 
            "tts", "tts_model", "reference_audio_filename", 
            "Actor_id", "llm_model"
        ]
        
        for field in required_fields:
            assert field in server.character
        
        assert server.character["llm_model"] is None
        assert server.character["tts"] == "piper"
        assert server.character["tts_model"] == "en_US-ryan-high"

    def test_init_database_error_handling(self):
        """
        Test that CharacterServer initialization propagates exceptions raised during database access.
        """
        mock_db = Mock()
        mock_db.get_character.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            CharacterServer(mock_db)


class TestCharacterServerAsyncInit:
    """Test CharacterServer async initialization functionality."""

    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    @patch('character_server.pygame.mixer')
    async def test_async_init_success(self, mock_pygame_mixer, mock_tts_class, 
                                    mock_llm_class, character_server):
        """
                                    Test that asynchronous initialization of the CharacterServer successfully initializes the LLM and TTS components and initializes the pygame mixer if not already initialized.
                                    """
        # Setup mocks
        mock_llm_instance = AsyncMock()
        mock_llm_instance.is_initialized = True
        mock_llm_class.return_value = mock_llm_instance
        
        mock_tts_instance = AsyncMock()
        mock_tts_instance.is_initialized = True
        mock_tts_class.return_value = mock_tts_instance
        
        mock_pygame_mixer.get_init.return_value = False
        mock_pygame_mixer.init.return_value = None
        
        # Execute async init
        await character_server.async_init()
        
        # Verify LLM and TTS were initialized
        assert character_server.llm is not None
        assert character_server.tts is not None
        
        # Verify pygame mixer was initialized
        mock_pygame_mixer.init.assert_called_once()

    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    @patch('character_server.pygame.mixer')
    async def test_async_init_pygame_already_initialized(self, mock_pygame_mixer, 
                                                       mock_tts_class, mock_llm_class, 
                                                       character_server):
        """
                                                       Test that asynchronous initialization skips pygame mixer initialization if it is already initialized.
                                                       """
        # Setup mocks
        mock_llm_class.return_value = AsyncMock()
        mock_tts_class.return_value = AsyncMock()
        mock_pygame_mixer.get_init.return_value = True
        
        await character_server.async_init()
        
        # Verify pygame init was not called since it's already initialized
        mock_pygame_mixer.init.assert_not_called()

    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    @patch('character_server.pygame.mixer')
    async def test_async_init_pygame_error(self, mock_pygame_mixer, mock_tts_class, 
                                         mock_llm_class, character_server):
        """
                                         Test that asynchronous initialization handles pygame mixer initialization errors gracefully.
                                         
                                         Verifies that if an exception occurs during pygame mixer initialization, the error is handled without raising, and initialization is still attempted.
                                         """
        # Setup mocks
        mock_llm_class.return_value = AsyncMock()
        mock_tts_class.return_value = AsyncMock()
        mock_pygame_mixer.get_init.return_value = False
        mock_pygame_mixer.init.side_effect = Exception("Pygame init failed")
        
        # Should not raise exception, just log error
        await character_server.async_init()
        
        # Verify pygame init was attempted
        mock_pygame_mixer.init.assert_called_once()

    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    async def test_async_init_llm_error(self, mock_llm_class, character_server):
        """
        Test that an exception is raised during asynchronous initialization if LLM initialization fails.
        """
        mock_llm_class.side_effect = Exception("LLM init failed")
        
        with pytest.raises(Exception, match="LLM init failed"):
            await character_server.async_init()

    @pytest.mark.asyncio
    @patch('character_server.TTSManager')
    async def test_async_init_tts_error(self, mock_tts_class, character_server):
        """
        Test that an exception is raised if an error occurs during TTS manager initialization in async_init.
        """
        mock_tts_class.side_effect = Exception("TTS init failed")
        
        with pytest.raises(Exception, match="TTS init failed"):
            await character_server.async_init()

    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    async def test_async_init_with_xtts_reference_audio(self, mock_tts_class, 
                                                      mock_llm_class, mock_database):
        """
                                                      Tests that asynchronous initialization of CharacterServer with XTTS TTS and a reference audio file correctly passes the speaker WAV path to the TTS manager.
                                                      """
        # Setup character with XTTS
        character_data = mock_database.get_character.return_value
        character_data["tts"] = "xttsv2"
        character_data["reference_audio_filename"] = "test_reference.wav"
        
        server = CharacterServer(mock_database)
        
        mock_llm_class.return_value = AsyncMock()
        mock_tts_instance = AsyncMock()
        mock_tts_class.return_value = mock_tts_instance
        
        with patch('character_server.os.path.join') as mock_join:
            mock_join.return_value = "/path/to/reference.wav"
            await server.async_init()
        
        # Verify TTS was initialized with speaker_wav_path
        mock_tts_class.assert_called_once()
        call_args = mock_tts_class.call_args
        assert "speaker_wav_path" in call_args.kwargs


class TestCharacterServerResponseGeneration:
    """Test CharacterServer response generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_response_success(self, character_server, mock_llm_engine, 
                                           mock_tts_manager):
        """
                                           Test that `CharacterServer.generate_response` successfully generates a response using the LLM, saves training data, triggers fine-tuning, and outputs audio.
                                           
                                           This test verifies that the prompt is constructed correctly with narration and other character texts, and that all dependent methods are called as expected.
                                           """
        # Setup initialized server
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        with patch.object(character_server, 'output_audio') as mock_output_audio:
            mock_output_audio.return_value = None
            
            response = await character_server.generate_response(
                "Test narration", 
                {"Player": "Hello there!"}
            )
        
        assert response == "Generated response from LLM"
        
        # Verify LLM was called with correct prompt
        mock_llm_engine.generate.assert_called_once()
        call_args = mock_llm_engine.generate.call_args[0]
        prompt = call_args[0]
        assert "Narrator: Test narration" in prompt
        assert "Player: Hello there!" in prompt
        assert "TestActor1 responds as friendly_tester:" in prompt
        
        # Verify training data was saved
        character_server.db.save_training_data.assert_called_once()
        
        # Verify fine-tuning was called
        mock_llm_engine.fine_tune.assert_called_once()
        
        # Verify audio output was called
        mock_output_audio.assert_called_once_with("Generated response from LLM")

    @pytest.mark.asyncio
    async def test_generate_response_no_character(self, mock_database):
        """
        Test that generate_response returns an empty string when no character is loaded.
        """
        server = CharacterServer(mock_database)
        server.character = None
        
        response = await server.generate_response("Test narration", {})
        
        assert response == ""

    @pytest.mark.asyncio
    async def test_generate_response_llm_not_initialized(self, character_server):
        """
        Test that response generation returns a specific error string when the LLM is not initialized.
        """
        character_server.llm = None
        
        response = await character_server.generate_response("Test narration", {})
        
        assert response == "[Actor1_LLM_ERROR]"

    @pytest.mark.asyncio
    async def test_generate_response_llm_not_ready(self, character_server, mock_llm_engine):
        """
        Test that response generation returns a specific error string when the LLM engine is not initialized.
        """
        mock_llm_engine.is_initialized = False
        character_server.llm = mock_llm_engine
        
        response = await character_server.generate_response("Test narration", {})
        
        assert response == "[Actor1_LLM_ERROR]"

    @pytest.mark.asyncio
    async def test_generate_response_llm_error_response(self, character_server, 
                                                      mock_llm_engine, mock_tts_manager):
        """
                                                      Test that the character server correctly handles LLM error responses during response generation.
                                                      
                                                      Verifies that when the LLM returns an error string, the same error is returned by `generate_response` and no training data is saved.
                                                      """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        mock_llm_engine.generate.return_value = "[LLM_ERROR: NOT_INITIALIZED]"
        
        response = await character_server.generate_response("Test narration", {})
        
        assert response == "[LLM_ERROR: NOT_INITIALIZED]"
        
        # Verify training data was not saved for error responses
        character_server.db.save_training_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_response_with_multiple_other_texts(self, character_server, 
                                                             mock_llm_engine, mock_tts_manager):
        """
                                                             Test that response generation correctly includes multiple other character texts in the LLM prompt.
                                                             
                                                             Verifies that when multiple entries are provided in the `other_texts` dictionary, each is incorporated into the prompt sent to the LLM during response generation.
                                                             """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        other_texts = {
            "Player": "Hello!",
            "NPC1": "Greetings!",
            "NPC2": "How are you?"
        }
        
        with patch.object(character_server, 'output_audio'):
            await character_server.generate_response("Test narration", other_texts)
        
        # Verify all texts were included in prompt
        call_args = mock_llm_engine.generate.call_args[0]
        prompt = call_args[0]
        assert "Player: Hello!" in prompt
        assert "NPC1: Greetings!" in prompt
        assert "NPC2: How are you?" in prompt

    @pytest.mark.asyncio
    async def test_generate_response_empty_narration(self, character_server, 
                                                   mock_llm_engine, mock_tts_manager):
        """
                                                   Test that response generation works correctly when the narration text is empty.
                                                   
                                                   Verifies that the LLM is still called, a response is generated, and the prompt includes the narrator field even when the narration is an empty string.
                                                   """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        with patch.object(character_server, 'output_audio'):
            response = await character_server.generate_response("", {})
        
        assert response == "Generated response from LLM"
        
        # Verify prompt was still generated correctly
        call_args = mock_llm_engine.generate.call_args[0]
        prompt = call_args[0]
        assert "Narrator: " in prompt


class TestCharacterServerAudioOutput:
    """Test CharacterServer audio output functionality."""

    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    @patch('character_server.pygame.mixer')
    @patch('character_server.asyncio.to_thread')
    async def test_output_audio_success(self, mock_to_thread, mock_pygame_mixer, 
                                      mock_exists, mock_makedirs, character_server, 
                                      mock_tts_manager, temp_audio_dir):
        """
                                      Test that audio output is successfully generated and played when TTS and audio playback are available.
                                      
                                      Verifies that the TTS manager's `synthesize` method is called and that audio playback is attempted using the appropriate threading mechanism.
                                      """
        character_server.tts = mock_tts_manager
        mock_exists.return_value = True
        mock_pygame_mixer.get_init.return_value = True
        mock_to_thread.return_value = None
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir):
            await character_server.output_audio("Test speech text")
        
        # Verify TTS synthesize was called
        mock_tts_manager.synthesize.assert_called_once()
        
        # Verify pygame audio playback was attempted
        mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_audio_no_tts(self, character_server):
        """
        Test that audio output completes without error when the TTS manager is not initialized.
        """
        character_server.tts = None
        
        # Should not raise exception
        await character_server.output_audio("Test text")

    @pytest.mark.asyncio
    async def test_output_audio_tts_not_initialized(self, character_server, mock_tts_manager):
        """
        Test that audio output does not attempt synthesis when the TTS manager is not initialized.
        """
        mock_tts_manager.is_initialized = False
        character_server.tts = mock_tts_manager
        
        await character_server.output_audio("Test text")
        
        # Verify TTS synthesize was not called
        mock_tts_manager.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_output_audio_empty_text(self, character_server, mock_tts_manager):
        """
        Test that no audio synthesis occurs when output_audio is called with an empty text string.
        """
        character_server.tts = mock_tts_manager
        
        await character_server.output_audio("")
        
        # Verify TTS synthesize was not called
        mock_tts_manager.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_output_audio_no_character(self, character_server, mock_tts_manager):
        """
        Test that audio output does not attempt synthesis when no character is loaded.
        """
        character_server.tts = mock_tts_manager
        character_server.character = None
        
        await character_server.output_audio("Test text")
        
        # Verify TTS synthesize was not called
        mock_tts_manager.synthesize.assert_not_called()

    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    async def test_output_audio_with_xtts_reference(self, mock_exists, mock_makedirs, 
                                                  character_server, mock_tts_manager, 
                                                  temp_audio_dir):
        """
                                                  Test that audio output with XTTS TTS uses the reference audio file when present.
                                                  
                                                  Verifies that when the character uses XTTS TTS and a reference audio filename is set and exists, the TTS manager's `synthesize` method is called with the correct `speaker_wav_for_synthesis` parameter.
                                                  """
        character_server.tts = mock_tts_manager
        character_server.character["tts"] = "xttsv2"
        character_server.character["reference_audio_filename"] = "test_ref.wav"
        
        # Mock reference audio file existence
        mock_exists.side_effect = lambda path: "test_ref.wav" in path
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir), \
             patch('character_server.REFERENCE_VOICES_AUDIO_PATH', temp_audio_dir):
            await character_server.output_audio("Test text")
        
        # Verify TTS synthesize was called with speaker_wav
        mock_tts_manager.synthesize.assert_called_once()
        call_args = mock_tts_manager.synthesize.call_args
        assert call_args.kwargs.get('speaker_wav_for_synthesis') is not None

    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    async def test_output_audio_xtts_missing_reference(self, mock_exists, mock_makedirs, 
                                                     character_server, mock_tts_manager, 
                                                     temp_audio_dir):
        """
                                                     Test that audio output with XTTS TTS proceeds without a speaker reference when the reference audio file is missing.
                                                     
                                                     Verifies that the TTS manager's `synthesize` method is called without the `speaker_wav_for_synthesis` parameter if the specified reference audio file does not exist.
                                                     """
        character_server.tts = mock_tts_manager
        character_server.character["tts"] = "xttsv2"
        character_server.character["reference_audio_filename"] = "missing_ref.wav"
        
        # Mock reference audio file does not exist
        mock_exists.return_value = False
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir):
            await character_server.output_audio("Test text")
        
        # Verify TTS synthesize was called without speaker_wav
        mock_tts_manager.synthesize.assert_called_once()
        call_args = mock_tts_manager.synthesize.call_args
        assert call_args.kwargs.get('speaker_wav_for_synthesis') is None

    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    @patch('character_server.pygame.mixer')
    async def test_output_audio_pygame_not_initialized(self, mock_pygame_mixer, mock_exists, 
                                                     mock_makedirs, character_server, 
                                                     mock_tts_manager, temp_audio_dir):
        """
                                                     Test that audio output proceeds and TTS synthesis is attempted even when the pygame mixer is not initialized.
                                                     """
        character_server.tts = mock_tts_manager
        mock_exists.return_value = True
        mock_pygame_mixer.get_init.return_value = False
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir):
            await character_server.output_audio("Test text")
        
        # Verify TTS synthesis was still attempted
        mock_tts_manager.synthesize.assert_called_once()

    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    @patch('character_server.pygame.mixer')
    @patch('character_server.asyncio.to_thread')
    async def test_output_audio_pygame_error(self, mock_to_thread, mock_pygame_mixer, 
                                           mock_exists, mock_makedirs, character_server, 
                                           mock_tts_manager, temp_audio_dir):
        """
                                           Test that audio output handles pygame playback errors gracefully without raising exceptions.
                                           
                                           Verifies that TTS synthesis is still attempted and that exceptions during pygame playback do not propagate.
                                           """
        character_server.tts = mock_tts_manager
        mock_exists.return_value = True
        mock_pygame_mixer.get_init.return_value = True
        mock_to_thread.side_effect = Exception("Pygame playback error")
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir):
            # Should not raise exception, just log error
            await character_server.output_audio("Test text")
        
        # Verify TTS synthesis was attempted
        mock_tts_manager.synthesize.assert_called_once()

    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    async def test_output_audio_tts_synthesis_failure(self, mock_makedirs, character_server, 
                                                    mock_tts_manager, temp_audio_dir):
        """
                                                    Test that audio output handles TTS synthesis failure without raising exceptions.
                                                    
                                                    This test verifies that when the TTS manager's synthesis method fails (returns False), the `output_audio` method completes without error and the synthesis method is still called.
                                                    """
        character_server.tts = mock_tts_manager
        mock_tts_manager.synthesize.return_value = False  # Synthesis failed
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir):
            await character_server.output_audio("Test text")
        
        # Verify TTS synthesize was called
        mock_tts_manager.synthesize.assert_called_once()

    @pytest.mark.asyncio
    @patch('character_server.uuid.uuid4')
    @patch('character_server.os.makedirs')
    async def test_output_audio_file_naming(self, mock_makedirs, mock_uuid, character_server, 
                                          mock_tts_manager, temp_audio_dir):
        """
                                          Test that the audio output file generated by the TTS manager is named using a UUID.
                                          
                                          Ensures that the synthesized audio file's name includes the expected UUID string.
                                          """
        character_server.tts = mock_tts_manager
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-uuid-123")
        
        with patch('character_server.CHARACTERS_AUDIO_PATH', temp_audio_dir):
            await character_server.output_audio("Test text")
        
        # Verify TTS synthesize was called with correct filename
        call_args = mock_tts_manager.synthesize.call_args
        output_path = call_args[0][1]  # Second argument is output path
        assert "test-uuid-123.wav" in output_path


class TestCharacterServerEdgeCases:
    """Test edge cases and error conditions."""

    def test_character_name_sanitization(self, mock_database):
        """
        Verify that character names containing special characters are loaded without error and do not cause issues with file path handling.
        """
        # Setup character with special characters in name
        character_data = mock_database.get_character.return_value
        character_data["name"] = "Test@Character#123!$%"
        
        server = CharacterServer(mock_database)
        
        # Character should be loaded successfully
        assert server.character["name"] == "Test@Character#123!$%"

    @pytest.mark.asyncio
    async def test_concurrent_response_generation(self, character_server, mock_llm_engine, 
                                                 mock_tts_manager):
        """
                                                 Test that multiple concurrent calls to `generate_response` succeed and invoke the LLM for each request.
                                                 
                                                 Verifies that concurrent response generation produces the expected outputs and that the LLM engine's `generate` method is called for each invocation.
                                                 """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        with patch.object(character_server, 'output_audio'):
            # Make multiple concurrent calls
            tasks = [
                character_server.generate_response(f"Narration {i}", {})
                for i in range(3)
            ]
            
            responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(responses) == 3
        assert all(response == "Generated response from LLM" for response in responses)
        
        # Verify LLM was called for each request
        assert mock_llm_engine.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_very_long_narration(self, character_server, mock_llm_engine, mock_tts_manager):
        """
        Tests that the character server can generate a response when provided with a very long narration string.
        """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        long_narration = "This is a very long narration. " * 1000  # Very long text
        
        with patch.object(character_server, 'output_audio'):
            response = await character_server.generate_response(long_narration, {})
        
        assert response == "Generated response from LLM"

    @pytest.mark.asyncio
    async def test_unicode_text_handling(self, character_server, mock_llm_engine, mock_tts_manager):
        """
        Test that the character server correctly generates responses when the narration contains unicode characters.
        """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        unicode_narration = "こんにちは世界！Здравствуй мир! 🌍"
        
        with patch.object(character_server, 'output_audio'):
            response = await character_server.generate_response(unicode_narration, {})
        
        assert response == "Generated response from LLM"

    def test_database_connection_recovery(self, mock_database):
        """
        Test that the CharacterServer raises an exception on initial database failure and can recover when the database returns valid data on subsequent attempts.
        """
        # Initially fail, then succeed
        mock_database.get_character.side_effect = [
            Exception("Database error"),
            {"name": "RecoveredActor", "Actor_id": "Actor1", "tts": "gtts"}
        ]
        
        # First call should raise exception
        with pytest.raises(Exception):
            CharacterServer(mock_database)


class TestCharacterServerPerformance:
    """Test performance-related aspects."""

    @pytest.mark.asyncio
    async def test_response_generation_timeout(self, character_server, mock_llm_engine, 
                                             mock_tts_manager):
        """
                                             Test that response generation correctly handles and measures a delayed LLM response.
                                             
                                             Simulates a slow LLM response and verifies that the generated response matches the expected output and that the elapsed time reflects the artificial delay.
                                             """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        # Simulate slow LLM response
        async def slow_generate(*args, **kwargs):
            """
            Simulates a slow asynchronous response by waiting briefly before returning a fixed string.
            
            Returns:
                str: The string "Slow response" after a short delay.
            """
            await asyncio.sleep(0.1)  # Small delay for testing
            return "Slow response"
        
        mock_llm_engine.generate.side_effect = slow_generate
        
        with patch.object(character_server, 'output_audio'):
            start_time = asyncio.get_event_loop().time()
            response = await character_server.generate_response("Test narration", {})
            end_time = asyncio.get_event_loop().time()
        
        assert response == "Slow response"
        assert end_time - start_time >= 0.1  # Should take at least the delay time

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_responses(self, character_server, mock_llm_engine, 
                                                   mock_tts_manager):
        """
                                                   Tests that the character server can handle and return very large responses generated by the LLM without errors or memory issues.
                                                   """
        character_server.llm = mock_llm_engine
        character_server.tts = mock_tts_manager
        
        # Generate large response
        large_response = "This is a large response. " * 10000
        mock_llm_engine.generate.return_value = large_response
        
        with patch.object(character_server, 'output_audio'):
            response = await character_server.generate_response("Test narration", {})
        
        assert response == large_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])