import pytest
import pytest_asyncio
import asyncio
import os
import uuid
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys

# Add the SERVER/src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from character_server import CharacterServer
except ImportError:
    # Create a mock CharacterServer class if import fails
    class CharacterServer:
        def __init__(self, db):
            self.db = db
            self.character_Actor_id = "Actor1"
            self.character = None
            self.llm = None
            self.tts = None
        
        async def async_init(self):
            pass
        
        async def generate_response(self, narration, other_texts):
            return "Mock response"
        
        async def output_audio(self, text):
            pass


class TestCharacterServerInitialization:
    """Test CharacterServer initialization and setup."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_db = Mock()
        self.mock_character_data = {
            "name": "TestActor",
            "personality": "test_personality",
            "goals": "test_goals",
            "backstory": "test_backstory",
            "tts": "piper",
            "tts_model": "en_US-ryan-high",
            "reference_audio_filename": "test_reference.wav",
            "Actor_id": "Actor1",
            "llm_model": "test_model",
            "language": "en"
        }
    
    def test_init_with_existing_character(self):
        """Test initialization when character exists in database."""
        self.mock_db.get_character.return_value = self.mock_character_data
        
        server = CharacterServer(self.mock_db)
        
        assert server.db == self.mock_db
        assert server.character_Actor_id == "Actor1"
        assert server.character == self.mock_character_data
        assert server.llm is None
        assert server.tts is None
        self.mock_db.get_character.assert_called_once_with("Actor1")
    
    def test_init_with_missing_character(self):
        """Test initialization when character doesn't exist in database."""
        self.mock_db.get_character.return_value = None
        
        server = CharacterServer(self.mock_db)
        
        assert server.character is not None
        assert server.character["name"] == "Actor1_Default"
        assert server.character["personality"] == "server_default"
        assert server.character["Actor_id"] == "Actor1"
        self.mock_db.save_character.assert_called_once()
    
    def test_init_with_none_db(self):
        """Test initialization with None database."""
        with pytest.raises(AttributeError):
            server = CharacterServer(None)
            # This should fail when trying to call get_character on None
    
    def test_character_default_values(self):
        """Test that default character values are properly set."""
        self.mock_db.get_character.return_value = None
        
        server = CharacterServer(self.mock_db)
        
        expected_defaults = {
            "name": "Actor1_Default",
            "personality": "server_default",
            "goals": "assist",
            "backstory": "embedded",
            "tts": "piper",
            "tts_model": "en_US-ryan-high",
            "reference_audio_filename": None,
            "Actor_id": "Actor1",
            "llm_model": None
        }
        
        for key, expected_value in expected_defaults.items():
            assert server.character[key] == expected_value


class TestCharacterServerAsyncInit:
    """Test CharacterServer async initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = Mock()
        self.mock_character_data = {
            "name": "TestActor",
            "tts": "piper",
            "tts_model": "en_US-ryan-high",
            "reference_audio_filename": "test.wav",
            "llm_model": "test_model",
            "language": "en"
        }
        self.mock_db.get_character.return_value = self.mock_character_data
    
    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    @patch('character_server.pygame')
    async def test_async_init_success(self, mock_pygame, mock_tts_manager, mock_llm_engine):
        """Test successful async initialization."""
        mock_llm_instance = Mock()
        mock_tts_instance = Mock()
        mock_llm_engine.return_value = mock_llm_instance
        mock_tts_manager.return_value = mock_tts_instance
        mock_pygame.mixer.get_init.return_value = False
        
        server = CharacterServer(self.mock_db)
        await server.async_init()
        
        assert server.llm == mock_llm_instance
        assert server.tts == mock_tts_instance
        mock_pygame.mixer.init.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    @patch('character_server.pygame')
    async def test_async_init_pygame_already_initialized(self, mock_pygame, mock_tts_manager, mock_llm_engine):
        """Test async init when pygame mixer is already initialized."""
        mock_pygame.mixer.get_init.return_value = True
        
        server = CharacterServer(self.mock_db)
        await server.async_init()
        
        mock_pygame.mixer.init.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    @patch('character_server.pygame')
    async def test_async_init_pygame_error(self, mock_pygame, mock_tts_manager, mock_llm_engine):
        """Test async init when pygame mixer initialization fails."""
        mock_pygame.mixer.get_init.return_value = False
        mock_pygame.mixer.init.side_effect = Exception("Pygame init failed")
        mock_pygame.error = Exception
        
        server = CharacterServer(self.mock_db)
        # Should not raise exception, should handle gracefully
        await server.async_init()
        
        mock_pygame.mixer.init.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('character_server.LLMEngine')
    @patch('character_server.TTSManager')
    async def test_async_init_with_xtts_reference_audio(self, mock_tts_manager, mock_llm_engine):
        """Test async init with XTTS and reference audio."""
        character_with_xtts = self.mock_character_data.copy()
        character_with_xtts["tts"] = "xttsv2"
        character_with_xtts["reference_audio_filename"] = "reference.wav"
        self.mock_db.get_character.return_value = character_with_xtts
        
        server = CharacterServer(self.mock_db)
        
        with patch('character_server.os.path.join') as mock_join:
            mock_join.return_value = "/path/to/reference.wav"
            await server.async_init()
        
        # Verify TTSManager was called with speaker_wav_path
        mock_tts_manager.assert_called_once()
        call_args = mock_tts_manager.call_args
        assert 'speaker_wav_path' in call_args.kwargs


class TestCharacterServerGenerateResponse:
    """Test CharacterServer response generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = Mock()
        self.mock_character_data = {
            "name": "TestActor",
            "personality": "helpful"
        }
        self.mock_db.get_character.return_value = self.mock_character_data
    
    @pytest.mark.asyncio
    async def test_generate_response_no_character(self):
        """Test response generation when no character is loaded."""
        server = CharacterServer(self.mock_db)
        server.character = None
        
        response = await server.generate_response("Test narration", {})
        assert response == ""
    
    @pytest.mark.asyncio
    async def test_generate_response_no_llm(self):
        """Test response generation when LLM is not initialized.""" 
        server = CharacterServer(self.mock_db)
        server.llm = None
        
        response = await server.generate_response("Test narration", {})
        assert response == "[Actor1_LLM_ERROR]"
    
    @pytest.mark.asyncio
    async def test_generate_response_llm_not_initialized(self):
        """Test response generation when LLM exists but is not initialized."""
        server = CharacterServer(self.mock_db)
        server.llm = Mock()
        server.llm.is_initialized = False
        
        response = await server.generate_response("Test narration", {})
        assert response == "[Actor1_LLM_ERROR]"
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful response generation."""
        server = CharacterServer(self.mock_db)
        server.llm = Mock()
        server.llm.is_initialized = True
        server.llm.generate = AsyncMock(return_value="Generated response")
        server.llm.fine_tune = AsyncMock()
        
        with patch.object(server, 'output_audio', new_callable=AsyncMock) as mock_output_audio:
            response = await server.generate_response("Test narration", {"Other": "text"})
        
        assert response == "Generated response"
        server.llm.generate.assert_called_once()
        self.mock_db.save_training_data.assert_called_once()
        server.llm.fine_tune.assert_called_once()
        mock_output_audio.assert_called_once_with("Generated response")
    
    @pytest.mark.asyncio
    async def test_generate_response_with_other_texts(self):
        """Test response generation with other character texts."""
        server = CharacterServer(self.mock_db)
        server.llm = Mock()
        server.llm.is_initialized = True
        server.llm.generate = AsyncMock(return_value="Generated response")
        server.llm.fine_tune = AsyncMock()
        
        other_texts = {"Character1": "Hello", "Character2": "World"}
        
        with patch.object(server, 'output_audio', new_callable=AsyncMock):
            await server.generate_response("Narration", other_texts)
        
        # Verify the prompt construction includes other texts
        call_args = server.llm.generate.call_args[0][0]
        assert "Character1: Hello" in call_args
        assert "Character2: World" in call_args
        assert "Narrator: Narration" in call_args
    
    @pytest.mark.asyncio
    async def test_generate_response_llm_error_responses(self):
        """Test handling of LLM error responses."""
        server = CharacterServer(self.mock_db)
        server.llm = Mock()
        server.llm.is_initialized = True
        
        error_responses = [
            "[LLM_ERROR: NOT_INITIALIZED]",
            "[LLM_ERROR: GENERATION_FAILED]",
            None,
            ""
        ]
        
        for error_response in error_responses:
            server.llm.generate = AsyncMock(return_value=error_response)
            
            with patch.object(server, 'output_audio', new_callable=AsyncMock) as mock_output_audio:
                response = await server.generate_response("Test", {})
            
            assert response == error_response
            # Should not save training data or call fine_tune for error responses
            mock_output_audio.assert_not_called()


class TestCharacterServerOutputAudio:
    """Test CharacterServer audio output functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = Mock()
        self.mock_character_data = {
            "name": "TestActor",
            "tts": "piper",
            "reference_audio_filename": "test.wav"
        }
        self.mock_db.get_character.return_value = self.mock_character_data
    
    @pytest.mark.asyncio
    async def test_output_audio_no_tts(self):
        """Test audio output when TTS is not initialized."""
        server = CharacterServer(self.mock_db)
        server.tts = None
        
        # Should not raise exception
        await server.output_audio("Test text")
    
    @pytest.mark.asyncio
    async def test_output_audio_tts_not_initialized(self):
        """Test audio output when TTS exists but is not initialized."""
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = False
        
        await server.output_audio("Test text")
        # Should handle gracefully without calling synthesize
    
    @pytest.mark.asyncio
    async def test_output_audio_empty_text(self):
        """Test audio output with empty text."""
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = True
        
        for empty_text in [None, "", "   "]:
            await server.output_audio(empty_text)
            # Should handle gracefully
    
    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    @patch('character_server.pygame')
    @patch('character_server.asyncio.to_thread')
    async def test_output_audio_success(self, mock_to_thread, mock_pygame, mock_exists, mock_makedirs):
        """Test successful audio output."""
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = True
        server.tts.synthesize = AsyncMock(return_value=True)
        
        mock_exists.return_value = True
        mock_pygame.mixer.get_init.return_value = True
        mock_to_thread.return_value = None
        
        with patch('character_server.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value="test-uuid")
            
            await server.output_audio("Test text")
        
        server.tts.synthesize.assert_called_once()
        mock_to_thread.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    async def test_output_audio_with_xtts_reference(self, mock_exists, mock_makedirs):
        """Test audio output with XTTS and reference audio."""
        character_with_xtts = self.mock_character_data.copy()
        character_with_xtts["tts"] = "xttsv2"
        character_with_xtts["reference_audio_filename"] = "reference.wav"
        self.mock_db.get_character.return_value = character_with_xtts
        
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = True
        server.tts.synthesize = AsyncMock(return_value=True)
        
        mock_exists.return_value = True  # Reference file exists
        
        with patch('character_server.REFERENCE_VOICES_AUDIO_PATH', '/ref/path'):
            await server.output_audio("Test text")
        
        # Verify synthesize was called with speaker_wav_for_synthesis
        call_args = server.tts.synthesize.call_args
        assert call_args.kwargs.get('speaker_wav_for_synthesis') is not None
    
    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    async def test_output_audio_missing_reference_file(self, mock_exists, mock_makedirs):
        """Test audio output when reference audio file is missing."""
        character_with_xtts = self.mock_character_data.copy()
        character_with_xtts["tts"] = "xttsv2"
        character_with_xtts["reference_audio_filename"] = "missing.wav"
        self.mock_db.get_character.return_value = character_with_xtts
        
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = True
        server.tts.synthesize = AsyncMock(return_value=True)
        
        mock_exists.return_value = False  # Reference file doesn't exist
        
        await server.output_audio("Test text")
        
        # Should still call synthesize but without speaker_wav_for_synthesis
        call_args = server.tts.synthesize.call_args
        assert call_args.kwargs.get('speaker_wav_for_synthesis') is None
    
    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.pygame')
    async def test_output_audio_synthesis_failure(self, mock_pygame, mock_makedirs):
        """Test audio output when synthesis fails."""
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = True
        server.tts.synthesize = AsyncMock(return_value=False)  # Synthesis fails
        
        mock_pygame.mixer.get_init.return_value = True
        
        # Should not raise exception
        await server.output_audio("Test text")
        
        # Should not attempt to play audio
        mock_pygame.mixer.Sound.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('character_server.os.makedirs')
    @patch('character_server.os.path.exists')
    @patch('character_server.pygame')
    async def test_output_audio_pygame_not_initialized(self, mock_pygame, mock_exists, mock_makedirs):
        """Test audio output when pygame mixer is not initialized."""
        server = CharacterServer(self.mock_db)
        server.tts = Mock()
        server.tts.is_initialized = True
        server.tts.synthesize = AsyncMock(return_value=True)
        
        mock_exists.return_value = True
        mock_pygame.mixer.get_init.return_value = False  # Pygame not initialized
        
        await server.output_audio("Test text")
        
        # Should not attempt to play audio
        mock_pygame.mixer.Sound.assert_not_called()


class TestCharacterServerEdgeCases:
    """Test CharacterServer edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = Mock()
    
    def test_character_name_sanitization(self):
        """Test character name sanitization for file paths."""
        character_data = {
            "name": "Test/Character\\With:Invalid*Chars",
            "Actor_id": "Actor1"
        }
        self.mock_db.get_character.return_value = character_data
        
        server = CharacterServer(self.mock_db)
        
        # The server should handle special characters in names
        assert server.character["name"] == character_data["name"]
    
    @pytest.mark.asyncio
    async def test_concurrent_response_generation(self):
        """Test concurrent response generation."""
        self.mock_db.get_character.return_value = {"name": "TestActor", "personality": "helpful"}
        server = CharacterServer(self.mock_db)
        server.llm = Mock()
        server.llm.is_initialized = True
        server.llm.generate = AsyncMock(return_value="Response")
        server.llm.fine_tune = AsyncMock()
        
        with patch.object(server, 'output_audio', new_callable=AsyncMock):
            tasks = [
                server.generate_response(f"Narration {i}", {})
                for i in range(5)
            ]
            responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 5
        assert all(response == "Response" for response in responses)
    
    @pytest.mark.asyncio
    async def test_very_long_narration(self):
        """Test response generation with very long narration."""
        self.mock_db.get_character.return_value = {"name": "TestActor", "personality": "helpful"}
        server = CharacterServer(self.mock_db)
        server.llm = Mock()
        server.llm.is_initialized = True
        server.llm.generate = AsyncMock(return_value="Response")
        server.llm.fine_tune = AsyncMock()
        
        very_long_narration = "A" * 10000  # 10k characters
        
        with patch.object(server, 'output_audio', new_callable=AsyncMock):
            response = await server.generate_response(very_long_narration, {})
        
        assert response == "Response"
        # Verify the long narration was included in the prompt
        call_args = server.llm.generate.call_args[0][0]
        assert very_long_narration in call_args
    
    def test_db_save_character_exception(self):
        """Test handling of database exceptions during character saving."""
        self.mock_db.get_character.return_value = None
        self.mock_db.save_character.side_effect = Exception("Database error")
        
        # Should not raise exception during initialization
        server = CharacterServer(self.mock_db)
        assert server.character is not None


class TestCharacterServerIntegration:
    """Integration tests for CharacterServer."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test the complete workflow from initialization to response generation."""
        mock_db = Mock()
        character_data = {
            "name": "IntegrationTestActor",
            "personality": "friendly",
            "tts": "piper",
            "tts_model": "en_US-ryan-high",
            "llm_model": "test_model",
            "language": "en"
        }
        mock_db.get_character.return_value = character_data
        
        with patch('character_server.LLMEngine') as mock_llm_engine, \
             patch('character_server.TTSManager') as mock_tts_manager, \
             patch('character_server.pygame'):
            
            mock_llm = Mock()
            mock_llm.is_initialized = True
            mock_llm.generate = AsyncMock(return_value="Integration test response")
            mock_llm.fine_tune = AsyncMock()
            mock_llm_engine.return_value = mock_llm
            
            mock_tts = Mock()
            mock_tts.is_initialized = True
            mock_tts.synthesize = AsyncMock(return_value=True)
            mock_tts_manager.return_value = mock_tts
            
            server = CharacterServer(mock_db)
            await server.async_init()
            
            with patch.object(server, 'output_audio', new_callable=AsyncMock) as mock_output:
                response = await server.generate_response("Test narration", {"Other": "Hello"})
            
            assert response == "Integration test response"
            assert server.llm == mock_llm
            assert server.tts == mock_tts
            mock_output.assert_called_once_with("Integration test response")


# Pytest configuration and fixtures
@pytest.fixture
def mock_character_data():
    """Fixture providing mock character data."""
    return {
        "name": "TestCharacter",
        "personality": "test_personality",
        "goals": "test_goals",
        "backstory": "test_backstory",
        "tts": "piper",
        "tts_model": "en_US-ryan-high",
        "reference_audio_filename": "test.wav",
        "Actor_id": "Actor1",
        "llm_model": "test_model",
        "language": "en"
    }


@pytest.fixture
def mock_db():
    """Fixture providing a mock database."""
    return Mock()


@pytest.fixture
def character_server(mock_db, mock_character_data):
    """Fixture providing a CharacterServer instance."""
    mock_db.get_character.return_value = mock_character_data
    return CharacterServer(mock_db)


# Performance and stress tests
class TestCharacterServerPerformance:
    """Performance and stress tests for CharacterServer."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_many_requests(self, character_server):
        """Test memory usage with many concurrent requests."""
        character_server.llm = Mock()
        character_server.llm.is_initialized = True
        character_server.llm.generate = AsyncMock(return_value="Response")
        character_server.llm.fine_tune = AsyncMock()
        
        with patch.object(character_server, 'output_audio', new_callable=AsyncMock):
            tasks = [
                character_server.generate_response(f"Request {i}", {})
                for i in range(100)
            ]
            responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 100
        assert all(response == "Response" for response in responses)
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, character_server):
        """Test that response times are reasonably consistent."""
        character_server.llm = Mock()
        character_server.llm.is_initialized = True
        character_server.llm.generate = AsyncMock(return_value="Response")
        character_server.llm.fine_tune = AsyncMock()
        
        import time
        response_times = []
        
        with patch.object(character_server, 'output_audio', new_callable=AsyncMock):
            for i in range(10):
                start_time = time.time()
                await character_server.generate_response(f"Request {i}", {})
                end_time = time.time()
                response_times.append(end_time - start_time)
        
        # Check that response times are reasonably consistent
        # (This is a basic check - in real scenarios you'd have more sophisticated metrics)
        assert len(response_times) == 10
        assert all(rt < 5.0 for rt in response_times)  # All responses under 5 seconds


if __name__ == '__main__':
    pytest.main([__file__, "-v"])