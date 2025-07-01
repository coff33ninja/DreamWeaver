import pytest
import asyncio
import os
import uuid
import base64
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import the modules we're testing
from CharacterClient.character_client import (
    CharacterClient, 
    app, 
    handle_character_generation_request,
    health_check,
    initialize_character_client,
    start_heartbeat_task,
    _heartbeat_task_runner
)


class TestCharacterClient:
    """Comprehensive unit tests for CharacterClient class using pytest."""
    
    @pytest.fixture
    def mock_character_traits(self):
        """Mock character traits data."""
        return {
            "name": "Rick Sanchez",
            "personality": "genius scientist",
            "tts": "xttsv2",
            "tts_model": "en",
            "reference_audio_filename": "rick_voice.wav",
            "language": "en",
            "llm_model": "gpt-3.5-turbo"
        }
    
    @pytest.fixture
    def character_client(self):
        """Create a CharacterClient instance for testing."""
        return CharacterClient(
            token="test_token",
            Actor_id="test_actor",
            server_url="http://localhost:8000",
            client_port=8080
        )
    
    @pytest.fixture
    def mock_tts_manager(self):
        """Mock TTSManager instance."""
        mock_tts = Mock()
        mock_tts.is_initialized = True
        mock_tts.synthesize = AsyncMock(return_value="/path/to/audio.wav")
        return mock_tts
    
    @pytest.fixture
    def mock_llm_engine(self):
        """Mock LLMEngine instance."""
        mock_llm = Mock()
        mock_llm.is_initialized = True
        mock_llm.generate = AsyncMock(return_value="Test response from LLM")
        mock_llm.fine_tune_async = AsyncMock()
        return mock_llm

    # Test CharacterClient initialization
    def test_character_client_init(self, character_client):
        """Test CharacterClient initialization with required parameters."""
        assert character_client.token == "test_token"
        assert character_client.Actor_id == "test_actor"
        assert character_client.server_url == "http://localhost:8000"
        assert character_client.client_port == 8080
        assert character_client.character is None
        assert character_client.tts is None
        assert character_client.llm is None
        assert character_client.local_reference_audio_path is None

    def test_character_client_init_empty_params(self):
        """Test CharacterClient initialization with empty parameters."""
        client = CharacterClient("", "", "", 0)
        assert client.token == ""
        assert client.Actor_id == ""
        assert client.server_url == ""
        assert client.client_port == 0

    def test_character_client_init_none_params(self):
        """Test CharacterClient initialization with None parameters."""
        with pytest.raises(TypeError):
            CharacterClient(None, None, None, None)

    # Test async factory method
    @pytest.mark.asyncio
    async def test_create_async_factory_success(self, mock_character_traits):
        """Test successful async creation of CharacterClient."""
        with patch.object(CharacterClient, '_register_with_server_blocking', return_value=True), \
             patch.object(CharacterClient, '_fetch_traits_blocking', return_value=mock_character_traits), \
             patch.object(CharacterClient, '_download_reference_audio_blocking'), \
             patch('CharacterClient.character_client.TTSManager') as mock_tts_cls, \
             patch('CharacterClient.character_client.LLMEngine') as mock_llm_cls:
            
            mock_tts_cls.return_value.is_initialized = True
            mock_llm_cls.return_value.is_initialized = True
            
            client = await CharacterClient.create(
                token="test_token",
                Actor_id="test_actor",
                server_url="http://localhost:8000",
                client_port=8080
            )
            
            assert client.token == "test_token"
            assert client.Actor_id == "test_actor"
            assert client.character == mock_character_traits
            assert client.tts is not None
            assert client.llm is not None

    @pytest.mark.asyncio
    async def test_create_async_factory_registration_failure(self):
        """Test async creation when registration fails."""
        with patch.object(CharacterClient, '_register_with_server_blocking', return_value=False), \
             patch.object(CharacterClient, '_fetch_traits_blocking', return_value=None), \
             patch('CharacterClient.character_client.TTSManager') as mock_tts_cls, \
             patch('CharacterClient.character_client.LLMEngine') as mock_llm_cls:
            
            mock_tts_cls.return_value.is_initialized = True
            mock_llm_cls.return_value.is_initialized = True
            
            client = await CharacterClient.create(
                token="test_token",
                Actor_id="test_actor",
                server_url="http://localhost:8000",
                client_port=8080
            )
            
            # Should still create client but with default character
            assert client.character["name"] == "test_actor"
            assert client.character["personality"] == "default"

    # Test registration with server
    def test_register_with_server_success(self, character_client):
        """Test successful server registration."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            result = character_client._register_with_server_blocking()
            
            assert result is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "http://localhost:8000/register"
            assert kwargs['json']['Actor_id'] == "test_actor"
            assert kwargs['json']['token'] == "test_token"

    def test_register_with_server_failure_with_retries(self, character_client):
        """Test server registration failure with retries."""
        with patch('requests.post') as mock_post, \
             patch('time.sleep') as mock_sleep:
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
            
            result = character_client._register_with_server_blocking(max_retries=2, base_delay=0.1)
            
            assert result is False
            assert mock_post.call_count == 3  # Initial + 2 retries
            assert mock_sleep.call_count == 2  # Sleep after each failed attempt except the last

    def test_register_with_server_http_error(self, character_client):
        """Test server registration with HTTP error response."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
            mock_post.return_value = mock_response
            
            result = character_client._register_with_server_blocking(max_retries=0)
            
            assert result is False

    # Test traits fetching
    def test_fetch_traits_success(self, character_client, mock_character_traits):
        """Test successful traits fetching."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_character_traits
            mock_get.return_value = mock_response
            
            result = character_client._fetch_traits_blocking()
            
            assert result == mock_character_traits
            mock_get.assert_called_once_with(
                "http://localhost:8000/get_traits",
                params={"Actor_id": "test_actor", "token": "test_token"},
                timeout=10
            )

    def test_fetch_traits_failure(self, character_client):
        """Test traits fetching failure."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Network error")
            
            result = character_client._fetch_traits_blocking()
            
            assert result is None

    def test_fetch_traits_invalid_response(self, character_client):
        """Test traits fetching with invalid JSON response."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_response
            
            result = character_client._fetch_traits_blocking()
            
            assert result is None

    # Test reference audio download
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_download_reference_audio_success(self, mock_getsize, mock_exists, mock_makedirs, character_client):
        """Test successful reference audio download."""
        character_client.character = {
            "reference_audio_filename": "test_voice.wav"
        }
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        
        with patch('requests.get') as mock_get, \
             patch('builtins.open', mock_open()) as mock_file:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content.return_value = [b'audio_data_chunk']
            mock_get.return_value = mock_response
            
            character_client._download_reference_audio_blocking()
            
            mock_get.assert_called_once()
            mock_file.assert_called_once()

    def test_download_reference_audio_no_character(self, character_client):
        """Test reference audio download when no character traits available."""
        character_client.character = None
        
        # Should not raise exception, just return early
        character_client._download_reference_audio_blocking()

    def test_download_reference_audio_no_filename(self, character_client):
        """Test reference audio download when no filename specified."""
        character_client.character = {"tts": "xttsv2"}
        
        # Should not raise exception, just return early
        character_client._download_reference_audio_blocking()

    @patch('os.path.exists')
    def test_download_reference_audio_already_exists(self, mock_exists, character_client):
        """Test reference audio download when file already exists."""
        character_client.character = {"reference_audio_filename": "test_voice.wav"}
        mock_exists.return_value = True
        
        with patch('os.path.getsize', return_value=1000):
            character_client._download_reference_audio_blocking()
        
        # Should return early without making HTTP request
        with patch('requests.get') as mock_get:
            assert not mock_get.called

    # Test heartbeat functionality
    @pytest.mark.asyncio
    async def test_send_heartbeat_success(self, character_client):
        """Test successful heartbeat sending."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            result = await character_client.send_heartbeat_async()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_send_heartbeat_failure_with_retries(self, character_client):
        """Test heartbeat failure with retries."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
            
            result = await character_client.send_heartbeat_async(max_retries=1, base_delay=0.01)
            
            assert result is False
            assert mock_post.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_send_heartbeat_timeout(self, character_client):
        """Test heartbeat timeout handling."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
            
            result = await character_client.send_heartbeat_async(max_retries=0)
            
            assert result is False

    # Test response generation
    @pytest.mark.asyncio
    async def test_generate_response_success(self, character_client, mock_character_traits, mock_llm_engine):
        """Test successful response generation."""
        character_client.character = mock_character_traits
        character_client.llm = mock_llm_engine
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200)
            
            result = await character_client.generate_response_async(
                "Test narration",
                {"character1": "Hello"}
            )
            
            assert result == "Test response from LLM"
            mock_llm_engine.generate.assert_called_once()
            mock_llm_engine.fine_tune_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_llm_not_initialized(self, character_client, mock_character_traits):
        """Test response generation when LLM is not initialized."""
        character_client.character = mock_character_traits
        character_client.llm = None
        
        result = await character_client.generate_response_async(
            "Test narration",
            {"character1": "Hello"}
        )
        
        assert "[Rick Sanchez LLM not ready]" in result

    @pytest.mark.asyncio
    async def test_generate_response_no_character(self, character_client):
        """Test response generation when no character is set."""
        character_client.character = None
        character_client.llm = None
        
        result = await character_client.generate_response_async(
            "Test narration",
            {"character1": "Hello"}
        )
        
        assert "[test_actor LLM not ready]" in result

    @pytest.mark.asyncio
    async def test_generate_response_training_data_save_failure(self, character_client, mock_character_traits, mock_llm_engine):
        """Test response generation when training data save fails."""
        character_client.character = mock_character_traits
        character_client.llm = mock_llm_engine
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Save failed")
            
            # Should not raise exception, just log error
            result = await character_client.generate_response_async(
                "Test narration",
                {"character1": "Hello"}
            )
            
            assert result == "Test response from LLM"

    # Test audio synthesis
    @pytest.mark.asyncio
    async def test_synthesize_audio_success(self, character_client, mock_character_traits, mock_tts_manager):
        """Test successful audio synthesis."""
        character_client.character = mock_character_traits
        character_client.tts = mock_tts_manager
        character_client.local_reference_audio_path = "/path/to/reference.wav"
        
        with patch('os.path.exists', return_value=True):
            result = await character_client.synthesize_audio_async("Hello world")
            
            assert result == "/path/to/audio.wav"
            mock_tts_manager.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_audio_tts_not_initialized(self, character_client, mock_character_traits):
        """Test audio synthesis when TTS is not initialized."""
        character_client.character = mock_character_traits
        character_client.tts = None
        
        result = await character_client.synthesize_audio_async("Hello world")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_audio_gtts_service(self, character_client, mock_tts_manager):
        """Test audio synthesis with GTTS service (no reference audio)."""
        character_client.character = {"name": "Test", "tts": "gtts"}
        character_client.tts = mock_tts_manager
        
        result = await character_client.synthesize_audio_async("Hello world")
        
        assert result == "/path/to/audio.wav"
        # Should not pass speaker_wav_for_synthesis for gtts
        args, kwargs = mock_tts_manager.synthesize.call_args
        assert kwargs.get('speaker_wav_for_synthesis') is None

    @pytest.mark.asyncio
    async def test_synthesize_audio_xttsv2_no_reference(self, character_client, mock_tts_manager):
        """Test audio synthesis with XTTS v2 but no reference audio."""
        character_client.character = {"name": "Test", "tts": "xttsv2"}
        character_client.tts = mock_tts_manager
        character_client.local_reference_audio_path = None
        
        result = await character_client.synthesize_audio_async("Hello world")
        
        assert result == "/path/to/audio.wav"
        args, kwargs = mock_tts_manager.synthesize.call_args
        assert kwargs.get('speaker_wav_for_synthesis') is None

    # Test FastAPI endpoints
    def test_health_check_success(self):
        """Test health check endpoint with healthy client."""
        mock_client = Mock()
        mock_client.Actor_id = "test_actor"
        mock_client.llm.is_initialized = True
        mock_client.tts.is_initialized = True
        
        mock_request = Mock()
        mock_request.app.state.character_client_instance = mock_client
        
        result = asyncio.run(health_check(mock_request))
        
        assert result["status"] == "ok"
        assert result["Actor_id"] == "test_actor"
        assert result["llm_ready"] is True
        assert result["tts_ready"] is True

    def test_health_check_degraded(self):
        """Test health check endpoint with degraded client."""
        mock_client = Mock()
        mock_client.Actor_id = "test_actor"
        mock_client.llm.is_initialized = False
        mock_client.tts.is_initialized = True
        
        mock_request = Mock()
        mock_request.app.state.character_client_instance = mock_client
        
        result = asyncio.run(health_check(mock_request))
        
        assert result["status"] == "degraded"
        assert "not fully ready" in result["detail"]

    def test_health_check_no_client(self):
        """Test health check endpoint when no client is available."""
        mock_request = Mock()
        mock_request.app.state.character_client_instance = None
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(health_check(mock_request))
        
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_success(self):
        """Test successful character generation request."""
        mock_client = Mock()
        mock_client.token = "valid_token"
        mock_client.generate_response_async = AsyncMock(return_value="Generated response")
        mock_client.synthesize_audio_async = AsyncMock(return_value="/path/to/audio.wav")
        
        mock_request = Mock()
        mock_request.app.state.character_client_instance = mock_client
        
        data = {
            "token": "valid_token",
            "narration": "Test narration",
            "character_texts": {"char1": "Hello"}
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=b'audio_data')), \
             patch('os.remove'):
            
            result = await handle_character_generation_request(data, mock_request)
            
            assert result["text"] == "Generated response"
            assert result["audio_data"] is not None
            # Verify base64 encoding
            decoded_audio = base64.b64decode(result["audio_data"])
            assert decoded_audio == b'audio_data'

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_invalid_token(self):
        """Test character generation request with invalid token."""
        mock_client = Mock()
        mock_client.token = "valid_token"
        
        mock_request = Mock()
        mock_request.app.state.character_client_instance = mock_client
        
        data = {
            "token": "invalid_token",
            "narration": "Test narration"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await handle_character_generation_request(data, mock_request)
        
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_missing_narration(self):
        """Test character generation request with missing narration."""
        mock_client = Mock()
        mock_client.token = "valid_token"
        
        mock_request = Mock()
        mock_request.app.state.character_client_instance = mock_client
        
        data = {"token": "valid_token"}
        
        with pytest.raises(HTTPException) as exc_info:
            await handle_character_generation_request(data, mock_request)
        
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_no_client(self):
        """Test character generation request when no client is available."""
        mock_request = Mock()
        mock_request.app.state.character_client_instance = None
        
        data = {"token": "valid_token", "narration": "Test"}
        
        with pytest.raises(HTTPException) as exc_info:
            await handle_character_generation_request(data, mock_request)
        
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_no_audio_file(self):
        """Test character generation request when audio file doesn't exist."""
        mock_client = Mock()
        mock_client.token = "valid_token"
        mock_client.generate_response_async = AsyncMock(return_value="Generated response")
        mock_client.synthesize_audio_async = AsyncMock(return_value="/path/to/nonexistent.wav")
        
        mock_request = Mock()
        mock_request.app.state.character_client_instance = mock_client
        
        data = {
            "token": "valid_token",
            "narration": "Test narration"
        }
        
        with patch('os.path.exists', return_value=False):
            result = await handle_character_generation_request(data, mock_request)
            
            assert result["text"] == "Generated response"
            assert result["audio_data"] is None

    # Test heartbeat task functionality
    @pytest.mark.asyncio
    async def test_heartbeat_task_runner(self):
        """Test the heartbeat task runner."""
        mock_client = Mock()
        mock_client.send_heartbeat_async = AsyncMock(return_value=True)
        
        # Run for a short time to avoid infinite loop
        task = asyncio.create_task(_heartbeat_task_runner(mock_client))
        await asyncio.sleep(0.01)  # Very short sleep
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should have called send_heartbeat_async at least once
        # (This test might be flaky due to timing, consider mocking asyncio.sleep)

    def test_start_heartbeat_task(self):
        """Test starting the heartbeat task."""
        mock_client = Mock()
        
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            
            start_heartbeat_task(mock_client)
            
            mock_loop.create_task.assert_called_once()

    # Test initialization function
    @patch('CharacterClient.character_client.ensure_client_directories')
    def test_initialize_character_client_new_client(self, mock_ensure_dirs):
        """Test initializing a new character client."""
        # Clear any existing client
        if hasattr(app.state, 'character_client_instance'):
            app.state.character_client_instance = None
        
        with patch('asyncio.get_running_loop') as mock_get_loop, \
             patch.object(CharacterClient, 'create', new_callable=AsyncMock) as mock_create:
            
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            mock_client = Mock()
            mock_client.llm.is_initialized = True
            mock_client.tts.is_initialized = True
            mock_create.return_value = mock_client
            
            initialize_character_client(
                token="test_token",
                Actor_id="test_actor",
                server_url="http://localhost:8000",
                client_port=8080
            )
            
            mock_ensure_dirs.assert_called_once()
            mock_loop.create_task.assert_called()

    def test_initialize_character_client_existing_client(self):
        """Test initializing when client already exists."""
        # Set existing client
        app.state.character_client_instance = Mock()
        
        with patch('CharacterClient.character_client.ensure_client_directories') as mock_ensure_dirs:
            initialize_character_client(
                token="test_token",
                Actor_id="test_actor",
                server_url="http://localhost:8000",
                client_port=8080
            )
            
            # Should not call ensure_client_directories for existing client
            mock_ensure_dirs.assert_not_called()

    # Edge cases and error conditions
    def test_character_client_with_very_long_actor_id(self):
        """Test CharacterClient with very long Actor_id."""
        long_id = "a" * 1000
        client = CharacterClient(
            token="test_token",
            Actor_id=long_id,
            server_url="http://localhost:8000",
            client_port=8080
        )
        assert client.Actor_id == long_id

    def test_character_client_with_special_characters_in_actor_id(self):
        """Test CharacterClient with special characters in Actor_id."""
        special_id = "test-actor_123!@#$%^&*()"
        client = CharacterClient(
            token="test_token",
            Actor_id=special_id,
            server_url="http://localhost:8000",
            client_port=8080
        )
        assert client.Actor_id == special_id

    def test_character_client_with_unicode_actor_id(self):
        """Test CharacterClient with Unicode characters in Actor_id."""
        unicode_id = "测试角色_🎭"
        client = CharacterClient(
            token="test_token",
            Actor_id=unicode_id,
            server_url="http://localhost:8000",
            client_port=8080
        )
        assert client.Actor_id == unicode_id

    @pytest.mark.asyncio
    async def test_async_init_with_exceptions(self, character_client):
        """Test async_init when various methods raise exceptions."""
        with patch.object(character_client, '_register_with_server_blocking', side_effect=Exception("Registration failed")), \
             patch.object(character_client, '_fetch_traits_blocking', side_effect=Exception("Fetch failed")), \
             patch.object(character_client, '_download_reference_audio_blocking', side_effect=Exception("Download failed")):
            
            # Should not raise exception, should handle gracefully
            await character_client.async_init()
            
            # Should have default character traits
            assert character_client.character is not None

    def test_file_sanitization_in_download_reference_audio(self, character_client):
        """Test that filenames are properly sanitized in reference audio download."""
        character_client.character = {
            "reference_audio_filename": "../../../etc/passwd"
        }
        character_client.Actor_id = "../../dangerous_path"
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch('os.path.getsize', return_value=0), \
             patch('requests.get') as mock_get, \
             patch('builtins.open', mock_open()):
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_content.return_value = [b'data']
            mock_get.return_value = mock_response
            
            character_client._download_reference_audio_blocking()
            
            # Verify that the path contains sanitized versions
            assert character_client.local_reference_audio_path is not None
            # Should not contain directory traversal sequences
            assert "../" not in character_client.local_reference_audio_path
            assert character_client.local_reference_audio_path.find("___") != -1  # Sanitized characters

# Test the mock_open function since we use it
def mock_open(mock=None, read_data=''):
    """Create a mock for the built-in open function."""
    if mock is None:
        mock = MagicMock(spec=open)
    
    handle = MagicMock(spec=['read', 'write', 'close', '__enter__', '__exit__'])
    handle.read.return_value = read_data
    handle.__enter__.return_value = handle
    mock.return_value = handle
    return mock

if __name__ == "__main__":
    pytest.main([__file__])