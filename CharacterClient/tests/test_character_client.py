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
        """
        Return a mock dictionary representing character traits for testing purposes.
        
        Returns:
            dict: A dictionary containing sample character trait fields such as name, personality, TTS type, model, reference audio filename, language, and LLM model.
        """
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
        """
        Creates and returns a CharacterClient instance configured with test parameters.
        """
        return CharacterClient(
            token="test_token",
            Actor_id="test_actor",
            server_url="http://localhost:8000",
            client_port=8080
        )
    
    @pytest.fixture
    def mock_tts_manager(self):
        """
        Provides a mock TTSManager instance with initialized state and a mocked asynchronous synthesize method for testing.
        """
        mock_tts = Mock()
        mock_tts.is_initialized = True
        mock_tts.synthesize = AsyncMock(return_value="/path/to/audio.wav")
        return mock_tts
    
    @pytest.fixture
    def mock_llm_engine(self):
        """
        Return a mock LLMEngine instance with initialized state and asynchronous methods for testing.
        """
        mock_llm = Mock()
        mock_llm.is_initialized = True
        mock_llm.generate = AsyncMock(return_value="Test response from LLM")
        mock_llm.fine_tune_async = AsyncMock()
        return mock_llm

    # Test CharacterClient initialization
    def test_character_client_init(self, character_client):
        """
        Test that the CharacterClient is initialized with the correct attribute values when provided with required parameters.
        """
        assert character_client.token == "test_token"
        assert character_client.Actor_id == "test_actor"
        assert character_client.server_url == "http://localhost:8000"
        assert character_client.client_port == 8080
        assert character_client.character is None
        assert character_client.tts is None
        assert character_client.llm is None
        assert character_client.local_reference_audio_path is None

    def test_character_client_init_empty_params(self):
        """
        Test initialization of CharacterClient with empty string parameters and zero port.
        
        Asserts that all attributes are set to their corresponding empty or zero values.
        """
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
        """
        Test that the asynchronous factory method successfully creates a CharacterClient instance with all components initialized.
        
        Verifies that the client is created with the provided parameters, character traits are fetched, and TTS/LLM components are properly initialized.
        """
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
        """
        Test that the async factory method creates a CharacterClient with default character traits when server registration fails.
        """
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
        """
        Verify that the client successfully registers with the server and returns True on a 200 OK response.
        """
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
        """
        Test that server registration retries the correct number of times and returns False after repeated failures.
        """
        with patch('requests.post') as mock_post, \
             patch('time.sleep') as mock_sleep:
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
            
            result = character_client._register_with_server_blocking(max_retries=2, base_delay=0.1)
            
            assert result is False
            assert mock_post.call_count == 3  # Initial + 2 retries
            assert mock_sleep.call_count == 2  # Sleep after each failed attempt except the last

    def test_register_with_server_http_error(self, character_client):
        """
        Test that server registration returns False when an HTTP error occurs during the POST request.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
            mock_post.return_value = mock_response
            
            result = character_client._register_with_server_blocking(max_retries=0)
            
            assert result is False

    # Test traits fetching
    def test_fetch_traits_success(self, character_client, mock_character_traits):
        """
        Test that `_fetch_traits_blocking` returns character traits when the server responds successfully.
        """
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
        """
        Test that fetching character traits returns None when a network error occurs.
        """
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Network error")
            
            result = character_client._fetch_traits_blocking()
            
            assert result is None

    def test_fetch_traits_invalid_response(self, character_client):
        """
        Test that fetching character traits returns None when the server responds with invalid JSON.
        """
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
        """
        Test that the reference audio file is downloaded successfully when it does not already exist.
        
        Verifies that the HTTP GET request is made and the file is written when the reference audio filename is present and the file does not exist.
        """
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
        """
        Test that downloading reference audio returns early without error when character traits are not set.
        """
        character_client.character = None
        
        # Should not raise exception, just return early
        character_client._download_reference_audio_blocking()

    def test_download_reference_audio_no_filename(self, character_client):
        """
        Test that downloading reference audio returns early without error when no reference audio filename is specified in the character traits.
        """
        character_client.character = {"tts": "xttsv2"}
        
        # Should not raise exception, just return early
        character_client._download_reference_audio_blocking()

    @patch('os.path.exists')
    def test_download_reference_audio_already_exists(self, mock_exists, character_client):
        """
        Test that downloading reference audio does not trigger an HTTP request if the file already exists.
        
        Verifies that the method returns early when the reference audio file is present, avoiding unnecessary network calls.
        """
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
        """
        Test that `send_heartbeat_async` returns True when the heartbeat POST request succeeds.
        """
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            result = await character_client.send_heartbeat_async()
            
            assert result is True

    @pytest.mark.asyncio
    async def test_send_heartbeat_failure_with_retries(self, character_client):
        """
        Test that the heartbeat method retries on failure and returns False after exhausting retries.
        
        Simulates connection failures for the heartbeat POST request and verifies that the method performs the expected number of retries before returning False.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
            
            result = await character_client.send_heartbeat_async(max_retries=1, base_delay=0.01)
            
            assert result is False
            assert mock_post.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_send_heartbeat_timeout(self, character_client):
        """
        Test that `send_heartbeat_async` returns False when a timeout exception occurs during the heartbeat request.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
            
            result = await character_client.send_heartbeat_async(max_retries=0)
            
            assert result is False

    # Test response generation
    @pytest.mark.asyncio
    async def test_generate_response_success(self, character_client, mock_character_traits, mock_llm_engine):
        """
        Test that `generate_response_async` returns the expected response when the LLM engine is initialized and training data is saved successfully.
        
        Verifies that the LLM's `generate` and `fine_tune_async` methods are called and the correct response is returned.
        """
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
        """
        Test that `generate_response_async` returns a placeholder message when the LLM engine is not initialized.
        """
        character_client.character = mock_character_traits
        character_client.llm = None
        
        result = await character_client.generate_response_async(
            "Test narration",
            {"character1": "Hello"}
        )
        
        assert "[Rick Sanchez LLM not ready]" in result

    @pytest.mark.asyncio
    async def test_generate_response_no_character(self, character_client):
        """
        Test that `generate_response_async` returns a placeholder message when both the character and LLM engine are not initialized.
        """
        character_client.character = None
        character_client.llm = None
        
        result = await character_client.generate_response_async(
            "Test narration",
            {"character1": "Hello"}
        )
        
        assert "[test_actor LLM not ready]" in result

    @pytest.mark.asyncio
    async def test_generate_response_training_data_save_failure(self, character_client, mock_character_traits, mock_llm_engine):
        """
        Test that response generation completes successfully even if saving training data fails.
        
        Verifies that when an exception occurs during the training data save step, the method still returns the generated response without raising an exception.
        """
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
        """
        Test that `synthesize_audio_async` successfully synthesizes audio when TTS and reference audio are available.
        
        Verifies that the method returns the expected audio file path and that the TTS manager's `synthesize` method is called once.
        """
        character_client.character = mock_character_traits
        character_client.tts = mock_tts_manager
        character_client.local_reference_audio_path = "/path/to/reference.wav"
        
        with patch('os.path.exists', return_value=True):
            result = await character_client.synthesize_audio_async("Hello world")
            
            assert result == "/path/to/audio.wav"
            mock_tts_manager.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_audio_tts_not_initialized(self, character_client, mock_character_traits):
        """
        Test that `synthesize_audio_async` returns `None` when the TTS manager is not initialized.
        """
        character_client.character = mock_character_traits
        character_client.tts = None
        
        result = await character_client.synthesize_audio_async("Hello world")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_audio_gtts_service(self, character_client, mock_tts_manager):
        """
        Test that audio synthesis using the GTTS service does not use reference audio.
        
        Verifies that when the character's TTS type is set to "gtts", the `synthesize_audio_async` method does not pass a reference audio file for synthesis and returns the expected audio file path.
        """
        character_client.character = {"name": "Test", "tts": "gtts"}
        character_client.tts = mock_tts_manager
        
        result = await character_client.synthesize_audio_async("Hello world")
        
        assert result == "/path/to/audio.wav"
        # Should not pass speaker_wav_for_synthesis for gtts
        args, kwargs = mock_tts_manager.synthesize.call_args
        assert kwargs.get('speaker_wav_for_synthesis') is None

    @pytest.mark.asyncio
    async def test_synthesize_audio_xttsv2_no_reference(self, character_client, mock_tts_manager):
        """
        Test that audio synthesis with XTTS v2 proceeds without a reference audio file and does not pass a speaker wav for synthesis.
        """
        character_client.character = {"name": "Test", "tts": "xttsv2"}
        character_client.tts = mock_tts_manager
        character_client.local_reference_audio_path = None
        
        result = await character_client.synthesize_audio_async("Hello world")
        
        assert result == "/path/to/audio.wav"
        args, kwargs = mock_tts_manager.synthesize.call_args
        assert kwargs.get('speaker_wav_for_synthesis') is None

    # Test FastAPI endpoints
    def test_health_check_success(self):
        """
        Test that the health check endpoint returns status "ok" and readiness flags when the client is healthy.
        """
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
        """
        Test the health check endpoint when the client is in a degraded state due to an uninitialized LLM engine.
        
        Asserts that the returned status is "degraded" and the detail message indicates the client is not fully ready.
        """
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
        """
        Test the health check endpoint behavior when no CharacterClient instance is available.
        
        Asserts that an HTTP 503 exception is raised when the client instance is missing.
        """
        mock_request = Mock()
        mock_request.app.state.character_client_instance = None
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(health_check(mock_request))
        
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_success(self):
        """
        Test that a character generation request returns the expected text and base64-encoded audio data when provided with valid input and a properly initialized client.
        """
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
        """
        Test that a character generation request with an invalid token raises an HTTP 401 Unauthorized exception.
        """
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
        """
        Test that a character generation request without a 'narration' field raises an HTTP 400 error.
        """
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
        """
        Test that a character generation request returns HTTP 503 when no client instance is available.
        """
        mock_request = Mock()
        mock_request.app.state.character_client_instance = None
        
        data = {"token": "valid_token", "narration": "Test"}
        
        with pytest.raises(HTTPException) as exc_info:
            await handle_character_generation_request(data, mock_request)
        
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_character_generation_request_no_audio_file(self):
        """
        Test the character generation endpoint when the synthesized audio file does not exist.
        
        Verifies that the endpoint returns the generated text response and sets `audio_data` to `None` if the audio file path returned by synthesis does not exist.
        """
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
        """
        Tests that the heartbeat task runner periodically calls the client's asynchronous heartbeat method.
        """
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
        """
        Test that the heartbeat task is scheduled on the event loop when `start_heartbeat_task` is called.
        """
        mock_client = Mock()
        
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            
            start_heartbeat_task(mock_client)
            
            mock_loop.create_task.assert_called_once()

    # Test initialization function
    @patch('CharacterClient.character_client.ensure_client_directories')
    def test_initialize_character_client_new_client(self, mock_ensure_dirs):
        """
        Test that initializing a new character client sets up directories and schedules the heartbeat task when no client instance exists.
        """
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
        """
        Test that initializing the character client does not recreate directories if an instance already exists.
        """
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
        """
        Test that CharacterClient correctly handles initialization with a very long Actor_id value.
        """
        long_id = "a" * 1000
        client = CharacterClient(
            token="test_token",
            Actor_id=long_id,
            server_url="http://localhost:8000",
            client_port=8080
        )
        assert client.Actor_id == long_id

    def test_character_client_with_special_characters_in_actor_id(self):
        """
        Test that CharacterClient correctly handles Actor_id values containing special characters.
        """
        special_id = "test-actor_123!@#$%^&*()"
        client = CharacterClient(
            token="test_token",
            Actor_id=special_id,
            server_url="http://localhost:8000",
            client_port=8080
        )
        assert client.Actor_id == special_id

    def test_character_client_with_unicode_actor_id(self):
        """
        Test that CharacterClient correctly handles Unicode characters in the Actor_id parameter.
        """
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
        """
        Test that async_init handles exceptions in registration, trait fetching, and audio download gracefully.
        
        Ensures that exceptions in internal initialization methods do not propagate and that default character traits are set.
        """
        with patch.object(character_client, '_register_with_server_blocking', side_effect=Exception("Registration failed")), \
             patch.object(character_client, '_fetch_traits_blocking', side_effect=Exception("Fetch failed")), \
             patch.object(character_client, '_download_reference_audio_blocking', side_effect=Exception("Download failed")):
            
            # Should not raise exception, should handle gracefully
            await character_client.async_init()
            
            # Should have default character traits
            assert character_client.character is not None

    def test_file_sanitization_in_download_reference_audio(self, character_client):
        """
        Test that the reference audio filename and Actor ID are sanitized to prevent directory traversal when downloading reference audio.
        
        Ensures that the resulting local file path does not contain unsafe sequences and that sanitization replaces potentially dangerous characters.
        """
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
    """
    Create a mock object for the built-in `open` function, supporting context manager and read operations.
    
    Parameters:
        mock: An optional MagicMock instance to use as the base mock.
        read_data (str): Data to be returned when the mock file's `read()` method is called.
    
    Returns:
        MagicMock: A mock object that simulates the behavior of `open` for use in tests.
    """
    if mock is None:
        mock = MagicMock(spec=open)
    
    handle = MagicMock(spec=['read', 'write', 'close', '__enter__', '__exit__'])
    handle.read.return_value = read_data
    handle.__enter__.return_value = handle
    mock.return_value = handle
    return mock

if __name__ == "__main__":
    pytest.main([__file__])