import pytest
import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock, call
from datetime import datetime, timezone, timedelta

# Add CharacterClient/src to sys.path to allow importing character_client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from character_client import CharacterClient, app as fastapi_app, initialize_character_client
from config import CLIENT_TTS_REFERENCE_VOICES_PATH

# Mock external dependencies for CharacterClient
MockTTSManager = MagicMock()
MockLLMEngine = MagicMock()
MockRequestsPost = MagicMock() # For requests.post
MockRequestsGet = MagicMock()  # For requests.get
MockWebsocketsConnect = AsyncMock() # For websockets.connect

# --- Fixtures ---
@pytest.fixture(autouse=True) # Apply to all tests in this file
def reset_singleton_mocks():
    """Resets mocks that might be stateful if not reset."""
    MockTTSManager.reset_mock()
    MockLLMEngine.reset_mock()
    MockRequestsPost.reset_mock()
    MockRequestsGet.reset_mock()
    MockWebsocketsConnect.reset_mock()

    # Default successful response for requests.post
    mock_response_post = MagicMock()
    mock_response_post.raise_for_status.return_value = None
    mock_response_post.json.return_value = {}
    MockRequestsPost.return_value = mock_response_post

    # Default successful response for requests.get
    mock_response_get = MagicMock()
    mock_response_get.raise_for_status.return_value = None
    mock_response_get.json.return_value = {}
    MockRequestsGet.return_value = mock_response_get

    # Default successful websocket connection
    mock_ws_connection = AsyncMock()
    mock_ws_connection.close = AsyncMock()
    mock_ws_connection.recv = AsyncMock(side_effect=asyncio.TimeoutError) # Default to timeout on recv
    mock_ws_connection.ping = AsyncMock()
    MockWebsocketsConnect.return_value = mock_ws_connection

    # Mock instances returned by constructors for LLM and TTS
    # These are used when CharacterClient instantiates them.
    MockTTSManager.return_value = MagicMock(is_initialized=True, service_name="mock_tts", model_name="mock_model", language="en", speaker_wav_path=None)
    MockLLMEngine.return_value = MagicMock(is_initialized=True)


@pytest.fixture
def mock_external_classes(monkeypatch):
    monkeypatch.setattr("character_client.TTSManager", MockTTSManager)
    monkeypatch.setattr("character_client.LLMEngine", MockLLMEngine)

@pytest.fixture
def mock_network_calls(monkeypatch):
    monkeypatch.setattr("requests.post", MockRequestsPost)
    monkeypatch.setattr("requests.get", MockRequestsGet)
    monkeypatch.setattr("websockets.connect", MockWebsocketsConnect)


@pytest.fixture
def character_client_bare(mock_external_classes, mock_network_calls):
    """Provides a CharacterClient instance without calling async_init."""
    with patch.object(CharacterClient, "async_init", AsyncMock()) as mock_async_init_on_instance:
        client = CharacterClient(
            token="test_token",
            Actor_id="test_actor",
            server_url="http://testserver:8000",
            client_port=8001
        )
        # Ensure the instance's async_init is the mock, not the class's after patch is exited
        client.async_init = mock_async_init_on_instance
    return client

@pytest.fixture
async def character_client_initialized(character_client_bare):
    """Provides a CharacterClient instance that has run a mocked async_init."""
    client = character_client_bare

    # Restore actual async_init for this fixture's purpose, then mock its internal calls
    original_async_init = CharacterClient.async_init.__func__ # Get original unbound method

    # Mock blocking calls made via to_thread within the actual async_init
    with patch("asyncio.to_thread") as mock_to_thread:
        mock_register_result = True
        mock_traits_result = {
            "name": "TestActor", "personality": "tester", "tts": "gtts",
            "tts_model": "en", "reference_audio_filename": None, "language": "en", "llm_model": "tiny"
        }
        mock_download_ref_audio_result = None # Assume no download by default

        # Order of calls in async_init: _register, _fetch_traits, _download_reference_audio (conditionally)
        def to_thread_side_effect(func, *args, **kwargs):
            if func.__name__ == "_register_with_server_blocking": return mock_register_result
            if func.__name__ == "_fetch_traits_blocking": return mock_traits_result
            if func.__name__ == "_download_reference_audio_blocking": return mock_download_ref_audio_result
            raise ValueError(f"Unexpected function call in to_thread: {func.__name__}")
        mock_to_thread.side_effect = to_thread_side_effect

        # Mock async helper methods called by async_init
        client._perform_handshake_async = AsyncMock(return_value=None)
        client._connect_websocket_with_retry_async = AsyncMock(return_value=None)

        # Call the actual async_init
        await original_async_init(client)

    # Simulate successful initialization state
    client.is_initialized = True
    client.character = client.character # This should be set by async_init
    client.llm = MockLLMEngine.return_value # Ensure these are the mocked instances
    client.tts = MockTTSManager.return_value

    return client


# --- Test Classes ---

class TestCharacterClientInitializationAndFactory:
    def test_character_client_init(self, character_client_bare):
        client = character_client_bare
        assert client.token == "test_token"
        assert client.Actor_id == "test_actor"
        assert client.server_url == "http://testserver:8000"
        assert client.client_port == 8001
        assert client.session_token is None
        assert client.session_token_expiry is None
        assert client.ws_connection is None
        assert client.character is None
        assert client.tts is None # Set by async_init
        assert client.llm is None  # Set by async_init

    @patch("character_client.CharacterClient.async_init", new_callable=AsyncMock) # Patch on the class
    async def test_character_client_create_calls_async_init(self, mock_async_init_method_on_class, mock_external_classes, mock_network_calls):
        # When CharacterClient.create is called, it instantiates CharacterClient,
        # then calls async_init on that instance.
        # The mock_external_classes fixture ensures that when CharacterClient is instantiated,
        # its attempts to create LLMEngine/TTSManager use mocks.

        # We are patching CharacterClient.async_init at the class level.
        # So, when create() calls self.async_init(), it calls our mock.

        client_instance = await CharacterClient.create(
            token="create_token", Actor_id="create_actor", server_url="http://create", client_port=8002
        )

        # Assert that the async_init method *of the instance created by CharacterClient.create* was called.
        # mock_async_init_method_on_class refers to the mock on the class, not the instance.
        # This is a bit subtle. The patch replaces the method on the class.
        # When an instance is made, it gets this mocked method.
        # So, the assertion should be on the class-level mock.
        mock_async_init_method_on_class.assert_awaited_once()


@pytest.mark.asyncio
class TestCharacterClientAsyncInit:

    @patch("asyncio.to_thread")
    async def test_async_init_success_path(self, mock_to_thread, character_client_bare, mock_external_classes, mock_network_calls):
        client = character_client_bare # __init__ has run, async_init (mocked by fixture) has not.

        # Restore real async_init for this test, mocking its sub-components
        client.async_init = CharacterClient.async_init.__func__.__get__(client, CharacterClient)


        mock_register_result = True
        mock_traits_result = {
            "name": "FetchedActor", "personality": "fetched", "tts": "xttsv2",
            "tts_model": "xtts_model", "reference_audio_filename": "ref.wav",
            "language": "es", "llm_model": "fetched_llm"
        }
        mock_download_ref_audio_result = os.path.join(CLIENT_TTS_REFERENCE_VOICES_PATH, f"{client.Actor_id}_ref.wav")

        def to_thread_side_effect(func, *args, **kwargs):
            if func.__name__ == "_register_with_server_blocking": return mock_register_result
            if func.__name__ == "_fetch_traits_blocking": return mock_traits_result
            if func.__name__ == "_download_reference_audio_blocking": return mock_download_ref_audio_result
            pytest.fail(f"Unexpected function call in to_thread: {func.__name__}")
        mock_to_thread.side_effect = to_thread_side_effect

        client._perform_handshake_async = AsyncMock(return_value=None)
        # Simulate handshake providing a session token to trigger websocket connection
        async def mock_handshake_effect(): client.session_token = "fake_session_token"
        client._perform_handshake_async.side_effect = mock_handshake_effect
        client._connect_websocket_with_retry_async = AsyncMock(return_value=None)

        await client.async_init()

        assert client.character == mock_traits_result
        MockLLMEngine.assert_called_once_with(model_name="fetched_llm", Actor_id=client.Actor_id)
        MockTTSManager.assert_called_once_with(
            tts_service_name="xttsv2", model_name="xtts_model", language="es",
            speaker_wav_path=mock_download_ref_audio_result
        )
        assert client.llm == MockLLMEngine.return_value
        assert client.tts == MockTTSManager.return_value
        client._perform_handshake_async.assert_awaited_once()
        client._connect_websocket_with_retry_async.assert_awaited_once() # Called due to session token

    @patch("asyncio.to_thread")
    async def test_async_init_registration_fails(self, mock_to_thread, character_client_bare, mock_external_classes):
        client = character_client_bare
        client.async_init = CharacterClient.async_init.__func__.__get__(client, CharacterClient)

        mock_to_thread.side_effect = [False] # _register_with_server_blocking returns False
        client._perform_handshake_async = AsyncMock()

        logger = logging.getLogger("dreamweaver_client")
        with patch.object(logger, "critical") as mock_log_critical:
            await client.async_init()
            mock_log_critical.assert_any_call(
                f"CharacterClient ({client.Actor_id}): Failed to register with server after retries. Functionality may be severely impaired."
            )
        client._perform_handshake_async.assert_not_awaited()

@pytest.mark.asyncio
class TestCharacterClientHandshake:
    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_perform_handshake_async_success(self, mock_to_thread, character_client_bare, mock_network_calls):
        client = character_client_bare

        mock_challenge_resp_json = {"challenge": "test_challenge"}
        mock_session_resp_json = {"session_token": "test_session_token", "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}

        # Mock the return values of requests.post().json()
        mock_resp_challenge = MagicMock()
        mock_resp_challenge.json.return_value = mock_challenge_resp_json
        mock_resp_challenge.raise_for_status.return_value = None

        mock_resp_session = MagicMock()
        mock_resp_session.json.return_value = mock_session_resp_json
        mock_resp_session.raise_for_status.return_value = None

        mock_to_thread.side_effect = [mock_resp_challenge, mock_resp_session]

        await client._perform_handshake_async()

        assert client.session_token == mock_session_resp_json["session_token"]
        assert client.session_token_expiry == datetime.fromisoformat(mock_session_resp_json["expires_at"])

        assert MockRequestsPost.call_count == 2
        # Further assertions on MockRequestsPost.call_args_list can be added if needed

    @patch("asyncio.to_thread", new_callable=AsyncMock)
    async def test_perform_handshake_async_http_error(self, mock_to_thread, character_client_bare, mock_network_calls):
        client = character_client_bare

        # Simulate HTTPError by making requests.post raise it
        mock_http_error = requests.exceptions.HTTPError("HTTP Error")
        mock_http_error.response = MagicMock(status_code=500, text="Server Error")
        MockRequestsPost.side_effect = mock_http_error # requests.post itself will raise

        # mock_to_thread will call the mocked (raising) requests.post
        # Need to ensure that if the wrapped function raises, to_thread propagates it
        async def raising_side_effect(func, *args, **kwargs):
            return func(*args, **kwargs) # This will raise if func (requests.post) raises
        mock_to_thread.side_effect = raising_side_effect

        logger = logging.getLogger("dreamweaver_client")
        with patch.object(logger, "error") as mock_log_error:
            await client._perform_handshake_async()
            assert client.session_token is None
            mock_log_error.assert_any_call(
                f"CharacterClient ({client.Actor_id}): Handshake HTTP error 500 - Server Error", exc_info=True
            )

# Keep the existing TestHandleConfigUpdate class, ensuring it uses pytest-asyncio correctly
# For pytest, IsolatedAsyncioTestCase is not standard. We use @pytest.mark.asyncio
# and standard pytest fixtures.
@pytest.mark.asyncio
class TestHandleConfigUpdate: # Removed (IsolatedAsyncioTestCase)
    @pytest.fixture(autouse=True) # Auto-use this fixture for all methods in this class
    async def setup_client_for_config_update(self, monkeypatch): # Renamed from asyncSetUp
        # Patch LLM and TTS for CharacterClient's full instantiation via create()
        self.mock_llm_engine_patch = patch("character_client.LLMEngine", MagicMock())
        self.mock_tts_manager_patch = patch("character_client.TTSManager", MagicMock())

        self.MockLLMEngine_cls = self.mock_llm_engine_patch.start()
        self.MockTTSManager_cls = self.mock_tts_manager_patch.start()

        self.mock_llm_instance = self.MockLLMEngine_cls.return_value
        self.mock_llm_instance.is_initialized = True

        self.mock_tts_instance = self.MockTTSManager_cls.return_value
        self.mock_tts_instance.is_initialized = True
        self.mock_tts_instance.service_name = "gtts"
        self.mock_tts_instance.model_name = "en"
        self.mock_tts_instance.language = "en"
        self.mock_tts_instance.speaker_wav_path = None

        # Mock network calls that CharacterClient.create might trigger via async_init
        with patch("requests.post"), patch("requests.get"), patch("websockets.connect"):
             # Mock the internal blocking calls of async_init
            with patch.object(CharacterClient, "_register_with_server_blocking", return_value=True):
                with patch.object(CharacterClient, "_fetch_traits_blocking", return_value={"tts":"gtts", "llm_model":""}):
                    with patch.object(CharacterClient, "_download_reference_audio_blocking", return_value=None):
                        with patch.object(CharacterClient, "_perform_handshake_async", AsyncMock()):
                             with patch.object(CharacterClient, "_connect_websocket_with_retry_async", AsyncMock()):
                                self.client = await CharacterClient.create(
                                    token="test_token_cfg", Actor_id="test_actor_cfg",
                                    server_url="http://cfgtest.com", client_port=8089
                                )

        # Ensure the instance uses the right mocks after its own async_init
        self.client.tts = self.mock_tts_instance
        self.client.character = {"tts": "gtts", "tts_model": "en", "language": "en"} # Simplified for test
        self.client.local_reference_audio_path = None

        yield # For pytest fixture teardown

        self.mock_llm_engine_patch.stop()
        self.mock_tts_manager_patch.stop()

    @patch("logging.getLogger")
    async def test_log_level_update(self, mock_get_logger):
        payload = {"log_level": "DEBUG"}
        mock_logger_instance = mock_get_logger.return_value

        await self.client._handle_config_update(payload)
        mock_logger_instance.setLevel.assert_called_once_with(logging.DEBUG)

    async def test_invalid_log_level(self, caplog): # Use caplog fixture
        payload = {"log_level": "INVALID_LEVEL"}
        client_logger = logging.getLogger("dreamweaver_client") # Target specific logger

        with caplog.at_level(logging.WARNING, logger="dreamweaver_client"):
             await self.client._handle_config_update(payload)

        assert "Invalid log level received: INVALID_LEVEL" in caplog.text


    async def test_tts_reinitialization(self):
        payload = {
            "tts_service_name": "xttsv2", "tts_model_name": "new_model", "tts_language": "fr",
        }
        new_mock_tts_instance = MagicMock(is_initialized=True, service_name="xttsv2", model_name="new_model", language="fr")
        self.MockTTSManager_cls.return_value = new_mock_tts_instance # Next call to TTSManager() gets this

        await self.client._handle_config_update(payload)

        self.MockTTSManager_cls.assert_called_with(
            tts_service_name="xttsv2", model_name="new_model", language="fr", speaker_wav_path=None,
        )
        assert self.client.tts == new_mock_tts_instance

    async def test_no_tts_changes(self):
        payload = {"log_level": "INFO"}
        original_tts_instance = self.client.tts

        self.MockTTSManager_cls.reset_mock() # Reset from setup call

        await self.client._handle_config_update(payload)

        assert self.client.tts == original_tts_instance # Should be the same instance
        self.MockTTSManager_cls.assert_not_called() # Constructor should not be called again

if __name__ == "__main__":
    pytest.main(["-v", __file__])

# TODO: Add tests for:
# - _get_active_token (various session token states)
# - WebSocket methods: _connect_websocket_with_retry_async, _websocket_listener_async, close_websocket_async
# - send_heartbeat_async
# - generate_response_async (mock LLM)
# - synthesize_audio_async (mock TTS)
# - initialize_character_client (module level function)
# - FastAPI endpoints (requires TestClient) - these would be integration tests.
```
