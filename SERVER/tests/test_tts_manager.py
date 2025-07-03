import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock

# Corrected import path for TTSManager from SERVER.src/
from SERVER.src.tts_manager import TTSManager, TTS_MODELS_PATH
from SERVER.src.config import MODELS_PATH # TTS_MODELS_PATH is now imported from tts_manager
import logging # Import logging for patching getLogger

# Use TTS_MODELS_PATH imported from SERVER.src.tts_manager
# This is where the TTSManager will attempt to create subdirectories.

# Get a logger instance, similar to how it's done in tts_manager.py
logger = logging.getLogger("dreamweaver_server_tests") # Use a test-specific logger name


@pytest.fixture
def server_tts_manager_gtts_success(monkeypatch):
    """Fixture for server TTSManager with gTTS mocked for success."""
    mock_gtts_lib = MagicMock()
    monkeypatch.setattr("SERVER.src.tts_manager.gtts", mock_gtts_lib)

    # Mock os.makedirs called during __init__
    with patch("os.makedirs") as mock_os_makedirs:
        # Mock os.environ if TTS_HOME is critical for gTTS part (it's more for Coqui)
        with patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
             manager = TTSManager(tts_service_name="gtts", language="en")
    return manager, mock_gtts_lib, mock_os_makedirs

@pytest.fixture
def server_tts_manager_xttsv2_success(monkeypatch):
    """Fixture for server TTSManager with CoquiTTS mocked for success."""
    mock_coqui_tts_lib = MagicMock()
    mock_coqui_tts_instance = MagicMock()
    mock_coqui_tts_lib.return_value = mock_coqui_tts_instance
    monkeypatch.setattr("SERVER.src.tts_manager.CoquiTTS", mock_coqui_tts_lib)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False # Assume CPU
    monkeypatch.setattr("SERVER.src.tts_manager.torch", mock_torch)

    # Import ModelManager here for patching if not already globally imported for tests
    from SERVER.src.model_manager import ModelManager

    with patch("os.makedirs") as mock_os_makedirs:
        with patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            # Patching the method on the ModelManager class
            with patch.object(ModelManager, "get_or_prepare_tts_model_path", return_value="mock_xtts_model") as mock_get_path:
                manager = TTSManager(
                    tts_service_name="xttsv2",
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2", # Example model
                    language="en"
                )
    return manager, mock_coqui_tts_lib, mock_coqui_tts_instance, mock_os_makedirs


class TestServerTTSManagerInitialization:
    def test_init_gtts_service_success(self, server_tts_manager_gtts_success):
        manager, mock_gtts_lib, _ = server_tts_manager_gtts_success
        assert manager.service_name == "gtts"
        assert manager.language == "en"
        assert manager.is_initialized
        assert manager.tts_instance == manager._gtts_synthesize_blocking

    def test_init_gtts_service_failure_lib_missing(self, monkeypatch):
        monkeypatch.setattr("SERVER.src.tts_manager.gtts", None)
        with patch("os.makedirs"), patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                manager = TTSManager(tts_service_name="gtts")
                assert not manager.is_initialized
                mock_log_instance.error.assert_any_call(
                    "Server TTSManager: Error - gTTS service selected but gtts library not found."
                )

    def test_init_xttsv2_service_success_cpu(self, server_tts_manager_xttsv2_success, monkeypatch):
        manager, mock_coqui_tts_lib, mock_coqui_tts_instance, _ = server_tts_manager_xttsv2_success
        assert manager.service_name == "xttsv2"
        assert manager.is_initialized
        mock_coqui_tts_lib.assert_called_once_with(
            model_name="mock_xtts_model", progress_bar=True
        )
        mock_coqui_tts_instance.to.assert_called_once_with("cpu")

    def test_init_xttsv2_service_success_gpu(self, monkeypatch):
        mock_coqui_tts_lib = MagicMock()
        mock_coqui_tts_instance = MagicMock()
        mock_coqui_tts_lib.return_value = mock_coqui_tts_instance
        monkeypatch.setattr("SERVER.src.tts_manager.CoquiTTS", mock_coqui_tts_lib)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True # Simulate GPU
        monkeypatch.setattr("SERVER.src.tts_manager.torch", mock_torch)

        # Import ModelManager here for patching
        from SERVER.src.model_manager import ModelManager
        with patch("os.makedirs"), patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(ModelManager, "get_or_prepare_tts_model_path", return_value="gpu_model_id"):
                manager = TTSManager(tts_service_name="xttsv2", model_name="gpu_model_id")
                assert manager.is_initialized
                mock_coqui_tts_instance.to.assert_called_once_with("cuda")

    def test_init_xttsv2_failure_lib_missing(self, monkeypatch):
        monkeypatch.setattr("SERVER.src.tts_manager.CoquiTTS", None)
        # Import ModelManager here for patching
        from SERVER.src.model_manager import ModelManager
        with patch("os.makedirs"), patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                # Even if CoquiTTS is missing, the call to ModelManager would happen if model_name is provided
                # No, if CoquiTTS is None, and service is xttsv2, it logs error and returns early before ModelManager call.
                # The patch for get_or_prepare_tts_model_path is not strictly needed here if that's the expected flow.
                # However, if there was a model_name, it would have called it.
                # Let's keep it for safety, or verify the exact execution path.
                # Upon review of TTSManager._initialize_service:
                # If self.service_name == "xttsv2", it proceeds to call ModelManager.
                # Then, it checks if CoquiTTS is available.
                # So, the ModelManager.get_or_prepare_tts_model_path would be called.
                with patch.object(ModelManager, "get_or_prepare_tts_model_path", return_value="any_model") as mock_prepare_path:
                    manager = TTSManager(tts_service_name="xttsv2", model_name="any_model")
                    assert not manager.is_initialized
                    # mock_prepare_path.assert_called_once() # Verify it was called
                    mock_log_instance.error.assert_any_call(
                        "Server TTSManager: Error - Coqui TTS library not available for XTTSv2."
                    )

    def test_init_unsupported_service(self, monkeypatch):
        monkeypatch.setattr("SERVER.src.tts_manager.gtts", None)
        monkeypatch.setattr("SERVER.src.tts_manager.CoquiTTS", None)
        with patch("os.makedirs"), patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                manager = TTSManager(tts_service_name="fake_service")
                assert not manager.is_initialized
                mock_log_instance.error.assert_any_call(
                    "Server TTSManager: Unsupported TTS service 'fake_service'."
                )

@pytest.mark.asyncio
class TestServerTTSManagerSynthesizeAsync:

    @patch("SERVER.src.tts_manager.os.makedirs")
    @patch("SERVER.src.tts_manager.os.path.exists")
    @patch("SERVER.src.tts_manager.os.path.getsize")
    @patch("SERVER.src.tts_manager.asyncio.to_thread")
    async def test_synthesize_gtts_success(self, mock_to_thread, mock_getsize, mock_exists, mock_makedirs, server_tts_manager_gtts_success):
        manager, _, _ = server_tts_manager_gtts_success
        mock_exists.return_value = True
        mock_getsize.return_value = 100

        test_text = "Hello gTTS Server"
        output_path = "/server_tmp/test_gtts.mp3" # Example path

        result_ok = await manager.synthesize(test_text, output_path)

        assert result_ok
        mock_to_thread.assert_called_once_with(
            manager._gtts_synthesize_blocking, test_text, output_path, manager.language
        )
        mock_makedirs.assert_called_with(os.path.dirname(output_path), exist_ok=True)

    @patch("SERVER.src.tts_manager.os.makedirs")
    @patch("SERVER.src.tts_manager.os.path.exists")
    @patch("SERVER.src.tts_manager.os.path.getsize")
    @patch("SERVER.src.tts_manager.asyncio.to_thread")
    async def test_synthesize_xttsv2_success(self, mock_to_thread, mock_getsize, mock_exists, mock_makedirs, server_tts_manager_xttsv2_success):
        manager, _, _, _ = server_tts_manager_xttsv2_success
        mock_exists.return_value = True
        mock_getsize.return_value = 100

        test_text = "Hello XTTS Server"
        output_path = "/server_tmp/test_xtts.wav"
        speaker_wav = "/path/to/server_speaker.wav"

        result_ok = await manager.synthesize(test_text, output_path, speaker_wav_for_synthesis=speaker_wav)

        assert result_ok
        mock_to_thread.assert_called_once_with(
            manager._xttsv2_synthesize_blocking, test_text, output_path, speaker_wav, manager.language
        )
        mock_makedirs.assert_called_with(os.path.dirname(output_path), exist_ok=True)

    async def test_synthesize_not_initialized_raises_error(self):
        manager = TTSManager.__new__(TTSManager)
        manager.is_initialized = False
        manager.tts_instance = None
        manager.service_name = "test_service_server"

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_log_instance = MagicMock()
            mock_get_logger.return_value = mock_log_instance
            with pytest.raises(RuntimeError, match=r"TTSManager \(test_service_server\) is not initialized."):
                await manager.synthesize("test", "output.wav")
            # Log message for this specific path
            mock_log_instance.error.assert_any_call(
                 f"Server TTSManager (test_service_server): Not initialized, cannot synthesize text: 'test'."
            )

    @patch("SERVER.src.tts_manager.os.makedirs")
    @patch("SERVER.src.tts_manager.os.path.exists", return_value=False)
    @patch("SERVER.src.tts_manager.asyncio.to_thread")
    async def test_synthesize_output_file_missing(self, mock_to_thread, mock_exists, mock_makedirs, server_tts_manager_gtts_success):
        manager, _, _ = server_tts_manager_gtts_success

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_log_instance = MagicMock()
            mock_get_logger.return_value = mock_log_instance
            result_ok = await manager.synthesize("test", "/server_tmp/output.mp3")
            assert not result_ok
            mock_log_instance.error.assert_any_call(
                f"Server TTSManager: Synthesis completed but output file /server_tmp/output.mp3 is missing or empty."
            )

    @patch("SERVER.src.tts_manager.os.makedirs")
    @patch("SERVER.src.tts_manager.os.path.exists")
    @patch("SERVER.src.tts_manager.os.remove")
    @patch("SERVER.src.tts_manager.asyncio.to_thread", side_effect=Exception("TTS lib error server"))
    async def test_synthesize_handles_exception_and_cleans_up(self, mock_to_thread, mock_remove, mock_exists, mock_makedirs, server_tts_manager_gtts_success):
        manager, _, _ = server_tts_manager_gtts_success
        mock_exists.return_value = True

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_log_instance = MagicMock()
            mock_get_logger.return_value = mock_log_instance
            result_ok = await manager.synthesize("test", "/server_tmp/error_output.mp3")
            assert not result_ok
            mock_log_instance.error.assert_any_call(
                f"Server TTSManager: Error during async TTS synthesis with gtts for text 'test...': TTS lib error server",
                exc_info=True
            )
            mock_remove.assert_called_once_with("/server_tmp/error_output.mp3")


class TestServerTTSManagerStaticMethods:
    @patch("SERVER.src.tts_manager.gtts", MagicMock())
    @patch("SERVER.src.tts_manager.CoquiTTS", MagicMock())
    def test_list_services_both_available(self):
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" in services

    @patch("SERVER.src.tts_manager.gtts", None)
    @patch("SERVER.src.tts_manager.CoquiTTS", MagicMock())
    def test_list_services_only_xttsv2_available(self):
        services = TTSManager.list_services()
        assert "gtts" not in services
        assert "xttsv2" in services

    def test_get_available_models(self):
        assert TTSManager.get_available_models("gtts") == ["N/A (uses language codes)"]
        assert TTSManager.get_available_models("xttsv2") == ["tts_models/multilingual/multi-dataset/xtts_v2"]
        assert TTSManager.get_available_models("unknown_service") == []

# TestServerTTSManagerPrivateHelpers is removed as _get_or_download_model_blocking was moved to ModelManager.
# Tests for ModelManager.get_or_prepare_tts_model_path will be in test_model_manager.py.

class TestServerTTSManagerInternalBlockingMethods: # Renamed for clarity
    @patch("SERVER.src.tts_manager.gtts")
    def test_gtts_synthesize_blocking(self, mock_gtts_lib, server_tts_manager_gtts_success):
        manager, _, _ = server_tts_manager_gtts_success
        mock_gtts_obj = MagicMock()
        mock_gtts_lib.gTTS.return_value = mock_gtts_obj

        manager._gtts_synthesize_blocking("server text", "/srv_path", "srv_lang")
        mock_gtts_lib.gTTS.assert_called_once_with(text="server text", lang="srv_lang")
        mock_gtts_obj.save.assert_called_once_with("/srv_path")

    @patch("SERVER.src.tts_manager.os.path.exists", return_value=True)
    def test_xttsv2_synthesize_blocking_with_speaker(self, mock_os_exists, server_tts_manager_xttsv2_success):
        manager, _, mock_coqui_tts_instance, _ = server_tts_manager_xttsv2_success
        manager.tts_instance = mock_coqui_tts_instance
        manager.speaker_wav_path = "/srv_valid/speaker.wav"

        manager._xttsv2_synthesize_blocking("srv text", "/srv_path", speaker_wav="/srv_explicit/speaker.wav", lang="srv_lang")
        mock_coqui_tts_instance.tts_to_file.assert_called_with(text="srv text", speaker_wav="/srv_explicit/speaker.wav", language="srv_lang", file_path="/srv_path")

        manager._xttsv2_synthesize_blocking("srv text2", "/srv_path2", speaker_wav=None, lang="srv_lang2")
        mock_coqui_tts_instance.tts_to_file.assert_called_with(text="srv text2", speaker_wav="/srv_valid/speaker.wav", language="srv_lang2", file_path="/srv_path2")

# Basic logging setup for pytest output if needed
logging.basicConfig(level=logging.DEBUG)


# --- Functional Test for TTSManager Synthesis ---
@pytest.mark.asyncio
async def test_functional_tts_manager_synthesis(tmp_path):
    """
    Asynchronously tests the TTSManager with available TTS services and saves synthesized audio outputs.
    This is a functional test that may download models and perform real synthesis.
    It uses tmp_path fixture from pytest for temporary output directory.
    """
    # Configure logging for the test
    test_logger = logging.getLogger("dreamweaver_server_functional_test")
    test_logger.info("--- Server TTSManager Async Functional Test ---")

    # Ensure MODELS_PATH (from server config) and subdirs are writable by the actual TTSManager
    # The TTSManager itself will handle os.makedirs for TTS_MODELS_PATH and its subdirectories.
    # For this test, we'll direct specific outputs to a pytest-managed temporary directory.
    functional_test_output_dir = tmp_path / "test_outputs_server_async_functional"
    os.makedirs(functional_test_output_dir, exist_ok=True)
    test_logger.info(f"Functional test output directory: {functional_test_output_dir}")

    # Test gTTS
    if "gtts" in TTSManager.list_services():
        test_logger.info("Testing gTTS (async functional)...")
        tts_g = TTSManager(tts_service_name="gtts", language="es")
        if tts_g.is_initialized:
            out_g = functional_test_output_dir / "server_gtts_async_functional_test.mp3"
            if await tts_g.synthesize("Hola mundo funcional desde el servidor.", str(out_g)):
                test_logger.info(f"gTTS async functional test audio saved to {out_g}")
                assert os.path.exists(out_g)
                assert os.path.getsize(out_g) > 0
            else:
                test_logger.error("gTTS async functional test synthesis failed.")
                pytest.fail("gTTS functional synthesis failed.")
        else:
            test_logger.warning("gTTS manager not initialized, skipping functional test.")
            # Depending on strictness, you might want to fail here if gTTS is expected to work
            # pytest.fail("gTTS manager failed to initialize for functional test.")

    # Test XTTSv2 (CoquiTTS)
    # This test might be slow as it can download models if not cached.
    # Consider marking as slow or integration if it becomes an issue.
    if "xttsv2" in TTSManager.list_services():
        test_logger.info("Testing XTTSv2 (async functional)...")
        # XTTS model will be downloaded by Coqui library to server's MODELS_PATH/tts_models/
        # if not already present.
        try:
            tts_x = TTSManager(
                tts_service_name="xttsv2",
                model_name="tts_models/multilingual/multi-dataset/xtts_v2", # A common Coqui model
                language="en",
            )
            if tts_x.is_initialized:
                out_x = functional_test_output_dir / "server_xtts_async_functional_test.wav"
                test_phrase = "Hello from server-side Coqui XTTS, this is an asynchronous functional test."
                if await tts_x.synthesize(test_phrase, str(out_x)):
                    test_logger.info(f"XTTSv2 async functional test audio saved to {out_x}")
                    assert os.path.exists(out_x)
                    assert os.path.getsize(out_x) > 0
                else:
                    test_logger.error("XTTSv2 async functional test synthesis failed.")
                    pytest.fail("XTTSv2 functional synthesis failed.")
            else:
                test_logger.error("XTTSv2 manager not initialized for functional test.")
                # This is a more critical failure if xttsv2 is expected
                pytest.fail("XTTSv2 manager failed to initialize for functional test.")
        except ImportError:
            test_logger.warning("Coqui TTS library not found, skipping XTTSv2 functional test.")
        except Exception as e:
            test_logger.error(f"An error occurred during XTTSv2 functional test setup or synthesis: {e}", exc_info=True)
            pytest.fail(f"XTTSv2 functional test failed due to exception: {e}")
    else:
        test_logger.warning("XTTSv2 service not listed, skipping functional test.")

    test_logger.info("--- Server TTSManager Async Functional Test Complete ---")
