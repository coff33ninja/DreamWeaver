import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock

# Ensure correct import path for TTSManager from src/
from src.tts_manager import TTSManager
from SERVER.src.config import MODELS_PATH, TTS_MODELS_PATH as SERVER_TTS_MODELS_PATH # Renamed to avoid clash if used directly
import logging # Import logging for patching getLogger

# Use SERVER_TTS_MODELS_PATH which is derived from SERVER.src.config.MODELS_PATH
# This is where the TTSManager will attempt to create subdirectories.

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

    with patch("os.makedirs") as mock_os_makedirs:
        with patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(TTSManager, "_get_or_download_model_blocking", return_value="mock_xtts_model"):
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

        with patch("os.makedirs"), patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(TTSManager, "_get_or_download_model_blocking", return_value="gpu_model_id"):
                manager = TTSManager(tts_service_name="xttsv2", model_name="gpu_model_id")
                assert manager.is_initialized
                mock_coqui_tts_instance.to.assert_called_once_with("cuda")

    def test_init_xttsv2_failure_lib_missing(self, monkeypatch):
        monkeypatch.setattr("SERVER.src.tts_manager.CoquiTTS", None)
        with patch("os.makedirs"), patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                with patch.object(TTSManager, "_get_or_download_model_blocking", return_value="any_model"):
                    manager = TTSManager(tts_service_name="xttsv2", model_name="any_model")
                    assert not manager.is_initialized
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

class TestServerTTSManagerPrivateHelpers:
    @patch("SERVER.src.tts_manager.os.makedirs")
    def test_get_or_download_model_blocking(self, mock_makedirs, monkeypatch):
        monkeypatch.setattr("SERVER.src.tts_manager.gtts", MagicMock())
        with patch.dict(os.environ, {"TTS_HOME": MODELS_PATH}):
            manager = TTSManager(tts_service_name="gtts")

        result_xtts = manager._get_or_download_model_blocking("xttsv2", "server_xtts_model")
        assert result_xtts == "server_xtts_model"
        mock_makedirs.assert_any_call(os.path.join(SERVER_TTS_MODELS_PATH, "xttsv2"), exist_ok=True)

        result_gtts = manager._get_or_download_model_blocking("gtts", "en")
        assert result_gtts is None # gTTS returns None here as per implementation
        mock_makedirs.assert_any_call(os.path.join(SERVER_TTS_MODELS_PATH, "gtts"), exist_ok=True)

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
