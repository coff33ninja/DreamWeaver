import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure correct import path for TTSManager from CharacterClient/src/
# This might require sys.path manipulation if tests are run from a different root
from CharacterClient.src.tts_manager import TTSManager
from CharacterClient.src.config import CLIENT_TTS_MODELS_PATH, CLIENT_TEMP_AUDIO_PATH


@pytest.fixture
def tts_manager_gtts_success(monkeypatch):
    """Fixture for TTSManager with gTTS mocked for success."""
    mock_gtts_lib = MagicMock()
    monkeypatch.setattr("CharacterClient.src.tts_manager.gtts", mock_gtts_lib)

    # Mock os.makedirs called during __init__ for CLIENT_TTS_MODELS_PATH/tts_models
    with patch("os.makedirs") as mock_os_makedirs:
        manager = TTSManager(tts_service_name="gtts", language="en")
    return manager, mock_gtts_lib, mock_os_makedirs

@pytest.fixture
def tts_manager_xttsv2_success(monkeypatch):
    """Fixture for TTSManager with CoquiTTS mocked for success."""
    mock_coqui_tts_lib = MagicMock()
    mock_coqui_tts_instance = MagicMock()
    mock_coqui_tts_lib.return_value = mock_coqui_tts_instance # Mock CoquiTTS() constructor call
    monkeypatch.setattr("CharacterClient.src.tts_manager.CoquiTTS", mock_coqui_tts_lib)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False # Assume CPU for simplicity
    monkeypatch.setattr("CharacterClient.src.tts_manager.torch", mock_torch)

    # Mock os.makedirs called during __init__
    with patch("os.makedirs") as mock_os_makedirs:
        # Mock _get_or_download_model_blocking to avoid actual model path logic for this unit test
        with patch.object(TTSManager, "_get_or_download_model_blocking", return_value="mock_model_path_or_name"):
            manager = TTSManager(
                tts_service_name="xttsv2",
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                language="en"
            )
    return manager, mock_coqui_tts_lib, mock_coqui_tts_instance, mock_os_makedirs


class TestTTSManagerInitialization:
    def test_init_gtts_service_success(self, tts_manager_gtts_success):
        manager, mock_gtts_lib, _ = tts_manager_gtts_success
        assert manager.service_name == "gtts"
        assert manager.language == "en"
        assert manager.is_initialized
        assert manager.tts_instance == manager._gtts_synthesize_blocking
        mock_gtts_lib.gTTS.assert_not_called() # Actual gTTS object is not created, only method stored

    def test_init_gtts_service_failure_lib_missing(self, monkeypatch):
        monkeypatch.setattr("CharacterClient.src.tts_manager.gtts", None)
        with patch("os.makedirs"):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                manager = TTSManager(tts_service_name="gtts")
                assert not manager.is_initialized
                mock_log_instance.error.assert_any_call(
                    "Client TTSManager: Error - gTTS library not found. gTTS service unavailable."
                )

    def test_init_xttsv2_service_success_cpu(self, tts_manager_xttsv2_success, monkeypatch):
        manager, mock_coqui_tts_lib, mock_coqui_tts_instance, _ = tts_manager_xttsv2_success

        assert manager.service_name == "xttsv2"
        assert manager.is_initialized
        mock_coqui_tts_lib.assert_called_once_with(
            model_name="mock_model_path_or_name", progress_bar=True
        )
        mock_coqui_tts_instance.to.assert_called_once_with("cpu")

    def test_init_xttsv2_service_success_gpu(self, monkeypatch):
        mock_coqui_tts_lib = MagicMock()
        mock_coqui_tts_instance = MagicMock()
        mock_coqui_tts_lib.return_value = mock_coqui_tts_instance
        monkeypatch.setattr("CharacterClient.src.tts_manager.CoquiTTS", mock_coqui_tts_lib)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True # Simulate GPU
        monkeypatch.setattr("CharacterClient.src.tts_manager.torch", mock_torch)

        with patch("os.makedirs"):
            with patch.object(TTSManager, "_get_or_download_model_blocking", return_value="gpu_model"):
                manager = TTSManager(tts_service_name="xttsv2", model_name="gpu_model")
                assert manager.is_initialized
                mock_coqui_tts_instance.to.assert_called_once_with("cuda")

    def test_init_xttsv2_failure_lib_missing(self, monkeypatch):
        monkeypatch.setattr("CharacterClient.src.tts_manager.CoquiTTS", None)
        with patch("os.makedirs"):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                with patch.object(TTSManager, "_get_or_download_model_blocking", return_value="any_model"):
                    manager = TTSManager(tts_service_name="xttsv2", model_name="any_model")
                    assert not manager.is_initialized
                    mock_log_instance.error.assert_any_call(
                        "Client TTSManager: Error - Coqui TTS library not found. XTTSv2 service unavailable."
                    )

    def test_init_unsupported_service(self, monkeypatch):
        monkeypatch.setattr("CharacterClient.src.tts_manager.gtts", None) # Ensure gtts is not picked by chance
        monkeypatch.setattr("CharacterClient.src.tts_manager.CoquiTTS", None) # Ensure xtts is not picked
        with patch("os.makedirs"):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                manager = TTSManager(tts_service_name="fake_service")
                assert not manager.is_initialized
                mock_log_instance.error.assert_any_call(
                    "Client TTSManager: Unsupported TTS service 'fake_service'."
                )

    def test_init_xttsv2_no_model_name_logs_warning(self, monkeypatch):
        mock_coqui_tts_lib = MagicMock() # Assume CoquiTTS lib exists
        monkeypatch.setattr("CharacterClient.src.tts_manager.CoquiTTS", mock_coqui_tts_lib)
        mock_torch = MagicMock()
        monkeypatch.setattr("CharacterClient.src.tts_manager.torch", mock_torch)

        with patch("os.makedirs"):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_log_instance = MagicMock()
                mock_get_logger.return_value = mock_log_instance
                # _get_or_download_model_blocking will be called with model_name=""
                # and for xttsv2, if it returns None/empty, it should error out.
                with patch.object(TTSManager, "_get_or_download_model_blocking", return_value=""):
                     manager = TTSManager(tts_service_name="xttsv2", model_name="") # No model name
                     mock_log_instance.warning.assert_any_call(
                        "Client TTSManager: No model name specified for service 'xttsv2'. Initialization may fail or use defaults."
                     )
                     # It should then fail because _get_or_download_model_blocking result is empty
                     mock_log_instance.error.assert_any_call(
                        "Client TTSManager: Could not determine model path or name for '' for service 'xttsv2'. Cannot initialize."
                     )
                     assert not manager.is_initialized


@pytest.mark.asyncio
class TestTTSManagerSynthesizeAsync:

    @patch("CharacterClient.src.tts_manager.os.makedirs")
    @patch("CharacterClient.src.tts_manager.os.path.exists")
    @patch("CharacterClient.src.tts_manager.os.path.getsize")
    @patch("CharacterClient.src.tts_manager.asyncio.to_thread")
    async def test_synthesize_gtts_success(self, mock_to_thread, mock_getsize, mock_exists, mock_makedirs, tts_manager_gtts_success):
        manager, _, _ = tts_manager_gtts_success
        mock_exists.return_value = True
        mock_getsize.return_value = 100 # Non-empty file

        test_text = "Hello gTTS"
        output_filename = "test_gtts.mp3"
        expected_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, output_filename)

        result_path = await manager.synthesize(test_text, output_filename)

        assert result_path == expected_full_path
        mock_to_thread.assert_called_once_with(
            manager._gtts_synthesize_blocking, test_text, expected_full_path, manager.language
        )
        mock_makedirs.assert_called_with(os.path.dirname(expected_full_path), exist_ok=True)

    @patch("CharacterClient.src.tts_manager.os.makedirs")
    @patch("CharacterClient.src.tts_manager.os.path.exists")
    @patch("CharacterClient.src.tts_manager.os.path.getsize")
    @patch("CharacterClient.src.tts_manager.asyncio.to_thread")
    async def test_synthesize_xttsv2_success(self, mock_to_thread, mock_getsize, mock_exists, mock_makedirs, tts_manager_xttsv2_success):
        manager, _, _, _ = tts_manager_xttsv2_success
        mock_exists.return_value = True
        mock_getsize.return_value = 100

        test_text = "Hello XTTS"
        output_filename = "test_xtts.wav"
        speaker_wav = "/path/to/speaker.wav"
        expected_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, output_filename)

        result_path = await manager.synthesize(test_text, output_filename, speaker_wav_for_synthesis=speaker_wav)

        assert result_path == expected_full_path
        mock_to_thread.assert_called_once_with(
            manager._xttsv2_synthesize_blocking, test_text, expected_full_path, speaker_wav, manager.language
        )
        mock_makedirs.assert_called_with(os.path.dirname(expected_full_path), exist_ok=True)

    async def test_synthesize_not_initialized(self):
        manager = TTSManager.__new__(TTSManager) # Create without calling __init__
        manager.is_initialized = False
        manager.tts_instance = None
        manager.service_name = "test_service" # for logging

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_log_instance = MagicMock()
            mock_get_logger.return_value = mock_log_instance
            result = await manager.synthesize("test", "output.wav")
            assert result is None
            mock_log_instance.error.assert_any_call(
                f"Client TTSManager (test_service): Not initialized. Cannot synthesize text: 'test'."
            )

    @patch("CharacterClient.src.tts_manager.os.makedirs")
    @patch("CharacterClient.src.tts_manager.os.path.exists", return_value=False) # File does not exist after synthesis
    @patch("CharacterClient.src.tts_manager.asyncio.to_thread")
    async def test_synthesize_output_file_missing(self, mock_to_thread, mock_exists, mock_makedirs, tts_manager_gtts_success):
        manager, _, _ = tts_manager_gtts_success

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_log_instance = MagicMock()
            mock_get_logger.return_value = mock_log_instance
            result_path = await manager.synthesize("test", "output.mp3")
            assert result_path is None
            expected_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, "output.mp3")
            mock_log_instance.error.assert_any_call(
                f"Client TTSManager: Synthesis completed but output file {expected_full_path} is missing or empty."
            )

    @patch("CharacterClient.src.tts_manager.os.makedirs")
    @patch("CharacterClient.src.tts_manager.os.path.exists")
    @patch("CharacterClient.src.tts_manager.os.remove")
    @patch("CharacterClient.src.tts_manager.asyncio.to_thread", side_effect=Exception("TTS library error"))
    async def test_synthesize_handles_exception_and_cleans_up(self, mock_to_thread, mock_remove, mock_exists, mock_makedirs, tts_manager_gtts_success):
        manager, _, _ = tts_manager_gtts_success
        mock_exists.return_value = True # Assume file was created before error, or to test cleanup path

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_log_instance = MagicMock()
            mock_get_logger.return_value = mock_log_instance
            result_path = await manager.synthesize("test", "error_output.mp3")
            assert result_path is None
            mock_log_instance.error.assert_any_call(
                f"Client TTSManager: Error during async TTS synthesis with gtts for 'test...': TTS library error",
                exc_info=True
            )
            mock_remove.assert_called_once_with(os.path.join(CLIENT_TEMP_AUDIO_PATH, "error_output.mp3"))


class TestTTSManagerStaticMethods:
    @patch("CharacterClient.src.tts_manager.gtts", MagicMock())
    @patch("CharacterClient.src.tts_manager.CoquiTTS", MagicMock())
    def test_list_services_both_available(self):
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" in services

    @patch("CharacterClient.src.tts_manager.gtts", None)
    @patch("CharacterClient.src.tts_manager.CoquiTTS", MagicMock())
    def test_list_services_only_xttsv2_available(self):
        services = TTSManager.list_services()
        assert "gtts" not in services
        assert "xttsv2" in services

    def test_get_available_models(self):
        assert TTSManager.get_available_models("gtts") == ["N/A (uses language codes)"]
        assert TTSManager.get_available_models("xttsv2") == ["tts_models/multilingual/multi-dataset/xtts_v2"]
        assert TTSManager.get_available_models("unknown_service") == []

# Basic test for _get_or_download_model_blocking
# More detailed tests would require mocking os.makedirs and os.getenv
class TestTTSManagerPrivateHelpers:
    @patch("CharacterClient.src.tts_manager.os.makedirs")
    def test_get_or_download_model_blocking(self, mock_makedirs, monkeypatch):
        # Need a TTSManager instance to call this method
        monkeypatch.setattr("CharacterClient.src.tts_manager.gtts", MagicMock()) # Mock gtts for init
        manager = TTSManager(tts_service_name="gtts") # Service doesn't matter much for this helper test

        # Test for xttsv2
        result_xtts = manager._get_or_download_model_blocking("xttsv2", "test_xtts_model")
        assert result_xtts == "test_xtts_model"
        mock_makedirs.assert_any_call(os.path.join(CLIENT_TTS_MODELS_PATH, "xttsv2"), exist_ok=True)

        # Test for gtts (should return the identifier as is, if not None)
        result_gtts = manager._get_or_download_model_blocking("gtts", "en")
        assert result_gtts == "en" # Returns the identifier
        mock_makedirs.assert_any_call(os.path.join(CLIENT_TTS_MODELS_PATH, "gtts"), exist_ok=True)

        # Test for other service (should return identifier)
        result_other = manager._get_or_download_model_blocking("other_service", "other_model")
        assert result_other == "other_model"
        mock_makedirs.assert_any_call(os.path.join(CLIENT_TTS_MODELS_PATH, "other_service"), exist_ok=True)

    # Tests for _gtts_synthesize_blocking and _xttsv2_synthesize_blocking
    # These are called via asyncio.to_thread, so their direct testing is part of
    # testing the public `synthesize` method correctly by mocking `to_thread`.
    # However, if direct unit tests are desired:

    @patch("CharacterClient.src.tts_manager.gtts")
    def test_gtts_synthesize_blocking(self, mock_gtts_lib, tts_manager_gtts_success):
        manager, _, _ = tts_manager_gtts_success
        mock_gtts_obj = MagicMock()
        mock_gtts_lib.gTTS.return_value = mock_gtts_obj

        manager._gtts_synthesize_blocking("text", "path", "lang")
        mock_gtts_lib.gTTS.assert_called_once_with(text="text", lang="lang")
        mock_gtts_obj.save.assert_called_once_with("path")

    @patch("CharacterClient.src.tts_manager.os.path.exists", return_value=True)
    def test_xttsv2_synthesize_blocking_with_speaker(self, mock_os_exists, tts_manager_xttsv2_success):
        manager, _, mock_coqui_tts_instance, _ = tts_manager_xttsv2_success

        # Ensure tts_instance is the Coqui mock, not the gTTS method
        manager.tts_instance = mock_coqui_tts_instance
        manager.speaker_wav_path = "/valid/speaker.wav" # Set on manager for one branch

        # Test with speaker_wav provided in call
        manager._xttsv2_synthesize_blocking("text", "path", speaker_wav="/explicit/speaker.wav", lang="lang_code")
        mock_coqui_tts_instance.tts_to_file.assert_called_with(text="text", speaker_wav="/explicit/speaker.wav", language="lang_code", file_path="path")

        # Test with speaker_wav from self.speaker_wav_path (when speaker_wav arg is None)
        manager._xttsv2_synthesize_blocking("text2", "path2", speaker_wav=None, lang="lang_code2")
        mock_coqui_tts_instance.tts_to_file.assert_called_with(text="text2", speaker_wav="/valid/speaker.wav", language="lang_code2", file_path="path2")

    @patch("CharacterClient.src.tts_manager.os.path.exists", return_value=False) # Speaker wav does not exist
    def test_xttsv2_synthesize_blocking_no_speaker_fallback(self, mock_os_exists, tts_manager_xttsv2_success):
        manager, _, mock_coqui_tts_instance, _ = tts_manager_xttsv2_success
        manager.tts_instance = mock_coqui_tts_instance
        manager.speaker_wav_path = "/invalid/speaker.wav"

        manager._xttsv2_synthesize_blocking("text", "path", lang="lang_code")
        # Should call without speaker_wav if path is invalid
        mock_coqui_tts_instance.tts_to_file.assert_called_with(text="text", language="lang_code", file_path="path")

# Configure logging for pytest capture if needed
import logging
logging.basicConfig(level=logging.DEBUG) # Or use pytest's caplog fixture
