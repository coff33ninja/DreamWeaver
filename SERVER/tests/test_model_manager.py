import pytest
import os
from unittest.mock import patch, MagicMock

from SERVER.src.model_manager import ModelManager, TTS_MODELS_PATH # Import necessary items
from SERVER.src.config import MODELS_PATH # For setting TTS_HOME if necessary for context

# Basic logger for tests if needed, though ModelManager itself logs.
import logging
logger = logging.getLogger("dreamweaver_server_model_manager_tests")

class TestModelManagerUnit:

    @patch("SERVER.src.model_manager.os.makedirs")
    def test_get_or_prepare_tts_model_path_xttsv2(self, mock_makedirs):
        """Test that for xttsv2, the model identifier is returned and directory is created."""
        model_manager = ModelManager() # Not strictly needed since it's a static method, but good for consistency if it becomes instance-based
        service_name = "xttsv2"
        model_identifier = "tts_models/multilingual/multi-dataset/xtts_v2"

        expected_path = os.path.join(TTS_MODELS_PATH, service_name.lower())

        result = ModelManager.get_or_prepare_tts_model_path(service_name, model_identifier)

        assert result == model_identifier
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    @patch("SERVER.src.model_manager.os.makedirs")
    def test_get_or_prepare_tts_model_path_gtts(self, mock_makedirs):
        """Test that for gtts, None is returned and directory is created."""
        service_name = "gtts"
        model_identifier = "en" # gTTS uses language codes

        expected_path = os.path.join(TTS_MODELS_PATH, service_name.lower())

        result = ModelManager.get_or_prepare_tts_model_path(service_name, model_identifier)

        assert result is None
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    @patch("SERVER.src.model_manager.os.makedirs")
    def test_get_or_prepare_tts_model_path_other_service(self, mock_makedirs):
        """Test that for other services, None is returned and directory is created."""
        service_name = "some_other_tts"
        model_identifier = "some_model"

        expected_path = os.path.join(TTS_MODELS_PATH, service_name.lower())

        result = ModelManager.get_or_prepare_tts_model_path(service_name, model_identifier)

        assert result is None
        mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    def test_get_or_prepare_tts_model_path_empty_service_or_model(self):
        """Test that None is returned if service_name or model_identifier is empty."""
        assert ModelManager.get_or_prepare_tts_model_path("", "model_id") is None
        assert ModelManager.get_or_prepare_tts_model_path("xttsv2", "") is None
        assert ModelManager.get_or_prepare_tts_model_path("", "") is None

    @patch("SERVER.src.model_manager.os.makedirs", side_effect=OSError("Test OS Error"))
    def test_get_or_prepare_tts_model_path_makedirs_os_error(self, mock_makedirs_error):
        """Test that None is returned if os.makedirs raises an OSError."""
        service_name = "xttsv2"
        model_identifier = "tts_models/multilingual/multi-dataset/xtts_v2"

        # Expect an error log message
        with patch.object(logging.getLogger("dreamweaver_server"), 'error') as mock_log_error:
            result = ModelManager.get_or_prepare_tts_model_path(service_name, model_identifier)
            assert result is None
            mock_log_error.assert_called_once()
            # Check that the error message contains the path and the error
            args, _ = mock_log_error.call_args
            assert "Could not create directory" in args[0]
            assert "Test OS Error" in str(args[1]) # Check the exc_info

# To run these tests, navigate to the SERVER directory and run:
# python -m pytest tests/test_model_manager.py
# Ensure __init__.py is in SERVER/tests/ and SERVER/src/ for imports to work correctly.
