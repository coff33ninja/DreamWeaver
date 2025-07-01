import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import torch
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, 'CharacterClient/src')
from tts_manager import TTSManager


class TestTTSManagerInitialization:
    """Test suite for TTSManager initialization scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts', None)
    @patch('tts_manager.CoquiTTS', None)
    def test_initialization_no_libraries_available(self, mock_ensure_dirs):
        """
        Test that TTSManager initializes with `is_initialized` set to False and no TTS instance when no TTS libraries are available.
        """
        manager = TTSManager(tts_service_name="gtts")
        assert manager.service_name == "gtts"
        assert manager.is_initialized is False
        assert manager.tts_instance is None
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_gtts_success(self, mock_gtts, mock_ensure_dirs):
        """
        Test that TTSManager initializes successfully with the gTTS service.
        
        Verifies that the service name, language, initialization status, and TTS instance are correctly set when gTTS is available.
        """
        manager = TTSManager(tts_service_name="gtts", language="en")
        assert manager.service_name == "gtts"
        assert manager.language == "en"
        assert manager.is_initialized is True
        assert callable(manager.tts_instance)
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts', None)
    def test_initialization_gtts_not_available(self, mock_ensure_dirs):
        """
        Test that TTSManager fails to initialize with gTTS when the gTTS library is not available.
        """
        manager = TTSManager(tts_service_name="gtts")
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('torch.cuda.is_available', return_value=True)
    def test_initialization_xttsv2_success_cuda(self, mock_cuda, mock_coqui, mock_ensure_dirs):
        """
        Test that XTTSv2 initializes successfully with CUDA available.
        
        Verifies that the TTSManager sets the correct service and model names, is marked as initialized, and moves the TTS instance to the CUDA device.
        """
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2", 
            model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        )
        
        assert manager.service_name == "xttsv2"
        assert manager.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
        assert manager.is_initialized is True
        mock_tts_instance.to.assert_called_with("cuda")
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('torch.cuda.is_available', return_value=False)
    def test_initialization_xttsv2_success_cpu(self, mock_cuda, mock_coqui, mock_ensure_dirs):
        """
        Test that XTTSv2 initializes successfully on CPU when CUDA is not available.
        
        Verifies that the TTS instance is moved to the CPU device during initialization.
        """
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        )
        
        mock_tts_instance.to.assert_called_with("cpu")
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_initialization_xttsv2_exception(self, mock_coqui, mock_ensure_dirs):
        """
        Test that XTTSv2 initialization sets `is_initialized` to False when an exception occurs during model loading.
        """
        mock_coqui.side_effect = Exception("Model loading failed")
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        )
        
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_no_model_name_for_xttsv2(self, mock_ensure_dirs):
        """
        Test that XTTSv2 initialization without a model name results in the TTSManager not being initialized.
        """
        manager = TTSManager(tts_service_name="xttsv2")
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_unsupported_service(self, mock_ensure_dirs):
        """
        Test that initializing TTSManager with an unsupported service name results in the manager not being initialized.
        """
        manager = TTSManager(tts_service_name="unsupported_service")
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_default_values(self, mock_ensure_dirs):
        """
        Test that TTSManager initializes with correct default values for model name, speaker wav path, and language when not explicitly provided.
        """
        manager = TTSManager(tts_service_name="gtts")
        assert manager.model_name == ""
        assert manager.speaker_wav_path == ""
        assert manager.language == "en"
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_none_language(self, mock_ensure_dirs):
        """
        Test that initializing TTSManager with a None language sets the language to the default value "en".
        """
        manager = TTSManager(tts_service_name="gtts", language=None)
        assert manager.language == "en"
        
    @patch('tts_manager.ensure_client_directories')
    @patch('os.makedirs')
    def test_initialization_creates_directories(self, mock_makedirs, mock_ensure_dirs):
        """
        Test that TTSManager initialization triggers creation of required directories.
        """
        TTSManager(tts_service_name="gtts")
        mock_ensure_dirs.assert_called_once()


class TestTTSManagerGTTSSynthesis:
    """Test suite for gTTS synthesis functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_gtts_synthesize_blocking_success(self, mock_gtts_module, mock_ensure_dirs):
        """
        Test that the blocking gTTS synthesis method successfully creates and saves an audio file for the given text and language.
        """
        mock_gtts_instance = Mock()
        mock_gtts_module.gTTS.return_value = mock_gtts_instance
        
        manager = TTSManager(tts_service_name="gtts", language="fr")
        manager._gtts_synthesize_blocking("Bonjour", "/path/to/output.mp3", "fr")
        
        mock_gtts_module.gTTS.assert_called_with(text="Bonjour", lang="fr")
        mock_gtts_instance.save.assert_called_with("/path/to/output.mp3")
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts', None)
    def test_gtts_synthesize_blocking_not_available(self, mock_ensure_dirs, capsys):
        """
        Test that gTTS synthesis outputs an appropriate message when the gTTS library is not available.
        """
        manager = TTSManager(tts_service_name="gtts")
        manager._gtts_synthesize_blocking("Hello", "/path/to/output.mp3", "en")
        
        captured = capsys.readouterr()
        assert "gTTS is not available" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_gtts_synthesize_blocking_no_gtts_class(self, mock_gtts_module, mock_ensure_dirs, capsys):
        """
        Test that gTTS synthesis outputs an error message when the gTTS class is missing from the gtts module.
        """
        # Mock gtts module without gTTS class
        delattr(mock_gtts_module, 'gTTS')
        
        manager = TTSManager(tts_service_name="gtts")
        manager._gtts_synthesize_blocking("Hello", "/path/to/output.mp3", "en")
        
        captured = capsys.readouterr()
        assert "gTTS is not available" in captured.out


class TestTTSManagerXTTSv2Synthesis:
    """Test suite for XTTSv2 synthesis functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('os.path.exists', return_value=True)
    def test_xttsv2_synthesize_blocking_with_speaker(self, mock_exists, mock_coqui, mock_ensure_dirs):
        """
        Test that XTTSv2 synthesis correctly uses a provided speaker WAV file and calls the synthesis method with the expected arguments.
        """
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en", "es", "fr"]
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model",
            speaker_wav_path="/path/to/speaker.wav"
        )
        
        manager._xttsv2_synthesize_blocking(
            "Hello world", 
            "/path/to/output.wav", 
            "/path/to/speaker.wav", 
            "en"
        )
        
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello world",
            speaker_wav="/path/to/speaker.wav",
            language="en",
            file_path="/path/to/output.wav"
        )
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('os.path.exists', return_value=False)
    def test_xttsv2_synthesize_blocking_no_speaker(self, mock_exists, mock_coqui, mock_ensure_dirs, capsys):
        """
        Test that XTTSv2 synthesis proceeds without a speaker wav file, issuing a warning and synthesizing with default voice.
        
        Verifies that when the specified speaker wav file does not exist, a warning is printed and synthesis is performed without the speaker wav parameter.
        """
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en", "es", "fr"]
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        manager._xttsv2_synthesize_blocking(
            "Hello world",
            "/path/to/output.wav",
            "/nonexistent/speaker.wav",
            "en"
        )
        
        captured = capsys.readouterr()
        assert "speaker_wav" in captured.out and "not found" in captured.out
        
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello world",
            language="en",
            file_path="/path/to/output.wav"
        )
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_synthesize_blocking_unsupported_language(self, mock_coqui, mock_ensure_dirs):
        """
        Test that XTTSv2 synthesis falls back to the first available language when an unsupported language code is provided.
        """
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en", "es", "fr"]
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        manager._xttsv2_synthesize_blocking(
            "Hello world",
            "/path/to/output.wav",
            None,
            "zh"  # Unsupported language
        )
        
        # Should fall back to first available language
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello world",
            language="en",  # First in languages list
            file_path="/path/to/output.wav"
        )
        
    @patch('tts_manager.ensure_client_directories')
    def test_xttsv2_synthesize_blocking_no_instance(self, mock_ensure_dirs, capsys):
        """
        Test that XTTSv2 synthesis outputs an error message when the TTS instance is not available.
        """
        manager = TTSManager(tts_service_name="xttsv2")
        manager.tts_instance = None
        
        manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav")
        
        captured = capsys.readouterr()
        assert "not available or invalid" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    def test_xttsv2_synthesize_blocking_invalid_instance(self, mock_ensure_dirs, capsys):
        """
        Test that XTTSv2 synthesis outputs an error message when the TTS instance is invalid.
        """
        manager = TTSManager(tts_service_name="xttsv2")
        manager.tts_instance = "invalid_instance"  # Not a proper TTS instance
        
        manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav")
        
        captured = capsys.readouterr()
        assert "not available or invalid" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_synthesize_blocking_none_language(self, mock_coqui, mock_ensure_dirs):
        """
        Test that XTTSv2 synthesis defaults to "en" when the language parameter is None.
        """
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en", "es"]
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        manager._xttsv2_synthesize_blocking(
            "Hello world",
            "/path/to/output.wav",
            None,
            None  # None language
        )
        
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello world",
            language="en",  # Should default to "en"
            file_path="/path/to/output.wav"
        )


class TestTTSManagerAsyncSynthesis:
    """Test suite for async synthesis functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CLIENT_TEMP_AUDIO_PATH', '/tmp/test_audio')
    @patch('os.makedirs')
    @patch('asyncio.to_thread')
    async def test_synthesize_gtts_success(self, mock_to_thread, mock_makedirs, mock_gtts, mock_ensure_dirs):
        """
        Test that asynchronous synthesis with gTTS completes successfully and returns the expected output file path.
        """
        mock_to_thread.return_value = None  # Successful synthesis
        
        manager = TTSManager(tts_service_name="gtts", language="en")
        result = await manager.synthesize("Hello world", "test_output.mp3")
        
        assert result == "/tmp/test_audio/test_output.mp3"
        mock_makedirs.assert_called_with('/tmp/test_audio', exist_ok=True)
        mock_to_thread.assert_called_once()
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('tts_manager.CLIENT_TEMP_AUDIO_PATH', '/tmp/test_audio')
    @patch('os.makedirs')
    @patch('asyncio.to_thread')
    async def test_synthesize_xttsv2_success(self, mock_to_thread, mock_makedirs, mock_coqui, mock_ensure_dirs):
        """
        Test that asynchronous XTTSv2 synthesis completes successfully and returns the expected output file path.
        """
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        mock_to_thread.return_value = None
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        result = await manager.synthesize("Hello world", "test_output.wav", "/path/to/speaker.wav")
        
        assert result == "/tmp/test_audio/test_output.wav"
        mock_to_thread.assert_called_once()
        
    @patch('tts_manager.ensure_client_directories')
    async def test_synthesize_not_initialized(self, mock_ensure_dirs, capsys):
        """
        Test that asynchronous synthesis returns None and outputs an error message when the TTSManager is not initialized.
        """
        manager = TTSManager(tts_service_name="unsupported")
        result = await manager.synthesize("Hello world", "test_output.wav")
        
        assert result is None
        captured = capsys.readouterr()
        assert "Not initialized" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('asyncio.to_thread')
    async def test_synthesize_exception_handling(self, mock_to_thread, mock_gtts, mock_ensure_dirs, capsys):
        """
        Test that the asynchronous synthesis method returns None and outputs an error message when an exception occurs during synthesis.
        """
        mock_to_thread.side_effect = Exception("Synthesis failed")
        
        manager = TTSManager(tts_service_name="gtts")
        result = await manager.synthesize("Hello world", "test_output.mp3")
        
        assert result is None
        captured = capsys.readouterr()
        assert "Error during async TTS synthesis" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('asyncio.to_thread')
    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    async def test_synthesize_cleanup_on_error(self, mock_remove, mock_exists, mock_to_thread, mock_gtts, mock_ensure_dirs):
        """
        Test that the output file is removed if an exception occurs during asynchronous synthesis.
        """
        mock_to_thread.side_effect = Exception("Synthesis failed")
        
        manager = TTSManager(tts_service_name="gtts")
        await manager.synthesize("Hello world", "test_output.mp3")
        
        mock_remove.assert_called_once()
        
    @patch('tts_manager.ensure_client_directories')
    async def test_synthesize_unsupported_service(self, mock_ensure_dirs, capsys):
        """
        Test that asynchronous synthesis returns None and outputs an error message when using an unsupported TTS service.
        """
        manager = TTSManager(tts_service_name="unsupported")
        manager.is_initialized = True  # Force initialized state
        manager.tts_instance = Mock()
        
        result = await manager.synthesize("Hello world", "test_output.wav")
        
        assert result is None
        captured = capsys.readouterr()
        assert "No async synthesis method" in captured.out


class TestTTSManagerStaticMethods:
    """Test suite for static methods"""
    
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    def test_list_services_both_available(self, mock_coqui, mock_gtts):
        """
        Test that `TTSManager.list_services()` returns both "gtts" and "xttsv2" when both services are available.
        """
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" in services
        assert len(services) == 2
        
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS', None)
    def test_list_services_only_gtts(self, mock_gtts):
        """
        Test that `TTSManager.list_services()` returns only "gtts" when gTTS is available and XTTSv2 is not.
        """
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" not in services
        assert len(services) == 1
        
    @patch('tts_manager.gtts', None)
    @patch('tts_manager.CoquiTTS')
    def test_list_services_only_xttsv2(self, mock_coqui):
        """
        Test that `list_services` returns only 'xttsv2' when only the XTTSv2 service is available.
        """
        services = TTSManager.list_services()
        assert "gtts" not in services
        assert "xttsv2" in services
        assert len(services) == 1
        
    @patch('tts_manager.gtts', None)
    @patch('tts_manager.CoquiTTS', None)
    def test_list_services_none_available(self):
        """
        Test that `TTSManager.list_services()` returns an empty list when no TTS services are available.
        """
        services = TTSManager.list_services()
        assert len(services) == 0
        
    def test_get_available_models_gtts(self):
        """
        Test that `get_available_models` returns the expected placeholder for gTTS, indicating that model selection is based on language codes.
        """
        models = TTSManager.get_available_models("gtts")
        assert models == ["N/A (uses language codes)"]
        
    def test_get_available_models_xttsv2(self):
        """
        Test that the `get_available_models` method returns the expected model names for the XTTSv2 service.
        """
        models = TTSManager.get_available_models("xttsv2")
        assert "tts_models/multilingual/multi-dataset/xtts_v2" in models
        
    def test_get_available_models_unsupported(self):
        """
        Test that requesting available models for an unsupported TTS service returns an empty list.
        """
        models = TTSManager.get_available_models("unsupported")
        assert models == []


class TestTTSManagerModelHandling:
    """Test suite for model handling functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CLIENT_TTS_MODELS_PATH', '/tmp/test_models')
    @patch('os.makedirs')
    def test_get_or_download_model_blocking_xttsv2(self, mock_makedirs, mock_ensure_dirs):
        """
        Test that the XTTSv2 model handling method returns the model name and ensures necessary directories are created.
        """
        manager = TTSManager(tts_service_name="gtts")  # Any service for testing
        result = manager._get_or_download_model_blocking("xttsv2", "test_model")
        
        assert result == "test_model"
        mock_makedirs.assert_called()
        
    @patch('tts_manager.ensure_client_directories')
    def test_get_or_download_model_blocking_unsupported(self, mock_ensure_dirs):
        """
        Test that attempting to get or download a model for an unsupported TTS service returns None.
        """
        manager = TTSManager(tts_service_name="gtts")
        result = manager._get_or_download_model_blocking("unsupported", "test_model")
        
        assert result is None


class TestTTSManagerEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_empty_text(self, mock_gtts, mock_ensure_dirs):
        """
        Test that synthesizing with empty text still invokes the synthesis process.
        
        Verifies that the TTSManager attempts synthesis even when the input text is an empty string, relying on the TTS service to handle the empty input.
        """
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            result = await manager.synthesize("", "output.mp3")
            # Should still attempt synthesis - TTS service handles empty text
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_very_long_text(self, mock_gtts, mock_ensure_dirs):
        """
        Test that the TTSManager can synthesize audio from a very long text input without errors.
        
        This test verifies that the asynchronous synthesis method is called correctly when provided with a long string, ensuring that the system can handle large input sizes.
        """
        long_text = "A" * 10000
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize(long_text, "output.mp3")
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_special_characters(self, mock_gtts, mock_ensure_dirs):
        """
        Test that the TTSManager can synthesize speech from text containing special Unicode characters without errors.
        """
        special_text = "Hello! 👋 How are you? 🤔 Fine, thanks! 😊"
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize(special_text, "output.mp3")
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_none_filename(self, mock_gtts, mock_ensure_dirs):
        """
        Test that synthesizing with a None filename raises a TypeError or AttributeError.
        """
        manager = TTSManager(tts_service_name="gtts")
        
        # This should raise an exception due to os.path.join with None
        with pytest.raises((TypeError, AttributeError)):
            await manager.synthesize("Hello", None)
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_invalid_filename(self, mock_gtts, mock_ensure_dirs):
        """
        Test that synthesis is attempted even when the output filename contains invalid characters.
        
        Verifies that the synthesis method is called and relies on the operating system to handle any filename errors.
        """
        manager = TTSManager(tts_service_name="gtts")
        invalid_filename = "output<>:\"|?*.mp3"
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize("Hello", invalid_filename)
            # Should still attempt - OS handles invalid filenames
            mock_to_thread.assert_called_once()


class TestTTSManagerLanguageHandling:
    """Test suite for language-specific functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_various_languages(self, mock_gtts, mock_ensure_dirs):
        """
        Test that TTSManager initializes correctly with a variety of valid language codes.
        
        Verifies that the language attribute is set to the provided code for each supported language.
        """
        languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
        
        for lang in languages:
            manager = TTSManager(tts_service_name="gtts", language=lang)
            assert manager.language == lang
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_invalid_language_code(self, mock_gtts, mock_ensure_dirs):
        """
        Test that TTSManager initialization accepts an invalid language code string without modification.
        """
        manager = TTSManager(tts_service_name="gtts", language="invalid_lang")
        assert manager.language == "invalid_lang"  # Should accept any string
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_empty_language(self, mock_gtts, mock_ensure_dirs):
        """
        Test that initializing TTSManager with an empty language string defaults the language to "en".
        """
        manager = TTSManager(tts_service_name="gtts", language="")
        assert manager.language == "en"  # Should default to "en"


class TestTTSManagerConcurrency:
    """Test suite for concurrent operations"""
    
    @patch('tts_manager.ensure_client_directories')  
    @patch('tts_manager.gtts')
    async def test_concurrent_synthesis_requests(self, mock_gtts, mock_ensure_dirs):
        """
        Test that multiple concurrent synthesis requests are handled correctly by TTSManager.
        
        Verifies that all concurrent synthesis tasks complete successfully and that the synthesis method is invoked for each request.
        """
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Create multiple concurrent synthesis tasks
            tasks = []
            for i in range(5):
                task = manager.synthesize(f"Text {i}", f"output_{i}.mp3")
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete successfully
            assert len(results) == 5
            assert all(r is not None for r in results if not isinstance(r, Exception))
            assert mock_to_thread.call_count == 5


class TestTTSManagerFileSystemOperations:
    """Test suite for file system related operations"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('os.makedirs', side_effect=PermissionError("Permission denied"))
    async def test_synthesize_makedirs_permission_error(self, mock_makedirs, mock_gtts, mock_ensure_dirs):
        """
        Test that synthesis raises a PermissionError when directory creation fails due to insufficient permissions.
        """
        manager = TTSManager(tts_service_name="gtts")
        
        with pytest.raises(PermissionError):
            await manager.synthesize("Hello", "output.mp3")
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('os.remove', side_effect=OSError("Removal failed"))
    @patch('os.path.exists', return_value=True)
    @patch('asyncio.to_thread', side_effect=Exception("Synthesis failed"))
    async def test_synthesize_cleanup_failure(self, mock_to_thread, mock_exists, mock_remove, mock_gtts, mock_ensure_dirs):
        """
        Test that synthesis cleanup does not raise exceptions if file removal fails during error handling.
        
        Verifies that when file removal fails during synthesis cleanup, the method returns None and does not propagate the exception.
        """
        manager = TTSManager(tts_service_name="gtts")
        
        # Should not raise exception even if cleanup fails
        result = await manager.synthesize("Hello", "output.mp3")
        assert result is None
        mock_remove.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])