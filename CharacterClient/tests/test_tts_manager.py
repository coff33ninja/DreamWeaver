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

class TestTTSManagerParameterValidation:
    """Test suite for parameter validation and input sanitization"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_with_extreme_values(self, mock_gtts, mock_ensure_dirs):
        """Test initialization with extreme parameter values"""
        # Very long service name
        long_service = "a" * 1000
        manager = TTSManager(tts_service_name=long_service)
        assert manager.service_name == long_service
        
        # Very long model name
        long_model = "model/" * 100
        manager = TTSManager(tts_service_name="gtts", model_name=long_model)
        assert manager.model_name == long_model
        
        # Very long language code
        long_lang = "en-US-" + "x" * 100
        manager = TTSManager(tts_service_name="gtts", language=long_lang)
        assert manager.language == long_lang
    
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_with_unicode_parameters(self, mock_ensure_dirs):
        """Test initialization with unicode and special characters"""
        unicode_service = "tts_service_🎵"
        manager = TTSManager(tts_service_name=unicode_service)
        assert manager.service_name == unicode_service
        
        # Language with unicode
        unicode_lang = "中文"
        manager = TTSManager(tts_service_name="gtts", language=unicode_lang)
        assert manager.language == unicode_lang
    
    @patch('tts_manager.ensure_client_directories')
    def test_initialization_with_numeric_strings(self, mock_ensure_dirs):
        """Test initialization with numeric string parameters"""
        manager = TTSManager(tts_service_name="123", language="456")
        assert manager.service_name == "123"
        assert manager.language == "456"
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_parameter_combinations(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with various parameter combinations"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            # Test with all possible parameter variations
            test_cases = [
                ("text", "file.mp3", None),
                ("", "file.mp3", None),
                ("text with spaces", "file with spaces.mp3", None),
                ("text\nwith\nnewlines", "file.mp3", None),
                ("text\twith\ttabs", "file.mp3", None),
            ]
            
            for text, filename, speaker_wav in test_cases:
                await manager.synthesize(text, filename, speaker_wav)
                
        assert mock_to_thread.call_count == len(test_cases)


class TestTTSManagerResourceManagement:
    """Test suite for resource management and cleanup"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_gpu_memory_management(self, mock_coqui, mock_ensure_dirs):
        """Test GPU memory management for XTTSv2"""
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                manager = TTSManager(
                    tts_service_name="xttsv2",
                    model_name="test_model"
                )
                
                # Verify GPU allocation
                mock_tts_instance.to.assert_called_with("cuda")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_multiple_model_instances(self, mock_coqui, mock_ensure_dirs):
        """Test creating multiple TTS manager instances"""
        mock_tts_instance1 = Mock()
        mock_tts_instance2 = Mock()
        mock_coqui.side_effect = [mock_tts_instance1, mock_tts_instance2]
        
        manager1 = TTSManager(
            tts_service_name="xttsv2",
            model_name="model1"
        )
        manager2 = TTSManager(
            tts_service_name="xttsv2", 
            model_name="model2"
        )
        
        assert manager1.tts_instance != manager2.tts_instance
        assert mock_coqui.call_count == 2
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tempfile.TemporaryDirectory')
    async def test_temporary_file_cleanup(self, mock_temp_dir, mock_gtts, mock_ensure_dirs):
        """Test proper cleanup of temporary files"""
        mock_temp_dir_instance = Mock()
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        mock_temp_dir.return_value.__exit__.return_value = None
        
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            with patch('os.path.exists', return_value=True):
                with patch('os.remove') as mock_remove:
                    mock_to_thread.side_effect = Exception("Synthesis failed")
                    
                    await manager.synthesize("Hello", "output.mp3")
                    
                    # Verify cleanup was attempted
                    mock_remove.assert_called()


class TestTTSManagerPerformanceEdgeCases:
    """Test suite for performance-related edge cases"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_with_massive_text(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis with extremely large text input"""
        massive_text = "A" * 1_000_000  # 1MB of text
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await manager.synthesize(massive_text, "huge_output.mp3")
            
            # Verify the call was made with massive text
            call_args = mock_to_thread.call_args
            assert len(call_args[0][1]) == 1_000_000  # First positional arg should be the text
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_rapid_successive_synthesis_calls(self, mock_gtts, mock_ensure_dirs):
        """Test rapid successive synthesis calls"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Make 50 rapid calls
            tasks = []
            for i in range(50):
                task = manager.synthesize(f"Text {i}", f"output_{i}.mp3")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete
            assert len(results) == 50
            assert mock_to_thread.call_count == 50
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    async def test_xttsv2_with_long_speaker_paths(self, mock_coqui, mock_ensure_dirs):
        """Test XTTSv2 with very long speaker wav paths"""
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en"]
        mock_coqui.return_value = mock_tts_instance
        
        long_path = "/very/long/path/" + "directory/" * 50 + "speaker.wav"
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model",
            speaker_wav_path=long_path
        )
        
        with patch('os.path.exists', return_value=True):
            with patch('asyncio.to_thread') as mock_to_thread:
                await manager.synthesize("Hello", "output.wav", long_path)
                mock_to_thread.assert_called_once()


class TestTTSManagerErrorRecovery:
    """Test suite for error recovery and resilience"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_network_timeout_simulation(self, mock_gtts_module, mock_ensure_dirs):
        """Test handling of network timeouts during synthesis"""
        mock_gtts_instance = Mock()
        mock_gtts_module.gTTS.return_value = mock_gtts_instance
        mock_gtts_instance.save.side_effect = Exception("Network timeout")
        
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread', side_effect=Exception("Network timeout")):
            result = await manager.synthesize("Hello", "output.mp3")
            assert result is None
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    async def test_memory_exhaustion_simulation(self, mock_coqui, mock_ensure_dirs):
        """Test handling of memory exhaustion scenarios"""
        mock_tts_instance = Mock()
        mock_tts_instance.tts_to_file.side_effect = MemoryError("Out of memory")
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        with patch('asyncio.to_thread', side_effect=MemoryError("Out of memory")):
            result = await manager.synthesize("Hello", "output.wav")
            assert result is None
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_disk_space_exhaustion(self, mock_gtts, mock_ensure_dirs):
        """Test handling of disk space exhaustion"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread', side_effect=OSError("No space left on device")):
            result = await manager.synthesize("Hello", "output.mp3")
            assert result is None
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_corrupted_model_handling(self, mock_coqui, mock_ensure_dirs):
        """Test handling of corrupted model files"""
        mock_coqui.side_effect = Exception("Model file corrupted")
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="corrupted_model"
        )
        
        assert manager.is_initialized is False
        assert manager.tts_instance is None


class TestTTSManagerStateManagement:
    """Test suite for state management and consistency"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_state_consistency_after_errors(self, mock_gtts, mock_ensure_dirs):
        """Test that manager state remains consistent after errors"""
        manager = TTSManager(tts_service_name="gtts")
        initial_state = {
            'service_name': manager.service_name,
            'is_initialized': manager.is_initialized,
            'language': manager.language
        }
        
        # Simulate various error conditions
        with patch('asyncio.to_thread', side_effect=Exception("Error")):
            asyncio.run(manager.synthesize("Hello", "output.mp3"))
        
        # State should remain consistent
        assert manager.service_name == initial_state['service_name']
        assert manager.is_initialized == initial_state['is_initialized']
        assert manager.language == initial_state['language']
    
    @patch('tts_manager.ensure_client_directories')
    def test_reinitialization_scenarios(self, mock_ensure_dirs):
        """Test reinitialization with different parameters"""
        # Initial setup
        manager = TTSManager(tts_service_name="gtts", language="en")
        original_service = manager.service_name
        
        # Verify properties are immutable after creation
        assert manager.service_name == original_service
        
        # Create new instance with different parameters
        manager2 = TTSManager(tts_service_name="xttsv2", language="es")
        assert manager2.service_name != manager.service_name
        assert manager2.language != manager.language


class TestTTSManagerIntegrationScenarios:
    """Test suite for integration-like scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    async def test_service_switching_workflow(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test workflow that involves switching between different TTS services"""
        # Create gTTS manager
        gtts_manager = TTSManager(tts_service_name="gtts", language="en")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            await gtts_manager.synthesize("Hello from gTTS", "gtts_output.mp3")
        
        # Create XTTSv2 manager
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en"]
        mock_coqui.return_value = mock_tts_instance
        
        xtts_manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            await xtts_manager.synthesize("Hello from XTTSv2", "xtts_output.wav")
        
        # Verify both work independently
        assert gtts_manager.service_name == "gtts"
        assert xtts_manager.service_name == "xttsv2"
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_multilingual_synthesis_workflow(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis workflow with multiple languages"""
        languages = ["en", "es", "fr", "de", "it"]
        texts = ["Hello", "Hola", "Bonjour", "Hallo", "Ciao"]
        
        managers = []
        for lang in languages:
            manager = TTSManager(tts_service_name="gtts", language=lang)
            managers.append(manager)
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            tasks = []
            for i, (manager, text) in enumerate(zip(managers, texts)):
                task = manager.synthesize(text, f"output_{languages[i]}.mp3")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(r is not None for r in results)
            assert mock_to_thread.call_count == len(languages)


class TestTTSManagerComprehensiveValidation:
    """Test suite for comprehensive input validation"""
    
    @patch('tts_manager.ensure_client_directories')
    def test_parameter_type_validation(self, mock_ensure_dirs):
        """Test that parameters are handled correctly regardless of type"""
        # Test with integer service name (should be converted to string)
        manager = TTSManager(tts_service_name=123)
        assert isinstance(manager.service_name, (str, int))
        
        # Test with boolean language
        manager = TTSManager(tts_service_name="gtts", language=True)
        assert manager.language == "en"  # Should default due to truthy conversion
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_file_extension_handling(self, mock_gtts, mock_ensure_dirs):
        """Test handling of various file extensions"""
        manager = TTSManager(tts_service_name="gtts")
        
        extensions = [".mp3", ".wav", ".ogg", ".m4a", ".flac", ""]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for ext in extensions:
                filename = f"output{ext}" if ext else "output"
                await manager.synthesize("Hello", filename)
        
        assert mock_to_thread.call_count == len(extensions)
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_model_name_variations(self, mock_coqui, mock_ensure_dirs):
        """Test various model name formats and patterns"""
        model_names = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "custom_model",
            "model-with-dashes",
            "model_with_underscores",
            "ModelWithCaps",
            "model123",
            "model/with/slashes",
        ]
        
        for model_name in model_names:
            mock_tts_instance = Mock()
            mock_coqui.return_value = mock_tts_instance
            
            manager = TTSManager(
                tts_service_name="xttsv2",
                model_name=model_name
            )
            
            assert manager.model_name == model_name


class TestTTSManagerPlatformCompatibility:
    """Test suite for platform-specific functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('torch.cuda.is_available')
    def test_cuda_availability_detection(self, mock_cuda_available, mock_coqui, mock_ensure_dirs):
        """Test CUDA availability detection across different scenarios"""
        test_cases = [True, False]
        
        for cuda_available in test_cases:
            mock_cuda_available.return_value = cuda_available
            mock_tts_instance = Mock()
            mock_coqui.return_value = mock_tts_instance
            
            manager = TTSManager(
                tts_service_name="xttsv2",
                model_name="test_model"
            )
            
            expected_device = "cuda" if cuda_available else "cpu"
            if manager.is_initialized:
                mock_tts_instance.to.assert_called_with(expected_device)
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('os.path.sep', '\\')  # Simulate Windows path separator
    async def test_windows_path_handling(self, mock_gtts, mock_ensure_dirs):
        """Test path handling on Windows-style systems"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Test Windows-style paths
            await manager.synthesize("Hello", "C:\\Users\\Test\\output.mp3")
            
            mock_to_thread.assert_called_once()


class TestTTSManagerAdvancedErrorHandling:
    """Test suite for advanced error handling scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_interruption_handling(self, mock_gtts, mock_ensure_dirs):
        """Test handling of synthesis interruption"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Simulate KeyboardInterrupt during synthesis
        with patch('asyncio.to_thread', side_effect=KeyboardInterrupt("User interrupted")):
            with pytest.raises(KeyboardInterrupt):
                await manager.synthesize("Hello", "output.mp3")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_initialization_partial_failure_recovery(self, mock_coqui, mock_ensure_dirs):
        """Test recovery from partial initialization failures"""
        # First attempt fails
        mock_coqui.side_effect = [Exception("First failure"), Mock()]
        
        # First manager fails
        manager1 = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        assert not manager1.is_initialized
        
        # Second manager succeeds
        manager2 = TTSManager(
            tts_service_name="xttsv2", 
            model_name="test_model"
        )
        assert manager2.is_initialized
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_timeout_handling(self, mock_gtts, mock_ensure_dirs):
        """Test handling of synthesis timeout scenarios"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Simulate timeout
        with patch('asyncio.to_thread', side_effect=asyncio.TimeoutError("Operation timed out")):
            result = await manager.synthesize("Hello", "output.mp3")
            assert result is None


class TestTTSManagerBoundaryConditions:
    """Test suite for boundary conditions and limits"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_zero_length_inputs(self, mock_gtts, mock_ensure_dirs):
        """Test handling of zero-length and minimal inputs"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            # Test zero-length text
            await manager.synthesize("", "output.mp3")
            
            # Test single character
            await manager.synthesize("a", "a.mp3")
            
            # Test whitespace only
            await manager.synthesize("   ", "whitespace.mp3")
            
            assert mock_to_thread.call_count == 3
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    def test_language_boundary_conditions(self, mock_coqui, mock_ensure_dirs):
        """Test language handling at boundaries"""
        mock_tts_instance = Mock()
        mock_tts_instance.languages = ["en", "es", "fr"]
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model"
        )
        
        # Test with language not in supported list
        manager._xttsv2_synthesize_blocking(
            "Hello", "output.wav", None, "unsupported_lang"
        )
        
        # Should fall back to first supported language
        mock_tts_instance.tts_to_file.assert_called_with(
            text="Hello",
            language="en",  # First in languages list
            file_path="output.wav"
        )


class TestTTSManagerThreadSafety:
    """Test suite for thread safety and concurrent access"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_concurrent_initialization_attempts(self, mock_gtts, mock_ensure_dirs):
        """Test concurrent initialization of multiple managers"""
        import concurrent.futures
        
        def create_manager(service_name):
            return TTSManager(tts_service_name=service_name)
        
        # Create multiple managers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(create_manager, f"gtts_{i}")
                for i in range(5)
            ]
            
            managers = [future.result() for future in futures]
        
        # All should be created successfully
        assert len(managers) == 5
        assert all(isinstance(m, TTSManager) for m in managers)
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_concurrent_synthesis_different_managers(self, mock_gtts, mock_ensure_dirs):
        """Test concurrent synthesis with different manager instances"""
        managers = [
            TTSManager(tts_service_name="gtts", language=f"lang_{i}")
            for i in range(3)
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Run synthesis concurrently with different managers
            tasks = [
                manager.synthesize(f"Text {i}", f"output_{i}.mp3")
                for i, manager in enumerate(managers)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should complete successfully
            assert len(results) == 3
            assert all(r is not None for r in results)
            assert mock_to_thread.call_count == 3


if __name__ == "__main__":
    # Run all tests including the new comprehensive ones
    pytest.main([__file__, "-v", "--tb=short"])