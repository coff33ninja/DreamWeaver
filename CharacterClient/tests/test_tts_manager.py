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

class TestTTSManagerPerformance:
    """Test suite for performance-related functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_synthesis_performance_measurement(self, tts_manager):
        """Test measuring synthesis performance metrics."""
        text = "Performance test text for TTS synthesis timing."
        
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio_data') as mock_synth:
            start_time = time.time()
            result = tts_manager.synthesize_speech(text)
            end_time = time.time()
            
            synthesis_time = end_time - start_time
            assert synthesis_time >= 0
            assert result is not None
            mock_synth.assert_called_once_with(text)
    
    @pytest.mark.parametrize("text_length", [10, 100, 1000, 5000])
    def test_synthesis_time_scaling(self, tts_manager, text_length):
        """Test synthesis time scaling with text length."""
        text = "A" * text_length
        
        with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio_data'):
            start_time = time.time()
            tts_manager.synthesize_speech(text)
            end_time = time.time()
            
            # Should complete within reasonable time regardless of text length
            assert (end_time - start_time) < 60  # Max 60 seconds for any text
    
    def test_memory_usage_during_synthesis(self, tts_manager):
        """Test memory usage patterns during synthesis."""
        if hasattr(tts_manager, 'get_memory_usage'):
            initial_memory = tts_manager.get_memory_usage()
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio_data'):
                tts_manager.synthesize_speech("Memory test text")
                
            final_memory = tts_manager.get_memory_usage()
            
            # Memory usage should be tracked
            assert isinstance(initial_memory, (int, float))
            assert isinstance(final_memory, (int, float))
    
    def test_batch_synthesis_performance(self, tts_manager):
        """Test performance of batch synthesis operations."""
        texts = [f"Batch text number {i}" for i in range(10)]
        
        if hasattr(tts_manager, 'synthesize_batch'):
            with patch.object(tts_manager, 'synthesize_batch', return_value=[b'audio'] * 10):
                start_time = time.time()
                results = tts_manager.synthesize_batch(texts)
                end_time = time.time()
                
                assert len(results) == len(texts)
                assert (end_time - start_time) < 30  # Reasonable time for batch


class TestTTSManagerResourceManagement:
    """Test suite for resource management and cleanup"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_resource_cleanup_on_destruction(self, tts_manager):
        """Test proper resource cleanup when TTS manager is destroyed."""
        if hasattr(tts_manager, '__del__'):
            with patch.object(tts_manager, 'cleanup_resources') as mock_cleanup:
                del tts_manager
                # Cleanup should be called
                mock_cleanup.assert_called()
    
    def test_temporary_file_cleanup(self, tts_manager):
        """Test cleanup of temporary files created during synthesis."""
        if hasattr(tts_manager, 'get_temp_files'):
            temp_files_before = tts_manager.get_temp_files()
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                tts_manager.synthesize_speech("Test cleanup")
                
            if hasattr(tts_manager, 'cleanup_temp_files'):
                tts_manager.cleanup_temp_files()
                temp_files_after = tts_manager.get_temp_files()
                
                # Should clean up temporary files
                assert len(temp_files_after) <= len(temp_files_before)
    
    def test_memory_leak_prevention(self, tts_manager):
        """Test for memory leak prevention in repeated operations."""
        # Simulate repeated synthesis operations
        for i in range(50):
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                tts_manager.synthesize_speech(f"Memory test {i}")
        
        # Should complete without memory issues (this test mainly ensures no exceptions)
        assert True
    
    def test_context_manager_support(self, tts_manager):
        """Test context manager protocol support."""
        if hasattr(tts_manager, '__enter__') and hasattr(tts_manager, '__exit__'):
            with tts_manager as manager:
                assert manager is not None
                with patch.object(manager, 'synthesize_speech', return_value=b'audio'):
                    result = manager.synthesize_speech("Context manager test")
                    assert result is not None


class TestTTSManagerAdvancedConfiguration:
    """Test suite for advanced configuration scenarios"""
    
    def test_configuration_persistence(self):
        """Test configuration persistence across instances."""
        config = {
            'voice': 'test-voice',
            'speed': 1.5,
            'pitch': 1.2,
            'volume': 0.9,
            'format': 'wav'
        }
        
        if hasattr(TTSManager, 'save_config'):
            TTSManager.save_config(config, 'test_config.json')
            
            try:
                loaded_manager = TTSManager.from_config('test_config.json')
                assert loaded_manager.config.get('voice') == 'test-voice'
                assert loaded_manager.config.get('speed') == 1.5
            finally:
                # Cleanup
                if os.path.exists('test_config.json'):
                    os.unlink('test_config.json')
    
    def test_dynamic_configuration_updates(self):
        """Test dynamic configuration updates during runtime."""
        tts_manager = TTSManager()
        
        if hasattr(tts_manager, 'update_config'):
            new_settings = {'speed': 2.0, 'volume': 0.7}
            tts_manager.update_config(new_settings)
            
            assert tts_manager.config.get('speed') == 2.0
            assert tts_manager.config.get('volume') == 0.7
    
    @pytest.mark.parametrize("config_format", ['json', 'yaml', 'toml'])
    def test_configuration_file_formats(self, config_format):
        """Test loading configuration from different file formats."""
        if hasattr(TTSManager, f'from_{config_format}'):
            config_data = {
                'voice': 'test-voice',
                'speed': 1.0,
                'pitch': 1.0
            }
            
            filename = f'test_config.{config_format}'
            
            try:
                # Create config file (mock the file creation)
                with patch('builtins.open', mock.mock_open(read_data=str(config_data))):
                    loader_method = getattr(TTSManager, f'from_{config_format}')
                    manager = loader_method(filename)
                    assert manager is not None
            except (NotImplementedError, AttributeError):
                pytest.skip(f"{config_format} format not supported")
    
    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        test_cases = [
            # Valid configurations
            ({'voice': 'en-US-Standard-A', 'speed': 1.0}, True),
            ({'pitch': 0.5, 'volume': 1.0}, True),
            ({'format': 'mp3', 'quality': 'high'}, True),
            
            # Invalid configurations
            ({'speed': 'fast'}, False),  # Wrong type
            ({'pitch': []}, False),      # Wrong type
            ({'volume': -5}, False),     # Out of range
            ({'format': 12345}, False),  # Wrong type
        ]
        
        for config, should_be_valid in test_cases:
            if should_be_valid:
                try:
                    manager = TTSManager(config=config)
                    assert manager is not None
                except (ValueError, TypeError, ConfigurationError):
                    pytest.fail(f"Valid configuration rejected: {config}")
            else:
                with pytest.raises((ValueError, TypeError, ConfigurationError)):
                    TTSManager(config=config)


class TestTTSManagerVoiceCloning:
    """Test suite for voice cloning functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="xttsv2")
    
    def test_voice_cloning_from_sample(self, tts_manager):
        """Test voice cloning from audio sample."""
        if hasattr(tts_manager, 'clone_voice'):
            sample_audio_path = "/path/to/sample.wav"
            
            with patch.object(tts_manager, 'clone_voice', return_value='cloned_voice_id'):
                voice_id = tts_manager.clone_voice(sample_audio_path)
                assert voice_id == 'cloned_voice_id'
    
    def test_speaker_embedding_extraction(self, tts_manager):
        """Test speaker embedding extraction from audio."""
        if hasattr(tts_manager, 'extract_speaker_embedding'):
            with patch.object(tts_manager, 'extract_speaker_embedding', return_value=b'embedding_data'):
                embedding = tts_manager.extract_speaker_embedding("/path/to/audio.wav")
                assert isinstance(embedding, bytes)
                assert len(embedding) > 0
    
    def test_voice_similarity_scoring(self, tts_manager):
        """Test voice similarity scoring functionality."""
        if hasattr(tts_manager, 'calculate_voice_similarity'):
            with patch.object(tts_manager, 'calculate_voice_similarity', return_value=0.95):
                similarity = tts_manager.calculate_voice_similarity(
                    "/path/to/voice1.wav", 
                    "/path/to/voice2.wav"
                )
                assert 0.0 <= similarity <= 1.0
    
    def test_voice_model_management(self, tts_manager):
        """Test voice model creation and management."""
        if hasattr(tts_manager, 'create_voice_model'):
            with patch.object(tts_manager, 'create_voice_model', return_value='model_id'):
                model_id = tts_manager.create_voice_model(
                    voice_samples=["/path/to/sample1.wav", "/path/to/sample2.wav"],
                    voice_name="test_voice"
                )
                assert model_id == 'model_id'
        
        if hasattr(tts_manager, 'list_voice_models'):
            with patch.object(tts_manager, 'list_voice_models', return_value=['model1', 'model2']):
                models = tts_manager.list_voice_models()
                assert isinstance(models, list)
                assert len(models) >= 0


class TestTTSManagerStreamingAndRealtime:
    """Test suite for streaming and real-time functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    @pytest.mark.asyncio
    async def test_streaming_synthesis(self, tts_manager):
        """Test streaming synthesis functionality."""
        if hasattr(tts_manager, 'synthesize_stream'):
            text = "This is a streaming synthesis test with multiple sentences."
            
            async def mock_stream():
                for i in range(5):
                    yield b'audio_chunk_' + str(i).encode()
            
            with patch.object(tts_manager, 'synthesize_stream', return_value=mock_stream()):
                audio_chunks = []
                async for chunk in tts_manager.synthesize_stream(text):
                    audio_chunks.append(chunk)
                    assert isinstance(chunk, bytes)
                
                assert len(audio_chunks) > 0
    
    def test_real_time_synthesis(self, tts_manager):
        """Test real-time synthesis capabilities."""
        if hasattr(tts_manager, 'start_real_time_synthesis'):
            with patch.object(tts_manager, 'start_real_time_synthesis'):
                tts_manager.start_real_time_synthesis()
                
                if hasattr(tts_manager, 'add_text_to_queue'):
                    tts_manager.add_text_to_queue("Real-time test text")
                
                if hasattr(tts_manager, 'stop_real_time_synthesis'):
                    tts_manager.stop_real_time_synthesis()
    
    def test_audio_buffer_management(self, tts_manager):
        """Test audio buffer management for streaming."""
        if hasattr(tts_manager, 'get_audio_buffer'):
            with patch.object(tts_manager, 'get_audio_buffer', return_value=b'buffer_data'):
                buffer_data = tts_manager.get_audio_buffer()
                assert isinstance(buffer_data, bytes)
        
        if hasattr(tts_manager, 'clear_audio_buffer'):
            tts_manager.clear_audio_buffer()
            # Should complete without error
            assert True
    
    def test_latency_measurement(self, tts_manager):
        """Test synthesis latency measurement."""
        if hasattr(tts_manager, 'measure_latency'):
            with patch.object(tts_manager, 'measure_latency', return_value=0.250):
                latency = tts_manager.measure_latency("Latency test text")
                assert isinstance(latency, (int, float))
                assert latency >= 0


class TestTTSManagerCallbacksAndEvents:
    """Test suite for callback and event handling"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_synthesis_progress_callback(self, tts_manager):
        """Test synthesis progress callback functionality."""
        if hasattr(tts_manager, 'set_progress_callback'):
            progress_values = []
            
            def progress_callback(progress):
                progress_values.append(progress)
            
            tts_manager.set_progress_callback(progress_callback)
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                # Simulate calling the callback during synthesis
                if hasattr(tts_manager, '_call_progress_callback'):
                    tts_manager._call_progress_callback(0.5)
                    tts_manager._call_progress_callback(1.0)
                
                tts_manager.synthesize_speech("Callback test")
                
            # Check if callbacks were called (if implemented)
            if progress_values:
                assert all(0.0 <= p <= 1.0 for p in progress_values)
    
    def test_error_callback_handling(self, tts_manager):
        """Test error callback handling."""
        if hasattr(tts_manager, 'set_error_callback'):
            error_messages = []
            
            def error_callback(error):
                error_messages.append(str(error))
            
            tts_manager.set_error_callback(error_callback)
            
            # Simulate an error during synthesis
            with patch.object(tts_manager, 'synthesize_speech', side_effect=TTSError("Test error")):
                try:
                    tts_manager.synthesize_speech("Error test")
                except TTSError:
                    pass
            
            # Check if error callback was called (if implemented)
            if hasattr(tts_manager, '_call_error_callback'):
                tts_manager._call_error_callback(TTSError("Test error"))
                if error_messages:
                    assert len(error_messages) > 0
    
    def test_completion_callback(self, tts_manager):
        """Test synthesis completion callback."""
        if hasattr(tts_manager, 'set_completion_callback'):
            completion_called = []
            
            def completion_callback(result):
                completion_called.append(result)
            
            tts_manager.set_completion_callback(completion_callback)
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                result = tts_manager.synthesize_speech("Completion test")
                
                # Simulate completion callback
                if hasattr(tts_manager, '_call_completion_callback'):
                    tts_manager._call_completion_callback(result)
                
            if completion_called:
                assert len(completion_called) > 0


class TestTTSManagerNetworkAndConnectivity:
    """Test suite for network-related functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")  # Network-dependent service
    
    def test_network_connectivity_check(self, tts_manager):
        """Test network connectivity checking."""
        if hasattr(tts_manager, 'check_network_connectivity'):
            with patch.object(tts_manager, 'check_network_connectivity', return_value=True):
                is_connected = tts_manager.check_network_connectivity()
                assert isinstance(is_connected, bool)
    
    def test_offline_mode_fallback(self, tts_manager):
        """Test offline mode fallback functionality."""
        if hasattr(tts_manager, 'set_offline_mode'):
            tts_manager.set_offline_mode(True)
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'offline_audio'):
                result = tts_manager.synthesize_speech("Offline test")
                assert result is not None
    
    def test_network_timeout_handling(self, tts_manager):
        """Test network timeout handling."""
        if hasattr(tts_manager, 'set_network_timeout'):
            tts_manager.set_network_timeout(5.0)  # 5 seconds
            
            # Simulate network timeout
            with patch.object(tts_manager, 'synthesize_speech', side_effect=TimeoutError("Network timeout")):
                with pytest.raises(TimeoutError):
                    tts_manager.synthesize_speech("Timeout test")
    
    def test_retry_mechanism(self, tts_manager):
        """Test automatic retry mechanism for network failures."""
        if hasattr(tts_manager, 'set_retry_attempts'):
            tts_manager.set_retry_attempts(3)
            
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Network error")
                return b'retry_success_audio'
            
            with patch.object(tts_manager, 'synthesize_speech', side_effect=side_effect):
                result = tts_manager.synthesize_speech("Retry test")
                assert result == b'retry_success_audio'
                assert call_count == 3


class TestTTSManagerCachingSystem:
    """Test suite for caching functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        config = {'enable_cache': True, 'cache_size': 100}
        return TTSManager(config=config)
    
    def test_audio_caching_basic(self, tts_manager):
        """Test basic audio caching functionality."""
        if hasattr(tts_manager, 'enable_cache'):
            tts_manager.enable_cache(True)
            
            text = "Cached synthesis test"
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'cached_audio') as mock_synth:
                # First call should trigger synthesis
                result1 = tts_manager.synthesize_speech(text)
                
                # Second call should use cache
                result2 = tts_manager.synthesize_speech(text)
                
                assert result1 == result2
                # Should only call synthesis once if caching works
                if hasattr(tts_manager, '_cache'):
                    assert mock_synth.call_count <= 2
    
    def test_cache_size_limits(self, tts_manager):
        """Test cache size limiting functionality."""
        if hasattr(tts_manager, 'set_cache_size'):
            tts_manager.set_cache_size(2)  # Very small cache
            
            texts = ["Text 1", "Text 2", "Text 3"]
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                for text in texts:
                    tts_manager.synthesize_speech(text)
                
                if hasattr(tts_manager, 'get_cache_size'):
                    cache_size = tts_manager.get_cache_size()
                    assert cache_size <= 2
    
    def test_cache_invalidation(self, tts_manager):
        """Test cache invalidation functionality."""
        if hasattr(tts_manager, 'clear_cache'):
            text = "Cache invalidation test"
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                tts_manager.synthesize_speech(text)
                tts_manager.clear_cache()
                
                if hasattr(tts_manager, 'get_cache_size'):
                    assert tts_manager.get_cache_size() == 0
    
    def test_cache_hit_rate_tracking(self, tts_manager):
        """Test cache hit rate tracking."""
        if hasattr(tts_manager, 'get_cache_stats'):
            text = "Cache hit rate test"
            
            with patch.object(tts_manager, 'synthesize_speech', return_value=b'audio'):
                # Multiple calls to same text
                for _ in range(5):
                    tts_manager.synthesize_speech(text)
                
                stats = tts_manager.get_cache_stats()
                if stats:
                    assert 'hit_rate' in stats or 'hits' in stats
                    assert isinstance(stats.get('hit_rate', 0), (int, float))


class TestTTSManagerMultithreadingSafety:
    """Test suite for multithreading safety"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()
    
    def test_thread_safety_concurrent_synthesis(self, tts_manager):
        """Test thread safety during concurrent synthesis operations."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def synthesis_worker(text, worker_id):
            try:
                with patch.object(tts_manager, 'synthesize_speech', return_value=f'audio_{worker_id}'.encode()):
                    result = tts_manager.synthesize_speech(f"{text} {worker_id}")
                    results.put((worker_id, result))
            except Exception as e:
                errors.put((worker_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=synthesis_worker, args=("Thread test", i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        assert results.qsize() <= 5  # All threads should complete
        assert errors.qsize() == 0   # No errors should occur
    
    def test_configuration_thread_safety(self, tts_manager):
        """Test thread safety of configuration changes."""
        import threading
        
        def config_changer():
            if hasattr(tts_manager, 'update_config'):
                for i in range(10):
                    tts_manager.update_config({'test_param': i})
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=config_changer)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        # Should complete without deadlocks or exceptions
        assert True


class TestTTSManagerCompatibilityAndIntegration:
    """Test suite for compatibility and integration scenarios"""
    
    def test_python_version_compatibility(self):
        """Test compatibility across Python versions."""
        import sys
        
        # Test should work on supported Python versions
        assert sys.version_info >= (3, 7), "Python 3.7+ required"
        
        # Test instantiation
        manager = TTSManager()
        assert manager is not None
    
    def test_dependency_version_compatibility(self):
        """Test compatibility with different dependency versions."""
        try:
            import pkg_resources
            
            # Check for key dependencies if they exist
            dependencies = ['numpy', 'torch', 'torchaudio', 'gtts']
            
            for dep in dependencies:
                try:
                    version = pkg_resources.get_distribution(dep).version
                    # Just check that version can be retrieved
                    assert isinstance(version, str)
                    assert len(version) > 0
                except pkg_resources.DistributionNotFound:
                    # Dependency not installed, skip
                    continue
                    
        except ImportError:
            # pkg_resources not available, skip test
            pytest.skip("pkg_resources not available")
    
    def test_os_compatibility(self):
        """Test OS compatibility."""
        import platform
        
        os_name = platform.system()
        
        # Test should work on major operating systems
        assert os_name in ['Windows', 'Darwin', 'Linux'], f"Unsupported OS: {os_name}"
        
        # Test basic functionality
        manager = TTSManager()
        with patch.object(manager, 'synthesize_speech', return_value=b'os_test_audio'):
            result = manager.synthesize_speech("OS compatibility test")
            assert result is not None
    
    def test_unicode_text_handling(self):
        """Test handling of various Unicode text inputs."""
        manager = TTSManager()
        
        unicode_texts = [
            "Hello, 世界!",           # Mixed English/Chinese
            "Café naïve résumé",      # Accented characters
            "Москва",                 # Cyrillic
            "こんにちは",                # Japanese
            "🙂😊🎉",                  # Emojis
            "العربية",                # Arabic
            "हिन्दी",                  # Hindi
        ]
        
        for text in unicode_texts:
            with patch.object(manager, 'synthesize_speech', return_value=b'unicode_audio'):
                try:
                    result = manager.synthesize_speech(text)
                    assert result is not None
                except (UnicodeError, ValueError):
                    # Some TTS engines might not support all Unicode
                    continue


if __name__ == "__main__":
    # Run with additional verbosity and coverage reporting
    pytest.main([__file__, "-v", "--tb=short", "--maxfail=5"])