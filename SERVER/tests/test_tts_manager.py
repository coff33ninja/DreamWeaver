import unittest
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
import pytest
import asyncio
import os
import tempfile
from pathlib import Path
import sys
from typing import Optional

# Add the source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tts_manager import TTSManager
except ImportError:
    # If direct import fails, try alternative import paths
    try:
        from SERVER.src.tts_manager import TTSManager
    except ImportError:
        from src.tts_manager import TTSManager


class TestTTSManagerInitialization(unittest.TestCase):
    """Test TTSManager initialization and configuration."""
    
    def test_init_gtts_service(self):
        """Test initialization with gTTS service."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(tts_service_name="gtts", language="en")
            self.assertEqual(manager.service_name, "gtts")
            self.assertEqual(manager.language, "en")
            self.assertIsNotNone(manager)
    
    def test_init_xttsv2_service(self):
        """Test initialization with XTTSv2 service."""
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui.__bool__ = Mock(return_value=True)
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(
                        tts_service_name="xttsv2", 
                        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                        language="en"
                    )
                    self.assertEqual(manager.service_name, "xttsv2")
                    self.assertEqual(manager.language, "en")
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(tts_service_name="gtts")
            self.assertEqual(manager.language, "en")
            self.assertEqual(manager.model_name, "")
            self.assertEqual(manager.speaker_wav_path, "")
            self.assertFalse(manager.is_initialized)
    
    def test_init_with_speaker_wav_path(self):
        """Test initialization with speaker wav path for XTTS."""
        speaker_path = "/path/to/speaker.wav"
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui.__bool__ = Mock(return_value=True)
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(
                        tts_service_name="xttsv2",
                        model_name="test_model",
                        speaker_wav_path=speaker_path
                    )
                    self.assertEqual(manager.speaker_wav_path, speaker_path)
    
    def test_init_none_values_converted_to_strings(self):
        """Test that None values are properly converted to empty strings."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(
                tts_service_name="gtts",
                model_name=None,
                speaker_wav_path=None,
                language=None
            )
            self.assertEqual(manager.model_name, "")
            self.assertEqual(manager.speaker_wav_path, "")
            self.assertEqual(manager.language, "en")  # None should default to "en"


class TestTTSManagerServiceInitialization(unittest.TestCase):
    """Test service-specific initialization logic."""
    
    @patch('tts_manager.gtts')
    def test_gtts_initialization_success(self, mock_gtts):
        """Test successful gTTS initialization."""
        mock_gtts.__bool__ = Mock(return_value=True)
        manager = TTSManager(tts_service_name="gtts")
        self.assertTrue(manager.is_initialized)
        self.assertIsNotNone(manager.tts_instance)
    
    @patch('tts_manager.gtts', None)
    def test_gtts_initialization_library_not_found(self):
        """Test gTTS initialization when library is not available."""
        with patch('builtins.print') as mock_print:
            manager = TTSManager(tts_service_name="gtts")
            self.assertFalse(manager.is_initialized)
            mock_print.assert_called_with("Server TTSManager: Error - gTTS service selected but library not found.")
    
    @patch('tts_manager.CoquiTTS')
    @patch('tts_manager.torch')
    def test_xttsv2_initialization_success(self, mock_torch, mock_coqui):
        """Test successful XTTSv2 initialization."""
        mock_torch.cuda.is_available.return_value = True
        mock_coqui_instance = Mock()
        mock_coqui.return_value = mock_coqui_instance
        mock_coqui.__bool__ = Mock(return_value=True)
        
        with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
            manager = TTSManager(tts_service_name="xttsv2", model_name="test_model")
            self.assertTrue(manager.is_initialized)
            mock_coqui.assert_called_once_with(model_name="test_model", progress_bar=True)
            mock_coqui_instance.to.assert_called_once_with("cuda")
    
    @patch('tts_manager.CoquiTTS', None)
    def test_xttsv2_initialization_library_not_found(self):
        """Test XTTSv2 initialization when Coqui TTS is not available."""
        with patch('builtins.print') as mock_print:
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                manager = TTSManager(tts_service_name="xttsv2", model_name="test_model")
                self.assertFalse(manager.is_initialized)
                mock_print.assert_called_with("Server TTSManager: Error - Coqui TTS library not available for XTTSv2.")
    
    def test_xttsv2_no_model_name(self):
        """Test XTTSv2 initialization without model name."""
        with patch('builtins.print') as mock_print:
            manager = TTSManager(tts_service_name="xttsv2")
            self.assertFalse(manager.is_initialized)
            mock_print.assert_called_with("Server TTSManager: Warning - No model name provided for TTS service 'xttsv2'.")
    
    def test_unsupported_service(self):
        """Test initialization with unsupported service."""
        with patch('builtins.print') as mock_print:
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                manager = TTSManager(tts_service_name="unsupported", model_name="test_model")
                self.assertFalse(manager.is_initialized)
                mock_print.assert_called_with("Server TTSManager: Unsupported TTS service 'unsupported'.")


class TestTTSManagerSynthesis(unittest.TestCase):
    """Test text-to-speech synthesis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_text = "Hello, this is a test message."
        self.output_path = "/tmp/test_output.wav"
    
    def test_synthesize_gtts_success(self):
        """Test successful gTTS synthesis."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(tts_service_name="gtts", language="en")
            
            async def run_test():
                with patch('tts_manager.asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.return_value = None
                    with patch('tts_manager.os.makedirs'):
                        result = await manager.synthesize(self.test_text, self.output_path)
                        self.assertTrue(result)
                        mock_to_thread.assert_called_once()
            
            asyncio.run(run_test())
    
    def test_synthesize_xttsv2_success(self):
        """Test successful XTTSv2 synthesis."""
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui.__bool__ = Mock(return_value=True)
            mock_coqui_instance = Mock()
            mock_coqui.return_value = mock_coqui_instance
            
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(tts_service_name="xttsv2", model_name="test_model")
                    
                    async def run_test():
                        with patch('tts_manager.asyncio.to_thread') as mock_to_thread:
                            mock_to_thread.return_value = None
                            with patch('tts_manager.os.makedirs'):
                                result = await manager.synthesize(self.test_text, self.output_path)
                                self.assertTrue(result)
                                mock_to_thread.assert_called_once()
                    
                    asyncio.run(run_test())
    
    def test_synthesize_not_initialized(self):
        """Test synthesis when manager is not initialized."""
        manager = TTSManager.__new__(TTSManager)  # Create without calling __init__
        manager.is_initialized = False
        manager.tts_instance = None
        
        async def run_test():
            with patch('builtins.print') as mock_print:
                result = await manager.synthesize(self.test_text, self.output_path)
                self.assertFalse(result)
                mock_print.assert_called_with("Server TTSManager: Not initialized, cannot synthesize.")
        
        asyncio.run(run_test())
    
    def test_synthesize_exception_handling(self):
        """Test synthesis exception handling and cleanup."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(tts_service_name="gtts")
            
            async def run_test():
                with patch('tts_manager.asyncio.to_thread', side_effect=Exception("Test error")):
                    with patch('tts_manager.os.makedirs'):
                        with patch('tts_manager.os.path.exists', return_value=True):
                            with patch('tts_manager.os.remove') as mock_remove:
                                with patch('builtins.print') as mock_print:
                                    result = await manager.synthesize(self.test_text, self.output_path)
                                    self.assertFalse(result)
                                    mock_remove.assert_called_once_with(self.output_path)
                                    mock_print.assert_called_with("Server TTSManager: Error during async TTS synthesis with gtts: Test error")
            
            asyncio.run(run_test())
    
    def test_synthesize_xttsv2_with_speaker_wav(self):
        """Test XTTSv2 synthesis with speaker wav file."""
        speaker_wav = "/path/to/speaker.wav"
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui.__bool__ = Mock(return_value=True)
            mock_coqui_instance = Mock()
            mock_coqui.return_value = mock_coqui_instance
            
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(tts_service_name="xttsv2", model_name="test_model", speaker_wav_path=speaker_wav)
                    
                    async def run_test():
                        with patch('tts_manager.asyncio.to_thread') as mock_to_thread:
                            mock_to_thread.return_value = None
                            with patch('tts_manager.os.makedirs'):
                                result = await manager.synthesize(self.test_text, self.output_path, speaker_wav_for_synthesis=speaker_wav)
                                self.assertTrue(result)
                    
                    asyncio.run(run_test())


class TestTTSManagerBlockingSynthesis(unittest.TestCase):
    """Test blocking synthesis methods."""
    
    def test_gtts_synthesize_blocking(self):
        """Test gTTS blocking synthesis method."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts_instance = Mock()
            mock_gtts.gTTS.return_value = mock_gtts_instance
            mock_gtts.__bool__ = Mock(return_value=True)
            
            manager = TTSManager(tts_service_name="gtts")
            manager._gtts_synthesize_blocking("test text", "/tmp/output.wav", "en")
            
            mock_gtts.gTTS.assert_called_once_with(text="test text", lang="en")
            mock_gtts_instance.save.assert_called_once_with("/tmp/output.wav")
    
    def test_xttsv2_synthesize_blocking_with_speaker_wav(self):
        """Test XTTSv2 blocking synthesis with speaker wav file."""
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui_instance = Mock()
            mock_coqui_instance.tts_to_file.return_value = None
            mock_coqui.return_value = mock_coqui_instance
            mock_coqui.__bool__ = Mock(return_value=True)
            
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(tts_service_name="xttsv2", model_name="test_model")
                    
                    manager._xttsv2_synthesize_blocking(
                        "test text", 
                        "/tmp/output.wav", 
                        "en", 
                        speaker_wav_for_synthesis="/path/to/speaker.wav"
                    )
                    
                    mock_coqui_instance.tts_to_file.assert_called_once()
    
    def test_xttsv2_synthesize_blocking_without_speaker_wav(self):
        """Test XTTSv2 blocking synthesis without speaker wav file."""
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui_instance = Mock()
            mock_coqui_instance.tts_to_file.return_value = None
            mock_coqui.return_value = mock_coqui_instance
            mock_coqui.__bool__ = Mock(return_value=True)
            
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(tts_service_name="xttsv2", model_name="test_model")
                    
                    manager._xttsv2_synthesize_blocking("test text", "/tmp/output.wav", "en")
                    
                    mock_coqui_instance.tts_to_file.assert_called_once()

    def test_xttsv2_synthesize_blocking_with_configured_speaker_wav(self):
        """Test XTTSv2 blocking synthesis with configured speaker wav path."""
        with patch('tts_manager.CoquiTTS') as mock_coqui:
            mock_coqui_instance = Mock()
            mock_coqui_instance.tts_to_file.return_value = None
            mock_coqui.return_value = mock_coqui_instance
            mock_coqui.__bool__ = Mock(return_value=True)
            
            with patch.object(TTSManager, '_get_or_download_model_blocking', return_value="test_model"):
                with patch('tts_manager.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    manager = TTSManager(
                        tts_service_name="xttsv2", 
                        model_name="test_model",
                        speaker_wav_path="/configured/speaker.wav"
                    )
                    
                    manager._xttsv2_synthesize_blocking("test text", "/tmp/output.wav", "en")
                    
                    # Should use configured speaker wav
                    mock_coqui_instance.tts_to_file.assert_called_once_with(
                        text="test text",
                        file_path="/tmp/output.wav",
                        speaker_wav="/configured/speaker.wav",
                        language="en"
                    )


class TestTTSManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(tts_service_name="gtts")
            
            async def run_test():
                with patch('tts_manager.asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.return_value = None
                    with patch('tts_manager.os.makedirs'):
                        result = await manager.synthesize("", "/tmp/test.wav")
                        self.assertTrue(result)
                        mock_to_thread.assert_called_once()
            
            asyncio.run(run_test())
    
    def test_synthesize_whitespace_only_text(self):
        """Test synthesis with whitespace-only text."""
        with patch('tts_manager.gtts') as mock_gtts:
            mock_gtts.__bool__ = Mock(return_value=True)
            manager = TTSManager(tts_service_name="gtts")
            
            async def run_test():
                with patch('tts_manager.asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.return_value = None
                    with patch('tts_manager.os.makedirs'):
                        result = await manager.synthesize("   \n\t   ", "/tmp/test.wav")
                        self.assertTrue(result)
            
            asyncio.run(run_test())
    
    def test_synthesize_very_long_text(self):
        """Test synthesis with very long text."""
        long_text = "Hello world! " * 1000  # Very long text (13,000 characters)
        with patch('tts_manager.gtts') as moc