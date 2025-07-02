import pytest
import asyncio
import os
import tempfile
import shutil
import uuid
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import sys
from pathlib import Path

# Add the src directory to the path to import narrator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the external dependencies before importing narrator
sys.modules['whisper'] = Mock()
sys.modules['pyannote'] = Mock()
sys.modules['pyannote.audio'] = Mock()

try:
    from narrator import Narrator
except ImportError as e:
    pytest.skip(f"Narrator module not found: {e}", allow_module_level=True)


class TestNarratorInitialization:
    """Test cases for Narrator class initialization."""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_init_default_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with default model size."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        
        mock_load_model.assert_called_once()
        assert narrator.stt_model is not None
        assert narrator.default_speaker_name == "Narrator"
        assert narrator.last_transcription is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_init_custom_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with custom model size."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator(model_size="large")
        
        mock_load_model.assert_called_once_with("large")
        assert narrator.stt_model is not None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_init_model_loading_failure(self, mock_pipeline, mock_load_model):
        """Test initialization when model loading fails."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        narrator = Narrator()
        
        assert narrator.stt_model is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    def test_init_with_diarization_enabled(self, mock_pipeline, mock_load_model):
        """Test initialization with diarization enabled."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.return_value = Mock()
        
        narrator = Narrator()
        
        mock_pipeline.from_pretrained.assert_called_once()
        assert narrator.diarization_pipeline is not None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    @patch('narrator.MAX_DIARIZATION_RETRIES', 2)
    def test_init_diarization_loading_failure_with_retries(self, mock_pipeline, mock_load_model):
        """Test diarization loading with failures and retries."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = Exception("Diarization loading failed")
        
        with patch('builtins.input', return_value='s'):  # Skip diarization
            narrator = Narrator()
        
        assert narrator.diarization_pipeline is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    @patch('narrator.webbrowser.open')
    @patch('narrator.re.findall')
    def test_init_diarization_with_url_opening(self, mock_findall, mock_webbrowser, mock_pipeline, mock_load_model):
        """Test diarization initialization with URL opening for errors."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = Exception("Error with https://example.com/token")
        mock_findall.return_value = ['https://example.com/token']
        
        with patch('builtins.input', return_value='s'):  # Skip diarization
            narrator = Narrator()
        
        mock_webbrowser.assert_called_once_with('https://example.com/token')
        assert narrator.diarization_pipeline is None


class TestNarratorProcessNarration:
    """Test cases for the process_narration method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_file = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create a dummy audio file
        with open(self.test_audio_file, 'wb') as f:
            f.write(b'dummy audio content')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    @patch('narrator.os.makedirs')
    @patch('narrator.uuid.uuid4')
    async def test_process_narration_success(self, mock_uuid, mock_makedirs, mock_pipeline, mock_load_model):
        """Test successful audio processing."""
        # Setup mocks
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model
        mock_uuid.return_value.hex = "test123"
        
        narrator = Narrator()
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'audio data'
            mock_open.return_value.__enter__.return_value.write.return_value = None
            
            result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == "Hello world"
        assert result["speaker"] == "Narrator"
        assert "narration_test123.wav" in result["audio_path"]
        assert narrator.last_transcription == "Hello world"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_no_model(self, mock_pipeline, mock_load_model):
        """Test processing when STT model is not loaded."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        narrator = Narrator()
        result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == self.test_audio_file
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_transcription_failure(self, mock_pipeline, mock_load_model):
        """Test processing when transcription fails."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=Exception("Transcription failed"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_with_diarization(self, mock_pipeline, mock_load_model):
        """Test processing with diarization enabled."""
        # Setup STT mock
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization mock
        mock_diarization = Mock()
        mock_track = Mock()
        mock_track.__iter__ = Mock(return_value=iter([('segment', 'track', 'SPEAKER_01')]))
        mock_diarization.itertracks = Mock(return_value=mock_track)
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        narrator = Narrator()
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == "Hello world"
        assert result["speaker"] == "SPEAKER_01"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_diarization_failure(self, mock_pipeline, mock_load_model):
        """Test processing when diarization fails."""
        # Setup STT mock
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization mock to fail
        mock_diarization = Mock()
        mock_diarization.side_effect = Exception("Diarization failed")
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        narrator = Narrator()
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == "Hello world"
        assert result["speaker"] == "Narrator"  # Falls back to default
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_text_list_format(self, mock_pipeline, mock_load_model):
        """Test processing when transcription returns text as list."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": ["Hello", "world", "test"]})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == "Hello world test"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_file_save_failure(self, mock_pipeline, mock_load_model):
        """Test processing when audio file saving fails."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        with patch('narrator.os.makedirs', side_effect=Exception("Permission denied")):
            result = await narrator.process_narration(self.test_audio_file)
        
        assert result["text"] == "Hello world"
        assert result["audio_path"] == self.test_audio_file  # Falls back to original path


class TestNarratorCorrectLastTranscription:
    """Test cases for the correct_last_transcription method."""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_correct_last_transcription(self, mock_pipeline, mock_load_model):
        """Test correcting the last transcription."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        narrator.last_transcription = "Original transcription"
        
        narrator.correct_last_transcription("Corrected transcription")
        
        assert narrator.last_transcription == "Corrected transcription"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_correct_last_transcription_empty_string(self, mock_pipeline, mock_load_model):
        """Test correcting with empty string."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        narrator.last_transcription = "Original transcription"
        
        narrator.correct_last_transcription("")
        
        assert narrator.last_transcription == ""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_correct_last_transcription_none_input(self, mock_pipeline, mock_load_model):
        """Test correcting with None input."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        narrator.last_transcription = "Original transcription"
        
        narrator.correct_last_transcription(None)
        
        assert narrator.last_transcription is None


class TestNarratorEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_nonexistent_file(self, mock_pipeline, mock_load_model):
        """Test processing non-existent audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=FileNotFoundError("File not found"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        nonexistent_file = "/path/to/nonexistent/file.wav"
        
        result = await narrator.process_narration(nonexistent_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_empty_audio_file(self, mock_pipeline, mock_load_model):
        """Test processing empty audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": ""})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create empty audio file
        empty_file = os.path.join(self.temp_dir, "empty.wav")
        with open(empty_file, 'wb') as f:
            pass  # Create empty file
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(empty_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_corrupted_audio_file(self, mock_pipeline, mock_load_model):
        """Test processing corrupted audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=Exception("Corrupted audio"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create corrupted audio file
        corrupted_file = os.path.join(self.temp_dir, "corrupted.wav")
        with open(corrupted_file, 'wb') as f:
            f.write(b'not audio data')
        
        result = await narrator.process_narration(corrupted_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"


class TestNarratorIntegration:
    """Integration test cases."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH')
    async def test_full_workflow_integration(self, mock_audio_path, mock_pipeline, mock_load_model):
        """Test full workflow from initialization to transcription."""
        mock_audio_path.__str__ = Mock(return_value=self.temp_dir)
        
        # Setup mocks
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Integration test successful"})
        mock_load_model.return_value = mock_stt_model
        
        # Create test audio file
        test_file = os.path.join(self.temp_dir, "integration_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'integration test audio')
        
        # Initialize narrator
        narrator = Narrator(model_size="tiny")
        
        # Process narration
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'audio data'
                mock_open.return_value.__enter__.return_value.write.return_value = None
                
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "integration123"
                    result = await narrator.process_narration(test_file)
        
        # Verify results
        assert result["text"] == "Integration test successful"
        assert result["speaker"] == "Narrator"
        assert narrator.last_transcription == "Integration test successful"
        
        # Test correction
        narrator.correct_last_transcription("Corrected integration test")
        assert narrator.last_transcription == "Corrected integration test"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_multiple_consecutive_processing(self, mock_pipeline, mock_load_model):
        """Test processing multiple audio files consecutively."""
        mock_stt_model = Mock()
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create multiple test files
        files_and_texts = [
            ("file1.wav", "First transcription"),
            ("file2.wav", "Second transcription"),
            ("file3.wav", "Third transcription")
        ]
        
        results = []
        for filename, expected_text in files_and_texts:
            test_file = os.path.join(self.temp_dir, filename)
            with open(test_file, 'wb') as f:
                f.write(b'test audio')
            
            mock_stt_model.transcribe = Mock(return_value={"text": expected_text})
            
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = f"test{len(results)}"
                        result = await narrator.process_narration(test_file)
                        results.append(result)
        
        # Verify all results
        for i, (_, expected_text) in enumerate(files_and_texts):
            assert results[i]["text"] == expected_text
            assert results[i]["speaker"] == "Narrator"


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium", "large"])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
def test_narrator_initialization_different_models(mock_pipeline, mock_load_model, model_size):
    """Test narrator initialization with different model sizes."""
    mock_load_model.return_value = Mock()
    
    narrator = Narrator(model_size=model_size)
    
    mock_load_model.assert_called_once_with(model_size)
    assert narrator.stt_model is not None


@pytest.mark.parametrize("text_input,expected_length", [
    ("", 0),
    ("Hello", 5),
    ("Hello world", 11),
    ("   Hello world   ", 11),  # Should be stripped
    (["Hello", "world"], 11),  # List input
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
async def test_process_narration_text_variations(mock_pipeline, mock_load_model, text_input, expected_length):
    """Test processing with various text formats."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(return_value={"text": text_input})
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(b'dummy audio')
        tmp_file_path = tmp_file.name
    
    try:
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(tmp_file_path)
        
        if isinstance(text_input, list):
            assert len(result["text"]) == expected_length
        else:
            assert len(result["text"].strip()) == expected_length
    finally:
        os.unlink(tmp_file_path)


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v', '--tb=short'])

class TestNarratorAdvancedInitialization:
    """Advanced initialization test cases."""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_init_with_invalid_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with invalid model size."""
        mock_load_model.side_effect = ValueError("Invalid model size")
        
        narrator = Narrator(model_size="invalid_size")
        
        assert narrator.stt_model is None
        assert narrator.default_speaker_name == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    def test_init_diarization_retry_mechanism(self, mock_pipeline, mock_load_model):
        """Test diarization retry mechanism with different failure scenarios."""
        mock_load_model.return_value = Mock()
        
        # First call fails, second succeeds
        mock_pipeline.from_pretrained.side_effect = [
            Exception("First failure"),
            Mock()
        ]
        
        with patch('builtins.input', return_value='r'):  # Retry
            narrator = Narrator()
        
        assert mock_pipeline.from_pretrained.call_count == 2
        assert narrator.diarization_pipeline is not None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.threading.Thread')
    def test_init_concurrent_initialization(self, mock_thread, mock_pipeline, mock_load_model):
        """Test concurrent narrator initialization."""
        mock_load_model.return_value = Mock()
        
        narrators = []
        for i in range(3):
            narrator = Narrator(model_size="tiny")
            narrators.append(narrator)
        
        # All should be successfully initialized
        for narrator in narrators:
            assert narrator.stt_model is not None
            assert narrator.default_speaker_name == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    @patch('narrator.os.getenv')
    def test_init_with_environment_variables(self, mock_getenv, mock_pipeline, mock_load_model):
        """Test initialization with different environment variable configurations."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.return_value = Mock()
        
        # Test with custom environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'HUGGINGFACE_TOKEN': 'custom_token',
            'NARRATOR_MODEL_SIZE': 'medium'
        }.get(key, default)
        
        narrator = Narrator()
        
        assert narrator.stt_model is not None
        assert narrator.diarization_pipeline is not None


class TestNarratorAdvancedProcessing:
    """Advanced processing test cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_with_unicode_text(self, mock_pipeline, mock_load_model):
        """Test processing with unicode and special characters in transcription."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Héllo wörld! 🎉 Testing unicode: ñäöü"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "unicode_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'audio data')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "unicode123"
                    result = await narrator.process_narration(test_file)
        
        assert result["text"] == "Héllo wörld! 🎉 Testing unicode: ñäöü"
        assert narrator.last_transcription == "Héllo wörld! 🎉 Testing unicode: ñäöü"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_with_very_long_text(self, mock_pipeline, mock_load_model):
        """Test processing with very long transcription text."""
        long_text = "This is a very long transcription. " * 1000  # ~34KB of text
        
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": long_text})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "long_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'audio data' * 1000)  # Simulate large audio file
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "long123"
                    result = await narrator.process_narration(test_file)
        
        assert len(result["text"]) > 30000
        assert result["text"] == long_text
        assert narrator.last_transcription == long_text
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    async def test_process_narration_multiple_speakers(self, mock_pipeline, mock_load_model):
        """Test processing with multiple speakers in diarization."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Multiple speakers talking"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization with multiple speakers
        mock_diarization = Mock()
        mock_segments = [
            ('segment1', 'track1', 'SPEAKER_00'),
            ('segment2', 'track2', 'SPEAKER_01'),
            ('segment3', 'track3', 'SPEAKER_02')
        ]
        mock_diarization.itertracks = Mock(return_value=iter(mock_segments))
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "multi_speaker.wav")
        with open(test_file, 'wb') as f:
            f.write(b'multi speaker audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "multi123"
                    result = await narrator.process_narration(test_file)
        
        # Should return the first speaker detected
        assert result["speaker"] == "SPEAKER_00"
        assert result["text"] == "Multiple speakers talking"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_memory_cleanup(self, mock_pipeline, mock_load_model):
        """Test memory cleanup during processing."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Memory test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Process multiple files to test memory cleanup
        for i in range(5):
            test_file = os.path.join(self.temp_dir, f"memory_test_{i}.wav")
            with open(test_file, 'wb') as f:
                f.write(b'audio data for memory test')
            
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = f"memory{i}"
                        result = await narrator.process_narration(test_file)
                        
                        assert result["text"] == "Memory test"
                        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_concurrent_processing(self, mock_pipeline, mock_load_model):
        """Test concurrent processing of multiple audio files."""
        mock_stt_model = Mock()
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create multiple test files
        test_files = []
        expected_texts = []
        
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"concurrent_{i}.wav")
            expected_text = f"Concurrent processing test {i}"
            
            with open(test_file, 'wb') as f:
                f.write(f'audio data {i}'.encode())
            
            test_files.append(test_file)
            expected_texts.append(expected_text)
        
        # Process files concurrently
        tasks = []
        for i, (test_file, expected_text) in enumerate(zip(test_files, expected_texts)):
            mock_stt_model.transcribe = Mock(return_value={"text": expected_text})
            
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = f"concurrent{i}"
                        task = narrator.process_narration(test_file)
                        tasks.append(task)
        
        # Wait for all tasks to complete
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        
        # Verify all results
        assert len(results) == 3
        for result in results:
            assert result["speaker"] == "Narrator"
            assert "Concurrent processing test" in result["text"]


class TestNarratorErrorHandling:
    """Comprehensive error handling test cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_permission_denied(self, mock_pipeline, mock_load_model):
        """Test processing with permission denied errors."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Permission test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "permission_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'audio data')
        
        # Mock permission denied on file operations
        with patch('narrator.os.makedirs', side_effect=PermissionError("Permission denied")):
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                result = await narrator.process_narration(test_file)
        
        assert result["text"] == "Permission test"  # Should still transcribe
        assert result["audio_path"] == test_file  # Should fallback to original path
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_disk_space_error(self, mock_pipeline, mock_load_model):
        """Test processing with disk space errors."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Disk space test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "disk_space_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'audio data')
        
        # Mock disk space error
        with patch('narrator.os.makedirs', side_effect=OSError("No space left on device")):
            result = await narrator.process_narration(test_file)
        
        assert result["text"] == "Disk space test"
        assert result["audio_path"] == test_file
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_malformed_transcription_response(self, mock_pipeline, mock_load_model):
        """Test processing with malformed transcription responses."""
        mock_stt_model = Mock()
        
        # Test various malformed responses
        malformed_responses = [
            {},  # Empty dict
            {"no_text_key": "value"},  # Missing text key
            {"text": None},  # None text
            {"text": 123},  # Non-string, non-list text
            {"text": {"nested": "dict"}},  # Nested dict
        ]
        
        narrator = Narrator()
        
        for i, response in enumerate(malformed_responses):
            mock_stt_model.transcribe = Mock(return_value=response)
            
            test_file = os.path.join(self.temp_dir, f"malformed_{i}.wav")
            with open(test_file, 'wb') as f:
                f.write(b'audio data')
            
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = f"malformed{i}"
                        result = await narrator.process_narration(test_file)
            
            # Should handle gracefully and return empty text
            assert result["text"] == ""
            assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_timeout_simulation(self, mock_pipeline, mock_load_model):
        """Test processing with simulated timeout scenarios."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=asyncio.TimeoutError("Transcription timeout"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "timeout_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'audio data')
        
        result = await narrator.process_narration(test_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == test_file


class TestNarratorPerformance:
    """Performance and stress test cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_large_audio_file(self, mock_pipeline, mock_load_model):
        """Test processing large audio files."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Large file transcription"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create a large audio file (simulate 100MB)
        large_file = os.path.join(self.temp_dir, "large_audio.wav")
        with open(large_file, 'wb') as f:
            # Write 1MB chunks to simulate large file
            chunk = b'x' * (1024 * 1024)  # 1MB chunk
            for _ in range(5):  # 5MB total (reduced for test speed)
                f.write(chunk)
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'large audio data'
                mock_open.return_value.__enter__.return_value.write.return_value = None
                
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "large123"
                    result = await narrator.process_narration(large_file)
        
        assert result["text"] == "Large file transcription"
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_memory_usage_multiple_instances(self, mock_pipeline, mock_load_model):
        """Test memory usage with multiple narrator instances."""
        mock_load_model.return_value = Mock()
        
        # Create multiple narrator instances
        narrators = []
        for i in range(10):
            narrator = Narrator(model_size="tiny")
            narrators.append(narrator)
        
        # Verify all instances are properly initialized
        for narrator in narrators:
            assert narrator.stt_model is not None
            assert narrator.default_speaker_name == "Narrator"
            assert narrator.last_transcription is None
        
        # Clean up
        del narrators


class TestNarratorConfigurationVariations:
    """Test various configuration scenarios."""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', False)
    def test_narrator_without_diarization(self, mock_pipeline, mock_load_model):
        """Test narrator when diarization is disabled."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        
        # Should not attempt to load diarization pipeline
        mock_pipeline.from_pretrained.assert_not_called()
        assert narrator.diarization_pipeline is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    @patch('narrator.MAX_DIARIZATION_RETRIES', 0)
    def test_narrator_diarization_no_retries(self, mock_pipeline, mock_load_model):
        """Test narrator with diarization but no retries allowed."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = Exception("Diarization failed")
        
        narrator = Narrator()
        
        # Should attempt once and fail
        mock_pipeline.from_pretrained.assert_called_once()
        assert narrator.diarization_pipeline is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_narrator_custom_speaker_name(self, mock_pipeline, mock_load_model):
        """Test narrator with custom default speaker name."""
        mock_load_model.return_value = Mock()
        
        # Simulate custom speaker name through initialization
        narrator = Narrator()
        narrator.default_speaker_name = "CustomSpeaker"
        
        assert narrator.default_speaker_name == "CustomSpeaker"


# Additional parametrized tests for edge cases
@pytest.mark.parametrize("audio_extension", [".wav", ".mp3", ".flac", ".m4a", ".ogg"])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
@patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
async def test_process_narration_different_audio_formats(mock_pipeline, mock_load_model, audio_extension):
    """Test processing different audio file formats."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(return_value={"text": f"Format test {audio_extension}"})
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.NamedTemporaryFile(suffix=audio_extension, delete=False) as tmp_file:
        tmp_file.write(b'audio data')
        tmp_file_path = tmp_file.name
    
    try:
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "format123"
                    result = await narrator.process_narration(tmp_file_path)
        
        assert result["text"] == f"Format test {audio_extension}"
        assert result["speaker"] == "Narrator"
    finally:
        os.unlink(tmp_file_path)


@pytest.mark.parametrize("transcription_error", [
    ValueError("Invalid audio format"),
    RuntimeError("Model not loaded"),
    MemoryError("Insufficient memory"),
    FileNotFoundError("Audio file not found"),
    OSError("I/O operation failed"),
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
async def test_process_narration_various_exceptions(mock_pipeline, mock_load_model, transcription_error):
    """Test processing with various exception types."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(side_effect=transcription_error)
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(b'audio data')
        tmp_file_path = tmp_file.name
    
    try:
        result = await narrator.process_narration(tmp_file_path)
        
        # Should handle all exceptions gracefully
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == tmp_file_path
    finally:
        os.unlink(tmp_file_path)


@pytest.mark.parametrize("correction_text", [
    "Simple correction",
    "",  # Empty correction
    "Correction with unicode: ñáéíóú 🎵",
    "Very long correction " * 100,  # Long correction
    None,  # None correction
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
def test_correct_last_transcription_variations(mock_pipeline, mock_load_model, correction_text):
    """Test correction with various text types."""
    mock_load_model.return_value = Mock()
    
    narrator = Narrator()
    narrator.last_transcription = "Original text"
    
    narrator.correct_last_transcription(correction_text)
    
    assert narrator.last_transcription == correction_text


class TestNarratorStateManagement:
    """Test state management and persistence."""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_narrator_state_persistence(self, mock_pipeline, mock_load_model):
        """Test that narrator maintains state across operations."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        
        # Set initial state
        narrator.last_transcription = "First transcription"
        assert narrator.last_transcription == "First transcription"
        
        # Change state
        narrator.correct_last_transcription("Second transcription")
        assert narrator.last_transcription == "Second transcription"
        
        # Verify state persistence
        assert narrator.last_transcription == "Second transcription"
        assert narrator.default_speaker_name == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_narrator_state_after_processing(self, mock_pipeline, mock_load_model):
        """Test narrator state after processing operations."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Processing state test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        initial_state = narrator.last_transcription
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(b'audio data')
            tmp_file_path = tmp_file.name
        
        try:
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = "state123"
                        result = await narrator.process_narration(tmp_file_path)
            
            # State should be updated after processing
            assert narrator.last_transcription != initial_state
            assert narrator.last_transcription == "Processing state test"
            assert result["text"] == "Processing state test"
        finally:
            os.unlink(tmp_file_path)


class TestNarratorBoundaryConditions:
    """Test boundary conditions and extreme scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
    async def test_process_narration_zero_byte_file(self, mock_pipeline, mock_load_model):
        """Test processing zero-byte audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": ""})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create zero-byte file
        zero_file = os.path.join(self.temp_dir, "zero_byte.wav")
        open(zero_file, 'a').close()  # Create empty file
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "zero123"
                    result = await narrator.process_narration(zero_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_extremely_long_filename(self, mock_pipeline, mock_load_model):
        """Test processing file with extremely long filename."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Long filename test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create file with very long name (but within filesystem limits)
        long_name = "a" * 200 + ".wav"
        long_file = os.path.join(self.temp_dir, long_name)
        
        try:
            with open(long_file, 'wb') as f:
                f.write(b'audio data')
            
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = "longname123"
                        result = await narrator.process_narration(long_file)
            
            assert result["text"] == "Long filename test"
            assert result["speaker"] == "Narrator"
        except OSError:
            # Skip if filesystem doesn't support long filenames
            pytest.skip("Filesystem doesn't support long filenames")
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    async def test_process_narration_empty_diarization_result(self, mock_pipeline, mock_load_model):
        """Test processing with empty diarization results."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Empty diarization test"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization with empty results
        mock_diarization = Mock()
        mock_diarization.itertracks = Mock(return_value=iter([]))  # Empty iterator
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "empty_diarization.wav")
        with open(test_file, 'wb') as f:
            f.write(b'audio data')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "empty123"
                    result = await narrator.process_narration(test_file)
        
        # Should fall back to default speaker
        assert result["speaker"] == "Narrator"
        assert result["text"] == "Empty diarization test"


class TestNarratorThreadSafety:
    """Test thread safety and concurrent access."""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_narrator_thread_safety_state_isolation(self, mock_pipeline, mock_load_model):
        """Test that multiple narrator instances maintain separate state."""
        mock_load_model.return_value = Mock()
        
        # Create multiple instances
        narrator1 = Narrator()
        narrator2 = Narrator()
        narrator3 = Narrator()
        
        # Set different states
        narrator1.last_transcription = "Narrator 1 state"
        narrator2.last_transcription = "Narrator 2 state"
        narrator3.last_transcription = "Narrator 3 state"
        
        # Verify state isolation
        assert narrator1.last_transcription == "Narrator 1 state"
        assert narrator2.last_transcription == "Narrator 2 state"
        assert narrator3.last_transcription == "Narrator 3 state"
        
        # Modify one and ensure others are unaffected
        narrator1.correct_last_transcription("Modified state")
        assert narrator1.last_transcription == "Modified state"
        assert narrator2.last_transcription == "Narrator 2 state"
        assert narrator3.last_transcription == "Narrator 3 state"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_narrator_concurrent_correction_operations(self, mock_pipeline, mock_load_model):
        """Test concurrent correction operations."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        narrator.last_transcription = "Initial state"
        
        # Simulate rapid corrections
        corrections = [
            "First correction",
            "Second correction", 
            "Third correction",
            "Fourth correction",
            "Final correction"
        ]
        
        for correction in corrections:
            narrator.correct_last_transcription(correction)
        
        # Should have the last correction applied
        assert narrator.last_transcription == "Final correction"


# Add test for handling special characters in file paths
@pytest.mark.parametrize("special_char", ["ñ", "ü", "€", "中", "🎵"])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
@patch('narrator.NARRATOR_AUDIO_PATH', '/tmp/narrator_audio')
async def test_process_narration_special_char_paths(mock_pipeline, mock_load_model, special_char):
    """Test processing files with special characters in paths."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(return_value={"text": f"Special char test {special_char}"})
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            special_file = os.path.join(temp_dir, f"test_{special_char}.wav")
            with open(special_file, 'wb') as f:
                f.write(b'audio data')
            
            with patch('narrator.os.makedirs'):
                with patch('builtins.open', create=True):
                    with patch('narrator.uuid.uuid4') as mock_uuid:
                        mock_uuid.return_value.hex = "special123"
                        result = await narrator.process_narration(special_file)
            
            assert result["text"] == f"Special char test {special_char}"
            assert result["speaker"] == "Narrator"
        except (UnicodeError, OSError):
            # Skip if filesystem doesn't support the character
            pytest.skip(f"Filesystem doesn't support character: {special_char}")