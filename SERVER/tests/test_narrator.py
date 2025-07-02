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

class TestNarratorAdvancedEdgeCases:
    """Advanced edge cases and boundary conditions for Narrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_narrator_with_none_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with None model size."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator(model_size=None)
        
        # Should handle None gracefully and use default
        mock_load_model.assert_called_once_with(None)
        assert narrator.stt_model is not None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_narrator_with_invalid_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with invalid model size."""
        mock_load_model.side_effect = ValueError("Invalid model size")
        
        narrator = Narrator(model_size="invalid_size")
        
        assert narrator.stt_model is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    def test_diarization_pipeline_memory_error(self, mock_pipeline, mock_load_model):
        """Test diarization initialization with memory error."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = MemoryError("Not enough memory")
        
        with patch('builtins.input', return_value='s'):
            narrator = Narrator()
        
        assert narrator.diarization_pipeline is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    def test_diarization_pipeline_timeout_error(self, mock_pipeline, mock_load_model):
        """Test diarization initialization with timeout error."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = TimeoutError("Connection timeout")
        
        with patch('builtins.input', return_value='s'):
            narrator = Narrator()
        
        assert narrator.diarization_pipeline is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_very_large_file(self, mock_pipeline, mock_load_model):
        """Test processing a very large audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Large file processed"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create a large dummy file
        large_file = os.path.join(self.temp_dir, "large_audio.wav")
        with open(large_file, 'wb') as f:
            f.write(b'x' * 1024 * 1024)  # 1MB file
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'x' * 1024 * 1024
                mock_open.return_value.__enter__.return_value.write.return_value = None
                
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "large123"
                    result = await narrator.process_narration(large_file)
        
        assert result["text"] == "Large file processed"
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_unicode_text(self, mock_pipeline, mock_load_model):
        """Test processing with unicode text in transcription."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "こんにちは 世界 🌍 émojis"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "unicode_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'unicode audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "unicode123"
                    result = await narrator.process_narration(test_file)
        
        assert result["text"] == "こんにちは 世界 🌍 émojis"
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_special_characters_in_path(self, mock_pipeline, mock_load_model):
        """Test processing with special characters in file path."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Special path test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create directory with special characters
        special_dir = os.path.join(self.temp_dir, "special (dir) [test]")
        os.makedirs(special_dir, exist_ok=True)
        test_file = os.path.join(special_dir, "test file (1).wav")
        with open(test_file, 'wb') as f:
            f.write(b'special path audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "special123"
                    result = await narrator.process_narration(test_file)
        
        assert result["text"] == "Special path test"
        assert result["speaker"] == "Narrator"


class TestNarratorConcurrency:
    """Test concurrent processing scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_concurrent_processing_different_files(self, mock_pipeline, mock_load_model):
        """Test processing multiple files concurrently."""
        mock_stt_model = Mock()
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"concurrent_test_{i}.wav")
            with open(test_file, 'wb') as f:
                f.write(f'concurrent audio {i}'.encode())
            files.append(test_file)
        
        # Mock transcribe to return different results for each file
        transcribe_results = [
            {"text": "First concurrent result"},
            {"text": "Second concurrent result"},
            {"text": "Third concurrent result"}
        ]
        mock_stt_model.transcribe.side_effect = transcribe_results
        
        # Process files concurrently
        tasks = []
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "concurrent123"
                    
                    for file_path in files:
                        task = narrator.process_narration(file_path)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
        
        # Verify results
        for i, result in enumerate(results):
            assert result["text"] == transcribe_results[i]["text"]
            assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_concurrent_processing_same_file(self, mock_pipeline, mock_load_model):
        """Test processing the same file concurrently."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Same file concurrent"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "same_file_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'same file audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "same123"
                    
                    # Process same file multiple times concurrently
                    tasks = [narrator.process_narration(test_file) for _ in range(3)]
                    results = await asyncio.gather(*tasks)
        
        # All results should be the same
        for result in results:
            assert result["text"] == "Same file concurrent"
            assert result["speaker"] == "Narrator"


class TestNarratorErrorRecovery:
    """Test error recovery and resilience scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_permission_denied(self, mock_pipeline, mock_load_model):
        """Test processing when file permissions are denied."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=PermissionError("Permission denied"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "permission_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'permission audio')
        
        result = await narrator.process_narration(test_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == test_file
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_disk_full(self, mock_pipeline, mock_load_model):
        """Test processing when disk is full."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Disk full test"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "disk_full_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'disk full audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'audio data'
                mock_open.return_value.__enter__.return_value.write.side_effect = OSError("No space left on device")
                
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "diskfull123"
                    result = await narrator.process_narration(test_file)
        
        assert result["text"] == "Disk full test"
        assert result["audio_path"] == test_file  # Falls back to original path
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_network_interruption(self, mock_pipeline, mock_load_model):
        """Test processing when network is interrupted during model loading."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=ConnectionError("Network interrupted"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "network_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'network audio')
        
        result = await narrator.process_narration(test_file)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_correct_last_transcription_with_very_long_text(self, mock_pipeline, mock_load_model):
        """Test correcting transcription with very long text."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        narrator.last_transcription = "Original"
        
        # Create very long text (10KB)
        long_text = "Very long transcription text. " * 350  # ~10KB
        
        narrator.correct_last_transcription(long_text)
        
        assert narrator.last_transcription == long_text
        assert len(narrator.last_transcription) > 10000
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_correct_last_transcription_with_special_characters(self, mock_pipeline, mock_load_model):
        """Test correcting transcription with special characters."""
        mock_load_model.return_value = Mock()
        
        narrator = Narrator()
        narrator.last_transcription = "Original"
        
        special_text = "Special chars: \n\t\r\0\x01\x02 and unicode: éñ中文🎵"
        
        narrator.correct_last_transcription(special_text)
        
        assert narrator.last_transcription == special_text


class TestNarratorDiarizationAdvanced:
    """Advanced diarization testing scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    async def test_diarization_with_multiple_speakers(self, mock_pipeline, mock_load_model):
        """Test diarization with multiple speakers."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Multiple speakers"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization mock with multiple speakers
        mock_diarization = Mock()
        mock_track = Mock()
        speakers = [
            ('segment1', 'track1', 'SPEAKER_00'),
            ('segment2', 'track2', 'SPEAKER_01'),
            ('segment3', 'track3', 'SPEAKER_02')
        ]
        mock_track.__iter__ = Mock(return_value=iter(speakers))
        mock_diarization.itertracks = Mock(return_value=mock_track)
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
        
        # Should return the first speaker found
        assert result["text"] == "Multiple speakers"
        assert result["speaker"] == "SPEAKER_00"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    async def test_diarization_with_empty_segments(self, mock_pipeline, mock_load_model):
        """Test diarization with empty segments."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Empty segments test"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization mock with empty segments
        mock_diarization = Mock()
        mock_track = Mock()
        mock_track.__iter__ = Mock(return_value=iter([]))  # Empty iterator
        mock_diarization.itertracks = Mock(return_value=mock_track)
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "empty_segments.wav")
        with open(test_file, 'wb') as f:
            f.write(b'empty segments audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "empty123"
                    result = await narrator.process_narration(test_file)
        
        # Should fall back to default speaker
        assert result["text"] == "Empty segments test"
        assert result["speaker"] == "Narrator"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    @patch('narrator.DIARIZATION_ENABLED', True)
    async def test_diarization_with_malformed_data(self, mock_pipeline, mock_load_model):
        """Test diarization with malformed data."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Malformed data test"})
        mock_load_model.return_value = mock_stt_model
        
        # Setup diarization mock with malformed data
        mock_diarization = Mock()
        mock_track = Mock()
        # Simulate malformed data that might cause unpacking errors
        mock_track.__iter__ = Mock(return_value=iter([('incomplete_data',)]))
        mock_diarization.itertracks = Mock(return_value=mock_track)
        mock_pipeline.from_pretrained.return_value = mock_diarization
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "malformed.wav")
        with open(test_file, 'wb') as f:
            f.write(b'malformed audio')
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "malformed123"
                    result = await narrator.process_narration(test_file)
        
        # Should handle malformed data gracefully and fall back to default
        assert result["text"] == "Malformed data test"
        assert result["speaker"] == "Narrator"


class TestNarratorPerformance:
    """Performance and resource management tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_with_timeout_simulation(self, mock_pipeline, mock_load_model):
        """Test processing with simulated timeout."""
        mock_stt_model = Mock()
        
        async def slow_transcribe(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate slow processing
            return {"text": "Slow transcription"}
        
        # Note: Since transcribe is likely synchronous, we'll simulate the delay differently
        mock_stt_model.transcribe = Mock(return_value={"text": "Slow transcription"})
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        test_file = os.path.join(self.temp_dir, "slow_test.wav")
        with open(test_file, 'wb') as f:
            f.write(b'slow audio')
        
        start_time = asyncio.get_event_loop().time()
        
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "slow123"
                    result = await narrator.process_narration(test_file)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 1.0
        assert result["text"] == "Slow transcription"
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    def test_narrator_memory_cleanup(self, mock_pipeline, mock_load_model):
        """Test that narrator properly cleans up resources."""
        mock_stt_model = Mock()
        mock_load_model.return_value = mock_stt_model
        
        # Create multiple narrator instances
        narrators = []
        for i in range(5):
            narrator = Narrator(model_size="tiny")
            narrators.append(narrator)
        
        # Verify all instances are created
        assert len(narrators) == 5
        
        # Clean up references
        for narrator in narrators:
            narrator.stt_model = None
            narrator.diarization_pipeline = None
        
        # Should not raise any exceptions
        del narrators


class TestNarratorInputValidation:
    """Test input validation and sanitization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_empty_file_path(self, mock_pipeline, mock_load_model):
        """Test processing with empty file path."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=FileNotFoundError("File not found"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        result = await narrator.process_narration("")
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == ""
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_none_file_path(self, mock_pipeline, mock_load_model):
        """Test processing with None file path."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=TypeError("Invalid file path"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        result = await narrator.process_narration(None)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] is None
    
    @patch('narrator.load_model')
    @patch('narrator.Pipeline')
    async def test_process_narration_directory_instead_of_file(self, mock_pipeline, mock_load_model):
        """Test processing when given a directory instead of file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=IsADirectoryError("Is a directory"))
        mock_load_model.return_value = mock_stt_model
        
        narrator = Narrator()
        
        result = await narrator.process_narration(self.temp_dir)
        
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == self.temp_dir


# Additional parametrized tests for comprehensive coverage
@pytest.mark.parametrize("transcription_format", [
    {"text": "Simple text"},
    {"text": ["List", "of", "words"]},
    {"text": ""},
    {"text": None},
    {"text": 123},  # Invalid type
    {"text": {"nested": "dict"}},  # Invalid nested structure
    {"segments": [{"text": "segment1"}, {"text": "segment2"}]},  # Alternative structure
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
async def test_process_narration_various_transcription_formats(mock_pipeline, mock_load_model, transcription_format):
    """Test processing with various transcription result formats."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(return_value=transcription_format)
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(b'test audio data')
        tmp_file_path = tmp_file.name
    
    try:
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "format123"
                    result = await narrator.process_narration(tmp_file_path)
        
        # Should handle all formats gracefully
        assert "text" in result
        assert "speaker" in result
        assert "audio_path" in result
        assert result["speaker"] == "Narrator"
        
        # Text should be converted to string format
        assert isinstance(result["text"], str)
        
    finally:
        os.unlink(tmp_file_path)


@pytest.mark.parametrize("error_type", [
    FileNotFoundError,
    PermissionError,
    OSError,
    MemoryError,
    ValueError,
    TypeError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
async def test_process_narration_various_errors(mock_pipeline, mock_load_model, error_type):
    """Test processing with various error types."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(side_effect=error_type("Test error"))
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(b'error test audio')
        tmp_file_path = tmp_file.name
    
    try:
        result = await narrator.process_narration(tmp_file_path)
        
        # Should handle all error types gracefully
        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == tmp_file_path
        
    finally:
        os.unlink(tmp_file_path)


@pytest.mark.parametrize("correction_input", [
    "Normal correction",
    "",
    None,
    123,
    ["list", "input"],
    {"dict": "input"},
    "Unicode: éñ中文🎵",
    "Very long text " * 100,
    "\n\t\r Special chars",
    True,
    False,
    0.123,
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
def test_correct_last_transcription_various_inputs(mock_pipeline, mock_load_model, correction_input):
    """Test correction with various input types."""
    mock_load_model.return_value = Mock()
    
    narrator = Narrator()
    narrator.last_transcription = "Original"
    
    # Should handle all input types without crashing
    narrator.correct_last_transcription(correction_input)
    
    # The correction should be stored as-is
    assert narrator.last_transcription == correction_input


@pytest.mark.parametrize("audio_file_extension", [
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".wma",
    ".aac",
    "",  # No extension
    ".txt",  # Wrong extension
])
@patch('narrator.load_model')
@patch('narrator.Pipeline')
async def test_process_narration_various_file_extensions(mock_pipeline, mock_load_model, audio_file_extension):
    """Test processing with various audio file extensions."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(return_value={"text": f"Processed {audio_file_extension} file"})
    mock_load_model.return_value = mock_stt_model
    
    narrator = Narrator()
    
    with tempfile.NamedTemporaryFile(suffix=audio_file_extension, delete=False) as tmp_file:
        tmp_file.write(b'audio data with extension')
        tmp_file_path = tmp_file.name
    
    try:
        with patch('narrator.os.makedirs'):
            with patch('builtins.open', create=True):
                with patch('narrator.uuid.uuid4') as mock_uuid:
                    mock_uuid.return_value.hex = "ext123"
                    result = await narrator.process_narration(tmp_file_path)
        
        # Should handle all extensions
        assert result["text"] == f"Processed {audio_file_extension} file"
        assert result["speaker"] == "Narrator"
        
    finally:
        os.unlink(tmp_file_path)


if __name__ == '__main__':
    # Run all tests including the new ones
    pytest.main([__file__, '-v', '--tb=short', '--maxfail=10'])