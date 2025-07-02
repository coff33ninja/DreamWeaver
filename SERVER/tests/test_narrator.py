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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock the external dependencies before importing narrator
sys.modules["whisper"] = Mock()
sys.modules["pyannote"] = Mock()
sys.modules["pyannote.audio"] = Mock()

try:
    from narrator import Narrator
except ImportError as e:
    pytest.skip(f"Narrator module not found: {e}", allow_module_level=True)


class TestNarratorInitialization:
    """Test cases for Narrator class initialization."""

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    def test_init_default_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with default model size."""
        mock_load_model.return_value = Mock()

        narrator = Narrator()

        mock_load_model.assert_called_once()
        assert narrator.stt_model is not None
        assert narrator.default_speaker_name == "Narrator"
        assert narrator.last_transcription is None

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    def test_init_custom_model_size(self, mock_pipeline, mock_load_model):
        """Test initialization with custom model size."""
        mock_load_model.return_value = Mock()

        narrator = Narrator(model_size="large")

        mock_load_model.assert_called_once_with("large")
        assert narrator.stt_model is not None

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    def test_init_model_loading_failure(self, mock_pipeline, mock_load_model):
        """Test initialization when model loading fails."""
        mock_load_model.side_effect = Exception("Model loading failed")

        narrator = Narrator()

        assert narrator.stt_model is None

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.DIARIZATION_ENABLED", True)
    def test_init_with_diarization_enabled(self, mock_pipeline, mock_load_model):
        """Test initialization with diarization enabled."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.return_value = Mock()

        narrator = Narrator()

        mock_pipeline.from_pretrained.assert_called_once()
        assert narrator.diarization_pipeline is not None

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.DIARIZATION_ENABLED", True)
    @patch("narrator.MAX_DIARIZATION_RETRIES", 2)
    def test_init_diarization_loading_failure_with_retries(
        self, mock_pipeline, mock_load_model
    ):
        """Test diarization loading with failures and retries."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = Exception(
            "Diarization loading failed"
        )

        with patch("builtins.input", return_value="s"):  # Skip diarization
            narrator = Narrator()

        assert narrator.diarization_pipeline is None

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.DIARIZATION_ENABLED", True)
    @patch("narrator.webbrowser.open")
    @patch("narrator.re.findall")
    def test_init_diarization_with_url_opening(
        self, mock_findall, mock_webbrowser, mock_pipeline, mock_load_model
    ):
        """Test diarization initialization with URL opening for errors."""
        mock_load_model.return_value = Mock()
        mock_pipeline.from_pretrained.side_effect = Exception(
            "Error with https://example.com/token"
        )
        mock_findall.return_value = ["https://example.com/token"]

        with patch("builtins.input", return_value="s"):  # Skip diarization
            narrator = Narrator()

        mock_webbrowser.assert_called_once_with("https://example.com/token")
        assert narrator.diarization_pipeline is None


class TestNarratorProcessNarration:
    """Test cases for the process_narration method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_file = os.path.join(self.temp_dir, "test_audio.wav")

        # Create a dummy audio file
        with open(self.test_audio_file, "wb") as f:
            f.write(b"dummy audio content")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.NARRATOR_AUDIO_PATH", "/tmp/narrator_audio")
    @patch("narrator.os.makedirs")
    @patch("narrator.uuid.uuid4")
    async def test_process_narration_success(
        self, mock_uuid, mock_makedirs, mock_pipeline, mock_load_model
    ):
        """Test successful audio processing."""
        # Setup mocks
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model
        mock_uuid.return_value.hex = "test123"

        narrator = Narrator()

        # Mock file operations
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"audio data"
            )
            mock_open.return_value.__enter__.return_value.write.return_value = None

            result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == "Hello world"
        assert result["speaker"] == "Narrator"
        assert "narration_test123.wav" in result["audio_path"]
        assert narrator.last_transcription == "Hello world"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    async def test_process_narration_no_model(self, mock_pipeline, mock_load_model):
        """Test processing when STT model is not loaded."""
        mock_load_model.side_effect = Exception("Model loading failed")

        narrator = Narrator()
        result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == ""
        assert result["speaker"] == "Narrator"
        assert result["audio_path"] == self.test_audio_file

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.NARRATOR_AUDIO_PATH", "/tmp/narrator_audio")
    async def test_process_narration_transcription_failure(
        self, mock_pipeline, mock_load_model
    ):
        """Test processing when transcription fails."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=Exception("Transcription failed"))
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()

        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True):
                result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == ""
        assert result["speaker"] == "Narrator"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.DIARIZATION_ENABLED", True)
    @patch("narrator.NARRATOR_AUDIO_PATH", "/tmp/narrator_audio")
    async def test_process_narration_with_diarization(
        self, mock_pipeline, mock_load_model
    ):
        """Test processing with diarization enabled."""
        # Setup STT mock
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model

        # Setup diarization mock
        mock_diarization = Mock()
        mock_track = Mock()
        mock_track.__iter__ = Mock(
            return_value=iter([("segment", "track", "SPEAKER_01")])
        )
        mock_diarization.itertracks = Mock(return_value=mock_track)
        mock_pipeline.from_pretrained.return_value = mock_diarization

        narrator = Narrator()

        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True):
                with patch("narrator.uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == "Hello world"
        assert result["speaker"] == "SPEAKER_01"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.DIARIZATION_ENABLED", True)
    @patch("narrator.NARRATOR_AUDIO_PATH", "/tmp/narrator_audio")
    async def test_process_narration_diarization_failure(
        self, mock_pipeline, mock_load_model
    ):
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

        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True):
                with patch("narrator.uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == "Hello world"
        assert result["speaker"] == "Narrator"  # Falls back to default

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.NARRATOR_AUDIO_PATH", "/tmp/narrator_audio")
    async def test_process_narration_text_list_format(
        self, mock_pipeline, mock_load_model
    ):
        """Test processing when transcription returns text as list."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(
            return_value={"text": ["Hello", "world", "test"]}
        )
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()

        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True):
                with patch("narrator.uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == "Hello world test"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.NARRATOR_AUDIO_PATH", "/tmp/narrator_audio")
    async def test_process_narration_file_save_failure(
        self, mock_pipeline, mock_load_model
    ):
        """Test processing when audio file saving fails."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": "Hello world"})
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()

        with patch("narrator.os.makedirs", side_effect=Exception("Permission denied")):
            result = await narrator.process_narration(self.test_audio_file)

        assert result["text"] == "Hello world"
        assert (
            result["audio_path"] == self.test_audio_file
        )  # Falls back to original path


class TestNarratorCorrectLastTranscription:
    """Test cases for the correct_last_transcription method."""

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    def test_correct_last_transcription(self, mock_pipeline, mock_load_model):
        """Test correcting the last transcription."""
        mock_load_model.return_value = Mock()

        narrator = Narrator()
        narrator.last_transcription = "Original transcription"

        narrator.correct_last_transcription("Corrected transcription")

        assert narrator.last_transcription == "Corrected transcription"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    def test_correct_last_transcription_empty_string(
        self, mock_pipeline, mock_load_model
    ):
        """Test correcting with empty string."""
        mock_load_model.return_value = Mock()

        narrator = Narrator()
        narrator.last_transcription = "Original transcription"

        narrator.correct_last_transcription("")

        assert narrator.last_transcription == ""

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    def test_correct_last_transcription_none_input(
        self, mock_pipeline, mock_load_model
    ):
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

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    async def test_process_nonexistent_file(self, mock_pipeline, mock_load_model):
        """Test processing non-existent audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(
            side_effect=FileNotFoundError("File not found")
        )
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()
        nonexistent_file = "/path/to/nonexistent/file.wav"

        result = await narrator.process_narration(nonexistent_file)

        assert result["text"] == ""
        assert result["speaker"] == "Narrator"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    async def test_process_empty_audio_file(self, mock_pipeline, mock_load_model):
        """Test processing empty audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(return_value={"text": ""})
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()

        # Create empty audio file
        empty_file = os.path.join(self.temp_dir, "empty.wav")
        with open(empty_file, "wb") as f:
            pass  # Create empty file

        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True):
                with patch("narrator.uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(empty_file)

        assert result["text"] == ""
        assert result["speaker"] == "Narrator"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    async def test_process_corrupted_audio_file(self, mock_pipeline, mock_load_model):
        """Test processing corrupted audio file."""
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(side_effect=Exception("Corrupted audio"))
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()

        # Create corrupted audio file
        corrupted_file = os.path.join(self.temp_dir, "corrupted.wav")
        with open(corrupted_file, "wb") as f:
            f.write(b"not audio data")

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

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    @patch("narrator.NARRATOR_AUDIO_PATH")
    async def test_full_workflow_integration(
        self, mock_audio_path, mock_pipeline, mock_load_model
    ):
        """Test full workflow from initialization to transcription."""
        mock_audio_path.__str__ = Mock(return_value=self.temp_dir)

        # Setup mocks
        mock_stt_model = Mock()
        mock_stt_model.transcribe = Mock(
            return_value={"text": "Integration test successful"}
        )
        mock_load_model.return_value = mock_stt_model

        # Create test audio file
        test_file = os.path.join(self.temp_dir, "integration_test.wav")
        with open(test_file, "wb") as f:
            f.write(b"integration test audio")

        # Initialize narrator
        narrator = Narrator(model_size="tiny")

        # Process narration
        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    b"audio data"
                )
                mock_open.return_value.__enter__.return_value.write.return_value = None

                with patch("narrator.uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value.hex = "integration123"
                    result = await narrator.process_narration(test_file)

        # Verify results
        assert result["text"] == "Integration test successful"
        assert result["speaker"] == "Narrator"
        assert narrator.last_transcription == "Integration test successful"

        # Test correction
        narrator.correct_last_transcription("Corrected integration test")
        assert narrator.last_transcription == "Corrected integration test"

    @patch("narrator.load_model")
    @patch("narrator.Pipeline")
    async def test_multiple_consecutive_processing(
        self, mock_pipeline, mock_load_model
    ):
        """Test processing multiple audio files consecutively."""
        mock_stt_model = Mock()
        mock_load_model.return_value = mock_stt_model

        narrator = Narrator()

        # Create multiple test files
        files_and_texts = [
            ("file1.wav", "First transcription"),
            ("file2.wav", "Second transcription"),
            ("file3.wav", "Third transcription"),
        ]

        results = []
        for filename, expected_text in files_and_texts:
            test_file = os.path.join(self.temp_dir, filename)
            with open(test_file, "wb") as f:
                f.write(b"test audio")

            mock_stt_model.transcribe = Mock(return_value={"text": expected_text})

            with patch("narrator.os.makedirs"):
                with patch("builtins.open", create=True):
                    with patch("narrator.uuid.uuid4") as mock_uuid:
                        mock_uuid.return_value.hex = f"test{len(results)}"
                        result = await narrator.process_narration(test_file)
                        results.append(result)

        # Verify all results
        for i, (_, expected_text) in enumerate(files_and_texts):
            assert results[i]["text"] == expected_text
            assert results[i]["speaker"] == "Narrator"


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium", "large"])
@patch("narrator.load_model")
@patch("narrator.Pipeline")
def test_narrator_initialization_different_models(
    mock_pipeline, mock_load_model, model_size
):
    """Test narrator initialization with different model sizes."""
    mock_load_model.return_value = Mock()

    narrator = Narrator(model_size=model_size)

    mock_load_model.assert_called_once_with(model_size)
    assert narrator.stt_model is not None


@pytest.mark.parametrize(
    "text_input,expected_length",
    [
        ("", 0),
        ("Hello", 5),
        ("Hello world", 11),
        ("   Hello world   ", 11),  # Should be stripped
        (["Hello", "world"], 11),  # List input
    ],
)
@patch("narrator.load_model")
@patch("narrator.Pipeline")
async def test_process_narration_text_variations(
    mock_pipeline, mock_load_model, text_input, expected_length
):
    """Test processing with various text formats."""
    mock_stt_model = Mock()
    mock_stt_model.transcribe = Mock(return_value={"text": text_input})
    mock_load_model.return_value = mock_stt_model

    narrator = Narrator()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(b"dummy audio")
        tmp_file_path = tmp_file.name

    try:
        with patch("narrator.os.makedirs"):
            with patch("builtins.open", create=True):
                with patch("narrator.uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value.hex = "test123"
                    result = await narrator.process_narration(tmp_file_path)

        if isinstance(text_input, list):
            assert len(result["text"]) == expected_length
        else:
            assert len(result["text"].strip()) == expected_length
    finally:
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
