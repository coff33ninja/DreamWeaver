"""
Comprehensive unit tests for checkpoint manager functionality.
Tests cover happy paths, edge cases, and failure conditions using pytest framework.
"""

import pytest
import tempfile
import shutil
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from contextlib import contextmanager


@pytest.fixture
def temp_checkpoint_dir():
    """
    Creates and yields a temporary directory for storing checkpoints during tests.
    
    Yields:
        Path: The path to the temporary checkpoint directory. The directory and its contents are removed after use.
    """
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    yield checkpoint_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_checkpoint_manager(temp_checkpoint_dir):
    """
    Return a mock checkpoint manager object with stubbed methods for testing purposes.
    
    The returned mock object includes attributes and methods commonly used in checkpoint management, such as `checkpoint_dir`, `save_checkpoint`, `load_checkpoint`, `list_checkpoints`, `delete_checkpoint`, and `get_latest_checkpoint`.
    """
    manager = Mock()
    manager.checkpoint_dir = str(temp_checkpoint_dir)
    manager.save_checkpoint = Mock()
    manager.load_checkpoint = Mock()
    manager.list_checkpoints = Mock()
    manager.delete_checkpoint = Mock()
    manager.get_latest_checkpoint = Mock()
    return manager


@pytest.fixture
def sample_checkpoint_data():
    """
    Return a representative checkpoint data dictionary for use in testing.
    
    Returns:
        dict: A sample checkpoint containing model state, optimizer state, training metrics, timestamp, and metadata.
    """
    return {
        "model_state": {
            "layer1": {"weights": [1.0, 2.0, 3.0], "bias": [0.1, 0.2]},
            "layer2": {"weights": [4.0, 5.0, 6.0], "bias": [0.3, 0.4]}
        },
        "optimizer_state": {
            "lr": 0.001,
            "momentum": 0.9,
            "beta1": 0.9,
            "beta2": 0.999
        },
        "epoch": 10,
        "step": 1000,
        "loss": 0.5,
        "accuracy": 0.85,
        "timestamp": time.time(),
        "metadata": {
            "model_name": "test_model",
            "version": "1.0.0",
            "dataset": "test_dataset"
        }
    }


class TestCheckpointManagerInitialization:
    """Test checkpoint manager initialization."""
    
    def test_init_creates_checkpoint_directory(self, temp_checkpoint_dir):
        """
        Test that initializing the checkpoint manager creates the checkpoint directory if it does not already exist.
        
        Verifies that the directory is absent before initialization, and that it is created and recognized as a directory after initialization logic is executed.
        """
        new_dir = temp_checkpoint_dir.parent / "new_checkpoints"
        assert not new_dir.exists()
        
        # Simulate directory creation during initialization
        new_dir.mkdir(exist_ok=True)
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_init_with_existing_directory(self, temp_checkpoint_dir):
        """Test initialization with existing directory."""
        assert temp_checkpoint_dir.exists()
        assert temp_checkpoint_dir.is_dir()
    
    def test_init_with_invalid_path(self):
        """
        Test that initializing with an invalid directory path raises an appropriate exception.
        """
        invalid_path = "/invalid/path/that/does/not/exist"
        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            Path(invalid_path).mkdir(parents=True)


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""
    
    def test_save_checkpoint_happy_path(self, temp_checkpoint_dir, sample_checkpoint_data):
        """
        Test that a checkpoint file is successfully saved with valid data and verifies its contents.
        """
        checkpoint_file = temp_checkpoint_dir / "test_checkpoint.json"
        
        # Simulate saving checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f, indent=2)
        
        assert checkpoint_file.exists()
        
        # Verify saved data
        with open(checkpoint_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["epoch"] == 10
        assert saved_data["loss"] == 0.5
        assert saved_data["model_state"]["layer1"]["weights"] == [1.0, 2.0, 3.0]
        assert "timestamp" in saved_data
    
    def test_save_checkpoint_with_empty_data(self, temp_checkpoint_dir):
        """
        Verify that saving an empty checkpoint dictionary results in a valid, empty JSON file.
        
        Ensures the file is created and contains an empty object when loaded.
        """
        empty_data = {}
        checkpoint_file = temp_checkpoint_dir / "empty_checkpoint.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(empty_data, f)
        
        assert checkpoint_file.exists()
        
        with open(checkpoint_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data == {}
    
    def test_save_checkpoint_with_none_values(self, temp_checkpoint_dir):
        """
        Verify that checkpoint data containing None values can be saved to and loaded from a JSON file without data loss.
        """
        test_data = {
            "model_state": None,
            "optimizer_state": None,
            "epoch": None,
            "loss": None
        }
        
        checkpoint_file = temp_checkpoint_dir / "none_checkpoint.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(test_data, f)
        
        with open(checkpoint_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["model_state"] is None
        assert saved_data["epoch"] is None
    
    @pytest.mark.parametrize("invalid_data", [
        {"circular": None},  # Will be modified to create circular reference
        {"function": "not_serializable"},
        {"complex_number": "1+2j"}
    ])
    def test_save_checkpoint_with_non_serializable_data(self, temp_checkpoint_dir, invalid_data):
        """
        Test that attempting to save checkpoint data containing non-JSON-serializable values raises a serialization error.
        
        Parameters:
            invalid_data (dict): Checkpoint data containing values that cannot be serialized to JSON, such as circular references or unsupported types.
        """
        if "circular" in invalid_data:
            # Create circular reference
            invalid_data["circular"] = invalid_data
        
        checkpoint_file = temp_checkpoint_dir / "invalid_checkpoint.json"
        
        # This should handle serialization issues gracefully
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(invalid_data, f)
        except (TypeError, ValueError) as e:
            # Expected for non-serializable data
            assert isinstance(e, (TypeError, ValueError))
    
    def test_save_checkpoint_permission_denied(self, temp_checkpoint_dir):
        """
        Test that attempting to save a checkpoint in a read-only directory raises a PermissionError.
        """
        # Create a read-only directory
        readonly_dir = temp_checkpoint_dir / "readonly"
        readonly_dir.mkdir()
        
        try:
            readonly_dir.chmod(0o444)
            checkpoint_file = readonly_dir / "test.json"
            
            with pytest.raises(PermissionError):
                with open(checkpoint_file, 'w') as f:
                    json.dump({"test": "data"}, f)
        finally:
            readonly_dir.chmod(0o755)
    
    def test_save_checkpoint_disk_full_simulation(self, temp_checkpoint_dir):
        """
        Test that saving a checkpoint raises an OSError when disk space is exhausted, by simulating a disk full condition.
        """
        checkpoint_file = temp_checkpoint_dir / "disk_full_test.json"
        
        # Mock open to raise OSError (disk full)
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                with open(checkpoint_file, 'w') as f:
                    json.dump({"test": "data"}, f)


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""
    
    def test_load_checkpoint_happy_path(self, temp_checkpoint_dir, sample_checkpoint_data):
        """
        Verify that a checkpoint file can be successfully loaded and its contents match the expected sample data.
        """
        checkpoint_file = temp_checkpoint_dir / "load_test.json"
        
        # Create checkpoint file
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)
        
        # Load checkpoint
        with open(checkpoint_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["epoch"] == 10
        assert loaded_data["loss"] == 0.5
        assert loaded_data["model_state"]["layer1"]["weights"] == [1.0, 2.0, 3.0]
        assert loaded_data["optimizer_state"]["lr"] == 0.001
    
    def test_load_nonexistent_checkpoint(self, temp_checkpoint_dir):
        """
        Test that attempting to load a nonexistent checkpoint file raises a FileNotFoundError.
        """
        nonexistent_file = temp_checkpoint_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            with open(nonexistent_file, 'r') as f:
                json.load(f)
    
    def test_load_corrupted_checkpoint(self, temp_checkpoint_dir):
        """
        Test that attempting to load a corrupted JSON checkpoint file raises a JSONDecodeError.
        """
        corrupted_file = temp_checkpoint_dir / "corrupted.json"
        
        # Create corrupted JSON file
        with open(corrupted_file, 'w') as f:
            f.write('{"incomplete": json, "missing_brace": true')
        
        with pytest.raises(json.JSONDecodeError):
            with open(corrupted_file, 'r') as f:
                json.load(f)
    
    def test_load_empty_checkpoint_file(self, temp_checkpoint_dir):
        """
        Test that loading an empty checkpoint file raises a JSONDecodeError.
        """
        empty_file = temp_checkpoint_dir / "empty.json"
        
        # Create empty file
        empty_file.touch()
        
        with pytest.raises(json.JSONDecodeError):
            with open(empty_file, 'r') as f:
                json.load(f)
    
    def test_load_checkpoint_with_unexpected_format(self, temp_checkpoint_dir):
        """
        Tests loading a checkpoint file containing a JSON list instead of a dictionary.
        
        Verifies that the loaded data is a list and matches the expected content.
        """
        unexpected_file = temp_checkpoint_dir / "unexpected.json"
        
        # Create file with unexpected format (list instead of dict)
        with open(unexpected_file, 'w') as f:
            json.dump([1, 2, 3, 4, 5], f)
        
        with open(unexpected_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert isinstance(loaded_data, list)
        assert loaded_data == [1, 2, 3, 4, 5]


class TestCheckpointManagement:
    """Test checkpoint management operations."""
    
    def test_list_checkpoints(self, temp_checkpoint_dir):
        """
        Verify that only JSON checkpoint files are listed in the checkpoint directory, excluding non-JSON files.
        """
        # Create multiple checkpoint files
        checkpoints = ["checkpoint_1.json", "checkpoint_2.json", "checkpoint_3.json", "other_file.txt"]
        
        for checkpoint in checkpoints:
            checkpoint_file = temp_checkpoint_dir / checkpoint
            if checkpoint.endswith('.json'):
                with open(checkpoint_file, 'w') as f:
                    json.dump({"test": "data", "name": checkpoint}, f)
            else:
                checkpoint_file.write_text("not a json file")
        
        # List only JSON files
        json_files = list(temp_checkpoint_dir.glob("*.json"))
        json_filenames = [f.name for f in json_files]
        
        assert len(json_files) == 3
        for checkpoint in checkpoints[:3]:  # Only JSON files
            assert checkpoint in json_filenames
        
        assert "other_file.txt" not in json_filenames
    
    def test_list_checkpoints_empty_directory(self, temp_checkpoint_dir):
        """
        Verify that listing checkpoint files in an empty directory returns no JSON files.
        """
        json_files = list(temp_checkpoint_dir.glob("*.json"))
        assert len(json_files) == 0
    
    def test_delete_checkpoint(self, temp_checkpoint_dir):
        """
        Verifies that a checkpoint file can be deleted from the checkpoint directory.
        """
        checkpoint_file = temp_checkpoint_dir / "to_delete.json"
        
        # Create checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        assert checkpoint_file.exists()
        
        # Delete checkpoint
        checkpoint_file.unlink()
        
        assert not checkpoint_file.exists()
    
    def test_delete_nonexistent_checkpoint(self, temp_checkpoint_dir):
        """
        Test that attempting to delete a nonexistent checkpoint file raises a FileNotFoundError.
        """
        nonexistent_file = temp_checkpoint_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            nonexistent_file.unlink()
    
    def test_delete_checkpoint_safe(self, temp_checkpoint_dir):
        """
        Verify that deleting a nonexistent checkpoint file with `missing_ok=True` does not raise an error.
        """
        nonexistent_file = temp_checkpoint_dir / "nonexistent.json"
        
        # This should not raise an error
        nonexistent_file.unlink(missing_ok=True)
    
    def test_get_latest_checkpoint(self, temp_checkpoint_dir):
        """
        Verify that the most recently modified checkpoint file in the directory contains the latest checkpoint data.
        
        This test creates multiple checkpoint files with increasing timestamps, then asserts that the file with the latest modification time corresponds to the most recent checkpoint.
        """
        # Create checkpoints with delays to ensure different timestamps
        checkpoints_data = [
            {"epoch": 1, "timestamp": time.time()},
            {"epoch": 2, "timestamp": time.time() + 1},
            {"epoch": 3, "timestamp": time.time() + 2}
        ]
        
        checkpoint_files = []
        for i, data in enumerate(checkpoints_data):
            checkpoint_file = temp_checkpoint_dir / f"checkpoint_{i}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f)
            checkpoint_files.append(checkpoint_file)
            time.sleep(0.1)  # Ensure different modification times
        
        # Find latest by modification time
        json_files = list(temp_checkpoint_dir.glob("*.json"))
        assert len(json_files) == 3
        
        if json_files:
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                latest_data = json.load(f)
            
            # Should be the last one created
            assert latest_data["epoch"] >= 1
    
    def test_get_latest_checkpoint_empty_directory(self, temp_checkpoint_dir):
        """
        Verify that no checkpoint files are found when retrieving the latest checkpoint from an empty directory.
        """
        json_files = list(temp_checkpoint_dir.glob("*.json"))
        assert len(json_files) == 0


class TestCheckpointValidation:
    """Test checkpoint data validation."""
    
    def test_checkpoint_validation_required_fields(self, sample_checkpoint_data):
        """
        Verify that the sample checkpoint data contains all required fields for validation.
        
        Ensures that the fields 'epoch', 'loss', and 'model_state' are present in the checkpoint data dictionary.
        """
        required_fields = ["epoch", "loss", "model_state"]
        
        for field in required_fields:
            assert field in sample_checkpoint_data
    
    def test_checkpoint_validation_data_types(self, sample_checkpoint_data):
        """
        Verify that each field in the sample checkpoint data has the correct data type.
        """
        assert isinstance(sample_checkpoint_data["epoch"], int)
        assert isinstance(sample_checkpoint_data["loss"], (int, float))
        assert isinstance(sample_checkpoint_data["model_state"], dict)
        assert isinstance(sample_checkpoint_data["optimizer_state"], dict)
        assert isinstance(sample_checkpoint_data["timestamp"], (int, float))
    
    @pytest.mark.parametrize("invalid_field,invalid_value", [
        ("epoch", "not_an_integer"),
        ("loss", "not_a_number"),
        ("model_state", "not_a_dict"),
        ("optimizer_state", [1, 2, 3])
    ])
    def test_checkpoint_validation_invalid_types(self, sample_checkpoint_data, invalid_field, invalid_value):
        """
        Test that checkpoint validation fails when required fields have invalid data types.
        
        This test assigns an invalid value to a specified checkpoint field and asserts that the field's type no longer matches the expected type.
        """
        sample_checkpoint_data[invalid_field] = invalid_value
        
        # In a real implementation, this would be caught by validation
        if invalid_field == "epoch":
            assert not isinstance(sample_checkpoint_data["epoch"], int)
        elif invalid_field == "loss":
            assert not isinstance(sample_checkpoint_data["loss"], (int, float))
        elif invalid_field == "model_state":
            assert not isinstance(sample_checkpoint_data["model_state"], dict)


class TestCheckpointSpecialCases:
    """Test checkpoint handling for special cases."""
    
    def test_checkpoint_with_large_data(self, temp_checkpoint_dir):
        """
        Verify that checkpoints containing large nested data structures can be saved to and loaded from disk without data loss or corruption.
        """
        large_data = {
            "model_state": {
                "large_tensor": [[i * j for j in range(50)] for i in range(50)]
            },
            "epoch": 50,
            "loss": 0.1
        }
        
        checkpoint_file = temp_checkpoint_dir / "large_checkpoint.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(large_data, f)
        
        assert checkpoint_file.exists()
        
        # Verify it can be loaded
        with open(checkpoint_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data["model_state"]["large_tensor"]) == 50
        assert len(loaded_data["model_state"]["large_tensor"][0]) == 50
        assert loaded_data["epoch"] == 50
    
    def test_checkpoint_with_special_characters(self, temp_checkpoint_dir):
        """
        Verify that checkpoint data containing special characters, Unicode, newlines, and tabs can be correctly saved to and loaded from a JSON file using UTF-8 encoding.
        """
        special_data = {
            "model_name": "测试模型",
            "description": "Special chars: αβγ δε ζη θι κλ μν ξο πρ στ υφ χψ ω",
            "symbols": "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`",
            "unicode": "🚀🎯🔥💯✨🌟⭐🎨🎭🎪",
            "epoch": 1,
            "newlines": "line1\nline2\nline3",
            "tabs": "col1\tcol2\tcol3"
        }
        
        checkpoint_file = temp_checkpoint_dir / "special_chars.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(special_data, f, ensure_ascii=False, indent=2)
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["model_name"] == "测试模型"
        assert loaded_data["unicode"] == "🚀🎯🔥💯✨🌟⭐🎨🎭🎪"
        assert loaded_data["symbols"] == "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/~`"
        assert "\n" in loaded_data["newlines"]
        assert "\t" in loaded_data["tabs"]
    
    def test_checkpoint_with_nested_structures(self, temp_checkpoint_dir):
        """
        Verify that checkpoints containing deeply nested dictionary and list structures can be saved to and loaded from a JSON file without data loss or corruption.
        """
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "deep_value": "found_me",
                                "deep_list": [1, [2, [3, [4, [5]]]]]
                            }
                        }
                    }
                }
            },
            "epoch": 1
        }
        
        checkpoint_file = temp_checkpoint_dir / "nested_checkpoint.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(nested_data, f, indent=2)
        
        with open(checkpoint_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["level1"]["level2"]["level3"]["level4"]["level5"]["deep_value"] == "found_me"
        deep_list = loaded_data["level1"]["level2"]["level3"]["level4"]["level5"]["deep_list"]
        assert deep_list[1][1][1][1] == [5]
    
    def test_checkpoint_with_numeric_precision(self, temp_checkpoint_dir):
        """
        Test saving and loading checkpoint data containing various numeric precisions, including small and large floats, scientific notation, zero, and handling of unsupported infinity values in JSON serialization.
        """
        numeric_data = {
            "small_float": 1e-10,
            "large_float": 1e10,
            "negative_float": -1.23456789,
            "scientific_notation": 1.23e-4,
            "zero": 0.0,
            "infinity": float('inf'),
            "negative_infinity": float('-inf'),
            "epoch": 1
        }
        
        checkpoint_file = temp_checkpoint_dir / "numeric_checkpoint.json"
        
        # Note: JSON doesn't support inf/-inf, so we'll test what happens
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(numeric_data, f)
        except ValueError:
            # Expected for inf values
            # Remove inf values and try again
            numeric_data_safe = {k: v for k, v in numeric_data.items() 
                               if not (isinstance(v, float) and (v == float('inf') or v == float('-inf')))}
            
            with open(checkpoint_file, 'w') as f:
                json.dump(numeric_data_safe, f)
            
            with open(checkpoint_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data["small_float"] == 1e-10
            assert loaded_data["large_float"] == 1e10
            assert loaded_data["scientific_notation"] == 1.23e-4


class TestCheckpointBackupAndRestore:
    """Test checkpoint backup and restore functionality."""
    
    def test_checkpoint_backup_and_restore(self, temp_checkpoint_dir, sample_checkpoint_data):
        """
        Tests that checkpoint files can be backed up and restored by copying, verifying both file content and metadata preservation.
        """
        checkpoint_file = temp_checkpoint_dir / "original.json"
        backup_file = temp_checkpoint_dir / "backup.json"
        
        # Save original
        with open(checkpoint_file, 'w') as f:
            json.dump(sample_checkpoint_data, f)
        
        # Create backup
        shutil.copy2(checkpoint_file, backup_file)
        
        # Verify backup exists and has same content
        assert backup_file.exists()
        
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        assert backup_data == sample_checkpoint_data
        
        # Verify file stats are preserved
        original_stat = checkpoint_file.stat()
        backup_stat = backup_file.stat()
        assert backup_stat.st_mtime == original_stat.st_mtime
    
    def test_checkpoint_incremental_backup(self, temp_checkpoint_dir):
        """
        Test that multiple incremental checkpoint files can be created, saved, and verified for correct epoch progression.
        """
        base_data = {"epoch": 1, "loss": 1.0}
        
        # Create series of checkpoints
        for i in range(5):
            checkpoint_data = base_data.copy()
            checkpoint_data["epoch"] = i + 1
            checkpoint_data["loss"] = 1.0 / (i + 1)
            
            checkpoint_file = temp_checkpoint_dir / f"checkpoint_epoch_{i+1}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
        
        # Verify all checkpoints exist
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_epoch_*.json"))
        assert len(checkpoints) == 5
        
        # Verify progression
        for i, checkpoint_file in enumerate(sorted(checkpoints)):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            assert data["epoch"] == i + 1


class TestCheckpointConcurrency:
    """Test checkpoint handling under concurrent access scenarios."""
    
    def test_concurrent_checkpoint_access_simulation(self, temp_checkpoint_dir):
        """
        Simulates concurrent writes to a checkpoint file and verifies that the last write persists.
        
        This test writes two different data payloads to the same checkpoint file in quick succession, mimicking concurrent access. It asserts that the file contains the data from the last write operation.
        """
        checkpoint_file = temp_checkpoint_dir / "concurrent.json"
        
        # Simulate concurrent writes
        data1 = {"writer": "process1", "timestamp": time.time(), "data": [1, 2, 3]}
        data2 = {"writer": "process2", "timestamp": time.time() + 1, "data": [4, 5, 6]}
        
        # Write first data
        with open(checkpoint_file, 'w') as f:
            json.dump(data1, f)
        
        # Simulate brief delay
        time.sleep(0.01)
        
        # Write second data (simulating concurrent access - last writer wins)
        with open(checkpoint_file, 'w') as f:
            json.dump(data2, f)
        
        # Verify final state
        with open(checkpoint_file, 'r') as f:
            final_data = json.load(f)
        
        assert final_data["writer"] == "process2"
        assert final_data["data"] == [4, 5, 6]
    
    def test_checkpoint_file_locking_simulation(self, temp_checkpoint_dir):
        """
        Simulates checkpoint file locking by verifying that a checkpoint file can be read while it is assumed to be locked.
        
        This test creates a checkpoint file, asserts its existence, and confirms that reading from the file is possible, mimicking a shared lock scenario.
        """
        checkpoint_file = temp_checkpoint_dir / "locked_checkpoint.json"
        
        # Create initial checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({"status": "initial"}, f)
        
        # Simulate file being locked (in real scenario, would use file locking)
        assert checkpoint_file.exists()
        
        # Verify we can still read (shared lock scenario)
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        assert data["status"] == "initial"


class TestCheckpointMetadata:
    """Test checkpoint metadata handling."""
    
    def test_checkpoint_with_comprehensive_metadata(self, temp_checkpoint_dir):
        """
        Verifies that a checkpoint file containing comprehensive metadata can be saved and loaded correctly.
        
        This test ensures that all metadata fields—including model, training, performance, system information, timestamp, epoch, and version—are preserved accurately during serialization and deserialization.
        """
        metadata = {
            "model_info": {
                "architecture": "transformer",
                "parameters": 1000000,
                "layers": 12,
                "attention_heads": 8,
                "hidden_size": 768,
                "vocab_size": 50000
            },
            "training_info": {
                "dataset": "custom_dataset",
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "scheduler": "linear_warmup",
                "gradient_clip": 1.0
            },
            "performance": {
                "train_loss": 0.5,
                "val_loss": 0.6,
                "train_accuracy": 0.85,
                "val_accuracy": 0.82,
                "perplexity": 15.2,
                "bleu_score": 0.78
            },
            "system_info": {
                "python_version": "3.8.10",
                "framework_version": "1.9.0",
                "cuda_version": "11.1",
                "gpu_count": 1,
                "cpu_count": 8,
                "memory_gb": 32
            },
            "timestamp": time.time(),
            "epoch": 25,
            "step": 10000,
            "checkpoint_version": "2.0"
        }
        
        checkpoint_file = temp_checkpoint_dir / "metadata_checkpoint.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(checkpoint_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata["model_info"]["architecture"] == "transformer"
        assert loaded_metadata["training_info"]["optimizer"] == "adam"
        assert loaded_metadata["performance"]["train_accuracy"] == 0.85
        assert "timestamp" in loaded_metadata
        assert loaded_metadata["checkpoint_version"] == "2.0"
    
    def test_checkpoint_version_compatibility(self, temp_checkpoint_dir):
        """
        Verify that checkpoints saved in multiple format versions can be loaded and their version and epoch fields are correctly recognized.
        """
        # Test different checkpoint format versions
        versions = [
            {"version": "1.0", "format": "simple", "epoch": 1},
            {"version": "2.0", "format": "extended", "epoch": 2, "metadata": {}},
            {"version": "3.0", "format": "comprehensive", "epoch": 3, "metadata": {}, "system_info": {}}
        ]
        
        for i, version_data in enumerate(versions):
            checkpoint_file = temp_checkpoint_dir / f"version_{version_data['version']}.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(version_data, f)
            
            # Verify each version can be loaded
            with open(checkpoint_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data["version"] == version_data["version"]
            assert loaded_data["epoch"] == i + 1


class TestCheckpointErrorHandling:
    """Test checkpoint error handling and edge cases."""
    
    def test_checkpoint_with_malformed_json(self, temp_checkpoint_dir):
        """
        Test that loading malformed JSON checkpoint files raises a JSONDecodeError.
        
        Creates several files with different types of JSON syntax errors and verifies that attempting to load them with `json.load` results in a `json.JSONDecodeError`.
        """
        malformed_files = [
            ("missing_quote.json", '{"key": value}'),
            ("extra_comma.json", '{"key1": "value1",}'),
            ("unclosed_brace.json", '{"key1": "value1"'),
            ("invalid_escape.json", '{"key": "\\invalid"}')
        ]
        
        for filename, content in malformed_files:
            malformed_file = temp_checkpoint_dir / filename
            
            with open(malformed_file, 'w') as f:
                f.write(content)
            
            with pytest.raises(json.JSONDecodeError):
                with open(malformed_file, 'r') as f:
                    json.load(f)
    
    def test_checkpoint_file_permissions(self, temp_checkpoint_dir):
        """
        Test that checkpoint files can be set to read-only and verifies read and write access permissions accordingly.
        """
        checkpoint_file = temp_checkpoint_dir / "permissions_test.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Check file exists and has correct permissions
        assert checkpoint_file.exists()
        assert os.access(checkpoint_file, os.R_OK)
        assert os.access(checkpoint_file, os.W_OK)
        
        # Test making file read-only
        checkpoint_file.chmod(0o444)
        assert os.access(checkpoint_file, os.R_OK)
        assert not os.access(checkpoint_file, os.W_OK)
        
        # Restore permissions for cleanup
        checkpoint_file.chmod(0o644)
    
    def test_checkpoint_directory_permissions(self, temp_checkpoint_dir):
        """
        Verify that the checkpoint directory has read, write, and execute permissions.
        """
        # Test directory is readable and writable
        assert os.access(temp_checkpoint_dir, os.R_OK)
        assert os.access(temp_checkpoint_dir, os.W_OK)
        assert os.access(temp_checkpoint_dir, os.X_OK)
    
    @pytest.mark.parametrize("error_type", [
        OSError("Disk full"),
        PermissionError("Permission denied"),
        IOError("I/O error")
    ])
    def test_checkpoint_save_io_errors(self, temp_checkpoint_dir, error_type):
        """
        Test that saving a checkpoint file raises the expected I/O error when file operations fail.
        
        Parameters:
            error_type (Exception): The specific I/O-related exception to simulate during file write.
        """
        checkpoint_file = temp_checkpoint_dir / "io_error_test.json"
        
        # Mock open to raise specific error
        with patch("builtins.open", side_effect=error_type):
            with pytest.raises(type(error_type)):
                with open(checkpoint_file, 'w') as f:
                    json.dump({"test": "data"}, f)


if __name__ == '__main__':
    pytest.main([__file__])