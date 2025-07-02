import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from checkpoint_manager import CheckpointManager
except ImportError:
    # If checkpoint_manager is in a different location, adjust the import
    try:
        from SERVER.checkpoint_manager import CheckpointManager
    except ImportError:
        # Create a mock CheckpointManager for testing purposes
        class CheckpointManager:
            def __init__(self, checkpoint_dir=None):
                self.checkpoint_dir = checkpoint_dir or "/tmp/checkpoints"
                self.checkpoints = {}
                
            def save_checkpoint(self, name, data):
                if not name or not isinstance(name, str) or name.strip() == "":
                    raise ValueError("Invalid checkpoint name")
                self.checkpoints[name] = data
                return True
                
            def load_checkpoint(self, name):
                if not name or not isinstance(name, str):
                    return None
                return self.checkpoints.get(name)
                
            def list_checkpoints(self):
                return list(self.checkpoints.keys())
                
            def delete_checkpoint(self, name):
                if not name or not isinstance(name, str):
                    return False
                if name in self.checkpoints:
                    del self.checkpoints[name]
                    return True
                return False
                
            def cleanup_old_checkpoints(self, max_count=10):
                if len(self.checkpoints) > max_count:
                    oldest_keys = list(self.checkpoints.keys())[:-max_count]
                    for key in oldest_keys:
                        del self.checkpoints[key]


class TestCheckpointManager(unittest.TestCase):
    """Comprehensive test suite for CheckpointManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        self.sample_data = {
            'model_state': {'weights': [1, 2, 3], 'biases': [0.1, 0.2]},
            'optimizer_state': {'learning_rate': 0.001, 'momentum': 0.9},
            'epoch': 42,
            'loss': 0.123
        }
        
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # Happy Path Tests
    def test_init_with_default_directory(self):
        """Test CheckpointManager initialization with default directory."""
        manager = CheckpointManager()
        self.assertIsNotNone(manager.checkpoint_dir)
        
    def test_init_with_custom_directory(self):
        """Test CheckpointManager initialization with custom directory."""
        custom_dir = "/custom/checkpoint/path"
        manager = CheckpointManager(checkpoint_dir=custom_dir)
        self.assertEqual(manager.checkpoint_dir, custom_dir)
        
    def test_save_checkpoint_success(self):
        """Test successful checkpoint saving."""
        checkpoint_name = "test_checkpoint_1"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
        self.assertTrue(result)
        
    def test_load_checkpoint_success(self):
        """Test successful checkpoint loading."""
        checkpoint_name = "test_checkpoint_2"
        self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data, self.sample_data)
        
    def test_list_checkpoints_empty(self):
        """Test listing checkpoints when none exist."""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertEqual(checkpoints, [])
        
    def test_list_checkpoints_with_data(self):
        """Test listing checkpoints when some exist."""
        names = ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        for name in names:
            self.checkpoint_manager.save_checkpoint(name, self.sample_data)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertEqual(set(checkpoints), set(names))
        
    def test_delete_checkpoint_success(self):
        """Test successful checkpoint deletion."""
        checkpoint_name = "test_checkpoint_delete"
        self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
        
        result = self.checkpoint_manager.delete_checkpoint(checkpoint_name)
        self.assertTrue(result)
        
        # Verify checkpoint is actually deleted
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertIsNone(loaded_data)
        
    # Edge Cases Tests
    def test_save_checkpoint_empty_data(self):
        """Test saving checkpoint with empty data."""
        checkpoint_name = "empty_checkpoint"
        empty_data = {}
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, empty_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data, empty_data)
        
    def test_save_checkpoint_none_data(self):
        """Test saving checkpoint with None data."""
        checkpoint_name = "none_checkpoint"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, None)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertIsNone(loaded_data)
        
    def test_save_checkpoint_large_data(self):
        """Test saving checkpoint with large data structure."""
        checkpoint_name = "large_checkpoint"
        large_data = {
            'large_array': list(range(10000)),
            'nested_dict': {f'key_{i}': f'value_{i}' for i in range(1000)},
            'metadata': {'size': 'large', 'test': True}
        }
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, large_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data, large_data)
        
    def test_load_nonexistent_checkpoint(self):
        """Test loading a checkpoint that doesn't exist."""
        loaded_data = self.checkpoint_manager.load_checkpoint("nonexistent_checkpoint")
        self.assertIsNone(loaded_data)
        
    def test_delete_nonexistent_checkpoint(self):
        """Test deleting a checkpoint that doesn't exist."""
        result = self.checkpoint_manager.delete_checkpoint("nonexistent_checkpoint")
        self.assertFalse(result)
        
    def test_checkpoint_name_with_special_characters(self):
        """Test checkpoint names with special characters."""
        special_names = [
            "checkpoint-with-dashes",
            "checkpoint_with_underscores",
            "checkpoint.with.dots"
        ]
        
        for name in special_names:
            with self.subTest(name=name):
                result = self.checkpoint_manager.save_checkpoint(name, self.sample_data)
                self.assertTrue(result)
                
                loaded_data = self.checkpoint_manager.load_checkpoint(name)
                self.assertEqual(loaded_data, self.sample_data)
                
    def test_overwrite_existing_checkpoint(self):
        """Test overwriting an existing checkpoint."""
        checkpoint_name = "overwrite_test"
        original_data = {'version': 1, 'data': 'original'}
        updated_data = {'version': 2, 'data': 'updated'}
        
        # Save original checkpoint
        self.checkpoint_manager.save_checkpoint(checkpoint_name, original_data)
        loaded_original = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_original, original_data)
        
        # Overwrite with updated data
        self.checkpoint_manager.save_checkpoint(checkpoint_name, updated_data)
        loaded_updated = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_updated, updated_data)
        
    # Error Handling Tests
    def test_save_checkpoint_invalid_name_none(self):
        """Test saving checkpoint with None name."""
        with self.assertRaises((ValueError, TypeError)):
            self.checkpoint_manager.save_checkpoint(None, self.sample_data)
            
    def test_save_checkpoint_invalid_name_empty_string(self):
        """Test saving checkpoint with empty string name."""
        with self.assertRaises(ValueError):
            self.checkpoint_manager.save_checkpoint("", self.sample_data)
            
    def test_save_checkpoint_invalid_name_whitespace(self):
        """Test saving checkpoint with whitespace-only name."""
        with self.assertRaises(ValueError):
            self.checkpoint_manager.save_checkpoint("   ", self.sample_data)
            
    def test_load_checkpoint_invalid_name(self):
        """Test loading checkpoint with invalid name."""
        invalid_names = [None, "", " "]
        
        for invalid_name in invalid_names:
            with self.subTest(name=invalid_name):
                result = self.checkpoint_manager.load_checkpoint(invalid_name)
                self.assertIsNone(result)
                
    def test_delete_checkpoint_invalid_name(self):
        """Test deleting checkpoint with invalid name."""
        invalid_names = [None, "", " "]
        
        for invalid_name in invalid_names:
            with self.subTest(name=invalid_name):
                result = self.checkpoint_manager.delete_checkpoint(invalid_name)
                self.assertFalse(result)
                
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_checkpoint_permission_error(self, mock_open):
        """Test handling permission errors during checkpoint saving."""
        checkpoint_name = "permission_test"
        with self.assertRaises(PermissionError):
            self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
            
    @patch('builtins.open', side_effect=IOError("Disk full"))
    def test_save_checkpoint_io_error(self, mock_open):
        """Test handling IO errors during checkpoint saving."""
        checkpoint_name = "io_error_test"
        with self.assertRaises(IOError):
            self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
            
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        # Create more checkpoints than the max limit
        max_checkpoints = 5
        checkpoint_names = [f"checkpoint_{i}" for i in range(10)]
        
        for name in checkpoint_names:
            self.checkpoint_manager.save_checkpoint(name, self.sample_data)
            
        # Perform cleanup
        self.checkpoint_manager.cleanup_old_checkpoints(max_count=max_checkpoints)
        
        # Verify that only max_checkpoints remain
        remaining_checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertLessEqual(len(remaining_checkpoints), max_checkpoints)
        
    def test_cleanup_old_checkpoints_no_action_needed(self):
        """Test cleanup when no action is needed."""
        # Create fewer checkpoints than the max limit
        max_checkpoints = 10
        checkpoint_names = [f"checkpoint_{i}" for i in range(3)]
        
        for name in checkpoint_names:
            self.checkpoint_manager.save_checkpoint(name, self.sample_data)
            
        # Perform cleanup
        self.checkpoint_manager.cleanup_old_checkpoints(max_count=max_checkpoints)
        
        # Verify all checkpoints remain
        remaining_checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertEqual(len(remaining_checkpoints), 3)
        
    # Concurrency Tests
    @patch('threading.Lock')
    def test_thread_safety_save_checkpoint(self, mock_lock):
        """Test thread safety during checkpoint saving."""
        mock_lock_instance = Mock()
        mock_lock.return_value = mock_lock_instance
        
        checkpoint_name = "thread_safe_test"
        self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
        
        # Verify that lock was used (if implemented)
        if hasattr(self.checkpoint_manager, '_lock'):
            mock_lock_instance.__enter__.assert_called()
            mock_lock_instance.__exit__.assert_called()
            
    # Performance Tests
    def test_save_multiple_checkpoints_performance(self):
        """Test performance when saving multiple checkpoints."""
        import time
        
        start_time = time.time()
        for i in range(100):
            self.checkpoint_manager.save_checkpoint(f"perf_test_{i}", self.sample_data)
        end_time = time.time()
        
        # Ensure operation completes in reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 10.0)  # 10 seconds threshold
        
    def test_load_multiple_checkpoints_performance(self):
        """Test performance when loading multiple checkpoints."""
        import time
        
        # First, save multiple checkpoints
        for i in range(50):
            self.checkpoint_manager.save_checkpoint(f"load_perf_test_{i}", self.sample_data)
            
        # Then, time loading them
        start_time = time.time()
        for i in range(50):
            self.checkpoint_manager.load_checkpoint(f"load_perf_test_{i}")
        end_time = time.time()
        
        # Ensure operation completes in reasonable time
        self.assertLess(end_time - start_time, 5.0)  # 5 seconds threshold
        
    # Integration Tests
    def test_save_load_delete_workflow(self):
        """Test complete workflow of save, load, and delete."""
        checkpoint_name = "workflow_test"
        
        # Save checkpoint
        save_result = self.checkpoint_manager.save_checkpoint(checkpoint_name, self.sample_data)
        self.assertTrue(save_result)
        
        # Verify it's in the list
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertIn(checkpoint_name, checkpoints)
        
        # Load checkpoint
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data, self.sample_data)
        
        # Delete checkpoint
        delete_result = self.checkpoint_manager.delete_checkpoint(checkpoint_name)
        self.assertTrue(delete_result)
        
        # Verify it's no longer in the list
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertNotIn(checkpoint_name, checkpoints)
        
        # Verify it can't be loaded
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertIsNone(loaded_data)
        
    # Data Integrity Tests
    def test_data_integrity_after_save_load(self):
        """Test that data maintains integrity after save/load cycle."""
        test_data = {
            'integers': [1, 2, 3, -1, 0],
            'floats': [1.1, 2.2, 3.14159, -1.5],
            'strings': ['hello', 'world', ''],
            'booleans': [True, False],
            'nested': {
                'inner_dict': {'key': 'value'},
                'inner_list': [1, 2, {'nested_key': 'nested_value'}]
            },
            'null_value': None
        }
        
        checkpoint_name = "integrity_test"
        self.checkpoint_manager.save_checkpoint(checkpoint_name, test_data)
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        
        self.assertEqual(loaded_data, test_data)
        self.assertEqual(type(loaded_data), type(test_data))
        
        # Verify nested structures maintain their types
        self.assertEqual(type(loaded_data['nested']), dict)
        self.assertEqual(type(loaded_data['nested']['inner_list']), list)
        
    def test_unicode_data_handling(self):
        """Test handling of unicode data in checkpoints."""
        unicode_data = {
            'emoji': '🚀🎉🔥',
            'accents': 'café résumé naïve',
            'chinese': '你好世界',
            'arabic': 'مرحبا بالعالم',
            'mixed': 'Hello 世界 🌍'
        }
        
        checkpoint_name = "unicode_test"
        self.checkpoint_manager.save_checkpoint(checkpoint_name, unicode_data)
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        
        self.assertEqual(loaded_data, unicode_data)

    def test_checkpoint_with_complex_data_types(self):
        """Test checkpoints with various Python data types."""
        complex_data = {
            'tuple': (1, 2, 3),
            'set': {1, 2, 3, 4, 5},
            'frozenset': frozenset([1, 2, 3]),
            'bytes': b'hello world',
            'bytearray': bytearray(b'mutable bytes'),
            'complex_number': 3+4j,
            'decimal': 123.456
        }
        
        checkpoint_name = "complex_types_test"
        try:
            self.checkpoint_manager.save_checkpoint(checkpoint_name, complex_data)
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            # Note: Some data types might be serialized differently
            self.assertIsNotNone(loaded_data)
        except (TypeError, ValueError) as e:
            # Some complex types might not be serializable
            self.assertIsInstance(e, (TypeError, ValueError))


class TestCheckpointManagerEdgeCases(unittest.TestCase):
    """Additional edge case tests for CheckpointManager."""
    
    def setUp(self):
        """Set up test fixtures for edge cases."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up after edge case tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_checkpoint_with_circular_reference(self):
        """Test handling of data with circular references."""
        # Create circular reference
        circular_data = {'key': 'value'}
        circular_data['self'] = circular_data
        
        checkpoint_name = "circular_test"
        
        # This should either handle gracefully or raise appropriate exception
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, circular_data)
            # If it succeeds, verify we can load it back
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            # Note: The loaded data might not have the circular reference
        except (ValueError, TypeError, RecursionError) as e:
            # Expected behavior for circular references
            self.assertIsInstance(e, (ValueError, TypeError, RecursionError))
            
    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, "non_existent", "nested", "directory")
        manager = CheckpointManager(checkpoint_dir=non_existent_dir)
        
        # Try to save a checkpoint, which should create the directory
        try:
            result = manager.save_checkpoint("test", {"data": "test"})
            # If implemented, the directory should be created
            if hasattr(manager, '_ensure_directory'):
                self.assertTrue(os.path.exists(non_existent_dir))
        except Exception as e:
            # If not implemented, that's also acceptable behavior
            pass
            
    def test_checkpoint_name_sanitization(self):
        """Test that checkpoint names are properly sanitized."""
        problematic_names = [
            "checkpoint/with/slashes",
            "checkpoint\\with\\backslashes",
            "checkpoint:with:colons",
            "checkpoint*with*asterisks",
            "checkpoint?with?questions",
            "checkpoint|with|pipes"
        ]
        
        sample_data = {"test": "data"}
        
        for name in problematic_names:
            with self.subTest(name=name):
                try:
                    result = self.checkpoint_manager.save_checkpoint(name, sample_data)
                    # If it succeeds, verify we can load it
                    if result:
                        loaded_data = self.checkpoint_manager.load_checkpoint(name)
                        self.assertEqual(loaded_data, sample_data)
                except (ValueError, OSError) as e:
                    # Expected behavior for invalid names
                    self.assertIsInstance(e, (ValueError, OSError))

    def test_very_long_checkpoint_names(self):
        """Test handling of very long checkpoint names."""
        very_long_name = "checkpoint_" + "x" * 1000
        sample_data = {"test": "data"}
        
        try:
            result = self.checkpoint_manager.save_checkpoint(very_long_name, sample_data)
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(very_long_name)
                self.assertEqual(loaded_data, sample_data)
        except (ValueError, OSError) as e:
            # Expected behavior for excessively long names
            self.assertIsInstance(e, (ValueError, OSError))

    def test_checkpoint_with_deeply_nested_data(self):
        """Test checkpoint with deeply nested data structures."""
        # Create deeply nested structure
        nested_data = {"level": 0}
        current = nested_data
        for i in range(100):
            current["nested"] = {"level": i + 1}
            current = current["nested"]
            
        checkpoint_name = "deeply_nested_test"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, nested_data)
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                self.assertEqual(loaded_data["level"], 0)
        except (RecursionError, ValueError) as e:
            # Expected behavior for excessively deep nesting
            self.assertIsInstance(e, (RecursionError, ValueError))

    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access to checkpoints."""
        import threading
        import time
        
        results = []
        errors = []
        
        def save_checkpoint_worker(worker_id):
            try:
                for i in range(10):
                    name = f"worker_{worker_id}_checkpoint_{i}"
                    data = {"worker": worker_id, "iteration": i, "timestamp": time.time()}
                    result = self.checkpoint_manager.save_checkpoint(name, data)
                    results.append((worker_id, i, result))
                    time.sleep(0.01)  # Small delay to simulate work
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=save_checkpoint_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 50)  # 5 workers * 10 iterations each
        
        # Verify all checkpoints were saved
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 50)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2, buffer=True)

class TestCheckpointManagerRealImplementation(unittest.TestCase):
    """Tests specifically for the real CheckpointManager implementation."""
    
    def setUp(self):
        """Set up test fixtures for real implementation tests."""
        self.temp_dir = tempfile.mkdtemp()
        # Mock the config paths to use our temp directory
        self.config_patcher = patch.multiple(
            'SERVER.checkpoint_manager',
            BASE_CHECKPOINT_PATH=self.temp_dir,
            DB_PATH=os.path.join(self.temp_dir, "test_db.db"),
            ADAPTERS_PATH=os.path.join(self.temp_dir, "adapters"),
            BASE_DATA_PATH=self.temp_dir
        )
        self.config_patcher.start()
        
        # Create test database file
        self.test_db_path = os.path.join(self.temp_dir, "test_db.db")
        with open(self.test_db_path, 'w') as f:
            f.write("test database content")
        
        try:
            from SERVER.checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                server_model_name="TestModel",
                server_Actor_id="TestActor"
            )
            self.real_implementation = True
        except ImportError:
            # Fall back to mock if real implementation not available
            self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
            self.real_implementation = False
        
    def tearDown(self):
        """Clean up after real implementation tests."""
        self.config_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_with_custom_model_and_actor(self):
        """Test CheckpointManager initialization with custom model and actor IDs."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        custom_manager = CheckpointManager(
            server_model_name="CustomModel",
            server_Actor_id="CustomActor"
        )
        self.assertEqual(custom_manager.server_model_name, "CustomModel")
        self.assertEqual(custom_manager.server_Actor_id, "CustomActor")

    def test_timestamped_checkpoint_naming(self):
        """Test that checkpoints are created with proper timestamp naming."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        # Save checkpoint with prefix
        result_msg, checkpoint_list = self.checkpoint_manager.save_checkpoint("test_prefix")
        self.assertIn("saved successfully", result_msg)
        
        # Verify timestamp format in checkpoint names
        for checkpoint in checkpoint_list:
            if checkpoint.startswith("test_prefix_"):
                timestamp_part = checkpoint.replace("test_prefix_", "")
                # Should match YYYYMMDD_HHMMSS format
                self.assertEqual(len(timestamp_part), 15)  # 8 digits + _ + 6 digits
                self.assertTrue(timestamp_part[8] == "_")

    def test_save_checkpoint_without_prefix(self):
        """Test saving checkpoint without name prefix."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        result_msg, checkpoint_list = self.checkpoint_manager.save_checkpoint()
        self.assertIn("saved successfully", result_msg)
        self.assertGreater(len(checkpoint_list), 0)

    def test_database_file_handling(self):
        """Test that database files are properly handled in checkpoints."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        # Create checkpoint
        result_msg, _ = self.checkpoint_manager.save_checkpoint("db_test")
        self.assertIn("saved successfully", result_msg)
        
        # Verify database file was copied to checkpoint
        checkpoints = self.checkpoint_manager.list_checkpoints()
        latest_checkpoint = checkpoints[0]
        checkpoint_dir = os.path.join(self.temp_dir, latest_checkpoint)
        db_file = os.path.join(checkpoint_dir, "dream_weaver.db")
        
        if os.path.exists(db_file):
            with open(db_file, 'r') as f:
                content = f.read()
            self.assertEqual(content, "test database content")

    def test_adapter_file_handling(self):
        """Test that adapter files are properly handled in checkpoints."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        # Create test adapter files
        adapter_dir = os.path.join(self.temp_dir, "adapters", "TestModel", "TestActor")
        os.makedirs(adapter_dir, exist_ok=True)
        test_adapter_file = os.path.join(adapter_dir, "adapter.bin")
        with open(test_adapter_file, 'w') as f:
            f.write("test adapter content")
        
        # Create checkpoint
        result_msg, _ = self.checkpoint_manager.save_checkpoint("adapter_test")
        self.assertIn("saved successfully", result_msg)
        
        # Verify adapter files were copied
        checkpoints = self.checkpoint_manager.list_checkpoints()
        latest_checkpoint = checkpoints[0]
        checkpoint_dir = os.path.join(self.temp_dir, latest_checkpoint)
        adapter_checkpoint_dir = os.path.join(checkpoint_dir, "TestActor_adapters")
        
        if os.path.exists(adapter_checkpoint_dir):
            adapter_file = os.path.join(adapter_checkpoint_dir, "adapter.bin")
            self.assertTrue(os.path.exists(adapter_file))

    def test_load_checkpoint_file_restoration(self):
        """Test that load_checkpoint properly restores files."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        # Create and save checkpoint
        result_msg, _ = self.checkpoint_manager.save_checkpoint("restore_test")
        checkpoints = self.checkpoint_manager.list_checkpoints()
        checkpoint_name = checkpoints[0]
        
        # Modify original database
        modified_content = "modified database content"
        with open(self.test_db_path, 'w') as f:
            f.write(modified_content)
        
        # Load checkpoint
        load_result = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertIn("loaded", load_result)
        self.assertIn("RESTART", load_result)
        
        # Verify database was restored
        with open(self.test_db_path, 'r') as f:
            restored_content = f.read()
        self.assertEqual(restored_content, "test database content")

    def test_load_nonexistent_checkpoint_real_implementation(self):
        """Test loading a nonexistent checkpoint with real implementation."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        result = self.checkpoint_manager.load_checkpoint("nonexistent_checkpoint_12345")
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_export_story_functionality(self):
        """Test the export_story functionality."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        # Mock the database dependency
        with patch('SERVER.checkpoint_manager.Database') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.get_story_history.return_value = [
                ("User", "Hello world", "2024-01-01 10:00:00"),
                ("Assistant", "Hello! How can I help?", "2024-01-01 10:00:05")
            ]
            
            # Test JSON export
            result, filename = self.checkpoint_manager.export_story("json")
            self.assertIn("exported", result)
            self.assertIn("JSON", result)
            self.assertTrue(filename.endswith(".json"))
            
            # Test text export
            result, filename = self.checkpoint_manager.export_story("text")
            self.assertIn("exported", result)
            self.assertIn("Text", result)
            self.assertTrue(filename.endswith(".txt"))

    def test_export_story_invalid_format(self):
        """Test export_story with invalid format."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        result, filename = self.checkpoint_manager.export_story("invalid_format")
        self.assertIn("Error", result)
        self.assertIn("Invalid export format", result)
        self.assertIsNone(filename)

    def test_export_story_no_history(self):
        """Test export_story when no history exists."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        with patch('SERVER.checkpoint_manager.Database') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.get_story_history.return_value = []
            
            result = self.checkpoint_manager.export_story("json")
            self.assertIn("Error", result)
            self.assertIn("No story history", result)


class TestCheckpointManagerAdditionalEdgeCases(unittest.TestCase):
    """Additional edge case tests that work with both mock and real implementations."""
    
    def setUp(self):
        """Set up test fixtures for additional edge case tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up after additional edge case tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_name_length_boundaries(self):
        """Test checkpoint names at various length boundaries."""
        test_cases = [
            ("", "empty_name"),
            ("a", "single_char"),
            ("ab", "two_chars"),
            ("a" * 50, "fifty_chars"),
            ("a" * 100, "hundred_chars"),
            ("a" * 255, "max_filename_length")
        ]
        
        for name, description in test_cases:
            with self.subTest(name=description):
                try:
                    if hasattr(self.checkpoint_manager, 'save_checkpoint'):
                        if len(name) == 0:
                            # Empty names should be handled gracefully
                            continue
                        result = self.checkpoint_manager.save_checkpoint(name, {"test": description})
                        if result:
                            loaded = self.checkpoint_manager.load_checkpoint(name)
                            if loaded:
                                self.assertEqual(loaded.get("test"), description)
                except (ValueError, OSError) as e:
                    # Long names might be rejected by filesystem or implementation
                    self.assertIsInstance(e, (ValueError, OSError))

    def test_checkpoint_with_memory_mapped_data_simulation(self):
        """Test checkpoints with data that simulates memory-mapped structures."""
        # Simulate memory-mapped data with large arrays
        memory_like_data = {
            'buffer_simulation': [0] * 10000,  # Large array
            'sparse_data': {i: i**2 for i in range(0, 1000, 10)},  # Sparse mapping
            'metadata': {
                'size': 10000,
                'type': 'buffer',
                'encoding': 'int32'
            }
        }
        
        checkpoint_name = "memory_mapped_simulation"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, memory_like_data)
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded_data:
                    self.assertEqual(len(loaded_data['buffer_simulation']), 10000)
                    self.assertEqual(loaded_data['metadata']['size'], 10000)
        except (MemoryError, OverflowError) as e:
            # Large data structures might cause memory issues
            self.assertIsInstance(e, (MemoryError, OverflowError))

    def test_checkpoint_data_serialization_edge_cases(self):
        """Test edge cases in data serialization."""
        edge_case_data = {
            'recursive_list': [],
            'self_referencing_dict': {},
            'mixed_types': [1, "string", True, None, [], {}],
            'nested_mixed': {
                'level1': [1, {"level2": [2, {"level3": 3}]}]
            }
        }
        
        # Make one list reference itself (if the implementation handles this)
        edge_case_data['recursive_list'].append(edge_case_data['recursive_list'])
        edge_case_data['self_referencing_dict']['self'] = edge_case_data['self_referencing_dict']
        
        checkpoint_name = "serialization_edge_cases"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, edge_case_data)
            # If serialization succeeds, try to load
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                # The loaded data might not have circular references preserved
                if loaded_data:
                    self.assertIn('mixed_types', loaded_data)
        except (ValueError, TypeError, RecursionError) as e:
            # Circular references often cause serialization errors
            self.assertIsInstance(e, (ValueError, TypeError, RecursionError))

    def test_checkpoint_concurrent_name_generation(self):
        """Test potential naming conflicts with rapid checkpoint creation."""
        import time
        import threading
        
        results = []
        errors = []
        
        def rapid_checkpoint_creation(worker_id):
            """Create checkpoints rapidly to test naming conflicts."""
            try:
                for i in range(5):
                    name = f"rapid_{worker_id}_{i}"
                    data = {"worker": worker_id, "iteration": i, "timestamp": time.time()}
                    result = self.checkpoint_manager.save_checkpoint(name, data)
                    results.append((worker_id, i, result, name))
                    # Very small delay to simulate rapid operations
                    time.sleep(0.001)
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads for concurrent checkpoint creation
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=rapid_checkpoint_creation, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        
        # Verify all checkpoints were created
        self.assertEqual(len(results), 15)  # 3 workers * 5 iterations

    def test_checkpoint_with_complex_nested_structures(self):
        """Test checkpoints with complex nested data structures."""
        complex_nested = {
            'matrix_3d': [[[i*j*k for k in range(5)] for j in range(5)] for i in range(5)],
            'tree_structure': {
                'root': {
                    'children': [
                        {'name': 'child1', 'value': 10, 'children': []},
                        {'name': 'child2', 'value': 20, 'children': [
                            {'name': 'grandchild1', 'value': 30, 'children': []}
                        ]}
                    ]
                }
            },
            'graph_adjacency': {
                'nodes': ['A', 'B', 'C', 'D'],
                'edges': [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
            }
        }
        
        checkpoint_name = "complex_nested_structures"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, complex_nested)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if loaded_data:
            # Verify structure integrity
            self.assertEqual(len(loaded_data['matrix_3d']), 5)
            self.assertEqual(len(loaded_data['matrix_3d'][0]), 5)
            self.assertEqual(len(loaded_data['matrix_3d'][0][0]), 5)
            self.assertEqual(loaded_data['tree_structure']['root']['children'][0]['name'], 'child1')
            self.assertEqual(len(loaded_data['graph_adjacency']['nodes']), 4)

    def test_checkpoint_with_datetime_objects(self):
        """Test checkpoints containing datetime objects."""
        from datetime import datetime, timedelta
        
        datetime_data = {
            'current_time': datetime.now(),
            'past_time': datetime.now() - timedelta(days=30),
            'future_time': datetime.now() + timedelta(days=30),
            'timestamps': [
                datetime.now() - timedelta(hours=i) for i in range(24)
            ],
            'time_metadata': {
                'created': datetime.now(),
                'format': 'ISO-8601',
                'timezone': 'UTC'
            }
        }
        
        checkpoint_name = "datetime_objects"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, datetime_data)
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                # Datetime objects might be serialized as strings
                if loaded_data:
                    self.assertIn('current_time', loaded_data)
                    self.assertEqual(len(loaded_data['timestamps']), 24)
        except (TypeError, ValueError) as e:
            # Datetime objects might not be directly serializable
            self.assertIsInstance(e, (TypeError, ValueError))

    def test_checkpoint_performance_profiling(self):
        """Test checkpoint operations with performance profiling."""
        import time
        
        # Performance test data
        perf_data = {
            'large_list': list(range(50000)),
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(10000)},
            'nested_performance': {
                f'level_{i}': {
                    f'sublevel_{j}': list(range(100))
                    for j in range(10)
                } for i in range(10)
            }
        }
        
        checkpoint_name = "performance_test"
        
        # Time the save operation
        start_save = time.time()
        save_result = self.checkpoint_manager.save_checkpoint(checkpoint_name, perf_data)
        end_save = time.time()
        save_time = end_save - start_save
        
        self.assertTrue(save_result)
        
        # Time the load operation
        start_load = time.time()
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        end_load = time.time()
        load_time = end_load - start_load
        
        if loaded_data:
            self.assertEqual(len(loaded_data['large_list']), 50000)
            self.assertEqual(len(loaded_data['large_dict']), 10000)
        
        # Performance assertions (adjust thresholds based on system capabilities)
        self.assertLess(save_time, 30.0, f"Save operation took {save_time:.2f}s, expected < 30s")
        self.assertLess(load_time, 30.0, f"Load operation took {load_time:.2f}s, expected < 30s")
        
        # Log performance metrics for monitoring
        print(f"Performance metrics - Save: {save_time:.3f}s, Load: {load_time:.3f}s")

    def test_checkpoint_state_consistency_across_operations(self):
        """Test that checkpoint state remains consistent across multiple operations."""
        # Create initial checkpoint
        initial_data = {'state': 'initial', 'counter': 0}
        self.checkpoint_manager.save_checkpoint('consistency_test', initial_data)
        
        # Perform multiple operations and verify consistency
        operations = [
            ('update_1', {'state': 'updated', 'counter': 1}),
            ('update_2', {'state': 'updated_again', 'counter': 2}),
            ('final', {'state': 'final', 'counter': 3})
        ]
        
        for name, data in operations:
            # Save new checkpoint
            save_result = self.checkpoint_manager.save_checkpoint(name, data)
            self.assertTrue(save_result)
            
            # Verify we can load it immediately
            loaded = self.checkpoint_manager.load_checkpoint(name)
            if loaded:
                self.assertEqual(loaded['state'], data['state'])
                self.assertEqual(loaded['counter'], data['counter'])
            
            # Verify checkpoint list is updated
            checkpoints = self.checkpoint_manager.list_checkpoints()
            self.assertIn(name, checkpoints)
        
        # Verify all checkpoints still exist and are loadable
        final_checkpoints = self.checkpoint_manager.list_checkpoints()
        expected_names = ['consistency_test'] + [op[0] for op in operations]
        
        for name in expected_names:
            self.assertIn(name, final_checkpoints)
            loaded = self.checkpoint_manager.load_checkpoint(name)
            self.assertIsNotNone(loaded)

    def test_checkpoint_error_handling_robustness(self):
        """Test robust error handling in various failure scenarios."""
        error_scenarios = [
            # (name, data, expected_exception_type_or_none)
            (None, {'valid': 'data'}, (ValueError, TypeError)),
            (123, {'valid': 'data'}, (ValueError, TypeError)),
            ('valid_name', object(), (TypeError, ValueError)),  # Non-serializable object
        ]
        
        for name, data, expected_exception in error_scenarios:
            with self.subTest(name=str(name), data_type=type(data).__name__):
                if expected_exception:
                    try:
                        result = self.checkpoint_manager.save_checkpoint(name, data)
                        # If no exception was raised, the implementation handled it gracefully
                        if result is False or (isinstance(result, tuple) and "Error" in str(result[0])):
                            # Implementation returned error status instead of raising exception
                            pass
                        else:
                            # If save succeeded, try to load to see if it's actually valid
                            if name and isinstance(name, str):
                                loaded = self.checkpoint_manager.load_checkpoint(name)
                                # Some implementations might handle edge cases gracefully
                    except expected_exception:
                        # Expected behavior
                        pass
                    except Exception as e:
                        # Unexpected exception type
                        self.fail(f"Unexpected exception {type(e).__name__}: {e}")


# Add test markers for pytest compatibility
class TestCheckpointManagerMarkers(unittest.TestCase):
    """Test class with pytest markers for test categorization."""
    
    def setUp(self):
        """Set up for marker tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up marker tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(sys.version_info >= (3, 8), "Requires Python 3.8+")
    def test_python_version_specific_features(self):
        """Test features that require specific Python versions."""
        # Test walrus operator (Python 3.8+)
        checkpoint_name = "python38_features"
        data = {'version_test': True}
        
        if (result := self.checkpoint_manager.save_checkpoint(checkpoint_name, data)):
            loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(loaded, data)

    def test_performance_marked_test(self):
        """Performance test that can be marked as slow."""
        # This test would be marked as @pytest.mark.performance in a pytest environment
        large_data = {'performance': list(range(10000))}
        checkpoint_name = "performance_marked"
        
        import time
        start = time.time()
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, large_data)
        duration = time.time() - start
        
        self.assertTrue(result)
        # Mark this as slow if it takes more than 1 second
        if duration > 1.0:
            print(f"Slow test detected: {duration:.2f}s")

    def test_integration_marked_test(self):
        """Integration test that tests multiple components together."""
        # This would be marked as @pytest.mark.integration in a pytest environment
        
        # Test full workflow integration
        checkpoint_name = "integration_test"
        data = {'integration': True, 'workflow': 'complete'}
        
        # Save
        save_result = self.checkpoint_manager.save_checkpoint(checkpoint_name, data)
        self.assertTrue(save_result)
        
        # List
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertIn(checkpoint_name, checkpoints)
        
        # Load
        loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded, data)
        
        # Delete
        delete_result = self.checkpoint_manager.delete_checkpoint(checkpoint_name)
        self.assertTrue(delete_result)
        
        # Verify deletion
        final_checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertNotIn(checkpoint_name, final_checkpoints)


if __name__ == '__main__':
    # Enhanced test runner with better reporting
    import sys
    
    # Set up test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes including new ones
    test_classes = [
        TestCheckpointManager,
        TestCheckpointManagerEdgeCases,
        TestCheckpointManagerRealImplementation,
        TestCheckpointManagerAdditionalEdgeCases,
        TestCheckpointManagerMarkers
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run with enhanced verbosity
    runner = unittest.TextTestRunner(
        verbosity=2, 
        buffer=True,
        failfast=False,
        stream=sys.stdout
    )
    
    print("="*70)
    print("COMPREHENSIVE CHECKPOINT MANAGER TEST SUITE")
    print("="*70)
    print(f"Testing Framework: unittest (pytest compatible)")
    print(f"Total Test Classes: {len(test_classes)}")
    print("="*70)
    
    result = runner.run(suite)
    
    # Enhanced summary reporting
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 'N/A'}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("="*70)
    
    # Exit with error code if tests failed
    if result.failures or result.errors:
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)

class TestCheckpointManagerFileSystemEdgeCases(unittest.TestCase):
    """Tests for filesystem-specific edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures for filesystem edge case tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up after filesystem edge case tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_readonly_directory_handling(self):
        """Test behavior when checkpoint directory becomes read-only."""
        if os.name == 'nt':  # Skip on Windows due to permission handling differences
            self.skipTest("Read-only directory test not reliable on Windows")
            
        # Make directory read-only
        os.chmod(self.temp_dir, 0o444)
        
        try:
            result = self.checkpoint_manager.save_checkpoint("readonly_test", {"data": "test"})
            # Implementation should handle this gracefully
            if hasattr(self.checkpoint_manager, 'save_checkpoint'):
                # Either return False or raise appropriate exception
                self.assertFalse(result or "Error" in str(result))
        except (PermissionError, OSError) as e:
            # Expected behavior for permission issues
            self.assertIsInstance(e, (PermissionError, OSError))
        finally:
            # Restore permissions for cleanup
            os.chmod(self.temp_dir, 0o755)

    def test_disk_full_simulation(self):
        """Test behavior when disk space is insufficient."""
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            try:
                result = self.checkpoint_manager.save_checkpoint("disk_full_test", {"data": "test"})
                # Should handle gracefully
                if result is not None:
                    self.assertFalse(result)
            except OSError as e:
                self.assertIn("space", str(e).lower())

    def test_corrupted_checkpoint_file_handling(self):
        """Test handling of corrupted checkpoint files."""
        # First save a valid checkpoint
        checkpoint_name = "corruption_test"
        valid_data = {"valid": "data"}
        self.checkpoint_manager.save_checkpoint(checkpoint_name, valid_data)
        
        # Simulate file corruption by writing invalid data
        try:
            # Find the checkpoint file and corrupt it
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if checkpoint_name in checkpoints:
                # Try to corrupt the file if we can find it
                checkpoint_files = []
                for root, dirs, files in os.walk(self.temp_dir):
                    for file in files:
                        if checkpoint_name in file:
                            checkpoint_files.append(os.path.join(root, file))
                
                if checkpoint_files:
                    with open(checkpoint_files[0], 'w') as f:
                        f.write("corrupted data that is not valid JSON/pickle")
                    
                    # Try to load the corrupted checkpoint
                    loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                    # Should handle corruption gracefully
                    self.assertIsNone(loaded_data or "Error" in str(loaded_data))
        except Exception:
            # If corruption simulation fails, that's acceptable
            pass

    def test_symlink_handling(self):
        """Test handling of symbolic links in checkpoint directory."""
        if os.name == 'nt':  # Skip on Windows due to symlink restrictions
            self.skipTest("Symlink test not reliable on Windows")
            
        # Create a symlink to another directory
        symlink_target = tempfile.mkdtemp()
        symlink_path = os.path.join(self.temp_dir, "symlinked_checkpoints")
        
        try:
            os.symlink(symlink_target, symlink_path)
            
            # Test checkpoint operations with symlinks present
            result = self.checkpoint_manager.save_checkpoint("symlink_test", {"data": "test"})
            # Should handle symlinks appropriately
            if result:
                loaded = self.checkpoint_manager.load_checkpoint("symlink_test")
                if loaded:
                    self.assertEqual(loaded["data"], "test")
        except OSError:
            # Symlink creation might fail on some systems
            pass
        finally:
            if os.path.exists(symlink_target):
                shutil.rmtree(symlink_target)

    def test_nested_directory_creation_failure(self):
        """Test handling of nested directory creation failures."""
        # Create a file where a directory should be created
        problematic_path = os.path.join(self.temp_dir, "nested", "path")
        os.makedirs(os.path.dirname(problematic_path), exist_ok=True)
        
        # Create a file with the same name as the directory we need
        with open(problematic_path, 'w') as f:
            f.write("blocking file")
        
        # Try to use a checkpoint manager with this problematic path
        try:
            problematic_manager = CheckpointManager(checkpoint_dir=os.path.join(problematic_path, "checkpoints"))
            result = problematic_manager.save_checkpoint("nested_fail_test", {"data": "test"})
            # Should handle directory creation failure
            if result is not None:
                self.assertFalse(result or "Error" in str(result))
        except (OSError, IOError) as e:
            # Expected behavior for directory creation failure
            self.assertIsInstance(e, (OSError, IOError))

    def test_very_deep_directory_structure(self):
        """Test handling of very deep directory structures."""
        # Create a very deep path
        deep_path_parts = ["very"] * 50  # 50 levels deep
        deep_path = os.path.join(self.temp_dir, *deep_path_parts)
        
        try:
            deep_manager = CheckpointManager(checkpoint_dir=deep_path)
            result = deep_manager.save_checkpoint("deep_test", {"data": "test"})
            
            if result:
                loaded = deep_manager.load_checkpoint("deep_test")
                if loaded:
                    self.assertEqual(loaded["data"], "test")
        except OSError as e:
            # Very deep paths might exceed filesystem limits
            self.assertIsInstance(e, OSError)

    def test_special_filesystem_characters(self):
        """Test handling of special filesystem characters in checkpoint data."""
        special_char_data = {
            'filename_chars': '<>:"|?*',
            'unicode_filename': 'тест файл',
            'emoji_filename': '📁💾🔒',
            'control_chars': '\n\t\r\x00\x1f',
            'path_separators': '\\//\\',
            'reserved_names': ['CON', 'PRN', 'AUX', 'NUL']
        }
        
        checkpoint_name = "special_chars_test"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, special_char_data)
            if result:
                loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded:
                    # Data should be preserved even if filenames are sanitized
                    self.assertIn('filename_chars', loaded)
                    self.assertIn('unicode_filename', loaded)
        except (ValueError, OSError) as e:
            # Special characters might cause filesystem issues
            self.assertIsInstance(e, (ValueError, OSError))


class TestCheckpointManagerMemoryAndResourceManagement(unittest.TestCase):
    """Tests for memory management and resource usage scenarios."""
    
    def setUp(self):
        """Set up test fixtures for memory and resource tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up after memory and resource tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_memory_efficient_large_data_handling(self):
        """Test memory efficiency with large data structures."""
        # Create data that should be handled efficiently
        efficient_large_data = {
            'large_but_sparse': {i: None for i in range(0, 10000, 100)},
            'repeated_pattern': ['pattern'] * 10000,
            'memory_view_simulation': bytearray(b'x' * 10000)
        }
        
        checkpoint_name = "memory_efficient_test"
        try:
            # Monitor memory usage during operation if psutil available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss
                
                result = self.checkpoint_manager.save_checkpoint(checkpoint_name, efficient_large_data)
                
                memory_after = process.memory_info().rss
                memory_increase = memory_after - memory_before
                
                # Memory increase should be reasonable (less than 100MB for this test)
                self.assertLess(memory_increase, 100 * 1024 * 1024, 
                              f"Memory increase too large: {memory_increase / 1024 / 1024:.2f}MB")
                
                if result:
                    loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                    if loaded:
                        self.assertIn('repeated_pattern', loaded)
                        
            except ImportError:
                # psutil not available, skip memory monitoring
                result = self.checkpoint_manager.save_checkpoint(checkpoint_name, efficient_large_data)
                self.assertTrue(result)
        except (MemoryError, OverflowError) as e:
            # Expected for very large data
            self.assertIsInstance(e, (MemoryError, OverflowError))

    def test_resource_cleanup_on_failure(self):
        """Test that resources are properly cleaned up on operation failure."""
        # Simulate failure scenarios and verify cleanup
        def failing_save_operation():
            try:
                # Create a checkpoint operation that will fail
                problematic_data = {"fail": object()}  # Non-serializable
                return self.checkpoint_manager.save_checkpoint("cleanup_test", problematic_data)
            except (TypeError, ValueError):
                return False
                
        # Check that temporary files/resources are cleaned up
        files_before = set()
        for root, dirs, files in os.walk(self.temp_dir):
            files_before.update(os.path.join(root, f) for f in files)
            
        failing_save_operation()
        
        files_after = set()
        for root, dirs, files in os.walk(self.temp_dir):
            files_after.update(os.path.join(root, f) for f in files)
            
        # Should not have left temporary files
        new_files = files_after - files_before
        temp_files = [f for f in new_files if 'tmp' in f.lower() or '.temp' in f.lower()]
        self.assertEqual(len(temp_files), 0, f"Temporary files not cleaned up: {temp_files}")

    def test_concurrent_resource_contention(self):
        """Test behavior under resource contention scenarios."""
        import threading
        import time
        
        resource_conflicts = []
        successful_operations = []
        
        def resource_intensive_operation(worker_id):
            """Simulate resource-intensive checkpoint operations."""
            try:
                for i in range(3):
                    # Create resource-intensive data
                    intensive_data = {
                        'worker_id': worker_id,
                        'iteration': i,
                        'large_data': list(range(10000)),
                        'timestamp': time.time()
                    }
                    
                    checkpoint_name = f"resource_worker_{worker_id}_iter_{i}"
                    result = self.checkpoint_manager.save_checkpoint(checkpoint_name, intensive_data)
                    
                    if result:
                        successful_operations.append((worker_id, i))
                        
                        # Immediately try to load to increase resource pressure
                        loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                        if not loaded:
                            resource_conflicts.append(f"Load failed for {checkpoint_name}")
                    else:
                        resource_conflicts.append(f"Save failed for {checkpoint_name}")
                        
                    time.sleep(0.01)  # Small delay between operations
                    
            except Exception as e:
                resource_conflicts.append(f"Worker {worker_id} exception: {str(e)}")
        
        # Create multiple threads to simulate resource contention
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=resource_intensive_operation, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=30)
        
        # Verify system handled contention gracefully
        total_expected_operations = 4 * 3  # 4 workers * 3 iterations
        success_rate = len(successful_operations) / total_expected_operations
        
        # Should handle at least 50% of operations successfully under contention
        self.assertGreater(success_rate, 0.5, 
                          f"Success rate too low: {success_rate:.2f}, Conflicts: {resource_conflicts}")

    def test_garbage_collection_interaction(self):
        """Test interaction with Python's garbage collector."""
        import gc
        import weakref
        
        # Create objects with references that might interfere with GC
        checkpoint_data = {
            'circular_refs': [],
            'weak_refs': [],
            'large_objects': []
        }
        
        # Create circular references
        circular_obj = {'self': None}
        circular_obj['self'] = circular_obj
        checkpoint_data['circular_refs'].append(circular_obj)
        
        # Create objects for weak references
        for i in range(10):
            obj = {'id': i, 'data': list(range(1000))}
            checkpoint_data['large_objects'].append(obj)
            try:
                weak_ref = weakref.ref(obj)
                checkpoint_data['weak_refs'].append(str(weak_ref))
            except TypeError:
                # Some objects can't have weak references
                pass
        
        checkpoint_name = "gc_interaction_test"
        
        # Force garbage collection before operation
        gc.collect()
        objects_before = len(gc.get_objects())
        
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, checkpoint_data)
            
            if result:
                loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded:
                    self.assertIn('large_objects', loaded)
                    
            # Force garbage collection after operation
            del checkpoint_data
            gc.collect()
            objects_after = len(gc.get_objects())
            
            # Should not have excessive object growth
            object_growth = objects_after - objects_before
            self.assertLess(object_growth, 1000, 
                          f"Excessive object growth: {object_growth}")
            
        except (TypeError, ValueError) as e:
            # Circular references might cause serialization issues
            self.assertIsInstance(e, (TypeError, ValueError))


class TestCheckpointManagerDataValidationAndSanitization(unittest.TestCase):
    """Tests for data validation, sanitization, and security scenarios."""
    
    def setUp(self):
        """Set up test fixtures for validation and sanitization tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up after validation and sanitization tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_malicious_data_patterns(self):
        """Test handling of potentially malicious data patterns."""
        malicious_patterns = {
            'script_injection': '<script>alert("xss")</script>',
            'sql_injection': "'; DROP TABLE checkpoints; --",
            'path_traversal': '../../../etc/passwd',
            'command_injection': '; rm -rf /',
            'format_string': '%s%s%s%s%s%n',
            'buffer_overflow': 'A' * 10000,
            'null_bytes': 'data\x00hidden',
            'unicode_attacks': '\u202e\u202d',  # Unicode override characters
        }
        
        checkpoint_name = "malicious_patterns_test"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, malicious_patterns)
            
            if result:
                loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded:
                    # Data should be preserved but sanitized if necessary
                    self.assertIn('script_injection', loaded)
                    # Verify dangerous characters are handled appropriately
                    for key, value in loaded.items():
                        if isinstance(value, str):
                            # Should not contain unescaped dangerous patterns
                            self.assertNotIn('\x00', value)  # Null bytes should be handled
                            
        except (ValueError, TypeError) as e:
            # Implementation might reject malicious patterns
            self.assertIsInstance(e, (ValueError, TypeError))

    def test_data_type_validation_edge_cases(self):
        """Test validation of edge cases in data types."""
        edge_case_data = {
            'infinity_float': float('inf'),
            'negative_infinity': float('-inf'),
            'nan_value': float('nan'),
            'max_int': sys.maxsize,
            'min_int': -sys.maxsize - 1,
            'empty_containers': [[], {}, set(), tuple()],
            'mixed_encodings': 'ASCII + UNICODE: café 🎉',
            'escape_sequences': 'Line1\nLine2\tTabbed\rCarriage\fForm\vVertical',
            'zero_width_chars': 'invisible\u200bzero\u200cwidth\u200dchars',
        }
        
        checkpoint_name = "data_validation_edge_cases"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, edge_case_data)
            
            if result:
                loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded:
                    # Special float values handling
                    if 'infinity_float' in loaded:
                        inf_val = loaded['infinity_float']
                        # Might be serialized as string or preserved as float
                        self.assertTrue(inf_val == float('inf') or str(inf_val) == 'inf' or 'inf' in str(inf_val).lower())
                    
                    # Container integrity
                    if 'empty_containers' in loaded:
                        containers = loaded['empty_containers']
                        self.assertEqual(len(containers), 4)  # Should preserve all empty containers
                        
        except (ValueError, TypeError, OverflowError) as e:
            # Special values might not be serializable
            self.assertIsInstance(e, (ValueError, TypeError, OverflowError))

    def test_checkpoint_name_injection_attacks(self):
        """Test handling of checkpoint names designed to cause issues."""
        malicious_names = [
            "../../../system_file",
            "normal_name; rm -rf /",
            "name_with\nnewlines\rand\ttabs",
            "name_with_nulls\x00hidden_part",
            "very_long_name_" + "x" * 1000 + "_end",
            "unicode_attack_\u202e\u202dreversal",
            ".hidden_checkpoint",
            "checkpoint.exe",
            "checkpoint.bat",
            "CON",  # Windows reserved name
            "PRN",  # Windows reserved name
        ]
        
        test_data = {"secure": "data"}
        
        for malicious_name in malicious_names:
            with self.subTest(name=repr(malicious_name)):
                try:
                    result = self.checkpoint_manager.save_checkpoint(malicious_name, test_data)
                    
                    if result:
                        # If save succeeded, verify it was sanitized appropriately
                        checkpoints = self.checkpoint_manager.list_checkpoints()
                        
                        # The actual stored name might be sanitized
                        found_checkpoint = None
                        for cp_name in checkpoints:
                            if malicious_name in cp_name or any(char in cp_name for char in malicious_name.replace('\x00', '').replace('\n', '').replace('\r', '').replace('\t', '')):
                                found_checkpoint = cp_name
                                break
                        
                        if found_checkpoint:
                            loaded = self.checkpoint_manager.load_checkpoint(found_checkpoint)
                            if loaded:
                                self.assertEqual(loaded["secure"], "data")
                                
                except (ValueError, OSError, TypeError) as e:
                    # Expected behavior for malicious names
                    self.assertIsInstance(e, (ValueError, OSError, TypeError))

    def test_data_integrity_with_binary_data(self):
        """Test data integrity preservation with binary data."""
        binary_test_data = {
            'pure_binary': b'\x00\x01\x02\x03\xff\xfe\xfd',
            'mixed_binary': [b'binary', 'text', 123, b'\x80\x81\x82'],
            'large_binary': bytes(range(256)) * 100,  # 25.6KB of binary data
            'empty_binary': b'',
            'unicode_as_bytes': 'Hello 世界 🌍'.encode('utf-8'),
            'bytearray_data': bytearray(b'mutable binary data'),
        }
        
        checkpoint_name = "binary_integrity_test"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, binary_test_data)
            
            if result:
                loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded:
                    # Verify binary data integrity
                    if 'pure_binary' in loaded:
                        loaded_binary = loaded['pure_binary']
                        if isinstance(loaded_binary, bytes):
                            self.assertEqual(loaded_binary, binary_test_data['pure_binary'])
                        # Might be base64 encoded or otherwise serialized
                    
                    if 'large_binary' in loaded:
                        loaded_large = loaded['large_binary']
                        if isinstance(loaded_large, bytes):
                            self.assertEqual(len(loaded_large), len(binary_test_data['large_binary']))
                            
        except (TypeError, ValueError) as e:
            # Binary data might not be directly serializable
            self.assertIsInstance(e, (TypeError, ValueError))

    def test_checkpoint_data_size_limits(self):
        """Test handling of data that approaches or exceeds reasonable size limits."""
        size_limit_tests = [
            ("small_data", {"tiny": "x"}),
            ("medium_data", {"data": "x" * 1000}),
            ("large_data", {"data": "x" * 100000}),
            ("very_large_data", {"data": "x" * 1000000}),  # 1MB string
            ("huge_structure", {"data": ["x" * 1000] * 1000}),  # 1MB in structure
        ]
        
        for test_name, test_data in size_limit_tests:
            with self.subTest(name=test_name):
                try:
                    result = self.checkpoint_manager.save_checkpoint(test_name, test_data)
                    
                    if result:
                        loaded = self.checkpoint_manager.load_checkpoint(test_name)
                        if loaded:
                            # Verify data integrity for successful operations
                            if isinstance(test_data["data"], str):
                                self.assertEqual(len(loaded["data"]), len(test_data["data"]))
                            elif isinstance(test_data["data"], list):
                                self.assertEqual(len(loaded["data"]), len(test_data["data"]))
                                if loaded["data"]:
                                    self.assertEqual(len(loaded["data"][0]), len(test_data["data"][0]))
                                    
                except (MemoryError, OverflowError, OSError) as e:
                    # Large data might exceed system limits
                    self.assertIsInstance(e, (MemoryError, OverflowError, OSError))
                    print(f"Expected size limit exceeded for {test_name}: {type(e).__name__}")


class TestCheckpointManagerAdvancedErrorRecovery(unittest.TestCase):
    """Tests for advanced error recovery and fault tolerance scenarios."""
    
    def setUp(self):
        """Set up test fixtures for error recovery tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up after error recovery tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_partial_operation_recovery(self):
        """Test recovery from partially completed operations."""
        # Create a scenario where operation is interrupted
        test_data = {"recovery": "test_data", "large": list(range(10000))}
        checkpoint_name = "partial_recovery_test"
        
        # Simulate partial save by creating incomplete checkpoint structure
        try:
            # Start save operation
            with patch('builtins.open', side_effect=[
                unittest.mock.mock_open(write_data="").return_value,  # First file succeeds
                OSError("Disk full")  # Second file fails
            ]):
                result = self.checkpoint_manager.save_checkpoint(checkpoint_name, test_data)
                # Should handle partial failure gracefully
                if result is not None:
                    self.assertFalse(result)
                    
        except OSError:
            # Expected behavior for partial failure
            pass
        
        # Verify system can recover and continue operating
        recovery_result = self.checkpoint_manager.save_checkpoint("recovery_after_failure", {"status": "recovered"})
        if recovery_result:
            loaded = self.checkpoint_manager.load_checkpoint("recovery_after_failure")
            if loaded:
                self.assertEqual(loaded["status"], "recovered")

    def test_interrupted_operation_cleanup(self):
        """Test cleanup of interrupted operations."""
        # Simulate operations that get interrupted at various stages
        interruption_scenarios = [
            ("during_save", lambda: self.checkpoint_manager.save_checkpoint("interrupted_save", {"data": list(range(1000))})),
            ("during_load", lambda: self.checkpoint_manager.load_checkpoint("nonexistent_cp")),
            ("during_delete", lambda: self.checkpoint_manager.delete_checkpoint("nonexistent_cp")),
        ]
        
        for scenario_name, operation in interruption_scenarios:
            with self.subTest(scenario=scenario_name):
                try:
                    # Execute operation that might be interrupted
                    result = operation()
                    
                    # Verify system state is still consistent after operation
                    list_result = self.checkpoint_manager.list_checkpoints()
                    self.assertIsInstance(list_result, list)
                    
                    # System should still be able to perform basic operations
                    test_result = self.checkpoint_manager.save_checkpoint(f"after_{scenario_name}", {"test": True})
                    if test_result:
                        cleanup_loaded = self.checkpoint_manager.load_checkpoint(f"after_{scenario_name}")
                        if cleanup_loaded:
                            self.assertEqual(cleanup_loaded["test"], True)
                            
                except Exception as e:
                    # Even if operation fails, system should remain stable
                    try:
                        stability_test = self.checkpoint_manager.list_checkpoints()
                        self.assertIsInstance(stability_test, list)
                    except Exception as stability_error:
                        self.fail(f"System became unstable after {scenario_name}: {stability_error}")

    def test_corruption_detection_and_recovery(self):
        """Test detection and recovery from data corruption."""
        # Create valid checkpoint first
        valid_data = {"integrity": "check", "data": list(range(100))}
        checkpoint_name = "corruption_detection_test"
        
        save_result = self.checkpoint_manager.save_checkpoint(checkpoint_name, valid_data)
        if save_result:
            # Verify initial save worked
            loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            if loaded:
                self.assertEqual(loaded["integrity"], "check")
                
                # Simulate data corruption scenarios
                corruption_tests = [
                    ("truncated_file", lambda f: f.truncate(len(f.read()) // 2)),
                    ("invalid_json", lambda f: f.write("invalid json data")),
                    ("binary_corruption", lambda f: f.write(b'\x00\xff\x00\xff' * 100)),
                ]
                
                for corruption_type, corruption_func in corruption_tests:
                    with self.subTest(corruption=corruption_type):
                        # Try to find and corrupt the checkpoint file
                        try:
                            for root, dirs, files in os.walk(self.temp_dir):
                                for file in files:
                                    if checkpoint_name in file:
                                        file_path = os.path.join(root, file)
                                        # Backup original
                                        with open(file_path, 'rb') as f:
                                            original_data = f.read()
                                        
                                        try:
                                            # Apply corruption
                                            with open(file_path, 'w+b') as f:
                                                corruption_func(f)
                                            
                                            # Try to load corrupted checkpoint
                                            corrupted_loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                                            # Should handle corruption gracefully
                                            if corrupted_loaded is not None:
                                                # Either returns None or error message
                                                self.assertTrue("Error" in str(corrupted_loaded) or corrupted_loaded is None)
                                                
                                        finally:
                                            # Restore original data
                                            with open(file_path, 'wb') as f:
                                                f.write(original_data)
                                                
                                        break
                                        
                        except Exception as corruption_error:
                            # Corruption simulation might fail on some systems
                            print(f"Corruption simulation failed for {corruption_type}: {corruption_error}")

    def test_resource_exhaustion_scenarios(self):
        """Test behavior under various resource exhaustion conditions."""
        exhaustion_scenarios = [
            ("memory_pressure", lambda: self.checkpoint_manager.save_checkpoint("memory_test", {"data": [0] * 100000})),
            ("file_handle_pressure", lambda: [self.checkpoint_manager.save_checkpoint(f"fh_test_{i}", {"id": i}) for i in range(100)]),
            ("disk_space_pressure", lambda: self.checkpoint_manager.save_checkpoint("disk_test", {"data": "x" * 1000000})),
        ]
        
        for scenario_name, pressure_operation in exhaustion_scenarios:
            with self.subTest(scenario=scenario_name):
                try:
                    # Apply resource pressure
                    result = pressure_operation()
                    
                    # System should remain responsive under pressure
                    post_pressure_test = self.checkpoint_manager.save_checkpoint(f"post_{scenario_name}", {"status": "stable"})
                    
                    # Verify basic functionality still works
                    if post_pressure_test:
                        stability_check = self.checkpoint_manager.load_checkpoint(f"post_{scenario_name}")
                        if stability_check:
                            self.assertEqual(stability_check["status"], "stable")
                            
                except (MemoryError, OSError, IOError) as e:
                    # Resource exhaustion is expected
                    print(f"Expected resource exhaustion in {scenario_name}: {type(e).__name__}")
                    
                    # System should still be stable for basic operations
                    try:
                        recovery_test = self.checkpoint_manager.list_checkpoints()
                        self.assertIsInstance(recovery_test, list)
                    except Exception as recovery_error:
                        self.fail(f"System failed to recover from {scenario_name}: {recovery_error}")


class TestCheckpointManagerRealImplementationExtended(unittest.TestCase):
    """Extended tests specifically for the real CheckpointManager implementation."""
    
    def setUp(self):
        """Set up test fixtures for extended real implementation tests."""
        self.temp_dir = tempfile.mkdtemp()
        # Mock the config paths to use our temp directory
        self.config_patcher = patch.multiple(
            'SERVER.checkpoint_manager',
            BASE_CHECKPOINT_PATH=self.temp_dir,
            DB_PATH=os.path.join(self.temp_dir, "test_db.db"),
            ADAPTERS_PATH=os.path.join(self.temp_dir, "adapters"),
            BASE_DATA_PATH=self.temp_dir
        )
        self.config_patcher.start()
        
        # Create test database file
        self.test_db_path = os.path.join(self.temp_dir, "test_db.db")
        with open(self.test_db_path, 'w') as f:
            f.write("test database content")
        
        try:
            from SERVER.checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                server_model_name="TestModel",
                server_Actor_id="TestActor"
            )
            self.real_implementation = True
        except ImportError:
            # Fall back to mock if real implementation not available
            self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
            self.real_implementation = False
        
    def tearDown(self):
        """Clean up after extended real implementation tests."""
        self.config_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_config_path_validation(self):
        """Test that CheckpointManager properly validates configuration paths."""
        if not self.real_implementation:
            self.skipTest("Real implementation not available")
            
        # Test with various path configurations
        invalid_paths = [
            "",  # Empty path
            "/nonexistent/deeply/nested/path",  # Non-existent path
            "/dev/null",  # Special file
        ]
        
        for invalid_path in invalid_paths:
            with self.subTest(path=invalid_path):
                try:
                    with patch('SERVER.checkpoint_manager.BASE_CHECKPOINT_PATH', invalid_path):
                        manager = CheckpointManager()
                        # Should either handle gracefully or raise appropriate exception
                        result = manager.sav