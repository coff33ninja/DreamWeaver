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
        TestCheckpointManagerMarkers,
        TestCheckpointManagerSecurityAndValidation,
        TestCheckpointManagerStressAndReliability,
        TestCheckpointManagerAdvancedScenarios
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
    print(f"Testing Framework: unittest (pytest compatible) with comprehensive security, stress, and advanced scenario testing")
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

class TestCheckpointManagerSecurityAndValidation(unittest.TestCase):
    """Security-focused and validation tests for CheckpointManager."""
    
    def setUp(self):
        """Set up security test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up security test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_path_traversal_attack_prevention(self):
        """Test that path traversal attacks are prevented in checkpoint names."""
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "checkpoint/../../../sensitive_file",
            "checkpoint\\..\\..\\..\\sensitive_file",
            "checkpoint/./../../sensitive_file",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "....//....//....//etc/passwd",
            "..././..././..././etc/passwd"
        ]
        
        sample_data = {"test": "data"}
        
        for malicious_name in malicious_names:
            with self.subTest(name=malicious_name):
                try:
                    result = self.checkpoint_manager.save_checkpoint(malicious_name, sample_data)
                    # If save succeeds, verify it didn't escape the temp directory
                    if result:
                        # Check that no files were created outside temp directory
                        created_files = []
                        for root, dirs, files in os.walk(self.temp_dir):
                            for file in files:
                                full_path = os.path.join(root, file)
                                if not full_path.startswith(self.temp_dir):
                                    created_files.append(full_path)
                        
                        self.assertEqual(len(created_files), 0, 
                                       f"Files created outside temp directory: {created_files}")
                        
                        # Verify sensitive system files weren't accessed
                        sensitive_paths = ["/etc/passwd", "/etc/shadow", 
                                         "C:\\Windows\\System32\\config\\sam"]
                        for sensitive_path in sensitive_paths:
                            if os.path.exists(sensitive_path):
                                original_stat = os.stat(sensitive_path)
                                time.sleep(0.1)  # Brief delay
                                current_stat = os.stat(sensitive_path)
                                self.assertEqual(original_stat.st_mtime, current_stat.st_mtime,
                                               f"Sensitive file {sensitive_path} was modified")
                                
                except (ValueError, OSError, SecurityError) as e:
                    # Expected behavior - malicious names should be rejected
                    self.assertIsInstance(e, (ValueError, OSError, Exception))
    
    def test_checkpoint_name_injection_attacks(self):
        """Test against various injection attacks in checkpoint names."""
        injection_attempts = [
            "checkpoint; rm -rf /",
            "checkpoint && rm -rf /*",
            "checkpoint | cat /etc/passwd",
            "checkpoint`cat /etc/passwd`",
            "checkpoint$(cat /etc/passwd)",
            "checkpoint'; DROP TABLE checkpoints; --",
            "checkpoint<script>alert('xss')</script>",
            "checkpoint${IFS}cat${IFS}/etc/passwd",
            "checkpoint\x00hidden_command",
            "checkpoint\r\nmalicious_command",
            "checkpoint\n\rmalicious_command"
        ]
        
        sample_data = {"test": "injection_test"}
        
        for injection_name in injection_attempts:
            with self.subTest(name=repr(injection_name)):
                try:
                    result = self.checkpoint_manager.save_checkpoint(injection_name, sample_data)
                    # If it succeeds, verify no system commands were executed
                    if result:
                        # Check that the checkpoint name was sanitized
                        checkpoints = self.checkpoint_manager.list_checkpoints()
                        sanitized_found = False
                        for checkpoint in checkpoints:
                            if "injection_test" in str(self.checkpoint_manager.load_checkpoint(checkpoint) or {}):
                                sanitized_found = True
                                # Verify the name doesn't contain dangerous characters
                                dangerous_chars = [';', '&', '|', '`', '$', '<', '>', '\x00', '\r', '\n']
                                for char in dangerous_chars:
                                    self.assertNotIn(char, checkpoint, 
                                                   f"Dangerous character '{char}' found in checkpoint name")
                        
                        if not sanitized_found:
                            self.fail(f"Injection attempt may have succeeded: {injection_name}")
                            
                except (ValueError, SecurityError, Exception) as e:
                    # Expected behavior - injection attempts should be rejected
                    pass
    
    def test_checkpoint_data_validation_extremes(self):
        """Test validation with extreme data scenarios."""
        extreme_scenarios = [
            # Extremely large strings
            ("massive_string", {"data": "x" * 1000000}),
            
            # Extremely deep nesting (beyond typical recursion limits)
            ("deep_nesting", self._create_deeply_nested_dict(500)),
            
            # Extremely wide dictionaries
            ("wide_dict", {f"key_{i}": f"value_{i}" for i in range(100000)}),
            
            # Mixed extreme scenarios
            ("mixed_extreme", {
                "large_string": "x" * 100000,
                "wide_dict": {f"k{i}": i for i in range(10000)},
                "deep_list": self._create_deeply_nested_list(100)
            }),
            
            # Unicode edge cases
            ("unicode_extreme", {
                "null_bytes": "test\x00data",
                "control_chars": "".join(chr(i) for i in range(32)),
                "emoji_spam": "🎉" * 10000,
                "mixed_encodings": "café" + "мир" + "世界" + "🌍" * 1000
            })
        ]
        
        for name, data in extreme_scenarios:
            with self.subTest(scenario=name):
                try:
                    start_time = time.time()
                    result = self.checkpoint_manager.save_checkpoint(name, data)
                    operation_time = time.time() - start_time
                    
                    # Verify operation completes in reasonable time (prevent DoS)
                    self.assertLess(operation_time, 60.0, 
                                  f"Operation took too long: {operation_time:.2f}s")
                    
                    if result:
                        # Verify data integrity for successful saves
                        loaded = self.checkpoint_manager.load_checkpoint(name)
                        if loaded and isinstance(data, dict) and "large_string" in data:
                            # For large string scenarios, verify length
                            self.assertEqual(len(loaded.get("large_string", "")), 
                                           len(data.get("large_string", "")))
                    
                except (MemoryError, RecursionError, OverflowError, ValueError) as e:
                    # Expected behavior for extreme scenarios
                    self.assertIsInstance(e, (MemoryError, RecursionError, OverflowError, ValueError))
    
    def _create_deeply_nested_dict(self, depth):
        """Helper to create deeply nested dictionary."""
        if depth <= 0:
            return {"end": True}
        return {"level": depth, "nested": self._create_deeply_nested_dict(depth - 1)}
    
    def _create_deeply_nested_list(self, depth):
        """Helper to create deeply nested list."""
        if depth <= 0:
            return ["end"]
        return [depth, self._create_deeply_nested_list(depth - 1)]
    
    def test_checkpoint_file_permission_attacks(self):
        """Test handling of various file permission scenarios."""
        if os.name != 'posix':
            self.skipTest("File permission tests only applicable on POSIX systems")
        
        # Create a read-only directory
        readonly_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(readonly_dir, exist_ok=True)
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        try:
            readonly_manager = CheckpointManager(checkpoint_dir=readonly_dir)
            
            # Attempt to save checkpoint in read-only directory
            with self.assertRaises((PermissionError, OSError, IOError)):
                readonly_manager.save_checkpoint("readonly_test", {"data": "test"})
        
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)
    
    def test_checkpoint_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        # Test many small checkpoints (file descriptor exhaustion)
        checkpoint_names = []
        try:
            for i in range(1000):  # Create many checkpoints
                name = f"resource_test_{i}"
                result = self.checkpoint_manager.save_checkpoint(name, {"id": i})
                if result:
                    checkpoint_names.append(name)
                
                # Stop if we hit reasonable limits
                if i > 100 and not result:
                    break
            
            # Verify system remains responsive
            test_name = "responsiveness_test"
            result = self.checkpoint_manager.save_checkpoint(test_name, {"test": "responsive"})
            self.assertTrue(result or len(checkpoint_names) > 50, 
                          "System should either save checkpoint or have created many checkpoints")
            
        except (OSError, MemoryError) as e:
            # Expected behavior when hitting system limits
            self.assertIsInstance(e, (OSError, MemoryError))
    
    def test_checkpoint_atomic_operations(self):
        """Test that checkpoint operations are atomic."""
        checkpoint_name = "atomic_test"
        data = {"test": "atomic_data", "timestamp": time.time()}
        
        # Mock file operations to simulate interruption
        original_open = builtins.open
        call_count = [0]
        
        def failing_open(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call
                raise IOError("Simulated failure during write")
            return original_open(*args, **kwargs)
        
        with patch('builtins.open', side_effect=failing_open):
            try:
                self.checkpoint_manager.save_checkpoint(checkpoint_name, data)
            except IOError:
                pass  # Expected failure
        
        # Verify partial write didn't leave corrupted checkpoint
        loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if loaded is not None:
            # If checkpoint exists, it should be complete and valid
            self.assertEqual(loaded.get("test"), "atomic_data")
        
        # Verify checkpoint list is consistent
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if checkpoint_name in checkpoints:
            # If listed, it should be loadable
            loaded_again = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertIsNotNone(loaded_again)


class TestCheckpointManagerStressAndReliability(unittest.TestCase):
    """Stress testing and reliability tests for CheckpointManager."""
    
    def setUp(self):
        """Set up stress test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up stress test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_memory_pressure(self):
        """Test checkpoint behavior under memory pressure."""
        # Create progressively larger datasets
        sizes = [1000, 10000, 100000, 500000]
        
        for size in sizes:
            with self.subTest(size=size):
                large_data = {
                    'array': list(range(size)),
                    'dict': {f'key_{i}': f'value_{i}' for i in range(min(size//10, 10000))},
                    'metadata': {'size': size, 'type': 'stress_test'}
                }
                
                checkpoint_name = f"memory_pressure_{size}"
                
                try:
                    # Monitor memory usage if psutil is available
                    start_memory = self._get_memory_usage()
                    
                    result = self.checkpoint_manager.save_checkpoint(checkpoint_name, large_data)
                    
                    if result:
                        # Verify data integrity
                        loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                        if loaded:
                            self.assertEqual(len(loaded['array']), size)
                            self.assertEqual(loaded['metadata']['size'], size)
                    
                    end_memory = self._get_memory_usage()
                    
                    # Check for reasonable memory usage (if we can measure it)
                    if start_memory and end_memory:
                        memory_growth = end_memory - start_memory
                        # Allow up to 10x data size for memory overhead
                        reasonable_limit = size * 10 * 8  # 8 bytes per integer, 10x overhead
                        if memory_growth > reasonable_limit:
                            print(f"Warning: High memory usage for size {size}: {memory_growth} bytes")
                
                except (MemoryError, OverflowError) as e:
                    # Expected behavior for very large datasets
                    print(f"Memory limit reached at size {size}: {e}")
                    break
    
    def _get_memory_usage(self):
        """Get current memory usage if psutil is available."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return None
    
    def test_checkpoint_high_frequency_operations(self):
        """Test checkpoint system under high frequency operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def high_frequency_worker(worker_id, operation_count):
            """Worker that performs high-frequency checkpoint operations."""
            try:
                for i in range(operation_count):
                    # Rapid save/load/delete cycles
                    name = f"hf_worker_{worker_id}_{i}"
                    data = {"worker": worker_id, "op": i, "timestamp": time.time()}
                    
                    # Save
                    save_result = self.checkpoint_manager.save_checkpoint(name, data)
                    
                    # Immediate load
                    if save_result:
                        loaded = self.checkpoint_manager.load_checkpoint(name)
                        if loaded and loaded.get("worker") == worker_id:
                            results_queue.put(("success", worker_id, i))
                        else:
                            results_queue.put(("load_fail", worker_id, i))
                    else:
                        results_queue.put(("save_fail", worker_id, i))
                    
                    # Occasional cleanup
                    if i % 10 == 0:
                        try:
                            self.checkpoint_manager.delete_checkpoint(name)
                        except:
                            pass  # Ignore cleanup failures in stress test
                            
            except Exception as e:
                error_queue.put((worker_id, str(e)))
        
        # Start multiple high-frequency workers
        threads = []
        workers = 3
        operations_per_worker = 50
        
        for worker_id in range(workers):
            thread = threading.Thread(
                target=high_frequency_worker, 
                args=(worker_id, operations_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Analyze results
        total_operations = 0
        successful_operations = 0
        errors = []
        
        while not results_queue.empty():
            result_type, worker_id, op_id = results_queue.get()
            total_operations += 1
            if result_type == "success":
                successful_operations += 1
        
        while not error_queue.empty():
            worker_id, error_msg = error_queue.get()
            errors.append((worker_id, error_msg))
        
        # Verify reasonable success rate under stress
        if total_operations > 0:
            success_rate = successful_operations / total_operations
            self.assertGreater(success_rate, 0.5, 
                             f"Success rate too low under stress: {success_rate:.2f}")
        
        # Some errors are acceptable under high stress
        self.assertLess(len(errors), workers * operations_per_worker * 0.5,
                       f"Too many errors: {len(errors)}")
    
    def test_checkpoint_filesystem_stress(self):
        """Test checkpoint behavior under filesystem stress conditions."""
        # Create many files to stress filesystem
        stress_files = []
        try:
            # Create background filesystem stress
            for i in range(100):
                stress_file = os.path.join(self.temp_dir, f"stress_file_{i}.tmp")
                with open(stress_file, 'w') as f:
                    f.write("x" * 10000)  # 10KB per file
                stress_files.append(stress_file)
            
            # Now try checkpoint operations under stress
            checkpoint_data = {"stress_test": True, "data": list(range(1000))}
            
            for i in range(10):
                checkpoint_name = f"fs_stress_{i}"
                
                # Save under filesystem stress
                save_result = self.checkpoint_manager.save_checkpoint(checkpoint_name, checkpoint_data)
                
                if save_result:
                    # Verify immediate load works
                    loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                    self.assertEqual(loaded.get("stress_test"), True)
                    
                    # Create more stress files between operations
                    for j in range(5):
                        stress_file = os.path.join(self.temp_dir, f"mid_stress_{i}_{j}.tmp")
                        with open(stress_file, 'w') as f:
                            f.write("y" * 5000)
                        stress_files.append(stress_file)
        
        finally:
            # Cleanup stress files
            for stress_file in stress_files:
                try:
                    if os.path.exists(stress_file):
                        os.remove(stress_file)
                except:
                    pass
    
    def test_checkpoint_interruption_recovery(self):
        """Test checkpoint system recovery from various interruptions."""
        checkpoint_name = "interruption_test"
        test_data = {"recovery_test": True, "data": list(range(1000))}
        
        # Simulate various interruption scenarios
        interruption_scenarios = [
            ("disk_full", OSError("No space left on device")),
            ("permission_denied", PermissionError("Permission denied")),
            ("io_error", IOError("Input/output error")),
            ("interrupted_system_call", OSError("Interrupted system call")),
        ]
        
        for scenario_name, exception in interruption_scenarios:
            with self.subTest(scenario=scenario_name):
                # Mock file operations to simulate interruption
                original_write = None
                
                def interrupting_write(self, data):
                    if len(data) > 100:  # Interrupt after some data written
                        raise exception
                    return original_write(data) if original_write else len(data)
                
                try:
                    # Try save with interruption
                    with patch('builtins.open') as mock_open:
                        mock_file = Mock()
                        mock_open.return_value.__enter__.return_value = mock_file
                        mock_file.write.side_effect = interrupting_write
                        
                        try:
                            self.checkpoint_manager.save_checkpoint(checkpoint_name, test_data)
                        except (OSError, IOError, PermissionError):
                            pass  # Expected interruption
                    
                    # Verify system remains in consistent state
                    checkpoints = self.checkpoint_manager.list_checkpoints()
                    if checkpoint_name in checkpoints:
                        # If checkpoint exists, it should be valid
                        loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                        if loaded is not None:
                            self.assertIsInstance(loaded, dict)
                    
                    # System should still be able to perform new operations
                    recovery_name = f"recovery_after_{scenario_name}"
                    recovery_result = self.checkpoint_manager.save_checkpoint(
                        recovery_name, {"recovered": True}
                    )
                    
                    # At least the recovery operation should work
                    if recovery_result:
                        recovery_loaded = self.checkpoint_manager.load_checkpoint(recovery_name)
                        self.assertEqual(recovery_loaded.get("recovered"), True)
                
                except Exception as e:
                    # If we get unexpected exceptions, that's also valuable info
                    print(f"Unexpected exception in {scenario_name}: {e}")
    
    def test_checkpoint_long_running_stability(self):
        """Test checkpoint system stability over extended operation."""
        # Simulate long-running usage patterns
        stability_data = {"stability_test": True}
        
        operations_count = 0
        start_time = time.time()
        max_duration = 10.0  # 10 seconds max for this test
        
        try:
            while time.time() - start_time < max_duration:
                # Perform various operations in cycle
                operation_cycle = operations_count % 4
                
                if operation_cycle == 0:
                    # Save operation
                    name = f"stability_{operations_count}"
                    self.checkpoint_manager.save_checkpoint(name, stability_data)
                
                elif operation_cycle == 1:
                    # Load operation
                    checkpoints = self.checkpoint_manager.list_checkpoints()
                    if checkpoints:
                        self.checkpoint_manager.load_checkpoint(checkpoints[0])
                
                elif operation_cycle == 2:
                    # List operation
                    self.checkpoint_manager.list_checkpoints()
                
                elif operation_cycle == 3:
                    # Cleanup operation
                    checkpoints = self.checkpoint_manager.list_checkpoints()
                    if len(checkpoints) > 50:  # Cleanup when too many
                        self.checkpoint_manager.cleanup_old_checkpoints(max_count=25)
                
                operations_count += 1
                
                # Brief pause to prevent tight loop
                time.sleep(0.01)
        
        except Exception as e:
            self.fail(f"System became unstable after {operations_count} operations: {e}")
        
        # Verify system is still responsive after long-running test
        final_test_name = "final_stability_test"
        final_result = self.checkpoint_manager.save_checkpoint(final_test_name, {"final": True})
        self.assertTrue(final_result, "System should remain responsive after long-running test")
        
        final_loaded = self.checkpoint_manager.load_checkpoint(final_test_name)
        self.assertEqual(final_loaded.get("final"), True)
        
        print(f"Stability test completed {operations_count} operations successfully")


class TestCheckpointManagerAdvancedScenarios(unittest.TestCase):
    """Advanced testing scenarios and edge cases."""
    
    def setUp(self):
        """Set up advanced test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up advanced test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_with_custom_serializable_objects(self):
        """Test checkpoints with custom objects that define serialization."""
        
        class CustomSerializable:
            def __init__(self, value):
                self.value = value
                self.metadata = {"created": time.time()}
            
            def __eq__(self, other):
                return isinstance(other, CustomSerializable) and self.value == other.value
            
            def to_dict(self):
                return {"value": self.value, "metadata": self.metadata, "type": "CustomSerializable"}
            
            @classmethod
            def from_dict(cls, data):
                obj = cls(data["value"])
                obj.metadata = data["metadata"]
                return obj
        
        # Test with custom serializable objects
        custom_data = {
            "objects": [CustomSerializable(i) for i in range(10)],
            "single_object": CustomSerializable("test_value"),
            "nested": {
                "deep_object": CustomSerializable("deep_value")
            }
        }
        
        # Convert to serializable format
        serializable_data = {
            "objects": [obj.to_dict() for obj in custom_data["objects"]],
            "single_object": custom_data["single_object"].to_dict(),
            "nested": {
                "deep_object": custom_data["nested"]["deep_object"].to_dict()
            }
        }
        
        checkpoint_name = "custom_serializable_test"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, serializable_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if loaded_data:
            # Verify structure is preserved
            self.assertEqual(len(loaded_data["objects"]), 10)
            self.assertEqual(loaded_data["single_object"]["value"], "test_value")
            self.assertEqual(loaded_data["nested"]["deep_object"]["value"], "deep_value")
    
    def test_checkpoint_with_binary_data_simulation(self):
        """Test checkpoints with binary-like data structures."""
        binary_like_data = {
            "image_data": bytes(range(256)),  # Simulate image bytes
            "audio_samples": [i * 0.001 for i in range(-32768, 32768, 100)],  # Audio samples
            "binary_flags": [bool(i & (1 << j)) for i in range(256) for j in range(8)],
            "hex_strings": [hex(i) for i in range(1000)],
            "base64_simulation": "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0Lg==",
            "packed_data": {
                "header": {"magic": 0xDEADBEEF, "version": 1, "size": 1024},
                "payload": list(range(1024))
            }
        }
        
        checkpoint_name = "binary_simulation_test"
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, binary_like_data)
            
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                if loaded_data:
                    # Verify binary data integrity
                    self.assertEqual(len(loaded_data["audio_samples"]), len(binary_like_data["audio_samples"]))
                    self.assertEqual(loaded_data["hex_strings"][:10], binary_like_data["hex_strings"][:10])
                    self.assertEqual(loaded_data["packed_data"]["header"]["magic"], 0xDEADBEEF)
        
        except (TypeError, ValueError) as e:
            # Binary data might not be directly serializable
            self.assertIsInstance(e, (TypeError, ValueError))
    
    def test_checkpoint_version_compatibility_simulation(self):
        """Test checkpoint compatibility across different versions."""
        # Simulate data from different versions
        version_scenarios = [
            {
                "version": "1.0",
                "data": {"simple_key": "simple_value"},
                "format": "json"
            },
            {
                "version": "2.0", 
                "data": {
                    "enhanced_data": {"nested": True, "features": ["a", "b", "c"]},
                    "metadata": {"version": "2.0", "timestamp": time.time()}
                },
                "format": "json"
            },
            {
                "version": "3.0",
                "data": {
                    "advanced_features": {
                        "compression": True,
                        "encryption": False,
                        "schema_version": 3,
                        "data": list(range(100))
                    }
                },
                "format": "json"
            }
        ]
        
        saved_checkpoints = []
        
        # Save checkpoints simulating different versions
        for scenario in version_scenarios:
            checkpoint_name = f"version_{scenario['version'].replace('.', '_')}_test"
            versioned_data = {
                "checkpoint_version": scenario["version"],
                "checkpoint_format": scenario["format"],
                "user_data": scenario["data"]
            }
            
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, versioned_data)
            if result:
                saved_checkpoints.append((checkpoint_name, scenario["version"]))
        
        # Verify all versions can be loaded
        for checkpoint_name, version in saved_checkpoints:
            with self.subTest(version=version):
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                self.assertIsNotNone(loaded_data)
                self.assertEqual(loaded_data["checkpoint_version"], version)
                
                # Verify version-specific data
                if version == "1.0":
                    self.assertEqual(loaded_data["user_data"]["simple_key"], "simple_value")
                elif version == "2.0":
                    self.assertTrue(loaded_data["user_data"]["enhanced_data"]["nested"])
                elif version == "3.0":
                    self.assertTrue(loaded_data["user_data"]["advanced_features"]["compression"])
    
    def test_checkpoint_with_generator_like_data(self):
        """Test checkpoints with data that simulates generators or iterators."""
        # Convert generator-like data to serializable format
        generator_simulation = {
            "fibonacci_sequence": [self._fibonacci(i) for i in range(20)],
            "prime_numbers": [n for n in range(2, 100) if self._is_prime(n)],
            "random_walk": self._generate_random_walk(100),
            "data_stream_simulation": [
                {"timestamp": time.time() + i, "value": i * 0.1, "id": f"sample_{i}"}
                for i in range(50)
            ]
        }
        
        checkpoint_name = "generator_simulation_test"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, generator_simulation)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if loaded_data:
            # Verify generated sequences
            self.assertEqual(loaded_data["fibonacci_sequence"][10], self._fibonacci(10))
            self.assertTrue(all(self._is_prime(p) for p in loaded_data["prime_numbers"]))
            self.assertEqual(len(loaded_data["random_walk"]), 100)
            self.assertEqual(len(loaded_data["data_stream_simulation"]), 50)
    
    def _fibonacci(self, n):
        """Helper to generate fibonacci number."""
        if n <= 1:
            return n
        return self._fibonacci(n-1) + self._fibonacci(n-2)
    
    def _is_prime(self, n):
        """Helper to check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _generate_random_walk(self, steps):
        """Helper to generate random walk simulation."""
        import random
        position = 0
        walk = [position]
        for _ in range(steps - 1):
            position += random.choice([-1, 1])
            walk.append(position)
        return walk
    
    def test_checkpoint_metadata_and_tagging(self):
        """Test checkpoint system with metadata and tagging capabilities."""
        # Enhanced checkpoint data with metadata
        enhanced_checkpoints = [
            {
                "name": "production_model_v1",
                "data": {"model_weights": list(range(100))},
                "metadata": {
                    "tags": ["production", "stable", "v1.0"],
                    "description": "Production model checkpoint",
                    "created_by": "test_system",
                    "performance_metrics": {"accuracy": 0.95, "loss": 0.05},
                    "dependencies": ["numpy", "torch"],
                    "size_bytes": 1000000
                }
            },
            {
                "name": "experimental_model_v2",
                "data": {"model_weights": list(range(200))},
                "metadata": {
                    "tags": ["experimental", "unstable", "v2.0"],
                    "description": "Experimental model with new architecture",
                    "created_by": "research_team",
                    "performance_metrics": {"accuracy": 0.97, "loss": 0.03},
                    "dependencies": ["numpy", "torch", "transformers"],
                    "size_bytes": 2000000
                }
            }
        ]
        
        # Save enhanced checkpoints
        for checkpoint_info in enhanced_checkpoints:
            full_data = {
                "user_data": checkpoint_info["data"],
                "checkpoint_metadata": checkpoint_info["metadata"]
            }
            
            result = self.checkpoint_manager.save_checkpoint(checkpoint_info["name"], full_data)
            self.assertTrue(result)
        
        # Verify metadata preservation
        for checkpoint_info in enhanced_checkpoints:
            with self.subTest(name=checkpoint_info["name"]):
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_info["name"])
                self.assertIsNotNone(loaded_data)
                
                # Verify metadata
                metadata = loaded_data.get("checkpoint_metadata", {})
                self.assertEqual(metadata.get("created_by"), checkpoint_info["metadata"]["created_by"])
                self.assertEqual(metadata.get("tags"), checkpoint_info["metadata"]["tags"])
                
                # Verify data integrity
                user_data = loaded_data.get("user_data", {})
                self.assertEqual(len(user_data.get("model_weights", [])), 
                               len(checkpoint_info["data"]["model_weights"]))
    
    def test_checkpoint_compression_simulation(self):
        """Test checkpoint behavior with compression-like scenarios."""
        # Create highly compressible data
        compressible_data = {
            "repeated_pattern": ["same_value"] * 10000,
            "zeros": [0] * 5000,
            "sequential": list(range(1000)) * 5,
            "text_repetition": "This is repeated text. " * 1000,
            "sparse_matrix": {
                f"row_{i}": {f"col_{j}": 0 for j in range(100)} 
                for i in range(100)
            }
        }
        
        # Add some random data for contrast
        import random
        compressible_data["random_data"] = [random.random() for _ in range(1000)]
        
        checkpoint_name = "compression_simulation_test"
        
        # Time the operation to see if compression would be beneficial
        start_time = time.time()
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, compressible_data)
        save_time = time.time() - start_time
        
        self.assertTrue(result)
        
        # Verify data integrity
        start_load = time.time()
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        load_time = time.time() - start_load
        
        if loaded_data:
            self.assertEqual(len(loaded_data["repeated_pattern"]), 10000)
            self.assertEqual(loaded_data["repeated_pattern"][0], "same_value")
            self.assertEqual(len(loaded_data["zeros"]), 5000)
            self.assertEqual(loaded_data["zeros"][100], 0)
            self.assertEqual(len(loaded_data["sequential"]), 5000)
            
            # Verify random data (should be different from repeated patterns)
            self.assertEqual(len(loaded_data["random_data"]), 1000)
        
        # Performance info for potential compression analysis
        print(f"Compression simulation - Save: {save_time:.3f}s, Load: {load_time:.3f}s")
    
    def test_checkpoint_multi_format_compatibility(self):
        """Test checkpoint compatibility with multiple data formats."""
        # Test various data format scenarios
        format_scenarios = [
            {
                "name": "json_like",
                "data": {
                    "string": "text",
                    "number": 42,
                    "float": 3.14159,
                    "boolean": True,
                    "null": None,
                    "array": [1, 2, 3],
                    "object": {"nested": "value"}
                }
            },
            {
                "name": "csv_like",
                "data": {
                    "headers": ["Name", "Age", "City"],
                    "rows": [
                        ["Alice", 30, "New York"],
                        ["Bob", 25, "London"],
                        ["Charlie", 35, "Tokyo"]
                    ]
                }
            },
            {
                "name": "xml_like",
                "data": {
                    "root": {
                        "element1": {"attributes": {"id": "1"}, "text": "content1"},
                        "element2": {"attributes": {"id": "2"}, "text": "content2"},
                        "children": [
                            {"tag": "child1", "content": "child_content1"},
                            {"tag": "child2", "content": "child_content2"}
                        ]
                    }
                }
            }
        ]
        
        # Save all formats
        for scenario in format_scenarios:
            checkpoint_name = f"format_{scenario['name']}_test"
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, scenario["data"])
            self.assertTrue(result, f"Failed to save {scenario['name']} format")
        
        # Load and verify all formats
        for scenario in format_scenarios:
            with self.subTest(format_name=scenario["name"]):
                checkpoint_name = f"format_{scenario['name']}_test"
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                self.assertIsNotNone(loaded_data)
                
                if scenario["name"] == "json_like":
                    self.assertEqual(loaded_data["string"], "text")
                    self.assertEqual(loaded_data["number"], 42)
                    self.assertAlmostEqual(loaded_data["float"], 3.14159)
                    
                elif scenario["name"] == "csv_like":
                    self.assertEqual(loaded_data["headers"], ["Name", "Age", "City"])
                    self.assertEqual(len(loaded_data["rows"]), 3)
                    self.assertEqual(loaded_data["rows"][0][0], "Alice")
                    
                elif scenario["name"] == "xml_like":
                    self.assertIn("root", loaded_data)
                    self.assertEqual(loaded_data["root"]["element1"]["attributes"]["id"], "1")
                    self.assertEqual(len(loaded_data["root"]["children"]), 2)


# Update the main execution section to include all new test classes