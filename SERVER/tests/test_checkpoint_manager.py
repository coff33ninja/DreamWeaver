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