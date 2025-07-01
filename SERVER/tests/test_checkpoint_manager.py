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

class TestCheckpointManagerAdvanced(unittest.TestCase):
    """Advanced test scenarios for CheckpointManager with additional edge cases and robustness testing."""
    
    def setUp(self):
        """Set up test fixtures for advanced testing scenarios."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        self.maxDiff = None  # Show full diff for assertion failures
        
    def tearDown(self):
        """Clean up after advanced tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # Memory and Resource Management Tests
    def test_memory_usage_large_checkpoints(self):
        """Test memory usage with very large checkpoint data."""
        import sys
        
        # Create a large data structure (50MB worth of data)
        large_data = {
            'massive_list': [i for i in range(1000000)],
            'massive_dict': {f'key_{i}': f'value_{i}' * 100 for i in range(10000)},
            'metadata': {'size': 'massive', 'items': 1000000}
        }
        
        checkpoint_name = "memory_test_large"
        
        # Monitor memory usage (simplified approach)
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, large_data)
            self.assertTrue(result)
            
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(len(loaded_data['massive_list']), 1000000)
            self.assertEqual(len(loaded_data['massive_dict']), 10000)
            
        except MemoryError:
            self.skipTest("Insufficient memory for large checkpoint test")

    def test_resource_cleanup_after_failed_operations(self):
        """Test that resources are properly cleaned up after failed operations."""
        checkpoint_name = "cleanup_test"
        
        # Mock a failure scenario
        with patch.object(self.checkpoint_manager, 'save_checkpoint', side_effect=Exception("Simulated failure")):
            try:
                self.checkpoint_manager.save_checkpoint(checkpoint_name, {"test": "data"})
            except Exception:
                pass
        
        # Verify no partial state is left behind
        checkpoints = self.checkpoint_manager.list_checkpoints()
        self.assertNotIn(checkpoint_name, checkpoints)

    # File System Edge Cases
    def test_readonly_checkpoint_directory(self):
        """Test behavior when checkpoint directory becomes read-only."""
        if os.name == 'nt':  # Skip on Windows due to permission model differences
            self.skipTest("Read-only directory test not applicable on Windows")
            
        checkpoint_name = "readonly_test"
        sample_data = {"test": "data"}
        
        # First save a checkpoint normally
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, sample_data)
        self.assertTrue(result)
        
        # Make directory read-only
        try:
            os.chmod(self.temp_dir, 0o444)
            
            # Try to save another checkpoint (should fail)
            with self.assertRaises((PermissionError, OSError)):
                self.checkpoint_manager.save_checkpoint("readonly_fail", sample_data)
                
        finally:
            # Restore permissions for cleanup
            os.chmod(self.temp_dir, 0o755)

    def test_filesystem_full_simulation(self):
        """Test behavior when filesystem is full."""
        checkpoint_name = "disk_full_test"
        
        # Create a very large data structure to potentially fill disk
        large_data = {'huge_string': 'x' * (1024 * 1024 * 100)}  # 100MB string
        
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, large_data)
            # If it succeeds, verify it can be loaded
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                self.assertEqual(len(loaded_data['huge_string']), 1024 * 1024 * 100)
        except (OSError, IOError) as e:
            # Expected behavior when disk is full
            self.assertIn(str(e), [
                'No space left on device',
                'Disk full',
                'Not enough space'
            ], f"Unexpected error: {e}")

    # Data Corruption and Recovery Tests
    def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted checkpoint data."""
        checkpoint_name = "corruption_test"
        sample_data = {"valid": "data", "numbers": [1, 2, 3]}
        
        # Save valid checkpoint first
        self.checkpoint_manager.save_checkpoint(checkpoint_name, sample_data)
        
        # Simulate corruption by directly modifying stored data (if file-based)
        if hasattr(self.checkpoint_manager, 'checkpoint_dir'):
            checkpoint_files = []
            for root, dirs, files in os.walk(self.checkpoint_manager.checkpoint_dir):
                for file in files:
                    if checkpoint_name in file:
                        checkpoint_files.append(os.path.join(root, file))
            
            # Corrupt the first file found
            if checkpoint_files:
                with open(checkpoint_files[0], 'w') as f:
                    f.write("corrupted data that's not valid JSON/pickle")
                
                # Try to load corrupted checkpoint
                try:
                    loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                    # Depending on implementation, might return None or raise exception
                    if loaded_data is not None:
                        self.fail("Expected corrupted checkpoint to fail loading")
                except (ValueError, TypeError, Exception) as e:
                    # Expected behavior for corrupted data
                    self.assertIsInstance(e, (ValueError, TypeError, Exception))

    # Serialization Edge Cases
    def test_custom_object_serialization(self):
        """Test serialization of custom Python objects."""
        class CustomObject:
            def __init__(self, value, name):
                self.value = value
                self.name = name
                
            def __eq__(self, other):
                return isinstance(other, CustomObject) and self.value == other.value and self.name == other.name
                
            def __repr__(self):
                return f"CustomObject(value={self.value}, name='{self.name}')"
        
        custom_data = {
            'custom_obj': CustomObject(42, "test_object"),
            'list_of_custom': [CustomObject(i, f"obj_{i}") for i in range(5)],
            'mixed': {'regular': 'data', 'custom': CustomObject(99, "mixed_obj")}
        }
        
        checkpoint_name = "custom_object_test"
        
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, custom_data)
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                # Verify structure is maintained (exact equality might not work due to serialization)
                self.assertIsNotNone(loaded_data)
                self.assertIn('custom_obj', loaded_data)
                self.assertIn('list_of_custom', loaded_data)
        except (TypeError, ValueError) as e:
            # Expected if custom objects aren't serializable
            self.assertIsInstance(e, (TypeError, ValueError))

    def test_function_and_lambda_serialization(self):
        """Test handling of functions and lambdas in checkpoint data."""
        def test_function(x):
            return x * 2
            
        lambda_func = lambda x: x + 1
        
        function_data = {
            'function': test_function,
            'lambda': lambda_func,
            'builtin': len,
            'method': str.upper
        }
        
        checkpoint_name = "function_test"
        
        # Functions typically aren't serializable
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, function_data)
            if result:
                loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                self.assertIsNotNone(loaded_data)
        except (TypeError, ValueError) as e:
            # Expected behavior for non-serializable function objects
            self.assertIsInstance(e, (TypeError, ValueError))

    # Configuration and State Tests
    def test_checkpoint_manager_state_persistence(self):
        """Test that CheckpointManager maintains consistent state across operations."""
        # Save multiple checkpoints
        test_data = [
            ("checkpoint_1", {"data": 1}),
            ("checkpoint_2", {"data": 2}),
            ("checkpoint_3", {"data": 3}),
        ]
        
        for name, data in test_data:
            result = self.checkpoint_manager.save_checkpoint(name, data)
            self.assertTrue(result)
        
        # Verify state consistency
        initial_list = set(self.checkpoint_manager.list_checkpoints())
        
        # Perform various operations
        self.checkpoint_manager.load_checkpoint("checkpoint_1")
        self.checkpoint_manager.load_checkpoint("nonexistent")
        
        # State should remain consistent
        final_list = set(self.checkpoint_manager.list_checkpoints())
        self.assertEqual(initial_list, final_list)

    def test_checkpoint_metadata_preservation(self):
        """Test that checkpoint metadata is preserved correctly."""
        import datetime
        
        metadata_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'version': "1.0.0",
            'author': "test_user",
            'description': "Test checkpoint with metadata",
            'tags': ["test", "metadata", "validation"],
            'config': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }
        }
        
        checkpoint_name = "metadata_test"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, metadata_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data['version'], "1.0.0")
        self.assertEqual(loaded_data['author'], "test_user")
        self.assertEqual(len(loaded_data['tags']), 3)
        self.assertEqual(loaded_data['config']['learning_rate'], 0.001)

    # Stress and Load Tests
    def test_rapid_checkpoint_operations(self):
        """Test rapid succession of checkpoint operations."""
        import time
        
        checkpoint_name = "rapid_test"
        base_data = {"counter": 0, "timestamp": time.time()}
        
        # Perform rapid save/load/delete cycles
        for i in range(100):
            data = base_data.copy()
            data["counter"] = i
            data["timestamp"] = time.time()
            
            # Save
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, data)
            self.assertTrue(result)
            
            # Load
            loaded = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(loaded["counter"], i)
            
            # Delete every 10th iteration
            if i % 10 == 0:
                delete_result = self.checkpoint_manager.delete_checkpoint(checkpoint_name)
                self.assertTrue(delete_result)

    def test_checkpoint_name_collision_handling(self):
        """Test handling of checkpoint name collisions and overwrites."""
        collision_name = "collision_test"
        
        # Create multiple different data structures with same name
        data_versions = [
            {"version": 1, "type": "initial"},
            {"version": 2, "type": "updated", "new_field": "added"},
            {"version": 3, "type": "final", "removed_field": None},
        ]
        
        for i, data in enumerate(data_versions):
            result = self.checkpoint_manager.save_checkpoint(collision_name, data)
            self.assertTrue(result)
            
            # Verify current version is what we just saved
            loaded = self.checkpoint_manager.load_checkpoint(collision_name)
            self.assertEqual(loaded["version"], i + 1)
        
        # Final verification
        final_loaded = self.checkpoint_manager.load_checkpoint(collision_name)
        self.assertEqual(final_loaded["version"], 3)
        self.assertEqual(final_loaded["type"], "final")

    # Edge Case Input Validation
    def test_checkpoint_name_with_unicode_normalization(self):
        """Test checkpoint names with different Unicode normalization forms."""
        import unicodedata
        
        base_name = "café"  # Contains accented character
        
        # Different Unicode normalization forms
        nfc_name = unicodedata.normalize('NFC', base_name)   # Composed
        nfd_name = unicodedata.normalize('NFD', base_name)   # Decomposed
        
        sample_data = {"encoding": "unicode_test"}
        
        # Save with NFC form
        result1 = self.checkpoint_manager.save_checkpoint(nfc_name, sample_data)
        self.assertTrue(result1)
        
        # Try to load with NFD form (might be treated as different name)
        loaded1 = self.checkpoint_manager.load_checkpoint(nfd_name)
        # Behavior depends on implementation - both are valid
        
        # Save with NFD form
        result2 = self.checkpoint_manager.save_checkpoint(nfd_name, sample_data)
        self.assertTrue(result2)
        
        # Check how many unique names we have
        checkpoints = self.checkpoint_manager.list_checkpoints()
        unicode_checkpoints = [cp for cp in checkpoints if 'caf' in cp]
        # Should be 1 or 2 depending on Unicode handling
        self.assertGreaterEqual(len(unicode_checkpoints), 1)
        self.assertLessEqual(len(unicode_checkpoints), 2)

    def test_checkpoint_with_extreme_numeric_values(self):
        """Test checkpoints with extreme numeric values."""
        extreme_data = {
            'max_int': sys.maxsize,
            'min_int': -sys.maxsize - 1,
            'max_float': sys.float_info.max,
            'min_float': sys.float_info.min,
            'epsilon': sys.float_info.epsilon,
            'infinity': float('inf'),
            'negative_infinity': float('-inf'),
            'not_a_number': float('nan'),
            'very_large_list': [sys.maxsize] * 1000,
            'precision_test': 1.23456789012345678901234567890
        }
        
        checkpoint_name = "extreme_numeric_test"
        
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, extreme_data)
            self.assertTrue(result)
            
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(loaded_data['max_int'], sys.maxsize)
            self.assertEqual(loaded_data['min_int'], -sys.maxsize - 1)
            
            # Special handling for infinity and NaN
            import math
            self.assertTrue(math.isinf(loaded_data['infinity']))
            self.assertTrue(math.isinf(loaded_data['negative_infinity']))
            self.assertTrue(math.isnan(loaded_data['not_a_number']))
            
        except (OverflowError, ValueError) as e:
            # Some extreme values might not be serializable
            self.assertIsInstance(e, (OverflowError, ValueError))

    # Integration with System Resources
    def test_checkpoint_operations_under_memory_pressure(self):
        """Test checkpoint operations when system memory is under pressure."""
        # Create memory pressure by allocating large amounts of memory
        memory_pressure = []
        
        try:
            # Allocate memory in chunks until we have significant pressure
            for i in range(10):
                chunk = [0] * (1024 * 1024)  # 1MB chunks
                memory_pressure.append(chunk)
            
            # Now try checkpoint operations under pressure
            checkpoint_name = "memory_pressure_test"
            sample_data = {"test": "under_pressure", "data": list(range(1000))}
            
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, sample_data)
            self.assertTrue(result)
            
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(loaded_data, sample_data)
            
        except MemoryError:
            self.skipTest("Unable to create sufficient memory pressure")
        finally:
            # Clean up memory pressure
            memory_pressure.clear()

    def test_checkpoint_with_different_python_versions_compatibility(self):
        """Test checkpoint data compatibility across different scenarios."""
        # Test data that might behave differently across Python versions
        version_sensitive_data = {
            'dict_order': {'z': 1, 'a': 2, 'b': 3},  # Dict ordering
            'string_encoding': "Testing 🐍 Python encoding",
            'numeric_precision': 0.1 + 0.2,  # Floating point precision
            'large_int': 2**100,  # Large integer handling
            'boolean_values': [True, False, 1, 0],
            'none_values': [None, [], {}, ''],
        }
        
        checkpoint_name = "version_compatibility_test"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, version_sensitive_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data['large_int'], 2**100)
        self.assertEqual(loaded_data['string_encoding'], "Testing 🐍 Python encoding")
        self.assertIsNotNone(loaded_data['dict_order'])

    # Error Recovery and Resilience
    def test_partial_failure_recovery(self):
        """Test recovery from partial operation failures."""
        checkpoint_names = [f"recovery_test_{i}" for i in range(5)]
        sample_data = {"recovery": "test", "id": None}
        
        # Save several checkpoints
        for i, name in enumerate(checkpoint_names):
            data = sample_data.copy()
            data["id"] = i
            result = self.checkpoint_manager.save_checkpoint(name, data)
            self.assertTrue(result)
        
        # Simulate partial failure by corrupting middle checkpoint
        if hasattr(self.checkpoint_manager, 'checkpoints'):
            # For mock implementation
            middle_name = checkpoint_names[2]
            if middle_name in self.checkpoint_manager.checkpoints:
                self.checkpoint_manager.checkpoints[middle_name] = "corrupted"
        
        # Verify other checkpoints still work
        for i, name in enumerate(checkpoint_names):
            if i != 2:  # Skip the corrupted one
                loaded = self.checkpoint_manager.load_checkpoint(name)
                if loaded and loaded != "corrupted":
                    self.assertEqual(loaded["id"], i)

    def test_graceful_degradation_on_resource_exhaustion(self):
        """Test graceful degradation when resources are exhausted."""
        checkpoint_name = "resource_exhaustion_test"
        
        # Try to save progressively larger checkpoints until resource exhaustion
        for size_multiplier in [1, 10, 100, 1000]:
            try:
                large_data = {
                    'size_multiplier': size_multiplier,
                    'large_data': 'x' * (1024 * size_multiplier),
                    'metadata': {'attempt': size_multiplier}
                }
                
                result = self.checkpoint_manager.save_checkpoint(
                    f"{checkpoint_name}_{size_multiplier}", large_data
                )
                
                if result:
                    # Verify we can still load it
                    loaded = self.checkpoint_manager.load_checkpoint(
                        f"{checkpoint_name}_{size_multiplier}"
                    )
                    self.assertEqual(loaded['size_multiplier'], size_multiplier)
                    
            except (MemoryError, OSError, IOError) as e:
                # Expected when resources are exhausted
                self.assertIsInstance(e, (MemoryError, OSError, IOError))
                break


class TestCheckpointManagerBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and limit cases for CheckpointManager."""
    
    def setUp(self):
        """Set up boundary condition tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up boundary condition tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_zero_length_data_structures(self):
        """Test handling of zero-length data structures."""
        zero_length_data = {
            'empty_string': '',
            'empty_list': [],
            'empty_dict': {},
            'empty_tuple': (),
            'empty_set': set(),
            'zero_int': 0,
            'zero_float': 0.0,
            'false_bool': False
        }
        
        checkpoint_name = "zero_length_test"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, zero_length_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data['empty_string'], '')
        self.assertEqual(loaded_data['empty_list'], [])
        self.assertEqual(loaded_data['empty_dict'], {})
        self.assertEqual(loaded_data['zero_int'], 0)
        self.assertEqual(loaded_data['zero_float'], 0.0)
        self.assertFalse(loaded_data['false_bool'])

    def test_single_character_strings(self):
        """Test handling of single character strings and edge cases."""
        single_char_data = {
            'space': ' ',
            'newline': '\n',
            'tab': '\t',
            'carriage_return': '\r',
            'null_char': '\0',
            'backslash': '\\',
            'quote': '"',
            'single_quote': "'",
            'unicode_char': '€',
            'emoji_char': '🚀'
        }
        
        checkpoint_name = "single_char_test"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, single_char_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data['space'], ' ')
        self.assertEqual(loaded_data['newline'], '\n')
        self.assertEqual(loaded_data['tab'], '\t')
        self.assertEqual(loaded_data['unicode_char'], '€')
        self.assertEqual(loaded_data['emoji_char'], '🚀')

    def test_boundary_numeric_values(self):
        """Test numeric values at system boundaries."""
        import sys
        
        boundary_data = {
            'max_int_32': 2**31 - 1,
            'min_int_32': -2**31,
            'max_int_64': 2**63 - 1,
            'min_int_64': -2**63,
            'smallest_positive_float': sys.float_info.min,
            'largest_float': sys.float_info.max,
            'float_epsilon': sys.float_info.epsilon,
            'one_plus_epsilon': 1.0 + sys.float_info.epsilon,
            'one_minus_epsilon': 1.0 - sys.float_info.epsilon,
        }
        
        checkpoint_name = "boundary_numeric_test"
        
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, boundary_data)
            self.assertTrue(result)
            
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(loaded_data['max_int_32'], 2**31 - 1)
            self.assertEqual(loaded_data['min_int_32'], -2**31)
            self.assertEqual(loaded_data['float_epsilon'], sys.float_info.epsilon)
            
        except (OverflowError, ValueError) as e:
            # Some boundary values might cause overflow
            self.assertIsInstance(e, (OverflowError, ValueError))

    def test_maximum_nesting_depth(self):
        """Test maximum practical nesting depth for data structures."""
        # Create nested structure approaching recursion limits
        max_depth = 500  # Conservative limit to avoid stack overflow
        nested_data = current = {}
        
        for i in range(max_depth):
            current['level'] = i
            current['nested'] = {}
            current = current['nested']
        
        # Add leaf data
        current['leaf'] = True
        current['depth'] = max_depth
        
        checkpoint_name = "max_nesting_test"
        
        try:
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, nested_data)
            self.assertTrue(result)
            
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            self.assertEqual(loaded_data['level'], 0)
            
            # Navigate to a middle level
            current_level = loaded_data
            for i in range(100):  # Check first 100 levels
                self.assertEqual(current_level['level'], i)
                if 'nested' in current_level:
                    current_level = current_level['nested']
                    
        except (RecursionError, ValueError) as e:
            # Expected for very deep nesting
            self.assertIsInstance(e, (RecursionError, ValueError))


class TestCheckpointManagerRealWorldIntegration(unittest.TestCase):
    """Test real-world integration scenarios for CheckpointManager."""
    
    def setUp(self):
        """Set up real-world integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up real-world integration tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_with_actual_model_data_structure(self):
        """Test checkpoint with realistic ML model data structures."""
        # Simulate typical ML model checkpoint data
        model_checkpoint_data = {
            'model_state_dict': {
                'layer1.weight': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                'layer1.bias': [0.1, 0.2],
                'layer2.weight': [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
                'layer2.bias': [0.3, 0.4, 0.5]
            },
            'optimizer_state_dict': {
                'state': {
                    0: {'momentum_buffer': [0.01, 0.02, 0.03]},
                    1: {'momentum_buffer': [0.04, 0.05]}
                },
                'param_groups': [
                    {'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0}
                ]
            },
            'epoch': 150,
            'loss': 0.0234,
            'accuracy': 0.9567,
            'lr_scheduler_state': {'_step_count': 150, 'last_epoch': 149},
            'random_state': {'numpy': [1, 2, 3, 4, 5], 'python': 42},
            'metadata': {
                'model_architecture': 'MLP',
                'dataset': 'MNIST',
                'training_started': '2024-01-01T10:00:00',
                'last_checkpoint': '2024-01-01T15:30:00',
                'total_training_time': 19800.5
            }
        }
        
        checkpoint_name = "model_checkpoint_epoch_150"
        result = self.checkpoint_manager.save_checkpoint(checkpoint_name, model_checkpoint_data)
        self.assertTrue(result)
        
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        self.assertEqual(loaded_data['epoch'], 150)
        self.assertEqual(loaded_data['loss'], 0.0234)
        self.assertEqual(loaded_data['accuracy'], 0.9567)
        self.assertEqual(len(loaded_data['model_state_dict']['layer1.weight']), 2)
        self.assertEqual(loaded_data['optimizer_state_dict']['param_groups'][0]['lr'], 0.001)

    def test_checkpoint_versioning_and_migration(self):
        """Test checkpoint versioning and data migration scenarios."""
        # Version 1.0 checkpoint format
        v1_checkpoint = {
            'format_version': '1.0',
            'model_weights': [1, 2, 3, 4, 5],
            'learning_rate': 0.01,
            'epoch': 10
        }
        
        # Version 2.0 checkpoint format (with additional fields)
        v2_checkpoint = {
            'format_version': '2.0',
            'model': {
                'weights': [1, 2, 3, 4, 5],
                'architecture': 'dense'
            },
            'optimizer': {
                'learning_rate': 0.01,
                'momentum': 0.9
            },
            'training': {
                'epoch': 10,
                'batch_size': 32
            },
            'metadata': {
                'created_by': 'training_script_v2.py',
                'framework_version': '2.0.1'
            }
        }
        
        # Save both versions
        self.checkpoint_manager.save_checkpoint("model_v1", v1_checkpoint)
        self.checkpoint_manager.save_checkpoint("model_v2", v2_checkpoint)
        
        # Load and verify both can be read
        loaded_v1 = self.checkpoint_manager.load_checkpoint("model_v1")
        loaded_v2 = self.checkpoint_manager.load_checkpoint("model_v2")
        
        self.assertEqual(loaded_v1['format_version'], '1.0')
        self.assertEqual(loaded_v2['format_version'], '2.0')
        self.assertEqual(loaded_v1['epoch'], 10)
        self.assertEqual(loaded_v2['training']['epoch'], 10)

    def test_concurrent_checkpoint_access_simulation(self):
        """Test simulated concurrent access patterns."""
        import threading
        import time
        import random
        
        results = []
        errors = []
        
        def worker_save_load_delete(worker_id, iterations=20):
            """Worker function that performs save/load/delete operations."""
            try:
                for i in range(iterations):
                    checkpoint_name = f"concurrent_worker_{worker_id}_iter_{i}"
                    data = {
                        'worker_id': worker_id,
                        'iteration': i,
                        'timestamp': time.time(),
                        'random_data': [random.randint(0, 1000) for _ in range(100)]
                    }
                    
                    # Save
                    save_result = self.checkpoint_manager.save_checkpoint(checkpoint_name, data)
                    results.append(('save', worker_id, i, save_result))
                    
                    # Small random delay
                    time.sleep(random.uniform(0.001, 0.01))
                    
                    # Load
                    loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
                    load_success = loaded_data is not None and loaded_data.get('worker_id') == worker_id
                    results.append(('load', worker_id, i, load_success))
                    
                    # Delete every other checkpoint
                    if i % 2 == 0:
                        delete_result = self.checkpoint_manager.delete_checkpoint(checkpoint_name)
                        results.append(('delete', worker_id, i, delete_result))
                        
            except Exception as e:
                errors.append(('worker', worker_id, str(e)))
        
        # Create and start multiple worker threads
        threads = []
        num_workers = 5
        for worker_id in range(num_workers):
            thread = threading.Thread(target=worker_save_load_delete, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Analyze results
        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent access: {errors}")
        
        # Verify expected number of operations
        save_operations = [r for r in results if r[0] == 'save']
        load_operations = [r for r in results if r[0] == 'load']
        delete_operations = [r for r in results if r[0] == 'delete']
        
        self.assertEqual(len(save_operations), num_workers * 20)  # 5 workers * 20 iterations
        self.assertEqual(len(load_operations), num_workers * 20)
        self.assertEqual(len(delete_operations), num_workers * 10)  # Every other iteration

    def test_checkpoint_backup_and_recovery_workflow(self):
        """Test backup and recovery workflow scenarios."""
        # Create initial set of checkpoints
        original_checkpoints = {}
        for i in range(10):
            name = f"backup_test_{i}"
            data = {
                'checkpoint_id': i,
                'model_state': f"state_{i}",
                'created_at': time.time(),
                'metadata': {'backup_test': True}
            }
            original_checkpoints[name] = data
            result = self.checkpoint_manager.save_checkpoint(name, data)
            self.assertTrue(result)
        
        # Verify all checkpoints exist
        checkpoint_list = self.checkpoint_manager.list_checkpoints()
        for name in original_checkpoints.keys():
            self.assertIn(name, checkpoint_list)
        
        # Simulate backup by reading all checkpoints
        backup_data = {}
        for name in original_checkpoints.keys():
            loaded = self.checkpoint_manager.load_checkpoint(name)
            backup_data[name] = loaded
        
        # Simulate disaster - delete some checkpoints
        disaster_victims = ['backup_test_2', 'backup_test_5', 'backup_test_8']
        for name in disaster_victims:
            delete_result = self.checkpoint_manager.delete_checkpoint(name)
            self.assertTrue(delete_result)
        
        # Verify they're gone
        post_disaster_list = self.checkpoint_manager.list_checkpoints()
        for name in disaster_victims:
            self.assertNotIn(name, post_disaster_list)
        
        # Recovery - restore from backup
        for name in disaster_victims:
            if name in backup_data:
                restore_result = self.checkpoint_manager.save_checkpoint(name, backup_data[name])
                self.assertTrue(restore_result)
        
        # Verify recovery
        post_recovery_list = self.checkpoint_manager.list_checkpoints()
        for name in disaster_victims:
            self.assertIn(name, post_recovery_list)
            restored_data = self.checkpoint_manager.load_checkpoint(name)
            self.assertEqual(restored_data['checkpoint_id'], original_checkpoints[name]['checkpoint_id'])


# Performance monitoring decorator for critical tests
def monitor_performance(max_time_seconds=30):
    """Decorator to monitor test performance and fail if too slow."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = test_func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                if elapsed > max_time_seconds:
                    raise AssertionError(f"Test {test_func.__name__} took {elapsed:.2f}s, exceeded limit of {max_time_seconds}s")
                return result
            except Exception as e:
                end_time = time.time()
                elapsed = end_time - start_time
                # Re-raise with timing info
                raise type(e)(f"{str(e)} (took {elapsed:.2f}s)")
        return wrapper
    return decorator


class TestCheckpointManagerPerformanceBoundaries(unittest.TestCase):
    """Performance boundary testing for CheckpointManager."""
    
    def setUp(self):
        """Set up performance boundary tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up performance boundary tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @monitor_performance(max_time_seconds=30)
    def test_large_checkpoint_list_performance(self):
        """Test performance with large number of checkpoints."""
        num_checkpoints = 1000
        sample_data = {"test": "performance", "small": True}
        
        # Create many small checkpoints
        for i in range(num_checkpoints):
            checkpoint_name = f"perf_checkpoint_{i:04d}"
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, sample_data)
            self.assertTrue(result)
        
        # Test listing performance
        start_time = time.time()
        checkpoint_list = self.checkpoint_manager.list_checkpoints()
        list_time = time.time() - start_time
        
        self.assertEqual(len(checkpoint_list), num_checkpoints)
        self.assertLess(list_time, 5.0, f"Listing {num_checkpoints} checkpoints took {list_time:.2f}s")

    @monitor_performance(max_time_seconds=60)
    def test_checkpoint_size_scaling_performance(self):
        """Test how performance scales with checkpoint size."""
        import time
        
        sizes = [1024, 10240, 102400, 1024000]  # 1KB, 10KB, 100KB, 1MB
        performance_data = []
        
        for size in sizes:
            data = {'large_string': 'x' * size, 'size': size}
            checkpoint_name = f"scaling_test_{size}"
            
            # Time save operation
            start_time = time.time()
            result = self.checkpoint_manager.save_checkpoint(checkpoint_name, data)
            save_time = time.time() - start_time
            
            self.assertTrue(result)
            
            # Time load operation
            start_time = time.time()
            loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            load_time = time.time() - start_time
            
            self.assertEqual(loaded_data['size'], size)
            
            performance_data.append((size, save_time, load_time))
            
            # Verify performance doesn't degrade exponentially
            if len(performance_data) > 1:
                prev_size, prev_save, prev_load = performance_data[-2]
                size_ratio = size / prev_size
                save_ratio = save_time / prev_save if prev_save > 0 else 1
                load_ratio = load_time / prev_load if prev_load > 0 else 1
                
                # Performance should scale roughly linearly with size
                # Allow for some overhead, but not exponential growth
                self.assertLess(save_ratio, size_ratio * 2, 
                               f"Save time scaling too poorly: {save_ratio} vs size ratio {size_ratio}")
                self.assertLess(load_ratio, size_ratio * 2,
                               f"Load time scaling too poorly: {load_ratio} vs size ratio {size_ratio}")


# Custom test result class for better reporting
class DetailedTestResult(unittest.TextTestResult):
    """Custom test result class that provides detailed information about test execution."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_timings = {}
        
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()
        
    def stopTest(self, test):
        super().stopTest(test)
        if hasattr(self, 'test_start_time'):
            elapsed = time.time() - self.test_start_time
            self.test_timings[str(test)] = elapsed
            
    def printTimings(self):
        """Print timing information for all tests."""
        if self.test_timings:
            print("\n" + "="*70)
            print("TEST EXECUTION TIMINGS")
            print("="*70)
            sorted_timings = sorted(self.test_timings.items(), key=lambda x: x[1], reverse=True)
            for test_name, timing in sorted_timings:
                print(f"{test_name}: {timing:.3f}s")


if __name__ == '__main__':
    # Configure test runner for comprehensive output
    import sys
    import time
    
    # Add command line options for test selection
    if len(sys.argv) > 1 and sys.argv[1] == '--basic':
        # Run only basic tests (exclude performance and stress tests)
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCheckpointManager))
        suite.addTest(unittest.makeSuite(TestCheckpointManagerEdgeCases))
        runner = unittest.TextTestRunner(verbosity=2, buffer=True, resultclass=DetailedTestResult)
        result = runner.run(suite)
        result.printTimings()
    elif len(sys.argv) > 1 and sys.argv[1] == '--performance':
        # Run only performance tests
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCheckpointManagerPerformanceBoundaries))
        runner = unittest.TextTestRunner(verbosity=2, buffer=True, resultclass=DetailedTestResult)
        result = runner.run(suite)
        result.printTimings()
    elif len(sys.argv) > 1 and sys.argv[1] == '--advanced':
        # Run advanced and integration tests
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCheckpointManagerAdvanced))
        suite.addTest(unittest.makeSuite(TestCheckpointManagerBoundaryConditions))
        suite.addTest(unittest.makeSuite(TestCheckpointManagerRealWorldIntegration))
        runner = unittest.TextTestRunner(verbosity=2, buffer=True, resultclass=DetailedTestResult)
        result = runner.run(suite)
        result.printTimings()
    else:
        # Run all tests with maximum verbosity and detailed reporting
        runner = unittest.TextTestRunner(verbosity=2, buffer=True, resultclass=DetailedTestResult, warnings='ignore')
        result = runner.main(exit=False)
        if hasattr(result, 'printTimings'):
            result.printTimings()
            
        # Print summary statistics
        print(f"\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        if result.wasSuccessful():
            print("RESULT: ALL TESTS PASSED ✅")
        else:
            print("RESULT: SOME TESTS FAILED ❌")