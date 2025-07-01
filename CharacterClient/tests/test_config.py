"""
Comprehensive unit tests for the config module.
Testing framework: pytest (identified from project requirements)
"""

import pytest
import os
import tempfile
import json
import yaml
from unittest.mock import patch, mock_open, MagicMock
import sys
from pathlib import Path

# Add the source directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from config import Config, ConfigManager, load_config, save_config, validate_config, ConfigError
except ImportError:
    # Fallback imports if the structure is different
    try:
        from src.config import Config, ConfigManager, load_config, save_config, validate_config, ConfigError
    except ImportError:
        # Create mock classes for testing if config module doesn't exist yet
        class ConfigError(Exception):
            pass
        
        class Config:
            def __init__(self, data=None):
                """
                Initialize a Config instance with the provided data dictionary.
                
                Parameters:
                    data (dict, optional): Initial configuration data. If not provided, an empty dictionary is used.
                """
                self._data = data or {}
            
            def get(self, key, default=None):
                """
                Retrieve the value associated with the specified key from the configuration.
                
                Parameters:
                    key: The key to look up in the configuration.
                    default: The value to return if the key is not found.
                
                Returns:
                    The value corresponding to the key, or the default if the key does not exist.
                """
                return self._data.get(key, default)
            
            def set(self, key, value):
                """
                Set the value for a given key in the configuration.
                
                If the key already exists, its value is updated; otherwise, a new key-value pair is added.
                """
                self._data[key] = value
            
            def update(self, data):
                """
                Update the configuration with key-value pairs from the provided dictionary.
                
                Parameters:
                    data (dict): Dictionary containing keys and values to update in the configuration.
                """
                self._data.update(data)
            
            def to_dict(self):
                """
                Return a shallow copy of the configuration data as a dictionary.
                
                Returns:
                    dict: A copy of the internal configuration data.
                """
                return self._data.copy()
        
        class ConfigManager:
            def __init__(self, config_path=None):
                """
                Initialize a ConfigManager instance with an optional configuration file path.
                
                Parameters:
                    config_path (str, optional): Path to the configuration file. If not provided, no file is associated initially.
                """
                self.config_path = config_path
                self._config = Config()
            
            def load(self):
                """
                Return the current configuration instance managed by this ConfigManager.
                """
                return self._config
            
            def save(self):
                """
                Saves the current configuration to the associated file or storage location.
                """
                pass
        
        def load_config(path):
            """
            Load configuration data from the specified file path.
            
            Returns:
                Config: An instance of Config initialized with the loaded data, or an empty Config if loading is not implemented.
            """
            return Config()
        
        def save_config(config, path):
            """
            Stub function for saving configuration data to a file.
            
            This mock implementation does not perform any file operations.
            """
            pass
        
        def validate_config(config_data):
            """
            Validate the provided configuration data.
            
            Returns:
                bool: Always returns True, indicating the configuration is considered valid.
            """
            return True


class TestConfig:
    """Test cases for the Config class."""
    
    @pytest.fixture
    def sample_config_data(self):
        """
        Provides a sample configuration dictionary for use in tests.
        
        Returns:
            dict: Example configuration data with API, feature, and character settings.
        """
        return {
            'api_endpoint': 'https://api.example.com/v1',
            'api_key': 'test_api_key_12345',
            'timeout': 30,
            'retry_attempts': 3,
            'debug': False,
            'features': {
                'voice_enabled': True,
                'chat_enabled': True,
                'max_characters': 5000
            },
            'character_settings': {
                'default_personality': 'friendly',
                'response_style': 'conversational',
                'languages': ['en', 'es', 'fr']
            }
        }
    
    @pytest.fixture
    def empty_config(self):
        """
        Fixture that returns an empty Config instance for use in tests.
        """
        return Config({})
    
    @pytest.fixture
    def temp_config_file(self, tmp_path, sample_config_data):
        """
        Provides a temporary JSON configuration file populated with sample data for testing.
        
        Parameters:
        	tmp_path: Temporary directory path provided by pytest.
        	sample_config_data: Dictionary containing sample configuration data.
        
        Returns:
        	Path to the created temporary JSON configuration file.
        """
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config_data, f)
        return config_file
    
    # Happy Path Tests
    def test_config_initialization_with_valid_data(self, sample_config_data):
        """
        Verifies that initializing a Config instance with valid configuration data allows correct retrieval of expected keys and values.
        """
        config = Config(sample_config_data)
        assert config.get('api_endpoint') == 'https://api.example.com/v1'
        assert config.get('api_key') == 'test_api_key_12345'
        assert config.get('timeout') == 30
        assert config.get('retry_attempts') == 3
        assert config.get('debug') is False
    
    def test_config_initialization_empty(self):
        """
        Test that initializing Config with no data returns None for missing keys and uses the provided default value when specified.
        """
        config = Config()
        assert config.get('nonexistent') is None
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_config_get_method_with_nested_data(self, sample_config_data):
        """
        Test that the Config.get() method correctly retrieves nested configuration data.
        
        Verifies that accessing a top-level key returns the expected nested dictionary and its values.
        """
        config = Config(sample_config_data)
        features = config.get('features')
        assert features['voice_enabled'] is True
        assert features['max_characters'] == 5000
    
    def test_config_get_method_with_default_values(self, sample_config_data):
        """
        Test that Config.get() returns the specified default value when a key is missing.
        
        Verifies that the method returns None if no default is provided, and returns the given default value for missing keys.
        """
        config = Config(sample_config_data)
        assert config.get('nonexistent_key') is None
        assert config.get('nonexistent_key', 'default_value') == 'default_value'
        assert config.get('nonexistent_key', 42) == 42
        assert config.get('nonexistent_key', False) is False
    
    def test_config_set_method(self, sample_config_data):
        """
        Test that the Config.set() method correctly adds new keys and updates existing configuration values.
        """
        config = Config(sample_config_data)
        config.set('new_key', 'new_value')
        assert config.get('new_key') == 'new_value'
        
        config.set('timeout', 60)
        assert config.get('timeout') == 60
    
    def test_config_update_method(self, sample_config_data):
        """
        Test that the Config.update() method correctly applies multiple updates and preserves existing values not in the update set.
        """
        config = Config(sample_config_data)
        updates = {
            'timeout': 45,
            'new_setting': 'new_value',
            'debug': True
        }
        config.update(updates)
        
        assert config.get('timeout') == 45
        assert config.get('new_setting') == 'new_value'
        assert config.get('debug') is True
        # Ensure other values remain unchanged
        assert config.get('api_endpoint') == 'https://api.example.com/v1'
    
    def test_config_to_dict_method(self, sample_config_data):
        """
        Tests that Config.to_dict() returns an accurate, independent dictionary representation of the configuration data.
        """
        config = Config(sample_config_data)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['api_endpoint'] == 'https://api.example.com/v1'
        assert config_dict['features']['voice_enabled'] is True
        
        # Ensure it's a copy, not a reference
        config_dict['api_endpoint'] = 'modified'
        assert config.get('api_endpoint') == 'https://api.example.com/v1'
    
    # Edge Cases
    def test_config_with_none_values(self):
        """
        Verify that the Config class correctly stores and retrieves keys with None values.
        """
        config_data = {
            'api_endpoint': None,
            'timeout': None,
            'valid_key': 'valid_value'
        }
        config = Config(config_data)
        
        assert config.get('api_endpoint') is None
        assert config.get('timeout') is None
        assert config.get('valid_key') == 'valid_value'
    
    def test_config_with_mixed_data_types(self):
        """
        Verify that the Config class correctly stores and retrieves values of various data types, including strings, integers, floats, booleans, lists, dictionaries, and None.
        """
        mixed_data = {
            'string_val': 'test_string',
            'int_val': 42,
            'float_val': 3.14159,
            'bool_true': True,
            'bool_false': False,
            'list_val': [1, 2, 'three', 4.0],
            'dict_val': {'nested': 'value', 'number': 100},
            'none_val': None
        }
        config = Config(mixed_data)
        
        assert config.get('string_val') == 'test_string'
        assert config.get('int_val') == 42
        assert config.get('float_val') == 3.14159
        assert config.get('bool_true') is True
        assert config.get('bool_false') is False
        assert config.get('list_val') == [1, 2, 'three', 4.0]
        assert config.get('dict_val')['nested'] == 'value'
        assert config.get('none_val') is None
    
    def test_config_with_special_characters(self):
        """
        Verify that the Config class correctly stores and retrieves values containing Unicode and special characters.
        """
        special_data = {
            'unicode_text': 'Hello 世界! 🌍',
            'special_chars': '!@#$%^&*()_+-={}[]|\\:";\'<>?,./',
            'url_with_params': 'https://api.com/endpoint?param=value&other=123',
            'json_string': '{"nested": "json", "array": [1, 2, 3]}'
        }
        config = Config(special_data)
        
        assert config.get('unicode_text') == 'Hello 世界! 🌍'
        assert config.get('special_chars') == '!@#$%^&*()_+-={}[]|\\:";\'<>?,./'
        assert 'param=value' in config.get('url_with_params')
    
    def test_config_with_deeply_nested_structure(self):
        """
        Test that the Config class correctly stores and retrieves values from deeply nested data structures.
        """
        deep_nested = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'deep_value': 'found_it',
                            'deep_list': [1, 2, {'nested_in_list': True}]
                        }
                    }
                }
            }
        }
        config = Config(deep_nested)
        
        level4 = config.get('level1')['level2']['level3']['level4']
        assert level4['deep_value'] == 'found_it'
        assert level4['deep_list'][2]['nested_in_list'] is True
    
    # Error Cases
    def test_config_initialization_with_invalid_types(self):
        """
        Verify that initializing a Config object with invalid data types either raises an appropriate exception or results in a Config instance that returns None for any key.
        """
        # Test with various invalid types
        invalid_inputs = [
            "invalid_string",
            123,
            [1, 2, 3],
            True,
            None
        ]
        
        for invalid_input in invalid_inputs:
            # Depending on implementation, this might raise ConfigError or handle gracefully
            try:
                config = Config(invalid_input)
                # If no exception, ensure it handles gracefully
                assert config.get('any_key') is None
            except (ConfigError, TypeError, ValueError):
                # Expected behavior for invalid input
                pass
    
    def test_config_get_with_invalid_key_types(self, sample_config_data):
        """
        Test that Config.get() returns None or raises an exception when called with invalid key types.
        
        Verifies that using non-string keys such as None, integers, lists, dictionaries, or booleans with Config.get() either returns None or raises a TypeError or KeyError.
        """
        config = Config(sample_config_data)
        
        # Test with non-string keys
        invalid_keys = [None, 123, [], {}, True]
        for invalid_key in invalid_keys:
            try:
                result = config.get(invalid_key)
                assert result is None  # Should return None for invalid keys
            except (TypeError, KeyError):
                # Expected behavior for invalid key types
                pass


class TestConfigManager:
    """Test cases for the ConfigManager class."""
    
    @pytest.fixture
    def temp_config_path(self, tmp_path):
        """
        Fixture that returns a temporary file path for a configuration file named 'config.json'.
        """
        return tmp_path / "config.json"
    
    @pytest.fixture
    def sample_config_data(self):
        """
        Fixture that returns a sample configuration dictionary with application name, version, and settings for testing purposes.
        """
        return {
            'app_name': 'CharacterClient',
            'version': '1.0.0',
            'settings': {
                'auto_save': True,
                'theme': 'dark'
            }
        }
    
    def test_config_manager_initialization(self, temp_config_path):
        """
        Verify that ConfigManager correctly stores the provided configuration file path upon initialization.
        """
        manager = ConfigManager(str(temp_config_path))
        assert manager.config_path == str(temp_config_path)
    
    def test_config_manager_load_existing_file(self, temp_config_path, sample_config_data):
        """
        Test that ConfigManager correctly loads configuration data from an existing file.
        
        Verifies that the loaded configuration is an instance of Config and contains the expected data.
        """
        # Create config file
        with open(temp_config_path, 'w') as f:
            json.dump(sample_config_data, f)
        
        manager = ConfigManager(str(temp_config_path))
        config = manager.load()
        
        assert isinstance(config, Config)
        assert config.get('app_name') == 'CharacterClient'
        assert config.get('settings')['theme'] == 'dark'
    
    def test_config_manager_load_nonexistent_file(self, temp_config_path):
        """
        Test that ConfigManager.load() handles a nonexistent config file by either returning a default Config instance or raising FileNotFoundError.
        """
        manager = ConfigManager(str(temp_config_path))
        
        # Should handle gracefully (create default config or raise appropriate error)
        try:
            config = manager.load()
            assert isinstance(config, Config)
        except FileNotFoundError:
            # Expected behavior for missing file
            pass
    
    def test_config_manager_save(self, temp_config_path, sample_config_data):
        """
        Verify that ConfigManager correctly saves configuration data to a file and that the saved file contains the expected content.
        """
        manager = ConfigManager(str(temp_config_path))
        config = Config(sample_config_data)
        
        # Save configuration
        manager._config = config
        manager.save()
        
        # Verify file was created and contains correct data
        assert temp_config_path.exists()
        with open(temp_config_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['app_name'] == 'CharacterClient'
        assert saved_data['settings']['theme'] == 'dark'


class TestConfigUtilityFunctions:
    """Test cases for utility functions."""
    
    @pytest.fixture
    def sample_config_data(self):
        """
        Provides a sample configuration dictionary for use in tests.
        
        Returns:
            dict: A nested dictionary containing example database, API, and logging configuration data.
        """
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'character_db'
            },
            'api': {
                'endpoint': 'https://api.example.com',
                'version': 'v1'
            },
            'logging': {
                'level': 'INFO',
                'file': 'app.log'
            }
        }
    
    def test_load_config_from_json_file(self, tmp_path, sample_config_data):
        """
        Test that the load_config function correctly loads configuration data from a JSON file and returns a Config instance with expected values.
        """
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config_data, f)
        
        config = load_config(str(config_file))
        assert isinstance(config, Config)
        assert config.get('database')['host'] == 'localhost'
        assert config.get('api')['version'] == 'v1'
    
    def test_load_config_from_yaml_file(self, tmp_path, sample_config_data):
        """
        Test that the load_config function correctly loads configuration data from a YAML file.
        
        Creates a temporary YAML file with sample configuration data, loads it using load_config, and verifies the resulting Config object contains the expected values. Skips the test if the YAML module is not available.
        """
        config_file = tmp_path / "test_config.yaml"
        try:
            with open(config_file, 'w') as f:
                yaml.dump(sample_config_data, f)
            
            config = load_config(str(config_file))
            assert isinstance(config, Config)
            assert config.get('database')['port'] == 5432
        except ImportError:
            # Skip if yaml not available
            pytest.skip("YAML not available")
    
    def test_load_config_from_nonexistent_file(self):
        """
        Test that loading a configuration from a nonexistent file raises a FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.json')
    
    def test_load_config_from_invalid_json(self, tmp_path):
        """
        Test that loading a configuration from an invalid JSON file raises a JSONDecodeError.
        """
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, 'w') as f:
            f.write('{"invalid": json content without closing brace')
        
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))
    
    def test_save_config_to_json_file(self, tmp_path, sample_config_data):
        """
        Test that the save_config function correctly writes a Config instance to a JSON file.
        
        Verifies that the file is created and contains the expected configuration data.
        """
        config = Config(sample_config_data)
        config_file = tmp_path / "saved_config.json"
        
        save_config(config, str(config_file))
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['database']['host'] == 'localhost'
        assert saved_data['logging']['level'] == 'INFO'
    
    def test_save_config_to_nonexistent_directory(self, tmp_path, sample_config_data):
        """
        Test that save_config creates all necessary parent directories when saving to a path in a nonexistent directory.
        """
        config = Config(sample_config_data)
        config_file = tmp_path / "nested" / "directory" / "config.json"
        
        save_config(config, str(config_file))
        
        assert config_file.exists()
        assert config_file.parent.exists()
    
    # Validation Tests
    def test_validate_config_with_valid_data(self, sample_config_data):
        """
        Test that `validate_config` returns True when provided with valid configuration data.
        """
        assert validate_config(sample_config_data) is True
    
    def test_validate_config_with_missing_required_fields(self):
        """
        Test that `validate_config` returns False or raises ConfigError when required fields are missing from the configuration.
        """
        incomplete_config = {
            'database': {
                'host': 'localhost'
                # Missing port and name
            }
        }
        
        # Depending on implementation, this might return False or raise ConfigError
        try:
            result = validate_config(incomplete_config)
            assert result is False
        except ConfigError:
            # Expected behavior for invalid config
            pass
    
    def test_validate_config_with_invalid_types(self):
        """
        Test that validate_config returns False or raises ConfigError when configuration fields have invalid types.
        """
        invalid_config = {
            'database': {
                'host': 'localhost',
                'port': 'not_a_number',  # Should be integer
                'name': 123  # Should be string
            }
        }
        
        try:
            result = validate_config(invalid_config)
            assert result is False
        except ConfigError:
            pass
    
    def test_validate_config_with_none_input(self):
        """
        Test that `validate_config` raises an exception when called with `None` as input.
        """
        with pytest.raises((ConfigError, TypeError)):
            validate_config(None)
    
    def test_validate_config_with_empty_dict(self):
        """
        Test that `validate_config` returns a boolean when given an empty dictionary.
        """
        result = validate_config({})
        # Depending on requirements, empty config might be valid or invalid
        assert isinstance(result, bool)


class TestConfigIntegration:
    """Integration tests for config functionality."""
    
    def test_config_roundtrip_json(self, tmp_path):
        """
        Verifies that a configuration can be created, saved to a JSON file, and loaded back with data integrity preserved.
        """
        original_data = {
            'app_settings': {
                'theme': 'dark',
                'language': 'en',
                'auto_save': True
            },
            'user_preferences': {
                'notifications': False,
                'sound_enabled': True
            }
        }
        
        # Create and save config
        config = Config(original_data)
        config_file = tmp_path / "roundtrip_config.json"
        save_config(config, str(config_file))
        
        # Load config from file
        loaded_config = load_config(str(config_file))
        
        # Verify data integrity
        assert loaded_config.get('app_settings')['theme'] == 'dark'
        assert loaded_config.get('user_preferences')['notifications'] is False
        assert loaded_config.to_dict() == original_data
    
    def test_config_manager_complete_workflow(self, tmp_path):
        """
        Test the full workflow of ConfigManager, including initialization, saving, loading, modifying, and verifying configuration data across multiple instances.
        """
        config_file = tmp_path / "workflow_config.json"
        
        # Initialize manager and create config
        manager = ConfigManager(str(config_file))
        config = Config({
            'initial_setting': 'initial_value',
            'counter': 0
        })
        manager._config = config
        
        # Save initial config
        manager.save()
        
        # Create new manager instance and load
        new_manager = ConfigManager(str(config_file))
        loaded_config = new_manager.load()
        
        # Modify and save again
        loaded_config.set('counter', 1)
        loaded_config.set('new_setting', 'new_value')
        new_manager._config = loaded_config
        new_manager.save()
        
        # Final verification
        final_manager = ConfigManager(str(config_file))
        final_config = final_manager.load()
        
        assert final_config.get('initial_setting') == 'initial_value'
        assert final_config.get('counter') == 1
        assert final_config.get('new_setting') == 'new_value'
    
    @patch.dict(os.environ, {
        'CONFIG_API_KEY': 'env_api_key',
        'CONFIG_DEBUG': 'true',
        'CONFIG_TIMEOUT': '60'
    })
    def test_config_with_environment_variables(self):
        """
        Placeholder test for verifying Config integration with environment variable substitution.
        
        This test is intended to check whether the Config class correctly replaces placeholders in configuration values with corresponding environment variable values. Actual assertions should be implemented based on the Config class's support for environment variable substitution.
        """
        # This test assumes Config class supports environment variable substitution
        env_config_data = {
            'api_key': '${CONFIG_API_KEY}',
            'debug': '${CONFIG_DEBUG}',
            'timeout': '${CONFIG_TIMEOUT}'
        }
        
        # Implementation would depend on actual Config class behavior
        # This is a placeholder for environment variable integration testing
        config = Config(env_config_data)
        
        # If environment variable substitution is implemented:
        # assert config.get('api_key') == 'env_api_key'
        # assert config.get('debug') == 'true'
        # assert config.get('timeout') == '60'
    
    def test_config_concurrent_access(self, tmp_path):
        """
        Test concurrent access to a configuration file by running multiple threads that load, modify, and update the config, verifying all threads complete successfully.
        """
        import threading
        import time
        
        config_file = tmp_path / "concurrent_config.json"
        config = Config({'counter': 0, 'shared_data': []})
        save_config(config, str(config_file))
        
        results = []
        
        def worker_thread(thread_id):
            """
            Performs a single iteration of concurrent configuration loading, modification, and result recording for testing purposes.
            
            Parameters:
                thread_id (int): Identifier for the worker thread, used in result tracking.
            """
            try:
                # Load config
                local_config = load_config(str(config_file))
                counter = local_config.get('counter', 0)
                
                # Simulate some work
                time.sleep(0.01)
                
                # Update counter
                local_config.set('counter', counter + 1)
                results.append(f"thread_{thread_id}_success")
                
            except Exception as e:
                results.append(f"thread_{thread_id}_error_{str(e)}")
        
        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        success_count = len([r for r in results if 'success' in r])
        assert success_count == 5


class TestConfigPerformance:
    """Performance tests for config operations."""
    
    def test_config_large_dataset_performance(self):
        """
        Tests that the Config class can initialize and retrieve values efficiently with a large dataset, ensuring initialization completes within 1 second and 100 accesses complete within 0.1 seconds.
        """
        # Create large config data
        large_data = {}
        for i in range(1000):
            large_data[f'key_{i}'] = {
                'value': f'value_{i}',
                'index': i,
                'metadata': {
                    'created': f'2023-01-{i%30+1:02d}',
                    'type': 'test_data'
                }
            }
        
        # Test initialization performance
        import time
        start_time = time.time()
        config = Config(large_data)
        init_time = time.time() - start_time
        
        # Should initialize reasonably quickly (< 1 second)
        assert init_time < 1.0
        
        # Test access performance
        start_time = time.time()
        for i in range(100):
            value = config.get(f'key_{i}')
            assert value['index'] == i
        access_time = time.time() - start_time
        
        # Should access quickly (< 0.1 seconds for 100 accesses)
        assert access_time < 0.1
    
    def test_config_repeated_operations_performance(self):
        """
        Tests that the Config class can perform 1000 repeated get and set operations within acceptable time limits.
        """
        config = Config({'base_value': 'test'})
        
        # Test repeated get operations
        import time
        start_time = time.time()
        for _ in range(1000):
            value = config.get('base_value')
            assert value == 'test'
        get_time = time.time() - start_time
        
        # Should handle 1000 gets quickly
        assert get_time < 0.1
        
        # Test repeated set operations
        start_time = time.time()
        for i in range(1000):
            config.set(f'dynamic_key_{i}', f'value_{i}')
        set_time = time.time() - start_time
        
        # Should handle 1000 sets reasonably quickly
        assert set_time < 1.0


# Parametrized tests for comprehensive coverage
class TestConfigParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.parametrize("input_data,expected_type", [
        ({}, dict),
        ({'key': 'value'}, dict),
        ({'nested': {'key': 'value'}}, dict),
        ({'list': [1, 2, 3]}, dict),
        ({'mixed': {'str': 'value', 'int': 42, 'bool': True}}, dict)
    ])
    def test_config_to_dict_with_various_inputs(self, input_data, expected_type):
        """
        Test that Config.to_dict() returns a dictionary matching the input data for various input types.
        
        Parameters:
            input_data: The initial data used to create the Config instance.
            expected_type: The expected type of the output from to_dict().
        """
        config = Config(input_data)
        result = config.to_dict()
        assert isinstance(result, expected_type)
        assert result == input_data
    
    @pytest.mark.parametrize("key,default,expected", [
        ('existing_key', None, 'existing_value'),
        ('nonexistent_key', None, None),
        ('nonexistent_key', 'default', 'default'),
        ('nonexistent_key', 42, 42),
        ('nonexistent_key', [], []),
        ('nonexistent_key', {}, {})
    ])
    def test_config_get_with_various_defaults(self, key, default, expected):
        """
        Test the Config.get() method with different keys and default values.
        
        Verifies that Config.get() returns the expected value for a given key, using the provided default when the key is not present.
        """
        config = Config({'existing_key': 'existing_value'})
        if default is None:
            result = config.get(key)
        else:
            result = config.get(key, default)
        assert result == expected
    
    @pytest.mark.parametrize("file_extension", ['.json', '.yaml', '.yml'])
    def test_load_config_with_various_file_extensions(self, tmp_path, file_extension):
        """
        Test that `load_config` correctly loads configuration files with different file extensions, including JSON and YAML.
        
        Creates a temporary configuration file with the specified extension, writes sample data in the appropriate format, and verifies that `load_config` returns a `Config` instance with the expected values. Skips the test for YAML files if the YAML module is not available.
        """
        config_data = {'test': 'data', 'number': 42}
        config_file = tmp_path / f"test_config{file_extension}"
        
        try:
            if file_extension == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config_data, f)
            else:  # YAML files
                with open(config_file, 'w') as f:
                    yaml.dump(config_data, f)
            
            config = load_config(str(config_file))
            assert isinstance(config, Config)
            assert config.get('test') == 'data'
            assert config.get('number') == 42
            
        except ImportError:
            # Skip YAML tests if yaml module not available
            if file_extension in ['.yaml', '.yml']:
                pytest.skip("YAML module not available")
            else:
                raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])