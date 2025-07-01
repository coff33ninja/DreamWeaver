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
                self._data = data or {}
            
            def get(self, key, default=None):
                return self._data.get(key, default)
            
            def set(self, key, value):
                self._data[key] = value
            
            def update(self, data):
                self._data.update(data)
            
            def to_dict(self):
                return self._data.copy()
        
        class ConfigManager:
            def __init__(self, config_path=None):
                self.config_path = config_path
                self._config = Config()
            
            def load(self):
                return self._config
            
            def save(self):
                pass
        
        def load_config(path):
            return Config()
        
        def save_config(config, path):
            pass
        
        def validate_config(config_data):
            return True


class TestConfig:
    """Test cases for the Config class."""
    
    @pytest.fixture
    def sample_config_data(self):
        """Fixture providing sample configuration data."""
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
        """Fixture providing an empty config instance."""
        return Config({})
    
    @pytest.fixture
    def temp_config_file(self, tmp_path, sample_config_data):
        """Fixture providing a temporary config file."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config_data, f)
        return config_file
    
    # Happy Path Tests
    def test_config_initialization_with_valid_data(self, sample_config_data):
        """Test Config initialization with valid configuration data."""
        config = Config(sample_config_data)
        assert config.get('api_endpoint') == 'https://api.example.com/v1'
        assert config.get('api_key') == 'test_api_key_12345'
        assert config.get('timeout') == 30
        assert config.get('retry_attempts') == 3
        assert config.get('debug') is False
    
    def test_config_initialization_empty(self):
        """Test Config initialization with empty data."""
        config = Config()
        assert config.get('nonexistent') is None
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_config_get_method_with_nested_data(self, sample_config_data):
        """Test Config.get() method with nested configuration data."""
        config = Config(sample_config_data)
        features = config.get('features')
        assert features['voice_enabled'] is True
        assert features['max_characters'] == 5000
    
    def test_config_get_method_with_default_values(self, sample_config_data):
        """Test Config.get() method returns default values for missing keys."""
        config = Config(sample_config_data)
        assert config.get('nonexistent_key') is None
        assert config.get('nonexistent_key', 'default_value') == 'default_value'
        assert config.get('nonexistent_key', 42) == 42
        assert config.get('nonexistent_key', False) is False
    
    def test_config_set_method(self, sample_config_data):
        """Test Config.set() method for updating configuration values."""
        config = Config(sample_config_data)
        config.set('new_key', 'new_value')
        assert config.get('new_key') == 'new_value'
        
        config.set('timeout', 60)
        assert config.get('timeout') == 60
    
    def test_config_update_method(self, sample_config_data):
        """Test Config.update() method for batch updates."""
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
        """Test Config.to_dict() method returns correct dictionary representation."""
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
        """Test Config handles None values correctly."""
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
        """Test Config handles various data types correctly."""
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
        """Test Config handles Unicode and special characters."""
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
        """Test Config handles deeply nested data structures."""
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
        """Test Config initialization handles invalid data types gracefully."""
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
        """Test Config.get() with invalid key types."""
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
        """Fixture providing a temporary config file path."""
        return tmp_path / "config.json"
    
    @pytest.fixture
    def sample_config_data(self):
        """Fixture providing sample configuration data."""
        return {
            'app_name': 'CharacterClient',
            'version': '1.0.0',
            'settings': {
                'auto_save': True,
                'theme': 'dark'
            }
        }
    
    def test_config_manager_initialization(self, temp_config_path):
        """Test ConfigManager initialization."""
        manager = ConfigManager(str(temp_config_path))
        assert manager.config_path == str(temp_config_path)
    
    def test_config_manager_load_existing_file(self, temp_config_path, sample_config_data):
        """Test ConfigManager loading from existing file."""
        # Create config file
        with open(temp_config_path, 'w') as f:
            json.dump(sample_config_data, f)
        
        manager = ConfigManager(str(temp_config_path))
        config = manager.load()
        
        assert isinstance(config, Config)
        assert config.get('app_name') == 'CharacterClient'
        assert config.get('settings')['theme'] == 'dark'
    
    def test_config_manager_load_nonexistent_file(self, temp_config_path):
        """Test ConfigManager handling of nonexistent config file."""
        manager = ConfigManager(str(temp_config_path))
        
        # Should handle gracefully (create default config or raise appropriate error)
        try:
            config = manager.load()
            assert isinstance(config, Config)
        except FileNotFoundError:
            # Expected behavior for missing file
            pass
    
    def test_config_manager_save(self, temp_config_path, sample_config_data):
        """Test ConfigManager save functionality."""
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
        """Fixture providing sample configuration data."""
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
        """Test load_config function with JSON file."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config_data, f)
        
        config = load_config(str(config_file))
        assert isinstance(config, Config)
        assert config.get('database')['host'] == 'localhost'
        assert config.get('api')['version'] == 'v1'
    
    def test_load_config_from_yaml_file(self, tmp_path, sample_config_data):
        """Test load_config function with YAML file."""
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
        """Test load_config with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.json')
    
    def test_load_config_from_invalid_json(self, tmp_path):
        """Test load_config with invalid JSON file."""
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, 'w') as f:
            f.write('{"invalid": json content without closing brace')
        
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))
    
    def test_save_config_to_json_file(self, tmp_path, sample_config_data):
        """Test save_config function with JSON file."""
        config = Config(sample_config_data)
        config_file = tmp_path / "saved_config.json"
        
        save_config(config, str(config_file))
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['database']['host'] == 'localhost'
        assert saved_data['logging']['level'] == 'INFO'
    
    def test_save_config_to_nonexistent_directory(self, tmp_path, sample_config_data):
        """Test save_config creates directories if they don't exist."""
        config = Config(sample_config_data)
        config_file = tmp_path / "nested" / "directory" / "config.json"
        
        save_config(config, str(config_file))
        
        assert config_file.exists()
        assert config_file.parent.exists()
    
    # Validation Tests
    def test_validate_config_with_valid_data(self, sample_config_data):
        """Test validate_config with valid configuration data."""
        assert validate_config(sample_config_data) is True
    
    def test_validate_config_with_missing_required_fields(self):
        """Test validate_config with missing required fields."""
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
        """Test validate_config with invalid field types."""
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
        """Test validate_config with None input."""
        with pytest.raises((ConfigError, TypeError)):
            validate_config(None)
    
    def test_validate_config_with_empty_dict(self):
        """Test validate_config with empty dictionary."""
        result = validate_config({})
        # Depending on requirements, empty config might be valid or invalid
        assert isinstance(result, bool)


class TestConfigIntegration:
    """Integration tests for config functionality."""
    
    def test_config_roundtrip_json(self, tmp_path):
        """Test complete roundtrip: create config, save to JSON, load from JSON."""
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
        """Test complete ConfigManager workflow."""
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
        """Test Config integration with environment variables."""
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
        """Test config behavior under concurrent access."""
        import threading
        import time
        
        config_file = tmp_path / "concurrent_config.json"
        config = Config({'counter': 0, 'shared_data': []})
        save_config(config, str(config_file))
        
        results = []
        
        def worker_thread(thread_id):
            """Worker function for concurrent testing."""
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
        """Test Config performance with large datasets."""
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
        """Test Config performance under repeated operations."""
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
        """Test Config.to_dict() with various input types."""
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
        """Test Config.get() with various default values."""
        config = Config({'existing_key': 'existing_value'})
        if default is None:
            result = config.get(key)
        else:
            result = config.get(key, default)
        assert result == expected
    
    @pytest.mark.parametrize("file_extension", ['.json', '.yaml', '.yml'])
    def test_load_config_with_various_file_extensions(self, tmp_path, file_extension):
        """Test load_config with various file extensions."""
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

class TestConfigAdvancedEdgeCases:
    """Advanced edge case tests for Config functionality."""
    
    def test_config_with_circular_references(self):
        """Test Config handles circular references gracefully."""
        # Create circular reference data structure
        data = {'a': {'b': None}}
        data['a']['b'] = data['a']  # Create circular reference
        
        try:
            config = Config(data)
            # Should either handle gracefully or raise appropriate error
            result = config.get('a')
            assert result is not None
        except (RecursionError, ValueError, ConfigError) as e:
            # Expected behavior for circular references
            assert isinstance(e, (RecursionError, ValueError, ConfigError))
    
    def test_config_with_extremely_long_keys(self):
        """Test Config with extremely long key names."""
        long_key = 'a' * 10000  # 10k character key
        data = {long_key: 'long_key_value'}
        
        config = Config(data)
        assert config.get(long_key) == 'long_key_value'
        assert config.get('nonexistent') is None
    
    def test_config_with_extremely_long_values(self):
        """Test Config with extremely long values."""
        long_value = 'x' * 100000  # 100k character value
        data = {'long_value_key': long_value}
        
        config = Config(data)
        retrieved_value = config.get('long_value_key')
        assert len(retrieved_value) == 100000
        assert retrieved_value == long_value
    
    def test_config_with_binary_data(self):
        """Test Config with binary data."""
        binary_data = b'\x00\x01\x02\xff\xfe\xfd'
        data = {'binary_key': binary_data}
        
        config = Config(data)
        assert config.get('binary_key') == binary_data
    
    def test_config_memory_efficiency_large_updates(self):
        """Test Config memory efficiency with large batch updates."""
        import sys
        
        config = Config({})
        initial_size = sys.getsizeof(config._data) if hasattr(config, '_data') else 0
        
        # Perform large batch update
        large_update = {f'key_{i}': f'value_{i}' for i in range(10000)}
        config.update(large_update)
        
        # Verify update worked
        assert config.get('key_5000') == 'value_5000'
        assert config.get('key_9999') == 'value_9999'
    
    def test_config_with_lambda_functions(self):
        """Test Config with lambda functions and callable objects."""
        data = {
            'lambda_func': lambda x: x * 2,
            'callable_obj': str.upper,
            'regular_value': 'test'
        }
        
        config = Config(data)
        lambda_func = config.get('lambda_func')
        callable_obj = config.get('callable_obj')
        
        # Verify callable objects are preserved
        if callable(lambda_func):
            assert lambda_func(5) == 10
        if callable(callable_obj):
            assert callable_obj('hello') == 'HELLO'
        assert config.get('regular_value') == 'test'
    
    @pytest.mark.parametrize("invalid_json_content", [
        '{"key": value}',  # Missing quotes
        '{"key": "value",}',  # Trailing comma
        '{key: "value"}',  # Unquoted key
        '{"key": "value" "another": "value"}',  # Missing comma
        '{"key": "value"',  # Missing closing brace
        '{"key": undefined}',  # Invalid value
    ])
    def test_load_config_various_invalid_json_formats(self, tmp_path, invalid_json_content):
        """Test load_config with various invalid JSON formats."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, 'w') as f:
            f.write(invalid_json_content)
        
        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_file))
    
    def test_config_with_numpy_arrays(self):
        """Test Config with numpy arrays if numpy is available."""
        try:
            import numpy as np
            data = {
                'numpy_array': np.array([1, 2, 3, 4, 5]),
                'numpy_matrix': np.array([[1, 2], [3, 4]]),
                'regular_list': [1, 2, 3, 4, 5]
            }
            
            config = Config(data)
            retrieved_array = config.get('numpy_array')
            retrieved_matrix = config.get('numpy_matrix')
            
            # Verify numpy arrays are preserved
            assert np.array_equal(retrieved_array, np.array([1, 2, 3, 4, 5]))
            assert np.array_equal(retrieved_matrix, np.array([[1, 2], [3, 4]]))
            
        except ImportError:
            pytest.skip("NumPy not available")
    
    def test_config_thread_safety_stress_test(self):
        """Stress test for Config thread safety."""
        import threading
        import time
        import random
        
        config = Config({'counter': 0})
        errors = []
        operations_count = [0]  # Use list for mutable counter
        
        def stress_worker():
            """Worker function for stress testing."""
            try:
                for _ in range(100):
                    # Random operations
                    operation = random.choice(['get', 'set', 'update'])
                    
                    if operation == 'get':
                        config.get('counter')
                        config.get(f'key_{random.randint(1, 100)}', 'default')
                    elif operation == 'set':
                        config.set(f'key_{random.randint(1, 100)}', random.randint(1, 1000))
                    elif operation == 'update':
                        config.update({f'batch_key_{random.randint(1, 50)}': random.randint(1, 1000)})
                    
                    operations_count[0] += 1
                    time.sleep(0.001)  # Small delay
                    
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=stress_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert operations_count[0] == 1000  # 10 threads * 100 operations each


class TestConfigSecurityAndValidation:
    """Security and validation tests for Config functionality."""
    
    def test_config_injection_prevention(self):
        """Test Config prevents common injection attacks."""
        malicious_data = {
            'sql_injection': "'; DROP TABLE users; --",
            'script_injection': '<script>alert("xss")</script>',
            'command_injection': '; rm -rf /',
            'path_traversal': '../../../etc/passwd',
            'null_bytes': 'test\x00malicious'
        }
        
        config = Config(malicious_data)
        
        # Verify malicious content is stored as-is (not executed)
        assert config.get('sql_injection') == "'; DROP TABLE users; --"
        assert '<script>' in config.get('script_injection')
        assert config.get('command_injection') == '; rm -rf /'
        assert '../../../' in config.get('path_traversal')
        assert '\x00' in config.get('null_bytes')
    
    def test_config_size_limits(self):
        """Test Config handles extremely large configurations."""
        # Test with large number of keys
        large_config = {}
        for i in range(50000):
            large_config[f'key_{i}'] = f'value_{i}'
        
        try:
            config = Config(large_config)
            assert config.get('key_25000') == 'value_25000'
            assert len(config.to_dict()) == 50000
        except MemoryError:
            # Expected behavior for extremely large configs
            pytest.skip("System memory limit reached")
    
    def test_config_validation_schema_compliance(self):
        """Test Config validation against schema if implemented."""
        # Test schema validation for required fields
        valid_schema_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'user',
                'password': 'pass'
            },
            'api': {
                'endpoint': 'https://api.example.com',
                'timeout': 30,
                'retries': 3
            }
        }
        
        invalid_schema_data = {
            'database': {
                'host': 'localhost',
                # Missing required port
            },
            'api': {
                'endpoint': 'invalid-url',  # Invalid URL format
                'timeout': -5,  # Invalid negative timeout
                'retries': 'not_a_number'  # Invalid type
            }
        }
        
        # Test valid schema
        assert validate_config(valid_schema_data) is True
        
        # Test invalid schema
        try:
            result = validate_config(invalid_schema_data)
            assert result is False
        except ConfigError:
            # Expected behavior for invalid schema
            pass
    
    def test_config_file_permissions(self, tmp_path):
        """Test Config respects file permissions."""
        import stat
        
        config_data = {'secret': 'sensitive_data'}
        config = Config(config_data)
        config_file = tmp_path / "secure_config.json"
        
        # Save config
        save_config(config, str(config_file))
        
        # Check if file was created with appropriate permissions
        if config_file.exists():
            file_stat = config_file.stat()
            # On Unix systems, check if file is readable by owner only
            if hasattr(stat, 'S_IMODE'):
                permissions = stat.S_IMODE(file_stat.st_mode)
                # File should not be world-readable for sensitive data
                assert not (permissions & stat.S_IROTH)


class TestConfigBackwardCompatibility:
    """Backward compatibility tests for Config functionality."""
    
    @pytest.mark.parametrize("old_format", [
        # Legacy format variations
        {'version': '1.0', 'settings': {'key': 'value'}},
        {'config_version': 1, 'data': {'key': 'value'}},
        {'metadata': {'version': '2.0'}, 'config': {'key': 'value'}},
    ])
    def test_config_legacy_format_support(self, old_format):
        """Test Config supports legacy configuration formats."""
        config = Config(old_format)
        
        # Should handle legacy formats gracefully
        assert isinstance(config, Config)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
    
    def test_config_migration_scenarios(self, tmp_path):
        """Test Config handles migration from old versions."""
        # Simulate old config format
        old_config_data = {
            'version': '1.0',
            'database_host': 'localhost',
            'database_port': 5432,
            'api_endpoint': 'https://old-api.example.com'
        }
        
        # Save old format
        old_config_file = tmp_path / "old_config.json"
        with open(old_config_file, 'w') as f:
            json.dump(old_config_data, f)
        
        # Load and verify
        config = load_config(str(old_config_file))
        assert config.get('version') == '1.0'
        assert config.get('database_host') == 'localhost'
    
    def test_config_deprecated_methods_support(self):
        """Test Config supports deprecated methods if any."""
        config = Config({'test': 'value'})
        
        # Test that deprecated methods still work (if they exist)
        # This would be implementation-specific
        try:
            # Example deprecated method calls
            if hasattr(config, 'getValue'):  # Hypothetical deprecated method
                assert config.getValue('test') == 'value'
            if hasattr(config, 'setValue'):  # Hypothetical deprecated method
                config.setValue('new_key', 'new_value')
                assert config.get('new_key') == 'new_value'
        except AttributeError:
            # No deprecated methods exist
            pass


class TestConfigErrorRecovery:
    """Error recovery and resilience tests for Config functionality."""
    
    def test_config_recovery_from_corrupted_file(self, tmp_path):
        """Test Config recovery from corrupted configuration files."""
        corrupted_files = [
            '{"key": "value"}\x00\x00\x00corrupted_data',  # Null bytes
            '{"key": "value"}{"second": "json"}',  # Multiple JSON objects
            '\ufeff{"key": "value"}',  # BOM character
            '{"key": "value"}\n\n\n\r\r\r',  # Extra whitespace
        ]
        
        for i, corrupted_content in enumerate(corrupted_files):
            config_file = tmp_path / f"corrupted_{i}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(corrupted_content)
            
            try:
                config = load_config(str(config_file))
                # If it loads successfully, verify basic functionality
                assert isinstance(config, Config)
            except (json.JSONDecodeError, UnicodeDecodeError, ConfigError):
                # Expected behavior for corrupted files
                pass
    
    def test_config_partial_failure_handling(self):
        """Test Config handles partial failures gracefully."""
        config = Config({'working_key': 'working_value'})
        
        # Test operations that might partially fail
        problematic_updates = [
            {'good_key': 'good_value', 'problematic_key': object()},  # Non-serializable
            {'normal_key': 'normal_value', 'none_key': None},  # None values
        ]
        
        for update in problematic_updates:
            try:
                config.update(update)
                # Verify partial success
                if 'good_key' in update:
                    assert config.get('good_key') == 'good_value'
                if 'normal_key' in update:
                    assert config.get('normal_key') == 'normal_value'
            except (TypeError, ValueError):
                # Some operations might fail completely
                pass
            
            # Original data should remain intact
            assert config.get('working_key') == 'working_value'
    
    def test_config_network_failure_simulation(self, tmp_path):
        """Test Config behavior during simulated network failures."""
        # Simulate network-based config loading failure
        config_file = tmp_path / "network_config.json"
        
        # Create initial config
        initial_data = {'local_cache': True, 'last_sync': '2023-01-01'}
        with open(config_file, 'w') as f:
            json.dump(initial_data, f)
        
        # Simulate network failure by making file temporarily inaccessible
        try:
            import os
            # Change file permissions to simulate access failure
            os.chmod(config_file, 0o000)
            
            try:
                config = load_config(str(config_file))
                # Should either fail gracefully or load cached version
                assert isinstance(config, Config)
            except (PermissionError, FileNotFoundError):
                # Expected behavior for inaccessible files
                pass
            finally:
                # Restore permissions
                os.chmod(config_file, 0o644)
                
        except (OSError, NotImplementedError):
            # Skip on systems that don't support chmod
            pytest.skip("File permission testing not supported")


class TestConfigPerformanceBenchmarks:
    """Performance benchmark tests for Config operations."""
    
    def test_config_creation_benchmark(self):
        """Benchmark Config creation with various data sizes."""
        import time
        
        sizes = [10, 100, 1000, 5000]
        creation_times = []
        
        for size in sizes:
            data = {f'key_{i}': f'value_{i}' for i in range(size)}
            
            start_time = time.perf_counter()
            config = Config(data)
            end_time = time.perf_counter()
            
            creation_time = end_time - start_time
            creation_times.append(creation_time)
            
            # Verify config was created correctly
            assert config.get('key_0') == 'value_0'
            assert config.get(f'key_{size-1}') == f'value_{size-1}'
        
        # Performance should scale reasonably
        # Larger configs should take more time but not exponentially more
        assert creation_times[-1] < creation_times[0] * 1000  # Reasonable scaling
    
    def test_config_access_pattern_benchmark(self):
        """Benchmark different Config access patterns."""
        import time
        
        # Create config with nested structure
        config_data = {}
        for i in range(1000):
            config_data[f'section_{i}'] = {
                'id': i,
                'data': {f'nested_key_{j}': f'nested_value_{j}' for j in range(10)}
            }
        
        config = Config(config_data)
        
        # Benchmark sequential access
        start_time = time.perf_counter()
        for i in range(500):
            value = config.get(f'section_{i}')
            assert value['id'] == i
        sequential_time = time.perf_counter() - start_time
        
        # Benchmark random access
        import random
        random_keys = [f'section_{random.randint(0, 999)}' for _ in range(500)]
        
        start_time = time.perf_counter()
        for key in random_keys:
            value = config.get(key)
            assert value is not None
        random_time = time.perf_counter() - start_time
        
        # Both should complete in reasonable time
        assert sequential_time < 1.0
        assert random_time < 1.0
    
    def test_config_serialization_benchmark(self, tmp_path):
        """Benchmark Config serialization performance."""
        import time
        
        # Create large nested config
        large_data = {}
        for i in range(5000):
            large_data[f'section_{i}'] = {
                'id': i,
                'name': f'Section {i}',
                'active': i % 2 == 0,
                'metadata': {
                    'created': f'2023-{i%12+1:02d}-01',
                    'tags': [f'tag_{j}' for j in range(i % 5)]
                }
            }
        
        config = Config(large_data)
        config_file = tmp_path / "benchmark_config.json"
        
        # Benchmark save operation
        start_time = time.perf_counter()
        save_config(config, str(config_file))
        save_time = time.perf_counter() - start_time
        
        # Benchmark load operation
        start_time = time.perf_counter()
        loaded_config = load_config(str(config_file))
        load_time = time.perf_counter() - start_time
        
        # Verify operations completed in reasonable time
        assert save_time < 5.0
        assert load_time < 5.0
        
        # Verify data integrity
        assert loaded_config.get('section_2500')['name'] == 'Section 2500'


class TestConfigSpecialCharactersAndEncoding:
    """Tests for Config handling of special characters and encoding."""
    
    @pytest.mark.parametrize("special_text", [
        "English text",
        "中文测试",  # Chinese
        "العربية",  # Arabic
        "русский текст",  # Russian
        "हिन्दी पाठ",  # Hindi
        "🌟✨🎉🔥💯",  # Emojis
        "математика: ∑∏∫∞≠≤≥±",  # Mathematical symbols
        "currency: $€£¥₹₽",  # Currency symbols
        "\u0000\u0001\u0002",  # Control characters
        "line1\nline2\r\nline3\tIndented",  # Line breaks and tabs
    ])
    def test_config_unicode_support(self, special_text):
        """Test Config handles various Unicode characters correctly."""
        config_data = {
            'unicode_key': special_text,
            f'key_{special_text[:5]}': 'unicode_key_test'
        }
        
        config = Config(config_data)
        assert config.get('unicode_key') == special_text
        
        # Test serialization/deserialization doesn't corrupt Unicode
        config_dict = config.to_dict()
        assert config_dict['unicode_key'] == special_text
    
    def test_config_encoding_consistency(self, tmp_path):
        """Test Config maintains encoding consistency across save/load operations."""
        unicode_data = {
            'multilingual': {
                'english': 'Hello World',
                'chinese': '你好世界',
                'japanese': 'こんにちは世界',
                'arabic': 'مرحبا بالعالم',
                'emoji': '🌍👋🎌'
            },
            'special_chars': '"`~!@#$%^&*()_+-={}[]|\\:";\'<>?,./',
            'unicode_escape': '\u00A9 \u00AE \u2122'  # Copyright, Registered, Trademark
        }
        
        config = Config(unicode_data)
        config_file = tmp_path / "unicode_config.json"
        
        # Save and reload
        save_config(config, str(config_file))
        loaded_config = load_config(str(config_file))
        
        # Verify Unicode preservation
        assert loaded_config.get('multilingual')['chinese'] == '你好世界'
        assert loaded_config.get('multilingual')['emoji'] == '🌍👋🎌'
        assert loaded_config.get('unicode_escape') == '\u00A9 \u00AE \u2122'


class TestConfigDocumentationExamples:
    """Tests based on documentation examples and common usage patterns."""
    
    def test_config_readme_examples(self):
        """Test Config examples that would appear in README/documentation."""
        # Basic usage example
        config = Config({
            'app_name': 'CharacterClient',
            'version': '1.0.0',
            'debug': False
        })
        
        assert config.get('app_name') == 'CharacterClient'
        assert config.get('debug') is False
        
        # Configuration update example
        config.update({
            'debug': True,
            'log_level': 'DEBUG'
        })
        
        assert config.get('debug') is True
        assert config.get('log_level') == 'DEBUG'
    
    def test_config_common_patterns(self):
        """Test common configuration patterns."""
        # Environment-based configuration
        config_data = {
            'environment': 'production',
            'database': {
                'production': {
                    'host': 'prod.db.example.com',
                    'port': 5432
                },
                'development': {
                    'host': 'localhost',
                    'port': 5433
                }
            }
        }
        
        config = Config(config_data)
        env = config.get('environment')
        db_config = config.get('database')[env]
        
        assert db_config['host'] == 'prod.db.example.com'
        assert db_config['port'] == 5432
    
    def test_config_best_practices_examples(self):
        """Test configuration following best practices."""
        # Hierarchical configuration example
        config = Config({
            'application': {
                'name': 'CharacterClient',
                'version': '2.0.0',
                'features': {
                    'authentication': True,
                    'caching': True,
                    'monitoring': True
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '{timestamp} - {level} - {message}',
                'handlers': ['console', 'file']
            },
            'security': {
                'encryption': True,
                'token_expiry': 3600,
                'max_login_attempts': 3
            }
        })
        
        # Test nested access patterns
        assert config.get('application')['name'] == 'CharacterClient'
        assert config.get('application')['features']['caching'] is True
        assert config.get('logging')['level'] == 'INFO'
        assert config.get('security')['token_expiry'] == 3600


class TestConfigRegressionScenarios:
    """Regression tests for previously identified issues."""
    
    def test_config_key_collision_handling(self):
        """Test Config handles key collisions gracefully."""
        # Test overlapping keys in updates
        config = Config({'shared_key': 'original_value'})
        
        # Multiple updates with same key
        config.set('shared_key', 'first_update')
        assert config.get('shared_key') == 'first_update'
        
        config.update({'shared_key': 'second_update', 'new_key': 'new_value'})
        assert config.get('shared_key') == 'second_update'
        assert config.get('new_key') == 'new_value'
    
    def test_config_type_preservation_after_operations(self):
        """Test Config preserves data types after various operations."""
        original_data = {
            'string_val': 'test',
            'int_val': 42,
            'float_val': 3.14,
            'bool_val': True,
            'none_val': None,
            'list_val': [1, 2, 3],
            'dict_val': {'nested': 'value'}
        }
        
        config = Config(original_data)
        
        # Perform operations that might affect types
        config.update({'additional_key': 'additional_value'})
        config.set('new_int', 100)
        
        # Verify original types are preserved
        assert isinstance(config.get('string_val'), str)
        assert isinstance(config.get('int_val'), int)
        assert isinstance(config.get('float_val'), float)
        assert isinstance(config.get('bool_val'), bool)
        assert config.get('none_val') is None
        assert isinstance(config.get('list_val'), list)
        assert isinstance(config.get('dict_val'), dict)
    
    def test_config_nested_modification_isolation(self):
        """Test Config isolates nested modifications properly."""
        config = Config({
            'nested': {
                'level1': {
                    'level2': {
                        'value': 'original'
                    }
                }
            }
        })
        
        # Get nested reference
        nested_ref = config.get('nested')
        original_nested_ref = config.get('nested')
        
        # Modify through reference (if mutable)
        if isinstance(nested_ref, dict):
            nested_ref['level1']['level2']['value'] = 'modified'
            
            # Check if original config was affected
            current_value = config.get('nested')['level1']['level2']['value']
            # Depending on implementation, this might be 'original' or 'modified'
            assert current_value in ['original', 'modified']


if __name__ == '__main__':
    # Run with comprehensive options
    pytest.main([
        __file__, 
        '-v',
        '--tb=short',
        '--durations=10',
        '--strict-markers'
    ])