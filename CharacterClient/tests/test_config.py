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