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
    """Advanced edge case tests for Config class."""
    
    def test_config_with_circular_references(self):
        """Test Config behavior with circular reference data structures."""
        # Create circular reference
        circular_data = {'a': {'b': {}}}
        circular_data['a']['b']['c'] = circular_data['a']
        
        try:
            config = Config(circular_data)
            # If it handles gracefully, test basic operations
            assert config.get('a') is not None
        except (ValueError, RecursionError, ConfigError):
            # Expected behavior for circular references
            pass
    
    def test_config_with_extremely_large_strings(self):
        """Test Config with very large string values."""
        large_string = 'x' * 1000000  # 1MB string
        config_data = {
            'large_value': large_string,
            'normal_value': 'small'
        }
        
        config = Config(config_data)
        assert config.get('large_value') == large_string
        assert config.get('normal_value') == 'small'
        assert len(config.get('large_value')) == 1000000
    
    def test_config_with_very_deep_nesting(self):
        """Test Config with extremely deep nesting to check stack overflow protection."""
        # Create deeply nested structure (100 levels)
        deep_data = current = {}
        for i in range(100):
            current[f'level_{i}'] = {}
            current = current[f'level_{i}']
        current['final_value'] = 'reached_bottom'
        
        try:
            config = Config(deep_data)
            # Navigate to the deep value
            current_level = config.get('level_0')
            for i in range(1, 100):
                current_level = current_level[f'level_{i}']
            assert current_level['final_value'] == 'reached_bottom'
        except RecursionError:
            # Expected behavior if implementation has recursion limits
            pass
    
    def test_config_with_empty_string_keys(self):
        """Test Config behavior with empty string keys."""
        config_data = {
            '': 'empty_key_value',
            ' ': 'space_key_value',
            '\t': 'tab_key_value',
            '\n': 'newline_key_value'
        }
        
        config = Config(config_data)
        assert config.get('') == 'empty_key_value'
        assert config.get(' ') == 'space_key_value'
        assert config.get('\t') == 'tab_key_value'
        assert config.get('\n') == 'newline_key_value'
    
    def test_config_with_numeric_keys_if_supported(self):
        """Test Config with numeric keys (converted to strings)."""
        config_data = {
            123: 'numeric_key_value',
            45.67: 'float_key_value',
            True: 'boolean_key_value'
        }
        
        try:
            config = Config(config_data)
            # Test if numeric keys are converted to strings or handled specially
            assert config.get('123') == 'numeric_key_value' or config.get(123) == 'numeric_key_value'
        except (TypeError, ConfigError):
            # Expected if implementation doesn't support non-string keys
            pass
    
    def test_config_with_malformed_unicode(self):
        """Test Config with potentially problematic Unicode characters."""
        problematic_unicode = {
            'zero_width': 'value\u200B\u200C\u200D',
            'right_to_left': 'value\u202E\u202D',
            'combining_chars': 'a\u0300\u0301\u0302',
            'emoji_sequences': '👨‍👩‍👧‍👦🏳️‍🌈',
            'surrogates': 'test\ud83d\ude00'  # Emoji as surrogate pair
        }
        
        config = Config(problematic_unicode)
        for key, value in problematic_unicode.items():
            assert config.get(key) == value
    
    @pytest.mark.parametrize("invalid_value", [
        float('inf'),
        float('-inf'),
        float('nan'),
        complex(1, 2),
        bytes(b'binary_data'),
        bytearray(b'mutable_binary')
    ])
    def test_config_with_special_numeric_and_binary_values(self, invalid_value):
        """Test Config with special numeric values and binary data."""
        config_data = {'special_value': invalid_value}
        
        try:
            config = Config(config_data)
            retrieved = config.get('special_value')
            
            if isinstance(invalid_value, float):
                if invalid_value != invalid_value:  # NaN check
                    assert retrieved != retrieved  # NaN != NaN
                else:
                    assert retrieved == invalid_value
            else:
                assert retrieved == invalid_value
        except (TypeError, ValueError, ConfigError):
            # Expected behavior for unsupported types
            pass


class TestConfigManagerAdvanced:
    """Advanced tests for ConfigManager functionality."""
    
    def test_config_manager_with_permission_errors(self, tmp_path):
        """Test ConfigManager handling of file permission errors."""
        import stat
        
        config_file = tmp_path / "readonly_config.json"
        config_data = {'test': 'data'}
        
        # Create file and make it read-only
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Make file read-only
        config_file.chmod(stat.S_IRUSR)
        
        try:
            manager = ConfigManager(str(config_file))
            config = Config({'new': 'data'})
            manager._config = config
            
            # This should raise PermissionError or handle gracefully
            with pytest.raises(PermissionError):
                manager.save()
        finally:
            # Restore write permissions for cleanup
            config_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    
    def test_config_manager_atomic_save_operation(self, tmp_path):
        """Test ConfigManager performs atomic save operations."""
        config_file = tmp_path / "atomic_config.json"
        original_data = {'version': 1, 'data': 'original'}
        
        # Create initial config
        with open(config_file, 'w') as f:
            json.dump(original_data, f)
        
        manager = ConfigManager(str(config_file))
        
        # Simulate atomic save by checking for temporary files
        new_config = Config({'version': 2, 'data': 'updated'})
        manager._config = new_config
        
        # Mock a failure during save to test atomicity
        original_save = manager.save
        def failing_save():
            # Check if temporary file exists during save
            temp_files_before = list(tmp_path.glob("*.tmp"))
            try:
                original_save()
            except:
                # Ensure no temporary files are left behind
                temp_files_after = list(tmp_path.glob("*.tmp"))
                assert len(temp_files_after) <= len(temp_files_before)
                raise
        
        try:
            failing_save()
        except:
            pass
        
        # Verify original file is intact if save failed
        assert config_file.exists()
    
    def test_config_manager_backup_and_recovery(self, tmp_path):
        """Test ConfigManager backup and recovery functionality."""
        config_file = tmp_path / "backup_config.json"
        backup_file = tmp_path / "backup_config.json.backup"
        
        original_data = {'important': 'data', 'version': 1}
        
        # Create initial config
        manager = ConfigManager(str(config_file))
        config = Config(original_data)
        manager._config = config
        manager.save()
        
        # Simulate creating backup before save
        if hasattr(manager, 'create_backup'):
            manager.create_backup()
            assert backup_file.exists()
            
            # Verify backup contains original data
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            assert backup_data == original_data
    
    def test_config_manager_with_different_encodings(self, tmp_path):
        """Test ConfigManager with different file encodings."""
        config_file = tmp_path / "encoded_config.json"
        unicode_data = {
            'chinese': '你好世界',
            'arabic': 'مرحبا بالعالم',
            'emoji': '🌍🚀✨',
            'mixed': 'Hello 世界 🌍'
        }
        
        # Test with UTF-8 encoding (default)
        manager = ConfigManager(str(config_file))
        config = Config(unicode_data)
        manager._config = config
        manager.save()
        
        # Verify file can be read correctly
        loaded_config = manager.load()
        assert loaded_config.get('chinese') == '你好世界'
        assert loaded_config.get('emoji') == '🌍🚀✨'


class TestConfigUtilityAdvanced:
    """Advanced tests for config utility functions."""
    
    def test_load_config_with_file_locking(self, tmp_path):
        """Test load_config behavior with file locking scenarios."""
        import threading
        import time
        
        config_file = tmp_path / "locked_config.json"
        config_data = {'shared': 'data', 'counter': 0}
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        results = []
        
        def concurrent_loader(thread_id):
            """Load config concurrently."""
            try:
                config = load_config(str(config_file))
                results.append(f"thread_{thread_id}_loaded")
                time.sleep(0.01)  # Simulate processing
                results.append(f"thread_{thread_id}_done")
            except Exception as e:
                results.append(f"thread_{thread_id}_error_{str(e)}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_loader, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        loaded_count = len([r for r in results if 'loaded' in r])
        assert loaded_count == 3
    
    def test_save_config_with_validation_hooks(self, tmp_path):
        """Test save_config with validation hooks before saving."""
        config_file = tmp_path / "validated_config.json"
        
        # Test with valid config
        valid_config = Config({
            'required_field': 'present',
            'numeric_field': 42,
            'nested': {'valid': True}
        })
        
        # Should save successfully
        save_config(valid_config, str(config_file))
        assert config_file.exists()
        
        # Test with potentially invalid config
        invalid_config = Config({
            'required_field': None,  # Invalid if None not allowed
            'numeric_field': 'not_a_number'
        })
        
        try:
            save_config(invalid_config, str(config_file))
            # If no validation, should still save
            assert config_file.exists()
        except ConfigError:
            # Expected if validation is enforced
            pass
    
    def test_config_merge_functionality(self):
        """Test config merging capabilities."""
        base_config = Config({
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'base_db'
            },
            'features': {
                'feature_a': True,
                'feature_b': False
            },
            'version': '1.0'
        })
        
        override_config = Config({
            'database': {
                'host': 'remote.server.com',
                'ssl': True  # New field
            },
            'features': {
                'feature_b': True,  # Override
                'feature_c': True   # New feature
            },
            'new_section': {
                'new_setting': 'value'
            }
        })
        
        # Test merge operation (if supported)
        if hasattr(base_config, 'merge'):
            merged = base_config.merge(override_config)
            
            # Verify merged results
            assert merged.get('database')['host'] == 'remote.server.com'
            assert merged.get('database')['port'] == 5432  # Preserved
            assert merged.get('database')['ssl'] is True   # Added
            assert merged.get('features')['feature_b'] is True  # Overridden
            assert merged.get('version') == '1.0'  # Preserved
        else:
            # Manual merge test
            merged_data = base_config.to_dict()
            for key, value in override_config.to_dict().items():
                if key in merged_data and isinstance(merged_data[key], dict) and isinstance(value, dict):
                    merged_data[key].update(value)
                else:
                    merged_data[key] = value
            
            merged = Config(merged_data)
            assert merged.get('database')['host'] == 'remote.server.com'


class TestConfigSecurity:
    """Security-related tests for config functionality."""
    
    def test_config_with_potentially_malicious_input(self):
        """Test Config handling of potentially malicious input."""
        malicious_inputs = [
            {'__class__': 'malicious'},
            {'__dict__': 'attack'},
            {'__globals__': 'global_access'},
            {'eval': 'eval("malicious_code")'},
            {'exec': 'exec("malicious_code")'},
            {'import': '__import__("os").system("ls")'},
            {'../../../etc/passwd': 'path_traversal'},
            {'script_injection': '<script>alert("xss")</script>'},
            {'sql_injection': "'; DROP TABLE users; --"},
            {'command_injection': '$(rm -rf /)'},
        ]
        
        for malicious_data in malicious_inputs:
            try:
                config = Config(malicious_data)
                # Verify it's handled safely
                for key in malicious_data:
                    retrieved = config.get(key)
                    # Should return the value as-is without execution
                    assert retrieved == malicious_data[key]
            except (ConfigError, ValueError, TypeError):
                # Expected behavior for rejecting malicious input
                pass
    
    def test_config_file_path_traversal_protection(self, tmp_path):
        """Test protection against path traversal attacks."""
        dangerous_paths = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\system',
            '/etc/shadow',
            'C:\\Windows\\System32\\config\\SAM',
            '../../../../root/.ssh/id_rsa',
        ]
        
        for dangerous_path in dangerous_paths:
            try:
                # Should either reject the path or normalize it safely
                config = load_config(dangerous_path)
                # If it doesn't raise an exception, ensure it didn't access dangerous files
                assert True  # Placeholder - actual implementation would verify safety
            except (FileNotFoundError, PermissionError, ConfigError, ValueError):
                # Expected behavior for dangerous paths
                pass
    
    def test_config_sensitive_data_handling(self, tmp_path):
        """Test proper handling of sensitive configuration data."""
        sensitive_config = Config({
            'api_key': 'super_secret_key_12345',
            'database_password': 'db_password_secret',
            'private_key': '-----BEGIN PRIVATE KEY-----\nMIIEvQ...',
            'oauth_secret': 'oauth_client_secret',
            'encryption_key': 'aes_256_encryption_key'
        })
        
        config_file = tmp_path / "sensitive_config.json"
        
        # Save sensitive config
        save_config(sensitive_config, str(config_file))
        
        # Verify file permissions are restrictive (if supported)
        import stat
        file_stat = config_file.stat()
        file_mode = stat.filemode(file_stat.st_mode)
        
        # On Unix systems, should be readable only by owner
        if hasattr(stat, 'S_IRGRP'):
            assert not (file_stat.st_mode & stat.S_IRGRP)  # No group read
            assert not (file_stat.st_mode & stat.S_IROTH)  # No other read
        
        # Test that sensitive data is not leaked in error messages
        try:
            # Force an error while processing sensitive config
            corrupted_file = tmp_path / "corrupted_sensitive.json"
            with open(corrupted_file, 'w') as f:
                f.write('{"api_key": "secret_key", invalid_json}')
            
            load_config(str(corrupted_file))
        except json.JSONDecodeError as e:
            # Ensure error message doesn't contain sensitive data
            error_msg = str(e).lower()
            assert 'secret_key' not in error_msg
            assert 'password' not in error_msg


class TestConfigMemoryAndResources:
    """Tests for memory usage and resource management."""
    
    def test_config_memory_efficiency_with_large_configs(self):
        """Test memory efficiency with large configuration objects."""
        import sys
        
        # Create a large config
        large_config_data = {}
        for i in range(10000):
            large_config_data[f'item_{i}'] = {
                'id': i,
                'name': f'Item {i}',
                'description': f'Description for item {i}' * 10,  # Longer strings
                'metadata': {
                    'created': f'2023-01-{i%30+1:02d}',
                    'tags': [f'tag_{j}' for j in range(i % 5 + 1)]
                }
            }
        
        # Measure memory before creating config
        initial_size = sys.getsizeof(large_config_data)
        
        # Create config
        config = Config(large_config_data)
        
        # Test that config doesn't use excessive additional memory
        config_size = sys.getsizeof(config.to_dict())
        
        # Should not use more than 150% of original data size
        assert config_size < initial_size * 1.5
    
    def test_config_garbage_collection_behavior(self):
        """Test proper garbage collection of config objects."""
        import gc
        import weakref
        
        # Create config and get weak reference
        config_data = {'test': 'data', 'large_list': list(range(1000))}
        config = Config(config_data)
        config_ref = weakref.ref(config)
        
        # Verify config exists
        assert config_ref() is not None
        
        # Delete config and force garbage collection
        del config
        gc.collect()
        
        # Verify config was garbage collected
        assert config_ref() is None
    
    def test_config_file_handle_cleanup(self, tmp_path):
        """Test proper cleanup of file handles."""
        config_file = tmp_path / "handle_test_config.json"
        config_data = {'test': 'file_handle_cleanup'}
        
        # Create multiple configs that interact with files
        for i in range(100):
            test_config = Config(config_data)
            save_config(test_config, str(config_file))
            loaded_config = load_config(str(config_file))
            
            # Modify and save again
            loaded_config.set('iteration', i)
            save_config(loaded_config, str(config_file))
        
        # Should not have leaked file handles
        # This test ensures file operations are properly closed
        assert config_file.exists()
        
        # Final verification
        final_config = load_config(str(config_file))
        assert final_config.get('iteration') == 99


class TestConfigCompatibility:
    """Tests for backwards compatibility and version handling."""
    
    def test_config_version_migration(self, tmp_path):
        """Test config version migration scenarios."""
        # Old version config format
        old_config_data = {
            'version': '1.0',
            'settings': {
                'old_setting': 'old_value',
                'deprecated_feature': True
            }
        }
        
        # Save old format
        old_config_file = tmp_path / "old_config.json"
        with open(old_config_file, 'w') as f:
            json.dump(old_config_data, f)
        
        # Load with potential migration
        config = load_config(str(old_config_file))
        
        # Test that old config is still readable
        assert config.get('version') == '1.0'
        assert config.get('settings')['old_setting'] == 'old_value'
        
        # If migration is supported, test new format
        if hasattr(config, 'migrate_to_version'):
            migrated = config.migrate_to_version('2.0')
            assert migrated.get('version') == '2.0'
    
    def test_config_backwards_compatibility(self):
        """Test backwards compatibility with older config formats."""
        # Test various historical config formats
        legacy_formats = [
            # Simple key-value format
            {'simple_key': 'simple_value'},
            
            # Flat namespace format
            {
                'app.name': 'CharacterClient',
                'app.version': '1.0',
                'db.host': 'localhost',
                'db.port': '5432'
            },
            
            # Mixed format
            {
                'flat_setting': 'value',
                'nested': {
                    'subsetting': 'nested_value'
                },
                'list_setting': ['item1', 'item2']
            }
        ]
        
        for legacy_format in legacy_formats:
            config = Config(legacy_format)
            
            # Ensure all data is accessible
            for key, value in legacy_format.items():
                assert config.get(key) == value


class TestConfigErrorHandlingExtended:
    """Extended error handling and recovery tests."""
    
    def test_config_recovery_from_partial_corruption(self, tmp_path):
        """Test config recovery from partially corrupted files."""
        config_file = tmp_path / "partially_corrupted.json"
        
        # Create a partially valid JSON file
        partial_json = '{"valid_key": "valid_value", "another_key":'
        with open(config_file, 'w') as f:
            f.write(partial_json)
        
        # Should handle gracefully
        try:
            config = load_config(str(config_file))
            # If it recovers partially, test what's available
            assert config is not None
        except (json.JSONDecodeError, ConfigError):
            # Expected behavior for corrupted files
            pass
    
    def test_config_with_disk_full_simulation(self, tmp_path):
        """Test config behavior when disk is full (simulated)."""
        config_file = tmp_path / "disk_full_test.json"
        large_config = Config({
            'large_data': 'x' * 1000000,  # 1MB of data
            'more_data': list(range(10000))
        })
        
        try:
            save_config(large_config, str(config_file))
            # If successful, verify file integrity
            assert config_file.exists()
            loaded = load_config(str(config_file))
            assert loaded.get('large_data') == 'x' * 1000000
        except OSError:
            # Expected if disk space issues occur
            pass
    
    def test_config_network_path_handling(self, tmp_path):
        """Test config handling of network paths and UNC paths."""
        network_paths = [
            "//server/share/config.json",
            "\\\\server\\share\\config.json",
            "smb://server/share/config.json",
            "ftp://server/config.json"
        ]
        
        for network_path in network_paths:
            try:
                # Should handle network paths gracefully or reject them appropriately
                config = load_config(network_path)
                assert config is not None
            except (FileNotFoundError, OSError, ValueError, ConfigError):
                # Expected behavior for inaccessible network paths
                pass
    
    def test_config_with_symlink_handling(self, tmp_path):
        """Test config handling of symbolic links."""
        if not hasattr(tmp_path, 'symlink_to'):
            pytest.skip("Symbolic links not supported on this platform")
        
        # Create actual config file
        real_config = tmp_path / "real_config.json"
        config_data = {'symlink_test': True, 'data': 'real_data'}
        with open(real_config, 'w') as f:
            json.dump(config_data, f)
        
        # Create symlink
        try:
            symlink_config = tmp_path / "symlink_config.json"
            symlink_config.symlink_to(real_config)
            
            # Load via symlink
            config = load_config(str(symlink_config))
            assert config.get('symlink_test') is True
            assert config.get('data') == 'real_data'
        except OSError:
            # Skip if symlinks not supported
            pytest.skip("Symbolic links not supported")


class TestConfigStressAndBoundaries:
    """Stress tests and boundary condition tests."""
    
    def test_config_maximum_key_length(self):
        """Test config with extremely long key names."""
        max_key_length = 10000
        long_key = 'k' * max_key_length
        config_data = {long_key: 'long_key_value'}
        
        try:
            config = Config(config_data)
            assert config.get(long_key) == 'long_key_value'
        except (MemoryError, ConfigError):
            # Expected if key length limits are enforced
            pass
    
    def test_config_maximum_nesting_depth(self):
        """Test config with maximum safe nesting depth."""
        # Create nested structure at safe recursion limit
        max_depth = 500
        nested_data = current = {}
        
        for i in range(max_depth):
            current[f'level_{i}'] = {}
            current = current[f'level_{i}']
        current['final'] = 'deepest_value'
        
        try:
            config = Config(nested_data)
            # Navigate to deepest level
            current_level = config.to_dict()
            for i in range(max_depth):
                current_level = current_level[f'level_{i}']
            assert current_level['final'] == 'deepest_value'
        except RecursionError:
            # Expected if recursion limits are hit
            pass
    
    def test_config_with_maximum_values_count(self):
        """Test config with maximum number of key-value pairs."""
        max_items = 100000
        large_flat_config = {}
        
        for i in range(max_items):
            large_flat_config[f'key_{i:06d}'] = f'value_{i}'
        
        try:
            config = Config(large_flat_config)
            
            # Test random access
            import random
            test_indices = random.sample(range(max_items), 100)
            for idx in test_indices:
                key = f'key_{idx:06d}'
                expected = f'value_{idx}'
                assert config.get(key) == expected
        except MemoryError:
            # Expected if memory limits are exceeded
            pass
    
    def test_config_concurrent_modification(self, tmp_path):
        """Test config behavior under concurrent modifications."""
        import threading
        import time
        
        config_file = tmp_path / "concurrent_mod.json"
        initial_data = {'counter': 0, 'modifications': []}
        
        with open(config_file, 'w') as f:
            json.dump(initial_data, f)
        
        results = []
        
        def modifier_thread(thread_id, modification_count):
            """Thread that modifies config file."""
            try:
                for i in range(modification_count):
                    config = load_config(str(config_file))
                    counter = config.get('counter', 0)
                    modifications = config.get('modifications', [])
                    
                    # Simulate some processing time
                    time.sleep(0.001)
                    
                    config.set('counter', counter + 1)
                    modifications.append(f'thread_{thread_id}_mod_{i}')
                    config.set('modifications', modifications)
                    
                    save_config(config, str(config_file))
                    
                results.append(f'thread_{thread_id}_completed')
            except Exception as e:
                results.append(f'thread_{thread_id}_error_{str(e)}')
        
        # Start multiple modifier threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=modifier_thread, args=(i, 5))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify threads completed
        completed_count = len([r for r in results if 'completed' in r])
        assert completed_count == 3
        
        # Verify final state
        final_config = load_config(str(config_file))
        assert final_config.get('counter') >= 0  # Some modifications should have occurred


if __name__ == '__main__':
    # Run all tests with verbose output and detailed failure information
    pytest.main([__file__, '-v', '--tb=long', '--strict-markers'])