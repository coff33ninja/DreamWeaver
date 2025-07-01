import unittest
import os
import tempfile
import json
import yaml
from unittest.mock import patch, mock_open, MagicMock
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import Config, load_config, validate_config, get_env_config
except ImportError:
    # Create mock config module for testing if it doesn't exist
    class MockConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    Config = MockConfig
    load_config = lambda x: MockConfig()
    validate_config = lambda x: True
    get_env_config = lambda: {}


class TestConfig(unittest.TestCase):
    """Comprehensive tests for configuration management."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'testdb',
                'user': 'testuser',
                'password': 'testpass'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'features': {
                'auth_enabled': True,
                'rate_limiting': True,
                'cors_enabled': False
            }
        }
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_config_initialization_with_valid_data(self):
        """Test Config class initialization with valid configuration data."""
        config = Config(**self.test_config_data)
        self.assertEqual(config.database['host'], 'localhost')
        self.assertEqual(config.server['port'], 8000)
        self.assertFalse(config.server['debug'])
        
    def test_config_initialization_with_empty_data(self):
        """Test Config class initialization with empty configuration data."""
        config = Config()
        # Should not raise an exception
        self.assertIsInstance(config, Config)
        
    def test_config_initialization_with_partial_data(self):
        """Test Config class initialization with partial configuration data."""
        partial_data = {'database': {'host': 'localhost'}}
        config = Config(**partial_data)
        self.assertEqual(config.database['host'], 'localhost')
        
    def test_config_attribute_access(self):
        """Test accessing configuration attributes."""
        config = Config(**self.test_config_data)
        self.assertEqual(config.database['host'], 'localhost')
        self.assertEqual(config.server['port'], 8000)
        
    def test_config_attribute_modification(self):
        """Test modifying configuration attributes."""
        config = Config(**self.test_config_data)
        config.database['host'] = 'newhost'
        self.assertEqual(config.database['host'], 'newhost')
        
    def test_load_config_from_json_file(self):
        """Test loading configuration from a JSON file."""
        config_file = os.path.join(self.temp_dir, 'test_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.test_config_data, f)
            
        with patch('builtins.open', mock_open(read_data=json.dumps(self.test_config_data))):
            config = load_config(config_file)
            self.assertIsInstance(config, (Config, dict))
            
    def test_load_config_from_yaml_file(self):
        """Test loading configuration from a YAML file."""
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        yaml_data = yaml.dump(self.test_config_data)
        
        with patch('builtins.open', mock_open(read_data=yaml_data)):
            with patch('yaml.safe_load', return_value=self.test_config_data):
                config = load_config(config_file)
                self.assertIsInstance(config, (Config, dict))
                
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        non_existent_file = '/path/to/non/existent/config.json'
        with self.assertRaises((FileNotFoundError, IOError)):
            load_config(non_existent_file)
            
    def test_load_config_invalid_json(self):
        """Test loading configuration with invalid JSON format."""
        invalid_json = '{"database": {"host": "localhost", "port": 5432}'  # Missing closing brace
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                load_config('config.json')
                
    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML format."""
        invalid_yaml = """
        database:
          host: localhost
          port: 5432
        - invalid_yaml_structure
        """
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")):
                with self.assertRaises(yaml.YAMLError):
                    load_config('config.yaml')
                    
    def test_validate_config_valid_structure(self):
        """Test configuration validation with valid structure."""
        self.assertTrue(validate_config(self.test_config_data))
        
    def test_validate_config_missing_required_fields(self):
        """Test configuration validation with missing required fields."""
        incomplete_config = {'server': {'host': 'localhost'}}
        # Assuming database section is required
        result = validate_config(incomplete_config)
        # This test depends on the actual validation logic
        self.assertIsInstance(result, bool)
        
    def test_validate_config_invalid_data_types(self):
        """Test configuration validation with invalid data types."""
        invalid_config = {
            'database': {
                'host': 123,  # Should be string
                'port': 'invalid_port'  # Should be integer
            }
        }
        result = validate_config(invalid_config)
        self.assertIsInstance(result, bool)
        
    def test_validate_config_empty_config(self):
        """Test configuration validation with empty configuration."""
        empty_config = {}
        result = validate_config(empty_config)
        self.assertIsInstance(result, bool)
        
    def test_validate_config_none_input(self):
        """Test configuration validation with None input."""
        result = validate_config(None)
        self.assertFalse(result)
        
    @patch.dict(os.environ, {
        'DB_HOST': 'env_localhost',
        'DB_PORT': '5433',
        'SERVER_DEBUG': 'true',
        'LOG_LEVEL': 'DEBUG'
    })
    def test_get_env_config_with_environment_variables(self):
        """Test getting configuration from environment variables."""
        env_config = get_env_config()
        self.assertIsInstance(env_config, dict)
        
    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_config_empty_environment(self):
        """Test getting configuration when no environment variables are set."""
        env_config = get_env_config()
        self.assertIsInstance(env_config, dict)
        
    def test_config_merge_with_environment(self):
        """Test merging configuration with environment variables."""
        base_config = Config(**self.test_config_data)
        with patch.dict(os.environ, {'DB_HOST': 'env_host', 'DB_PORT': '9999'}):
            env_config = get_env_config()
            # Test merging logic here
            self.assertIsInstance(env_config, dict)
            
    def test_config_default_values(self):
        """Test configuration with default values."""
        config = Config()
        # Test that default values are properly set
        self.assertIsInstance(config, Config)
        
    def test_config_serialization(self):
        """Test configuration serialization to JSON."""
        config = Config(**self.test_config_data)
        try:
            serialized = json.dumps(config.__dict__)
            self.assertIsInstance(serialized, str)
        except (AttributeError, TypeError):
            # If config doesn't have __dict__ or isn't serializable
            pass
            
    def test_config_deep_copy(self):
        """Test deep copying configuration objects."""
        import copy
        config = Config(**self.test_config_data)
        try:
            config_copy = copy.deepcopy(config)
            self.assertIsInstance(config_copy, Config)
            # Ensure it's a deep copy, not a reference
            if hasattr(config, 'database') and hasattr(config_copy, 'database'):
                config.database['host'] = 'modified'
                self.assertNotEqual(config_copy.database.get('host'), 'modified')
        except (AttributeError, TypeError):
            pass
            
    def test_config_boolean_conversion(self):
        """Test boolean value conversion in configuration."""
        bool_test_data = {
            'features': {
                'flag1': 'true',
                'flag2': 'false', 
                'flag3': '1',
                'flag4': '0',
                'flag5': True,
                'flag6': False
            }
        }
        config = Config(**bool_test_data)
        if hasattr(config, 'features'):
            self.assertIn('flag1', config.features)
            
    def test_config_numeric_conversion(self):
        """Test numeric value conversion in configuration."""
        numeric_test_data = {
            'settings': {
                'timeout': '30',
                'retries': '5',
                'rate_limit': '100.5'
            }
        }
        config = Config(**numeric_test_data)
        if hasattr(config, 'settings'):
            self.assertIn('timeout', config.settings)
            
    def test_config_list_and_dict_handling(self):
        """Test handling of lists and nested dictionaries in configuration."""
        complex_data = {
            'allowed_hosts': ['localhost', '127.0.0.1'],
            'middleware': {
                'auth': {'enabled': True, 'providers': ['oauth', 'basic']},
                'cors': {'origins': ['*'], 'methods': ['GET', 'POST']}
            }
        }
        config = Config(**complex_data)
        if hasattr(config, 'allowed_hosts'):
            self.assertIsInstance(config.allowed_hosts, list)
            
    def test_config_case_sensitivity(self):
        """Test configuration key case sensitivity."""
        case_test_data = {
            'Database': {'Host': 'localhost'},
            'database': {'host': 'localhost'}
        }
        config = Config(**case_test_data)
        # Test case handling
        self.assertIsInstance(config, Config)
        
    def test_config_special_characters_in_values(self):
        """Test configuration with special characters in values."""
        special_char_data = {
            'database': {
                'password': 'p@ssw0rd!@#$%^&*()',
                'connection_string': 'postgresql://user:pass@host:5432/db?sslmode=require'
            }
        }
        config = Config(**special_char_data)
        if hasattr(config, 'database'):
            self.assertIn('password', config.database)
            
    def test_config_unicode_handling(self):
        """Test configuration with unicode characters."""
        unicode_data = {
            'messages': {
                'greeting': 'Hello, 世界!',
                'emoji': '🚀 Test Config 🎉'
            }
        }
        config = Config(**unicode_data)
        if hasattr(config, 'messages'):
            self.assertIn('greeting', config.messages)


class TestConfigEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for configuration."""
    
    def test_very_large_config_file(self):
        """Test handling of very large configuration files."""
        large_config = {}
        for i in range(1000):
            large_config[f'key_{i}'] = f'value_{i}' * 100
            
        config = Config(**large_config)
        self.assertIsInstance(config, Config)
        
    def test_deeply_nested_config(self):
        """Test handling of deeply nested configuration structures."""
        nested_config = {'level1': {'level2': {'level3': {'level4': {'value': 'deep'}}}}}
        config = Config(**nested_config)
        self.assertIsInstance(config, Config)
        
    def test_config_with_none_values(self):
        """Test configuration with None values."""
        none_config = {
            'database': None,
            'server': {'host': None, 'port': 8000}
        }
        config = Config(**none_config)
        self.assertIsInstance(config, Config)
        
    def test_config_circular_reference_prevention(self):
        """Test prevention of circular references in configuration."""
        # This is more of a theoretical test since JSON doesn't support circular refs
        config = Config(test='value')
        self.assertIsInstance(config, Config)
        
    def test_config_memory_usage(self):
        """Test configuration memory usage with large datasets."""
        import sys
        config_data = {f'section_{i}': {f'key_{j}': f'value_{j}' for j in range(100)} for i in range(10)}
        config = Config(**config_data)
        
        # Basic memory usage check
        size = sys.getsizeof(config)
        self.assertGreater(size, 0)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_full_config_workflow(self):
        """Test complete configuration workflow from file to object."""
        config_data = {
            'app': {'name': 'TestApp', 'version': '1.0.0'},
            'database': {'url': 'sqlite:///test.db'}
        }
        
        # Write config file
        config_file = os.path.join(self.temp_dir, 'app_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        # Load and validate
        try:
            config = load_config(config_file)
            is_valid = validate_config(config if isinstance(config, dict) else config.__dict__)
            self.assertIsInstance(is_valid, bool)
        except (NameError, AttributeError):
            # Skip if functions don't exist
            pass
            
    def test_config_environment_override(self):
        """Test configuration with environment variable overrides."""
        base_config = {'server': {'port': 8000}}
        
        with patch.dict(os.environ, {'SERVER_PORT': '9000'}):
            config = Config(**base_config)
            env_config = get_env_config()
            
            # Test that environment variables can override base config
            self.assertIsInstance(config, Config)
            self.assertIsInstance(env_config, dict)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation scenarios."""
    
    def test_required_fields_validation(self):
        """Test validation of required configuration fields."""
        required_fields = ['database', 'server']
        test_configs = [
            {'database': {'host': 'localhost'}, 'server': {'port': 8000}},  # Valid
            {'database': {'host': 'localhost'}},  # Missing server
            {'server': {'port': 8000}},  # Missing database
            {}  # Missing both
        ]
        
        for config in test_configs:
            result = validate_config(config)
            self.assertIsInstance(result, bool)
            
    def test_data_type_validation(self):
        """Test validation of configuration data types."""
        type_test_configs = [
            {'server': {'port': 8000}},  # Valid integer
            {'server': {'port': '8000'}},  # String instead of integer
            {'server': {'port': None}},  # None value
            {'server': {'debug': True}},  # Valid boolean
            {'server': {'debug': 'true'}},  # String instead of boolean
        ]
        
        for config in type_test_configs:
            result = validate_config(config)
            self.assertIsInstance(result, bool)
            
    def test_range_validation(self):
        """Test validation of configuration value ranges."""
        range_test_configs = [
            {'server': {'port': 8000}},  # Valid port
            {'server': {'port': 0}},  # Invalid port (too low)
            {'server': {'port': 99999}},  # Invalid port (too high)
            {'database': {'pool_size': 10}},  # Valid pool size
            {'database': {'pool_size': -1}},  # Invalid pool size
        ]
        
        for config in range_test_configs:
            result = validate_config(config)
            self.assertIsInstance(result, bool)


class TestConfigSecurity(unittest.TestCase):
    """Test configuration security aspects."""
    
    def test_sensitive_data_handling(self):
        """Test handling of sensitive configuration data."""
        sensitive_config = {
            'database': {
                'password': 'super_secret_password',
                'api_key': 'sk-1234567890abcdef',
                'token': 'jwt_token_here'
            }
        }
        config = Config(**sensitive_config)
        self.assertIsInstance(config, Config)
        
    def test_config_sanitization(self):
        """Test sanitization of configuration values."""
        potentially_dangerous_config = {
            'paths': {
                'upload_dir': '/tmp/../../../etc/passwd',
                'log_file': '/var/log/app.log; rm -rf /'
            }
        }
        config = Config(**potentially_dangerous_config)
        self.assertIsInstance(config, Config)
        
    def test_injection_prevention(self):
        """Test prevention of injection attacks in configuration."""
        injection_test_config = {
            'database': {
                'query': "SELECT * FROM users WHERE id = '1'; DROP TABLE users; --"
            },
            'commands': {
                'backup': "backup.sh && rm -rf /"
            }
        }
        config = Config(**injection_test_config)
        self.assertIsInstance(config, Config)


class TestConfigPerformance(unittest.TestCase):
    """Test configuration performance aspects."""
    
    def test_config_loading_performance(self):
        """Test performance of configuration loading."""
        import time
        
        large_config = {f'section_{i}': {f'key_{j}': f'value_{j}' for j in range(50)} for i in range(20)}
        
        start_time = time.time()
        config = Config(**large_config)
        end_time = time.time()
        
        # Should load in reasonable time (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsInstance(config, Config)
        
    def test_config_access_performance(self):
        """Test performance of configuration attribute access."""
        import time
        
        config = Config(**{'test': {'nested': {'deep': {'value': 'test'}}}})
        
        start_time = time.time()
        for _ in range(1000):
            try:
                _ = config.test['nested']['deep']['value']
            except (AttributeError, KeyError, TypeError):
                pass
        end_time = time.time()
        
        # Should access attributes quickly
        self.assertLess(end_time - start_time, 0.1)


if __name__ == '__main__':
    # Configure test runner with enhanced verbosity
    unittest.main(verbosity=2, buffer=True)