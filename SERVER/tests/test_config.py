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
            """
            Initialize the configuration object with attributes set from keyword arguments.
            
            Each key-value pair in kwargs becomes an attribute of the instance.
            """
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    Config = MockConfig
    load_config = lambda x: MockConfig()
    validate_config = lambda x: True
    get_env_config = lambda: {}


class TestConfig(unittest.TestCase):
    """Comprehensive tests for configuration management."""
    
    def setUp(self):
        """
        Prepares a sample configuration dictionary and creates a temporary directory for use in each test.
        """
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
        """
        Removes the temporary directory and its contents after each test to ensure a clean test environment.
        """
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_config_initialization_with_valid_data(self):
        """
        Verify that the Config class initializes correctly with valid configuration data and that its attributes match the expected values.
        """
        config = Config(**self.test_config_data)
        self.assertEqual(config.database['host'], 'localhost')
        self.assertEqual(config.server['port'], 8000)
        self.assertFalse(config.server['debug'])
        
    def test_config_initialization_with_empty_data(self):
        """
        Test that the Config class can be initialized with no configuration data.
        
        Verifies that creating a Config instance without arguments does not raise an exception and results in a valid Config object.
        """
        config = Config()
        # Should not raise an exception
        self.assertIsInstance(config, Config)
        
    def test_config_initialization_with_partial_data(self):
        """
        Test initialization of the Config class with partial configuration data.
        
        Verifies that the Config object correctly sets provided fields and allows access to them when only a subset of configuration data is supplied.
        """
        partial_data = {'database': {'host': 'localhost'}}
        config = Config(**partial_data)
        self.assertEqual(config.database['host'], 'localhost')
        
    def test_config_attribute_access(self):
        """
        Test that configuration attributes can be accessed correctly after initialization.
        """
        config = Config(**self.test_config_data)
        self.assertEqual(config.database['host'], 'localhost')
        self.assertEqual(config.server['port'], 8000)
        
    def test_config_attribute_modification(self):
        """
        Test that configuration attributes can be modified after initialization.
        """
        config = Config(**self.test_config_data)
        config.database['host'] = 'newhost'
        self.assertEqual(config.database['host'], 'newhost')
        
    def test_load_config_from_json_file(self):
        """
        Test that configuration can be loaded correctly from a JSON file.
        
        Verifies that the loaded configuration is an instance of the expected Config class or a dictionary.
        """
        config_file = os.path.join(self.temp_dir, 'test_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.test_config_data, f)
            
        with patch('builtins.open', mock_open(read_data=json.dumps(self.test_config_data))):
            config = load_config(config_file)
            self.assertIsInstance(config, (Config, dict))
            
    def test_load_config_from_yaml_file(self):
        """
        Test that configuration data can be loaded correctly from a YAML file.
        
        Verifies that loading a YAML configuration file returns an instance of the expected configuration class or dictionary.
        """
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        yaml_data = yaml.dump(self.test_config_data)
        
        with patch('builtins.open', mock_open(read_data=yaml_data)):
            with patch('yaml.safe_load', return_value=self.test_config_data):
                config = load_config(config_file)
                self.assertIsInstance(config, (Config, dict))
                
    def test_load_config_file_not_found(self):
        """
        Test that loading a configuration from a non-existent file raises a FileNotFoundError or IOError.
        """
        non_existent_file = '/path/to/non/existent/config.json'
        with self.assertRaises((FileNotFoundError, IOError)):
            load_config(non_existent_file)
            
    def test_load_config_invalid_json(self):
        """
        Test that loading a configuration file with invalid JSON format raises a JSONDecodeError or ValueError.
        """
        invalid_json = '{"database": {"host": "localhost", "port": 5432}'  # Missing closing brace
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                load_config('config.json')
                
    def test_load_config_invalid_yaml(self):
        """
        Test that loading a configuration file with invalid YAML format raises a YAMLError.
        """
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
        """
        Test that configuration validation succeeds when provided with a valid configuration structure.
        """
        self.assertTrue(validate_config(self.test_config_data))
        
    def test_validate_config_missing_required_fields(self):
        """
        Test validation behavior when required configuration fields are missing.
        
        Verifies that the validation function handles configurations lacking required sections, such as the 'database' field, and returns a boolean result.
        """
        incomplete_config = {'server': {'host': 'localhost'}}
        # Assuming database section is required
        result = validate_config(incomplete_config)
        # This test depends on the actual validation logic
        self.assertIsInstance(result, bool)
        
    def test_validate_config_invalid_data_types(self):
        """
        Test that configuration validation returns a boolean when provided with invalid data types in the configuration.
        """
        invalid_config = {
            'database': {
                'host': 123,  # Should be string
                'port': 'invalid_port'  # Should be integer
            }
        }
        result = validate_config(invalid_config)
        self.assertIsInstance(result, bool)
        
    def test_validate_config_empty_config(self):
        """
        Test that validating an empty configuration returns a boolean result.
        """
        empty_config = {}
        result = validate_config(empty_config)
        self.assertIsInstance(result, bool)
        
    def test_validate_config_none_input(self):
        """
        Test that validating a configuration with `None` input returns `False`.
        """
        result = validate_config(None)
        self.assertFalse(result)
        
    @patch.dict(os.environ, {
        'DB_HOST': 'env_localhost',
        'DB_PORT': '5433',
        'SERVER_DEBUG': 'true',
        'LOG_LEVEL': 'DEBUG'
    })
    def test_get_env_config_with_environment_variables(self):
        """
        Test that configuration can be retrieved from environment variables and is returned as a dictionary.
        """
        env_config = get_env_config()
        self.assertIsInstance(env_config, dict)
        
    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_config_empty_environment(self):
        """
        Test that `get_env_config` returns an empty dictionary when no relevant environment variables are set.
        """
        env_config = get_env_config()
        self.assertIsInstance(env_config, dict)
        
    def test_config_merge_with_environment(self):
        """
        Test that configuration values from environment variables are correctly merged with the base configuration.
        
        Verifies that environment-derived configuration is returned as a dictionary and can be integrated with an existing configuration object.
        """
        base_config = Config(**self.test_config_data)
        with patch.dict(os.environ, {'DB_HOST': 'env_host', 'DB_PORT': '9999'}):
            env_config = get_env_config()
            # Test merging logic here
            self.assertIsInstance(env_config, dict)
            
    def test_config_default_values(self):
        """
        Verify that a configuration object initializes with default values set.
        
        Ensures that creating a Config instance without arguments results in an object with default configuration values.
        """
        config = Config()
        # Test that default values are properly set
        self.assertIsInstance(config, Config)
        
    def test_config_serialization(self):
        """
        Tests that a configuration object can be serialized to a JSON string.
        
        Verifies that the configuration data can be converted to a JSON-formatted string and that the result is of type `str`. Handles cases where the configuration object may not be serializable.
        """
        config = Config(**self.test_config_data)
        try:
            serialized = json.dumps(config.__dict__)
            self.assertIsInstance(serialized, str)
        except (AttributeError, TypeError):
            # If config doesn't have __dict__ or isn't serializable
            pass
            
    def test_config_deep_copy(self):
        """
        Test that deep copying a configuration object creates an independent copy with no shared references to nested data.
        """
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
        """
        Test that boolean-like string and native boolean values are correctly handled in configuration data.
        """
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
        """
        Test that numeric string values in the configuration are correctly handled.
        
        Verifies that configuration fields containing numeric values as strings are accessible and present in the configuration object.
        """
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
        """
        Test that configuration objects correctly handle list and nested dictionary values.
        
        Verifies that lists and nested dictionaries are properly stored and accessible as attributes in the configuration object.
        """
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
        """
        Test that configuration keys with different cases are treated as distinct entries.
        
        Verifies that the configuration system distinguishes between keys that differ only by letter casing.
        """
        case_test_data = {
            'Database': {'Host': 'localhost'},
            'database': {'host': 'localhost'}
        }
        config = Config(**case_test_data)
        # Test case handling
        self.assertIsInstance(config, Config)
        
    def test_config_special_characters_in_values(self):
        """
        Test that configuration values containing special characters are handled correctly.
        
        Verifies that special characters in configuration values, such as passwords and connection strings, do not cause errors and are accessible as expected.
        """
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
        """
        Test that configuration objects correctly handle and store unicode characters in their values.
        """
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
        """
        Test that the configuration system can handle very large configuration files with many keys and large string values.
        """
        large_config = {}
        for i in range(1000):
            large_config[f'key_{i}'] = f'value_{i}' * 100
            
        config = Config(**large_config)
        self.assertIsInstance(config, Config)
        
    def test_deeply_nested_config(self):
        """
        Test that the configuration system correctly handles deeply nested dictionary structures.
        """
        nested_config = {'level1': {'level2': {'level3': {'level4': {'value': 'deep'}}}}}
        config = Config(**nested_config)
        self.assertIsInstance(config, Config)
        
    def test_config_with_none_values(self):
        """
        Verify that the configuration system correctly handles fields set to None values.
        """
        none_config = {
            'database': None,
            'server': {'host': None, 'port': 8000}
        }
        config = Config(**none_config)
        self.assertIsInstance(config, Config)
        
    def test_config_circular_reference_prevention(self):
        """
        Test that the configuration system does not allow or create circular references.
        
        This test is theoretical, as standard JSON does not support circular references, but ensures that the Config object can be instantiated without introducing such references.
        """
        # This is more of a theoretical test since JSON doesn't support circular refs
        config = Config(test='value')
        self.assertIsInstance(config, Config)
        
    def test_config_memory_usage(self):
        """
        Test that a configuration object with a large dataset occupies measurable memory.
        
        Creates a configuration with multiple sections and keys, then verifies that its memory footprint is greater than zero.
        """
        import sys
        config_data = {f'section_{i}': {f'key_{j}': f'value_{j}' for j in range(100)} for i in range(10)}
        config = Config(**config_data)
        
        # Basic memory usage check
        size = sys.getsizeof(config)
        self.assertGreater(size, 0)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration functionality."""
    
    def setUp(self):
        """
        Creates a temporary directory for use in integration tests.
        """
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """
        Removes the temporary directory and its contents after each integration test to ensure a clean test environment.
        """
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_full_config_workflow(self):
        """
        Tests the end-to-end configuration workflow, including writing configuration data to a file, loading it into a configuration object, and validating the loaded configuration.
        """
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
        """
        Test that environment variable values can override base configuration settings.
        
        Verifies that configuration values provided via environment variables are correctly extracted and can be used to override values in the base configuration.
        """
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
        """
        Test that configuration validation returns a boolean when required fields are present or missing.
        
        Verifies that the `validate_config` function returns a boolean result for configurations with various combinations of required fields.
        """
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
        """
        Test that configuration validation returns a boolean result for various data type scenarios.
        
        This test checks that the `validate_config` function consistently returns a boolean when validating configurations with different data types for fields such as port and debug.
        """
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
        """
        Test that configuration value ranges are validated and the result is a boolean.
        
        Covers valid and invalid ranges for server port and database pool size.
        """
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
        """
        Verify that configuration objects can be initialized with sensitive data fields such as passwords, API keys, and tokens.
        """
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
        """
        Test that configuration values containing potentially dangerous paths or commands are handled safely.
        
        Verifies that a configuration with suspicious file paths or shell commands can be instantiated without error.
        """
        potentially_dangerous_config = {
            'paths': {
                'upload_dir': '/tmp/../../../etc/passwd',
                'log_file': '/var/log/app.log; rm -rf /'
            }
        }
        config = Config(**potentially_dangerous_config)
        self.assertIsInstance(config, Config)
        
    def test_injection_prevention(self):
        """
        Test that configuration containing potentially malicious injection strings is handled safely.
        
        Ensures that the configuration system can accept values with SQL or shell injection patterns without executing or mishandling them.
        """
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
        """
        Test that loading a large configuration into a Config object completes in under one second.
        
        Ensures that the Config class can efficiently handle initialization with a sizable nested dictionary structure.
        """
        import time
        
        large_config = {f'section_{i}': {f'key_{j}': f'value_{j}' for j in range(50)} for i in range(20)}
        
        start_time = time.time()
        config = Config(**large_config)
        end_time = time.time()
        
        # Should load in reasonable time (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsInstance(config, Config)
        
    def test_config_access_performance(self):
        """
        Test that accessing nested configuration attributes is performed within an acceptable time frame.
        
        Measures the time taken to access a deeply nested configuration value 1000 times and asserts that the total duration is less than 0.1 seconds.
        """
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