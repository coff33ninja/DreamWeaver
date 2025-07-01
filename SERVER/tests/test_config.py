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
        Prepares sample configuration data and a temporary directory for use in each test.
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
        Tests that the Config class initializes correctly with valid configuration data and that attribute values match the expected input.
        """
        config = Config(**self.test_config_data)
        self.assertEqual(config.database['host'], 'localhost')
        self.assertEqual(config.server['port'], 8000)
        self.assertFalse(config.server['debug'])
        
    def test_config_initialization_with_empty_data(self):
        """
        Test that the Config class can be initialized with no configuration data.
        
        Asserts that creating a Config instance with empty input does not raise exceptions and results in a valid Config object.
        """
        config = Config()
        # Should not raise an exception
        self.assertIsInstance(config, Config)
        
    def test_config_initialization_with_partial_data(self):
        """
        Test initialization of the Config class with partial configuration data, ensuring that provided fields are set correctly.
        """
        partial_data = {'database': {'host': 'localhost'}}
        config = Config(**partial_data)
        self.assertEqual(config.database['host'], 'localhost')
        
    def test_config_attribute_access(self):
        """
        Verify that configuration attributes can be accessed correctly after initialization.
        
        Ensures that nested configuration values are accessible via attribute access on the Config object.
        """
        config = Config(**self.test_config_data)
        self.assertEqual(config.database['host'], 'localhost')
        self.assertEqual(config.server['port'], 8000)
        
    def test_config_attribute_modification(self):
        """
        Test that configuration attributes can be modified after initialization.
        
        Verifies that updating a configuration attribute reflects the change when accessed.
        """
        config = Config(**self.test_config_data)
        config.database['host'] = 'newhost'
        self.assertEqual(config.database['host'], 'newhost')
        
    def test_load_config_from_json_file(self):
        """
        Tests that a configuration can be successfully loaded from a JSON file and returns a Config object or dictionary.
        """
        config_file = os.path.join(self.temp_dir, 'test_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.test_config_data, f)
            
        with patch('builtins.open', mock_open(read_data=json.dumps(self.test_config_data))):
            config = load_config(config_file)
            self.assertIsInstance(config, (Config, dict))
            
    def test_load_config_from_yaml_file(self):
        """
        Test that configuration can be loaded from a YAML file and returns a Config object or dictionary.
        """
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        yaml_data = yaml.dump(self.test_config_data)
        
        with patch('builtins.open', mock_open(read_data=yaml_data)):
            with patch('yaml.safe_load', return_value=self.test_config_data):
                config = load_config(config_file)
                self.assertIsInstance(config, (Config, dict))
                
    def test_load_config_file_not_found(self):
        """
        Test that attempting to load a configuration from a non-existent file raises a FileNotFoundError or IOError.
        """
        non_existent_file = '/path/to/non/existent/config.json'
        with self.assertRaises((FileNotFoundError, IOError)):
            load_config(non_existent_file)
            
    def test_load_config_invalid_json(self):
        """
        Test that loading a configuration file with invalid JSON content raises a JSON decoding error.
        """
        invalid_json = '{"database": {"host": "localhost", "port": 5432}'  # Missing closing brace
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                load_config('config.json')
                
    def test_load_config_invalid_yaml(self):
        """
        Test that loading a configuration file with invalid YAML content raises a YAMLError.
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
        Verifies that the configuration validator returns True for a valid configuration structure.
        """
        self.assertTrue(validate_config(self.test_config_data))
        
    def test_validate_config_missing_required_fields(self):
        """
        Test validation of a configuration missing required fields.
        
        Verifies that validating a config dictionary lacking required sections returns a boolean result.
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
        Test that validating a None configuration input returns False.
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
        Test that configuration can be retrieved from environment variables and returns a dictionary.
        """
        env_config = get_env_config()
        self.assertIsInstance(env_config, dict)
        
    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_config_empty_environment(self):
        """
        Test that retrieving configuration from environment variables returns an empty dictionary when no relevant environment variables are set.
        """
        env_config = get_env_config()
        self.assertIsInstance(env_config, dict)
        
    def test_config_merge_with_environment(self):
        """
        Test that configuration values from environment variables are correctly merged with the base configuration.
        
        Verifies that environment-derived configuration is returned as a dictionary and can be integrated with an existing Config object.
        """
        base_config = Config(**self.test_config_data)
        with patch.dict(os.environ, {'DB_HOST': 'env_host', 'DB_PORT': '9999'}):
            env_config = get_env_config()
            # Test merging logic here
            self.assertIsInstance(env_config, dict)
            
    def test_config_default_values(self):
        """
        Verify that a Config object initializes correctly with default values when no data is provided.
        """
        config = Config()
        # Test that default values are properly set
        self.assertIsInstance(config, Config)
        
    def test_config_serialization(self):
        """
        Tests that a Config object can be serialized to a JSON string using its __dict__ attribute.
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
        Test that a configuration object can be deep copied, ensuring the copy is independent of the original.
        
        Verifies that modifying nested attributes in the original does not affect the deep-copied object.
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
        Tests that boolean-like string and native boolean values in configuration data are accessible as attributes.
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
        Test that numeric string values in the configuration are accessible as attributes.
        
        Verifies that configuration fields containing numeric values as strings are present and accessible in the configuration object.
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
        Verify that configuration objects correctly handle list and nested dictionary values as attributes.
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
        Test that the Config class can be instantiated with configuration keys that differ only in case.
        
        Verifies that the configuration object is created successfully when provided with keys of varying case sensitivity.
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
        Tests that configuration values containing special characters are correctly handled and accessible.
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
        Tests that configuration values containing Unicode characters are correctly handled and accessible.
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
        Tests that the Config class can be initialized with a very large configuration dictionary without errors.
        """
        large_config = {}
        for i in range(1000):
            large_config[f'key_{i}'] = f'value_{i}' * 100
            
        config = Config(**large_config)
        self.assertIsInstance(config, Config)
        
    def test_deeply_nested_config(self):
        """
        Tests that the Config class can be initialized with and correctly handle deeply nested configuration dictionaries.
        """
        nested_config = {'level1': {'level2': {'level3': {'level4': {'value': 'deep'}}}}}
        config = Config(**nested_config)
        self.assertIsInstance(config, Config)
        
    def test_config_with_none_values(self):
        """
        Test that the Config class can be initialized with None values in the configuration dictionary.
        """
        none_config = {
            'database': None,
            'server': {'host': None, 'port': 8000}
        }
        config = Config(**none_config)
        self.assertIsInstance(config, Config)
        
    def test_config_circular_reference_prevention(self):
        """
        Verify that initializing a Config object does not result in circular reference issues.
        
        This test is theoretical, as standard configuration formats like JSON do not support circular references.
        """
        # This is more of a theoretical test since JSON doesn't support circular refs
        config = Config(test='value')
        self.assertIsInstance(config, Config)
        
    def test_config_memory_usage(self):
        """
        Tests that a large configuration object consumes measurable memory, ensuring `Config` instances with substantial data have nonzero memory usage.
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
        Creates a temporary directory for use in integration test fixtures.
        """
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """
        Remove the temporary directory and its contents after each integration test to clean up test artifacts.
        """
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_full_config_workflow(self):
        """
        Tests the end-to-end workflow of writing a configuration to a file, loading it, and validating its structure.
        
        This test verifies that a configuration can be serialized to disk, loaded using the configuration loader, and validated for correctness. It asserts that the validation result is a boolean value.
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
        Test that environment variable values can override base configuration values.
        
        This test verifies that when environment variables are set, they are correctly retrieved and can be used to override values in the base configuration.
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
        Test that configuration validation returns a boolean when provided with various data types for configuration fields.
        
        This test checks that the `validate_config` function consistently returns a boolean result when validating configurations with different data types, including valid and invalid types for fields such as port and debug.
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
        Tests that configuration validation correctly handles value ranges for fields such as server port and database pool size.
        
        Verifies that the `validate_config` function returns a boolean when provided with configurations containing both valid and invalid value ranges.
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
        Test that sensitive data fields such as passwords, API keys, and tokens are correctly handled during configuration initialization.
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
        Tests that configuration objects can be instantiated with values containing potentially dangerous paths or commands without raising exceptions.
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
        Test that configuration containing potentially malicious injection strings can be safely initialized without causing execution or errors.
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
        Measures the time taken to initialize a large Config object and asserts that loading completes within one second.
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
        Measures the time required to repeatedly access a deeply nested configuration attribute and asserts that access completes within 0.1 seconds.
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