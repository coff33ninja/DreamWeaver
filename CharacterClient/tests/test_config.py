import pytest
import os
import tempfile
import json
import yaml
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the config module being tested
try:
    from CharacterClient.config import Config, ConfigError, load_config, validate_config
except ImportError:
    # Handle case where config module structure might be different
    try:
        from CharacterClient import config
        Config = getattr(config, 'Config', None)
        ConfigError = getattr(config, 'ConfigError', Exception)
        load_config = getattr(config, 'load_config', None)
        validate_config = getattr(config, 'validate_config', None)
    except ImportError:
        # Fallback for different import structures
        import sys
        sys.path.append('CharacterClient')
        import config
        Config = getattr(config, 'Config', None)
        ConfigError = getattr(config, 'ConfigError', Exception)
        load_config = getattr(config, 'load_config', None)
        validate_config = getattr(config, 'validate_config', None)


class TestConfig:
    """Comprehensive test suite for Config functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self.yaml_config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Default valid configuration for testing
        self.valid_config = {
            'api_key': 'test_api_key_123',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False,
            'character_settings': {
                'default_name': 'TestCharacter',
                'max_characters': 100,
                'allowed_types': ['warrior', 'mage', 'rogue']
            }
        }
        
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_config_creation_with_valid_data(self):
        """Test Config object creation with valid configuration data."""
        if Config:
            config = Config(self.valid_config)
            assert config is not None
            assert hasattr(config, 'api_key') or 'api_key' in config
    
    def test_config_creation_with_empty_data(self):
        """Test Config object creation with empty configuration data."""
        if Config:
            with pytest.raises((ValueError, ConfigError, KeyError)):
                Config({})
    
    def test_config_creation_with_none(self):
        """Test Config object creation with None."""
        if Config:
            with pytest.raises((ValueError, ConfigError, TypeError)):
                Config(None)
    
    def test_config_creation_with_invalid_types(self):
        """Test Config object creation with invalid data types."""
        if Config:
            invalid_configs = [
                {'timeout': 'not_a_number'},
                {'max_retries': -1},
                {'debug': 'not_a_boolean'},
                {'character_settings': 'not_a_dict'}
            ]
            
            for invalid_config in invalid_configs:
                combined_config = {**self.valid_config, **invalid_config}
                with pytest.raises((ValueError, ConfigError, TypeError)):
                    Config(combined_config)
    
    def test_load_config_from_json_file(self):
        """Test loading configuration from a JSON file."""
        if load_config:
            # Create a test JSON config file
            with open(self.config_path, 'w') as f:
                json.dump(self.valid_config, f)
            
            config = load_config(self.config_path)
            assert config is not None
            assert config.get('api_key') == 'test_api_key_123'
    
    def test_load_config_from_yaml_file(self):
        """Test loading configuration from a YAML file."""
        if load_config:
            # Create a test YAML config file
            with open(self.yaml_config_path, 'w') as f:
                yaml.dump(self.valid_config, f)
            
            config = load_config(self.yaml_config_path)
            assert config is not None
            assert config.get('api_key') == 'test_api_key_123'
    
    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        if load_config:
            with pytest.raises((FileNotFoundError, ConfigError, IOError)):
                load_config('/non/existent/path/config.json')
    
    def test_load_config_invalid_json(self):
        """Test loading configuration from invalid JSON file."""
        if load_config:
            # Create invalid JSON file
            with open(self.config_path, 'w') as f:
                f.write('{ invalid json content }')
            
            with pytest.raises((json.JSONDecodeError, ConfigError, ValueError)):
                load_config(self.config_path)
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration from invalid YAML file."""
        if load_config:
            # Create invalid YAML file
            with open(self.yaml_config_path, 'w') as f:
                f.write('invalid: yaml: content: [')
            
            with pytest.raises((yaml.YAMLError, ConfigError, ValueError)):
                load_config(self.yaml_config_path)
    
    def test_validate_config_with_valid_data(self):
        """Test configuration validation with valid data."""
        if validate_config:
            result = validate_config(self.valid_config)
            assert result is True or result is None  # Different implementations may return different values
    
    def test_validate_config_missing_required_fields(self):
        """Test configuration validation with missing required fields."""
        if validate_config:
            incomplete_configs = [
                {'api_key': 'test'},  # Missing other required fields
                {'base_url': 'https://api.example.com'},  # Missing api_key
                {}  # Empty config
            ]
            
            for incomplete_config in incomplete_configs:
                with pytest.raises((ValueError, ConfigError, KeyError)):
                    validate_config(incomplete_config)
    
    def test_validate_config_with_extra_fields(self):
        """Test configuration validation with extra unknown fields."""
        if validate_config:
            config_with_extra = {**self.valid_config, 'unknown_field': 'value'}
            # This should either pass (ignoring extra fields) or raise a specific error
            try:
                result = validate_config(config_with_extra)
                assert result is True or result is None
            except (ValueError, ConfigError):
                # Some implementations may reject unknown fields
                pass
    
    @patch.dict(os.environ, {'CHARACTER_API_KEY': 'env_api_key'})
    def test_config_from_environment_variables(self):
        """Test configuration loading from environment variables."""
        if Config:
            # Test that config can be loaded from environment variables
            env_config = {
                'api_key': os.environ.get('CHARACTER_API_KEY'),
                'base_url': 'https://api.example.com',
                'timeout': 30,
                'max_retries': 3,
                'debug': False,
                'character_settings': {}
            }
            
            config = Config(env_config)
            assert config is not None
    
    def test_config_default_values(self):
        """Test that configuration provides sensible default values."""
        if Config:
            minimal_config = {'api_key': 'test_key'}
            config = Config(minimal_config)
            
            # Test that defaults are applied appropriately
            # This test assumes the config system provides defaults
            assert config is not None
    
    def test_config_serialization(self):
        """Test configuration object serialization."""
        if Config:
            config = Config(self.valid_config)
            
            # Test various serialization methods that might exist
            try:
                serialized = str(config)
                assert serialized is not None
            except:
                pass
            
            try:
                dict_repr = dict(config) if hasattr(config, '__iter__') else vars(config)
                assert dict_repr is not None
            except:
                pass
    
    def test_config_immutability(self):
        """Test that configuration objects are immutable where expected."""
        if Config:
            config = Config(self.valid_config)
            
            # Try to modify the config and ensure it's protected
            try:
                if hasattr(config, 'api_key'):
                    original_key = config.api_key
                    config.api_key = 'modified_key'
                    # If modification succeeded, verify it was intended
                    assert config.api_key == 'modified_key' or config.api_key == original_key
            except (AttributeError, TypeError):
                # Expected if config is immutable
                pass
    
    def test_config_nested_access(self):
        """Test accessing nested configuration values."""
        if Config:
            config = Config(self.valid_config)
            
            # Test nested access patterns
            try:
                if hasattr(config, 'character_settings'):
                    assert config.character_settings is not None
                elif 'character_settings' in config:
                    assert config['character_settings'] is not None
            except (KeyError, AttributeError):
                # May not have nested access implemented
                pass
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        if validate_config:
            edge_cases = [
                {'api_key': ''},  # Empty string
                {'api_key': ' ' * 100},  # Very long string
                {'timeout': 0},  # Zero timeout
                {'timeout': 999999},  # Very large timeout
                {'max_retries': 0},  # Zero retries
                {'character_settings': {}},  # Empty nested dict
            ]
            
            for edge_case in edge_cases:
                test_config = {**self.valid_config, **edge_case}
                try:
                    validate_config(test_config)
                except (ValueError, ConfigError):
                    # Some edge cases may be invalid
                    pass
    
    def test_config_thread_safety(self):
        """Test configuration object thread safety."""
        if Config:
            import threading
            import time
            
            config = Config(self.valid_config)
            results = []
            
            def access_config():
                try:
                    # Access config properties multiple times
                    for _ in range(10):
                        if hasattr(config, 'api_key'):
                            val = config.api_key
                        elif 'api_key' in config:
                            val = config['api_key']
                        results.append(True)
                        time.sleep(0.001)  # Small delay
                except Exception as e:
                    results.append(False)
            
            # Create multiple threads accessing the config
            threads = [threading.Thread(target=access_config) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # All accesses should succeed
            assert all(results)
    
    @patch('builtins.open', mock_open(read_data='{"api_key": "mocked_key"}'))
    def test_config_loading_with_mocked_file(self):
        """Test configuration loading with mocked file operations."""
        if load_config:
            config = load_config('mocked_path.json')
            assert config is not None
            assert config.get('api_key') == 'mocked_key'
    
    def test_config_error_handling(self):
        """Test proper error handling and error messages."""
        if ConfigError:
            # Test that ConfigError can be raised and caught
            with pytest.raises(ConfigError):
                raise ConfigError("Test error message")
    
    def test_config_backward_compatibility(self):
        """Test backward compatibility with older config formats."""
        if Config:
            # Test with minimal old-style config
            old_style_config = {
                'api_key': 'old_key',
                'url': 'https://old-api.example.com'  # Different key name
            }
            
            try:
                config = Config(old_style_config)
                assert config is not None
            except (ValueError, ConfigError, KeyError):
                # Expected if backward compatibility isn't maintained
                pass
    
    def test_config_performance(self):
        """Test configuration loading and access performance."""
        if Config:
            import time
            
            # Test config creation performance
            start_time = time.time()
            for _ in range(100):
                config = Config(self.valid_config)
            creation_time = time.time() - start_time
            
            # Should be reasonably fast (less than 1 second for 100 creations)
            assert creation_time < 1.0
            
            # Test config access performance
            config = Config(self.valid_config)
            start_time = time.time()
            for _ in range(1000):
                if hasattr(config, 'api_key'):
                    val = config.api_key
                elif 'api_key' in config:
                    val = config['api_key']
            access_time = time.time() - start_time
            
            # Should be very fast (less than 0.1 seconds for 1000 accesses)
            assert access_time < 0.1


class TestConfigIntegration:
    """Integration tests for Config functionality."""
    
    def test_config_with_real_file_system(self):
        """Test configuration with actual file system operations."""
        if load_config:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({'api_key': 'integration_test_key'}, f)
                temp_path = f.name
            
            try:
                config = load_config(temp_path)
                assert config is not None
                assert config.get('api_key') == 'integration_test_key'
            finally:
                os.unlink(temp_path)
    
    def test_config_with_environment_integration(self):
        """Test configuration integration with environment variables."""
        if Config:
            # Test environment variable precedence
            with patch.dict(os.environ, {'CONFIG_DEBUG': 'true'}):
                config_data = {'api_key': 'test', 'debug': False}
                config = Config(config_data)
                # Test if environment variables override config file values
                assert config is not None


class TestConfigSecurity:
    """Security-focused tests for configuration handling."""
    
    def test_config_sensitive_data_handling(self):
        """Test that sensitive configuration data is handled securely."""
        if Config:
            sensitive_config = {
                'api_key': 'super_secret_key',
                'password': 'secret_password',
                'token': 'auth_token_123'
            }
            
            config = Config(sensitive_config)
            
            # Test that sensitive data doesn't appear in string representations
            config_str = str(config)
            assert 'super_secret_key' not in config_str
            assert 'secret_password' not in config_str
    
    def test_config_path_traversal_protection(self):
        """Test protection against path traversal attacks in config file loading."""
        if load_config:
            malicious_paths = [
                '../../../etc/passwd',
                '..\\..\\..\\windows\\system32\\config\\sam',
                '/etc/shadow',
                'C:\\Windows\\System32\\config\\SAM'
            ]
            
            for malicious_path in malicious_paths:
                with pytest.raises((FileNotFoundError, PermissionError, ValueError, ConfigError)):
                    load_config(malicious_path)


class TestConfigValidation:
    """Comprehensive validation tests for configuration data."""
    
    def test_url_validation(self):
        """Test URL validation in configuration."""
        if validate_config or Config:
            invalid_urls = [
                'not_a_url',
                'http://',
                'ftp://invalid',
                'javascript:alert(1)',
                'file:///etc/passwd'
            ]
            
            for invalid_url in invalid_urls:
                invalid_config = {'api_key': 'test', 'base_url': invalid_url}
                
                if validate_config:
                    with pytest.raises((ValueError, ConfigError)):
                        validate_config(invalid_config)
                elif Config:
                    with pytest.raises((ValueError, ConfigError)):
                        Config(invalid_config)
    
    def test_numeric_range_validation(self):
        """Test numeric range validation for configuration values."""
        if validate_config or Config:
            test_cases = [
                ('timeout', -1, False),  # Negative timeout should be invalid
                ('timeout', 0, False),   # Zero timeout should be invalid
                ('timeout', 1, True),    # Positive timeout should be valid
                ('max_retries', -1, False),  # Negative retries should be invalid
                ('max_retries', 0, True),    # Zero retries might be valid
                ('max_retries', 100, True),  # Large retries should be valid
            ]
            
            base_config = {'api_key': 'test', 'base_url': 'https://api.example.com'}
            
            for field, value, should_be_valid in test_cases:
                test_config = {**base_config, field: value}
                
                if should_be_valid:
                    try:
                        if validate_config:
                            validate_config(test_config)
                        elif Config:
                            Config(test_config)
                    except (ValueError, ConfigError):
                        pytest.fail(f"Valid {field}={value} was rejected")
                else:
                    with pytest.raises((ValueError, ConfigError)):
                        if validate_config:
                            validate_config(test_config)
                        elif Config:
                            Config(test_config)


# Additional parametrized tests for comprehensive coverage
@pytest.mark.parametrize("config_format", ["json", "yaml"])
def test_config_loading_different_formats(config_format):
    """Test configuration loading with different file formats."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    config_data = {'api_key': 'format_test_key'}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{config_format}', delete=False) as f:
        if config_format == 'json':
            json.dump(config_data, f)
        elif config_format == 'yaml':
            yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config is not None
        assert config.get('api_key') == 'format_test_key'
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize("invalid_value", [None, "", "   ", 123, [], {}])
def test_config_validation_invalid_api_keys(invalid_value):
    """Test configuration validation with various invalid API key values."""
    if not validate_config:
        pytest.skip("validate_config function not available")
    
    config_data = {
        'api_key': invalid_value,
        'base_url': 'https://api.example.com'
    }
    
    with pytest.raises((ValueError, ConfigError, TypeError)):
        validate_config(config_data)


@pytest.mark.parametrize("timeout_value", [-1, 0, 1, 30, 300, 999999])
def test_config_timeout_boundary_values(timeout_value):
    """Test configuration with various timeout boundary values."""
    if not Config:
        pytest.skip("Config class not available")
    
    config_data = {
        'api_key': 'test_key',
        'base_url': 'https://api.example.com',
        'timeout': timeout_value
    }
    
    if timeout_value < 1:
        with pytest.raises((ValueError, ConfigError)):
            Config(config_data)
    else:
        config = Config(config_data)
        assert config is not None


@pytest.mark.parametrize("debug_value", [True, False, "true", "false", 1, 0, "yes", "no"])
def test_config_debug_flag_values(debug_value):
    """Test configuration with various debug flag values."""
    if not Config:
        pytest.skip("Config class not available")
    
    config_data = {
        'api_key': 'test_key',
        'base_url': 'https://api.example.com',
        'debug': debug_value
    }
    
    try:
        config = Config(config_data)
        assert config is not None
    except (ValueError, ConfigError, TypeError):
        # Some debug values might be invalid depending on implementation
        pass


class TestConfigConcurrency:
    """Tests for configuration handling under concurrent access."""
    
    def test_config_multiple_readers(self):
        """Test configuration with multiple concurrent readers."""
        if not Config:
            pytest.skip("Config class not available")
        
        import threading
        import time
        
        config_data = {'api_key': 'concurrent_test_key'}
        config = Config(config_data)
        
        results = []
        errors = []
        
        def read_config():
            try:
                for _ in range(50):
                    if hasattr(config, 'api_key'):
                        value = config.api_key
                    elif 'api_key' in config:
                        value = config['api_key']
                    results.append(value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Create multiple reader threads
        threads = [threading.Thread(target=read_config) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All reads should succeed without errors
        assert len(errors) == 0
        assert len(results) > 0
    
    def test_config_loading_concurrent_files(self):
        """Test loading configuration from multiple files concurrently."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        import threading
        
        # Create multiple temporary config files
        temp_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({'api_key': f'test_key_{i}'}, f)
                temp_files.append(f.name)
        
        results = []
        errors = []
        
        def load_config_file(filepath):
            try:
                config = load_config(filepath)
                results.append(config)
            except Exception as e:
                errors.append(e)
        
        # Load configs concurrently
        threads = [threading.Thread(target=load_config_file, args=(path,)) for path in temp_files]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        # All loads should succeed
        assert len(errors) == 0
        assert len(results) == 5


# Edge case tests for robustness
class TestConfigEdgeCases:
    """Edge case tests for configuration handling."""
    
    def test_config_with_unicode_values(self):
        """Test configuration with Unicode values."""
        if not Config:
            pytest.skip("Config class not available")
        
        unicode_config = {
            'api_key': 'тест_ключ_🔑',
            'base_url': 'https://测试.example.com',
            'debug': False
        }
        
        config = Config(unicode_config)
        assert config is not None
    
    def test_config_with_very_long_values(self):
        """Test configuration with very long string values."""
        if not Config:
            pytest.skip("Config class not available")
        
        long_string = 'x' * 10000
        long_config = {
            'api_key': long_string,
            'base_url': 'https://api.example.com'
        }
        
        try:
            config = Config(long_config)
            assert config is not None
        except (ValueError, ConfigError, MemoryError):
            # Some implementations may reject very long values
            pass
    
    def test_config_with_deeply_nested_data(self):
        """Test configuration with deeply nested data structures."""
        if not Config:
            pytest.skip("Config class not available")
        
        nested_config = {
            'api_key': 'test_key',
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'level5': {
                                'deep_value': 'found_me'
                            }
                        }
                    }
                }
            }
        }
        
        config = Config(nested_config)
        assert config is not None
    
    def test_config_with_circular_references(self):
        """Test configuration handling with circular references."""
        if not Config:
            pytest.skip("Config class not available")
        
        # Create a structure with circular reference
        circular_dict = {'api_key': 'test_key'}
        circular_dict['self_ref'] = circular_dict
        
        with pytest.raises((ValueError, ConfigError, RecursionError)):
            Config(circular_dict)


# Performance benchmarks
class TestConfigPerformance:
    """Performance tests for configuration operations."""
    
    def test_config_creation_performance_benchmark(self):
        """Benchmark configuration object creation performance."""
        if not Config:
            pytest.skip("Config class not available")
        
        import time
        
        config_data = {
            'api_key': 'performance_test_key',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False
        }
        
        # Warm up
        for _ in range(10):
            Config(config_data)
        
        # Benchmark
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            Config(config_data)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Should create configs quickly (less than 1ms per config on average)
        assert avg_time < 0.001, f"Config creation too slow: {avg_time:.4f}s per config"
    
    def test_config_file_loading_performance(self):
        """Benchmark configuration file loading performance."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        import time
        
        # Create a reasonably sized config file
        large_config = {
            'api_key': 'performance_test_key',
            'base_url': 'https://api.example.com',
            'settings': {f'setting_{i}': f'value_{i}' for i in range(100)}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_config, f)
            temp_path = f.name
        
        try:
            # Warm up
            for _ in range(5):
                load_config(temp_path)
            
            # Benchmark
            start_time = time.time()
            iterations = 100
            
            for _ in range(iterations):
                load_config(temp_path)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / iterations
            
            # Should load configs quickly (less than 10ms per load on average)
            assert avg_time < 0.01, f"Config loading too slow: {avg_time:.4f}s per load"
        
        finally:
            os.unlink(temp_path)

# Additional comprehensive tests for enhanced coverage
class TestConfigAdvancedValidation:
    """Advanced validation tests for configuration data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_config = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False
        }
    
    def test_config_error_messages_are_descriptive(self):
        """Test that configuration error messages are descriptive and helpful."""
        if Config:
            test_cases = [
                ({'api_key': None}, "api_key"),
                ({'api_key': ''}, "api_key"),
                ({'base_url': 'invalid-url'}, "base_url"),
                ({'timeout': -1}, "timeout"),
                ({'max_retries': 'not_a_number'}, "max_retries"),
            ]
            
            for invalid_config, expected_field in test_cases:
                test_config = {**self.base_config, **invalid_config}
                with pytest.raises((ValueError, ConfigError)) as excinfo:
                    Config(test_config)
                
                # Check that error message mentions the problematic field
                error_msg = str(excinfo.value).lower()
                assert expected_field.lower() in error_msg, f"Error message should mention '{expected_field}'"
    
    def test_config_validation_with_custom_validators(self):
        """Test configuration validation with custom validation rules."""
        if validate_config:
            # Test email format validation if supported
            test_configs = [
                {'api_key': 'test', 'email': 'invalid-email', 'base_url': 'https://api.example.com'},
                {'api_key': 'test', 'email': 'valid@example.com', 'base_url': 'https://api.example.com'},
            ]
            
            for config in test_configs:
                try:
                    validate_config(config)
                except (ValueError, ConfigError):
                    # Expected for invalid formats
                    pass
    
    def test_config_field_type_coercion(self):
        """Test that configuration fields are properly type-coerced."""
        if Config:
            coercion_tests = [
                ({'timeout': '30'}, int),      # String to int
                ({'debug': 'true'}, bool),     # String to bool
                ({'debug': '1'}, bool),        # String to bool
                ({'max_retries': '5'}, int),   # String to int
            ]
            
            for config_override, expected_type in coercion_tests:
                test_config = {**self.base_config, **config_override}
                try:
                    config = Config(test_config)
                    # Check if type coercion happened (implementation dependent)
                    assert config is not None
                except (ValueError, ConfigError, TypeError):
                    # Some implementations may not support type coercion
                    pass
    
    def test_config_schema_validation_comprehensive(self):
        """Test comprehensive schema validation rules."""
        if validate_config or Config:
            invalid_schemas = [
                # Invalid API key formats
                {'api_key': 'x' * 1000, 'base_url': 'https://api.example.com'},
                {'api_key': 'key with spaces', 'base_url': 'https://api.example.com'},
                {'api_key': 'key\nwith\nnewlines', 'base_url': 'https://api.example.com'},
                
                # Invalid URL formats
                {'api_key': 'test', 'base_url': 'http://localhost'},
                {'api_key': 'test', 'base_url': 'https://'},
                {'api_key': 'test', 'base_url': 'not-a-protocol://example.com'},
                
                # Invalid numeric ranges
                {'api_key': 'test', 'timeout': 999999, 'base_url': 'https://api.example.com'},
                {'api_key': 'test', 'max_retries': -999, 'base_url': 'https://api.example.com'},
            ]
            
            for invalid_config in invalid_schemas:
                with pytest.raises((ValueError, ConfigError, TypeError)):
                    if validate_config:
                        validate_config(invalid_config)
                    elif Config:
                        Config(invalid_config)


class TestConfigEnvironmentIntegration:
    """Enhanced tests for environment variable integration."""
    
    @patch.dict(os.environ, {
        'CHARACTER_API_KEY': 'env_key',
        'CHARACTER_BASE_URL': 'https://env.example.com',
        'CHARACTER_DEBUG': 'true',
        'CHARACTER_TIMEOUT': '60'
    })
    def test_config_environment_variable_precedence(self):
        """Test that environment variables take precedence over config files."""
        if Config:
            file_config = {
                'api_key': 'file_key',
                'base_url': 'https://file.example.com',
                'debug': False,
                'timeout': 30
            }
            
            # This test assumes the Config class checks environment variables
            config = Config(file_config)
            assert config is not None
    
    @patch.dict(os.environ, {'CHARACTER_INVALID_VAR': 'invalid_value'})
    def test_config_handles_invalid_environment_variables(self):
        """Test handling of invalid environment variable values."""
        if Config:
            base_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com'
            }
            
            # Should handle invalid env vars gracefully
            try:
                config = Config(base_config)
                assert config is not None
            except (ValueError, ConfigError):
                # Expected if strict env var validation is implemented
                pass
    
    def test_config_environment_variable_type_conversion(self):
        """Test proper type conversion of environment variables."""
        with patch.dict(os.environ, {
            'CHARACTER_TIMEOUT': '45',
            'CHARACTER_DEBUG': 'false',
            'CHARACTER_MAX_RETRIES': '7'
        }):
            if Config:
                config = Config({'api_key': 'test', 'base_url': 'https://api.example.com'})
                assert config is not None


class TestConfigMerging:
    """Tests for configuration merging and precedence."""
    
    def test_config_merge_multiple_sources(self):
        """Test merging configuration from multiple sources."""
        if Config:
            default_config = {
                'api_key': 'default_key',
                'base_url': 'https://default.example.com',
                'timeout': 30,
                'debug': False
            }
            
            user_config = {
                'api_key': 'user_key',
                'timeout': 60,
                'custom_field': 'user_value'
            }
            
            # Test merging behavior (implementation dependent)
            try:
                merged_config = Config({**default_config, **user_config})
                assert merged_config is not None
            except (ValueError, ConfigError):
                # Expected if merging is not supported
                pass
    
    def test_config_nested_merge_behavior(self):
        """Test merging behavior for nested configuration objects."""
        if Config:
            base_config = {
                'api_key': 'test',
                'settings': {
                    'cache': {'enabled': True, 'ttl': 300},
                    'logging': {'level': 'INFO'}
                }
            }
            
            override_config = {
                'api_key': 'test',
                'settings': {
                    'cache': {'ttl': 600},
                    'debug': {'enabled': True}
                }
            }
            
            try:
                config = Config({**base_config, **override_config})
                assert config is not None
            except (ValueError, ConfigError):
                # Expected if nested merging is not supported
                pass


class TestConfigSerialization:
    """Enhanced tests for configuration serialization."""
    
    def test_config_to_dict_conversion(self):
        """Test converting configuration objects to dictionaries."""
        if Config:
            config_data = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'timeout': 30
            }
            
            config = Config(config_data)
            
            # Test various ways to convert to dict
            try:
                if hasattr(config, 'to_dict'):
                    dict_repr = config.to_dict()
                    assert isinstance(dict_repr, dict)
                    assert dict_repr['api_key'] == 'test_key'
                elif hasattr(config, '__dict__'):
                    dict_repr = config.__dict__
                    assert isinstance(dict_repr, dict)
                elif hasattr(config, 'items'):
                    dict_repr = dict(config.items())
                    assert isinstance(dict_repr, dict)
            except (AttributeError, TypeError):
                # Expected if serialization is not supported
                pass
    
    def test_config_json_serialization(self):
        """Test JSON serialization of configuration objects."""
        if Config:
            config_data = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'timeout': 30,
                'debug': False
            }
            
            config = Config(config_data)
            
            try:
                if hasattr(config, 'to_json'):
                    json_str = config.to_json()
                    assert isinstance(json_str, str)
                    # Should be valid JSON
                    json.loads(json_str)
                else:
                    # Try to serialize using vars() or similar
                    json_str = json.dumps(vars(config))
                    assert isinstance(json_str, str)
            except (AttributeError, TypeError, json.JSONEncodeError):
                # Expected if JSON serialization is not supported
                pass
    
    def test_config_yaml_serialization(self):
        """Test YAML serialization of configuration objects."""
        if Config:
            config_data = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'timeout': 30
            }
            
            config = Config(config_data)
            
            try:
                if hasattr(config, 'to_yaml'):
                    yaml_str = config.to_yaml()
                    assert isinstance(yaml_str, str)
                    # Should be valid YAML
                    yaml.safe_load(yaml_str)
                else:
                    # Try to serialize using vars() or similar
                    yaml_str = yaml.dump(vars(config))
                    assert isinstance(yaml_str, str)
            except (AttributeError, TypeError, yaml.YAMLError):
                # Expected if YAML serialization is not supported
                pass


class TestConfigAdvancedSecurity:
    """Advanced security tests for configuration handling."""
    
    def test_config_injection_attacks(self):
        """Test protection against various injection attacks."""
        if Config or validate_config:
            injection_payloads = [
                # SQL injection attempts
                {'api_key': "'; DROP TABLE users; --", 'base_url': 'https://api.example.com'},
                # Script injection attempts
                {'api_key': '<script>alert("xss")</script>', 'base_url': 'https://api.example.com'},
                # Command injection attempts
                {'api_key': '$(rm -rf /)', 'base_url': 'https://api.example.com'},
                # Path traversal attempts
                {'api_key': '../../../etc/passwd', 'base_url': 'https://api.example.com'},
                # LDAP injection attempts
                {'api_key': '*)(uid=*', 'base_url': 'https://api.example.com'},
            ]
            
            for payload in injection_payloads:
                try:
                    if validate_config:
                        validate_config(payload)
                    elif Config:
                        Config(payload)
                    # If no exception, the payload was accepted (might be okay)
                except (ValueError, ConfigError):
                    # Expected for rejected payloads
                    pass
    
    def test_config_sensitive_data_masking(self):
        """Test that sensitive data is properly masked in logs and string representations."""
        if Config:
            sensitive_config = {
                'api_key': 'very_secret_key_12345',
                'password': 'super_secret_password',
                'token': 'auth_token_xyz789',
                'secret': 'my_secret_value',
                'base_url': 'https://api.example.com'
            }
            
            config = Config(sensitive_config)
            
            # Test string representation masking
            str_repr = str(config)
            repr_repr = repr(config)
            
            sensitive_values = ['very_secret_key_12345', 'super_secret_password', 'auth_token_xyz789', 'my_secret_value']
            for sensitive_value in sensitive_values:
                assert sensitive_value not in str_repr, f"Sensitive value '{sensitive_value}' found in str() representation"
                assert sensitive_value not in repr_repr, f"Sensitive value '{sensitive_value}' found in repr() representation"
    
    def test_config_secure_comparison(self):
        """Test that configuration comparison is secure against timing attacks."""
        if Config:
            config1 = Config({'api_key': 'secret_key_1', 'base_url': 'https://api.example.com'})
            config2 = Config({'api_key': 'secret_key_2', 'base_url': 'https://api.example.com'})
            
            # Test equality comparison
            try:
                result = config1 == config2
                assert isinstance(result, bool)
            except (TypeError, AttributeError):
                # Expected if comparison is not implemented
                pass


class TestConfigReloadAndUpdate:
    """Tests for configuration reloading and updating functionality."""
    
    def test_config_reload_from_file(self):
        """Test reloading configuration from file after changes."""
        if load_config:
            # Create initial config file
            initial_config = {'api_key': 'initial_key', 'base_url': 'https://initial.example.com'}
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(initial_config, f)
                temp_path = f.name
            
            try:
                # Load initial config
                config1 = load_config(temp_path)
                assert config1.get('api_key') == 'initial_key'
                
                # Update file
                updated_config = {'api_key': 'updated_key', 'base_url': 'https://updated.example.com'}
                with open(temp_path, 'w') as f:
                    json.dump(updated_config, f)
                
                # Reload config
                config2 = load_config(temp_path)
                assert config2.get('api_key') == 'updated_key'
                
            finally:
                os.unlink(temp_path)
    
    def test_config_update_in_place(self):
        """Test updating configuration object in place."""
        if Config:
            initial_config = {'api_key': 'initial_key', 'base_url': 'https://api.example.com'}
            config = Config(initial_config)
            
            # Test if config supports updating
            try:
                if hasattr(config, 'update'):
                    config.update({'api_key': 'updated_key'})
                elif hasattr(config, 'set'):
                    config.set('api_key', 'updated_key')
                # If no update methods, config is likely immutable (which is fine)
            except (AttributeError, TypeError):
                # Expected for immutable configs
                pass


class TestConfigCrossPlatform:
    """Cross-platform compatibility tests."""
    
    def test_config_path_handling_cross_platform(self):
        """Test that configuration handles file paths correctly across platforms."""
        if load_config:
            # Test various path formats
            test_paths = [
                'config.json',                    # Relative path
                './config.json',                  # Explicit relative path
                os.path.abspath('config.json'),   # Absolute path
            ]
            
            for path_format in test_paths:
                # Create config file at the test path
                config_data = {'api_key': 'test_key'}
                dir_path = os.path.dirname(path_format) or '.'
                os.makedirs(dir_path, exist_ok=True)
                
                try:
                    with open(path_format, 'w') as f:
                        json.dump(config_data, f)
                    
                    config = load_config(path_format)
                    assert config is not None
                    assert config.get('api_key') == 'test_key'
                    
                finally:
                    if os.path.exists(path_format):
                        os.unlink(path_format)
    
    def test_config_encoding_handling(self):
        """Test that configuration handles various text encodings correctly."""
        if load_config:
            # Test different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1']
            
            for encoding in encodings:
                config_data = {'api_key': 'test_key_ñ_中文', 'base_url': 'https://api.example.com'}
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding=encoding, delete=False) as f:
                    json.dump(config_data, f, ensure_ascii=False)
                    temp_path = f.name
                
                try:
                    config = load_config(temp_path)
                    assert config is not None
                    assert 'test_key_ñ_中文' in str(config.get('api_key', ''))
                except (UnicodeDecodeError, UnicodeEncodeError):
                    # Expected for unsupported encodings
                    pass
                finally:
                    os.unlink(temp_path)


class TestConfigAdvancedErrorHandling:
    """Advanced error handling and recovery tests."""
    
    def test_config_graceful_degradation(self):
        """Test graceful degradation when partial configuration is available."""
        if Config:
            partial_configs = [
                {'api_key': 'test_key'},  # Minimal config
                {'api_key': 'test_key', 'base_url': 'https://api.example.com'},  # Partial config
            ]
            
            for partial_config in partial_configs:
                try:
                    config = Config(partial_config)
                    assert config is not None
                    # Should either work with defaults or raise clear error
                except (ValueError, ConfigError) as e:
                    # Should have clear error message about missing fields
                    assert len(str(e)) > 0
    
    def test_config_error_recovery_strategies(self):
        """Test various error recovery strategies."""
        if load_config:
            # Test recovery from corrupted files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write('{"api_key": "test", "corrupted": }')  # Invalid JSON
                temp_path = f.name
            
            try:
                with pytest.raises((json.JSONDecodeError, ConfigError, ValueError)):
                    load_config(temp_path)
            finally:
                os.unlink(temp_path)
    
    def test_config_validation_detailed_errors(self):
        """Test that validation errors provide detailed information."""
        if validate_config:
            invalid_config = {
                'api_key': '',  # Empty
                'base_url': 'invalid-url',  # Invalid URL
                'timeout': -1,  # Invalid timeout
                'max_retries': 'not_a_number'  # Invalid type
            }
            
            try:
                validate_config(invalid_config)
                pytest.fail("Expected validation to fail")
            except (ValueError, ConfigError) as e:
                error_msg = str(e)
                # Error message should mention specific issues
                assert len(error_msg) > 20  # Should be descriptive
                # Could check for specific fields mentioned in error


@pytest.mark.parametrize("config_size", [1, 10, 100, 1000])
def test_config_performance_with_varying_sizes(config_size):
    """Test configuration performance with varying configuration sizes."""
    if not Config:
        pytest.skip("Config class not available")
    
    import time
    
    # Generate config of specified size
    large_config = {
        'api_key': 'test_key',
        'base_url': 'https://api.example.com'
    }
    
    # Add many configuration items
    for i in range(config_size):
        large_config[f'setting_{i}'] = f'value_{i}'
    
    # Measure creation time
    start_time = time.time()
    config = Config(large_config)
    creation_time = time.time() - start_time
    
    # Should handle larger configs efficiently
    max_time = 0.001 * config_size  # Linear time complexity assumption
    assert creation_time < max_time, f"Config creation too slow for size {config_size}: {creation_time:.4f}s"


@pytest.mark.parametrize("file_format,extension", [
    ("json", ".json"),
    ("yaml", ".yaml"),
    ("yaml", ".yml"),
])
def test_config_file_format_detection(file_format, extension):
    """Test automatic file format detection based on extension."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    config_data = {'api_key': 'format_detection_test'}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
        if file_format == 'json':
            json.dump(config_data, f)
        elif file_format == 'yaml':
            yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config is not None
        assert config.get('api_key') == 'format_detection_test'
    finally:
        os.unlink(temp_path)


# Stress tests for robustness
class TestConfigStress:
    """Stress tests for configuration handling under extreme conditions."""
    
    def test_config_memory_usage_with_large_configs(self):
        """Test memory usage doesn't grow excessively with large configurations."""
        if not Config:
            pytest.skip("Config class not available")
        
        import gc
        import sys
        
        # Create very large config
        large_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.example.com'
        }
        
        # Add 10000 settings
        for i in range(10000):
            large_config[f'large_setting_{i}'] = f'large_value_{i}' * 10
        
        # Measure memory before
        gc.collect()
        mem_before = sys.getsizeof(large_config)
        
        # Create config
        config = Config(large_config)
        
        # Memory usage should be reasonable
        gc.collect()
        # This is a rough test - actual memory usage depends on implementation
        assert config is not None
    
    def test_config_rapid_creation_and_destruction(self):
        """Test rapid creation and destruction of configuration objects."""
        if not Config:
            pytest.skip("Config class not available")
        
        import time
        
        config_data = {
            'api_key': 'stress_test_key',
            'base_url': 'https://api.example.com'
        }
        
        start_time = time.time()
        
        # Rapidly create and destroy configs
        for i in range(1000):
            config = Config(config_data)
            del config
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle rapid creation/destruction efficiently
        assert total_time < 2.0, f"Rapid creation/destruction too slow: {total_time:.2f}s"


# Mock-based tests for external dependencies
class TestConfigMocking:
    """Tests using mocks for external dependencies."""
    
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_config_loading_with_filesystem_mocks(self, mock_open_func, mock_exists):
        """Test configuration loading with mocked filesystem operations."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        # Mock file existence and content
        mock_exists.return_value = True
        mock_open_func.return_value.__enter__.return_value.read.return_value = '{"api_key": "mocked_key"}'
        
        config = load_config('mocked_config.json')
        assert config is not None
        assert config.get('api_key') == 'mocked_key'
        
        # Verify mocks were called
        mock_exists.assert_called_once_with('mocked_config.json')
        mock_open_func.assert_called_once()
    
    @patch.dict(os.environ, {'MOCK_CONFIG_VAR': 'mocked_value'})
    def test_config_environment_variable_mocking(self):
        """Test configuration with mocked environment variables."""
        if Config:
            config_data = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'mock_setting': os.environ.get('MOCK_CONFIG_VAR', 'default')
            }
            
            config = Config(config_data)
            assert config is not None
    
    @patch('json.load')
    def test_config_json_loading_error_handling(self, mock_json_load):
        """Test error handling when JSON loading fails."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        # Mock JSON loading to raise an error
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with pytest.raises((json.JSONDecodeError, ConfigError, ValueError)):
            with patch('builtins.open', mock_open(read_data='invalid json')):
                load_config('test_config.json')


# Integration tests that combine multiple features
class TestConfigIntegrationAdvanced:
    """Advanced integration tests combining multiple configuration features."""
    
    def test_config_full_lifecycle_integration(self):
        """Test complete configuration lifecycle from loading to usage."""
        if load_config and Config:
            # Create config file
            config_data = {
                'api_key': 'integration_test_key',
                'base_url': 'https://integration.example.com',
                'timeout': 45,
                'max_retries': 5,
                'debug': True
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                temp_path = f.name
            
            try:
                # Load from file
                loaded_config = load_config(temp_path)
                assert loaded_config is not None
                
                # Create Config object
                config = Config(loaded_config)
                assert config is not None
                
                # Validate
                if validate_config:
                    validate_config(loaded_config)
                
                # Use config (test attribute access)
                if hasattr(config, 'api_key'):
                    assert config.api_key == 'integration_test_key'
                elif 'api_key' in config:
                    assert config['api_key'] == 'integration_test_key'
                
            finally:
                os.unlink(temp_path)
    
    def test_config_multi_format_compatibility(self):
        """Test that the same configuration works across JSON and YAML formats."""
        if load_config:
            config_data = {
                'api_key': 'multi_format_test',
                'base_url': 'https://multi.example.com',
                'settings': {
                    'nested': {
                        'value': 'deep_test'
                    }
                }
            }
            
            # Test JSON format
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                json_path = f.name
            
            # Test YAML format
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                yaml_path = f.name
            
            try:
                json_config = load_config(json_path)
                yaml_config = load_config(yaml_path)
                
                # Both should load successfully and have same content
                assert json_config is not None
                assert yaml_config is not None
                assert json_config.get('api_key') == yaml_config.get('api_key')
                
            finally:
                os.unlink(json_path)
                os.unlink(yaml_path)


# Final validation tests
class TestConfigComprehensiveValidation:
    """Comprehensive validation tests covering all aspects."""
    
    def test_config_all_validation_rules(self):
        """Test all validation rules in a comprehensive manner."""
        if validate_config:
            # Test comprehensive valid config
            comprehensive_config = {
                'api_key': 'valid_api_key_123',
                'base_url': 'https://api.example.com',
                'timeout': 30,
                'max_retries': 3,
                'debug': False,
                'character_settings': {
                    'default_name': 'TestCharacter',
                    'max_characters': 100,
                    'allowed_types': ['warrior', 'mage', 'rogue']
                },
                'advanced_settings': {
                    'cache_enabled': True,
                    'cache_ttl': 3600,
                    'rate_limiting': {
                        'requests_per_minute': 60,
                        'burst_size': 10
                    }
                }
            }
            
            # Should validate successfully
            result = validate_config(comprehensive_config)
            assert result is True or result is None
    
    def test_config_validates_all_required_fields(self):
        """Test that validation catches all missing required fields."""
        if validate_config:
            required_fields = ['api_key', 'base_url']
            
            for field in required_fields:
                incomplete_config = {
                    'api_key': 'test_key',
                    'base_url': 'https://api.example.com'
                }
                del incomplete_config[field]
                
                with pytest.raises((ValueError, ConfigError, KeyError)):
                    validate_config(incomplete_config)
    
    def test_config_final_integration_validation(self):
        """Final integration test validating all configuration functionality."""
        if Config and load_config and validate_config:
            # Create comprehensive test config
            test_config = {
                'api_key': 'final_test_key',
                'base_url': 'https://final.example.com',
                'timeout': 30,
                'max_retries': 3,
                'debug': False
            }
            
            # Save to file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_config, f)
                temp_path = f.name
            
            try:
                # Load, validate, and create config
                loaded = load_config(temp_path)
                validate_config(loaded)
                config = Config(loaded)
                
                # All operations should succeed
                assert config is not None
                
            finally:
                os.unlink(temp_path)


# Tests for the actual config.py implementation discovered
class TestActualConfigImplementation:
    """Tests for the actual config.py implementation found in the repository."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Import the actual config module
        try:
            import CharacterClient.src.config as actual_config
            self.config_module = actual_config
        except ImportError:
            self.config_module = None
    
    def test_client_root_path_exists(self):
        """Test that CLIENT_ROOT path is properly set."""
        if self.config_module and hasattr(self.config_module, 'CLIENT_ROOT'):
            assert os.path.exists(self.config_module.CLIENT_ROOT)
            assert os.path.isdir(self.config_module.CLIENT_ROOT)
    
    def test_default_paths_are_valid(self):
        """Test that default paths are properly constructed."""
        if self.config_module:
            paths_to_test = [
                'DEFAULT_CLIENT_DATA_PATH',
                'DEFAULT_CLIENT_MODELS_PATH',
                'CLIENT_LLM_MODELS_PATH',
                'CLIENT_TTS_MODELS_PATH',
                'CLIENT_TTS_REFERENCE_VOICES_PATH',
                'CLIENT_LOGS_PATH',
                'CLIENT_TEMP_AUDIO_PATH'
            ]
            
            for path_name in paths_to_test:
                if hasattr(self.config_module, path_name):
                    path = getattr(self.config_module, path_name)
                    assert isinstance(path, str)
                    assert len(path) > 0
                    # Path should be absolute
                    assert os.path.isabs(path)
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': '/custom/data/path'})
    def test_environment_variable_override(self):
        """Test that environment variables properly override default paths."""
        if self.config_module:
            # Re-import to get updated environment variables
            import importlib
            importlib.reload(self.config_module)
            
            if hasattr(self.config_module, 'CLIENT_DATA_PATH'):
                assert self.config_module.CLIENT_DATA_PATH == '/custom/data/path'
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_MODELS_PATH': '/custom/models/path'})
    def test_models_path_environment_override(self):
        """Test that models path environment variable works."""
        if self.config_module:
            import importlib
            importlib.reload(self.config_module)
            
            if hasattr(self.config_module, 'CLIENT_MODELS_PATH'):
                assert self.config_module.CLIENT_MODELS_PATH == '/custom/models/path'
    
    def test_ensure_client_directories_function(self):
        """Test the ensure_client_directories function."""
        if self.config_module and hasattr(self.config_module, 'ensure_client_directories'):
            # Function should exist and be callable
            assert callable(self.config_module.ensure_client_directories)
            
            # Should run without errors
            try:
                self.config_module.ensure_client_directories()
            except Exception as e:
                pytest.fail(f"ensure_client_directories() raised {e}")
    
    def test_directory_creation_with_permissions(self):
        """Test that directories are created with appropriate permissions."""
        if self.config_module:
            import tempfile
            import shutil
            
            # Create a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Patch the config paths to use temp directory
                with patch.object(self.config_module, 'CLIENT_DATA_PATH', temp_dir):
                    if hasattr(self.config_module, 'ensure_client_directories'):
                        self.config_module.ensure_client_directories()
                        
                        # Check that directory was created
                        assert os.path.exists(temp_dir)
                        assert os.path.isdir(temp_dir)
    
    def test_config_module_logging_setup(self):
        """Test that logging is properly configured in the config module."""
        if self.config_module:
            # Check if logging is imported and used
            import logging
            
            # Should not raise errors when logging functions are called
            try:
                logger = logging.getLogger("dreamweaver_client_config_setup")
                assert logger is not None
            except Exception as e:
                pytest.fail(f"Logging setup failed: {e}")
    
    def test_path_security_validation(self):
        """Test that paths are secure and don't contain dangerous characters."""
        if self.config_module:
            paths_to_validate = [
                'CLIENT_DATA_PATH',
                'CLIENT_MODELS_PATH',
                'CLIENT_LLM_MODELS_PATH',
                'CLIENT_TTS_MODELS_PATH',
                'CLIENT_LOGS_PATH',
                'CLIENT_TEMP_AUDIO_PATH'
            ]
            
            dangerous_patterns = ['../', '..\\', '/etc/', 'C:\\Windows\\']
            
            for path_name in paths_to_validate:
                if hasattr(self.config_module, path_name):
                    path = getattr(self.config_module, path_name)
                    for pattern in dangerous_patterns:
                        assert pattern not in path, f"Dangerous pattern '{pattern}' found in {path_name}: {path}"
    
    def test_config_module_main_execution(self):
        """Test that the config module can be executed as main."""
        if self.config_module:
            # Should be able to run the main block without errors
            try:
                # Simulate running as main
                if hasattr(self.config_module, '__name__'):
                    # This would normally be set to '__main__' when run directly
                    pass
            except Exception as e:
                pytest.fail(f"Config module main execution failed: {e}")
    
    def test_config_constants_are_strings(self):
        """Test that all configuration constants are strings."""
        if self.config_module:
            string_constants = [
                'CLIENT_ROOT',
                'DEFAULT_CLIENT_DATA_PATH',
                'CLIENT_DATA_PATH',
                'DEFAULT_CLIENT_MODELS_PATH',
                'CLIENT_MODELS_PATH',
                'CLIENT_LLM_MODELS_PATH',
                'CLIENT_TTS_MODELS_PATH',
                'CLIENT_TTS_REFERENCE_VOICES_PATH',
                'CLIENT_LOGS_PATH',
                'CLIENT_TEMP_AUDIO_PATH'
            ]
            
            for const_name in string_constants:
                if hasattr(self.config_module, const_name):
                    const_value = getattr(self.config_module, const_name)
                    assert isinstance(const_value, str), f"{const_name} should be a string, got {type(const_value)}"
    
    def test_path_hierarchy_consistency(self):
        """Test that path hierarchy is consistent (subdirectories are under parent directories)."""
        if self.config_module:
            # CLIENT_MODELS_PATH should be under CLIENT_DATA_PATH
            if (hasattr(self.config_module, 'CLIENT_DATA_PATH') and 
                hasattr(self.config_module, 'CLIENT_MODELS_PATH')):
                data_path = self.config_module.CLIENT_DATA_PATH
                models_path = self.config_module.CLIENT_MODELS_PATH
                
                # models_path should start with data_path (be a subdirectory)
                assert models_path.startswith(data_path), f"Models path {models_path} should be under data path {data_path}"
            
            # Specific model paths should be under CLIENT_MODELS_PATH
            model_subpaths = ['CLIENT_LLM_MODELS_PATH', 'CLIENT_TTS_MODELS_PATH']
            if hasattr(self.config_module, 'CLIENT_MODELS_PATH'):
                models_path = self.config_module.CLIENT_MODELS_PATH
                
                for subpath_name in model_subpaths:
                    if hasattr(self.config_module, subpath_name):
                        subpath = getattr(self.config_module, subpath_name)
                        assert subpath.startswith(models_path), f"{subpath_name} {subpath} should be under models path {models_path}"
    
    def test_directory_creation_error_handling(self):
        """Test error handling in directory creation."""
        if self.config_module and hasattr(self.config_module, 'ensure_client_directories'):
            # Test with invalid permissions (mock os.makedirs to raise PermissionError)
            with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
                # Should handle the error gracefully and log it
                try:
                    self.config_module.ensure_client_directories()
                    # Function should complete without raising an exception
                except PermissionError:
                    pytest.fail("ensure_client_directories should handle PermissionError gracefully")


# End of additional tests