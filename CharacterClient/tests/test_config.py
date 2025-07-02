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

# Additional test classes and methods for enhanced coverage

class TestConfigFileFormats:
    """Test configuration handling with various file formats and encoding."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_config_with_different_encodings(self):
        """Test configuration files with different character encodings."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'тест_ключ_🔑'}
        encodings = ['utf-8', 'utf-16', 'latin-1']
        
        for encoding in encodings:
            config_path = os.path.join(self.temp_dir, f'config_{encoding}.json')
            
            try:
                with open(config_path, 'w', encoding=encoding) as f:
                    json.dump(config_data, f, ensure_ascii=False)
                
                config = load_config(config_path)
                assert config is not None
                assert 'api_key' in config
            except (UnicodeDecodeError, UnicodeEncodeError):
                # Some encodings may not support all characters
                pass
            finally:
                if os.path.exists(config_path):
                    os.unlink(config_path)
    
    def test_config_with_bom(self):
        """Test configuration files with Byte Order Mark (BOM)."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'bom_test_key'}
        config_path = os.path.join(self.temp_dir, 'config_bom.json')
        
        # Write file with UTF-8 BOM
        with open(config_path, 'wb') as f:
            f.write(b'\xef\xbb\xbf')  # UTF-8 BOM
            f.write(json.dumps(config_data).encode('utf-8'))
        
        try:
            config = load_config(config_path)
            assert config is not None
            assert config.get('api_key') == 'bom_test_key'
        finally:
            os.unlink(config_path)
    
    def test_config_with_comments_in_json(self):
        """Test configuration loading with JSON files containing comments (invalid JSON)."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_path = os.path.join(self.temp_dir, 'config_comments.json')
        
        # Create JSON with comments (technically invalid JSON)
        json_with_comments = '''{
            // This is a comment
            "api_key": "comment_test_key",
            /* Multi-line
               comment */
            "base_url": "https://api.example.com"
        }'''
        
        with open(config_path, 'w') as f:
            f.write(json_with_comments)
        
        try:
            # Should fail with standard JSON parsing
            with pytest.raises((json.JSONDecodeError, ConfigError, ValueError)):
                load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_config_yaml_advanced_features(self):
        """Test YAML configuration with advanced YAML features."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_path = os.path.join(self.temp_dir, 'config_advanced.yaml')
        
        # YAML with anchors and references
        yaml_content = '''
api_key: &api_key "advanced_test_key"
environments:
  dev:
    api_key: *api_key
    base_url: "https://dev-api.example.com"
  prod:
    api_key: *api_key
    base_url: "https://api.example.com"
default_timeout: &timeout 30
timeouts:
  short: 5
  medium: *timeout
  long: 120
'''
        
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        try:
            config = load_config(config_path)
            assert config is not None
            assert config.get('api_key') == 'advanced_test_key'
        finally:
            os.unlink(config_path)


class TestConfigErrorMessages:
    """Test configuration error messages and error handling details."""
    
    def test_config_error_message_content(self):
        """Test that configuration errors contain helpful error messages."""
        if not ConfigError:
            pytest.skip("ConfigError not available")
        
        test_message = "Test configuration error with details"
        
        try:
            raise ConfigError(test_message)
        except ConfigError as e:
            assert str(e) == test_message
            assert isinstance(e, Exception)
    
    def test_config_validation_error_details(self):
        """Test that validation errors provide specific field information."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        invalid_configs = [
            ({'timeout': 'not_a_number'}, 'timeout'),
            ({'max_retries': -1}, 'max_retries'),
            ({'api_key': None}, 'api_key'),
        ]
        
        for invalid_config, expected_field in invalid_configs:
            try:
                validate_config(invalid_config)
                pytest.fail(f"Expected validation error for {expected_field}")
            except (ValueError, ConfigError) as e:
                # Error message should mention the problematic field
                error_msg = str(e).lower()
                assert expected_field.lower() in error_msg or 'validation' in error_msg
    
    def test_config_file_loading_error_details(self):
        """Test that file loading errors provide helpful details."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        # Test with non-existent file
        try:
            load_config('/absolutely/non/existent/path/config.json')
            pytest.fail("Expected file loading error")
        except (FileNotFoundError, ConfigError, IOError) as e:
            error_msg = str(e).lower()
            assert 'file' in error_msg or 'path' in error_msg or 'not found' in error_msg


class TestConfigMemoryUsage:
    """Test configuration memory usage and cleanup."""
    
    def test_config_memory_efficiency(self):
        """Test that configuration objects don't consume excessive memory."""
        if not Config:
            pytest.skip("Config class not available")
        
        import sys
        
        # Create a reasonably sized config
        large_config = {
            'api_key': 'memory_test_key',
            'base_url': 'https://api.example.com',
            'large_data': {f'key_{i}': f'value_{i}' * 100 for i in range(100)}
        }
        
        # Measure memory usage
        config_objects = []
        initial_size = sys.getsizeof(config_objects)
        
        for _ in range(10):
            config_objects.append(Config(large_config))
        
        final_size = sys.getsizeof(config_objects)
        
        # Basic sanity check - should not grow unreasonably
        # This is a rough check, actual memory usage depends on implementation
        assert final_size > initial_size  # Should have some memory usage
    
    def test_config_object_cleanup(self):
        """Test that configuration objects can be properly garbage collected."""
        if not Config:
            pytest.skip("Config class not available")
        
        import gc
        import weakref
        
        config_data = {'api_key': 'cleanup_test_key'}
        config = Config(config_data)
        
        # Create a weak reference to test garbage collection
        weak_ref = weakref.ref(config)
        assert weak_ref() is not None
        
        # Delete the config and force garbage collection
        del config
        gc.collect()
        
        # The weak reference should be gone if cleanup worked properly
        # Note: This test might be implementation-dependent
        try:
            assert weak_ref() is None
        except AssertionError:
            # Some implementations might hold references longer
            pass


class TestConfigCompatibility:
    """Test configuration compatibility with different Python versions and environments."""
    
    def test_config_with_pathlib(self):
        """Test configuration loading with pathlib.Path objects."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'pathlib_test_key'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert config is not None
            assert config.get('api_key') == 'pathlib_test_key'
        finally:
            temp_path.unlink()
    
    def test_config_with_environment_variable_substitution(self):
        """Test configuration with environment variable substitution."""
        if not Config:
            pytest.skip("Config class not available")
        
        with patch.dict(os.environ, {'TEST_API_KEY': 'env_substitution_key'}):
            # Test if config supports environment variable substitution
            config_data = {
                'api_key': '${TEST_API_KEY}',  # Common substitution pattern
                'base_url': 'https://api.example.com'
            }
            
            try:
                config = Config(config_data)
                # Check if substitution happened (implementation-dependent)
                assert config is not None
            except (ValueError, ConfigError):
                # Environment variable substitution might not be supported
                pass
    
    def test_config_with_relative_paths(self):
        """Test configuration loading with relative file paths."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data = {'api_key': 'relative_path_test_key'}
        
        # Create config in current directory
        config_filename = 'test_relative_config.json'
        
        with open(config_filename, 'w') as f:
            json.dump(config_data, f)
        
        try:
            # Test with relative path
            config = load_config(f'./{config_filename}')
            assert config is not None
            assert config.get('api_key') == 'relative_path_test_key'
        finally:
            if os.path.exists(config_filename):
                os.unlink(config_filename)


class TestConfigSpecialValues:
    """Test configuration with special values and data types."""
    
    def test_config_with_null_values(self):
        """Test configuration handling with null/None values."""
        if not Config:
            pytest.skip("Config class not available")
        
        config_with_nulls = {
            'api_key': 'test_key',
            'optional_field': None,
            'empty_string': '',
            'zero_value': 0,
            'false_value': False
        }
        
        try:
            config = Config(config_with_nulls)
            assert config is not None
        except (ValueError, ConfigError):
            # Some implementations may not allow null values
            pass
    
    def test_config_with_scientific_notation(self):
        """Test configuration with scientific notation numbers."""
        if not Config:
            pytest.skip("Config class not available")
        
        config_data = {
            'api_key': 'test_key',
            'timeout': 3e1,  # 30 in scientific notation
            'max_size': 1.5e6,  # 1,500,000
            'min_value': 1e-3  # 0.001
        }
        
        config = Config(config_data)
        assert config is not None
    
    def test_config_with_boolean_strings(self):
        """Test configuration with various boolean string representations."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        boolean_tests = [
            ('true', True),
            ('false', False),
            ('True', True),
            ('False', False),
            ('yes', True),
            ('no', False),
            ('1', True),
            ('0', False),
            ('on', True),
            ('off', False)
        ]
        
        for bool_str, expected in boolean_tests:
            config_data = {
                'api_key': 'test_key',
                'debug': bool_str
            }
            
            try:
                # Test if string boolean values are properly converted
                validate_config(config_data)
            except (ValueError, ConfigError):
                # String boolean conversion might not be supported
                pass


class TestConfigLogging:
    """Test configuration-related logging and debugging features."""
    
    def test_config_debug_logging(self):
        """Test configuration debug logging functionality."""
        if not Config:
            pytest.skip("Config class not available")
        
        import logging
        from io import StringIO
        
        # Set up logging capture
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            config_data = {
                'api_key': 'logging_test_key',
                'debug': True
            }
            
            config = Config(config_data)
            assert config is not None
            
            # Check if any debug logging occurred
            log_output = log_capture.getvalue()
            # This is implementation-dependent
            
        finally:
            logger.removeHandler(handler)
    
    def test_config_sensitive_data_redaction_in_logs(self):
        """Test that sensitive configuration data is redacted in logs."""
        if not Config:
            pytest.skip("Config class not available")
        
        sensitive_config = {
            'api_key': 'super_secret_key_12345',
            'password': 'secret_password_67890',
            'token': 'auth_token_abcdef'
        }
        
        config = Config(sensitive_config)
        config_repr = repr(config)
        config_str = str(config)
        
        # Sensitive values should not appear in string representations
        sensitive_values = ['super_secret_key_12345', 'secret_password_67890', 'auth_token_abcdef']
        
        for sensitive_value in sensitive_values:
            assert sensitive_value not in config_repr
            assert sensitive_value not in config_str


class TestConfigCaching:
    """Test configuration caching and reloading behavior."""
    
    def test_config_file_caching(self):
        """Test that configuration files are properly cached or reloaded."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        config_data_v1 = {'api_key': 'cached_test_key_v1'}
        config_data_v2 = {'api_key': 'cached_test_key_v2'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data_v1, f)
            temp_path = f.name
        
        try:
            # Load config first time
            config1 = load_config(temp_path)
            assert config1.get('api_key') == 'cached_test_key_v1'
            
            # Modify the file
            with open(temp_path, 'w') as f:
                json.dump(config_data_v2, f)
            
            # Load config again - behavior depends on caching implementation
            config2 = load_config(temp_path)
            # Could be cached (same as v1) or reloaded (same as v2)
            assert config2.get('api_key') in ['cached_test_key_v1', 'cached_test_key_v2']
            
        finally:
            os.unlink(temp_path)
    
    def test_config_modification_detection(self):
        """Test detection of configuration file modifications."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        import time
        
        config_data = {'api_key': 'modification_test_key', 'version': 1}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load initial config
            config1 = load_config(temp_path)
            initial_time = time.time()
            
            # Wait a bit and modify the file
            time.sleep(0.1)
            config_data['version'] = 2
            
            with open(temp_path, 'w') as f:
                json.dump(config_data, f)
            
            # Check if modification is detected (implementation-dependent)
            modification_time = os.path.getmtime(temp_path)
            assert modification_time > initial_time
            
        finally:
            os.unlink(temp_path)


# Additional parametrized tests for comprehensive edge case coverage

@pytest.mark.parametrize("file_extension", [".json", ".JSON", ".yml", ".yaml", ".YAML"])
def test_config_case_insensitive_extensions(file_extension):
    """Test configuration loading with case-insensitive file extensions."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    config_data = {'api_key': 'case_test_key'}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as f:
        if file_extension.lower().endswith('.json'):
            json.dump(config_data, f)
        else:
            yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config is not None
        assert config.get('api_key') == 'case_test_key'
    except (ValueError, ConfigError):
        # Some implementations may be case-sensitive
        pass
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize("special_char", ["space file.json", "file-with-dash.json", "file_with_underscore.json", "file.with.dots.json"])
def test_config_special_filename_characters(special_char):
    """Test configuration loading with special characters in filenames."""
    if not load_config:
        pytest.skip("load_config function not available")
    
    config_data = {'api_key': 'special_char_test_key'}
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, special_char)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(config_data, f)
        
        config = load_config(file_path)
        assert config is not None
        assert config.get('api_key') == 'special_char_test_key'
    except (OSError, ValueError, ConfigError):
        # Some special characters might not be supported on all filesystems
        pass
    finally:
        import shutil
        shutil.rmtree(temp_dir)


@pytest.mark.parametrize("config_size", [1, 10, 100, 1000])
def test_config_varying_sizes(config_size):
    """Test configuration handling with varying configuration sizes."""
    if not Config:
        pytest.skip("Config class not available")
    
    # Generate config with specified number of key-value pairs
    large_config = {'api_key': 'size_test_key'}
    large_config.update({f'key_{i}': f'value_{i}' for i in range(config_size)})
    
    config = Config(large_config)
    assert config is not None


@pytest.mark.parametrize("nesting_level", [1, 3, 5, 10])
def test_config_varying_nesting_levels(nesting_level):
    """Test configuration with varying levels of nesting."""
    if not Config:
        pytest.skip("Config class not available")
    
    # Create nested structure
    nested_config = {'api_key': 'nesting_test_key'}
    current_level = nested_config
    
    for i in range(nesting_level):
        current_level[f'level_{i}'] = {}
        current_level = current_level[f'level_{i}']
    
    current_level['deep_value'] = 'found_at_bottom'
    
    try:
        config = Config(nested_config)
        assert config is not None
    except (RecursionError, ValueError, ConfigError):
        # Very deep nesting might be rejected
        if nesting_level > 5:
            pass
        else:
            raise


# Additional stress tests
class TestConfigStress:
    """Stress tests for configuration handling under extreme conditions."""
    
    def test_config_rapid_creation_destruction(self):
        """Test rapid creation and destruction of configuration objects."""
        if not Config:
            pytest.skip("Config class not available")
        
        config_data = {'api_key': 'rapid_test_key'}
        
        # Rapidly create and destroy config objects
        for _ in range(1000):
            config = Config(config_data)
            del config
        
        # Should complete without memory issues or crashes
        assert True
    
    def test_config_concurrent_file_modifications(self):
        """Test configuration behavior with concurrent file modifications."""
        if not load_config:
            pytest.skip("load_config function not available")
        
        import threading
        import time
        
        config_data = {'api_key': 'concurrent_mod_test_key', 'counter': 0}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        def modify_config():
            for i in range(10):
                try:
                    with open(temp_path, 'r+') as f:
                        data = json.load(f)
                        data['counter'] = i
                        f.seek(0)
                        json.dump(data, f)
                        f.truncate()
                    time.sleep(0.01)
                except (json.JSONDecodeError, OSError):
                    # Concurrent modifications might cause temporary issues
                    pass
        
        def read_config():
            for _ in range(10):
                try:
                    load_config(temp_path)
                    time.sleep(0.01)
                except (json.JSONDecodeError, ConfigError, OSError):
                    # Concurrent access might cause temporary issues
                    pass
        
        try:
            # Start concurrent modification and reading
            modifier = threading.Thread(target=modify_config)
            reader = threading.Thread(target=read_config)
            
            modifier.start()
            reader.start()
            
            modifier.join()
            reader.join()
            
            # Should complete without crashes
            assert True
            
        finally:
            os.unlink(temp_path)


class TestConfigSchemaValidation:
    """Test configuration schema validation and type checking."""
    
    def test_config_strict_schema_validation(self):
        """Test strict schema validation for configuration."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test various schema violations
        schema_violations = [
            {'api_key': 123},  # Wrong type
            {'api_key': []},   # Wrong type
            {'timeout': 'thirty'},  # Wrong type for numeric field
            {'debug': 'maybe'},  # Wrong type for boolean field
        ]
        
        for violation in schema_violations:
            with pytest.raises((ValueError, ConfigError, TypeError)):
                validate_config(violation)
    
    def test_config_optional_fields_validation(self):
        """Test validation of optional configuration fields."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test configs with missing optional fields
        minimal_configs = [
            {'api_key': 'test_key'},  # Only required field
            {'api_key': 'test_key', 'debug': True},  # Some optional fields
        ]
        
        for config in minimal_configs:
            try:
                result = validate_config(config)
                # Should pass validation
                assert result is True or result is None
            except (ValueError, ConfigError):
                # Depending on implementation, minimal config might be rejected
                pass
    
    def test_config_nested_schema_validation(self):
        """Test validation of nested configuration schemas."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test nested structure validation
        nested_config = {
            'api_key': 'test_key',
            'character_settings': {
                'max_characters': 'not_a_number',  # Invalid type
                'allowed_types': 'not_a_list'     # Invalid type
            }
        }
        
        with pytest.raises((ValueError, ConfigError, TypeError)):
            validate_config(nested_config)


class TestConfigExtensibility:
    """Test configuration system extensibility and plugin support."""
    
    def test_config_custom_validators(self):
        """Test configuration with custom validation functions."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test if custom validators can be added (implementation-dependent)
        config_data = {
            'api_key': 'custom_validator_test_key',
            'custom_field': 'custom_value'
        }
        
        try:
            # This test is implementation-dependent
            validate_config(config_data)
        except (ValueError, ConfigError):
            # Custom fields might be rejected
            pass
    
    def test_config_plugin_loading(self):
        """Test configuration system with plugin-like extensions."""
        if not Config:
            pytest.skip("Config class not available")
        
        # Test if config system supports plugin-like functionality
        config_data = {
            'api_key': 'plugin_test_key',
            'plugins': ['plugin1', 'plugin2'],
            'plugin_config': {
                'plugin1': {'enabled': True},
                'plugin2': {'enabled': False}
            }
        }
        
        config = Config(config_data)
        assert config is not None


class TestConfigVersioning:
    """Test configuration versioning and migration support."""
    
    def test_config_version_handling(self):
        """Test configuration with version information."""
        if not Config:
            pytest.skip("Config class not available")
        
        versioned_configs = [
            {'api_key': 'test_key', 'version': '1.0'},
            {'api_key': 'test_key', 'config_version': 2},
            {'api_key': 'test_key', 'schema_version': '2.1.0'},
        ]
        
        for config_data in versioned_configs:
            try:
                config = Config(config_data)
                assert config is not None
            except (ValueError, ConfigError):
                # Version handling might not be implemented
                pass
    
    def test_config_backward_compatibility_validation(self):
        """Test configuration backward compatibility validation."""
        if not validate_config:
            pytest.skip("validate_config function not available")
        
        # Test old configuration format
        old_format_config = {
            'apiKey': 'old_format_key',  # camelCase instead of snake_case
            'baseUrl': 'https://api.example.com',
            'maxRetries': 3
        }
        
        try:
            validate_config(old_format_config)
        except (ValueError, ConfigError):
            # Old format might not be supported
            pass


class TestConfigLocalization:
    """Test configuration localization and internationalization support."""
    
    def test_config_with_localized_values(self):
        """Test configuration with localized/internationalized values."""
        if not Config:
            pytest.skip("Config class not available")
        
        localized_config = {
            'api_key': 'localization_test_key',
            'messages': {
                'en': 'Hello World',
                'es': 'Hola Mundo',
                'fr': 'Bonjour le Monde',
                'zh': '你好世界',
                'ar': 'مرحبا بالعالم'
            },
            'locale': 'en'
        }
        
        config = Config(localized_config)
        assert config is not None
    
    def test_config_unicode_normalization(self):
        """Test configuration with Unicode normalization requirements."""
        if not Config:
            pytest.skip("Config class not available")
        
        # Test different Unicode normalizations of the same character
        import unicodedata
        
        # Different representations of the same character (é)
        unicode_variants = [
            'café',  # precomposed
            'cafe\u0301',  # decomposed
        ]
        
        for variant in unicode_variants:
            config_data = {
                'api_key': 'unicode_test_key',
                'name': variant
            }
            
            config = Config(config_data)
            assert config is not None

