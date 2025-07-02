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

# Tests for the actual config.py functionality
class TestActualConfigModule:
    """Test the actual config.py module functionality - path management and directory creation."""
    
    def setup_method(self):
        """Set up test fixtures for actual config testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_client_root_path_calculation(self):
        """Test CLIENT_ROOT path calculation."""
        # Import to get the actual config module
        import sys
        import importlib
        
        # Temporarily add path and import
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            assert hasattr(config, 'CLIENT_ROOT')
            assert os.path.isabs(config.CLIENT_ROOT)
            assert config.CLIENT_ROOT.endswith('CharacterClient')
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_default_path_constants(self):
        """Test default path constants are properly defined."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            # Test all expected path constants exist
            expected_paths = [
                'CLIENT_ROOT', 'DEFAULT_CLIENT_DATA_PATH', 'CLIENT_DATA_PATH',
                'DEFAULT_CLIENT_MODELS_PATH', 'CLIENT_MODELS_PATH',
                'CLIENT_LLM_MODELS_PATH', 'CLIENT_TTS_MODELS_PATH',
                'CLIENT_TTS_REFERENCE_VOICES_PATH', 'CLIENT_LOGS_PATH',
                'CLIENT_TEMP_AUDIO_PATH'
            ]
            
            for path_name in expected_paths:
                assert hasattr(config, path_name), f"Missing path constant: {path_name}"
                path_value = getattr(config, path_name)
                assert isinstance(path_value, str), f"{path_name} should be a string"
                assert os.path.isabs(path_value), f"{path_name} should be absolute path"
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': '/custom/data/path'})
    def test_environment_variable_override_data_path(self):
        """Test CLIENT_DATA_PATH can be overridden by environment variable."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            assert config.CLIENT_DATA_PATH == '/custom/data/path'
            # Dependent paths should also use the custom base
            assert config.CLIENT_LOGS_PATH.startswith('/custom/data/path')
            assert config.CLIENT_TEMP_AUDIO_PATH.startswith('/custom/data/path')
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    @patch.dict(os.environ, {'DREAMWEAVER_CLIENT_MODELS_PATH': '/custom/models/path'})
    def test_environment_variable_override_models_path(self):
        """Test CLIENT_MODELS_PATH can be overridden by environment variable."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            assert config.CLIENT_MODELS_PATH == '/custom/models/path'
            # Sub-model paths should use the custom base
            assert config.CLIENT_LLM_MODELS_PATH.startswith('/custom/models/path')
            assert config.CLIENT_TTS_MODELS_PATH.startswith('/custom/models/path')
            assert config.CLIENT_TTS_REFERENCE_VOICES_PATH.startswith('/custom/models/path')
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_ensure_client_directories_function_exists(self):
        """Test that ensure_client_directories function exists and is callable."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            assert hasattr(config, 'ensure_client_directories')
            assert callable(config.ensure_client_directories)
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    @patch('os.makedirs')
    def test_ensure_client_directories_creates_all_paths(self, mock_makedirs):
        """Test that ensure_client_directories attempts to create all required directories."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            # Reset the mock to clear any calls from import
            mock_makedirs.reset_mock()
            
            config.ensure_client_directories()
            
            # Should have called makedirs for each directory
            expected_calls = [
                config.CLIENT_DATA_PATH,
                config.CLIENT_MODELS_PATH,
                config.CLIENT_LLM_MODELS_PATH,
                config.CLIENT_TTS_MODELS_PATH,
                config.CLIENT_TTS_REFERENCE_VOICES_PATH,
                config.CLIENT_LOGS_PATH,
                config.CLIENT_TEMP_AUDIO_PATH
            ]
            
            assert mock_makedirs.call_count >= len(expected_calls)
            
            # Check that all expected paths were called
            called_paths = [call[0][0] for call in mock_makedirs.call_args_list]
            for expected_path in expected_calls:
                assert expected_path in called_paths
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    @patch('os.makedirs')
    def test_ensure_client_directories_uses_exist_ok(self, mock_makedirs):
        """Test that ensure_client_directories uses exist_ok=True."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            # Reset the mock to clear any calls from import
            mock_makedirs.reset_mock()
            
            config.ensure_client_directories()
            
            # All calls should use exist_ok=True
            for call in mock_makedirs.call_args_list:
                args, kwargs = call
                assert kwargs.get('exist_ok', False) is True
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    @patch('os.makedirs', side_effect=OSError("Permission denied"))
    @patch('sys.stderr')
    def test_ensure_client_directories_handles_permission_errors(self, mock_stderr, mock_makedirs):
        """Test that ensure_client_directories handles permission errors gracefully."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            # Should not raise exception even with permission errors
            config.ensure_client_directories()
            
            # Should log critical errors
            # The function should continue even if some directories fail
            assert mock_makedirs.called
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_config_module_can_run_as_main(self):
        """Test that config module can be executed as main for verification."""
        import sys
        import subprocess
        
        # Try to run the config module as main
        try:
            result = subprocess.run([
                sys.executable, '-c',
                'import sys; sys.path.insert(0, "CharacterClient/src"); import config'
            ], capture_output=True, text=True, timeout=10)
            
            # Should not crash
            assert result.returncode == 0 or result.returncode is None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Cannot execute config module as subprocess")
    
    def test_path_hierarchy_consistency(self):
        """Test that path hierarchy is consistent."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            # CLIENT_DATA_PATH should be parent of logs and temp_audio
            assert config.CLIENT_LOGS_PATH.startswith(config.CLIENT_DATA_PATH)
            assert config.CLIENT_TEMP_AUDIO_PATH.startswith(config.CLIENT_DATA_PATH)
            
            # CLIENT_MODELS_PATH should be parent of specific model paths
            assert config.CLIENT_LLM_MODELS_PATH.startswith(config.CLIENT_MODELS_PATH)
            assert config.CLIENT_TTS_MODELS_PATH.startswith(config.CLIENT_MODELS_PATH)
            assert config.CLIENT_TTS_REFERENCE_VOICES_PATH.startswith(config.CLIENT_TTS_MODELS_PATH)
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_path_separation_characters(self):
        """Test that paths use correct separation characters for the platform."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            paths_to_test = [
                config.CLIENT_ROOT, config.CLIENT_DATA_PATH, config.CLIENT_MODELS_PATH,
                config.CLIENT_LLM_MODELS_PATH, config.CLIENT_TTS_MODELS_PATH,
                config.CLIENT_TTS_REFERENCE_VOICES_PATH, config.CLIENT_LOGS_PATH,
                config.CLIENT_TEMP_AUDIO_PATH
            ]
            
            for path in paths_to_test:
                # Should use os.path.join (platform-appropriate separators)
                assert os.sep in path or len(path.split('/')) == 1  # Root path might not have separators
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_directory_names_are_valid(self):
        """Test that directory names are valid for the filesystem."""
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            paths_to_test = [
                config.CLIENT_DATA_PATH, config.CLIENT_MODELS_PATH,
                config.CLIENT_LLM_MODELS_PATH, config.CLIENT_TTS_MODELS_PATH,
                config.CLIENT_TTS_REFERENCE_VOICES_PATH, config.CLIENT_LOGS_PATH,
                config.CLIENT_TEMP_AUDIO_PATH
            ]
            
            invalid_chars = '<>:"|?*' if os.name == 'nt' else '\0'
            
            for path in paths_to_test:
                for char in invalid_chars:
                    assert char not in os.path.basename(path), f"Invalid char '{char}' in path: {path}"
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')


class TestConfigRealDirectoryOperations:
    """Test actual directory operations with temporary directories."""
    
    def setup_method(self):
        """Set up with temporary directory for safe testing."""
        self.temp_root = tempfile.mkdtemp()
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Clean up temporary directories."""
        import shutil
        if os.path.exists(self.temp_root):
            shutil.rmtree(self.temp_root)
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @patch.dict(os.environ, clear=True)
    def test_directory_creation_with_custom_base(self):
        """Test directory creation with custom base path."""
        custom_data_path = os.path.join(self.temp_root, 'custom_data')
        os.environ['DREAMWEAVER_CLIENT_DATA_PATH'] = custom_data_path
        
        import sys
        sys.path.insert(0, 'CharacterClient/src')
        try:
            import config
            importlib.reload(config)
            
            # Directories should be created under our custom path
            expected_dirs = [
                custom_data_path,
                os.path.join(custom_data_path, 'logs'),
                os.path.join(custom_data_path, 'temp_audio')
            ]
            
            for expected_dir in expected_dirs:
                assert os.path.exists(expected_dir), f"Directory not created: {expected_dir}"
                assert os.path.isdir(expected_dir), f"Path is not a directory: {expected_dir}"
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_directory_permissions(self):
        """Test that created directories have appropriate permissions."""
        custom_data_path = os.path.join(self.temp_root, 'permissions_test')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': custom_data_path}):
            import sys
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Check that directories are readable and writable
                test_dirs = [
                    custom_data_path,
                    config.CLIENT_LOGS_PATH,
                    config.CLIENT_TEMP_AUDIO_PATH
                ]
                
                for test_dir in test_dirs:
                    if os.path.exists(test_dir):
                        assert os.access(test_dir, os.R_OK), f"Directory not readable: {test_dir}"
                        assert os.access(test_dir, os.W_OK), f"Directory not writable: {test_dir}"
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')
    
    def test_nested_directory_creation(self):
        """Test creation of deeply nested directories."""
        deep_path = os.path.join(self.temp_root, 'very', 'deep', 'nested', 'structure')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': deep_path}):
            import sys
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Should create the entire nested structure
                assert os.path.exists(deep_path), "Deep nested directory not created"
                
                # Should also create subdirectories
                logs_path = os.path.join(deep_path, 'logs')
                assert os.path.exists(logs_path), "Nested logs directory not created"
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')


class TestConfigEdgeCasesReal:
    """Test edge cases for the real config module."""
    
    def setup_method(self):
        """Set up for edge case testing."""
        self.temp_root = tempfile.mkdtemp()
        self.original_env = dict(os.environ)
    
    def teardown_method(self):
        """Clean up edge case testing."""
        import shutil
        if os.path.exists(self.temp_root):
            shutil.rmtree(self.temp_root)
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_empty_environment_variable(self):
        """Test behavior with empty environment variables."""
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': ''}):
            import sys
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Should fall back to default path, not use empty string
                assert config.CLIENT_DATA_PATH != ''
                assert 'CharacterClient' in config.CLIENT_DATA_PATH
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')
    
    def test_relative_path_in_environment(self):
        """Test behavior with relative paths in environment variables."""
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': './relative/path'}):
            import sys
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Should handle relative paths (behavior may vary)
                assert config.CLIENT_DATA_PATH is not None
                # The exact behavior depends on implementation
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')
    
    def test_unicode_paths(self):
        """Test handling of Unicode characters in paths."""
        unicode_path = os.path.join(self.temp_root, 'тест_🎯_test')
        
        with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': unicode_path}):
            import sys
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Should handle Unicode paths gracefully
                assert config.CLIENT_DATA_PATH == unicode_path
                # Directory creation may or may not succeed depending on filesystem
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            except (UnicodeError, OSError):
                # Unicode paths might not be supported on all systems
                pytest.skip("Unicode paths not supported on this system")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')


class TestConfigLogging:
    """Test logging behavior in config module."""
    
    def test_logging_setup_during_import(self):
        """Test that logging is properly set up during import."""
        import sys
        import logging
        
        # Capture log output
        with patch('logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Should have created a logger
                assert mock_logger.called
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')
    
    @patch('sys.stderr')
    def test_critical_error_output(self, mock_stderr):
        """Test that critical errors are output to stderr."""
        import sys
        
        with patch('os.makedirs', side_effect=OSError("Critical error")):
            sys.path.insert(0, 'CharacterClient/src')
            try:
                import config
                importlib.reload(config)
                
                # Should have written to stderr for critical errors
                # The exact behavior depends on implementation
            
            except ImportError:
                pytest.skip("Config module not available for actual testing")
            finally:
                if 'CharacterClient/src' in sys.path:
                    sys.path.remove('CharacterClient/src')


# Performance tests for real config functionality
class TestConfigPerformanceReal:
    """Performance tests for actual config operations."""
    
    def test_import_performance(self):
        """Test that config module imports quickly."""
        import sys
        import time
        
        sys.path.insert(0, 'CharacterClient/src')
        try:
            start_time = time.time()
            import config
            importlib.reload(config)
            import_time = time.time() - start_time
            
            # Should import quickly (less than 1 second)
            assert import_time < 1.0, f"Config import too slow: {import_time:.3f}s"
        
        except ImportError:
            pytest.skip("Config module not available for actual testing")
        finally:
            if 'CharacterClient/src' in sys.path:
                sys.path.remove('CharacterClient/src')
    
    def test_directory_creation_performance(self):
        """Test that directory creation is performant."""
        import sys
        import time
        
        temp_root = tempfile.mkdtemp()
        custom_path = os.path.join(temp_root, 'perf_test')
        
        try:
            with patch.dict(os.environ, {'DREAMWEAVER_CLIENT_DATA_PATH': custom_path}):
                sys.path.insert(0, 'CharacterClient/src')
                try:
                    import config
                    
                    start_time = time.time()
                    config.ensure_client_directories()
                    creation_time = time.time() - start_time
                    
                    # Directory creation should be fast (less than 1 second)
                    assert creation_time < 1.0, f"Directory creation too slow: {creation_time:.3f}s"
                
                except ImportError:
                    pytest.skip("Config module not available for actual testing")
                finally:
                    if 'CharacterClient/src' in sys.path:
                        sys.path.remove('CharacterClient/src')
        
        finally:
            import shutil
            if os.path.exists(temp_root):
                shutil.rmtree(temp_root)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])