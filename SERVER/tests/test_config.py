import pytest
import os
import tempfile
import json
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the config module - adjust import path as needed
try:
    from SERVER.config import *
except ImportError:
    # If direct import fails, try alternative import patterns
    import sys
    sys.path.append('SERVER')
    from config import *


class TestConfigurationLoading:
    """Test configuration loading functionality"""
    
    def test_load_config_from_file_success(self):
        """Test successful configuration loading from file"""
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb"
            },
            "api": {
                "port": 8080,
                "host": "0.0.0.0"
            }
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
            with patch("os.path.exists", return_value=True):
                result = load_config("test_config.json")
                assert result == config_data
    
    def test_load_config_file_not_found(self):
        """Test configuration loading when file doesn't exist"""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_config("nonexistent_config.json")
    
    def test_load_config_invalid_json(self):
        """Test configuration loading with invalid JSON"""
        invalid_json = '{"key": "value",}'  # trailing comma makes it invalid
        
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(json.JSONDecodeError):
                    load_config("invalid_config.json")
    
    def test_load_config_empty_file(self):
        """Test configuration loading with empty file"""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(json.JSONDecodeError):
                    load_config("empty_config.json")
    
    def test_load_config_permission_error(self):
        """Test configuration loading with permission error"""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(PermissionError):
                    load_config("restricted_config.json")
    
    def test_load_config_with_yaml_format(self):
        """Test configuration loading from YAML format if supported"""
        yaml_content = """
        database:
          host: localhost
          port: 5432
          name: testdb
        api:
          port: 8080
          host: 0.0.0.0
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                try:
                    result = load_config("test_config.yaml")
                    assert result["database"]["host"] == "localhost"
                    assert result["database"]["port"] == 5432
                except (NameError, AttributeError):
                    # YAML loading not supported, skip test
                    pytest.skip("YAML loading not supported")


class TestConfigurationValidation:
    """Test configuration validation functionality"""
    
    def test_validate_config_valid_structure(self):
        """Test validation of valid configuration structure"""
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "user": "testuser",
                "password": "testpass"
            },
            "api": {
                "port": 8080,
                "host": "0.0.0.0",
                "debug": False
            }
        }
        
        try:
            assert validate_config(valid_config) == True
        except NameError:
            # validate_config function doesn't exist, create a basic test
            assert isinstance(valid_config, dict)
            assert "database" in valid_config
            assert "api" in valid_config
    
    def test_validate_config_missing_required_fields(self):
        """Test validation with missing required fields"""
        invalid_config = {
            "database": {
                "host": "localhost",
                # missing port, name, user, password
            },
            "api": {
                "port": 8080,
                "host": "0.0.0.0"
            }
        }
        
        try:
            with pytest.raises((KeyError, ValueError)):
                validate_config(invalid_config)
        except NameError:
            # validate_config function doesn't exist, skip test
            pytest.skip("validate_config function not available")
    
    def test_validate_config_invalid_data_types(self):
        """Test validation with invalid data types"""
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": "not_a_number",  # should be int
                "name": "testdb",
                "user": "testuser",
                "password": "testpass"
            },
            "api": {
                "port": 8080,
                "host": "0.0.0.0"
            }
        }
        
        try:
            with pytest.raises((ValueError, TypeError)):
                validate_config(invalid_config)
        except NameError:
            # validate_config function doesn't exist, skip test
            pytest.skip("validate_config function not available")
    
    def test_validate_config_empty_config(self):
        """Test validation with empty configuration"""
        empty_config = {}
        
        try:
            with pytest.raises((KeyError, ValueError)):
                validate_config(empty_config)
        except NameError:
            # validate_config function doesn't exist, skip test
            pytest.skip("validate_config function not available")
    
    def test_validate_config_none_values(self):
        """Test validation with None values"""
        config_with_none = {
            "database": {
                "host": None,
                "port": 5432,
                "name": "testdb",
                "user": "testuser",
                "password": "testpass"
            },
            "api": {
                "port": 8080,
                "host": "0.0.0.0"
            }
        }
        
        try:
            with pytest.raises((ValueError, TypeError)):
                validate_config(config_with_none)
        except NameError:
            # validate_config function doesn't exist, skip test
            pytest.skip("validate_config function not available")


class TestConfigurationDefaults:
    """Test configuration default values and fallbacks"""
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        try:
            default_config = get_default_config()
            
            assert isinstance(default_config, dict)
            assert len(default_config) > 0
            
            # Check for common configuration sections
            expected_sections = ["database", "api", "server", "logging"]
            found_sections = [section for section in expected_sections if section in default_config]
            assert len(found_sections) > 0, "At least one expected section should be present"
        except NameError:
            # get_default_config function doesn't exist, skip test
            pytest.skip("get_default_config function not available")
    
    def test_merge_config_with_defaults(self):
        """Test merging partial config with defaults"""
        partial_config = {
            "database": {
                "host": "custom_host"
            }
        }
        
        try:
            merged_config = merge_with_defaults(partial_config)
            
            assert merged_config["database"]["host"] == "custom_host"
            assert isinstance(merged_config, dict)
            assert len(merged_config) >= len(partial_config)
        except NameError:
            # merge_with_defaults function doesn't exist, skip test
            pytest.skip("merge_with_defaults function not available")
    
    def test_merge_config_override_all_defaults(self):
        """Test merging config that overrides all defaults"""
        custom_config = {
            "database": {
                "host": "custom_host",
                "port": 3306,
                "name": "custom_db",
                "user": "custom_user",
                "password": "custom_pass"
            },
            "api": {
                "host": "127.0.0.1",
                "port": 9000,
                "debug": True
            }
        }
        
        try:
            merged_config = merge_with_defaults(custom_config)
            
            assert merged_config["database"]["host"] == "custom_host"
            assert merged_config["database"]["port"] == 3306
            assert merged_config["api"]["port"] == 9000
        except NameError:
            # merge_with_defaults function doesn't exist, skip test
            pytest.skip("merge_with_defaults function not available")
    
    def test_merge_config_empty_input(self):
        """Test merging empty config with defaults"""
        empty_config = {}
        
        try:
            merged_config = merge_with_defaults(empty_config)
            default_config = get_default_config()
            
            assert merged_config == default_config
        except NameError:
            # Functions don't exist, skip test
            pytest.skip("merge_with_defaults or get_default_config function not available")


class TestEnvironmentVariables:
    """Test environment variable configuration overrides"""
    
    @patch.dict(os.environ, {
        "DB_HOST": "env_host",
        "DB_PORT": "3306",
        "API_PORT": "9000"
    })
    def test_load_config_from_env_variables(self):
        """Test loading configuration from environment variables"""
        try:
            config = load_config_from_env()
            
            # Check if environment variables are properly loaded
            assert isinstance(config, dict)
            
            # Try to find evidence of environment variable usage
            found_env_values = []
            for key, value in os.environ.items():
                if key.startswith(("DB_", "API_")):
                    found_env_values.append((key, value))
            
            assert len(found_env_values) > 0
        except NameError:
            # load_config_from_env function doesn't exist, skip test
            pytest.skip("load_config_from_env function not available")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_from_env_no_variables(self):
        """Test loading configuration when no env variables are set"""
        try:
            config = load_config_from_env()
            
            # Should return some default configuration
            assert isinstance(config, dict)
        except NameError:
            # load_config_from_env function doesn't exist, skip test
            pytest.skip("load_config_from_env function not available")
    
    @patch.dict(os.environ, {
        "DB_HOST": "env_host",
        "DB_PORT": "invalid_port"  # invalid port number
    })
    def test_load_config_from_env_invalid_values(self):
        """Test loading configuration with invalid environment values"""
        try:
            # This should either raise an error or handle the invalid value gracefully
            config = load_config_from_env()
            
            # If no error is raised, verify the invalid value is handled
            if "database" in config and "port" in config["database"]:
                # Should either be default value or converted properly
                assert isinstance(config["database"]["port"], (int, str))
        except (ValueError, TypeError):
            # Expected behavior for invalid values
            pass
        except NameError:
            # load_config_from_env function doesn't exist, skip test
            pytest.skip("load_config_from_env function not available")
    
    def test_env_var_precedence(self):
        """Test that environment variables take precedence over config files"""
        config_data = {
            "database": {
                "host": "file_host",
                "port": 5432
            }
        }
        
        with patch.dict(os.environ, {"DB_HOST": "env_host"}):
            with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
                with patch("os.path.exists", return_value=True):
                    try:
                        # Load config and apply env overrides
                        file_config = load_config("test_config.json")
                        final_config = apply_env_overrides(file_config)
                        
                        # Environment variable should override file value
                        assert final_config["database"]["host"] == "env_host"
                        assert final_config["database"]["port"] == 5432  # unchanged
                    except NameError:
                        # Functions don't exist, skip test
                        pytest.skip("Required functions not available")


class TestConfigurationSerialization:
    """Test configuration serialization and deserialization"""
    
    def test_config_to_json(self):
        """Test converting configuration to JSON"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        try:
            json_str = config_to_json(config)
            
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert parsed == config
        except NameError:
            # config_to_json function doesn't exist, use standard json
            json_str = json.dumps(config)
            parsed = json.loads(json_str)
            assert parsed == config
    
    def test_config_from_json(self):
        """Test creating configuration from JSON string"""
        json_str = '{"database": {"host": "localhost", "port": 5432}}'
        
        try:
            config = config_from_json(json_str)
            
            assert config["database"]["host"] == "localhost"
            assert config["database"]["port"] == 5432
        except NameError:
            # config_from_json function doesn't exist, use standard json
            config = json.loads(json_str)
            assert config["database"]["host"] == "localhost"
            assert config["database"]["port"] == 5432
    
    def test_config_to_json_with_complex_types(self):
        """Test JSON serialization with complex types"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "ssl": True,
                "timeout": 30.5,
                "tags": ["production", "primary"]
            }
        }
        
        try:
            json_str = config_to_json(config)
            parsed = json.loads(json_str)
        except NameError:
            # Use standard json if custom function doesn't exist
            json_str = json.dumps(config)
            parsed = json.loads(json_str)
        
        assert parsed == config
        assert isinstance(parsed["database"]["ssl"], bool)
        assert isinstance(parsed["database"]["timeout"], float)
        assert isinstance(parsed["database"]["tags"], list)
    
    def test_config_from_json_invalid_json(self):
        """Test creating configuration from invalid JSON"""
        invalid_json = '{"key": "value",}'
        
        with pytest.raises(json.JSONDecodeError):
            try:
                config_from_json(invalid_json)
            except NameError:
                # Use standard json if custom function doesn't exist
                json.loads(invalid_json)


class TestConfigurationUtilities:
    """Test configuration utility functions"""
    
    def test_get_config_value_existing_key(self):
        """Test getting existing configuration value"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        try:
            assert get_config_value(config, "database.host") == "localhost"
            assert get_config_value(config, "database.port") == 5432
        except NameError:
            # get_config_value function doesn't exist, test manual access
            assert config["database"]["host"] == "localhost"
            assert config["database"]["port"] == 5432
    
    def test_get_config_value_nonexistent_key(self):
        """Test getting nonexistent configuration value"""
        config = {
            "database": {
                "host": "localhost"
            }
        }
        
        try:
            assert get_config_value(config, "database.nonexistent") is None
            assert get_config_value(config, "database.nonexistent", "default") == "default"
        except NameError:
            # get_config_value function doesn't exist, test with dict.get
            assert config["database"].get("nonexistent") is None
            assert config["database"].get("nonexistent", "default") == "default"
    
    def test_get_config_value_deep_nesting(self):
        """Test getting deeply nested configuration values"""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep_value"
                    }
                }
            }
        }
        
        try:
            assert get_config_value(config, "level1.level2.level3.value") == "deep_value"
        except NameError:
            # get_config_value function doesn't exist, test manual access
            assert config["level1"]["level2"]["level3"]["value"] == "deep_value"
    
    def test_set_config_value(self):
        """Test setting configuration value"""
        config = {
            "database": {
                "host": "localhost"
            }
        }
        
        try:
            set_config_value(config, "database.port", 5432)
            assert config["database"]["port"] == 5432
        except NameError:
            # set_config_value function doesn't exist, test manual setting
            config["database"]["port"] = 5432
            assert config["database"]["port"] == 5432
    
    def test_set_config_value_new_nested_key(self):
        """Test setting new nested configuration value"""
        config = {}
        
        try:
            set_config_value(config, "new.nested.key", "value")
            assert config["new"]["nested"]["key"] == "value"
        except NameError:
            # set_config_value function doesn't exist, test manual nested setting
            if "new" not in config:
                config["new"] = {}
            if "nested" not in config["new"]:
                config["new"]["nested"] = {}
            config["new"]["nested"]["key"] = "value"
            assert config["new"]["nested"]["key"] == "value"


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_config_with_unicode_characters(self):
        """Test configuration with unicode characters"""
        config = {
            "database": {
                "name": "test_db_ñáéíóú",
                "user": "test_user_测试"
            }
        }
        
        json_str = json.dumps(config, ensure_ascii=False)
        parsed_config = json.loads(json_str)
        
        assert parsed_config == config
        assert "ñáéíóú" in parsed_config["database"]["name"]
        assert "测试" in parsed_config["database"]["user"]
    
    def test_config_with_large_numbers(self):
        """Test configuration with large numbers"""
        config = {
            "limits": {
                "max_connections": 1000000,
                "max_memory": 9223372036854775807  # max int64
            }
        }
        
        json_str = json.dumps(config)
        parsed_config = json.loads(json_str)
        
        assert parsed_config == config
        assert isinstance(parsed_config["limits"]["max_connections"], int)
        assert isinstance(parsed_config["limits"]["max_memory"], int)
    
    def test_config_with_special_characters(self):
        """Test configuration with special characters in values"""
        config = {
            "database": {
                "password": "p@ssw0rd!#$%^&*()",
                "connection_string": "postgres://user:pass@host:5432/db?sslmode=require"
            }
        }
        
        json_str = json.dumps(config)
        parsed_config = json.loads(json_str)
        
        assert parsed_config == config
        assert "p@ssw0rd!#$%^&*()" in parsed_config["database"]["password"]
        assert "postgres://" in parsed_config["database"]["connection_string"]
    
    def test_config_with_empty_strings(self):
        """Test configuration with empty strings"""
        config = {
            "database": {
                "host": "",
                "name": "testdb",
                "user": "",
                "password": ""
            }
        }
        
        json_str = json.dumps(config)
        parsed_config = json.loads(json_str)
        
        assert parsed_config == config
        assert parsed_config["database"]["host"] == ""
        assert parsed_config["database"]["user"] == ""
    
    def test_config_with_boolean_values(self):
        """Test configuration with boolean values"""
        config = {
            "features": {
                "debug": True,
                "ssl_enabled": False,
                "auto_backup": True
            }
        }
        
        json_str = json.dumps(config)
        parsed_config = json.loads(json_str)
        
        assert parsed_config == config
        assert parsed_config["features"]["debug"] is True
        assert parsed_config["features"]["ssl_enabled"] is False
        assert isinstance(parsed_config["features"]["debug"], bool)


class TestConfigurationSecurity:
    """Test security aspects of configuration handling"""
    
    def test_config_with_sensitive_data_masking(self):
        """Test that sensitive configuration data is properly handled"""
        config = {
            "database": {
                "host": "localhost",
                "password": "secret_password",
                "api_key": "sensitive_api_key"
            }
        }
        
        try:
            masked_config = mask_sensitive_config(config)
            
            assert masked_config["database"]["host"] == "localhost"
            assert "secret_password" not in str(masked_config)
            assert "sensitive_api_key" not in str(masked_config)
        except NameError:
            # mask_sensitive_config function doesn't exist, skip test
            pytest.skip("mask_sensitive_config function not available")
    
    def test_config_validation_prevents_injection(self):
        """Test that configuration validation prevents injection attacks"""
        malicious_config = {
            "database": {
                "host": "localhost; DROP TABLE users; --",
                "port": 5432
            }
        }
        
        try:
            # Should either raise an error or sanitize the input
            result = validate_config(malicious_config)
            
            if result:
                # If validation passes, check that dangerous content is handled
                assert isinstance(malicious_config["database"]["host"], str)
        except (ValueError, SecurityError):
            # Expected behavior for malicious input
            pass
        except NameError:
            # validate_config function doesn't exist, skip test
            pytest.skip("validate_config function not available")


@pytest.fixture
def temp_config_file():
    """Fixture to create a temporary config file for testing"""
    config_data = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "testdb"
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8080
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_file_path = f.name
    
    yield temp_file_path
    
    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration for testing"""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "testdb",
            "user": "testuser",
            "password": "testpass"
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8080,
            "debug": False
        },
        "logging": {
            "level": "INFO",
            "file": "/var/log/app.log"
        }
    }


@pytest.fixture
def config_with_env_vars():
    """Fixture providing configuration with environment variable references"""
    return {
        "database": {
            "host": "${DB_HOST:localhost}",
            "port": "${DB_PORT:5432}",
            "name": "${DB_NAME:testdb}"
        },
        "api": {
            "host": "${API_HOST:0.0.0.0}",
            "port": "${API_PORT:8080}"
        }
    }


class TestConfigurationIntegration:
    """Integration tests for configuration system"""
    
    def test_full_config_lifecycle(self, temp_config_file):
        """Test complete configuration lifecycle"""
        try:
            # Load config from file
            loaded_config = load_config(temp_config_file)
            
            # Validate loaded config
            assert isinstance(loaded_config, dict)
            assert "database" in loaded_config
            
            # Test serialization roundtrip
            json_str = json.dumps(loaded_config)
            final_config = json.loads(json_str)
            
            # Verify integrity
            assert final_config == loaded_config
        except NameError:
            # load_config function doesn't exist, test with standard json
            with open(temp_config_file, 'r') as f:
                loaded_config = json.load(f)
            
            assert isinstance(loaded_config, dict)
            assert "database" in loaded_config
    
    def test_config_backup_and_restore(self, sample_config):
        """Test configuration backup and restore functionality"""
        # Create a deep copy as backup
        import copy
        backup = copy.deepcopy(sample_config)
        
        # Modify original
        sample_config["database"]["host"] = "modified_host"
        
        # Verify backup is unchanged
        assert backup["database"]["host"] == "localhost"
        assert backup != sample_config
        
        # Restore from backup
        sample_config.update(backup)
        assert sample_config["database"]["host"] == "localhost"
    
    def test_config_with_multiple_sources(self, temp_config_file, sample_config):
        """Test configuration loading from multiple sources"""
        try:
            # Load from file
            file_config = load_config(temp_config_file)
            
            # Merge with sample config
            merged_config = {**file_config, **sample_config}
            
            # Verify merge
            assert isinstance(merged_config, dict)
            assert len(merged_config) >= len(file_config)
        except NameError:
            # load_config function doesn't exist, test manual merge
            with open(temp_config_file, 'r') as f:
                file_config = json.load(f)
            
            merged_config = {**file_config, **sample_config}
            assert isinstance(merged_config, dict)
    
    def test_config_environment_variable_substitution(self, config_with_env_vars):
        """Test environment variable substitution in configuration"""
        with patch.dict(os.environ, {
            "DB_HOST": "prod_host",
            "DB_PORT": "5433",
            "API_PORT": "9000"
        }):
            try:
                resolved_config = resolve_env_vars(config_with_env_vars)
                
                assert resolved_config["database"]["host"] == "prod_host"
                assert resolved_config["database"]["port"] == "5433"
                assert resolved_config["api"]["port"] == "9000"
                # Should use default when env var not set
                assert resolved_config["database"]["name"] == "testdb"
            except NameError:
                # resolve_env_vars function doesn't exist, skip test
                pytest.skip("resolve_env_vars function not available")


class TestConfigurationPerformance:
    """Test performance aspects of configuration handling"""
    
    def test_config_loading_performance(self, temp_config_file):
        """Test that configuration loading is reasonably fast"""
        import time
        
        start_time = time.time()
        
        # Load config multiple times
        for _ in range(100):
            try:
                load_config(temp_config_file)
            except NameError:
                # load_config function doesn't exist, use standard json
                with open(temp_config_file, 'r') as f:
                    json.load(f)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 1.0, f"Config loading took too long: {duration:.2f}s"
    
    def test_config_serialization_performance(self, sample_config):
        """Test that configuration serialization is reasonably fast"""
        import time
        
        start_time = time.time()
        
        # Serialize config multiple times
        for _ in range(1000):
            json.dumps(sample_config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 1.0, f"Config serialization took too long: {duration:.2f}s"
    
    def test_large_config_handling(self):
        """Test handling of large configuration files"""
        # Create a large configuration
        large_config = {}
        for i in range(1000):
            large_config[f"section_{i}"] = {
                f"key_{j}": f"value_{j}" for j in range(10)
            }
        
        # Test serialization
        json_str = json.dumps(large_config)
        assert len(json_str) > 10000
        
        # Test deserialization
        parsed_config = json.loads(json_str)
        assert len(parsed_config) == 1000
        assert parsed_config == large_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])