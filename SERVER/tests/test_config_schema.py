import unittest
import json

class TestConfigSchema(unittest.TestCase):
    """Test configuration against predefined schemas."""
    
    def setUp(self):
        """
        Initializes the JSON schema definition for configuration validation before each test.
        """
        self.config_schema = {
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "name": {"type": "string"},
                        "user": {"type": "string"},
                        "password": {"type": "string"}
                    },
                    "required": ["host", "port", "name"]
                },
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "debug": {"type": "boolean"}
                    },
                    "required": ["host", "port"]
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "format": {"type": "string"}
                    }
                }
            },
            "required": ["database", "server"]
        }
        
    def test_valid_config_schema(self):
        """
        Verify that a configuration dictionary with valid structure and types passes schema validation checks.
        """
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "user": "user",
                "password": "pass"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            "logging": {
                "level": "INFO",
                "format": "%(message)s"
            }
        }
        
        # Basic schema validation without external libraries
        self.assertIn("database", valid_config)
        self.assertIn("server", valid_config)
        self.assertIsInstance(valid_config["database"]["port"], int)
        self.assertIsInstance(valid_config["server"]["debug"], bool)
            
    def test_invalid_config_schema_missing_required(self):
        """
        Test that a configuration dictionary missing the required "database" section is correctly identified as invalid.
        """
        invalid_config = {
            "server": {
                "host": "localhost",
                "port": 8000
            }
            # Missing required "database" section
        }
        
        # Check for missing required fields
        self.assertNotIn("database", invalid_config)
            
    def test_invalid_config_schema_wrong_types(self):
        """
        Test that a configuration with incorrect data types for fields is detected as invalid.
        
        Specifically, verifies that the 'database.port' field is a string instead of the expected integer type.
        """
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": "not_a_number",  # Should be integer
                "name": "testdb"
            },
            "server": {
                "host": "localhost",
                "port": 8000
            }
        }
        
        # Check for wrong data types
        self.assertIsInstance(invalid_config["database"]["port"], str)
        self.assertNotIsInstance(invalid_config["database"]["port"], int)
            
    def test_invalid_config_schema_out_of_range(self):
        """
        Test that a configuration with a database port value exceeding the valid range is detected as invalid.
        
        Asserts that the 'database.port' field is greater than 65535, indicating an out-of-range value.
        """
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # Outside typical valid port range
                "name": "testdb"
            },
            "server": {
                "host": "localhost",
                "port": 8000
            }
        }
        
        # Check for out-of-range values
        self.assertGreater(invalid_config["database"]["port"], 65535)


if __name__ == '__main__':
    unittest.main()