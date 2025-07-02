import pytest
import json
import os
from unittest.mock import patch
import sys

# Add the SERVER directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Testing framework: pytest
# Configuration and setup tests for character server


class TestCharacterServerConfiguration:
    """Test suite for character server configuration scenarios."""
    
    def test_character_server_default_configuration(self):
        """Test character server with default configuration."""
        from test_character_server import MockCharacterServer
        
        server = MockCharacterServer()
        
        # Test default behavior
        assert server.next_id == 1
        assert len(server.characters) == 0
        
        # Create character with defaults
        char_data = {'name': 'DefaultHero', 'class': 'Warrior', 'level': 1}
        char = server.create_character(char_data)
        
        assert char['id'] == 1
        assert char['name'] == 'DefaultHero'
    
    def test_character_server_with_custom_validation_rules(self):
        """Test character server with custom validation configuration."""
        from test_character_server import MockCharacterServer
        
        class CustomValidationServer(MockCharacterServer):
            @staticmethod
            def validate_character_data(data):
                # Custom validation rules
                required_fields = ['name', 'class', 'level', 'faction']
                if not all(field in data for field in required_fields):
                    return False
                
                # Name must be at least 3 characters
                if len(data.get('name', '')) < 3:
                    return False
                
                # Level must be between 1 and 200
                level = data.get('level', 0)
                if not isinstance(level, int) or level < 1 or level > 200:
                    return False
                
                # Faction must be valid
                valid_factions = ['Alliance', 'Horde', 'Neutral']
                if data.get('faction') not in valid_factions:
                    return False
                
                return True
        
        server = CustomValidationServer()
        
        # Test valid character with custom rules
        valid_char = {
            'name': 'CustomHero',
            'class': 'Warrior',
            'level': 50,
            'faction': 'Alliance'
        }
        result = server.create_character(valid_char)
        assert result['name'] == 'CustomHero'
        assert result['faction'] == 'Alliance'
        
        # Test invalid characters
        invalid_chars = [
            {'name': 'Hi', 'class': 'Warrior', 'level': 1, 'faction': 'Alliance'},  # Name too short
            {'name': 'Hero', 'class': 'Warrior', 'level': 300, 'faction': 'Alliance'},  # Level too high
            {'name': 'Hero', 'class': 'Warrior', 'level': 1, 'faction': 'InvalidFaction'},  # Invalid faction
            {'name': 'Hero', 'class': 'Warrior', 'level': 1},  # Missing faction
        ]
        
        for invalid_char in invalid_chars:
            with pytest.raises(ValueError):
                server.create_character(invalid_char)
    
    def test_character_server_with_environment_configuration(self):
        """Test character server behavior with environment-based configuration."""
        from test_character_server import MockCharacterServer
        
        class ConfigurableServer(MockCharacterServer):
            def __init__(self):
                super().__init__()
                # Read configuration from environment
                self.max_characters = int(os.getenv('MAX_CHARACTERS', '1000'))
                self.enable_validation = os.getenv('ENABLE_VALIDATION', 'true').lower() == 'true'
                self.default_level = int(os.getenv('DEFAULT_LEVEL', '1'))
            
            def create_character(self, character_data):
                # Check max characters limit
                if len(self.characters) >= self.max_characters:
                    raise ValueError("Maximum number of characters reached")
                
                # Apply default level if not specified
                if 'level' not in character_data:
                    character_data = {**character_data, 'level': self.default_level}
                
                # Skip validation if disabled
                if self.enable_validation and not self.validate_character_data(character_data):
                    raise ValueError("Invalid character data")
                elif not self.enable_validation:
                    # Minimal validation when disabled
                    if 'name' not in character_data:
                        raise ValueError("Name is required")
                
                character = {**character_data, 'id': self.next_id}
                self.characters[self.next_id] = character
                self.next_id += 1
                return character
        
        # Test with default environment
        server1 = ConfigurableServer()
        char_data = {'name': 'EnvHero', 'class': 'Warrior', 'level': 5}
        char = server1.create_character(char_data)
        assert char['level'] == 5
        
        # Test with custom environment
        with patch.dict(os.environ, {'MAX_CHARACTERS': '2', 'DEFAULT_LEVEL': '10', 'ENABLE_VALIDATION': 'false'}):
            server2 = ConfigurableServer()
            
            # Test default level application
            char_no_level = server2.create_character({'name': 'NoLevelHero', 'class': 'Mage'})
            assert char_no_level['level'] == 10
            
            # Test max characters limit
            server2.create_character({'name': 'Hero1', 'class': 'Warrior'})
            server2.create_character({'name': 'Hero2', 'class': 'Mage'})
            
            with pytest.raises(ValueError, match="Maximum number of characters reached"):
                server2.create_character({'name': 'Hero3', 'class': 'Rogue'})
            
            # Test disabled validation
            minimal_char = server2.create_character({'name': 'MinimalHero'})  # Missing class and level
            assert minimal_char['name'] == 'MinimalHero'
    
    def test_character_server_with_json_configuration_file(self):
        """Test character server with JSON configuration file."""
        import tempfile
        from test_character_server import MockCharacterServer
        
        # Create temporary config file
        config_data = {
            "validation_rules": {
                "min_name_length": 2,
                "max_name_length": 50,
                "min_level": 1,
                "max_level": 100,
                "required_fields": ["name", "class", "level"]
            },
            "server_settings": {
                "max_characters": 500,
                "auto_increment_start": 1000,
                "enable_caching": True
            },
            "character_defaults": {
                "hp": 100,
                "mp": 50,
                "status": "active"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            class ConfigFileServer(MockCharacterServer):
                def __init__(self, config_file):
                    super().__init__()
                    with open(config_file, 'r') as f:
                        self.config = json.load(f)
                    
                    # Apply configuration
                    self.next_id = self.config['server_settings']['auto_increment_start']
                    self.max_characters = self.config['server_settings']['max_characters']
                    self.character_defaults = self.config['character_defaults']
                
                def create_character(self, character_data):
                    # Apply defaults
                    full_data = {**self.character_defaults, **character_data}
                    
                    # Custom validation based on config
                    rules = self.config['validation_rules']
                    
                    # Check required fields
                    for field in rules['required_fields']:
                        if field not in full_data:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Check name length
                    name = full_data.get('name', '')
                    if len(name) < rules['min_name_length'] or len(name) > rules['max_name_length']:
                        raise ValueError("Name length out of bounds")
                    
                    # Check level bounds
                    level = full_data.get('level', 0)
                    if level < rules['min_level'] or level > rules['max_level']:
                        raise ValueError("Level out of bounds")
                    
                    # Check max characters
                    if len(self.characters) >= self.max_characters:
                        raise ValueError("Server at capacity")
                    
                    character = {**full_data, 'id': self.next_id}
                    self.characters[self.next_id] = character
                    self.next_id += 1
                    return character
            
            server = ConfigFileServer(config_file)
            
            # Test character creation with defaults applied
            char_data = {'name': 'ConfigHero', 'class': 'Warrior', 'level': 25}
            char = server.create_character(char_data)
            
            assert char['id'] == 1000  # Custom start ID
            assert char['hp'] == 100   # Applied default
            assert char['mp'] == 50    # Applied default
            assert char['status'] == 'active'  # Applied default
            
            # Test validation rules
            with pytest.raises(ValueError, match="Name length out of bounds"):
                server.create_character({'name': 'X', 'class': 'Warrior', 'level': 1})  # Too short
            
            with pytest.raises(ValueError, match="Level out of bounds"):
                server.create_character({'name': 'Hero', 'class': 'Warrior', 'level': 150})  # Too high
            
        finally:
            os.unlink(config_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])