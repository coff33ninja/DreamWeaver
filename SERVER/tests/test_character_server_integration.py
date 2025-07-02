import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the SERVER directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Testing framework: pytest
# Integration tests for character server functionality


class TestCharacterServerIntegration:
    """Integration test suite for character server with external dependencies."""
    
    def test_character_server_json_serialization(self):
        """Test character server data serialization to JSON."""
        from test_character_server import MockCharacterServer
        
        server = MockCharacterServer()
        
        # Create test characters
        test_chars = [
            {'name': 'Hero1', 'class': 'Warrior', 'level': 10},
            {'name': 'Hero2', 'class': 'Mage', 'level': 15},
            {'name': 'Hero3', 'class': 'Rogue', 'level': 8}
        ]
        
        created_chars = []
        for char_data in test_chars:
            char = server.create_character(char_data)
            created_chars.append(char)
        
        # Serialize to JSON
        all_chars = server.list_characters()
        json_data = json.dumps(all_chars, indent=2)
        
        # Verify JSON is valid and contains expected data
        parsed_data = json.loads(json_data)
        assert len(parsed_data) == 3
        assert parsed_data[0]['name'] == 'Hero1'
        assert parsed_data[1]['class'] == 'Mage'
        assert parsed_data[2]['level'] == 8
    
    def test_character_server_file_persistence_simulation(self):
        """Test character server data persistence to file (simulated)."""
        from test_character_server import MockCharacterServer
        
        server = MockCharacterServer()
        
        # Create characters
        char_data = {'name': 'PersistentHero', 'class': 'Paladin', 'level': 25}
        char = server.create_character(char_data)
        
        # Simulate saving to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            all_chars = server.list_characters()
            json.dump(all_chars, f)
            temp_filename = f.name
        
        try:
            # Simulate loading from file
            with open(temp_filename, 'r') as f:
                loaded_chars = json.load(f)
            
            # Verify data integrity
            assert len(loaded_chars) == 1
            assert loaded_chars[0]['name'] == 'PersistentHero'
            assert loaded_chars[0]['class'] == 'Paladin'
            assert loaded_chars[0]['level'] == 25
            
        finally:
            # Clean up temp file
            os.unlink(temp_filename)
    
    def test_character_server_with_external_validation_service(self):
        """Test character server integration with external validation service."""
        from test_character_server import MockCharacterServer
        
        class ExternalValidationService:
            @staticmethod
            def validate_character_name(name):
                """Simulate external name validation service."""
                forbidden_names = ['admin', 'system', 'root']
                return name.lower() not in forbidden_names
            
            @staticmethod
            def validate_character_class(char_class):
                """Simulate external class validation service."""
                valid_classes = ['Warrior', 'Mage', 'Rogue', 'Paladin', 'Archer']
                return char_class in valid_classes
        
        # Extend MockCharacterServer with external validation
        class ValidatedCharacterServer(MockCharacterServer):
            def create_character(self, character_data):
                # Use external validation
                if not ExternalValidationService.validate_character_name(character_data.get('name', '')):
                    raise ValueError("Character name not allowed by external service")
                
                if not ExternalValidationService.validate_character_class(character_data.get('class', '')):
                    raise ValueError("Character class not valid according to external service")
                
                return super().create_character(character_data)
        
        server = ValidatedCharacterServer()
        
        # Test valid character creation
        valid_char = {'name': 'ValidHero', 'class': 'Warrior', 'level': 1}
        result = server.create_character(valid_char)
        assert result['name'] == 'ValidHero'
        
        # Test invalid name
        with pytest.raises(ValueError, match="name not allowed"):
            server.create_character({'name': 'admin', 'class': 'Warrior', 'level': 1})
        
        # Test invalid class
        with pytest.raises(ValueError, match="class not valid"):
            server.create_character({'name': 'Hero', 'class': 'InvalidClass', 'level': 1})
    
    def test_character_server_with_logging_integration(self):
        """Test character server integration with logging system."""
        import logging
        from io import StringIO
        from test_character_server import MockCharacterServer
        
        # Set up logging capture
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('character_server')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Extend MockCharacterServer with logging
        class LoggingCharacterServer(MockCharacterServer):
            def create_character(self, character_data):
                logger.info(f"Creating character: {character_data.get('name')}")
                result = super().create_character(character_data)
                logger.info(f"Created character with ID: {result['id']}")
                return result
            
            def delete_character(self, character_id):
                logger.info(f"Deleting character ID: {character_id}")
                result = super().delete_character(character_id)
                if result:
                    logger.info(f"Successfully deleted character ID: {character_id}")
                else:
                    logger.warning(f"Failed to delete character ID: {character_id}")
                return result
        
        server = LoggingCharacterServer()
        
        # Perform operations
        char_data = {'name': 'LoggedHero', 'class': 'Warrior', 'level': 1}
        char = server.create_character(char_data)
        server.delete_character(char['id'])
        server.delete_character(9999)  # Should log warning
        
        # Verify logging
        log_output = log_capture.getvalue()
        assert 'Creating character: LoggedHero' in log_output
        assert f'Created character with ID: {char["id"]}' in log_output
        assert f'Successfully deleted character ID: {char["id"]}' in log_output
        assert 'Failed to delete character ID: 9999' in log_output
    
    def test_character_server_with_metrics_collection(self):
        """Test character server integration with metrics collection."""
        from test_character_server import MockCharacterServer
        from collections import defaultdict
        import time
        
        # Metrics collector
        class MetricsCollector:
            def __init__(self):
                self.operation_counts = defaultdict(int)
                self.operation_times = defaultdict(list)
            
            def record_operation(self, operation, duration):
                self.operation_counts[operation] += 1
                self.operation_times[operation].append(duration)
            
            def get_average_time(self, operation):
                times = self.operation_times[operation]
                return sum(times) / len(times) if times else 0
        
        # Extend MockCharacterServer with metrics
        class MetricsCharacterServer(MockCharacterServer):
            def __init__(self):
                super().__init__()
                self.metrics = MetricsCollector()
            
            def create_character(self, character_data):
                start_time = time.time()
                result = super().create_character(character_data)
                duration = time.time() - start_time
                self.metrics.record_operation('create', duration)
                return result
            
            def get_character(self, character_id):
                start_time = time.time()
                result = super().get_character(character_id)
                duration = time.time() - start_time
                self.metrics.record_operation('get', duration)
                return result
        
        server = MetricsCharacterServer()
        
        # Perform operations
        char_data = {'name': 'MetricsHero', 'class': 'Warrior', 'level': 1}
        char = server.create_character(char_data)
        server.get_character(char['id'])
        server.get_character(char['id'])
        server.create_character({**char_data, 'name': 'MetricsHero2'})
        
        # Verify metrics collection
        assert server.metrics.operation_counts['create'] == 2
        assert server.metrics.operation_counts['get'] == 2
        assert server.metrics.get_average_time('create') >= 0
        assert server.metrics.get_average_time('get') >= 0
    
    def test_character_server_with_caching_layer(self):
        """Test character server integration with caching layer."""
        from test_character_server import MockCharacterServer
        
        # Simple cache implementation
        class SimpleCache:
            def __init__(self):
                self.cache = {}
                self.hits = 0
                self.misses = 0
            
            def get(self, key):
                if key in self.cache:
                    self.hits += 1
                    return self.cache[key]
                self.misses += 1
                return None
            
            def set(self, key, value):
                self.cache[key] = value
            
            def delete(self, key):
                if key in self.cache:
                    del self.cache[key]
        
        # Extend MockCharacterServer with caching
        class CachedCharacterServer(MockCharacterServer):
            def __init__(self):
                super().__init__()
                self.cache = SimpleCache()
            
            def get_character(self, character_id):
                cache_key = f"char_{character_id}"
                cached_char = self.cache.get(cache_key)
                if cached_char:
                    return cached_char
                
                char = super().get_character(character_id)
                if char:
                    self.cache.set(cache_key, char)
                return char
            
            def update_character(self, character_id, update_data):
                result = super().update_character(character_id, update_data)
                if result:
                    # Update cache
                    cache_key = f"char_{character_id}"
                    self.cache.set(cache_key, result)
                return result
            
            def delete_character(self, character_id):
                result = super().delete_character(character_id)
                if result:
                    # Remove from cache
                    cache_key = f"char_{character_id}"
                    self.cache.delete(cache_key)
                return result
        
        server = CachedCharacterServer()
        
        # Create character
        char_data = {'name': 'CachedHero', 'class': 'Warrior', 'level': 1}
        char = server.create_character(char_data)
        
        # First get - should miss cache
        result1 = server.get_character(char['id'])
        assert result1 is not None
        assert server.cache.misses == 1
        assert server.cache.hits == 0
        
        # Second get - should hit cache
        result2 = server.get_character(char['id'])
        assert result2 is not None
        assert server.cache.hits == 1
        
        # Update character - should update cache
        server.update_character(char['id'], {'level': 10})
        
        # Get updated character - should hit cache with updated data
        result3 = server.get_character(char['id'])
        assert result3['level'] == 10
        assert server.cache.hits == 2
        
        # Delete character - should remove from cache
        server.delete_character(char['id'])
        
        # Try to get deleted character - should miss cache and return None
        result4 = server.get_character(char['id'])
        assert result4 is None
        assert server.cache.misses == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])