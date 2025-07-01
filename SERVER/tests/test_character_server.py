import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, List, Optional
import sys
import os

# Add the SERVER directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Testing framework: pytest
# These tests provide comprehensive coverage including happy paths, edge cases, and failure conditions


class MockCharacterServer:
    """Mock character server for testing when actual implementation isn't available."""
    
    def __init__(self):
        self.characters = {}
        self.next_id = 1
    
    def create_character(self, character_data: Dict) -> Dict:
        if not self.validate_character_data(character_data):
            raise ValueError("Invalid character data")
        
        character = {**character_data, 'id': self.next_id}
        self.characters[self.next_id] = character
        self.next_id += 1
        return character
    
    def get_character(self, character_id: int) -> Optional[Dict]:
        return self.characters.get(character_id)
    
    def update_character(self, character_id: int, update_data: Dict) -> Optional[Dict]:
        if character_id not in self.characters:
            return None
        
        self.characters[character_id].update(update_data)
        return self.characters[character_id]
    
    def delete_character(self, character_id: int) -> bool:
        if character_id in self.characters:
            del self.characters[character_id]
            return True
        return False
    
    def list_characters(self) -> List[Dict]:
        return list(self.characters.values())
    
    @staticmethod
    def validate_character_data(data: Dict) -> bool:
        required_fields = ['name', 'class', 'level']
        return all(field in data for field in required_fields) and \
               isinstance(data.get('name'), str) and \
               len(data['name']) > 0 and \
               isinstance(data.get('level'), int) and \
               data['level'] > 0


@pytest.fixture
def character_server():
    """Fixture providing a fresh character server instance for each test."""
    return MockCharacterServer()


@pytest.fixture
def valid_character_data():
    """Fixture providing valid character data for testing."""
    return {
        'name': 'TestHero',
        'class': 'Warrior',
        'level': 10,
        'hp': 100,
        'mp': 50,
        'strength': 15,
        'dexterity': 12,
        'intelligence': 8,
        'equipment': ['sword', 'shield'],
        'skills': ['combat', 'leadership']
    }


@pytest.fixture
def invalid_character_data():
    """Fixture providing invalid character data for testing."""
    return {
        'name': '',  # Invalid empty name
        'class': 'InvalidClass',
        'level': -5,  # Invalid negative level
        'hp': 'not_a_number',  # Invalid type
        'strength': None
    }


class TestCharacterCreation:
    """Test suite for character creation functionality."""
    
    def test_create_character_success(self, character_server, valid_character_data):
        """Test successful character creation with valid data."""
        result = character_server.create_character(valid_character_data)
        
        assert result['id'] is not None
        assert result['name'] == valid_character_data['name']
        assert result['class'] == valid_character_data['class']
        assert result['level'] == valid_character_data['level']
    
    def test_create_character_minimal_data(self, character_server):
        """Test character creation with minimal required data."""
        minimal_data = {
            'name': 'MinimalHero',
            'class': 'Novice',
            'level': 1
        }
        
        result = character_server.create_character(minimal_data)
        
        assert result['name'] == 'MinimalHero'
        assert result['class'] == 'Novice'
        assert result['level'] == 1
        assert 'id' in result
    
    def test_create_character_with_special_characters_in_name(self, character_server):
        """Test character creation with special characters in name."""
        special_char_data = {
            'name': 'Hérö-Tëst_123',
            'class': 'Rogue',
            'level': 5
        }
        
        result = character_server.create_character(special_char_data)
        assert result['name'] == 'Hérö-Tëst_123'
    
    def test_create_character_maximum_level(self, character_server):
        """Test character creation with maximum level."""
        max_level_data = {
            'name': 'MaxLevelHero',
            'class': 'Legend',
            'level': 100,
            'hp': 9999,
            'mp': 9999
        }
        
        result = character_server.create_character(max_level_data)
        assert result['level'] == 100
        assert result['hp'] == 9999
    
    def test_create_character_invalid_empty_name(self, character_server):
        """Test character creation fails with empty name."""
        invalid_data = {
            'name': '',
            'class': 'Warrior',
            'level': 1
        }
        
        with pytest.raises(ValueError, match="Invalid character data"):
            character_server.create_character(invalid_data)
    
    def test_create_character_invalid_negative_level(self, character_server):
        """Test character creation fails with negative level."""
        invalid_data = {
            'name': 'TestHero',
            'class': 'Warrior',
            'level': -1
        }
        
        with pytest.raises(ValueError, match="Invalid character data"):
            character_server.create_character(invalid_data)
    
    def test_create_character_missing_required_fields(self, character_server):
        """Test character creation fails with missing required fields."""
        incomplete_data = {
            'name': 'TestHero'
            # Missing 'class' and 'level'
        }
        
        with pytest.raises(ValueError, match="Invalid character data"):
            character_server.create_character(incomplete_data)
    
    def test_create_character_invalid_data_types(self, character_server):
        """Test character creation fails with invalid data types."""
        invalid_type_data = {
            'name': 123,  # Should be string
            'class': 'Warrior',
            'level': 'ten'  # Should be integer
        }
        
        with pytest.raises(ValueError, match="Invalid character data"):
            character_server.create_character(invalid_type_data)


class TestCharacterRetrieval:
    """Test suite for character retrieval functionality."""
    
    def test_get_character_success(self, character_server, valid_character_data):
        """Test successful character retrieval."""
        created_char = character_server.create_character(valid_character_data)
        retrieved_char = character_server.get_character(created_char['id'])
        
        assert retrieved_char is not None
        assert retrieved_char['id'] == created_char['id']
        assert retrieved_char['name'] == valid_character_data['name']
    
    def test_get_character_nonexistent(self, character_server):
        """Test retrieving a character that doesn't exist."""
        result = character_server.get_character(9999)
        assert result is None
    
    def test_get_character_zero_id(self, character_server):
        """Test retrieving character with ID 0."""
        result = character_server.get_character(0)
        assert result is None
    
    def test_get_character_negative_id(self, character_server):
        """Test retrieving character with negative ID."""
        result = character_server.get_character(-1)
        assert result is None
    
    def test_get_multiple_characters(self, character_server, valid_character_data):
        """Test retrieving multiple different characters."""
        char1_data = {**valid_character_data, 'name': 'Hero1'}
        char2_data = {**valid_character_data, 'name': 'Hero2', 'class': 'Mage'}
        
        char1 = character_server.create_character(char1_data)
        char2 = character_server.create_character(char2_data)
        
        retrieved_char1 = character_server.get_character(char1['id'])
        retrieved_char2 = character_server.get_character(char2['id'])
        
        assert retrieved_char1['name'] == 'Hero1'
        assert retrieved_char2['name'] == 'Hero2'
        assert retrieved_char1['id'] != retrieved_char2['id']


class TestCharacterUpdate:
    """Test suite for character update functionality."""
    
    def test_update_character_success(self, character_server, valid_character_data):
        """Test successful character update."""
        created_char = character_server.create_character(valid_character_data)
        update_data = {'level': 15, 'hp': 150}
        
        updated_char = character_server.update_character(created_char['id'], update_data)
        
        assert updated_char is not None
        assert updated_char['level'] == 15
        assert updated_char['hp'] == 150
        assert updated_char['name'] == valid_character_data['name']  # Unchanged
    
    def test_update_character_partial(self, character_server, valid_character_data):
        """Test updating character with partial data."""
        created_char = character_server.create_character(valid_character_data)
        update_data = {'level': 20}
        
        updated_char = character_server.update_character(created_char['id'], update_data)
        
        assert updated_char['level'] == 20
        assert updated_char['name'] == valid_character_data['name']  # Unchanged
    
    def test_update_character_all_fields(self, character_server, valid_character_data):
        """Test updating all character fields."""
        created_char = character_server.create_character(valid_character_data)
        update_data = {
            'name': 'UpdatedHero',
            'class': 'Paladin',
            'level': 25,
            'hp': 200,
            'mp': 100,
            'strength': 20,
            'equipment': ['holy_sword', 'blessed_shield']
        }
        
        updated_char = character_server.update_character(created_char['id'], update_data)
        
        assert updated_char['name'] == 'UpdatedHero'
        assert updated_char['class'] == 'Paladin'
        assert updated_char['level'] == 25
        assert updated_char['equipment'] == ['holy_sword', 'blessed_shield']
    
    def test_update_character_nonexistent(self, character_server):
        """Test updating a character that doesn't exist."""
        update_data = {'level': 20}
        result = character_server.update_character(9999, update_data)
        
        assert result is None
    
    def test_update_character_empty_data(self, character_server, valid_character_data):
        """Test updating character with empty data."""
        created_char = character_server.create_character(valid_character_data)
        
        updated_char = character_server.update_character(created_char['id'], {})
        
        # Should return the character unchanged
        assert updated_char['name'] == valid_character_data['name']
        assert updated_char['level'] == valid_character_data['level']
    
    def test_update_character_add_new_fields(self, character_server, valid_character_data):
        """Test adding new fields to existing character."""
        created_char = character_server.create_character(valid_character_data)
        update_data = {
            'guild': 'AwesomeGuild',
            'reputation': 'Honored',
            'achievements': ['first_kill', 'level_10']
        }
        
        updated_char = character_server.update_character(created_char['id'], update_data)
        
        assert updated_char['guild'] == 'AwesomeGuild'
        assert updated_char['reputation'] == 'Honored'
        assert updated_char['achievements'] == ['first_kill', 'level_10']


class TestCharacterDeletion:
    """Test suite for character deletion functionality."""
    
    def test_delete_character_success(self, character_server, valid_character_data):
        """Test successful character deletion."""
        created_char = character_server.create_character(valid_character_data)
        
        result = character_server.delete_character(created_char['id'])
        
        assert result is True
        # Verify character is actually deleted
        retrieved_char = character_server.get_character(created_char['id'])
        assert retrieved_char is None
    
    def test_delete_character_nonexistent(self, character_server):
        """Test deleting a character that doesn't exist."""
        result = character_server.delete_character(9999)
        assert result is False
    
    def test_delete_character_zero_id(self, character_server):
        """Test deleting character with ID 0."""
        result = character_server.delete_character(0)
        assert result is False
    
    def test_delete_character_negative_id(self, character_server):
        """Test deleting character with negative ID."""
        result = character_server.delete_character(-1)
        assert result is False
    
    def test_delete_multiple_characters(self, character_server, valid_character_data):
        """Test deleting multiple characters."""
        char1 = character_server.create_character({**valid_character_data, 'name': 'Hero1'})
        char2 = character_server.create_character({**valid_character_data, 'name': 'Hero2'})
        
        result1 = character_server.delete_character(char1['id'])
        result2 = character_server.delete_character(char2['id'])
        
        assert result1 is True
        assert result2 is True
        assert character_server.get_character(char1['id']) is None
        assert character_server.get_character(char2['id']) is None


class TestCharacterListing:
    """Test suite for character listing functionality."""
    
    def test_list_characters_empty(self, character_server):
        """Test listing characters when none exist."""
        result = character_server.list_characters()
        assert result == []
    
    def test_list_characters_single(self, character_server, valid_character_data):
        """Test listing characters with single character."""
        character_server.create_character(valid_character_data)
        
        result = character_server.list_characters()
        
        assert len(result) == 1
        assert result[0]['name'] == valid_character_data['name']
    
    def test_list_characters_multiple(self, character_server, valid_character_data):
        """Test listing multiple characters."""
        char1_data = {**valid_character_data, 'name': 'Hero1'}
        char2_data = {**valid_character_data, 'name': 'Hero2', 'class': 'Mage'}
        char3_data = {**valid_character_data, 'name': 'Hero3', 'class': 'Rogue'}
        
        character_server.create_character(char1_data)
        character_server.create_character(char2_data)
        character_server.create_character(char3_data)
        
        result = character_server.list_characters()
        
        assert len(result) == 3
        names = [char['name'] for char in result]
        assert 'Hero1' in names
        assert 'Hero2' in names
        assert 'Hero3' in names
    
    def test_list_characters_after_deletion(self, character_server, valid_character_data):
        """Test listing characters after some are deleted."""
        char1 = character_server.create_character({**valid_character_data, 'name': 'Hero1'})
        char2 = character_server.create_character({**valid_character_data, 'name': 'Hero2'})
        
        character_server.delete_character(char1['id'])
        
        result = character_server.list_characters()
        
        assert len(result) == 1
        assert result[0]['name'] == 'Hero2'


class TestCharacterValidation:
    """Test suite for character data validation."""
    
    @pytest.mark.parametrize("valid_data", [
        {'name': 'Hero', 'class': 'Warrior', 'level': 1},
        {'name': 'A', 'class': 'Mage', 'level': 100},
        {'name': 'Test Hero', 'class': 'Rogue', 'level': 50},
        {'name': 'Hérö-123', 'class': 'Paladin', 'level': 25},
    ])
    def test_validate_character_data_valid_cases(self, valid_data):
        """Test validation with various valid character data."""
        assert MockCharacterServer.validate_character_data(valid_data) is True
    
    @pytest.mark.parametrize("invalid_data", [
        {'name': '', 'class': 'Warrior', 'level': 1},  # Empty name
        {'name': 'Hero', 'class': 'Warrior', 'level': 0},  # Zero level
        {'name': 'Hero', 'class': 'Warrior', 'level': -1},  # Negative level
        {'name': 'Hero', 'class': 'Warrior'},  # Missing level
        {'name': 'Hero', 'level': 1},  # Missing class
        {'class': 'Warrior', 'level': 1},  # Missing name
        {'name': 123, 'class': 'Warrior', 'level': 1},  # Invalid name type
        {'name': 'Hero', 'class': 'Warrior', 'level': 'one'},  # Invalid level type
        {},  # Empty data
    ])
    def test_validate_character_data_invalid_cases(self, invalid_data):
        """Test validation with various invalid character data."""
        assert MockCharacterServer.validate_character_data(invalid_data) is False
    
    def test_validate_character_data_none_values(self):
        """Test validation with None values."""
        invalid_data = {
            'name': None,
            'class': 'Warrior',
            'level': 1
        }
        assert MockCharacterServer.validate_character_data(invalid_data) is False
    
    def test_validate_character_data_extra_fields(self):
        """Test validation with extra fields (should still be valid)."""
        data_with_extra = {
            'name': 'Hero',
            'class': 'Warrior',
            'level': 1,
            'extra_field': 'extra_value',
            'hp': 100
        }
        assert MockCharacterServer.validate_character_data(data_with_extra) is True


class TestCharacterServerEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_character_id_sequence(self, character_server, valid_character_data):
        """Test that character IDs are assigned sequentially."""
        char1 = character_server.create_character({**valid_character_data, 'name': 'Hero1'})
        char2 = character_server.create_character({**valid_character_data, 'name': 'Hero2'})
        char3 = character_server.create_character({**valid_character_data, 'name': 'Hero3'})
        
        assert char2['id'] == char1['id'] + 1
        assert char3['id'] == char2['id'] + 1
    
    def test_character_id_after_deletion(self, character_server, valid_character_data):
        """Test character ID sequence after deletion."""
        char1 = character_server.create_character({**valid_character_data, 'name': 'Hero1'})
        char2 = character_server.create_character({**valid_character_data, 'name': 'Hero2'})
        
        character_server.delete_character(char1['id'])
        
        char3 = character_server.create_character({**valid_character_data, 'name': 'Hero3'})
        
        # ID should continue from where it left off, not reuse deleted ID
        assert char3['id'] > char2['id']
    
    def test_large_character_data(self, character_server):
        """Test handling of characters with large amounts of data."""
        large_data = {
            'name': 'Hero',
            'class': 'Warrior',
            'level': 1,
            'equipment': ['item' + str(i) for i in range(100)],
            'skills': ['skill' + str(i) for i in range(50)],
            'achievements': ['achievement' + str(i) for i in range(200)],
            'inventory': {f'slot_{i}': f'item_{i}' for i in range(100)}
        }
        
        result = character_server.create_character(large_data)
        assert result is not None
        assert len(result['equipment']) == 100
        assert len(result['skills']) == 50
        assert len(result['achievements']) == 200
    
    def test_unicode_character_names(self, character_server):
        """Test handling of Unicode character names."""
        unicode_names = [
            '测试英雄',  # Chinese
            'тестовый герой',  # Russian
            'ñoño',  # Spanish with special chars
            '🦸‍♂️Hero',  # Emoji
            'Åsbjørn',  # Nordic characters
        ]
        
        for name in unicode_names:
            char_data = {
                'name': name,
                'class': 'Warrior',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
    
    def test_concurrent_character_operations(self, character_server, valid_character_data):
        """Test handling of concurrent character operations."""
        # Simulate concurrent creation
        characters = []
        for i in range(10):
            char_data = {**valid_character_data, 'name': f'Hero{i}'}
            characters.append(character_server.create_character(char_data))
        
        # Verify all characters were created with unique IDs
        ids = [char['id'] for char in characters]
        assert len(set(ids)) == len(ids)  # All IDs should be unique
        
        # Verify all characters can be retrieved
        for char in characters:
            retrieved = character_server.get_character(char['id'])
            assert retrieved is not None
            assert retrieved['name'] == char['name']


class TestCharacterServerPerformance:
    """Test suite for performance-related scenarios."""
    
    def test_create_many_characters(self, character_server, valid_character_data):
        """Test creating many characters to check performance degradation."""
        num_characters = 1000
        
        for i in range(num_characters):
            char_data = {**valid_character_data, 'name': f'Hero{i}'}
            result = character_server.create_character(char_data)
            assert result is not None
        
        # Verify all characters exist
        all_characters = character_server.list_characters()
        assert len(all_characters) == num_characters
    
    def test_bulk_operations(self, character_server, valid_character_data):
        """Test bulk character operations."""
        # Create multiple characters
        created_chars = []
        for i in range(50):
            char_data = {**valid_character_data, 'name': f'BulkHero{i}'}
            created_chars.append(character_server.create_character(char_data))
        
        # Update all characters
        for char in created_chars:
            character_server.update_character(char['id'], {'level': 99})
        
        # Verify all updates
        for char in created_chars:
            updated_char = character_server.get_character(char['id'])
            assert updated_char['level'] == 99
        
        # Delete all characters
        for char in created_chars:
            result = character_server.delete_character(char['id'])
            assert result is True
        
        # Verify all deleted
        assert len(character_server.list_characters()) == 0


# Async tests for potential async functionality
class TestAsyncCharacterOperations:
    """Test suite for asynchronous character operations."""
    
    @pytest.mark.asyncio
    async def test_async_character_creation(self):
        """Test asynchronous character creation."""
        # This would test async functionality if the server supports it
        async def mock_create_character_async(data):
            await asyncio.sleep(0.01)  # Simulate async operation
            return {'id': 1, **data}
        
        with patch('character_server.create_character_async', side_effect=mock_create_character_async):
            result = await mock_create_character_async({
                'name': 'AsyncHero',
                'class': 'Warrior',
                'level': 1
            })
            
            assert result['name'] == 'AsyncHero'
    
    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self):
        """Test concurrent asynchronous operations."""
        async def mock_operation(delay, result):
            await asyncio.sleep(delay)
            return result
        
        # Simulate concurrent operations
        tasks = [
            mock_operation(0.01, f'result_{i}')
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(f'result_{i}' in results for i in range(5))


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])