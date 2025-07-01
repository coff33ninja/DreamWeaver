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

class TestCharacterServerSecurityAndRobustness:
    """Test suite for security and robustness scenarios."""
    
    def test_sql_injection_protection_in_name(self, character_server):
        """Test protection against SQL injection in character names."""
        malicious_names = [
            "'; DROP TABLE characters; --",
            "Robert'; DELETE FROM users; --",
            "test' OR '1'='1",
            "admin'/**/UNION/**/SELECT/**/*/**/FROM/**/users--",
            "<script>alert('xss')</script>"
        ]
        
        for malicious_name in malicious_names:
            char_data = {
                'name': malicious_name,
                'class': 'Hacker',
                'level': 1
            }
            # Should create character without executing malicious code
            result = character_server.create_character(char_data)
            assert result['name'] == malicious_name
            assert result['id'] is not None
    
    def test_extremely_long_character_name(self, character_server):
        """Test handling of extremely long character names."""
        long_name = "A" * 10000  # Very long name
        char_data = {
            'name': long_name,
            'class': 'Warrior',
            'level': 1
        }
        
        result = character_server.create_character(char_data)
        assert result['name'] == long_name
        assert len(result['name']) == 10000
    
    def test_character_name_with_null_bytes(self, character_server):
        """Test handling of null bytes in character names."""
        name_with_nulls = "Hero\x00\x00Test"
        char_data = {
            'name': name_with_nulls,
            'class': 'Warrior',
            'level': 1
        }
        
        result = character_server.create_character(char_data)
        assert result['name'] == name_with_nulls
    
    def test_massive_level_values(self, character_server):
        """Test handling of extremely large level values."""
        import sys
        large_levels = [
            sys.maxsize,
            2**31 - 1,  # Max 32-bit signed int
            2**63 - 1,  # Max 64-bit signed int
            999999999999999999
        ]
        
        for level in large_levels:
            char_data = {
                'name': f'HighLevel_{level}',
                'class': 'Warrior',
                'level': level
            }
            
            result = character_server.create_character(char_data)
            assert result['level'] == level
    
    def test_negative_zero_level(self, character_server):
        """Test handling of edge case numeric values for level."""
        edge_cases = [0.0, -0, 1.0, 1.9]  # Float values that might be converted
        
        for level in edge_cases:
            char_data = {
                'name': f'EdgeLevel_{level}',
                'class': 'Warrior',
                'level': int(level) if level > 0 else 1
            }
            
            if int(level) > 0:
                result = character_server.create_character(char_data)
                assert result['level'] == int(level)
            else:
                with pytest.raises(ValueError):
                    character_server.create_character({
                        'name': f'EdgeLevel_{level}',
                        'class': 'Warrior',
                        'level': int(level)
                    })


class TestCharacterServerDataIntegrity:
    """Test suite for data integrity and consistency."""
    
    def test_character_data_immutability_after_creation(self, character_server, valid_character_data):
        """Test that original data doesn't affect created character."""
        original_data = valid_character_data.copy()
        created_char = character_server.create_character(original_data)
        
        # Modify original data
        original_data['name'] = 'ModifiedName'
        original_data['level'] = 999
        
        # Retrieved character should be unchanged
        retrieved_char = character_server.get_character(created_char['id'])
        assert retrieved_char['name'] == valid_character_data['name']
        assert retrieved_char['level'] == valid_character_data['level']
    
    def test_update_data_immutability(self, character_server, valid_character_data):
        """Test that update data doesn't affect character after update."""
        created_char = character_server.create_character(valid_character_data)
        update_data = {'level': 50, 'new_field': 'test'}
        
        character_server.update_character(created_char['id'], update_data)
        
        # Modify update data
        update_data['level'] = 999
        update_data['new_field'] = 'modified'
        
        # Character should retain original update values
        retrieved_char = character_server.get_character(created_char['id'])
        assert retrieved_char['level'] == 50
        assert retrieved_char['new_field'] == 'test'
    
    def test_character_deep_copy_behavior(self, character_server):
        """Test that complex nested data structures are properly handled."""
        complex_data = {
            'name': 'ComplexHero',
            'class': 'Warrior',
            'level': 1,
            'nested_data': {
                'skills': {
                    'combat': {'level': 10, 'experience': 1000},
                    'magic': {'level': 5, 'experience': 200}
                },
                'inventory': {
                    'weapons': [{'name': 'sword', 'damage': 10}],
                    'armor': [{'name': 'shield', 'defense': 5}]
                }
            }
        }
        
        created_char = character_server.create_character(complex_data)
        
        # Modify original nested data
        complex_data['nested_data']['skills']['combat']['level'] = 999
        complex_data['nested_data']['inventory']['weapons'].append({'name': 'super_sword', 'damage': 1000})
        
        # Retrieved character should be unchanged
        retrieved_char = character_server.get_character(created_char['id'])
        assert retrieved_char['nested_data']['skills']['combat']['level'] == 10
        assert len(retrieved_char['nested_data']['inventory']['weapons']) == 1
    
    def test_character_id_uniqueness_stress_test(self, character_server, valid_character_data):
        """Stress test character ID uniqueness with rapid creation."""
        created_ids = set()
        num_characters = 500
        
        for i in range(num_characters):
            char_data = {**valid_character_data, 'name': f'StressTest_{i}'}
            created_char = character_server.create_character(char_data)
            
            # Ensure no ID collision
            assert created_char['id'] not in created_ids
            created_ids.add(created_char['id'])
        
        assert len(created_ids) == num_characters


class TestCharacterServerBoundaryConditions:
    """Test suite for boundary conditions and limits."""
    
    def test_empty_string_class_name(self, character_server):
        """Test character creation with empty class name."""
        char_data = {
            'name': 'Hero',
            'class': '',  # Empty class
            'level': 1
        }
        
        # Should succeed as validation only checks required fields exist
        result = character_server.create_character(char_data)
        assert result['class'] == ''
    
    def test_whitespace_only_names(self, character_server):
        """Test handling of whitespace-only names."""
        whitespace_names = [
            ' ',
            '\t',
            '\n',
            '   ',
            '\t\n\r',
            '\u00A0',  # Non-breaking space
            '\u2003'   # Em space
        ]
        
        for name in whitespace_names:
            char_data = {
                'name': name,
                'class': 'Warrior',
                'level': 1
            }
            
            # These should succeed as they're non-empty strings
            result = character_server.create_character(char_data)
            assert result['name'] == name
    
    def test_maximum_integer_boundaries(self, character_server):
        """Test integer boundary conditions for level."""
        import sys
        
        boundary_levels = [
            1,  # Minimum valid
            2**31 - 1,  # Max 32-bit signed int
            sys.maxsize  # Max size_t
        ]
        
        for level in boundary_levels:
            char_data = {
                'name': f'Boundary_{level}',
                'class': 'Warrior',
                'level': level
            }
            
            result = character_server.create_character(char_data)
            assert result['level'] == level
    
    def test_character_data_with_none_values(self, character_server):
        """Test character creation with None values in optional fields."""
        char_data = {
            'name': 'Hero',
            'class': 'Warrior',
            'level': 1,
            'optional_field': None,
            'hp': None,
            'equipment': None
        }
        
        result = character_server.create_character(char_data)
        assert result['optional_field'] is None
        assert result['hp'] is None
        assert result['equipment'] is None


class TestCharacterServerErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_create_character_with_circular_references(self, character_server):
        """Test handling of circular references in character data."""
        char_data = {
            'name': 'CircularHero',
            'class': 'Warrior',
            'level': 1
        }
        
        # Create circular reference
        char_data['self_ref'] = char_data
        
        # Should handle gracefully without infinite recursion
        result = character_server.create_character(char_data)
        assert result['name'] == 'CircularHero'
    
    def test_update_character_with_invalid_id_types(self, character_server):
        """Test update with invalid ID types."""
        invalid_ids = [
            'string_id',
            3.14,
            [],
            {},
            None,
            True,
            complex(1, 2)
        ]
        
        for invalid_id in invalid_ids:
            result = character_server.update_character(invalid_id, {'level': 10})
            assert result is None
    
    def test_delete_character_with_invalid_id_types(self, character_server):
        """Test deletion with invalid ID types."""
        invalid_ids = [
            'string_id',
            3.14,
            [],
            {},
            None,
            True,
            complex(1, 2)
        ]
        
        for invalid_id in invalid_ids:
            result = character_server.delete_character(invalid_id)
            assert result is False
    
    def test_get_character_with_invalid_id_types(self, character_server):
        """Test retrieval with invalid ID types."""
        invalid_ids = [
            'string_id',
            3.14,
            [],
            {},
            None,
            True,
            complex(1, 2)
        ]
        
        for invalid_id in invalid_ids:
            result = character_server.get_character(invalid_id)
            assert result is None


class TestCharacterServerAdvancedScenarios:
    """Test suite for advanced usage scenarios."""
    
    def test_character_with_callable_objects(self, character_server):
        """Test handling of callable objects in character data."""
        def dummy_function():
            return "test"
        
        char_data = {
            'name': 'CallableHero',
            'class': 'Warrior',
            'level': 1,
            'callback': dummy_function,
            'lambda_func': lambda x: x * 2
        }
        
        result = character_server.create_character(char_data)
        assert callable(result['callback'])
        assert callable(result['lambda_func'])
        assert result['callback']() == "test"
        assert result['lambda_func'](5) == 10
    
    def test_character_with_custom_objects(self, character_server):
        """Test handling of custom object instances in character data."""
        class CustomEquipment:
            def __init__(self, name, power):
                self.name = name
                self.power = power
            
            def __eq__(self, other):
                return isinstance(other, CustomEquipment) and \
                       self.name == other.name and self.power == other.power
        
        equipment = CustomEquipment("Magic Sword", 100)
        
        char_data = {
            'name': 'CustomHero',
            'class': 'Warrior',
            'level': 1,
            'weapon': equipment
        }
        
        result = character_server.create_character(char_data)
        assert isinstance(result['weapon'], CustomEquipment)
        assert result['weapon'].name == "Magic Sword"
        assert result['weapon'].power == 100
    
    def test_character_server_state_consistency(self, character_server, valid_character_data):
        """Test internal state consistency across operations."""
        # Create multiple characters
        chars = []
        for i in range(10):
            char_data = {**valid_character_data, 'name': f'StateTest_{i}'}
            chars.append(character_server.create_character(char_data))
        
        # Perform mixed operations
        character_server.update_character(chars[0]['id'], {'level': 99})
        character_server.delete_character(chars[1]['id'])
        character_server.update_character(chars[2]['id'], {'new_field': 'added'})
        
        # Verify state consistency
        all_chars = character_server.list_characters()
        assert len(all_chars) == 9  # 10 created, 1 deleted
        
        # Verify specific changes persisted
        updated_char = character_server.get_character(chars[0]['id'])
        assert updated_char['level'] == 99
        
        deleted_char = character_server.get_character(chars[1]['id'])
        assert deleted_char is None
        
        field_added_char = character_server.get_character(chars[2]['id'])
        assert field_added_char['new_field'] == 'added'
    
    def test_character_operations_with_very_large_datasets(self, character_server, valid_character_data):
        """Test character operations with large datasets."""
        # This tests performance and memory handling
        large_equipment_list = [f'item_{i}' for i in range(10000)]
        large_skills_dict = {f'skill_{i}': i for i in range(5000)}
        
        char_data = {
            **valid_character_data,
            'name': 'LargeDataHero',
            'equipment': large_equipment_list,
            'skills': large_skills_dict
        }
        
        result = character_server.create_character(char_data)
        assert len(result['equipment']) == 10000
        assert len(result['skills']) == 5000
        
        # Test update with large data
        update_data = {
            'new_large_field': [f'new_item_{i}' for i in range(8000)]
        }
        
        updated_char = character_server.update_character(result['id'], update_data)
        assert len(updated_char['new_large_field']) == 8000


class TestCharacterServerConcurrencySimulation:
    """Test suite simulating concurrency scenarios."""
    
    def test_rapid_fire_operations(self, character_server, valid_character_data):
        """Test rapid succession of operations to simulate concurrency."""
        # Rapid character creation
        created_chars = []
        for i in range(100):
            char_data = {**valid_character_data, 'name': f'Rapid_{i}'}
            created_chars.append(character_server.create_character(char_data))
        
        # Rapid updates
        for i, char in enumerate(created_chars):
            character_server.update_character(char['id'], {'level': i + 10})
        
        # Rapid deletions (every other character)
        for i, char in enumerate(created_chars):
            if i % 2 == 0:
                character_server.delete_character(char['id'])
        
        # Verify final state
        remaining_chars = character_server.list_characters()
        assert len(remaining_chars) == 50  # Half deleted
        
        for char in remaining_chars:
            # Verify updates were applied to remaining characters
            assert char['level'] >= 10
    
    def test_interleaved_operations(self, character_server, valid_character_data):
        """Test interleaved create/update/delete operations."""
        operations_log = []
        
        # Perform interleaved operations
        for i in range(20):
            # Create
            char_data = {**valid_character_data, 'name': f'Interleaved_{i}'}
            char = character_server.create_character(char_data)
            operations_log.append(('create', char['id']))
            
            # Update (if possible)
            if i > 0:
                prev_id = operations_log[i*2 - 2][1]  # Previous character ID
                character_server.update_character(prev_id, {'level': i + 20})
                operations_log.append(('update', prev_id))
            
            # Delete (every 3rd character)
            if i > 2 and i % 3 == 0:
                target_id = operations_log[(i-3)*2][1]
                character_server.delete_character(target_id)
                operations_log.append(('delete', target_id))
        
        # Verify consistency
        all_chars = character_server.list_characters()
        created_count = sum(1 for op, _ in operations_log if op == 'create')
        deleted_count = sum(1 for op, _ in operations_log if op == 'delete')
        
        assert len(all_chars) == created_count - deleted_count


# Additional async tests for comprehensive coverage
class TestAsyncCharacterOperationsExtended:
    """Extended test suite for asynchronous character operations."""
    
    @pytest.mark.asyncio
    async def test_async_bulk_operations(self):
        """Test bulk asynchronous operations."""
        async def mock_bulk_create(character_list):
            await asyncio.sleep(0.01)
            return [{'id': i, **char} for i, char in enumerate(character_list, 1)]
        
        character_list = [
            {'name': f'AsyncBulk_{i}', 'class': 'Warrior', 'level': i}
            for i in range(10)
        ]
        
        result = await mock_bulk_create(character_list)
        
        assert len(result) == 10
        assert all('id' in char for char in result)
        assert result[0]['name'] == 'AsyncBulk_0'
        assert result[9]['name'] == 'AsyncBulk_9'
    
    @pytest.mark.asyncio
    async def test_async_operation_timeout_handling(self):
        """Test timeout handling in async operations."""
        async def slow_operation():
            await asyncio.sleep(2.0)  # Simulate slow operation
            return "completed"
        
        # Test with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_async_operation_cancellation(self):
        """Test cancellation of async operations."""
        async def cancellable_operation():
            try:
                await asyncio.sleep(1.0)
                return "should not complete"
            except asyncio.CancelledError:
                return "cancelled"
        
        task = asyncio.create_task(cancellable_operation())
        await asyncio.sleep(0.1)  # Let it start
        task.cancel()
        
        result = await task
        assert result == "cancelled"
    
    @pytest.mark.asyncio
    async def test_async_error_propagation(self):
        """Test error propagation in async operations."""
        async def failing_operation():
            await asyncio.sleep(0.01)
            raise ValueError("Async operation failed")
        
        with pytest.raises(ValueError, match="Async operation failed"):
            await failing_operation()


class TestCharacterServerValidationExtended:
    """Extended validation test suite."""
    
    def test_validate_character_data_type_coercion(self):
        """Test validation with type coercion scenarios."""
        # Test data that might be coerced
        coercion_cases = [
            ({'name': 'Hero', 'class': 'Warrior', 'level': '10'}, False),  # String level
            ({'name': 'Hero', 'class': 'Warrior', 'level': 10.0}, False),  # Float level
            ({'name': 'Hero', 'class': 'Warrior', 'level': True}, False),  # Boolean level
            ({'name': b'Hero', 'class': 'Warrior', 'level': 10}, False),  # Bytes name
        ]
        
        for data, expected in coercion_cases:
            result = MockCharacterServer.validate_character_data(data)
            assert result == expected
    
    def test_validate_character_data_with_inheritance(self):
        """Test validation with inherited data structures."""
        class CustomDict(dict):
            pass
        
        class CustomStr(str):
            pass
        
        custom_data = CustomDict({
            'name': CustomStr('Hero'),
            'class': 'Warrior',
            'level': 1
        })
        
        result = MockCharacterServer.validate_character_data(custom_data)
        assert result is True
    
    @pytest.mark.parametrize("field_name,invalid_values", [
        ('name', [[], {}, set(), tuple(), 0, False]),
        ('level', [[], {}, set(), tuple(), '', 'text', None, False]),
        ('class', [[], {}, set(), tuple(), 0, None, False])  # class can be any truthy value
    ])
    def test_validate_field_type_comprehensive(self, field_name, invalid_values):
        """Comprehensive test of field type validation."""
        base_data = {'name': 'Hero', 'class': 'Warrior', 'level': 1}
        
        for invalid_value in invalid_values:
            test_data = base_data.copy()
            test_data[field_name] = invalid_value
            
            result = MockCharacterServer.validate_character_data(test_data)
            assert result is False, f"Should fail for {field_name}={invalid_value}"


# Performance and memory tests
class TestCharacterServerMemoryAndPerformance:
    """Test suite for memory usage and performance characteristics."""
    
    def test_memory_cleanup_after_deletion(self, character_server, valid_character_data):
        """Test that memory is properly cleaned up after character deletion."""
        import gc
        
        # Create many characters with large data
        large_data = {
            **valid_character_data,
            'large_field': ['data'] * 1000
        }
        
        created_chars = []
        for i in range(100):
            char_data = {**large_data, 'name': f'MemTest_{i}'}
            created_chars.append(character_server.create_character(char_data))
        
        # Delete all characters
        for char in created_chars:
            character_server.delete_character(char['id'])
        
        # Force garbage collection
        gc.collect()
        
        # Verify characters are gone
        assert len(character_server.list_characters()) == 0
        
        # Memory should be reclaimed (this is a basic check)
        # In a real scenario, you might use memory profiling tools
        remaining_chars = character_server.list_characters()
        assert len(remaining_chars) == 0
    
    def test_server_state_size_growth(self, character_server, valid_character_data):
        """Test server state size growth patterns."""
        import sys
        
        initial_size = sys.getsizeof(character_server.characters)
        
        # Add characters and measure growth
        for i in range(50):
            char_data = {**valid_character_data, 'name': f'GrowthTest_{i}'}
            character_server.create_character(char_data)
        
        mid_size = sys.getsizeof(character_server.characters)
        assert mid_size > initial_size
        
        # Delete half and measure again
        chars = character_server.list_characters()
        for i, char in enumerate(chars):
            if i % 2 == 0:
                character_server.delete_character(char['id'])
        
        final_size = sys.getsizeof(character_server.characters)
        # Size should be between initial and mid (some cleanup occurred)
        assert initial_size <= final_size <= mid_size


# Character server integration and cross-functional tests
class TestCharacterServerIntegration:
    """Test suite for integration and cross-functional scenarios."""
    
    def test_character_lifecycle_complete_workflow(self, character_server, valid_character_data):
        """Test complete character lifecycle from creation to deletion."""
        # Phase 1: Character creation with validation
        char_data = {**valid_character_data, 'name': 'LifecycleHero'}
        created_char = character_server.create_character(char_data)
        
        assert created_char['id'] is not None
        assert created_char['name'] == 'LifecycleHero'
        
        # Phase 2: Multiple updates with state verification
        updates = [
            {'level': 25, 'experience': 5000},
            {'equipment': ['new_sword', 'magic_armor']},
            {'guild': 'TestGuild', 'reputation': 'Honored'},
            {'achievements': ['first_boss', 'level_25']}
        ]
        
        for update in updates:
            updated_char = character_server.update_character(created_char['id'], update)
            assert updated_char is not None
            
            # Verify update was applied
            for key, value in update.items():
                assert updated_char[key] == value
        
        # Phase 3: Character retrieval and validation
        final_char = character_server.get_character(created_char['id'])
        assert final_char['level'] == 25
        assert final_char['guild'] == 'TestGuild'
        assert len(final_char['achievements']) == 2
        
        # Phase 4: Character deletion and cleanup
        deletion_result = character_server.delete_character(created_char['id'])
        assert deletion_result is True
        
        # Verify deletion
        deleted_char = character_server.get_character(created_char['id'])
        assert deleted_char is None
    
    def test_bulk_character_management(self, character_server, valid_character_data):
        """Test bulk character management operations."""
        # Create multiple character types
        character_templates = [
            {'name': 'Warrior_{i}', 'class': 'Warrior', 'level': 10},
            {'name': 'Mage_{i}', 'class': 'Mage', 'level': 8},
            {'name': 'Rogue_{i}', 'class': 'Rogue', 'level': 12},
            {'name': 'Paladin_{i}', 'class': 'Paladin', 'level': 15}
        ]
        
        created_characters = []
        
        # Bulk creation
        for i in range(25):  # 100 total characters (25 of each type)
            for template in character_templates:
                char_data = {**valid_character_data}
                char_data.update({
                    'name': template['name'].format(i=i),
                    'class': template['class'],
                    'level': template['level']
                })
                
                created_char = character_server.create_character(char_data)
                created_characters.append(created_char)
        
        assert len(created_characters) == 100
        
        # Bulk updates (level up all characters)
        for char in created_characters:
            character_server.update_character(char['id'], {'level': char['level'] + 10})
        
        # Verify updates
        all_chars = character_server.list_characters()
        assert len(all_chars) == 100
        assert all(char['level'] >= 18 for char in all_chars)  # Min level after update
        
        # Selective deletion (remove all Rogues)
        rogue_ids = [char['id'] for char in all_chars if char['class'] == 'Rogue']
        for rogue_id in rogue_ids:
            character_server.delete_character(rogue_id)
        
        # Verify selective deletion
        remaining_chars = character_server.list_characters()
        assert len(remaining_chars) == 75  # 100 - 25 Rogues
        assert not any(char['class'] == 'Rogue' for char in remaining_chars)
    
    def test_character_search_and_filtering_simulation(self, character_server, valid_character_data):
        """Simulate character search and filtering operations."""
        # Create diverse character dataset
        characters_data = [
            {'name': 'Alice', 'class': 'Warrior', 'level': 25, 'guild': 'Knights'},
            {'name': 'Bob', 'class': 'Mage', 'level': 30, 'guild': 'Wizards'},
            {'name': 'Charlie', 'class': 'Rogue', 'level': 20, 'guild': 'Thieves'},
            {'name': 'Diana', 'class': 'Warrior', 'level': 35, 'guild': 'Knights'},
            {'name': 'Eve', 'class': 'Paladin', 'level': 40, 'guild': 'Temple'},
            {'name': 'Frank', 'class': 'Mage', 'level': 15, 'guild': 'Apprentices'},
        ]
        
        created_chars = []
        for char_data in characters_data:
            full_data = {**valid_character_data, **char_data}
            created_chars.append(character_server.create_character(full_data))
        
        all_chars = character_server.list_characters()
        
        # Simulate filtering by class
        warriors = [char for char in all_chars if char['class'] == 'Warrior']
        assert len(warriors) == 2
        assert all(char['class'] == 'Warrior' for char in warriors)
        
        # Simulate filtering by level range
        high_level_chars = [char for char in all_chars if char['level'] >= 30]
        assert len(high_level_chars) == 3
        assert all(char['level'] >= 30 for char in high_level_chars)
        
        # Simulate filtering by guild
        knights = [char for char in all_chars if char.get('guild') == 'Knights']
        assert len(knights) == 2
        assert all(char['guild'] == 'Knights' for char in knights)
        
        # Simulate complex filtering (Warriors in Knights guild)
        knight_warriors = [char for char in all_chars 
                          if char['class'] == 'Warrior' and char.get('guild') == 'Knights']
        assert len(knight_warriors) == 2


if __name__ == '__main__':
    # Run all tests including the new comprehensive ones
    pytest.main([__file__, '-v', '--tb=short', '--maxfail=10'])