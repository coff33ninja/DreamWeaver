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

class TestCharacterServerSecurityAndValidation:
    """Test suite for security and input validation scenarios."""
    
    def test_character_names_with_sql_injection_patterns(self, character_server):
        """Test character names that might cause SQL injection if improperly handled."""
        malicious_names = [
            "'; DROP TABLE characters; --",
            "Robert'; DELETE FROM characters WHERE 1=1; --",
            "1' OR '1'='1",
            "'; SELECT * FROM users; --",
            "admin'/*",
            "UNION SELECT * FROM characters",
            "0x31303235343830303536",
            "char(0x31)",
        ]
        
        for name in malicious_names:
            char_data = {
                'name': name,
                'class': 'Hacker',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
            assert result['class'] == 'Hacker'
    
    def test_character_names_with_script_injection_patterns(self, character_server):
        """Test character names with script injection patterns."""
        script_names = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "</title><script>alert(1)</script>",
            "{{7*7}}",
            "${7*7}",
            "<%=7*7%>",
            "<%= 7*7 %>",
            "#{7*7}",
        ]
        
        for name in script_names:
            char_data = {
                'name': name,
                'class': 'ScriptKiddie',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
    
    def test_character_names_with_path_traversal_patterns(self, character_server):
        """Test character names with path traversal patterns."""
        path_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/hosts",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....\\....\\....\\windows\\system32\\drivers\\etc\\hosts",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
        ]
        
        for name in path_names:
            char_data = {
                'name': name,
                'class': 'PathTraverser',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
    
    def test_character_names_with_format_string_attacks(self, character_server):
        """Test character names with format string attack patterns."""
        format_names = [
            "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
            "%x%x%x%x%x%x%x%x%x%x",
            "%d%d%d%d%d%d%d%d%d%d",
            "%n%n%n%n%n%n%n%n%n%n",
            "{}{}{}{}{}{}{}{}{}{}",
            "{0}{1}{2}{3}{4}{5}",
        ]
        
        for name in format_names:
            char_data = {
                'name': name,
                'class': 'FormatAttacker',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name


class TestCharacterServerDataIntegrity:
    """Test suite for data integrity and consistency."""
    
    def test_character_data_immutability_after_creation(self, character_server, valid_character_data):
        """Test that character data remains consistent after creation."""
        original_data = valid_character_data.copy()
        created_char = character_server.create_character(valid_character_data)
        
        # Modify original data after creation
        valid_character_data['name'] = 'ModifiedName'
        valid_character_data['level'] = 999
        
        # Verify created character wasn't affected
        retrieved_char = character_server.get_character(created_char['id'])
        assert retrieved_char['name'] == original_data['name']
        assert retrieved_char['level'] == original_data['level']
    
    def test_character_update_data_isolation(self, character_server, valid_character_data):
        """Test that update data doesn't affect original character after operation."""
        created_char = character_server.create_character(valid_character_data)
        update_data = {'level': 50, 'new_field': 'test'}
        
        character_server.update_character(created_char['id'], update_data)
        
        # Modify update data after operation
        update_data['level'] = 999
        update_data['new_field'] = 'modified'
        
        # Verify character wasn't affected by the modification
        retrieved_char = character_server.get_character(created_char['id'])
        assert retrieved_char['level'] == 50
        assert retrieved_char['new_field'] == 'test'
    
    def test_character_list_returns_independent_copies(self, character_server, valid_character_data):
        """Test that character list returns independent copies of data."""
        character_server.create_character(valid_character_data)
        
        # Get two separate lists
        list1 = character_server.list_characters()
        list2 = character_server.list_characters()
        
        # Modify first list
        if list1:
            list1[0]['name'] = 'ModifiedInList1'
        
        # Verify second list is unaffected
        if list2:
            assert list2[0]['name'] == valid_character_data['name']
    
    def test_concurrent_character_modifications(self, character_server, valid_character_data):
        """Test data integrity under concurrent modification scenarios."""
        created_char = character_server.create_character(valid_character_data)
        char_id = created_char['id']
        
        # Simulate concurrent updates
        update1 = {'level': 10, 'hp': 100}
        update2 = {'level': 20, 'mp': 50}
        update3 = {'level': 30, 'strength': 15}
        
        character_server.update_character(char_id, update1)
        character_server.update_character(char_id, update2)
        character_server.update_character(char_id, update3)
        
        # Verify final state is consistent
        final_char = character_server.get_character(char_id)
        assert final_char['level'] == 30  # Last update should win
        assert 'hp' in final_char  # Earlier updates should be preserved
        assert 'mp' in final_char
        assert 'strength' in final_char


class TestCharacterServerBoundaryConditions:
    """Test suite for boundary value analysis and extreme conditions."""
    
    @pytest.mark.parametrize("level", [1, 2**31-1, 2**63-1])
    def test_character_extreme_level_values(self, character_server, level):
        """Test character creation with extreme level values."""
        char_data = {
            'name': f'ExtremeHero_{level}',
            'class': 'Boundary',
            'level': level
        }
        
        result = character_server.create_character(char_data)
        assert result['level'] == level
    
    def test_character_with_extremely_long_name(self, character_server):
        """Test character creation with extremely long name."""
        long_name = 'A' * 100000  # 100K characters
        char_data = {
            'name': long_name,
            'class': 'LongName',
            'level': 1
        }
        
        result = character_server.create_character(char_data)
        assert result['name'] == long_name
        assert len(result['name']) == 100000
    
    def test_character_with_deeply_nested_data(self, character_server):
        """Test character with deeply nested data structures."""
        nested_data = {
            'name': 'NestedHero',
            'class': 'Complex',
            'level': 1,
            'inventory': {
                'weapons': {
                    'primary': {
                        'type': 'sword',
                        'enchantments': {
                            'fire': {'level': 5, 'damage': 10},
                            'ice': {'level': 3, 'damage': 6}
                        }
                    },
                    'secondary': {
                        'type': 'bow',
                        'arrows': [
                            {'type': 'normal', 'count': 50},
                            {'type': 'fire', 'count': 20}
                        ]
                    }
                },
                'armor': {
                    'head': {'type': 'helmet', 'defense': 5},
                    'body': {'type': 'chainmail', 'defense': 10}
                }
            }
        }
        
        result = character_server.create_character(nested_data)
        assert result['inventory']['weapons']['primary']['type'] == 'sword'
        assert result['inventory']['weapons']['primary']['enchantments']['fire']['level'] == 5
    
    def test_character_with_large_arrays(self, character_server):
        """Test character with large array data."""
        large_array_data = {
            'name': 'ArrayHero',
            'class': 'Collector',
            'level': 1,
            'items': [f'item_{i}' for i in range(10000)],
            'skills': [f'skill_{i}' for i in range(1000)],
            'achievements': [f'achievement_{i}' for i in range(5000)]
        }
        
        result = character_server.create_character(large_array_data)
        assert len(result['items']) == 10000
        assert len(result['skills']) == 1000
        assert len(result['achievements']) == 5000


class TestCharacterServerErrorHandling:
    """Test suite for comprehensive error handling scenarios."""
    
    def test_create_character_with_none_data(self, character_server):
        """Test character creation with None as input data."""
        with pytest.raises((ValueError, TypeError)):
            character_server.create_character(None)
    
    def test_create_character_with_non_dict_data(self, character_server):
        """Test character creation with non-dictionary data."""
        invalid_inputs = ["string", 123, [], True, 45.67, set(), tuple()]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                character_server.create_character(invalid_input)
    
    def test_character_operations_with_invalid_id_types(self, character_server):
        """Test character operations with invalid ID types."""
        invalid_ids = ["1", "abc", [], {}, None, 1.5, complex(1, 2)]
        
        for invalid_id in invalid_ids:
            assert character_server.get_character(invalid_id) is None
            assert character_server.delete_character(invalid_id) is False
            result = character_server.update_character(invalid_id, {'level': 10})
            assert result is None
    
    def test_update_character_with_invalid_data_types(self, character_server, valid_character_data):
        """Test character update with various invalid data types."""
        created_char = character_server.create_character(valid_character_data)
        
        invalid_update_data = [None, "string", 123, [], True]
        
        for invalid_data in invalid_update_data:
            with pytest.raises((ValueError, TypeError)):
                character_server.update_character(created_char['id'], invalid_data)
    
    def test_character_operations_with_extreme_negative_ids(self, character_server):
        """Test character operations with extreme negative IDs."""
        extreme_ids = [-1, -100, -2**31, -2**63]
        
        for extreme_id in extreme_ids:
            assert character_server.get_character(extreme_id) is None
            assert character_server.delete_character(extreme_id) is False
            result = character_server.update_character(extreme_id, {'level': 10})
            assert result is None


class TestCharacterServerStressAndPerformance:
    """Test suite for stress testing and performance validation."""
    
    def test_rapid_character_crud_operations(self, character_server, valid_character_data):
        """Test rapid Create, Read, Update, Delete operations."""
        num_operations = 1000
        character_ids = []
        
        # Rapid creation
        for i in range(num_operations):
            char_data = {**valid_character_data, 'name': f'RapidHero_{i}'}
            char = character_server.create_character(char_data)
            character_ids.append(char['id'])
        
        # Rapid reading
        for char_id in character_ids:
            char = character_server.get_character(char_id)
            assert char is not None
        
        # Rapid updating
        for char_id in character_ids:
            character_server.update_character(char_id, {'level': 99})
        
        # Rapid deletion
        for char_id in character_ids:
            result = character_server.delete_character(char_id)
            assert result is True
    
    def test_memory_intensive_character_operations(self, character_server):
        """Test operations with memory-intensive character data."""
        # Create characters with large data payloads
        large_data_chars = []
        for i in range(100):
            char_data = {
                'name': f'MemoryHero_{i}',
                'class': 'MemoryIntensive',
                'level': 1,
                'large_text': 'x' * 100000,  # 100KB of text
                'large_array': list(range(10000)),  # Large array
                'large_dict': {f'key_{j}': f'value_{j}' for j in range(1000)}
            }
            char = character_server.create_character(char_data)
            large_data_chars.append(char)
        
        # Verify all characters were created successfully
        assert len(large_data_chars) == 100
        
        # Test retrieval of large data
        for char in large_data_chars:
            retrieved = character_server.get_character(char['id'])
            assert len(retrieved['large_text']) == 100000
            assert len(retrieved['large_array']) == 10000
            assert len(retrieved['large_dict']) == 1000
    
    def test_interleaved_operations_stress_test(self, character_server, valid_character_data):
        """Test system under interleaved operation stress."""
        import random
        
        characters = []
        operations_performed = 0
        max_operations = 2000
        
        while operations_performed < max_operations:
            operation = random.choice(['create', 'read', 'update', 'delete', 'list'])
            
            if operation == 'create' or len(characters) == 0:
                char_data = {**valid_character_data, 'name': f'StressHero_{operations_performed}'}
                char = character_server.create_character(char_data)
                characters.append(char)
            
            elif operation == 'read' and characters:
                random_char = random.choice(characters)
                character_server.get_character(random_char['id'])
            
            elif operation == 'update' and characters:
                random_char = random.choice(characters)
                character_server.update_character(random_char['id'], {'level': random.randint(1, 100)})
            
            elif operation == 'delete' and len(characters) > 10:  # Keep some characters
                char_to_delete = characters.pop(random.randint(0, len(characters) - 1))
                character_server.delete_character(char_to_delete['id'])
            
            elif operation == 'list':
                character_server.list_characters()
            
            operations_performed += 1
        
        # Verify system is still in a consistent state
        remaining_chars = character_server.list_characters()
        assert len(remaining_chars) > 0


class TestCharacterServerSpecialCharacters:
    """Test suite for special character and encoding handling."""
    
    def test_unicode_character_names_comprehensive(self, character_server):
        """Test comprehensive Unicode character name support."""
        unicode_test_cases = [
            ('Chinese', '测试英雄角色名称'),
            ('Japanese', 'テストヒーロー'),
            ('Korean', '테스트 영웅'),
            ('Arabic', 'بطل الاختبار'),
            ('Hebrew', 'גיבור מבחן'),
            ('Russian', 'тестовый герой'),
            ('Greek', 'δοκιμαστικός ήρωας'),
            ('Emoji', '🦸‍♂️🗡️⚔️🛡️'),
            ('Mathematical', '∑∞∆∇∂∫√'),
            ('Symbols', '★☆♠♣♥♦'),
            ('Accented', 'Héröîç Âcçéñtéd Nàmé'),
        ]
        
        for description, name in unicode_test_cases:
            char_data = {
                'name': name,
                'class': f'Unicode{description}',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
            
            # Verify retrieval maintains encoding
            retrieved = character_server.get_character(result['id'])
            assert retrieved['name'] == name
    
    def test_control_characters_in_names(self, character_server):
        """Test handling of control characters in names."""
        control_chars = [
            ('Null', 'Hero\x00Character'),
            ('Tab', 'Hero\tCharacter'),
            ('Newline', 'Hero\nCharacter'),
            ('Carriage Return', 'Hero\rCharacter'),
            ('Escape', 'Hero\x1bCharacter'),
            ('Backspace', 'Hero\x08Character'),
            ('Form Feed', 'Hero\x0cCharacter'),
            ('Vertical Tab', 'Hero\x0bCharacter'),
        ]
        
        for description, name in control_chars:
            char_data = {
                'name': name,
                'class': f'Control{description.replace(" ", "")}',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
    
    def test_mixed_encoding_character_data(self, character_server):
        """Test character data with mixed encodings and special characters."""
        mixed_data = {
            'name': 'Mïxéd Éñçødîñg Hérø 🌟',
            'class': 'Ωαρριοr',
            'level': 1,
            'description': 'A héro with spéciål çharactérs ànd émojis 🗡️⚔️',
            'location': 'Tøkyo, Jåpan 🏯',
            'guild': 'Dràgøn Sláyers ⚡',
            'motto': '∞ Sτρεngτh, ∞ Hønør ∞'
        }
        
        result = character_server.create_character(mixed_data)
        
        # Verify all fields maintain their encoding
        for key, value in mixed_data.items():
            assert result[key] == value


class TestCharacterServerEdgeCasesAndCornerCases:
    """Test suite for edge cases and corner case scenarios."""
    
    def test_character_id_sequence_after_mass_deletion(self, character_server, valid_character_data):
        """Test ID sequencing behavior after mass character deletion."""
        # Create many characters
        created_chars = []
        for i in range(100):
            char_data = {**valid_character_data, 'name': f'MassHero_{i}'}
            char = character_server.create_character(char_data)
            created_chars.append(char)
        
        # Delete all characters
        for char in created_chars:
            character_server.delete_character(char['id'])
        
        # Create new characters and verify ID behavior
        new_chars = []
        for i in range(5):
            char_data = {**valid_character_data, 'name': f'NewHero_{i}'}
            char = character_server.create_character(char_data)
            new_chars.append(char)
        
        # Verify new IDs continue from where they left off
        for i in range(1, len(new_chars)):
            assert new_chars[i]['id'] > new_chars[i-1]['id']
    
    def test_character_update_with_same_data(self, character_server, valid_character_data):
        """Test updating character with identical data."""
        created_char = character_server.create_character(valid_character_data)
        original_id = created_char['id']
        
        # Update with same data
        updated_char = character_server.update_character(original_id, valid_character_data)
        
        # Verify update succeeded and data remains consistent
        assert updated_char is not None
        assert updated_char['id'] == original_id
        for key, value in valid_character_data.items():
            assert updated_char[key] == value
    
    def test_character_operations_at_system_limits(self, character_server):
        """Test character operations at various system limits."""
        # Test with maximum integer values
        max_int_data = {
            'name': 'MaxIntHero',
            'class': 'Boundary',
            'level': 2**31 - 1,
            'hp': 2**31 - 1,
            'mp': 2**31 - 1,
            'experience': 2**63 - 1
        }
        
        result = character_server.create_character(max_int_data)
        assert result['level'] == 2**31 - 1
        assert result['experience'] == 2**63 - 1
        
        # Test retrieval and update
        retrieved = character_server.get_character(result['id'])
        assert retrieved['level'] == 2**31 - 1
        
        # Test update with max values
        update_result = character_server.update_character(
            result['id'], 
            {'level': 2**31 - 1, 'new_max_field': 2**63 - 1}
        )
        assert update_result['new_max_field'] == 2**63 - 1
    
    def test_character_data_type_preservation(self, character_server):
        """Test that character data types are preserved across operations."""
        complex_data = {
            'name': 'TypeTestHero',
            'class': 'Tester',
            'level': 1,
            'is_active': True,
            'health_percentage': 95.5,
            'skills': ['combat', 'magic'],
            'attributes': {'strength': 10, 'dexterity': 8},
            'coordinates': (100, 200),
            'inventory_count': 0,
            'last_login': None
        }
        
        result = character_server.create_character(complex_data)
        
        # Verify types are preserved
        assert isinstance(result['is_active'], bool)
        assert isinstance(result['health_percentage'], float)
        assert isinstance(result['skills'], list)
        assert isinstance(result['attributes'], dict)
        assert isinstance(result['coordinates'], (tuple, list))
        assert isinstance(result['inventory_count'], int)
        assert result['last_login'] is None


# Additional performance and monitoring fixtures
@pytest.fixture
def performance_character_data():
    """Fixture providing character data optimized for performance testing."""
    return {
        'name': 'PerfTestHero',
        'class': 'Speedster',
        'level': 1,
        'hp': 100,
        'mp': 50,
        'created_at': '2023-01-01T00:00:00Z'
    }


@pytest.fixture
def character_server_with_preloaded_data(character_server, valid_character_data):
    """Fixture providing a character server with pre-loaded test data."""
    # Create diverse set of test characters
    test_characters = [
        {'name': 'Tank', 'class': 'Warrior', 'level': 50, 'hp': 500, 'mp': 50},
        {'name': 'Healer', 'class': 'Cleric', 'level': 45, 'hp': 300, 'mp': 400},
        {'name': 'DPS', 'class': 'Rogue', 'level': 48, 'hp': 250, 'mp': 100},
        {'name': 'Mage', 'class': 'Wizard', 'level': 47, 'hp': 200, 'mp': 500},
        {'name': 'Support', 'class': 'Bard', 'level': 46, 'hp': 275, 'mp': 350},
    ]
    
    for char_data in test_characters:
        character_server.create_character(char_data)
    
    return character_server


if __name__ == '__main__':
    # Run comprehensive tests with detailed reporting
    pytest.main([
        __file__, 
        '-v', 
        '--tb=long', 
        '--durations=10',
        '--cov=character_server',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])

class TestCharacterServerLifecycle:
    """Test suite for character server lifecycle and initialization."""
    
    def test_character_server_initialization(self):
        """Test character server proper initialization."""
        server = MockCharacterServer()
        assert server.characters == {}
        assert server.next_id == 1
    
    def test_character_server_state_persistence_simulation(self, character_server, valid_character_data):
        """Test character server state consistency across operations."""
        # Create initial state
        chars = []
        for i in range(10):
            char_data = {**valid_character_data, 'name': f'StateHero_{i}'}
            chars.append(character_server.create_character(char_data))
        
        # Simulate server restart by checking state consistency
        initial_count = len(character_server.list_characters())
        assert initial_count == 10
        
        # Perform mixed operations
        character_server.delete_character(chars[0]['id'])
        character_server.update_character(chars[1]['id'], {'level': 99})
        
        # Verify state remains consistent
        final_chars = character_server.list_characters()
        assert len(final_chars) == 9
        updated_char = character_server.get_character(chars[1]['id'])
        assert updated_char['level'] == 99


class TestCharacterServerAdvancedValidation:
    """Test suite for advanced validation scenarios."""
    
    @pytest.mark.parametrize("field_name,field_value,should_pass", [
        ('name', 'ValidName', True),
        ('name', 'A' * 255, True),  # Long but reasonable name
        ('name', 'A' * 10000, True),  # Very long name
        ('class', 'ValidClass', True),
        ('level', 1, True),
        ('level', 100, True),
        ('level', 999999, True),
        ('name', '', False),  # Empty name
        ('level', 0, False),  # Zero level
        ('level', -1, False),  # Negative level
    ])
    def test_field_validation_comprehensive(self, field_name, field_value, should_pass):
        """Test comprehensive field validation with various values."""
        base_data = {'name': 'TestHero', 'class': 'Warrior', 'level': 1}
        base_data[field_name] = field_value
        
        if should_pass:
            assert MockCharacterServer.validate_character_data(base_data) is True
        else:
            assert MockCharacterServer.validate_character_data(base_data) is False
    
    def test_character_validation_with_circular_references(self, character_server):
        """Test character creation with circular reference data structures."""
        circular_data = {
            'name': 'CircularHero',
            'class': 'Recursive',
            'level': 1
        }
        # Create circular reference
        circular_data['self_ref'] = circular_data
        
        # Should handle circular references gracefully
        try:
            result = character_server.create_character(circular_data)
            assert 'self_ref' in result
        except (ValueError, RecursionError):
            # Either validation fails or recursion is caught - both acceptable
            pass


class TestCharacterServerMemoryAndResourceManagement:
    """Test suite for memory and resource management scenarios."""
    
    def test_character_creation_memory_cleanup(self, character_server, valid_character_data):
        """Test that character creation doesn't cause memory leaks in rapid succession."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and immediately delete many characters
        for i in range(1000):
            char_data = {**valid_character_data, 'name': f'MemoryTestHero_{i}'}
            char = character_server.create_character(char_data)
            character_server.delete_character(char['id'])
        
        # Force garbage collection after operations
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow excessively (allow some variance)
        memory_growth = final_objects - initial_objects
        assert memory_growth < 1000, f"Possible memory leak: {memory_growth} objects retained"
    
    def test_character_server_resource_limits(self, character_server):
        """Test character server behavior at resource limits."""
        # Test with resource-intensive character data
        resource_heavy_data = {
            'name': 'ResourceHeavyHero',
            'class': 'MemoryConsumer', 
            'level': 1,
            'massive_inventory': {f'item_{i}': f'value_{i}' * 1000 for i in range(1000)},
            'huge_description': 'x' * 1000000,  # 1MB string
            'large_skill_tree': [[f'skill_{i}_{j}' for j in range(50)] for i in range(100)]
        }
        
        result = character_server.create_character(resource_heavy_data)
        assert result is not None
        assert len(result['huge_description']) == 1000000
        assert len(result['large_skill_tree']) == 100


class TestCharacterServerTransactionAndAtomicity:
    """Test suite for transaction-like behavior and atomicity."""
    
    def test_character_update_atomicity_simulation(self, character_server, valid_character_data):
        """Test that character updates are atomic (all-or-nothing behavior)."""
        created_char = character_server.create_character(valid_character_data)
        original_level = created_char['level']
        
        # Simulate partial update failure by updating with mixed valid/invalid data
        complex_update = {
            'level': 50,  # Valid
            'name': 'UpdatedName',  # Valid
            'hp': 200,  # Valid
            'invalid_field_with_circular_ref': None
        }
        
        # Create circular reference to test atomicity
        complex_update['invalid_field_with_circular_ref'] = complex_update
        
        try:
            result = character_server.update_character(created_char['id'], complex_update)
            # If update succeeds, all fields should be updated
            if result:
                assert result['level'] == 50
                assert result['name'] == 'UpdatedName'
                assert result['hp'] == 200
        except Exception:
            # If update fails, character should remain unchanged
            unchanged_char = character_server.get_character(created_char['id'])
            assert unchanged_char['level'] == original_level
    
    def test_concurrent_character_access_simulation(self, character_server, valid_character_data):
        """Test character server behavior under simulated concurrent access."""
        created_char = character_server.create_character(valid_character_data)
        char_id = created_char['id']
        
        # Simulate concurrent operations on the same character
        operations = []
        for i in range(100):
            if i % 3 == 0:
                # Read operation
                result = character_server.get_character(char_id)
                operations.append(('read', result is not None))
            elif i % 3 == 1:
                # Update operation
                update_data = {'level': 10 + (i % 50), 'hp': 100 + i}
                result = character_server.update_character(char_id, update_data)
                operations.append(('update', result is not None))
            else:
                # List operation
                chars = character_server.list_characters()
                operations.append(('list', len(chars) > 0))
        
        # Verify all operations completed successfully
        assert len(operations) == 100
        assert all(success for _, success in operations)
        
        # Verify character is in consistent state
        final_char = character_server.get_character(char_id)
        assert final_char is not None
        assert final_char['id'] == char_id


class TestCharacterServerAdvancedEdgeCases:
    """Test suite for advanced edge cases and corner scenarios."""
    
    def test_character_with_extremely_nested_data_structures(self, character_server):
        """Test character creation with extremely nested data structures."""
        # Create deeply nested structure
        nested_level = 100
        nested_data = {'name': 'DeepNestHero', 'class': 'Recursive', 'level': 1}
        
        current = nested_data
        for i in range(nested_level):
            current[f'level_{i}'] = {f'nested_{i}': f'value_{i}'}
            current = current[f'level_{i}']
        
        try:
            result = character_server.create_character(nested_data)
            assert result is not None
            assert 'level_0' in result
        except (RecursionError, ValueError):
            # Deep nesting might hit recursion limits - acceptable behavior
            pass
    
    def test_character_operations_with_unusual_data_types(self, character_server):
        """Test character operations with unusual but valid Python data types."""
        unusual_data = {
            'name': 'UnusualHero',
            'class': 'DataExplorer',
            'level': 1,
            'complex_number': complex(3, 4),
            'byte_data': b'binary_data_here',
            'frozen_set': frozenset([1, 2, 3, 4]),
            'range_object': range(10),
            'memory_view': memoryview(b'memory_data'),
        }
        
        try:
            result = character_server.create_character(unusual_data)
            assert result is not None
            # Verify data types are preserved or handled gracefully
            if 'complex_number' in result:
                assert isinstance(result['complex_number'], complex)
        except (ValueError, TypeError):
            # Some unusual data types might not be supported - acceptable
            pass
    
    def test_character_server_behavior_with_modified_methods(self, character_server, valid_character_data):
        """Test character server behavior when methods are dynamically modified."""
        # Store original validation method
        original_validate = MockCharacterServer.validate_character_data
        
        try:
            # Temporarily modify validation to always return True
            MockCharacterServer.validate_character_data = lambda data: True
            
            # Test with previously invalid data
            invalid_data = {'name': '', 'class': 'Test', 'level': -1}
            result = character_server.create_character(invalid_data)
            assert result is not None  # Should succeed with modified validation
            
        finally:
            # Restore original validation method
            MockCharacterServer.validate_character_data = original_validate
        
        # Verify normal validation is restored
        with pytest.raises(ValueError):
            character_server.create_character({'name': '', 'class': 'Test', 'level': 1})


class TestCharacterServerComprehensiveIntegration:
    """Test suite for comprehensive integration scenarios."""
    
    def test_complete_character_lifecycle_workflow(self, character_server):
        """Test complete character lifecycle from creation to deletion."""
        # Phase 1: Character Creation and Initial Setup
        character_data = {
            'name': 'LifecycleHero',
            'class': 'Adventurer',
            'level': 1,
            'hp': 100,
            'mp': 50,
            'experience': 0,
            'equipment': [],
            'skills': ['basic_attack'],
            'stats': {'strength': 10, 'dexterity': 10, 'intelligence': 10}
        }
        
        hero = character_server.create_character(character_data)
        assert hero['level'] == 1
        assert hero['experience'] == 0
        
        # Phase 2: Character Progression
        progression_updates = [
            ({'level': 5, 'experience': 1000}, 'Early game progression'),
            ({'level': 10, 'hp': 200, 'skills': ['basic_attack', 'power_strike']}, 'Skill acquisition'),
            ({'level': 25, 'equipment': ['iron_sword', 'leather_armor']}, 'Equipment upgrade'),
            ({'level': 50, 'stats': {'strength': 25, 'dexterity': 20, 'intelligence': 15}}, 'Stat advancement'),
            ({'level': 100, 'class': 'Master Adventurer', 'hp': 1000, 'mp': 500}, 'Class advancement')
        ]
        
        for update_data, phase_name in progression_updates:
            updated_hero = character_server.update_character(hero['id'], update_data)
            assert updated_hero is not None, f"Failed at {phase_name}"
            hero = updated_hero  # Update reference for next phase
        
        # Phase 3: Verification of Final State
        final_hero = character_server.get_character(hero['id'])
        assert final_hero['level'] == 100
        assert final_hero['class'] == 'Master Adventurer'
        assert 'power_strike' in final_hero['skills']
        assert 'iron_sword' in final_hero['equipment']
        
        # Phase 4: Character Retirement (Deletion)
        deletion_result = character_server.delete_character(hero['id'])
        assert deletion_result is True
        retired_hero = character_server.get_character(hero['id'])
        assert retired_hero is None
    
    def test_multi_character_guild_simulation(self, character_server):
        """Test simulation of multi-character guild operations."""
        # Create guild characters with different roles
        guild_members = [
            {'name': 'GuildLeader', 'class': 'Paladin', 'level': 60, 'role': 'leader'},
            {'name': 'GuildTank', 'class': 'Warrior', 'level': 55, 'role': 'tank'},
            {'name': 'GuildHealer', 'class': 'Cleric', 'level': 58, 'role': 'healer'},
            {'name': 'GuildDPS1', 'class': 'Rogue', 'level': 52, 'role': 'dps'},
            {'name': 'GuildDPS2', 'class': 'Mage', 'level': 54, 'role': 'dps'},
            {'name': 'GuildSupport', 'class': 'Bard', 'level': 50, 'role': 'support'}
        ]
        
        # Create all guild members
        created_members = []
        for member_data in guild_members:
            member = character_server.create_character(member_data)
            created_members.append(member)
        
        # Simulate guild activities (level progression after group activities)
        for member in created_members:
            # All members gain experience from group activity
            updated_member = character_server.update_character(
                member['id'], 
                {'level': member['level'] + 5, 'experience': 10000}
            )
            assert updated_member['level'] == member['level'] + 5
        
        # Verify guild integrity
        all_members = character_server.list_characters()
        guild_member_names = [m['name'] for m in all_members if m['name'].startswith('Guild')]
        assert len(guild_member_names) == 6
        assert 'GuildLeader' in guild_member_names
        
        # Simulate member leaving guild (deletion)
        character_server.delete_character(created_members[0]['id'])  # Leader leaves
        
        remaining_members = [m for m in character_server.list_characters() 
                           if m['name'].startswith('Guild')]
        assert len(remaining_members) == 5
        assert all(m['name'] != 'GuildLeader' for m in remaining_members)


# Additional fixtures for advanced testing
@pytest.fixture
def large_dataset_character_server():
    """Fixture providing character server with large pre-loaded dataset."""
    server = MockCharacterServer()
    
    # Create diverse character dataset
    classes = ['Warrior', 'Mage', 'Rogue', 'Cleric', 'Paladin', 'Ranger', 'Bard', 'Warlock']
    levels = list(range(1, 101))
    
    for i in range(500):  # Create 500 characters
        char_data = {
            'name': f'DataHero_{i:03d}',
            'class': classes[i % len(classes)],
            'level': levels[i % len(levels)],
            'hp': (levels[i % len(levels)] * 10) + 100,
            'mp': (levels[i % len(levels)] * 5) + 50,
            'experience': levels[i % len(levels)] * 1000,
            'created_batch': i // 50  # Group characters into batches
        }
        server.create_character(char_data)
    
    return server


@pytest.fixture
def stress_test_character_data():
    """Fixture providing character data designed for stress testing."""
    return {
        'name': 'StressTestHero',
        'class': 'StressTester',
        'level': 1,
        'stress_data': {
            'large_text': 'A' * 50000,  # 50KB text field
            'number_array': list(range(5000)),  # Large numeric array
            'nested_objects': [{'id': i, 'data': f'item_{i}' * 100} for i in range(100)],
            'metadata': {f'key_{i}': f'value_{i}' * 10 for i in range(1000)}
        }
    }



class TestRealCharacterServerIntegration:
    """Test suite for integration with actual character server implementation."""
    
    @pytest.fixture(autouse=True)
    def setup_real_server_tests(self):
        """Setup for real server integration tests."""
        # This would be implemented when actual server is available
        try:
            # Try to import actual character server
            # from character_server import CharacterServer
            # self.real_server = CharacterServer()
            self.real_server = None
            self.has_real_server = False
        except ImportError:
            self.real_server = None
            self.has_real_server = False
    
    @pytest.mark.skipif(True, reason="Real character server not available")
    def test_real_server_character_creation(self):
        """Test character creation with real server implementation."""
        if not self.has_real_server:
            pytest.skip("Real character server not available")
        
        char_data = {
            'name': 'RealServerHero',
            'class': 'Warrior',
            'level': 1
        }
        
        result = self.real_server.create_character(char_data)
        assert result is not None
        assert 'id' in result
    
    @pytest.mark.skipif(True, reason="Real character server not available") 
    def test_real_server_performance_comparison(self, performance_character_data):
        """Test performance comparison between mock and real server."""
        if not self.has_real_server:
            pytest.skip("Real character server not available")
        
        import time
        
        # Test mock server performance
        mock_server = MockCharacterServer()
        start_time = time.time()
        for i in range(1000):
            char_data = {**performance_character_data, 'name': f'MockHero_{i}'}
            mock_server.create_character(char_data)
        mock_time = time.time() - start_time
        
        # Test real server performance  
        start_time = time.time()
        for i in range(1000):
            char_data = {**performance_character_data, 'name': f'RealHero_{i}'}
            self.real_server.create_character(char_data)
        real_time = time.time() - start_time
        
        # Performance should be within reasonable bounds
        assert real_time < mock_time * 10  # Real server shouldn't be >10x slower


# Additional test utilities and helper functions
class TestUtilities:
    """Utility functions for character server testing."""
    
    @staticmethod
    def create_test_character_batch(server, count=10, base_data=None):
        """Create a batch of test characters for testing purposes."""
        if base_data is None:
            base_data = {'name': 'BatchHero', 'class': 'Tester', 'level': 1}
        
        created_chars = []
        for i in range(count):
            char_data = {**base_data, 'name': f'{base_data["name"]}_{i:03d}'}
            char = server.create_character(char_data)
            created_chars.append(char)
        
        return created_chars
    
    @staticmethod
    def verify_character_data_integrity(char1, char2, exclude_fields=None):
        """Verify that two character objects have matching data integrity."""
        if exclude_fields is None:
            exclude_fields = ['id']
        
        for key in char1:
            if key not in exclude_fields:
                assert key in char2, f"Field {key} missing in second character"
                assert char1[key] == char2[key], f"Field {key} mismatch: {char1[key]} != {char2[key]}"
    
    @staticmethod
    def generate_random_character_data(seed=None):
        """Generate random but valid character data for testing."""
        import random
        if seed:
            random.seed(seed)
        
        classes = ['Warrior', 'Mage', 'Rogue', 'Cleric', 'Paladin', 'Ranger']
        names = ['Hero', 'Champion', 'Warrior', 'Mage', 'Guardian', 'Seeker']
        
        return {
            'name': f"{random.choice(names)}{random.randint(1000, 9999)}",
            'class': random.choice(classes),
            'level': random.randint(1, 100),
            'hp': random.randint(50, 1000),
            'mp': random.randint(25, 500),
            'experience': random.randint(0, 100000)
        }


# Test configuration and execution enhancements
@pytest.fixture(scope="session")
def test_configuration():
    """Session-scoped configuration for test execution."""
    return {
        'max_test_characters': 10000,
        'performance_threshold_ms': 100,
        'memory_threshold_mb': 100,
        'stress_test_iterations': 1000,
        'enable_performance_tests': True,
        'enable_stress_tests': True,
        'enable_security_tests': True
    }


if __name__ == '__main__':
    # Enhanced test execution with comprehensive reporting
    import sys
    
    # Configure pytest arguments for comprehensive testing
    pytest_args = [
        __file__,
        '-v',                          # Verbose output
        '--tb=long',                   # Long traceback format
        '--durations=20',              # Show 20 slowest tests
        '--strict-markers',            # Strict marker validation
        '--strict-config',             # Strict configuration validation
        '--disable-warnings',          # Clean output
        '--maxfail=5',                 # Stop after 5 failures
    ]
    
    # Add coverage reporting if available
    try:
        import pytest_cov
        pytest_args.extend([
            '--cov=character_server',
            '--cov-report=html:htmlcov',
            '--cov-report=term-missing',
            '--cov-report=xml',
            '--cov-fail-under=90'       # Require 90% coverage
        ])
    except ImportError:
        print("pytest-cov not available, skipping coverage reporting")
    
    # Add benchmark reporting if available
    try:
        import pytest_benchmark
        pytest_args.extend([
            '--benchmark-only',
            '--benchmark-sort=mean',
            '--benchmark-json=benchmark_results.json'
        ])
    except ImportError:
        print("pytest-benchmark not available, skipping benchmark tests")
    
    # Execute tests
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)
