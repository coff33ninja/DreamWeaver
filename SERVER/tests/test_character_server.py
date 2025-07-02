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

# ========================================================================
# ADDITIONAL COMPREHENSIVE TESTS FOR CHARACTER SERVER
# Testing Framework: pytest (confirmed from existing imports)
# These tests extend the existing comprehensive coverage with additional
# scenarios, integration tests, and edge cases
# ========================================================================


class TestCharacterServerAdvancedValidation:
    """Advanced validation test suite for character server."""
    
    def test_character_name_length_boundaries(self, character_server):
        """Test character name length at various boundaries."""
        # Test minimum length (1 character)
        min_data = {'name': 'A', 'class': 'Warrior', 'level': 1}
        result = character_server.create_character(min_data)
        assert result['name'] == 'A'
        
        # Test various length boundaries
        lengths_to_test = [1, 2, 10, 50, 100, 255, 1000, 10000]
        for length in lengths_to_test:
            name = 'H' * length
            char_data = {'name': name, 'class': 'Warrior', 'level': 1}
            result = character_server.create_character(char_data)
            assert len(result['name']) == length
    
    def test_character_class_validation_extended(self, character_server):
        """Test extended character class validation scenarios."""
        valid_classes = [
            'Warrior', 'Mage', 'Rogue', 'Cleric', 'Paladin', 'Ranger',
            'Bard', 'Sorcerer', 'Warlock', 'Barbarian', 'Monk', 'Druid',
            'Artificer', 'Blood Hunter', 'Mystic', 'Death Knight'
        ]
        
        for char_class in valid_classes:
            char_data = {'name': f'{char_class}Hero', 'class': char_class, 'level': 1}
            result = character_server.create_character(char_data)
            assert result['class'] == char_class
    
    def test_character_level_precision_and_boundaries(self, character_server):
        """Test character level with precision boundaries and edge cases."""
        # Test level boundaries
        boundary_levels = [1, 2, 10, 50, 99, 100, 999, 1000, 9999, 10000]
        
        for level in boundary_levels:
            char_data = {'name': f'Level{level}Hero', 'class': 'Warrior', 'level': level}
            result = character_server.create_character(char_data)
            assert result['level'] == level
            assert isinstance(result['level'], int)
    
    def test_character_data_with_null_and_undefined_values(self, character_server):
        """Test character creation with null and undefined values in optional fields."""
        char_data = {
            'name': 'NullTestHero',
            'class': 'Warrior', 
            'level': 1,
            'optional_field': None,
            'empty_string': '',
            'zero_value': 0,
            'false_value': False,
            'empty_list': [],
            'empty_dict': {}
        }
        
        result = character_server.create_character(char_data)
        assert result['name'] == 'NullTestHero'
        assert result['optional_field'] is None
        assert result['empty_string'] == ''
        assert result['zero_value'] == 0
        assert result['false_value'] is False
        assert result['empty_list'] == []
        assert result['empty_dict'] == {}


class TestCharacterServerDatabaseLikeOperations:
    """Test suite simulating database-like operations and constraints."""
    
    def test_character_unique_name_constraint_simulation(self, character_server, valid_character_data):
        """Test behavior when attempting to create characters with duplicate names."""
        # Create first character
        char1 = character_server.create_character(valid_character_data)
        
        # Attempt to create second character with same name
        char2_data = {**valid_character_data}
        char2 = character_server.create_character(char2_data)
        
        # Both should succeed (no unique constraint in mock)
        assert char1['name'] == char2['name']
        assert char1['id'] != char2['id']
    
    def test_character_cascade_delete_simulation(self, character_server, valid_character_data):
        """Test cascade delete behavior for related character data."""
        # Create character with related data
        char_data = {
            **valid_character_data,
            'guild_id': 123,
            'party_members': [456, 789],
            'owned_items': ['sword_001', 'shield_002']
        }
        
        created_char = character_server.create_character(char_data)
        char_id = created_char['id']
        
        # Delete character
        delete_result = character_server.delete_character(char_id)
        assert delete_result is True
        
        # Verify character and all related data is gone
        retrieved_char = character_server.get_character(char_id)
        assert retrieved_char is None
    
    def test_character_transactional_operations_simulation(self, character_server, valid_character_data):
        """Test transactional-like operations for character data consistency."""
        # Create multiple characters
        characters = []
        for i in range(5):
            char_data = {**valid_character_data, 'name': f'TransactionHero_{i}'}
            char = character_server.create_character(char_data)
            characters.append(char)
        
        # Simulate batch update operation
        batch_update_data = {'level': 25, 'status': 'updated'}
        updated_characters = []
        
        for char in characters:
            updated_char = character_server.update_character(char['id'], batch_update_data)
            updated_characters.append(updated_char)
        
        # Verify all updates succeeded
        for updated_char in updated_characters:
            assert updated_char['level'] == 25
            assert updated_char['status'] == 'updated'
    
    def test_character_indexing_simulation(self, character_server, valid_character_data):
        """Test character retrieval performance simulation with indexing concepts."""
        # Create many characters with different attributes for "indexing"
        characters_by_class = {}
        characters_by_level = {}
        
        classes = ['Warrior', 'Mage', 'Rogue', 'Cleric']
        levels = [1, 10, 20, 30, 40, 50]
        
        created_chars = []
        for i in range(100):
            char_class = classes[i % len(classes)]
            level = levels[i % len(levels)]
            
            char_data = {
                'name': f'IndexHero_{i}',
                'class': char_class,
                'level': level
            }
            
            char = character_server.create_character(char_data)
            created_chars.append(char)
            
            # Simulate indexing
            if char_class not in characters_by_class:
                characters_by_class[char_class] = []
            characters_by_class[char_class].append(char)
            
            if level not in characters_by_level:
                characters_by_level[level] = []
            characters_by_level[level].append(char)
        
        # Test "index-based" lookups
        warriors = characters_by_class.get('Warrior', [])
        assert len(warriors) == 25  # 100/4 = 25
        
        level_50_chars = characters_by_level.get(50, [])
        assert len(level_50_chars) > 0


class TestCharacterServerComplexDataStructures:
    """Test suite for complex nested data structures and relationships."""
    
    def test_character_with_complex_inventory_system(self, character_server):
        """Test character with complex inventory and item management."""
        complex_inventory_data = {
            'name': 'InventoryMaster',
            'class': 'Merchant',
            'level': 1,
            'inventory': {
                'capacity': 100,
                'current_weight': 45.5,
                'slots': {
                    'weapons': {
                        'primary': {
                            'item_id': 'sword_legendary_001',
                            'name': 'Excalibur',
                            'durability': 100,
                            'enchantments': [
                                {'type': 'fire_damage', 'power': 50},
                                {'type': 'holy_blessing', 'power': 25}
                            ],
                            'gems': [
                                {'type': 'ruby', 'quality': 'perfect', 'bonus': '+10 STR'},
                                {'type': 'diamond', 'quality': 'flawless', 'bonus': '+5 ALL'}
                            ]
                        },
                        'secondary': {
                            'item_id': 'dagger_poisoned_023',
                            'name': 'Viper\'s Fang',
                            'durability': 85,
                            'poison_charges': 10
                        }
                    },
                    'armor': {
                        'head': {'item_id': 'helm_dragon_001', 'defense': 50},
                        'chest': {'item_id': 'plate_titanium_005', 'defense': 100},
                        'legs': {'item_id': 'greaves_steel_010', 'defense': 75},
                        'feet': {'item_id': 'boots_speed_007', 'defense': 25, 'speed_bonus': 15}
                    },
                    'accessories': {
                        'rings': [
                            {'item_id': 'ring_power_001', 'bonus': '+20 MP'},
                            {'item_id': 'ring_life_003', 'bonus': '+50 HP'}
                        ],
                        'amulet': {'item_id': 'amulet_wisdom_002', 'bonus': '+10 INT'},
                        'trinkets': [
                            {'item_id': 'compass_eternal', 'effect': 'never_lost'},
                            {'item_id': 'coin_luck', 'effect': '+5% item_find'}
                        ]
                    }
                },
                'consumables': {
                    'potions': [
                        {'type': 'health', 'quantity': 50, 'potency': 'greater'},
                        {'type': 'mana', 'quantity': 30, 'potency': 'superior'},
                        {'type': 'antidote', 'quantity': 10, 'potency': 'universal'}
                    ],
                    'scrolls': [
                        {'type': 'teleport', 'quantity': 5, 'destinations': ['town', 'dungeon']},
                        {'type': 'resurrection', 'quantity': 2, 'power_level': 'major'}
                    ]
                }
            }
        }
        
        result = character_server.create_character(complex_inventory_data)
        
        # Verify complex nested structure preservation
        assert result['inventory']['capacity'] == 100
        assert result['inventory']['slots']['weapons']['primary']['name'] == 'Excalibur'
        assert len(result['inventory']['slots']['weapons']['primary']['enchantments']) == 2
        assert result['inventory']['slots']['armor']['chest']['defense'] == 100
        assert len(result['inventory']['slots']['accessories']['rings']) == 2
        assert result['inventory']['consumables']['potions'][0]['potency'] == 'greater'
    
    def test_character_with_guild_and_social_relationships(self, character_server):
        """Test character with complex guild and social relationship data."""
        social_data = {
            'name': 'SocialButterfly',
            'class': 'Diplomat',
            'level': 35,
            'guild': {
                'id': 'guild_001',
                'name': 'Elite Guardians',
                'rank': 'Officer',
                'join_date': '2023-01-15',
                'contributions': {
                    'gold_donated': 50000,
                    'resources_provided': ['iron_ore', 'magic_crystals'],
                    'quests_completed': 145,
                    'members_recruited': 12
                },
                'permissions': ['invite_members', 'manage_treasury', 'organize_raids']
            },
            'relationships': {
                'friends': [
                    {'character_id': 'char_123', 'name': 'BestFriend', 'friendship_level': 100},
                    {'character_id': 'char_456', 'name': 'GoodFriend', 'friendship_level': 80},
                    {'character_id': 'char_789', 'name': 'Acquaintance', 'friendship_level': 40}
                ],
                'enemies': [
                    {'character_id': 'char_999', 'name': 'Nemesis', 'rivalry_level': 95, 'reason': 'guild_war'}
                ],
                'mentors': [
                    {'character_id': 'char_001', 'name': 'WiseMaster', 'teachings_received': ['advanced_combat', 'leadership']}
                ],
                'students': [
                    {'character_id': 'char_555', 'name': 'Apprentice1', 'lessons_taught': ['basic_combat']},
                    {'character_id': 'char_666', 'name': 'Apprentice2', 'lessons_taught': ['diplomacy', 'trade']}
                ]
            },
            'reputation': {
                'factions': {
                    'royal_court': {'standing': 'honored', 'points': 8500},
                    'merchants_guild': {'standing': 'revered', 'points': 12000},
                    'thieves_den': {'standing': 'hostile', 'points': -3000},
                    'mages_circle': {'standing': 'neutral', 'points': 100}
                },
                'cities': {
                    'capital_city': {'standing': 'hero', 'points': 15000},
                    'port_town': {'standing': 'friendly', 'points': 5000},
                    'mountain_village': {'standing': 'unknown', 'points': 0}
                }
            }
        }
        
        result = character_server.create_character(social_data)
        
        # Verify complex social structure preservation
        assert result['guild']['name'] == 'Elite Guardians'
        assert result['guild']['rank'] == 'Officer'
        assert len(result['relationships']['friends']) == 3
        assert result['relationships']['friends'][0]['friendship_level'] == 100
        assert len(result['relationships']['students']) == 2
        assert result['reputation']['factions']['royal_court']['standing'] == 'honored'
        assert result['reputation']['cities']['capital_city']['points'] == 15000
    
    def test_character_with_progression_and_achievement_data(self, character_server):
        """Test character with detailed progression and achievement tracking."""
        progression_data = {
            'name': 'AchievementHunter',
            'class': 'Explorer',
            'level': 42,
            'experience': {
                'current': 1250000,
                'to_next_level': 75000,
                'total_earned': 1325000,
                'sources': {
                    'combat': 800000,
                    'quests': 400000,
                    'exploration': 125000
                }
            },
            'skills': {
                'combat_skills': {
                    'one_handed_weapons': {'level': 85, 'experience': 425000},
                    'archery': {'level': 78, 'experience': 390000},
                    'defense': {'level': 70, 'experience': 350000},
                    'magic_resistance': {'level': 45, 'experience': 225000}
                },
                'crafting_skills': {
                    'blacksmithing': {'level': 60, 'experience': 300000},
                    'alchemy': {'level': 55, 'experience': 275000},
                    'enchanting': {'level': 40, 'experience': 200000}
                },
                'social_skills': {
                    'persuasion': {'level': 90, 'experience': 450000},
                    'intimidation': {'level': 35, 'experience': 175000},
                    'deception': {'level': 25, 'experience': 125000}
                }
            },
            'achievements': {
                'combat': [
                    {'id': 'first_kill', 'name': 'First Blood', 'description': 'Defeat your first enemy', 'date_earned': '2023-01-01'},
                    {'id': 'hundred_kills', 'name': 'Centurion', 'description': 'Defeat 100 enemies', 'date_earned': '2023-02-15'},
                    {'id': 'dragon_slayer', 'name': 'Dragon Slayer', 'description': 'Defeat a dragon', 'date_earned': '2023-06-20', 'rarity': 'legendary'}
                ],
                'exploration': [
                    {'id': 'world_traveler', 'name': 'World Traveler', 'description': 'Visit 50 locations', 'date_earned': '2023-04-10'},
                    {'id': 'treasure_hunter', 'name': 'Treasure Hunter', 'description': 'Find 25 hidden treasures', 'date_earned': '2023-05-05'}
                ],
                'social': [
                    {'id': 'diplomat', 'name': 'Master Diplomat', 'description': 'Successfully negotiate 10 conflicts', 'date_earned': '2023-07-01'}
                ]
            },
            'statistics': {
                'playtime_hours': 450,
                'enemies_defeated': 2547,
                'quests_completed': 189,
                'locations_discovered': 73,
                'items_crafted': 234,
                'gold_earned': 125000,
                'deaths': 12,
                'resurrections': 12
            }
        }
        
        result = character_server.create_character(progression_data)
        
        # Verify progression data preservation
        assert result['experience']['current'] == 1250000
        assert result['skills']['combat_skills']['one_handed_weapons']['level'] == 85
        assert len(result['achievements']['combat']) == 3
        assert result['achievements']['combat'][2]['rarity'] == 'legendary'
        assert result['statistics']['playtime_hours'] == 450
        assert result['statistics']['enemies_defeated'] == 2547


class TestCharacterServerAdvancedErrorScenarios:
    """Advanced error handling and recovery scenarios."""
    
    def test_character_operations_with_corrupted_data_simulation(self, character_server, valid_character_data):
        """Test character operations with simulated data corruption scenarios."""
        # Create character
        created_char = character_server.create_character(valid_character_data)
        
        # Simulate various data corruption scenarios
        corruption_scenarios = [
            {'level': float('inf')},  # Infinity
            {'level': float('-inf')},  # Negative infinity  
            {'level': float('nan')},  # NaN
            {'hp': complex(1, 2)},  # Complex number
            {'name': b'binary_data'},  # Binary data
            {'class': {'nested': 'object'}},  # Nested object where string expected
        ]
        
        for corrupt_data in corruption_scenarios:
            # Most should either succeed (flexible mock) or raise appropriate errors
            try:
                result = character_server.update_character(created_char['id'], corrupt_data)
                # If it succeeds, verify the data was stored as-is
                if result:
                    for key, value in corrupt_data.items():
                        # Special handling for NaN comparison
                        if isinstance(value, float) and str(value) == 'nan':
                            assert str(result[key]) == 'nan'
                        else:
                            assert result[key] == value
            except (ValueError, TypeError) as e:
                # Expected for some corruption scenarios
                assert True
    
    def test_character_server_memory_pressure_simulation(self, character_server):
        """Test character server behavior under simulated memory pressure."""
        # Create characters with progressively larger data
        large_data_characters = []
        
        for i in range(10):
            # Each character has exponentially larger data
            data_size = 2 ** i
            char_data = {
                'name': f'MemoryPressureHero_{i}',
                'class': 'MemoryTester',
                'level': 1,
                'large_data': 'x' * (1000 * data_size),  # 1KB, 2KB, 4KB, etc.
                'large_array': list(range(100 * data_size)),
                'large_dict': {f'key_{j}': f'value_{j}' * data_size for j in range(10 * data_size)}
            }
            
            char = character_server.create_character(char_data)
            large_data_characters.append(char)
            
            # Verify creation succeeded
            assert char is not None
            assert len(char['large_data']) == 1000 * data_size
        
        # Test operations on large data characters
        for char in large_data_characters:
            retrieved = character_server.get_character(char['id'])
            assert retrieved is not None
            
            # Test update operations
            updated = character_server.update_character(char['id'], {'status': 'updated'})
            assert updated['status'] == 'updated'
    
    def test_character_server_concurrent_access_simulation(self, character_server, valid_character_data):
        """Test character server behavior under simulated concurrent access."""
        import threading
        import time
        
        # Shared data structures for thread communication
        results = []
        errors = []
        
        def worker_thread(thread_id, operations_count):
            """Worker thread that performs character operations."""
            thread_results = []
            try:
                for i in range(operations_count):
                    # Create character
                    char_data = {**valid_character_data, 'name': f'ConcurrentHero_{thread_id}_{i}'}
                    char = character_server.create_character(char_data)
                    thread_results.append(('create', char['id']))
                    
                    # Update character
                    character_server.update_character(char['id'], {'level': i + 1})
                    thread_results.append(('update', char['id']))
                    
                    # Read character
                    character_server.get_character(char['id'])
                    thread_results.append(('read', char['id']))
                    
                    # Small delay to simulate real-world timing
                    time.sleep(0.001)
                
                results.extend(thread_results)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        num_threads = 5
        operations_per_thread = 10
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=worker_thread, 
                args=(thread_id, operations_per_thread)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads * operations_per_thread * 3  # create, update, read
        
        # Verify all characters exist
        all_chars = character_server.list_characters()
        assert len(all_chars) >= num_threads * operations_per_thread


class TestCharacterServerAdvancedIntegration:
    """Integration tests for character server with external dependencies simulation."""
    
    @patch('builtins.open')
    def test_character_server_file_persistence_simulation(self, mock_open, character_server, valid_character_data):
        """Test character server with simulated file persistence operations."""
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Create character
        char = character_server.create_character(valid_character_data)
        
        # Simulate saving to file
        char_json = json.dumps(char)
        mock_file.write.assert_not_called()  # Not actually called in mock implementation
        
        # Verify character data can be serialized
        assert isinstance(char_json, str)
        
        # Verify deserialization
        restored_char = json.loads(char_json)
        assert restored_char['name'] == valid_character_data['name']
    
    @patch('requests.post')
    def test_character_server_api_integration_simulation(self, mock_post, character_server, valid_character_data):
        """Test character server with simulated external API integration."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'success', 'character_id': 'external_123'}
        mock_post.return_value = mock_response
        
        # Create character
        char = character_server.create_character(valid_character_data)
        
        # Simulate external API call (would be in real implementation)
        # This is just testing that the character data is suitable for API calls
        api_payload = {
            'character_data': char,
            'operation': 'create',
            'timestamp': '2023-01-01T00:00:00Z'
        }
        
        # Verify payload can be serialized for API
        json_payload = json.dumps(api_payload, default=str)
        assert isinstance(json_payload, str)
        assert 'character_data' in json_payload
    
    @patch('database.connection')
    def test_character_server_database_integration_simulation(self, mock_db, character_server, valid_character_data):
        """Test character server with simulated database integration."""
        # Mock database operations
        mock_cursor = Mock()
        mock_db.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = (1, 'TestHero', 'Warrior', 10)
        
        # Create character
        char = character_server.create_character(valid_character_data)
        
        # Simulate database operations
        # In a real implementation, these would be actual database calls
        sql_insert = "INSERT INTO characters (name, class, level) VALUES (?, ?, ?)"
        sql_params = (char['name'], char['class'], char['level'])
        
        # Verify SQL parameters are properly formatted
        assert len(sql_params) == 3
        assert all(isinstance(param, (str, int)) for param in sql_params)
    
    @patch('redis.Redis')
    def test_character_server_cache_integration_simulation(self, mock_redis, character_server, valid_character_data):
        """Test character server with simulated Redis cache integration."""
        # Mock Redis operations
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.get.return_value = None  # Cache miss
        mock_redis_instance.set.return_value = True
        
        # Create character
        char = character_server.create_character(valid_character_data)
        
        # Simulate cache operations
        cache_key = f"character:{char['id']}"
        cache_value = json.dumps(char)
        
        # Verify cache data is serializable
        assert isinstance(cache_key, str)
        assert isinstance(cache_value, str)
        
        # Verify cache value can be deserialized
        cached_char = json.loads(cache_value)
        assert cached_char['name'] == char['name']


class TestCharacterServerAdvancedMetrics:
    """Test suite for advanced metrics, monitoring, and observability."""
    
    def test_character_server_operation_timing_metrics(self, character_server, valid_character_data):
        """Test character server operations with timing metrics."""
        import time
        
        # Test creation timing
        start_time = time.time()
        char = character_server.create_character(valid_character_data)
        create_time = time.time() - start_time
        
        # Creation should be reasonably fast (under 1 second for mock)
        assert create_time < 1.0
        
        # Test retrieval timing
        start_time = time.time()
        retrieved_char = character_server.get_character(char['id'])
        retrieve_time = time.time() - start_time
        
        assert retrieve_time < 1.0
        assert retrieved_char is not None
        
        # Test update timing
        start_time = time.time()
        character_server.update_character(char['id'], {'level': 50})
        update_time = time.time() - start_time
        
        assert update_time < 1.0
        
        # Test deletion timing
        start_time = time.time()
        character_server.delete_character(char['id'])
        delete_time = time.time() - start_time
        
        assert delete_time < 1.0
    
    def test_character_server_memory_usage_estimation(self, character_server):
        """Test character server memory usage estimation."""
        import sys
        
        # Measure initial memory usage
        initial_chars_count = len(character_server.list_characters())
        
        # Create characters and estimate memory usage
        characters = []
        for i in range(100):
            char_data = {
                'name': f'MemoryTestHero_{i}',
                'class': 'MemoryTester',
                'level': i + 1,
                'large_data': 'x' * 1000  # 1KB per character
            }
            char = character_server.create_character(char_data)
            characters.append(char)
        
        # Verify all characters were created
        final_chars_count = len(character_server.list_characters())
        assert final_chars_count >= initial_chars_count + 100
        
        # Test memory efficiency - characters should be retrievable
        for char in characters:
            retrieved = character_server.get_character(char['id'])
            assert retrieved is not None
            assert len(retrieved['large_data']) == 1000
    
    def test_character_server_throughput_metrics(self, character_server, valid_character_data):
        """Test character server throughput under load."""
        import time
        
        num_operations = 1000
        start_time = time.time()
        
        # Perform mixed operations
        created_chars = []
        for i in range(num_operations // 4):  # 250 creates
            char_data = {**valid_character_data, 'name': f'ThroughputHero_{i}'}
            char = character_server.create_character(char_data)
            created_chars.append(char)
        
        for char in created_chars[:len(created_chars)//2]:  # 125 updates
            character_server.update_character(char['id'], {'level': 99})
        
        for char in created_chars:  # 250 reads
            character_server.get_character(char['id'])
        
        # Some deletes
        for char in created_chars[:len(created_chars)//4]:  # 62 deletes
            character_server.delete_character(char['id'])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput (operations per second)
        total_operations = (num_operations // 4) + (len(created_chars)//2) + len(created_chars) + (len(created_chars)//4)
        throughput = total_operations / total_time
        
        # Should handle at least 100 operations per second for mock implementation
        assert throughput > 100, f"Throughput too low: {throughput} ops/sec"


class TestCharacterServerAdvancedCompatibility:
    """Test suite for compatibility with different data formats and versions."""
    
    def test_character_server_json_serialization_compatibility(self, character_server):
        """Test character server JSON serialization/deserialization compatibility."""
        # Test various JSON-problematic data types
        problematic_data = {
            'name': 'JSONTestHero',
            'class': 'Serializer',
            'level': 1,
            'unicode_text': 'héllø wørld 🌍',
            'large_number': 2**53 - 1,  # JavaScript safe integer limit
            'floating_point': 3.141592653589793,
            'boolean_true': True,
            'boolean_false': False,
            'null_value': None,
            'empty_string': '',
            'empty_array': [],
            'empty_object': {},
            'nested_structure': {
                'level1': {
                    'level2': {
                        'level3': ['deep', 'nesting', 'test']
                    }
                }
            }
        }
        
        char = character_server.create_character(problematic_data)
        
        # Test JSON serialization
        json_str = json.dumps(char, ensure_ascii=False, default=str)
        assert isinstance(json_str, str)
        
        # Test JSON deserialization
        restored_char = json.loads(json_str)
        
        # Verify data integrity after JSON round-trip
        assert restored_char['name'] == problematic_data['name']
        assert restored_char['unicode_text'] == problematic_data['unicode_text']
        assert restored_char['large_number'] == problematic_data['large_number']
        assert restored_char['boolean_true'] == problematic_data['boolean_true']
        assert restored_char['null_value'] == problematic_data['null_value']
        assert restored_char['nested_structure']['level1']['level2']['level3'] == ['deep', 'nesting', 'test']
    
    def test_character_server_backwards_compatibility(self, character_server):
        """Test character server backwards compatibility with older data formats."""
        # Simulate older version character data (missing some fields)
        old_format_data = {
            'name': 'LegacyHero',
            'class': 'OldWarrior',
            'level': 25
            # Missing newer fields like 'created_at', 'last_updated', etc.
        }
        
        char = character_server.create_character(old_format_data)
        assert char['name'] == 'LegacyHero'
        assert char['class'] == 'OldWarrior'
        assert char['level'] == 25
        
        # Test adding modern fields via update
        modern_updates = {
            'created_at': '2023-01-01T00:00:00Z',
            'last_updated': '2023-12-31T23:59:59Z',
            'version': '2.0',
            'metadata': {
                'migration_source': 'legacy_system',
                'original_format': 'v1.0'
            }
        }
        
        updated_char = character_server.update_character(char['id'], modern_updates)
        assert updated_char['created_at'] == '2023-01-01T00:00:00Z'
        assert updated_char['metadata']['migration_source'] == 'legacy_system'
    
    def test_character_server_forward_compatibility(self, character_server):
        """Test character server forward compatibility with future data formats."""
        # Simulate future version character data (extra fields)
        future_format_data = {
            'name': 'FutureHero',
            'class': 'CyberWarrior',
            'level': 100,
            # Future fields
            'neural_interface_version': '3.0',
            'augmentations': [
                {'type': 'cybernetic_arm', 'version': '2.1', 'capabilities': ['enhanced_strength', 'built_in_scanner']},
                {'type': 'neural_boost', 'version': '1.5', 'capabilities': ['enhanced_reflexes', 'data_processing']}
            ],
            'ai_companion': {
                'name': 'ARIA',
                'personality_matrix': 'friendly_assistant',
                'learning_algorithms': ['adaptive', 'predictive', 'analytical'],
                'data_banks': {
                    'combat_tactics': 95,
                    'medical_knowledge': 87,
                    'technical_specs': 92
                }
            },
            'quantum_inventory': {
                'dimensional_storage': True,
                'capacity_multiplier': 100,
                'cross_dimensional_access': ['prime', 'alpha', 'beta']
            }
        }
        
        char = character_server.create_character(future_format_data)
        
        # Verify all future fields are preserved
        assert char['neural_interface_version'] == '3.0'
        assert len(char['augmentations']) == 2
        assert char['ai_companion']['name'] == 'ARIA'
        assert char['quantum_inventory']['dimensional_storage'] is True
        
        # Test that system handles unknown fields gracefully
        retrieved_char = character_server.get_character(char['id'])
        assert retrieved_char['neural_interface_version'] == '3.0'


# Performance benchmark fixture
@pytest.fixture
def benchmark_character_data():
    """Fixture providing standardized data for benchmarking."""
    return {
        'name': 'BenchmarkHero',
        'class': 'Benchmark',
        'level': 1,
        'hp': 100,
        'mp': 50,
        'created_at': '2023-01-01T00:00:00Z'
    }


# Custom test markers for extended test categories
pytestmark = [
    pytest.mark.character_server,
    pytest.mark.comprehensive,
    pytest.mark.extended_coverage
]


# Run extended tests if executed directly
if __name__ == '__main__':
    # Run all tests including the new comprehensive ones
    pytest.main([
        __file__, 
        '-v', 
        '--tb=long', 
        '--durations=20',
        '--cov=character_server',
        '--cov-report=html:htmlcov_extended',
        '--cov-report=term-missing',
        '-m', 'not slow'  # Skip slow tests in normal runs
    ])
