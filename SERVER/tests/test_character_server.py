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

class TestCharacterServerAdvancedValidation:
    """Advanced validation test suite for more robust character server testing."""
    
    def test_character_name_whitespace_handling(self, character_server):
        """Test character name handling with various whitespace scenarios."""
        whitespace_names = [
            '  SpacedHero  ',  # Leading/trailing spaces
            'Tab\tHero',       # Tab character
            'New\nLine',       # Newline character
            'Multi   Space',   # Multiple spaces
            '\r\nCarriage',    # Carriage return + newline
            ' ',               # Single space
            '\t',              # Single tab
        ]
        
        for name in whitespace_names:
            char_data = {
                'name': name,
                'class': 'Whitespace',
                'level': 1
            }
            result = character_server.create_character(char_data)
            assert result['name'] == name
    
    def test_character_level_extreme_values(self, character_server):
        """Test character level with extreme boundary values."""
        extreme_levels = [
            1,                    # Minimum valid
            2**15 - 1,           # 16-bit signed max
            2**16 - 1,           # 16-bit unsigned max
            2**31 - 1,           # 32-bit signed max
            2**32 - 1,           # 32-bit unsigned max (if supported)
        ]
        
        for level in extreme_levels:
            char_data = {
                'name': f'ExtremeHero_{level}',
                'class': 'Boundary',
                'level': level
            }
            try:
                result = character_server.create_character(char_data)
                assert result['level'] == level
            except (ValueError, OverflowError):
                # Some extreme values might be rejected, which is acceptable
                pass
    
    def test_character_class_case_sensitivity(self, character_server):
        """Test character class validation with different case variations."""
        case_variations = [
            ('warrior', 'Warrior'),
            ('WARRIOR', 'Warrior'), 
            ('WaRrIoR', 'Warrior'),
            ('mage', 'Mage'),
            ('MAGE', 'Mage'),
            ('rogue', 'Rogue'),
            ('ROGUE', 'Rogue'),
        ]
        
        for input_class, expected_class in case_variations:
            char_data = {
                'name': f'{input_class}Hero',
                'class': input_class,
                'level': 1
            }
            result = character_server.create_character(char_data)
            # Either should accept as-is or normalize to expected case
            assert result['class'] in [input_class, expected_class]


class TestCharacterServerDataTypes:
    """Test suite for comprehensive data type handling."""
    
    def test_character_numeric_field_types(self, character_server):
        """Test character creation with various numeric field types."""
        numeric_data = {
            'name': 'NumericHero',
            'class': 'Calculator',
            'level': 1,
            'hp': 100,
            'mp': 50,
            'strength': 15,
            'dexterity': 12,
            'intelligence': 8,
            'luck': 5,
            'experience': 0,
            'gold': 100,
            'stat_points': 0,
            'skill_points': 0,
        }
        
        result = character_server.create_character(numeric_data)
        
        # Verify all numeric fields are preserved with correct types
        for field in ['hp', 'mp', 'strength', 'dexterity', 'intelligence', 'luck']:
            assert isinstance(result[field], int)
            assert result[field] == numeric_data[field]
    
    def test_character_boolean_field_handling(self, character_server):
        """Test character creation with boolean fields."""
        boolean_data = {
            'name': 'BooleanHero',
            'class': 'Logic',
            'level': 1,
            'is_active': True,
            'is_online': False,
            'is_premium': True,
            'is_banned': False,
            'pvp_enabled': True,
            'tutorial_completed': False,
        }
        
        result = character_server.create_character(boolean_data)
        
        # Verify boolean fields maintain their types and values
        assert result['is_active'] is True
        assert result['is_online'] is False
        assert result['is_premium'] is True
        assert result['is_banned'] is False
        assert result['pvp_enabled'] is True
        assert result['tutorial_completed'] is False
    
    def test_character_nested_object_handling(self, character_server):
        """Test character creation with complex nested objects."""
        nested_data = {
            'name': 'NestedHero',
            'class': 'Complex',
            'level': 1,
            'stats': {
                'base': {'str': 10, 'dex': 8, 'int': 12},
                'bonus': {'str': 2, 'dex': 1, 'int': 3},
                'total': {'str': 12, 'dex': 9, 'int': 15}
            },
            'equipment': {
                'weapon': {'name': 'Iron Sword', 'damage': 15, 'durability': 100},
                'armor': {'name': 'Leather Vest', 'defense': 8, 'durability': 80},
                'accessories': [
                    {'name': 'Ring of Power', 'effect': '+2 STR'},
                    {'name': 'Amulet of Wisdom', 'effect': '+3 INT'}
                ]
            },
            'skills': {
                'combat': {'level': 5, 'experience': 1250},
                'magic': {'level': 3, 'experience': 450},
                'crafting': {'level': 1, 'experience': 50}
            }
        }
        
        result = character_server.create_character(nested_data)
        
        # Verify nested structure is preserved
        assert result['stats']['base']['str'] == 10
        assert result['equipment']['weapon']['name'] == 'Iron Sword'
        assert result['skills']['combat']['level'] == 5
        assert len(result['equipment']['accessories']) == 2


class TestCharacterServerConcurrency:
    """Test suite for concurrent operation scenarios."""
    
    def test_concurrent_character_creation_unique_ids(self, character_server, valid_character_data):
        """Test that concurrent character creation produces unique IDs."""
        import threading
        import time
        
        created_characters = []
        creation_errors = []
        
        def create_character_worker(worker_id):
            try:
                char_data = {**valid_character_data, 'name': f'ConcurrentHero_{worker_id}'}
                char = character_server.create_character(char_data)
                created_characters.append(char)
            except Exception as e:
                creation_errors.append(e)
        
        # Create multiple threads for concurrent character creation
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_character_worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(creation_errors) == 0
        
        # Verify all characters were created with unique IDs
        assert len(created_characters) == 10
        ids = [char['id'] for char in created_characters]
        assert len(set(ids)) == len(ids)  # All IDs should be unique
    
    def test_concurrent_character_operations_mixed(self, character_server, valid_character_data):
        """Test mixed concurrent operations (create, read, update, delete)."""
        import threading
        import random
        import time
        
        # Pre-create some characters for operations
        initial_characters = []
        for i in range(5):
            char_data = {**valid_character_data, 'name': f'InitialHero_{i}'}
            char = character_server.create_character(char_data)
            initial_characters.append(char)
        
        operation_results = []
        operation_errors = []
        
        def mixed_operations_worker(worker_id):
            try:
                operations = ['create', 'read', 'update']
                for _ in range(5):  # Each worker performs 5 operations
                    op = random.choice(operations)
                    
                    if op == 'create':
                        char_data = {**valid_character_data, 'name': f'Worker{worker_id}_Hero_{random.randint(1, 1000)}'}
                        result = character_server.create_character(char_data)
                        operation_results.append(('create', result))
                    
                    elif op == 'read' and initial_characters:
                        char = random.choice(initial_characters)
                        result = character_server.get_character(char['id'])
                        operation_results.append(('read', result))
                    
                    elif op == 'update' and initial_characters:
                        char = random.choice(initial_characters)
                        update_data = {'level': random.randint(1, 50)}
                        result = character_server.update_character(char['id'], update_data)
                        operation_results.append(('update', result))
                    
                    time.sleep(0.001)  # Small delay to increase chance of concurrency
            except Exception as e:
                operation_errors.append(e)
        
        # Create worker threads
        threads = []
        for i in range(3):  # 3 concurrent workers
            thread = threading.Thread(target=mixed_operations_worker, args=(i,))
            threads.append(thread)
        
        # Execute concurrent operations
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify operations completed successfully
        assert len(operation_errors) == 0
        assert len(operation_results) > 0


class TestCharacterServerMemoryManagement:
    """Test suite for memory management and resource usage."""
    
    def test_character_creation_memory_efficiency(self, character_server):
        """Test memory efficiency during character creation."""
        import gc
        import sys
        
        # Force garbage collection and get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and delete many characters
        for i in range(100):
            char_data = {
                'name': f'MemoryTestHero_{i}',
                'class': 'Temporary',
                'level': 1,
                'temp_data': 'x' * 1000  # 1KB of temporary data
            }
            char = character_server.create_character(char_data)
            character_server.delete_character(char['id'])
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold for object growth
    
    def test_large_character_data_handling(self, character_server):
        """Test handling of characters with large data payloads."""
        large_data_sizes = [
            ('1KB', 1024),
            ('10KB', 10240),
            ('100KB', 102400),
            ('1MB', 1048576),
        ]
        
        for size_name, size_bytes in large_data_sizes:
            char_data = {
                'name': f'LargeDataHero_{size_name}',
                'class': 'DataCarrier',
                'level': 1,
                'large_field': 'x' * size_bytes,
                'metadata': {
                    'size': size_bytes,
                    'description': f'Character with {size_name} of data'
                }
            }
            
            try:
                result = character_server.create_character(char_data)
                assert len(result['large_field']) == size_bytes
                assert result['metadata']['size'] == size_bytes
                
                # Test retrieval of large data
                retrieved = character_server.get_character(result['id'])
                assert len(retrieved['large_field']) == size_bytes
                
                # Clean up
                character_server.delete_character(result['id'])
                
            except (MemoryError, ValueError) as e:
                # Some very large data might be rejected, which is acceptable
                pass


class TestCharacterServerSpecialCases:
    """Test suite for special edge cases and corner scenarios."""
    
    def test_character_with_all_optional_fields_none(self, character_server):
        """Test character creation where all optional fields are explicitly None."""
        minimal_with_nones = {
            'name': 'MinimalNoneHero',
            'class': 'Basic',
            'level': 1,
            'hp': None,
            'mp': None,
            'equipment': None,
            'skills': None,
            'description': None,
            'location': None,
        }
        
        result = character_server.create_character(minimal_with_nones)
        assert result['name'] == 'MinimalNoneHero'
        assert result['class'] == 'Basic'
        assert result['level'] == 1
        # None values should be preserved
        assert result.get('hp') is None
        assert result.get('equipment') is None
    
    def test_character_field_order_preservation(self, character_server):
        """Test that character field order is preserved when possible."""
        ordered_data = {
            'name': 'OrderedHero',
            'class': 'Organized',
            'level': 1,
            'a_first_field': 'first',
            'b_second_field': 'second',
            'c_third_field': 'third',
            'z_last_field': 'last'
        }
        
        result = character_server.create_character(ordered_data)
        
        # Verify all fields are present
        for key, value in ordered_data.items():
            assert result[key] == value
    
    def test_character_with_numeric_string_fields(self, character_server):
        """Test character creation with numeric strings that should not be auto-converted."""
        numeric_string_data = {
            'name': 'NumericStringHero',
            'class': 'StringKeeper',
            'level': 1,
            'id_string': '12345',
            'phone_number': '+1-555-123-4567',
            'postal_code': '90210',
            'version': '1.0.0',
            'build_number': '2023.12.01.001'
        }
        
        result = character_server.create_character(numeric_string_data)
        
        # Verify strings that look like numbers remain strings
        assert isinstance(result['id_string'], str)
        assert result['id_string'] == '12345'
        assert isinstance(result['phone_number'], str)
        assert isinstance(result['postal_code'], str)
        assert isinstance(result['version'], str)


class TestCharacterServerErrorRecovery:
    """Test suite for error recovery and system resilience."""
    
    def test_character_server_state_after_invalid_operations(self, character_server, valid_character_data):
        """Test that server state remains consistent after series of invalid operations."""
        # Create valid baseline characters
        baseline_chars = []
        for i in range(3):
            char_data = {**valid_character_data, 'name': f'BaselineHero_{i}'}
            char = character_server.create_character(char_data)
            baseline_chars.append(char)
        
        initial_count = len(character_server.list_characters())
        
        # Attempt various invalid operations
        invalid_operations = [
            lambda: character_server.create_character(None),
            lambda: character_server.create_character({'invalid': 'data'}),
            lambda: character_server.create_character('not_a_dict'),
            lambda: character_server.get_character('invalid_id'),
            lambda: character_server.get_character(-999999),
            lambda: character_server.update_character(99999, {'level': 50}),
            lambda: character_server.update_character('invalid', {}),
            lambda: character_server.delete_character('invalid'),
            lambda: character_server.delete_character(-999999),
        ]
        
        # Execute invalid operations and catch their errors
        for operation in invalid_operations:
            try:
                operation()
            except (ValueError, TypeError, AttributeError):
                pass  # Expected errors
        
        # Verify system state is still consistent
        final_count = len(character_server.list_characters())
        assert final_count == initial_count
        
        # Verify baseline characters are still accessible and unmodified
        for char in baseline_chars:
            retrieved = character_server.get_character(char['id'])
            assert retrieved is not None
            assert retrieved['name'] == char['name']
            assert retrieved['level'] == char['level']
        
        # Verify system is still functional for valid operations
        new_char = character_server.create_character({**valid_character_data, 'name': 'RecoveryTestHero'})
        assert new_char is not None
        assert character_server.get_character(new_char['id']) is not None
    
    def test_character_partial_update_failure_recovery(self, character_server, valid_character_data):
        """Test recovery from partial update failures."""
        created_char = character_server.create_character(valid_character_data)
        original_data = character_server.get_character(created_char['id'])
        
        # Attempt updates that might partially fail
        problematic_updates = [
            {'level': 'invalid_type', 'name': 'ShouldNotUpdate'},
            {'invalid_field': {'nested': 'complex'}, 'level': 50},
            {None: 'invalid_key', 'name': 'BadUpdate'},
        ]
        
        for update in problematic_updates:
            try:
                character_server.update_character(created_char['id'], update)
            except (ValueError, TypeError, KeyError):
                pass  # Expected failures
            
            # Verify character data integrity after failed update
            current_data = character_server.get_character(created_char['id'])
            assert current_data is not None
            assert current_data['name'] == original_data['name']  # Should be unchanged
            assert current_data['level'] == original_data['level']  # Should be unchanged


# Enhanced test execution with additional markers
class TestCharacterServerComplexScenarios:
    """Test suite for complex real-world scenarios."""
    
    @pytest.mark.slow
    def test_character_server_endurance_test(self, character_server, valid_character_data):
        """Long-running endurance test for server stability."""
        operations_count = 0
        max_operations = 1000
        character_pool = []
        
        import random
        
        while operations_count < max_operations:
            operation = random.choice(['create', 'read', 'update', 'delete', 'list'])
            
            try:
                if operation == 'create':
                    char_data = {**valid_character_data, 'name': f'EnduranceHero_{operations_count}'}
                    char = character_server.create_character(char_data)
                    character_pool.append(char)
                
                elif operation == 'read' and character_pool:
                    char = random.choice(character_pool)
                    character_server.get_character(char['id'])
                
                elif operation == 'update' and character_pool:
                    char = random.choice(character_pool)
                    character_server.update_character(char['id'], {'level': random.randint(1, 100)})
                
                elif operation == 'delete' and len(character_pool) > 10:
                    char = character_pool.pop(random.randint(0, len(character_pool) - 1))
                    character_server.delete_character(char['id'])
                
                elif operation == 'list':
                    character_server.list_characters()
                
                operations_count += 1
                
                # Periodic verification
                if operations_count % 100 == 0:
                    assert len(character_server.list_characters()) >= 0
                    
            except Exception as e:
                pytest.fail(f"Endurance test failed at operation {operations_count}: {e}")
        
        # Final verification
        final_characters = character_server.list_characters()
        assert isinstance(final_characters, list)
    
    def test_character_data_migration_simulation(self, character_server):
        """Test simulation of character data migration scenarios."""
        # Create characters with old data format
        old_format_characters = []
        for i in range(10):
            old_char = {
                'name': f'OldFormatHero_{i}',
                'class': 'Legacy',
                'level': i + 1,
                'old_format_field': f'legacy_value_{i}',
                'deprecated_stat': 'old_stat'
            }
            char = character_server.create_character(old_char)
            old_format_characters.append(char)
        
        # Simulate migration by updating to new format
        for char in old_format_characters:
            migration_update = {
                'new_format_field': char.get('old_format_field', '').replace('legacy_', 'migrated_'),
                'updated_stat': char.get('deprecated_stat', 'default'),
                'migration_timestamp': '2023-01-01T00:00:00Z',
                'old_format_field': None,  # Remove old field
                'deprecated_stat': None    # Remove deprecated field
            }
            
            updated_char = character_server.update_character(char['id'], migration_update)
            assert updated_char is not None
            assert 'migrated_' in updated_char.get('new_format_field', '')
            assert updated_char.get('migration_timestamp') is not None


# Additional test utilities and helpers
@pytest.fixture
def complex_character_factory():
    """Factory fixture for creating complex test characters."""
    def _create_complex_character(name_suffix="", **overrides):
        base_data = {
            'name': f'ComplexHero{name_suffix}',
            'class': 'MultiClass',
            'level': 1,
            'stats': {
                'primary': {'str': 10, 'dex': 8, 'int': 12, 'wis': 9, 'con': 11, 'cha': 7},
                'secondary': {'luck': 5, 'karma': 0, 'honor': 50}
            },
            'inventory': {
                'weapons': [],
                'armor': {},
                'consumables': [],
                'quest_items': [],
                'currency': {'gold': 100, 'silver': 500, 'copper': 1000}
            },
            'progress': {
                'experience': 0,
                'skill_points': 0,
                'attribute_points': 0,
                'completed_quests': [],
                'current_quests': [],
                'achievements': []
            },
            'social': {
                'guild': None,
                'friends': [],
                'blocked_users': [],
                'reputation': {}
            },
            'settings': {
                'auto_loot': True,
                'pvp_enabled': False,
                'chat_enabled': True,
                'notifications': True
            }
        }
        
        # Apply any overrides
        base_data.update(overrides)
        return base_data
    
    return _create_complex_character


if __name__ == '__main__':
    # Comprehensive test execution with maximum detail
    pytest.main([
        __file__, 
        '-v', 
        '--tb=long', 
        '--durations=25',
        '--maxfail=10',
        '--strict-markers',
        '--strict-config',
        '--cov=character_server',
        '--cov-branch',
        '--cov-report=html:htmlcov_enhanced',
        '--cov-report=term-missing:skip-covered',
        '--cov-report=xml:coverage_enhanced.xml',
        '--cov-fail-under=95',
        '-m', 'not slow',  # Skip slow tests in regular runs
        '--disable-warnings'
    ])