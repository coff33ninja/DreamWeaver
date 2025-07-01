import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException
import json
import logging

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from character_client import CharacterClient
from config import Config


class TestCharacterClient:
    """Comprehensive test suite for CharacterClient class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock(spec=Config)
        config.get.side_effect = lambda key, default=None: {
            'CHARACTER_API_URL': 'https://test-api.character.com/v1',
            'CHARACTER_API_KEY': 'test-api-key-123',
            'DEFAULT_TIMEOUT': 30
        }.get(key, default)
        return config
    
    @pytest.fixture
    def character_client(self, mock_config):
        """Create CharacterClient instance for testing."""
        with patch('character_client.Config', return_value=mock_config):
            return CharacterClient()
    
    @pytest.fixture
    def sample_character(self):
        """Sample character data for testing."""
        return {
            "id": 1,
            "name": "Test Warrior",
            "class": "Warrior",
            "level": 10,
            "health": 100,
            "mana": 50,
            "stats": {
                "strength": 15,
                "agility": 12,
                "intelligence": 8
            }
        }
    
    @pytest.fixture
    def sample_character_list(self, sample_character):
        """Sample character list for testing."""
        return [
            sample_character,
            {
                "id": 2,
                "name": "Test Mage",
                "class": "Mage",
                "level": 8,
                "health": 75,
                "mana": 120
            }
        ]

    # Initialization Tests
    def test_initialization_default_config(self):
        """Test CharacterClient initialization with default configuration."""
        with patch('character_client.Config') as mock_config_class:
            mock_config_instance = Mock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'CHARACTER_API_URL': 'https://api.character.com/v1',
                'CHARACTER_API_KEY': None
            }.get(key, default)
            mock_config_class.return_value = mock_config_instance
            
            client = CharacterClient()
            
            assert client.base_url == 'https://api.character.com/v1'
            assert client.api_key is None
            assert client.timeout == 30
            assert isinstance(client.session, requests.Session)
    
    def test_initialization_custom_params(self):
        """Test CharacterClient initialization with custom parameters."""
        custom_url = "https://custom-api.example.com/v1"
        custom_key = "custom-api-key"
        custom_timeout = 60
        
        with patch('character_client.Config') as mock_config_class:
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance
            
            client = CharacterClient(
                base_url=custom_url,
                api_key=custom_key,
                timeout=custom_timeout
            )
            
            assert client.base_url == custom_url
            assert client.api_key == custom_key
            assert client.timeout == custom_timeout
    
    def test_session_headers_with_api_key(self, mock_config):
        """Test that session headers are properly set when API key is provided."""
        with patch('character_client.Config', return_value=mock_config):
            client = CharacterClient(api_key="test-key")
            
            expected_headers = {
                'Authorization': 'Bearer test-key',
                'Content-Type': 'application/json',
                'User-Agent': 'CharacterClient/1.0'
            }
            
            for key, value in expected_headers.items():
                assert client.session.headers[key] == value
    
    def test_session_headers_without_api_key(self):
        """Test that session headers are properly set when no API key is provided."""
        with patch('character_client.Config') as mock_config_class:
            mock_config_instance = Mock()
            mock_config_instance.get.return_value = None
            mock_config_class.return_value = mock_config_instance
            
            client = CharacterClient()
            
            assert 'Authorization' not in client.session.headers
            assert client.session.headers['Content-Type'] == 'application/json'
            assert client.session.headers['User-Agent'] == 'CharacterClient/1.0'

    # Get Character Tests
    @patch('requests.Session.get')
    def test_get_character_success(self, mock_get, character_client, sample_character):
        """Test successful character retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = character_client.get_character(1)
        
        assert result == sample_character
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters/1',
            timeout=30
        )
    
    def test_get_character_invalid_id_zero(self, character_client):
        """Test get_character with zero ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character(0)
    
    def test_get_character_invalid_id_negative(self, character_client):
        """Test get_character with negative ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character(-1)
    
    def test_get_character_invalid_id_string(self, character_client):
        """Test get_character with string ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character("invalid")
    
    def test_get_character_invalid_id_none(self, character_client):
        """Test get_character with None ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character(None)
    
    @patch('requests.Session.get')
    def test_get_character_http_error(self, mock_get, character_client):
        """Test get_character with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.get_character(999)
    
    @patch('requests.Session.get')
    def test_get_character_connection_error(self, mock_get, character_client):
        """Test get_character with connection error."""
        mock_get.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            character_client.get_character(1)
    
    @patch('requests.Session.get')
    def test_get_character_timeout(self, mock_get, character_client):
        """Test get_character with timeout."""
        mock_get.side_effect = Timeout("Request timed out")
        
        with pytest.raises(Timeout):
            character_client.get_character(1)

    # Create Character Tests
    @patch('requests.Session.post')
    def test_create_character_success(self, mock_post, character_client):
        """Test successful character creation."""
        character_data = {"name": "New Warrior", "class": "Warrior", "level": 1}
        created_character = {**character_data, "id": 3}
        
        mock_response = Mock()
        mock_response.json.return_value = created_character
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = character_client.create_character(character_data)
        
        assert result == created_character
        mock_post.assert_called_once_with(
            'https://test-api.character.com/v1/characters',
            json=character_data,
            timeout=30
        )
    
    def test_create_character_empty_data(self, character_client):
        """Test create_character with empty data."""
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.create_character({})
    
    def test_create_character_none_data(self, character_client):
        """Test create_character with None data."""
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.create_character(None)
    
    def test_create_character_non_dict_data(self, character_client):
        """Test create_character with non-dictionary data."""
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.create_character("invalid")
    
    def test_create_character_missing_name(self, character_client):
        """Test create_character with missing name field."""
        character_data = {"class": "Warrior", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'name' is missing or empty"):
            character_client.create_character(character_data)
    
    def test_create_character_missing_class(self, character_client):
        """Test create_character with missing class field."""
        character_data = {"name": "Test Character", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'class' is missing or empty"):
            character_client.create_character(character_data)
    
    def test_create_character_empty_name(self, character_client):
        """Test create_character with empty name field."""
        character_data = {"name": "", "class": "Warrior", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'name' is missing or empty"):
            character_client.create_character(character_data)
    
    def test_create_character_empty_class(self, character_client):
        """Test create_character with empty class field."""
        character_data = {"name": "Test Character", "class": "", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'class' is missing or empty"):
            character_client.create_character(character_data)
    
    @patch('requests.Session.post')
    def test_create_character_http_error(self, mock_post, character_client):
        """Test create_character with HTTP error."""
        character_data = {"name": "Test Character", "class": "Warrior"}
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("422 Validation Error")
        mock_post.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.create_character(character_data)

    # Update Character Tests
    @patch('requests.Session.put')
    def test_update_character_success(self, mock_put, character_client, sample_character):
        """Test successful character update."""
        update_data = {"level": 15, "health": 120}
        updated_character = {**sample_character, **update_data}
        
        mock_response = Mock()
        mock_response.json.return_value = updated_character
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response
        
        result = character_client.update_character(1, update_data)
        
        assert result == updated_character
        mock_put.assert_called_once_with(
            'https://test-api.character.com/v1/characters/1',
            json=update_data,
            timeout=30
        )
    
    def test_update_character_invalid_id(self, character_client):
        """Test update_character with invalid ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.update_character(0, {"level": 20})
    
    def test_update_character_empty_data(self, character_client):
        """Test update_character with empty data."""
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.update_character(1, {})
    
    def test_update_character_none_data(self, character_client):
        """Test update_character with None data."""
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.update_character(1, None)
    
    @patch('requests.Session.put')
    def test_update_character_not_found(self, mock_put, character_client):
        """Test updating non-existent character."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_put.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.update_character(999, {"level": 20})

    # Delete Character Tests
    @patch('requests.Session.delete')
    def test_delete_character_success(self, mock_delete, character_client):
        """Test successful character deletion."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_response
        
        result = character_client.delete_character(1)
        
        assert result is True
        mock_delete.assert_called_once_with(
            'https://test-api.character.com/v1/characters/1',
            timeout=30
        )
    
    def test_delete_character_invalid_id(self, character_client):
        """Test delete_character with invalid ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.delete_character(-1)
    
    @patch('requests.Session.delete')
    def test_delete_character_not_found(self, mock_delete, character_client):
        """Test deleting non-existent character."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_delete.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.delete_character(999)

    # List Characters Tests
    @patch('requests.Session.get')
    def test_list_characters_success(self, mock_get, character_client, sample_character_list):
        """Test successful character listing."""
        response_data = {
            "characters": sample_character_list,
            "total": 2,
            "limit": 10,
            "offset": 0
        }
        
        mock_response = Mock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = character_client.list_characters()
        
        assert result == response_data
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters',
            params={'limit': 10, 'offset': 0},
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_list_characters_with_pagination(self, mock_get, character_client):
        """Test character listing with custom pagination."""
        response_data = {"characters": [], "total": 0, "limit": 5, "offset": 10}
        
        mock_response = Mock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = character_client.list_characters(limit=5, offset=10)
        
        assert result == response_data
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters',
            params={'limit': 5, 'offset': 10},
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_list_characters_with_filters(self, mock_get, character_client):
        """Test character listing with filters."""
        filters = {"class": "Warrior", "level_min": 5}
        expected_params = {'limit': 10, 'offset': 0, 'class': 'Warrior', 'level_min': 5}
        
        mock_response = Mock()
        mock_response.json.return_value = {"characters": [], "total": 0}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        character_client.list_characters(filters=filters)
        
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters',
            params=expected_params,
            timeout=30
        )
    
    def test_list_characters_invalid_limit_zero(self, character_client):
        """Test list_characters with zero limit."""
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.list_characters(limit=0)
    
    def test_list_characters_invalid_limit_negative(self, character_client):
        """Test list_characters with negative limit."""
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.list_characters(limit=-1)
    
    def test_list_characters_invalid_limit_string(self, character_client):
        """Test list_characters with string limit."""
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.list_characters(limit="invalid")
    
    def test_list_characters_invalid_offset_negative(self, character_client):
        """Test list_characters with negative offset."""
        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            character_client.list_characters(offset=-1)
    
    def test_list_characters_invalid_offset_string(self, character_client):
        """Test list_characters with string offset."""
        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            character_client.list_characters(offset="invalid")

    # Search Characters Tests
    @patch('requests.Session.get')
    def test_search_characters_success(self, mock_get, character_client, sample_character_list):
        """Test successful character search."""
        search_response = {"characters": sample_character_list}
        
        mock_response = Mock()
        mock_response.json.return_value = search_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = character_client.search_characters("warrior")
        
        assert result == sample_character_list
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters/search',
            params={'q': 'warrior', 'limit': 10},
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_search_characters_with_limit(self, mock_get, character_client):
        """Test character search with custom limit."""
        mock_response = Mock()
        mock_response.json.return_value = {"characters": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        character_client.search_characters("test", limit=5)
        
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters/search',
            params={'q': 'test', 'limit': 5},
            timeout=30
        )
    
    def test_search_characters_empty_query(self, character_client):
        """Test search_characters with empty query."""
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            character_client.search_characters("")
    
    def test_search_characters_none_query(self, character_client):
        """Test search_characters with None query."""
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            character_client.search_characters(None)
    
    def test_search_characters_non_string_query(self, character_client):
        """Test search_characters with non-string query."""
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            character_client.search_characters(123)
    
    def test_search_characters_invalid_limit(self, character_client):
        """Test search_characters with invalid limit."""
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.search_characters("test", limit=0)

    # Get Character Stats Tests
    @patch('requests.Session.get')
    def test_get_character_stats_success(self, mock_get, character_client):
        """Test successful character stats retrieval."""
        stats_data = {
            "character_id": 1,
            "stats": {
                "strength": 15,
                "agility": 12,
                "intelligence": 8,
                "vitality": 10
            },
            "combat_stats": {
                "attack_power": 45,
                "defense": 20,
                "critical_chance": 0.15
            }
        }
        
        mock_response = Mock()
        mock_response.json.return_value = stats_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = character_client.get_character_stats(1)
        
        assert result == stats_data
        mock_get.assert_called_once_with(
            'https://test-api.character.com/v1/characters/1/stats',
            timeout=30
        )
    
    def test_get_character_stats_invalid_id(self, character_client):
        """Test get_character_stats with invalid ID."""
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character_stats(0)
    
    @patch('requests.Session.get')
    def test_get_character_stats_not_found(self, mock_get, character_client):
        """Test get_character_stats with non-existent character."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.get_character_stats(999)

    # Edge Cases and Error Handling Tests
    @patch('requests.Session.get')
    def test_malformed_json_response(self, mock_get, character_client):
        """Test handling of malformed JSON responses."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with pytest.raises(json.JSONDecodeError):
            character_client.get_character(1)
    
    @patch('requests.Session.get')
    def test_request_exception_handling(self, mock_get, character_client):
        """Test handling of general request exceptions."""
        mock_get.side_effect = RequestException("General request error")
        
        with pytest.raises(RequestException):
            character_client.get_character(1)
    
    def test_logging_on_error(self, character_client, caplog):
        """Test that errors are properly logged."""
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection failed")
            
            with caplog.at_level(logging.ERROR):
                with pytest.raises(ConnectionError):
                    character_client.get_character(1)
            
            assert "Failed to get character 1" in caplog.text
            assert "Connection failed" in caplog.text
    
    def test_large_character_id(self, character_client):
        """Test handling of very large character IDs."""
        large_id = 999999999999999999
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"id": large_id, "name": "Test"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = character_client.get_character(large_id)
            
            assert result["id"] == large_id
            mock_get.assert_called_once_with(
                f'https://test-api.character.com/v1/characters/{large_id}',
                timeout=30
            )
    
    @patch('requests.Session.post')
    def test_unicode_character_data(self, mock_post, character_client):
        """Test handling of Unicode characters in character data."""
        unicode_data = {
            "name": "Tëst Châractér 🎮",
            "class": "Wârrior",
            "description": "A character with émojis ⚔️🛡️"
        }
        
        mock_response = Mock()
        mock_response.json.return_value = {**unicode_data, "id": 1}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = character_client.create_character(unicode_data)
        
        assert result["name"] == unicode_data["name"]
        assert result["description"] == unicode_data["description"]