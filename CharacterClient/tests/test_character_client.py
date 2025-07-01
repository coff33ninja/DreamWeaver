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
        """
        Provides a mocked configuration object with preset API URL, API key, and timeout values for testing purposes.
        
        Returns:
            Mock: A mock Config object with predefined settings for API URL, API key, and timeout.
        """
        config = Mock(spec=Config)
        config.get.side_effect = lambda key, default=None: {
            'CHARACTER_API_URL': 'https://test-api.character.com/v1',
            'CHARACTER_API_KEY': 'test-api-key-123',
            'DEFAULT_TIMEOUT': 30
        }.get(key, default)
        return config
    
    @pytest.fixture
    def character_client(self, mock_config):
        """
        Creates a CharacterClient instance using a mocked configuration for testing purposes.
        
        Parameters:
            mock_config: The mocked configuration object to be used when instantiating CharacterClient.
        
        Returns:
            CharacterClient: An instance of CharacterClient initialized with the mocked configuration.
        """
        with patch('character_client.Config', return_value=mock_config):
            return CharacterClient()
    
    @pytest.fixture
    def sample_character(self):
        """
        Return a dictionary representing a sample character for use in tests.
        
        Returns:
            dict: Sample character data including id, name, class, level, health, mana, and stats.
        """
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
        """
        Provides a sample list of character dictionaries for use in tests.
        
        Parameters:
        	sample_character (dict): A sample character dictionary to include in the list.
        
        Returns:
        	list: A list containing the provided sample character and an additional predefined character.
        """
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
        """
        Verify that CharacterClient initializes correctly with custom base URL, API key, and timeout values.
        """
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
        """
        Verify that the CharacterClient session includes the correct headers when an API key is provided.
        """
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
        """
        Verify that the session headers exclude Authorization and include Content-Type and User-Agent when no API key is provided.
        """
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
        """
        Tests that get_character successfully retrieves and returns character data for a valid character ID.
        """
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
        """
        Test that get_character raises ValueError when called with a character ID of zero.
        """
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character(0)
    
    def test_get_character_invalid_id_negative(self, character_client):
        """
        Test that get_character raises ValueError when called with a negative character ID.
        """
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character(-1)
    
    def test_get_character_invalid_id_string(self, character_client):
        """
        Test that get_character raises ValueError when called with a string character ID.
        """
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character("invalid")
    
    def test_get_character_invalid_id_none(self, character_client):
        """
        Test that get_character raises ValueError when called with None as the character ID.
        """
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.get_character(None)
    
    @patch('requests.Session.get')
    def test_get_character_http_error(self, mock_get, character_client):
        """
        Test that get_character raises an HTTPError when the API responds with an HTTP error status.
        """
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.get_character(999)
    
    @patch('requests.Session.get')
    def test_get_character_connection_error(self, mock_get, character_client):
        """
        Test that get_character raises a ConnectionError when a connection failure occurs.
        """
        mock_get.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            character_client.get_character(1)
    
    @patch('requests.Session.get')
    def test_get_character_timeout(self, mock_get, character_client):
        """
        Test that get_character raises a Timeout exception when the request times out.
        """
        mock_get.side_effect = Timeout("Request timed out")
        
        with pytest.raises(Timeout):
            character_client.get_character(1)

    # Create Character Tests
    @patch('requests.Session.post')
    def test_create_character_success(self, mock_post, character_client):
        """
        Verifies that creating a character with valid data returns the expected character object and sends the correct POST request.
        """
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
        """
        Test that creating a character with empty data raises a ValueError.
        """
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.create_character({})
    
    def test_create_character_none_data(self, character_client):
        """
        Test that create_character raises ValueError when provided with None as character data.
        """
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.create_character(None)
    
    def test_create_character_non_dict_data(self, character_client):
        """
        Test that create_character raises ValueError when provided non-dictionary data.
        """
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.create_character("invalid")
    
    def test_create_character_missing_name(self, character_client):
        """
        Test that create_character raises ValueError when the 'name' field is missing from the character data.
        """
        character_data = {"class": "Warrior", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'name' is missing or empty"):
            character_client.create_character(character_data)
    
    def test_create_character_missing_class(self, character_client):
        """
        Test that create_character raises ValueError when the 'class' field is missing from the character data.
        """
        character_data = {"name": "Test Character", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'class' is missing or empty"):
            character_client.create_character(character_data)
    
    def test_create_character_empty_name(self, character_client):
        """
        Test that creating a character with an empty 'name' field raises a ValueError.
        """
        character_data = {"name": "", "class": "Warrior", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'name' is missing or empty"):
            character_client.create_character(character_data)
    
    def test_create_character_empty_class(self, character_client):
        """
        Test that creating a character with an empty 'class' field raises a ValueError.
        """
        character_data = {"name": "Test Character", "class": "", "level": 1}
        
        with pytest.raises(ValueError, match="Required field 'class' is missing or empty"):
            character_client.create_character(character_data)
    
    @patch('requests.Session.post')
    def test_create_character_http_error(self, mock_post, character_client):
        """
        Test that create_character raises an HTTPError when the API responds with an HTTP error during character creation.
        """
        character_data = {"name": "Test Character", "class": "Warrior"}
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("422 Validation Error")
        mock_post.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.create_character(character_data)

    # Update Character Tests
    @patch('requests.Session.put')
    def test_update_character_success(self, mock_put, character_client, sample_character):
        """
        Test that updating a character with valid data returns the updated character and sends the correct PUT request.
        """
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
        """
        Test that update_character raises ValueError when given an invalid character ID.
        """
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.update_character(0, {"level": 20})
    
    def test_update_character_empty_data(self, character_client):
        """
        Test that updating a character with empty data raises a ValueError.
        """
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.update_character(1, {})
    
    def test_update_character_none_data(self, character_client):
        """
        Test that update_character raises ValueError when provided with None as character data.
        """
        with pytest.raises(ValueError, match="Character data must be a non-empty dictionary"):
            character_client.update_character(1, None)
    
    @patch('requests.Session.put')
    def test_update_character_not_found(self, mock_put, character_client):
        """
        Test that updating a non-existent character raises an HTTPError.
        
        Simulates a 404 Not Found response when attempting to update a character, and verifies that the client raises an HTTPError.
        """
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_put.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.update_character(999, {"level": 20})

    # Delete Character Tests
    @patch('requests.Session.delete')
    def test_delete_character_success(self, mock_delete, character_client):
        """
        Test that deleting a character with a valid ID returns True and sends the correct DELETE request.
        """
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
        """
        Test that delete_character raises ValueError when given an invalid character ID.
        """
        with pytest.raises(ValueError, match="Character ID must be a positive integer"):
            character_client.delete_character(-1)
    
    @patch('requests.Session.delete')
    def test_delete_character_not_found(self, mock_delete, character_client):
        """
        Test that attempting to delete a non-existent character raises an HTTPError.
        """
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_delete.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.delete_character(999)

    # List Characters Tests
    @patch('requests.Session.get')
    def test_list_characters_success(self, mock_get, character_client, sample_character_list):
        """
        Test that listing characters returns the expected response data when the API call is successful.
        
        Verifies that the `list_characters` method of `CharacterClient` correctly retrieves and returns a list of characters, and that the request is made with the appropriate URL, query parameters, and timeout.
        """
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
        """
        Test that listing characters with custom limit and offset parameters returns the expected paginated response.
        
        Verifies that the client sends the correct query parameters for pagination and processes the API response as expected.
        """
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
        """
        Verify that listing characters with filter parameters correctly merges filters into the request and calls the API with the expected query parameters.
        """
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
        """
        Test that calling list_characters with a limit of zero raises a ValueError.
        """
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.list_characters(limit=0)
    
    def test_list_characters_invalid_limit_negative(self, character_client):
        """
        Test that calling list_characters with a negative limit raises a ValueError.
        """
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.list_characters(limit=-1)
    
    def test_list_characters_invalid_limit_string(self, character_client):
        """
        Test that providing a non-integer string as the limit to list_characters raises a ValueError.
        """
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.list_characters(limit="invalid")
    
    def test_list_characters_invalid_offset_negative(self, character_client):
        """
        Test that list_characters raises a ValueError when called with a negative offset.
        """
        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            character_client.list_characters(offset=-1)
    
    def test_list_characters_invalid_offset_string(self, character_client):
        """
        Test that providing a non-integer string as the offset to list_characters raises a ValueError.
        """
        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            character_client.list_characters(offset="invalid")

    # Search Characters Tests
    @patch('requests.Session.get')
    def test_search_characters_success(self, mock_get, character_client, sample_character_list):
        """
        Verify that searching for characters with a valid query returns the expected list of characters and constructs the correct API request.
        """
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
        """
        Test that searching for characters with a custom limit parameter sends the correct request and parameters.
        """
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
        """
        Verify that searching for characters with an empty query string raises a ValueError.
        """
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            character_client.search_characters("")
    
    def test_search_characters_none_query(self, character_client):
        """
        Verify that passing None as the query to search_characters raises a ValueError.
        """
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            character_client.search_characters(None)
    
    def test_search_characters_non_string_query(self, character_client):
        """
        Test that search_characters raises ValueError when the query is not a string.
        """
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            character_client.search_characters(123)
    
    def test_search_characters_invalid_limit(self, character_client):
        """
        Test that `search_characters` raises a ValueError when called with a non-positive limit.
        """
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            character_client.search_characters("test", limit=0)

    # Get Character Stats Tests
    @patch('requests.Session.get')
    def test_get_character_stats_success(self, mock_get, character_client):
        """
        Verifies that character stats are successfully retrieved and returned when the API responds with valid data.
        """
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
        """
        Test that get_character_stats raises HTTPError when the character does not exist.
        """
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            character_client.get_character_stats(999)

    # Edge Cases and Error Handling Tests
    @patch('requests.Session.get')
    def test_malformed_json_response(self, mock_get, character_client):
        """
        Test that the client raises a JSONDecodeError when the API returns malformed JSON in the response.
        """
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with pytest.raises(json.JSONDecodeError):
            character_client.get_character(1)
    
    @patch('requests.Session.get')
    def test_request_exception_handling(self, mock_get, character_client):
        """
        Test that a general RequestException raised during a character retrieval is properly propagated.
        """
        mock_get.side_effect = RequestException("General request error")
        
        with pytest.raises(RequestException):
            character_client.get_character(1)
    
    def test_logging_on_error(self, character_client, caplog):
        """
        Verify that connection errors during character retrieval are logged at the error level.
        """
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection failed")
            
            with caplog.at_level(logging.ERROR):
                with pytest.raises(ConnectionError):
                    character_client.get_character(1)
            
            assert "Failed to get character 1" in caplog.text
            assert "Connection failed" in caplog.text
    
    def test_large_character_id(self, character_client):
        """
        Verify that the client correctly retrieves a character when provided with a very large character ID.
        """
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
        """
        Verify that the client correctly handles creation of character data containing Unicode characters, including emojis and accented letters.
        """
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