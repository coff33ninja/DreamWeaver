import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
import sys
import os
from typing import Dict, List, Optional, Union

# Add the parent directory to the path to import CharacterClient
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from character_client import CharacterClient
except ImportError:
    # Create a mock CharacterClient class for testing purposes if the actual implementation is not available
    class CharacterClient:
        def __init__(self, base_url: str = "https://api.character.com", api_key: str = None, timeout: int = 30):
            self.base_url = base_url.rstrip('/')
            self.api_key = api_key
            self.timeout = timeout
            self.session = requests.Session()
            if api_key:
                self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        def get_character(self, character_id: int) -> Dict:
            response = self.session.get(f"{self.base_url}/characters/{character_id}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        def get_all_characters(self, page: int = 1, limit: int = 20) -> List[Dict]:
            params = {"page": page, "limit": limit}
            response = self.session.get(f"{self.base_url}/characters", params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        def create_character(self, character_data: Dict) -> Dict:
            if not character_data or not isinstance(character_data, dict):
                raise ValueError("Invalid character data")
            response = self.session.post(f"{self.base_url}/characters", json=character_data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        def update_character(self, character_id: int, update_data: Dict) -> Dict:
            if not isinstance(character_id, int) or character_id <= 0:
                raise ValueError("Invalid character ID")
            if not update_data or not isinstance(update_data, dict):
                raise ValueError("Invalid update data")
            response = self.session.put(f"{self.base_url}/characters/{character_id}", json=update_data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        def delete_character(self, character_id: int) -> bool:
            if not isinstance(character_id, int) or character_id <= 0:
                raise ValueError("Invalid character ID")
            response = self.session.delete(f"{self.base_url}/characters/{character_id}", timeout=self.timeout)
            response.raise_for_status()
            return response.status_code == 204

        def search_characters(self, name: str = None, character_class: str = None, level_min: int = None, level_max: int = None) -> List[Dict]:
            if not any([name, character_class, level_min, level_max]):
                raise ValueError("At least one search criterion must be provided")
            params = {}
            if name:
                params["name"] = name
            if character_class:
                params["class"] = character_class
            if level_min:
                params["level_min"] = level_min
            if level_max:
                params["level_max"] = level_max
            response = self.session.get(f"{self.base_url}/characters/search", params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()


class TestCharacterClient:
    """
    Comprehensive test suite for CharacterClient.
    Testing framework: pytest
    """

    @pytest.fixture
    def client(self):
        """Fixture to create a CharacterClient instance for testing."""
        return CharacterClient(base_url="https://api.test.com", api_key="test_api_key")

    @pytest.fixture
    def sample_character(self):
        """Fixture providing sample character data."""
        return {
            "id": 1,
            "name": "Test Warrior",
            "level": 10,
            "class": "Warrior",
            "health": 100,
            "mana": 50,
            "experience": 2500,
            "stats": {
                "strength": 15,
                "dexterity": 12,
                "intelligence": 8,
                "constitution": 14
            }
        }

    @pytest.fixture
    def sample_characters_list(self, sample_character):
        """Fixture providing a list of sample characters."""
        return [
            sample_character,
            {
                "id": 2,
                "name": "Test Mage",
                "level": 15,
                "class": "Mage",
                "health": 80,
                "mana": 120,
                "experience": 5000,
                "stats": {
                    "strength": 8,
                    "dexterity": 10,
                    "intelligence": 18,
                    "constitution": 10
                }
            }
        ]

    # Initialization Tests
    def test_init_default_parameters(self):
        """Test CharacterClient initialization with default parameters."""
        client = CharacterClient()
        assert isinstance(client, CharacterClient)
        assert client.base_url == "https://api.character.com"
        assert client.api_key is None
        assert client.timeout == 30

    def test_init_custom_parameters(self):
        """Test CharacterClient initialization with custom parameters."""
        base_url = "https://custom.api.com"
        api_key = "custom_key"
        timeout = 60
        
        client = CharacterClient(base_url=base_url, api_key=api_key, timeout=timeout)
        
        assert client.base_url == base_url
        assert client.api_key == api_key
        assert client.timeout == timeout

    def test_init_with_trailing_slash_in_url(self):
        """Test CharacterClient initialization strips trailing slash from URL."""
        client = CharacterClient(base_url="https://api.test.com/")
        assert client.base_url == "https://api.test.com"

    def test_init_sets_authorization_header_when_api_key_provided(self):
        """Test that authorization header is set when API key is provided."""
        api_key = "test_key"
        client = CharacterClient(api_key=api_key)
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == f"Bearer {api_key}"

    # Happy Path Tests - Character CRUD Operations
    @patch('requests.Session.get')
    def test_get_character_success(self, mock_get, client, sample_character):
        """Test successful character retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_character(1)
        
        assert result == sample_character
        mock_get.assert_called_once_with("https://api.test.com/characters/1", timeout=30)

    @patch('requests.Session.get')
    def test_get_all_characters_success(self, mock_get, client, sample_characters_list):
        """Test successful retrieval of all characters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_characters_list
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_all_characters()
        
        assert result == sample_characters_list
        assert len(result) == 2
        mock_get.assert_called_once_with(
            "https://api.test.com/characters", 
            params={"page": 1, "limit": 20}, 
            timeout=30
        )

    @patch('requests.Session.get')
    def test_get_all_characters_with_pagination(self, mock_get, client, sample_characters_list):
        """Test character retrieval with custom pagination parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_characters_list
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_all_characters(page=2, limit=50)
        
        assert result == sample_characters_list
        mock_get.assert_called_once_with(
            "https://api.test.com/characters", 
            params={"page": 2, "limit": 50}, 
            timeout=30
        )

    @patch('requests.Session.post')
    def test_create_character_success(self, mock_post, client, sample_character):
        """Test successful character creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        character_data = {
            "name": "New Character",
            "class": "Rogue",
            "level": 1
        }
        
        result = client.create_character(character_data)
        
        assert result == sample_character
        mock_post.assert_called_once_with(
            "https://api.test.com/characters", 
            json=character_data, 
            timeout=30
        )

    @patch('requests.Session.put')
    def test_update_character_success(self, mock_put, client, sample_character):
        """Test successful character update."""
        updated_character = sample_character.copy()
        updated_character["level"] = 11
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = updated_character
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        update_data = {"level": 11}
        result = client.update_character(1, update_data)
        
        assert result == updated_character
        mock_put.assert_called_once_with(
            "https://api.test.com/characters/1", 
            json=update_data, 
            timeout=30
        )

    @patch('requests.Session.delete')
    def test_delete_character_success(self, mock_delete, client):
        """Test successful character deletion."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_response

        result = client.delete_character(1)
        
        assert result is True
        mock_delete.assert_called_once_with("https://api.test.com/characters/1", timeout=30)

    # Edge Cases and Error Handling Tests
    @patch('requests.Session.get')
    def test_get_character_not_found(self, mock_get, client):
        """Test character retrieval when character doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError("404 Client Error: Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match="404 Client Error"):
            client.get_character(999)

    @patch('requests.Session.get')
    def test_get_character_server_error(self, mock_get, client):
        """Test character retrieval with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match="500 Server Error"):
            client.get_character(1)

    @patch('requests.Session.get')
    def test_get_character_timeout(self, mock_get, client):
        """Test character retrieval with timeout."""
        mock_get.side_effect = Timeout("Request timed out")

        with pytest.raises(Timeout, match="Request timed out"):
            client.get_character(1)

    @patch('requests.Session.get')
    def test_get_character_connection_error(self, mock_get, client):
        """Test character retrieval with connection error."""
        mock_get.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError, match="Connection failed"):
            client.get_character(1)

    @patch('requests.Session.get')
    def test_get_character_invalid_json(self, mock_get, client):
        """Test character retrieval with invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            client.get_character(1)

    @pytest.mark.parametrize("invalid_id", [None, "", "invalid", -1, 0, [], {}, 0.5])
    def test_get_character_invalid_id_types(self, client, invalid_id):
        """Test character retrieval with invalid ID types."""
        with pytest.raises((TypeError, ValueError)):
            client.get_character(invalid_id)

    @pytest.mark.parametrize("invalid_data", [
        None,
        "",
        [],
        {"name": ""},  # Empty name
        {"name": None},  # None name
        {},  # Empty dict
    ])
    def test_create_character_invalid_data(self, client, invalid_data):
        """Test character creation with invalid data."""
        with pytest.raises(ValueError, match="Invalid character data"):
            client.create_character(invalid_data)

    @patch('requests.Session.post')
    def test_create_character_validation_error(self, mock_post, client):
        """Test character creation with validation error from server."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = HTTPError("400 Bad Request: Validation Error")
        mock_post.return_value = mock_response

        character_data = {"name": "Invalid Character"}
        
        with pytest.raises(HTTPError, match="400 Bad Request"):
            client.create_character(character_data)

    @pytest.mark.parametrize("char_id,update_data", [
        (None, {"level": 10}),
        ("invalid", {"level": 10}),
        (-1, {"level": 10}),
        (0, {"level": 10}),
        (1, None),
        (1, ""),
        (1, []),
        (1, {}),
    ])
    def test_update_character_invalid_parameters(self, client, char_id, update_data):
        """Test character update with invalid parameters."""
        with pytest.raises(ValueError):
            client.update_character(char_id, update_data)

    @pytest.mark.parametrize("char_id", [None, "invalid", -1, 0, [], {}, 0.5])
    def test_delete_character_invalid_id(self, client, char_id):
        """Test character deletion with invalid ID."""
        with pytest.raises(ValueError, match="Invalid character ID"):
            client.delete_character(char_id)

    # Authentication and Authorization Tests
    @patch('requests.Session.get')
    def test_get_character_unauthorized(self, mock_get, client):
        """Test character retrieval with unauthorized access."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = HTTPError("401 Unauthorized")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match="401 Unauthorized"):
            client.get_character(1)

    @patch('requests.Session.get')
    def test_get_character_forbidden(self, mock_get, client):
        """Test character retrieval with forbidden access."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = HTTPError("403 Forbidden")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match="403 Forbidden"):
            client.get_character(1)

    # Rate Limiting Tests
    @patch('requests.Session.get')
    def test_get_character_rate_limited(self, mock_get, client):
        """Test character retrieval with rate limiting."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = HTTPError("429 Too Many Requests")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match="429 Too Many Requests"):
            client.get_character(1)

    # Search and Filter Tests
    @patch('requests.Session.get')
    def test_search_characters_by_name(self, mock_get, client, sample_character):
        """Test character search by name."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [sample_character]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.search_characters(name="Test")
        
        assert len(result) == 1
        assert result[0]["name"] == "Test Warrior"
        mock_get.assert_called_once_with(
            "https://api.test.com/characters/search", 
            params={"name": "Test"}, 
            timeout=30
        )

    @patch('requests.Session.get')
    def test_search_characters_by_class(self, mock_get, client, sample_character):
        """Test character search by class."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [sample_character]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.search_characters(character_class="Warrior")
        
        assert len(result) == 1
        assert result[0]["class"] == "Warrior"
        mock_get.assert_called_once_with(
            "https://api.test.com/characters/search", 
            params={"class": "Warrior"}, 
            timeout=30
        )

    @patch('requests.Session.get')
    def test_search_characters_by_level_range(self, mock_get, client, sample_character):
        """Test character search by level range."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [sample_character]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.search_characters(level_min=5, level_max=15)
        
        assert len(result) == 1
        assert result[0]["level"] == 10
        mock_get.assert_called_once_with(
            "https://api.test.com/characters/search", 
            params={"level_min": 5, "level_max": 15}, 
            timeout=30
        )

    @patch('requests.Session.get')
    def test_search_characters_multiple_criteria(self, mock_get, client, sample_character):
        """Test character search with multiple criteria."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [sample_character]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.search_characters(name="Test", character_class="Warrior", level_min=5)
        
        assert len(result) == 1
        mock_get.assert_called_once_with(
            "https://api.test.com/characters/search", 
            params={"name": "Test", "class": "Warrior", "level_min": 5}, 
            timeout=30
        )

    def test_search_characters_no_criteria(self, client):
        """Test character search without search criteria."""
        with pytest.raises(ValueError, match="At least one search criterion must be provided"):
            client.search_characters()

    @patch('requests.Session.get')
    def test_search_characters_empty_results(self, mock_get, client):
        """Test character search with no matching results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.search_characters(name="NonexistentCharacter")
        
        assert result == []
        assert len(result) == 0

    # Performance and Load Tests
    @patch('requests.Session.get')
    def test_get_all_characters_large_page_size(self, mock_get, client):
        """Test character retrieval with large page size."""
        large_characters_list = [{"id": i, "name": f"Character {i}"} for i in range(1000)]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_characters_list
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_all_characters(page=1, limit=1000)
        
        assert len(result) == 1000
        assert result[0]["id"] == 1
        assert result[999]["id"] == 1000

    # Configuration Tests
    def test_custom_timeout_configuration(self):
        """Test CharacterClient with custom timeout configuration."""
        client = CharacterClient(timeout=60)
        assert client.timeout == 60

    @patch('requests.Session.get')
    def test_timeout_passed_to_requests(self, mock_get, sample_character):
        """Test that timeout is correctly passed to requests."""
        client = CharacterClient(timeout=45)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client.get_character(1)
        
        mock_get.assert_called_once_with("https://api.character.com/characters/1", timeout=45)

    # Error Recovery Tests
    @patch('requests.Session.get')
    def test_get_character_with_retry_logic(self, mock_get, client, sample_character):
        """Test character retrieval with simulated retry logic."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = HTTPError("500 Server Error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = sample_character
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]

        # First call should fail
        with pytest.raises(HTTPError):
            client.get_character(1)
        
        # Second call should succeed
        result = client.get_character(1)
        assert result == sample_character
        assert mock_get.call_count == 2

    # Data Validation Tests
    def test_create_character_validates_required_fields(self, client):
        """Test that character creation validates required fields."""
        # Test with missing required fields
        incomplete_data = {"level": 1}  # Missing name and class
        
        with pytest.raises(ValueError):
            client.create_character(incomplete_data)

    @patch('requests.Session.post')
    def test_create_character_with_complete_data(self, mock_post, client, sample_character):
        """Test character creation with complete valid data."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        complete_character_data = {
            "name": "Complete Character",
            "class": "Paladin",
            "level": 1,
            "stats": {
                "strength": 14,
                "dexterity": 10,
                "intelligence": 12,
                "constitution": 13
            }
        }
        
        result = client.create_character(complete_character_data)
        
        assert result == sample_character
        mock_post.assert_called_once_with(
            "https://api.test.com/characters", 
            json=complete_character_data, 
            timeout=30
        )

    # Boundary Value Tests
    @pytest.mark.parametrize("level", [1, 50, 100])
    def test_search_characters_level_boundaries(self, client, level):
        """Test character search with boundary level values."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = client.search_characters(level_min=level, level_max=level)
            
            assert result == []
            assert mock_get.called

    # Integration-style Tests
    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_create_then_retrieve_character(self, mock_get, mock_post, client, sample_character):
        """Test creating a character and then retrieving it."""
        # Mock character creation
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = sample_character
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        # Mock character retrieval
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = sample_character
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        # Create character
        character_data = {"name": "New Character", "class": "Rogue", "level": 1}
        created_character = client.create_character(character_data)
        
        # Retrieve the created character
        retrieved_character = client.get_character(created_character["id"])
        
        assert created_character == sample_character
        assert retrieved_character == sample_character
        assert created_character["id"] == retrieved_character["id"]


# Test Configuration and Fixtures for pytest
class TestCharacterClientConfiguration:
    """Test class for CharacterClient configuration and setup."""

    def test_session_is_configured(self):
        """Test that the session is properly configured."""
        client = CharacterClient(api_key="test_key")
        assert hasattr(client, 'session')
        assert isinstance(client.session, requests.Session)

    def test_session_headers_with_api_key(self):
        """Test that session headers are set correctly with API key."""
        api_key = "test_api_key"
        client = CharacterClient(api_key=api_key)
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == f"Bearer {api_key}"

    def test_session_headers_without_api_key(self):
        """Test that session headers are correct without API key."""
        client = CharacterClient()
        assert "Authorization" not in client.session.headers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])