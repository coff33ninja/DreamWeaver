import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock, call
import json
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
import sys
import os
import logging

# Add the parent directory to the path to import CharacterClient
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from character_client import CharacterClient
except ImportError:
    # Try alternative import paths
    try:
        from CharacterClient.character_client import CharacterClient
    except ImportError:
        from src.character_client import CharacterClient


class TestCharacterClient:
    """Comprehensive test suite for CharacterClient using pytest framework."""
    
    @pytest.fixture
    def client(self):
        """
        Returns a CharacterClient instance configured for testing with a test API base URL, API key, and timeout.
        """
        return CharacterClient(
            base_url="https://test-api.characters.com",
            api_key="test-api-key",
            timeout=15
        )
    
    @pytest.fixture
    def client_no_auth(self):
        """
        Create and return a CharacterClient instance configured without an API key for authentication.
        """
        return CharacterClient(base_url="https://test-api.characters.com")
    
    @pytest.fixture
    def sample_character_data(self):
        """
        Provides a sample character dictionary for use in tests.
        
        Returns:
            dict: Example character data including ID, name, level, class, attributes, equipment, and timestamps.
        """
        return {
            "id": "char_123",
            "name": "Test Hero",
            "level": 10,
            "class": "Warrior",
            "attributes": {
                "strength": 15,
                "dexterity": 12,
                "intelligence": 8,
                "health": 100
            },
            "equipment": ["iron_sword", "leather_armor"],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
    
    @pytest.fixture
    def sample_stats_data(self):
        """
        Provides sample statistics data for a character, including battles won and lost, experience points, gold, and achievements.
        
        Returns:
            dict: A dictionary containing sample character statistics.
        """
        return {
            "battles_won": 25,
            "battles_lost": 5,
            "experience_points": 1500,
            "gold": 750,
            "achievements": ["first_victory", "level_10"]
        }

    # Constructor Tests
    def test_constructor_with_default_values(self):
        """
        Verify that the CharacterClient constructor sets default base URL, API key, timeout, and creates a session.
        """
        client = CharacterClient()
        assert client.base_url == "https://api.characters.com"
        assert client.api_key is None
        assert client.timeout == 30
        assert hasattr(client, 'session')
        assert isinstance(client.session, requests.Session)
    
    def test_constructor_with_custom_values(self):
        """
        Test that the CharacterClient constructor correctly applies custom base URL, API key, and timeout values, including normalization of the base URL.
        """
        client = CharacterClient(
            base_url="https://custom.api.com/",  # Test trailing slash removal
            api_key="custom-key-123",
            timeout=45
        )
        assert client.base_url == "https://custom.api.com"  # Should strip trailing slash
        assert client.api_key == "custom-key-123"
        assert client.timeout == 45
    
    def test_constructor_base_url_normalization(self):
        """
        Verify that the CharacterClient constructor normalizes the base_url by removing any trailing slashes.
        """
        test_cases = [
            ("https://api.test.com", "https://api.test.com"),
            ("https://api.test.com/", "https://api.test.com"),
            ("https://api.test.com///", "https://api.test.com"),
        ]
        
        for input_url, expected_url in test_cases:
            client = CharacterClient(base_url=input_url)
            assert client.base_url == expected_url

    # _get_headers Tests
    def test_get_headers_with_api_key(self, client):
        """
        Verify that the _get_headers method includes the correct headers when an API key is provided.
        """
        headers = client._get_headers()
        
        expected_headers = {
            "Content-Type": "application/json",
            "User-Agent": "CharacterClient/1.0",
            "Authorization": "Bearer test-api-key"
        }
        assert headers == expected_headers
    
    def test_get_headers_without_api_key(self, client_no_auth):
        """
        Verify that the _get_headers method returns headers without the Authorization field when no API key is provided.
        """
        headers = client_no_auth._get_headers()
        
        expected_headers = {
            "Content-Type": "application/json",
            "User-Agent": "CharacterClient/1.0"
        }
        assert headers == expected_headers
        assert "Authorization" not in headers
    
    def test_get_headers_with_empty_api_key(self):
        """
        Verify that _get_headers includes an Authorization header with an empty API key.
        
        Ensures that when the API key is an empty string, the Authorization header is still present with the value 'Bearer '.
        """
        client = CharacterClient(api_key="")
        headers = client._get_headers()
        
        # Empty API key should still add Authorization header
        assert headers["Authorization"] == "Bearer "

    # _make_request Tests
    @patch('requests.Session.request')
    def test_make_request_success(self, mock_request, client, sample_character_data):
        """
        Test that _make_request returns parsed JSON data on a successful HTTP response.
        
        Verifies that the method constructs the request with correct parameters and headers, and that the returned result matches the expected character data.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = sample_character_data
        mock_response.raise_for_status.return_value = None
        mock_response.content = json.dumps(sample_character_data).encode()
        mock_request.return_value = mock_response
        
        result = client._make_request("GET", "/test-endpoint")
        
        assert result == sample_character_data
        mock_request.assert_called_once_with(
            method="GET",
            url="https://test-api.characters.com/test-endpoint",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "CharacterClient/1.0",
                "Authorization": "Bearer test-api-key"
            },
            timeout=15
        )
    
    @patch('requests.Session.request')
    def test_make_request_empty_response(self, mock_request, client):
        """
        Test that _make_request returns an empty dictionary when the HTTP response has no content.
        """
        mock_response = MagicMock()
        mock_response.content = b""
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        result = client._make_request("DELETE", "/test-endpoint")
        
        assert result == {}
    
    @patch('requests.Session.request')
    def test_make_request_with_kwargs(self, mock_request, client):
        """
        Verify that the _make_request method correctly handles and forwards additional keyword arguments such as JSON payloads and query parameters.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'{"success": true}'
        mock_request.return_value = mock_response
        
        test_data = {"name": "test"}
        result = client._make_request("POST", "/test", json=test_data, params={"param": "value"})
        
        mock_request.assert_called_once_with(
            method="POST",
            url="https://test-api.characters.com/test",
            headers=mock.ANY,
            timeout=15,
            json=test_data,
            params={"param": "value"}
        )
    
    @patch('requests.Session.request')
    def test_make_request_http_error(self, mock_request, client):
        """
        Test that _make_request raises an HTTPError when the response indicates an HTTP error.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_request.return_value = mock_response
        
        with pytest.raises(HTTPError, match="404 Not Found"):
            client._make_request("GET", "/nonexistent")
    
    @patch('requests.Session.request')
    def test_make_request_connection_error(self, mock_request, client):
        """
        Test that `_make_request` raises a `ConnectionError` when a connection failure occurs.
        """
        mock_request.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError, match="Connection failed"):
            client._make_request("GET", "/test")
    
    @patch('requests.Session.request')
    def test_make_request_timeout(self, mock_request, client):
        """
        Test that the client's _make_request method raises a Timeout exception when a request times out.
        """
        mock_request.side_effect = Timeout("Request timed out")
        
        with pytest.raises(Timeout, match="Request timed out"):
            client._make_request("GET", "/test")
    
    @patch('requests.Session.request')
    def test_make_request_json_decode_error(self, mock_request, client):
        """
        Test that _make_request raises a JSONDecodeError when the response contains invalid JSON.
        """
        mock_response = MagicMock()
        mock_response.content = b"invalid json"
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        with pytest.raises(json.JSONDecodeError):
            client._make_request("GET", "/test")

    # get_character Tests
    @patch.object(CharacterClient, '_make_request')
    def test_get_character_success(self, mock_make_request, client, sample_character_data):
        """
        Test that `get_character` successfully retrieves and returns character data when the API responds with valid data.
        """
        mock_make_request.return_value = sample_character_data
        
        result = client.get_character("char_123")
        
        assert result == sample_character_data
        mock_make_request.assert_called_once_with("GET", "/characters/char_123")
    
    @patch.object(CharacterClient, '_make_request')
    def test_get_character_not_found(self, mock_make_request, client):
        """
        Test that `get_character` raises an HTTPError when the character is not found.
        
        Simulates a 404 Not Found error and verifies that the client raises the appropriate exception.
        """
        mock_make_request.side_effect = HTTPError("404 Not Found")
        
        with pytest.raises(HTTPError, match="404 Not Found"):
            client.get_character("nonexistent_char")
    
    @pytest.mark.parametrize("character_id", [
        "char_123", "test-char", "CHAR_ABC", "123", "", "special@char!"
    ])
    @patch.object(CharacterClient, '_make_request')
    def test_get_character_various_ids(self, mock_make_request, client, character_id, sample_character_data):
        """
        Test retrieving a character using various ID formats.
        
        Verifies that the client correctly fetches character data for different types of character IDs and constructs the appropriate API request.
        """
        mock_make_request.return_value = sample_character_data
        
        result = client.get_character(character_id)
        
        assert result == sample_character_data
        mock_make_request.assert_called_once_with("GET", f"/characters/{character_id}")

    # create_character Tests
    @patch.object(CharacterClient, '_make_request')
    def test_create_character_success(self, mock_make_request, client, sample_character_data):
        """
        Verify that creating a character with valid data returns the expected character object and triggers the correct API request.
        """
        created_character = sample_character_data.copy()
        mock_make_request.return_value = created_character
        
        character_data = sample_character_data.copy()
        del character_data["id"]  # Remove ID as it should be generated
        
        result = client.create_character(character_data)
        
        assert result == created_character
        mock_make_request.assert_called_once_with("POST", "/characters", json=character_data)
    
    @patch.object(CharacterClient, '_make_request')
    def test_create_character_validation_error(self, mock_make_request, client):
        """
        Test that creating a character with invalid data raises a validation error.
        
        Simulates a 422 Validation Error when attempting to create a character with invalid attributes, and verifies that an HTTPError is raised.
        """
        mock_make_request.side_effect = HTTPError("422 Validation Error")
        
        invalid_data = {"name": "", "level": -1}
        
        with pytest.raises(HTTPError, match="422 Validation Error"):
            client.create_character(invalid_data)
    
    @patch.object(CharacterClient, '_make_request')
    def test_create_character_empty_data(self, mock_make_request, client):
        """
        Test that creating a character with empty data returns an error response.
        
        Verifies that the client handles empty input by returning the expected error dictionary from the API.
        """
        mock_make_request.return_value = {"error": "Invalid data"}
        
        result = client.create_character({})
        
        assert result == {"error": "Invalid data"}
        mock_make_request.assert_called_once_with("POST", "/characters", json={})

    # update_character Tests
    @patch.object(CharacterClient, '_make_request')
    def test_update_character_success(self, mock_make_request, client, sample_character_data):
        """
        Test that updating a character with valid data returns the updated character information.
        
        Verifies that the `update_character` method sends the correct request and returns the expected updated character data.
        """
        updated_character = sample_character_data.copy()
        updated_character["level"] = 15
        mock_make_request.return_value = updated_character
        
        update_data = {"level": 15}
        result = client.update_character("char_123", update_data)
        
        assert result == updated_character
        mock_make_request.assert_called_once_with("PUT", "/characters/char_123", json=update_data)
    
    @patch.object(CharacterClient, '_make_request')
    def test_update_character_not_found(self, mock_make_request, client):
        """
        Test that updating a non-existent character raises an HTTPError with a 404 Not Found message.
        """
        mock_make_request.side_effect = HTTPError("404 Not Found")
        
        with pytest.raises(HTTPError, match="404 Not Found"):
            client.update_character("nonexistent", {"level": 20})
    
    @patch.object(CharacterClient, '_make_request')
    def test_update_character_partial_update(self, mock_make_request, client):
        """
        Test that updating a character with partial data correctly updates only the specified fields.
        
        Verifies that the client sends a partial update request and receives the expected updated data.
        """
        mock_make_request.return_value = {"id": "char_123", "level": 25}
        
        partial_data = {"level": 25}
        result = client.update_character("char_123", partial_data)
        
        assert result["level"] == 25
        mock_make_request.assert_called_once_with("PUT", "/characters/char_123", json=partial_data)

    # delete_character Tests
    @patch.object(CharacterClient, '_make_request')
    def test_delete_character_success(self, mock_make_request, client):
        """
        Test that deleting a character by ID returns True on successful deletion.
        """
        mock_make_request.return_value = {}
        
        result = client.delete_character("char_123")
        
        assert result is True
        mock_make_request.assert_called_once_with("DELETE", "/characters/char_123")
    
    @patch.object(CharacterClient, '_make_request')
    def test_delete_character_not_found(self, mock_make_request, client):
        """
        Test that deleting a non-existent character raises an HTTPError with a 404 Not Found message.
        """
        mock_make_request.side_effect = HTTPError("404 Not Found")
        
        with pytest.raises(HTTPError, match="404 Not Found"):
            client.delete_character("nonexistent")
    
    @patch.object(CharacterClient, '_make_request')
    def test_delete_character_forbidden(self, mock_make_request, client):
        """
        Test that attempting to delete a character without sufficient permissions raises an HTTPError with a 403 Forbidden message.
        """
        mock_make_request.side_effect = HTTPError("403 Forbidden")
        
        with pytest.raises(HTTPError, match="403 Forbidden"):
            client.delete_character("char_123")

    # list_characters Tests
    @patch.object(CharacterClient, '_make_request')
    def test_list_characters_default_params(self, mock_make_request, client):
        """
        Verify that listing characters with default parameters returns the expected empty result and correct pagination values.
        """
        expected_response = {
            "characters": [],
            "total": 0,
            "limit": 10,
            "offset": 0
        }
        mock_make_request.return_value = expected_response
        
        result = client.list_characters()
        
        assert result == expected_response
        mock_make_request.assert_called_once_with(
            "GET", "/characters", params={"limit": 10, "offset": 0}
        )
    
    @patch.object(CharacterClient, '_make_request')
    def test_list_characters_with_pagination(self, mock_make_request, client, sample_character_data):
        """
        Verify that listing characters with custom limit and offset parameters returns the expected paginated response.
        """
        expected_response = {
            "characters": [sample_character_data],
            "total": 100,
            "limit": 25,
            "offset": 50
        }
        mock_make_request.return_value = expected_response
        
        result = client.list_characters(limit=25, offset=50)
        
        assert result == expected_response
        mock_make_request.assert_called_once_with(
            "GET", "/characters", params={"limit": 25, "offset": 50}
        )
    
    @patch.object(CharacterClient, '_make_request')
    def test_list_characters_with_filters(self, mock_make_request, client):
        """
        Test that listing characters with filter parameters correctly passes those filters to the API and returns the expected response.
        """
        expected_response = {"characters": [], "total": 0}
        mock_make_request.return_value = expected_response
        
        filters = {"class": "Warrior", "level_min": 10}
        result = client.list_characters(filters=filters)
        
        expected_params = {"limit": 10, "offset": 0, "class": "Warrior", "level_min": 10}
        mock_make_request.assert_called_once_with("GET", "/characters", params=expected_params)
    
    @pytest.mark.parametrize("limit,offset", [
        (1, 0),      # Minimum pagination
        (100, 0),    # Standard pagination
        (1000, 0),   # Large limit
        (10, 10000), # Large offset
        (0, 0),      # Edge case: zero limit
        (-1, -5),    # Edge case: negative values
    ])
    @patch.object(CharacterClient, '_make_request')
    def test_list_characters_boundary_conditions(self, mock_make_request, client, limit, offset):
        """
        Test that list_characters correctly handles various boundary values for limit and offset parameters.
        
        Verifies that the client constructs the request with the specified limit and offset, and that the underlying request method is called with the expected parameters.
        """
        mock_make_request.return_value = {"characters": [], "total": 0}
        
        client.list_characters(limit=limit, offset=offset)
        
        mock_make_request.assert_called_once_with(
            "GET", "/characters", params={"limit": limit, "offset": offset}
        )

    # search_characters Tests
    @patch.object(CharacterClient, '_make_request')
    def test_search_characters_by_name(self, mock_make_request, client, sample_character_data):
        """
        Test that searching for characters by name returns the expected results and constructs the correct API request.
        """
        expected_response = {
            "results": [sample_character_data],
            "total": 1
        }
        mock_make_request.return_value = expected_response
        
        result = client.search_characters("Test Hero")
        
        assert result == expected_response
        mock_make_request.assert_called_once_with(
            "GET", "/characters/search", params={"q": "Test Hero", "type": "name"}
        )
    
    @patch.object(CharacterClient, '_make_request')
    def test_search_characters_by_class(self, mock_make_request, client):
        """
        Test that searching for characters by class returns the expected results and constructs the correct API request.
        """
        expected_response = {"results": [], "total": 0}
        mock_make_request.return_value = expected_response
        
        result = client.search_characters("Warrior", search_type="class")
        
        assert result == expected_response
        mock_make_request.assert_called_once_with(
            "GET", "/characters/search", params={"q": "Warrior", "type": "class"}
        )
    
    @pytest.mark.parametrize("query,search_type", [
        ("", "name"),           # Empty query
        ("Test@Character", "name"),  # Special characters
        ("Test Character!", "name"), # Punctuation
        ("Test/Character", "name"),  # Forward slash
        ("Test Character?", "name"), # Question mark
        ("Test Character", "level"), # Different search type
        ("Test Character", "equipment"), # Equipment search
    ])
    @patch.object(CharacterClient, '_make_request')
    def test_search_characters_various_queries(self, mock_make_request, client, query, search_type):
        """
        Test that the `search_characters` method correctly handles various query strings and search types.
        
        Verifies that the client constructs the appropriate request parameters for different search scenarios.
        """
        mock_make_request.return_value = {"results": [], "total": 0}
        
        client.search_characters(query, search_type)
        
        mock_make_request.assert_called_once_with(
            "GET", "/characters/search", params={"q": query, "type": search_type}
        )

    # get_character_stats Tests
    @patch.object(CharacterClient, '_make_request')
    def test_get_character_stats_success(self, mock_make_request, client, sample_stats_data):
        """
        Test that `get_character_stats` successfully retrieves and returns character statistics data.
        """
        mock_make_request.return_value = sample_stats_data
        
        result = client.get_character_stats("char_123")
        
        assert result == sample_stats_data
        mock_make_request.assert_called_once_with("GET", "/characters/char_123/stats")
    
    @patch.object(CharacterClient, '_make_request')
    def test_get_character_stats_not_found(self, mock_make_request, client):
        """
        Test that retrieving stats for a non-existent character raises an HTTPError with a 404 message.
        """
        mock_make_request.side_effect = HTTPError("404 Not Found")
        
        with pytest.raises(HTTPError, match="404 Not Found"):
            client.get_character_stats("nonexistent")

    # update_character_stats Tests
    @patch.object(CharacterClient, '_make_request')
    def test_update_character_stats_success(self, mock_make_request, client, sample_stats_data):
        """
        Verify that updating a character's stats returns the updated stats data when the operation is successful.
        """
        updated_stats = sample_stats_data.copy()
        updated_stats["experience_points"] = 2000
        mock_make_request.return_value = updated_stats
        
        stats_update = {"experience_points": 2000}
        result = client.update_character_stats("char_123", stats_update)
        
        assert result == updated_stats
        mock_make_request.assert_called_once_with(
            "PUT", "/characters/char_123/stats", json=stats_update
        )
    
    @patch.object(CharacterClient, '_make_request')
    def test_update_character_stats_validation_error(self, mock_make_request, client):
        """
        Test that updating character stats with invalid data raises an HTTPError for validation errors.
        """
        mock_make_request.side_effect = HTTPError("422 Validation Error")
        
        invalid_stats = {"experience_points": -100}  # Invalid negative XP
        
        with pytest.raises(HTTPError, match="422 Validation Error"):
            client.update_character_stats("char_123", invalid_stats)

    # Integration-style Tests
    @patch.object(CharacterClient, '_make_request')
    def test_full_character_lifecycle(self, mock_make_request, client, sample_character_data):
        """
        Test the complete lifecycle of a character, including creation, retrieval, update, and deletion.
        
        Simulates each operation using mocked responses and verifies that the client methods return expected results and that all lifecycle steps are executed in sequence.
        """
        # Setup mock responses for each operation
        created_character = sample_character_data.copy()
        updated_character = created_character.copy()
        updated_character["level"] = 15
        
        mock_make_request.side_effect = [
            created_character,  # create
            created_character,  # get
            updated_character,  # update
            {}                  # delete
        ]
        
        # Create character
        character_data = sample_character_data.copy()
        del character_data["id"]
        created = client.create_character(character_data)
        
        # Get character
        retrieved = client.get_character(created["id"])
        
        # Update character
        updated = client.update_character(created["id"], {"level": 15})
        
        # Delete character
        deleted = client.delete_character(created["id"])
        
        # Verify results
        assert created == created_character
        assert retrieved == created_character
        assert updated["level"] == 15
        assert deleted is True
        
        # Verify all calls were made
        assert mock_make_request.call_count == 4

    # Error Handling and Edge Cases
    @patch.object(CharacterClient, '_make_request')
    def test_rate_limiting_error(self, mock_make_request, client):
        """
        Test that the client raises an HTTPError when a rate limiting (429) error occurs during a request.
        """
        mock_make_request.side_effect = HTTPError("429 Too Many Requests")
        
        with pytest.raises(HTTPError, match="429 Too Many Requests"):
            client.get_character("char_123")
    
    @patch.object(CharacterClient, '_make_request')
    def test_server_error_handling(self, mock_make_request, client):
        """
        Test that the client raises an HTTPError when a server error (500) occurs during a character retrieval request.
        """
        mock_make_request.side_effect = HTTPError("500 Internal Server Error")
        
        with pytest.raises(HTTPError, match="500 Internal Server Error"):
            client.get_character("char_123")
    
    def test_client_immutability_of_config(self, client):
        """
        Verify that the client's base URL, API key, and timeout configuration remain unchanged after performing various operations.
        """
        original_base_url = client.base_url
        original_api_key = client.api_key
        original_timeout = client.timeout
        
        # Simulate some operations that shouldn't change config
        with patch.object(client, '_make_request'):
            try:
                client.get_character("test")
                client.create_character({"name": "test"})
                client.list_characters()
            except:
                pass  # We don't care about the result, just that config doesn't change
        
        assert client.base_url == original_base_url
        assert client.api_key == original_api_key
        assert client.timeout == original_timeout

    # Session Management Tests
    def test_session_creation(self, client):
        """
        Verify that the client instance initializes a `requests.Session` object as its `session` attribute.
        """
        assert hasattr(client, 'session')
        assert isinstance(client.session, requests.Session)
    
    def test_session_reuse(self):
        """
        Verify that the CharacterClient instance reuses the same session object across multiple API calls.
        """
        client = CharacterClient()
        session1 = client.session
        
        # Make some calls and verify session doesn't change
        with patch.object(client, '_make_request'):
            client.get_character("test")
            session2 = client.session
        
        assert session1 is session2

    # Parametrized Tests for HTTP Status Codes
    @pytest.mark.parametrize("status_code,error_message", [
        (400, "400 Bad Request"),
        (401, "401 Unauthorized"),
        (403, "403 Forbidden"),
        (404, "404 Not Found"),
        (422, "422 Validation Error"),
        (429, "429 Too Many Requests"),
        (500, "500 Internal Server Error"),
        (502, "502 Bad Gateway"),
        (503, "503 Service Unavailable"),
    ])
    @patch.object(CharacterClient, '_make_request')
    def test_various_http_error_responses(self, mock_make_request, client, status_code, error_message):
        """
        Test that the client raises an HTTPError with the expected message for various HTTP error status codes.
        
        Parameters:
            status_code (int): The HTTP status code being simulated.
            error_message (str): The error message to be raised with the HTTPError.
        """
        mock_make_request.side_effect = HTTPError(error_message)
        
        with pytest.raises(HTTPError, match=error_message):
            client.get_character("char_123")

    # Timeout and Connection Tests
    @pytest.mark.parametrize("timeout_value", [1, 5, 30, 60, 120])
    def test_various_timeout_values(self, timeout_value):
        """
        Test that the CharacterClient correctly sets the timeout value during initialization for various input values.
        """
        client = CharacterClient(timeout=timeout_value)
        assert client.timeout == timeout_value

    # Logging Tests
    @patch('character_client.logger')
    @patch.object(CharacterClient, '_make_request')
    def test_logging_on_request_error(self, mock_make_request, mock_logger, client):
        """
        Verify that the client logs an error message when a connection error occurs during an HTTP request.
        """
        # Override the actual _make_request to test logging in the real method
        with patch('requests.Session.request') as mock_request:
            mock_request.side_effect = ConnectionError("Connection failed")
            
            with pytest.raises(ConnectionError):
                client._make_request("GET", "/test")
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args
            assert "Request failed" in args[0]

    # Complex Data Handling Tests
    @patch.object(CharacterClient, '_make_request')
    def test_complex_character_data(self, mock_make_request, client):
        """
        Verify that the client correctly retrieves and processes character data with deeply nested and complex structures.
        """
        complex_character = {
            "id": "complex_char",
            "name": "Complex Character",
            "attributes": {
                "primary": {"strength": 15, "dexterity": 12},
                "secondary": {"health": 100, "mana": 50}
            },
            "inventory": {
                "weapons": [
                    {"name": "sword", "damage": 10, "durability": 100},
                    {"name": "bow", "damage": 8, "durability": 80}
                ],
                "armor": {"head": None, "chest": {"name": "leather", "defense": 5}}
            },
            "quests": ["quest_1", "quest_2", "quest_3"],
            "metadata": {
                "created": "2024-01-01",
                "tags": ["hero", "adventurer"],
                "settings": {"auto_save": True, "difficulty": "normal"}
            }
        }
        
        mock_make_request.return_value = complex_character
        
        result = client.get_character("complex_char")
        
        assert result == complex_character
        assert result["inventory"]["weapons"][0]["damage"] == 10
        assert result["metadata"]["settings"]["auto_save"] is True


class TestCharacterClientIntegration:
    """Integration-style tests for CharacterClient."""
    
    @pytest.fixture
    def client(self):
        """
        Provides a CharacterClient instance configured for integration testing with a test base URL and API key.
        """
        return CharacterClient(
            base_url="https://integration-test.characters.com",
            api_key="integration-test-key"
        )
    
    @patch.object(CharacterClient, '_make_request')
    def test_character_and_stats_workflow(self, mock_make_request, client):
        """
        Test a complete workflow involving character creation, retrieval, stats retrieval, and stats update.
        
        Simulates the sequence of creating a character, retrieving its details, fetching initial stats, updating stats, and verifying the results and call count using mocked API responses.
        """
        character_data = {"id": "char_123", "name": "Test Character", "level": 1}
        initial_stats = {"experience_points": 0, "battles_won": 0}
        updated_stats = {"experience_points": 100, "battles_won": 1}
        
        mock_make_request.side_effect = [
            character_data,  # create character
            character_data,  # get character
            initial_stats,   # get initial stats
            updated_stats    # update stats
        ]
        
        # Create character
        created = client.create_character({"name": "Test Character", "level": 1})
        
        # Get character
        retrieved = client.get_character(created["id"])
        
        # Get initial stats
        stats = client.get_character_stats(created["id"])
        
        # Update stats after a battle
        new_stats = client.update_character_stats(created["id"], updated_stats)
        
        # Verify workflow
        assert created["name"] == "Test Character"
        assert retrieved["id"] == created["id"]
        assert stats["battles_won"] == 0
        assert new_stats["battles_won"] == 1
        assert mock_make_request.call_count == 4
    
    @patch.object(CharacterClient, '_make_request')
    def test_search_and_retrieve_workflow(self, mock_make_request, client):
        """
        Test the workflow of searching for characters by class and retrieving detailed information for each result.
        
        Simulates a search operation followed by fetching details for each character found, verifying that the correct data is returned and the expected number of API calls are made.
        """
        search_results = {
            "results": [
                {"id": "char_1", "name": "Warrior One"},
                {"id": "char_2", "name": "Warrior Two"}
            ],
            "total": 2
        }
        
        character_details_1 = {"id": "char_1", "name": "Warrior One", "level": 10, "class": "Warrior"}
        character_details_2 = {"id": "char_2", "name": "Warrior Two", "level": 12, "class": "Warrior"}
        
        mock_make_request.side_effect = [
            search_results,      # search
            character_details_1, # get char_1 details
            character_details_2  # get char_2 details
        ]
        
        # Search for warriors
        results = client.search_characters("Warrior", search_type="class")
        
        # Get details for each found character
        detailed_characters = []
        for char in results["results"]:
            details = client.get_character(char["id"])
            detailed_characters.append(details)
        
        # Verify workflow
        assert len(detailed_characters) == 2
        assert detailed_characters[0]["level"] == 10
        assert detailed_characters[1]["level"] == 12
        assert mock_make_request.call_count == 3


# Performance and Load Testing Utilities
class TestCharacterClientPerformance:
    """Performance-related tests for CharacterClient."""
    
    @pytest.fixture
    def client(self):
        """
        Returns a new instance of CharacterClient with default configuration.
        """
        return CharacterClient()
    
    @patch.object(CharacterClient, '_make_request')
    def test_bulk_operations_performance(self, mock_make_request, client):
        """
        Simulates bulk creation of characters to assess client performance and verifies that all operations are executed as expected.
        """
        # Simulate creating multiple characters
        mock_make_request.return_value = {"id": "char_bulk", "name": "Bulk Character"}
        
        characters_to_create = 100
        created_characters = []
        
        for i in range(characters_to_create):
            character_data = {"name": f"Character {i}", "level": i + 1}
            created = client.create_character(character_data)
            created_characters.append(created)
        
        # Verify all characters were "created"
        assert len(created_characters) == characters_to_create
        assert mock_make_request.call_count == characters_to_create
    
    @patch.object(CharacterClient, '_make_request')
    def test_pagination_performance(self, mock_make_request, client):
        """
        Tests that the client correctly paginates through a large list of characters, retrieving all items across multiple pages.
        
        Simulates paginated API responses and verifies that the client issues the expected number of requests and aggregates the correct total number of characters.
        """
        page_size = 50
        total_characters = 1000
        
        def mock_list_response(call_count=[0]):
            """
            Simulate a paginated API response for listing characters, incrementing call count on each invocation.
            
            Parameters:
                call_count (list, optional): Mutable counter tracking the number of times the function is called.
            
            Returns:
                dict: A simulated API response containing a list of character IDs, total count, current page limit, and offset.
            """
            call_count[0] += 1
            offset = (call_count[0] - 1) * page_size
            remaining = max(0, total_characters - offset)
            return {
                "characters": [{"id": f"char_{i}"} for i in range(min(page_size, remaining))],
                "total": total_characters,
                "limit": page_size,
                "offset": offset
            }
        
        mock_make_request.side_effect = lambda *args, **kwargs: mock_list_response()
        
        # Fetch all characters using pagination
        all_characters = []
        offset = 0
        
        while True:
            page = client.list_characters(limit=page_size, offset=offset)
            all_characters.extend(page["characters"])
            
            if len(page["characters"]) < page_size:
                break
            offset += page_size
        
        # Verify pagination worked correctly
        expected_calls = (total_characters + page_size - 1) // page_size  # Ceiling division
        assert mock_make_request.call_count == expected_calls
        assert len(all_characters) == total_characters


if __name__ == "__main__":
    # Allow running tests directly with various options
    pytest.main([__file__, "-v", "--tb=short"])