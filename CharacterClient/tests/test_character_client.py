import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
import json
from typing import Dict, List

# Import the CharacterClient and exceptions
from CharacterClient.character_client import CharacterClient
from CharacterClient.exceptions import (
    CharacterClientError, 
    CharacterNotFoundError, 
    ValidationError,
    AuthenticationError,
    RateLimitError
)


class TestCharacterClient(unittest.TestCase):
    """Comprehensive unit tests for CharacterClient class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.base_url = "https://api.test.com"
        self.api_key = "test_api_key_123"
        self.client = CharacterClient(base_url=self.base_url, api_key=self.api_key)
        
        # Sample character data for testing
        self.sample_character = {
            "id": 1,
            "name": "Test Hero",
            "class": "Warrior",
            "level": 10,
            "health": 100,
            "mana": 50,
            "attributes": {
                "strength": 15,
                "agility": 12,
                "intelligence": 8
            },
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        self.sample_characters_list = [
            self.sample_character,
            {
                "id": 2,
                "name": "Test Mage",
                "class": "Mage",
                "level": 8,
                "health": 80,
                "mana": 120,
                "attributes": {
                    "strength": 8,
                    "agility": 10,
                    "intelligence": 18
                }
            }
        ]
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    # Constructor Tests
    def test_init_with_valid_parameters(self):
        """Test CharacterClient initialization with valid parameters."""
        client = CharacterClient(base_url=self.base_url, api_key=self.api_key)
        self.assertEqual(client.base_url, self.base_url)
        self.assertEqual(client.api_key, self.api_key)
        self.assertEqual(client.timeout, 30)  # default timeout
        self.assertIsInstance(client.session, requests.Session)
    
    def test_init_with_custom_timeout(self):
        """Test CharacterClient initialization with custom timeout."""
        timeout = 60
        client = CharacterClient(base_url=self.base_url, api_key=self.api_key, timeout=timeout)
        self.assertEqual(client.timeout, timeout)
    
    def test_init_with_default_base_url(self):
        """Test CharacterClient initialization with default base URL."""
        client = CharacterClient(api_key=self.api_key)
        self.assertEqual(client.base_url, "https://api.characters.com")
    
    def test_init_with_none_base_url(self):
        """Test CharacterClient initialization with None base_url."""
        with self.assertRaises(ValueError) as context:
            CharacterClient(base_url=None, api_key=self.api_key)
        self.assertIn("base_url cannot be empty", str(context.exception))
    
    def test_init_with_empty_base_url(self):
        """Test CharacterClient initialization with empty base_url."""
        with self.assertRaises(ValueError) as context:
            CharacterClient(base_url="", api_key=self.api_key)
        self.assertIn("base_url cannot be empty", str(context.exception))
    
    def test_init_with_none_api_key(self):
        """Test CharacterClient initialization with None API key."""
        with self.assertRaises(ValueError) as context:
            CharacterClient(base_url=self.base_url, api_key=None)
        self.assertIn("API key is required", str(context.exception))
    
    def test_init_with_empty_api_key(self):
        """Test CharacterClient initialization with empty API key."""
        with self.assertRaises(ValueError) as context:
            CharacterClient(base_url=self.base_url, api_key="")
        self.assertIn("API key is required", str(context.exception))
    
    @patch('CharacterClient.utils.validate_url')
    def test_init_with_invalid_url_format(self, mock_validate_url):
        """Test CharacterClient initialization with invalid URL format."""
        mock_validate_url.return_value = False
        
        with self.assertRaises(ValueError) as context:
            CharacterClient(base_url="not-a-valid-url", api_key=self.api_key)
        self.assertIn("Invalid URL format", str(context.exception))
    
    def test_init_session_headers(self):
        """Test that session headers are set correctly."""
        client = CharacterClient(base_url=self.base_url, api_key=self.api_key)
        expected_headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'CharacterClient/1.0'
        }
        
        for key, value in expected_headers.items():
            self.assertEqual(client.session.headers[key], value)
    
    def test_init_base_url_trailing_slash_removal(self):
        """Test that trailing slashes are removed from base_url."""
        client = CharacterClient(base_url="https://api.test.com/", api_key=self.api_key)
        self.assertEqual(client.base_url, "https://api.test.com")
    
    # Character Retrieval Tests
    @patch('requests.Session.get')
    def test_get_character_success(self, mock_get):
        """Test successful character retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_character
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        
        self.assertEqual(result, self.sample_character)
        mock_get.assert_called_once_with(
            f"{self.base_url}/characters/1", 
            timeout=30
        )
    
    @patch('requests.Session.get')
    def test_get_character_not_found(self, mock_get):
        """Test character retrieval when character doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterNotFoundError) as context:
            self.client.get_character(999)
        self.assertIn("Character with ID 999 not found", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_authentication_error(self, mock_get):
        """Test character retrieval with authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Authentication failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_rate_limit_error(self, mock_get):
        """Test character retrieval with rate limit error."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_server_error(self, mock_get):
        """Test character retrieval with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("API error: 500", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_network_error(self, mock_get):
        """Test character retrieval with network error."""
        mock_get.side_effect = requests.ConnectionError("Network error")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_timeout_error(self, mock_get):
        """Test character retrieval with timeout error."""
        mock_get.side_effect = requests.Timeout("Request timeout")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    def test_get_character_invalid_id_type(self):
        """Test character retrieval with invalid ID type."""
        with self.assertRaises(TypeError) as context:
            self.client.get_character("invalid_id")
        self.assertIn("character_id must be an integer", str(context.exception))
    
    def test_get_character_negative_id(self):
        """Test character retrieval with negative ID."""
        with self.assertRaises(ValueError) as context:
            self.client.get_character(-1)
        self.assertIn("character_id must be positive", str(context.exception))
    
    def test_get_character_zero_id(self):
        """Test character retrieval with zero ID."""
        with self.assertRaises(ValueError) as context:
            self.client.get_character(0)
        self.assertIn("character_id must be positive", str(context.exception))
    
    # Character List Tests
    @patch('requests.Session.get')
    def test_get_characters_success(self, mock_get):
        """Test successful characters list retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"characters": self.sample_characters_list}
        mock_get.return_value = mock_response
        
        result = self.client.get_characters()
        
        self.assertEqual(result, self.sample_characters_list)
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_get_characters_with_pagination(self, mock_get):
        """Test characters list retrieval with pagination."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "characters": self.sample_characters_list[:1],
            "page": 2,
            "total_pages": 5
        }
        mock_get.return_value = mock_response
        
        result = self.client.get_characters(page=2, limit=1)
        
        self.assertEqual(len(result), 1)
        mock_get.assert_called_once()
        # Verify correct parameters were passed
        call_args = mock_get.call_args
        self.assertEqual(call_args[1]['params']['page'], 2)
        self.assertEqual(call_args[1]['params']['limit'], 1)
    
    @patch('requests.Session.get')
    def test_get_characters_with_class_filter(self, mock_get):
        """Test characters list retrieval with class filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        filtered_characters = [char for char in self.sample_characters_list if char["class"] == "Warrior"]
        mock_response.json.return_value = {"characters": filtered_characters}
        mock_get.return_value = mock_response
        
        result = self.client.get_characters(character_class="Warrior")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "Warrior")
        # Verify filter parameter was passed
        call_args = mock_get.call_args
        self.assertEqual(call_args[1]['params']['class'], "Warrior")
    
    @patch('requests.Session.get')
    def test_get_characters_empty_result(self, mock_get):
        """Test characters list retrieval with empty result."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"characters": []}
        mock_get.return_value = mock_response
        
        result = self.client.get_characters()
        
        self.assertEqual(result, [])
    
    @patch('requests.Session.get')
    def test_get_characters_missing_characters_key(self, mock_get):
        """Test characters list retrieval when response doesn't have 'characters' key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": self.sample_characters_list}
        mock_get.return_value = mock_response
        
        result = self.client.get_characters()
        
        self.assertEqual(result, [])  # Should return empty list when 'characters' key is missing
    
    def test_get_characters_invalid_page(self):
        """Test get_characters with invalid page number."""
        with self.assertRaises(ValueError) as context:
            self.client.get_characters(page=0)
        self.assertIn("page must be >= 1", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.client.get_characters(page=-1)
        self.assertIn("page must be >= 1", str(context.exception))
    
    def test_get_characters_invalid_limit(self):
        """Test get_characters with invalid limit."""
        with self.assertRaises(ValueError) as context:
            self.client.get_characters(limit=0)
        self.assertIn("limit must be between 1 and 100", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.client.get_characters(limit=101)
        self.assertIn("limit must be between 1 and 100", str(context.exception))
    
    # Character Creation Tests
    @patch('requests.Session.post')
    def test_create_character_success(self, mock_post):
        """Test successful character creation."""
        new_character_data = {
            "name": "New Character",
            "class": "Rogue",
            "level": 1
        }
        
        mock_response = Mock()
        mock_response.status_code = 201
        created_character = {**new_character_data, "id": 3}
        mock_response.json.return_value = created_character
        mock_post.return_value = mock_response
        
        result = self.client.create_character(new_character_data)
        
        self.assertEqual(result["name"], "New Character")
        self.assertEqual(result["id"], 3)
        mock_post.assert_called_once_with(
            f"{self.base_url}/characters",
            json=new_character_data,
            timeout=30
        )
    
    @patch('requests.Session.post')
    def test_create_character_validation_error(self, mock_post):
        """Test character creation with validation error."""
        invalid_data = {"name": "Test", "class": "Warrior", "level": 1}
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        with self.assertRaises(ValidationError) as context:
            self.client.create_character(invalid_data)
        self.assertIn("Invalid character data", str(context.exception))
    
    @patch('requests.Session.post')
    def test_create_character_server_error(self, mock_post):
        """Test character creation with server error."""
        character_data = {"name": "Test", "class": "Warrior", "level": 1}
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.create_character(character_data)
        self.assertIn("API error: 500", str(context.exception))
    
    def test_create_character_invalid_data_type(self):
        """Test character creation with invalid data type."""
        with self.assertRaises(TypeError) as context:
            self.client.create_character("not_a_dict")
        self.assertIn("character_data must be a dictionary", str(context.exception))
    
    def test_create_character_missing_required_fields(self):
        """Test character creation with missing required fields."""
        incomplete_data = {"level": 1}  # Missing name and class
        
        with self.assertRaises(ValueError) as context:
            self.client.create_character(incomplete_data)
        self.assertIn("Missing required field: name", str(context.exception))
    
    def test_create_character_empty_name(self):
        """Test character creation with empty name."""
        invalid_data = {"name": "", "class": "Warrior", "level": 1}
        
        with self.assertRaises(ValueError) as context:
            self.client.create_character(invalid_data)
        self.assertIn("Character name cannot be empty", str(context.exception))
    
    def test_create_character_whitespace_name(self):
        """Test character creation with whitespace-only name."""
        invalid_data = {"name": "   ", "class": "Warrior", "level": 1}
        
        with self.assertRaises(ValueError) as context:
            self.client.create_character(invalid_data)
        self.assertIn("Character name cannot be empty", str(context.exception))
    
    def test_create_character_empty_class(self):
        """Test character creation with empty class."""
        invalid_data = {"name": "Test", "class": "", "level": 1}
        
        with self.assertRaises(ValueError) as context:
            self.client.create_character(invalid_data)
        self.assertIn("Character class cannot be empty", str(context.exception))
    
    def test_create_character_invalid_level(self):
        """Test character creation with invalid level."""
        invalid_data = {"name": "Test", "class": "Warrior", "level": 0}
        
        with self.assertRaises(ValueError) as context:
            self.client.create_character(invalid_data)
        self.assertIn("Character level must be a positive integer", str(context.exception))
        
        invalid_data = {"name": "Test", "class": "Warrior", "level": "not_a_number"}
        
        with self.assertRaises(ValueError) as context:
            self.client.create_character(invalid_data)
        self.assertIn("Character level must be a positive integer", str(context.exception))
    
    # Character Update Tests
    @patch('requests.Session.put')
    def test_update_character_success(self, mock_put):
        """Test successful character update."""
        update_data = {"level": 11, "health": 110}
        
        mock_response = Mock()
        mock_response.status_code = 200
        updated_character = {**self.sample_character, **update_data}
        mock_response.json.return_value = updated_character
        mock_put.return_value = mock_response
        
        result = self.client.update_character(1, update_data)
        
        self.assertEqual(result["level"], 11)
        self.assertEqual(result["health"], 110)
        mock_put.assert_called_once_with(
            f"{self.base_url}/characters/1",
            json=update_data,
            timeout=30
        )
    
    @patch('requests.Session.put')
    def test_update_character_not_found(self, mock_put):
        """Test character update when character doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_put.return_value = mock_response
        
        with self.assertRaises(CharacterNotFoundError) as context:
            self.client.update_character(999, {"level": 5})
        self.assertIn("Character with ID 999 not found", str(context.exception))
    
    @patch('requests.Session.put')
    def test_update_character_server_error(self, mock_put):
        """Test character update with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_put.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.update_character(1, {"level": 5})
        self.assertIn("API error: 500", str(context.exception))
    
    def test_update_character_invalid_id(self):
        """Test character update with invalid ID."""
        with self.assertRaises(ValueError) as context:
            self.client.update_character(-1, {"level": 5})
        self.assertIn("character_id must be a positive integer", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.client.update_character("invalid", {"level": 5})
        self.assertIn("character_id must be a positive integer", str(context.exception))
    
    def test_update_character_empty_data(self):
        """Test character update with empty data."""
        with self.assertRaises(ValueError) as context:
            self.client.update_character(1, {})
        self.assertIn("update_data must be a non-empty dictionary", str(context.exception))
    
    def test_update_character_invalid_data_type(self):
        """Test character update with invalid data type."""
        with self.assertRaises(ValueError) as context:
            self.client.update_character(1, "not_a_dict")
        self.assertIn("update_data must be a non-empty dictionary", str(context.exception))
    
    # Character Deletion Tests
    @patch('requests.Session.delete')
    def test_delete_character_success(self, mock_delete):
        """Test successful character deletion."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_delete.return_value = mock_response
        
        result = self.client.delete_character(1)
        
        self.assertTrue(result)
        mock_delete.assert_called_once_with(
            f"{self.base_url}/characters/1",
            timeout=30
        )
    
    @patch('requests.Session.delete')
    def test_delete_character_not_found(self, mock_delete):
        """Test character deletion when character doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_delete.return_value = mock_response
        
        with self.assertRaises(CharacterNotFoundError) as context:
            self.client.delete_character(999)
        self.assertIn("Character with ID 999 not found", str(context.exception))
    
    @patch('requests.Session.delete')
    def test_delete_character_server_error(self, mock_delete):
        """Test character deletion with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_delete.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.delete_character(1)
        self.assertIn("API error: 500", str(context.exception))
    
    def test_delete_character_invalid_id(self):
        """Test character deletion with invalid ID."""
        with self.assertRaises(ValueError) as context:
            self.client.delete_character(0)
        self.assertIn("character_id must be a positive integer", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.client.delete_character("invalid")
        self.assertIn("character_id must be a positive integer", str(context.exception))
    
    # Batch Operations Tests
    @patch.object(CharacterClient, 'get_character')
    def test_get_characters_batch_success(self, mock_get_character):
        """Test successful batch character retrieval."""
        character_ids = [1, 2, 3]
        mock_get_character.side_effect = [
            {"id": 1, "name": "Character 1"},
            {"id": 2, "name": "Character 2"},
            {"id": 3, "name": "Character 3"}
        ]
        
        results = self.client.get_characters_batch(character_ids)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_get_character.call_count, 3)
        for i, result in enumerate(results):
            self.assertEqual(result["id"], i + 1)
    
    @patch.object(CharacterClient, 'get_character')
    def test_get_characters_batch_with_not_found(self, mock_get_character):
        """Test batch character retrieval with some characters not found."""
        character_ids = [1, 999, 3]
        
        def side_effect(char_id):
            if char_id == 999:
                raise CharacterNotFoundError("Character not found")
            return {"id": char_id, "name": f"Character {char_id}"}
        
        mock_get_character.side_effect = side_effect
        
        results = self.client.get_characters_batch(character_ids)
        
        self.assertEqual(len(results), 2)  # Should skip the not found character
        self.assertEqual(mock_get_character.call_count, 3)
    
    def test_get_characters_batch_invalid_input(self):
        """Test batch character retrieval with invalid input."""
        with self.assertRaises(TypeError) as context:
            self.client.get_characters_batch("not_a_list")
        self.assertIn("character_ids must be a list", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.client.get_characters_batch([])
        self.assertIn("character_ids cannot be empty", str(context.exception))
    
    # Utility Method Tests
    def test_build_url_basic(self):
        """Test URL building utility method."""
        expected_url = f"{self.base_url}/characters"
        result = self.client._build_url("characters")
        self.assertEqual(result, expected_url)
    
    def test_build_url_with_resource_id(self):
        """Test URL building with resource ID."""
        expected_url = f"{self.base_url}/characters/1"
        result = self.client._build_url("characters", resource_id=1)
        self.assertEqual(result, expected_url)
    
    def test_build_url_with_query_params(self):
        """Test URL building with query parameters."""
        params = {"page": 1, "limit": 10}
        result = self.client._build_url("characters", params=params)
        
        # Check that URL contains base path and parameters
        self.assertIn(f"{self.base_url}/characters", result)
        self.assertIn("page=1", result)
        self.assertIn("limit=10", result)
    
    def test_build_url_with_resource_id_and_params(self):
        """Test URL building with both resource ID and parameters."""
        params = {"include": "stats"}
        result = self.client._build_url("characters", resource_id=1, params=params)
        
        self.assertIn(f"{self.base_url}/characters/1", result)
        self.assertIn("include=stats", result)
    
    def test_validate_character_data_valid(self):
        """Test character data validation with valid data."""
        valid_data = {"name": "Test Character", "class": "Warrior", "level": 1}
        result = self.client._validate_character_data(valid_data)
        self.assertTrue(result)
    
    def test_validate_character_data_missing_fields(self):
        """Test character data validation with missing required fields."""
        test_cases = [
            ({"class": "Warrior", "level": 1}, "name"),
            ({"name": "Test", "level": 1}, "class"),
            ({"name": "Test", "class": "Warrior"}, "level")
        ]
        
        for invalid_data, missing_field in test_cases:
            with self.subTest(missing_field=missing_field):
                with self.assertRaises(ValueError) as context:
                    self.client._validate_character_data(invalid_data)
                self.assertIn(f"Missing required field: {missing_field}", str(context.exception))


class TestCharacterClientEdgeCases(unittest.TestCase):
    """Test edge cases and unusual scenarios for CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def test_extremely_long_character_name(self):
        """Test character creation with extremely long name."""
        long_name = "A" * 1000  # Very long name
        character_data = {
            "name": long_name,
            "class": "Warrior",
            "level": 1
        }
        
        # Should pass validation (length limits would be enforced by API)
        result = self.client._validate_character_data(character_data)
        self.assertTrue(result)
    
    def test_unicode_character_names(self):
        """Test character names with Unicode characters."""
        unicode_names = ["测试角色", "キャラクター", "Персонаж", "🧙‍♂️ Wizard"]
        
        for name in unicode_names:
            with self.subTest(name=name):
                character_data = {"name": name, "class": "Mage", "level": 1}
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_special_characters_in_name(self):
        """Test character names with special characters."""
        special_names = [
            "Character@123",
            "Hero-Prime",
            "Super_Warrior",
            "Knight.the.Great",
            "Mage (Apprentice)",
            "Rogue [Level 10]"
        ]
        
        for name in special_names:
            with self.subTest(name=name):
                character_data = {"name": name, "class": "Warrior", "level": 1}
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_boundary_level_values(self):
        """Test character levels at boundary values."""
        valid_levels = [1, 100, 999, 1000]
        invalid_levels = [0, -1, -100]
        
        for level in valid_levels:
            with self.subTest(level=level):
                character_data = {"name": "Test", "class": "Warrior", "level": level}
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
        
        for level in invalid_levels:
            with self.subTest(level=level):
                character_data = {"name": "Test", "class": "Warrior", "level": level}
                with self.assertRaises(ValueError):
                    self.client._validate_character_data(character_data)
    
    @patch('requests.Session.get')
    def test_malformed_json_response(self, mock_get):
        """Test handling of malformed JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
    
    @patch('requests.Session.get')
    def test_response_with_unexpected_structure(self, mock_get):
        """Test handling of responses with unexpected structure."""
        unexpected_responses = [
            [],  # Array instead of object
            "string_response",  # String instead of object
            None,  # Null response
            {"unexpected": "structure"}  # Object with unexpected keys
        ]
        
        for response_data in unexpected_responses:
            with self.subTest(response=response_data):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = response_data
                mock_get.return_value = mock_response
                
                # Should not raise an exception, just return the response
                result = self.client.get_character(1)
                self.assertEqual(result, response_data)
    
    def test_large_character_ids(self):
        """Test with very large character IDs."""
        large_ids = [999999999, 2**31 - 1, 2**63 - 1]
        
        for char_id in large_ids:
            with self.subTest(char_id=char_id):
                with patch('requests.Session.get') as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"id": char_id, "name": "Test"}
                    mock_get.return_value = mock_response
                    
                    result = self.client.get_character(char_id)
                    self.assertEqual(result["id"], char_id)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=True)

class TestCharacterClientAdvancedScenarios(unittest.TestCase):
    """Additional comprehensive tests for CharacterClient covering advanced scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
        self.sample_character = {
            "id": 1,
            "name": "Test Hero",
            "class": "Warrior",
            "level": 10,
            "health": 100,
            "mana": 50
        }
    
    # Session Management Tests
    def test_session_close_on_context_manager_exit(self):
        """Test that session is properly closed when using context manager."""
        with CharacterClient(base_url="https://test.com", api_key="test") as client:
            self.assertIsNotNone(client.session)
        # Session should be closed after context exit
        self.assertTrue(client.session.close.called if hasattr(client.session, 'close') else True)
    
    def test_session_reuse_across_requests(self):
        """Test that the same session is reused across multiple requests."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = self.sample_character
            mock_get.return_value = mock_response
            
            # Make multiple requests
            self.client.get_character(1)
            self.client.get_character(2)
            
            # Verify same session instance was used
            self.assertEqual(mock_get.call_count, 2)
    
    # Custom Headers and Authentication Tests
    def test_custom_headers_override(self):
        """Test that custom headers can be added and override defaults."""
        client = CharacterClient(
            base_url="https://test.com", 
            api_key="test_key",
            custom_headers={"User-Agent": "CustomAgent/2.0", "X-Custom": "value"}
        )
        
        self.assertEqual(client.session.headers.get("User-Agent"), "CustomAgent/2.0")
        self.assertEqual(client.session.headers.get("X-Custom"), "value")
        self.assertEqual(client.session.headers.get("Authorization"), "Bearer test_key")
    
    def test_api_key_with_different_auth_schemes(self):
        """Test different authentication schemes."""
        auth_schemes = [
            ("Bearer", "Bearer test_key"),
            ("Token", "Token test_key"),
            ("API-Key", "API-Key test_key")
        ]
        
        for scheme, expected_header in auth_schemes:
            with self.subTest(scheme=scheme):
                client = CharacterClient(
                    base_url="https://test.com", 
                    api_key="test_key",
                    auth_scheme=scheme
                )
                self.assertEqual(client.session.headers.get("Authorization"), expected_header)
    
    # Retry and Circuit Breaker Tests
    @patch('requests.Session.get')
    def test_retry_on_network_error(self, mock_get):
        """Test automatic retry on network errors."""
        # First call fails, second succeeds
        mock_get.side_effect = [
            requests.ConnectionError("Network error"),
            Mock(status_code=200, json=lambda: self.sample_character)
        ]
        
        client = CharacterClient(
            base_url="https://test.com", 
            api_key="test_key",
            max_retries=2
        )
        
        result = client.get_character(1)
        self.assertEqual(result, self.sample_character)
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('requests.Session.get')
    def test_retry_exhaustion(self, mock_get):
        """Test behavior when all retries are exhausted."""
        mock_get.side_effect = requests.ConnectionError("Persistent network error")
        
        client = CharacterClient(
            base_url="https://test.com", 
            api_key="test_key",
            max_retries=2
        )
        
        with self.assertRaises(CharacterClientError):
            client.get_character(1)
        self.assertEqual(mock_get.call_count, 3)  # Initial + 2 retries
    
    @patch('requests.Session.get')
    def test_exponential_backoff(self, mock_get):
        """Test exponential backoff between retries."""
        mock_get.side_effect = [
            requests.ConnectionError("Network error"),
            requests.ConnectionError("Network error"),
            Mock(status_code=200, json=lambda: self.sample_character)
        ]
        
        with patch('time.sleep') as mock_sleep:
            client = CharacterClient(
                base_url="https://test.com", 
                api_key="test_key",
                max_retries=3,
                backoff_factor=0.1
            )
            
            result = client.get_character(1)
            self.assertEqual(result, self.sample_character)
            
            # Verify exponential backoff was used
            sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
            self.assertTrue(len(sleep_calls) > 0)
            if len(sleep_calls) > 1:
                self.assertTrue(sleep_calls[1] > sleep_calls[0])  # Exponential increase
    
    # Rate Limiting and Throttling Tests
    @patch('requests.Session.get')
    def test_rate_limit_with_retry_after_header(self, mock_get):
        """Test handling of rate limit with Retry-After header."""
        # First response has rate limit with Retry-After header
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "2"}
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = self.sample_character
        
        mock_get.side_effect = [rate_limit_response, success_response]
        
        with patch('time.sleep') as mock_sleep:
            result = self.client.get_character(1)
            self.assertEqual(result, self.sample_character)
            mock_sleep.assert_called_with(2)
    
    @patch('requests.Session.get')
    def test_rate_limit_without_retry_after(self, mock_get):
        """Test handling of rate limit without Retry-After header."""
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}
        
        mock_get.return_value = rate_limit_response
        
        with self.assertRaises(RateLimitError):
            self.client.get_character(1)
    
    # Concurrent Request Tests
    @patch('requests.Session.get')
    def test_concurrent_requests_thread_safety(self, mock_get):
        """Test thread safety of concurrent requests."""
        import threading
        import time
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_character
        mock_get.return_value = mock_response
        
        results = []
        errors = []
        
        def make_request(char_id):
            try:
                result = self.client.get_character(char_id)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads making concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i + 1,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        self.assertEqual(mock_get.call_count, 5)
    
    # Caching Tests
    @patch('requests.Session.get')
    def test_response_caching(self, mock_get):
        """Test response caching functionality if implemented."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_character
        mock_get.return_value = mock_response
        
        client = CharacterClient(
            base_url="https://test.com", 
            api_key="test_key",
            enable_caching=True,
            cache_ttl=60
        )
        
        # Make the same request twice
        result1 = client.get_character(1)
        result2 = client.get_character(1)
        
        self.assertEqual(result1, result2)
        # If caching is implemented, second request should not hit the API
        expected_calls = 1 if hasattr(client, '_cache') else 2
        self.assertEqual(mock_get.call_count, expected_calls)
    
    # Data Validation and Sanitization Tests
    def test_sanitize_character_data(self):
        """Test data sanitization before sending to API."""
        dirty_data = {
            "name": "  Test Character  ",  # Extra whitespace
            "class": "WARRIOR",  # Wrong case
            "level": "10",  # String instead of int
            "description": "<script>alert('xss')</script>Normal description"  # Potential XSS
        }
        
        sanitized = self.client._sanitize_character_data(dirty_data)
        
        self.assertEqual(sanitized["name"], "Test Character")
        self.assertEqual(sanitized["class"], "Warrior")
        self.assertEqual(sanitized["level"], 10)
        self.assertNotIn("<script>", sanitized.get("description", ""))
    
    def test_validate_character_attributes(self):
        """Test validation of character attributes."""
        invalid_attributes = [
            {"strength": -5},  # Negative value
            {"agility": "fast"},  # Non-numeric value
            {"intelligence": 101},  # Value over maximum
            {"unknown_stat": 15}  # Unknown attribute
        ]
        
        for invalid_attr in invalid_attributes:
            with self.subTest(attributes=invalid_attr):
                character_data = {
                    "name": "Test",
                    "class": "Warrior",
                    "level": 1,
                    "attributes": invalid_attr
                }
                with self.assertRaises(ValidationError):
                    self.client._validate_character_data(character_data)
    
    # Pagination Edge Cases
    @patch('requests.Session.get')
    def test_get_all_characters_pagination(self, mock_get):
        """Test getting all characters across multiple pages."""
        # Mock responses for multiple pages
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "characters": [{"id": 1, "name": "Char1"}],
            "page": 1,
            "total_pages": 3,
            "has_next": True
        }
        
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "characters": [{"id": 2, "name": "Char2"}],
            "page": 2,
            "total_pages": 3,
            "has_next": True
        }
        
        page3_response = Mock()
        page3_response.status_code = 200
        page3_response.json.return_value = {
            "characters": [{"id": 3, "name": "Char3"}],
            "page": 3,
            "total_pages": 3,
            "has_next": False
        }
        
        mock_get.side_effect = [page1_response, page2_response, page3_response]
        
        all_characters = self.client.get_all_characters()
        
        self.assertEqual(len(all_characters), 3)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(all_characters[0]["id"], 1)
        self.assertEqual(all_characters[2]["id"], 3)
    
    # Search and Filtering Tests
    @patch('requests.Session.get')
    def test_search_characters_by_name(self, mock_get):
        """Test searching characters by name."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "characters": [{"id": 1, "name": "Test Hero"}]
        }
        mock_get.return_value = mock_response
        
        results = self.client.search_characters(name="Test Hero")
        
        self.assertEqual(len(results), 1)
        call_args = mock_get.call_args
        self.assertIn("name", call_args[1].get("params", {}))
    
    @patch('requests.Session.get')
    def test_search_characters_multiple_filters(self, mock_get):
        """Test searching characters with multiple filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"characters": []}
        mock_get.return_value = mock_response
        
        filters = {
            "class": "Warrior",
            "min_level": 10,
            "max_level": 20,
            "has_guild": True
        }
        
        results = self.client.search_characters(**filters)
        
        call_args = mock_get.call_args
        params = call_args[1].get("params", {})
        
        for key, value in filters.items():
            self.assertEqual(params.get(key), value)
    
    # Webhook and Event Handling Tests
    def test_register_webhook(self):
        """Test webhook registration."""
        webhook_url = "https://myapp.com/webhook"
        events = ["character.created", "character.updated", "character.deleted"]
        
        with patch('requests.Session.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"webhook_id": "webhook_123"}
            mock_post.return_value = mock_response
            
            result = self.client.register_webhook(webhook_url, events)
            
            self.assertEqual(result["webhook_id"], "webhook_123")
            mock_post.assert_called_once()
    
    def test_webhook_signature_validation(self):
        """Test webhook signature validation."""
        payload = '{"event": "character.created", "data": {"id": 1}}'
        signature = "sha256=valid_signature"
        secret = "webhook_secret"
        
        # Test valid signature
        is_valid = self.client.validate_webhook_signature(payload, signature, secret)
        self.assertTrue(is_valid)
        
        # Test invalid signature
        invalid_signature = "sha256=invalid_signature"
        is_valid = self.client.validate_webhook_signature(payload, invalid_signature, secret)
        self.assertFalse(is_valid)
    
    # Statistics and Analytics Tests
    @patch('requests.Session.get')
    def test_get_character_statistics(self, mock_get):
        """Test getting character statistics."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_characters": 1000,
            "average_level": 15.5,
            "class_distribution": {
                "Warrior": 300,
                "Mage": 250,
                "Rogue": 200,
                "Priest": 250
            }
        }
        mock_get.return_value = mock_response
        
        stats = self.client.get_statistics()
        
        self.assertEqual(stats["total_characters"], 1000)
        self.assertIn("class_distribution", stats)
        self.assertEqual(len(stats["class_distribution"]), 4)
    
    # Import/Export Tests
    @patch('requests.Session.post')
    def test_bulk_import_characters(self, mock_post):
        """Test bulk import of characters."""
        characters_data = [
            {"name": "Hero1", "class": "Warrior", "level": 5},
            {"name": "Hero2", "class": "Mage", "level": 8}
        ]
        
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "import_id": "import_123",
            "status": "processing",
            "total_count": 2
        }
        mock_post.return_value = mock_response
        
        result = self.client.bulk_import_characters(characters_data)
        
        self.assertEqual(result["import_id"], "import_123")
        self.assertEqual(result["total_count"], 2)
    
    @patch('requests.Session.get')
    def test_export_characters(self, mock_get):
        """Test exporting characters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "export_url": "https://api.com/exports/export_123.json",
            "expires_at": "2024-01-01T12:00:00Z"
        }
        mock_get.return_value = mock_response
        
        export_filters = {"class": "Warrior", "min_level": 10}
        result = self.client.export_characters(export_filters)
        
        self.assertIn("export_url", result)
        self.assertIn("expires_at", result)
    
    # Error Recovery Tests
    @patch('requests.Session.get')
    def test_partial_failure_recovery(self, mock_get):
        """Test recovery from partial failures in batch operations."""
        character_ids = [1, 2, 3, 4, 5]
        
        responses = [
            Mock(status_code=200, json=lambda: {"id": 1, "name": "Char1"}),
            Mock(status_code=404),  # Character not found
            Mock(status_code=200, json=lambda: {"id": 3, "name": "Char3"}),
            Mock(status_code=500),  # Server error
            Mock(status_code=200, json=lambda: {"id": 5, "name": "Char5"})
        ]
        
        mock_get.side_effect = responses
        
        # Should return successful results and handle failures gracefully
        results = self.client.get_characters_batch(character_ids, ignore_errors=True)
        
        # Should have 3 successful results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[1]["id"], 3)
        self.assertEqual(results[2]["id"], 5)
    
    # Configuration and Environment Tests
    def test_environment_based_configuration(self):
        """Test configuration from environment variables."""
        import os
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'CHARACTER_API_BASE_URL': 'https://prod.api.com',
            'CHARACTER_API_KEY': 'prod_key_123',
            'CHARACTER_API_TIMEOUT': '60'
        }):
            client = CharacterClient.from_environment()
            
            self.assertEqual(client.base_url, 'https://prod.api.com')
            self.assertEqual(client.api_key, 'prod_key_123')
            self.assertEqual(client.timeout, 60)
    
    def test_configuration_file_loading(self):
        """Test loading configuration from file."""
        config_data = {
            "base_url": "https://config.api.com",
            "api_key": "config_key_123",
            "timeout": 45,
            "max_retries": 3
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
            client = CharacterClient.from_config_file("config.json")
            
            self.assertEqual(client.base_url, "https://config.api.com")
            self.assertEqual(client.api_key, "config_key_123")
            self.assertEqual(client.timeout, 45)


class TestCharacterClientSecurityAndValidation(unittest.TestCase):
    """Security and data validation tests for CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.com", api_key="test_key")
    
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection in character names."""
        malicious_names = [
            "'; DROP TABLE characters; --",
            "Robert'; DELETE FROM users WHERE 't' = 't",
            "1' OR '1'='1",
            "admin'/*",
            "' UNION SELECT * FROM passwords --"
        ]
        
        for malicious_name in malicious_names:
            with self.subTest(name=malicious_name):
                character_data = {
                    "name": malicious_name,
                    "class": "Warrior",
                    "level": 1
                }
                
                # Should either sanitize or raise validation error
                try:
                    validated_data = self.client._validate_character_data(character_data)
                    # If validation passes, name should be sanitized
                    self.assertNotIn("DROP", validated_data["name"].upper())
                    self.assertNotIn("DELETE", validated_data["name"].upper())
                    self.assertNotIn("UNION", validated_data["name"].upper())
                except ValidationError:
                    # Validation error is acceptable for malicious input
                    pass
    
    def test_xss_prevention_in_character_data(self):
        """Test prevention of XSS attacks in character data."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "&#60;script&#62;alert('xss')&#60;/script&#62;"
        ]
        
        for payload in xss_payloads:
            with self.subTest(payload=payload):
                character_data = {
                    "name": f"Hero {payload}",
                    "class": "Warrior",
                    "level": 1,
                    "description": f"A hero with {payload} abilities"
                }
                
                try:
                    sanitized = self.client._sanitize_character_data(character_data)
                    # Should not contain script tags or javascript: URLs
                    self.assertNotIn("<script>", sanitized.get("name", ""))
                    self.assertNotIn("javascript:", sanitized.get("description", ""))
                    self.assertNotIn("<img", sanitized.get("description", ""))
                except ValidationError:
                    # Validation error is acceptable
                    pass
    
    def test_api_key_not_logged(self):
        """Test that API key is not exposed in logs or error messages."""
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = requests.HTTPError("Server error")
            
            with patch('logging.Logger.error') as mock_log:
                try:
                    self.client.get_character(1)
                except:
                    pass
                
                # Check that API key is not in any log messages
                for call in mock_log.call_args_list:
                    log_message = str(call)
                    self.assertNotIn("test_key", log_message)
    
    def test_request_size_limits(self):
        """Test handling of excessively large requests."""
        # Create character data that exceeds reasonable size limits
        huge_description = "A" * 1000000  # 1MB description
        
        character_data = {
            "name": "Test Character",
            "class": "Warrior",
            "level": 1,
            "description": huge_description
        }
        
        with self.assertRaises(ValidationError):
            self.client.create_character(character_data)
    
    def test_response_size_limits(self):
        """Test handling of excessively large responses."""
        with patch('requests.Session.get') as mock_get:
            # Mock response with huge data
            huge_response_data = {"data": "X" * 10000000}  # 10MB response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = huge_response_data
            mock_get.return_value = mock_response
            
            # Should handle large responses gracefully
            result = self.client.get_character(1)
            self.assertIn("data", result)
    
    def test_url_validation(self):
        """Test validation of URLs to prevent SSRF attacks."""
        malicious_urls = [
            "http://localhost:22/",  # SSH port
            "http://169.254.169.254/",  # AWS metadata
            "file:///etc/passwd",  # Local file
            "ftp://malicious.com/",  # Non-HTTP protocol
            "http://internal.network/",  # Internal network
        ]
        
        for url in malicious_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    CharacterClient(base_url=url, api_key="test")
    
    def test_rate_limiting_per_client(self):
        """Test that rate limiting is enforced per client."""
        with patch('requests.Session.get') as mock_get:
            # Mock rate limit response
            rate_limit_response = Mock()
            rate_limit_response.status_code = 429
            rate_limit_response.headers = {"X-RateLimit-Remaining": "0"}
            mock_get.return_value = rate_limit_response
            
            # Multiple rapid requests should trigger rate limiting
            with self.assertRaises(RateLimitError):
                for _ in range(10):
                    try:
                        self.client.get_character(1)
                    except RateLimitError:
                        raise  # Re-raise to test detection
                    except:
                        pass  # Ignore other errors for this test


class TestCharacterClientPerformance(unittest.TestCase):
    """Performance and stress tests for CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.com", api_key="test_key")
    
    @patch('requests.Session.get')
    def test_large_batch_request_performance(self, mock_get):
        """Test performance with large batch requests."""
        import time
        
        # Simulate processing 1000 character requests
        large_character_ids = list(range(1, 1001))
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_get.return_value = mock_response
        
        start_time = time.time()
        results = self.client.get_characters_batch(large_character_ids[:100])  # Limit for test
        end_time = time.time()
        
        # Basic performance assertion
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        self.assertEqual(len(results), 100)
    
    @patch('requests.Session.get')
    def test_memory_usage_with_large_responses(self, mock_get):
        """Test memory efficiency with large response data."""
        import sys
        
        # Create a large character object
        large_character = {
            "id": 1,
            "name": "Test Character",
            "description": "A" * 10000,  # Large description
            "inventory": [{"item": f"Item{i}"} for i in range(1000)],  # Large inventory
            "stats": {f"stat_{i}": i for i in range(1000)}  # Many stats
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_character
        mock_get.return_value = mock_response
        
        # Measure memory before and after
        initial_refs = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        result = self.client.get_character(1)
        
        final_refs = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # Verify the large character was returned correctly
        self.assertEqual(result["id"], 1)
        self.assertEqual(len(result["inventory"]), 1000)
        
        # Memory growth should be reasonable (this is a basic check)
        if hasattr(sys, 'gettotalrefcount'):
            ref_growth = final_refs - initial_refs
            self.assertLess(ref_growth, 10000)  # Arbitrary reasonable limit
    
    def test_connection_pooling_efficiency(self):
        """Test that connection pooling is working efficiently."""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Create multiple clients
            clients = [
                CharacterClient(base_url="https://test.com", api_key="test1"),
                CharacterClient(base_url="https://test.com", api_key="test2"),
                CharacterClient(base_url="https://test.com", api_key="test3")
            ]
            
            # Each client should have its own session
            self.assertEqual(mock_session_class.call_count, 3)
    
    @patch('requests.Session.get')
    def test_timeout_handling_performance(self, mock_get):
        """Test that timeouts don't significantly impact performance."""
        import time
        
        # Mock timeout after a short delay
        def timeout_side_effect(*args, **kwargs):
            time.sleep(0.1)  # Short delay
            raise requests.Timeout("Request timeout")
        
        mock_get.side_effect = timeout_side_effect
        
        start_time = time.time()
        
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should fail quickly, not hang
        self.assertLess(execution_time, 1.0)


class TestCharacterClientCompatibility(unittest.TestCase):
    """Compatibility tests for different environments and Python versions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.com", api_key="test_key")
    
    def test_json_serialization_compatibility(self):
        """Test JSON serialization with different data types."""
        from datetime import datetime, date
        from decimal import Decimal
        
        complex_data = {
            "name": "Test Character",
            "class": "Warrior",
            "level": 1,
            "created_at": datetime.now(),
            "birth_date": date.today(),
            "experience": Decimal("123.45"),
            "active": True,
            "scores": [1, 2, 3, 4, 5]
        }
        
        # Should handle complex data types gracefully
        try:
            sanitized = self.client._sanitize_character_data(complex_data)
            # Datetime objects should be converted to strings
            if "created_at" in sanitized:
                self.assertIsInstance(sanitized["created_at"], str)
        except (TypeError, ValidationError):
            # Acceptable to raise error for unsupported types
            pass
    
    def test_unicode_handling(self):
        """Test proper handling of Unicode characters."""
        unicode_data = {
            "name": "英雄",  # Chinese characters
            "class": "Воин",  # Cyrillic characters
            "level": 1,
            "description": "A hero with émojis 🎮⚔️🛡️"
        }
        
        # Should handle Unicode properly
        result = self.client._validate_character_data(unicode_data)
        self.assertEqual(result["name"], "英雄")
        self.assertIn("🎮", result.get("description", ""))
    
    def test_python_version_compatibility(self):
        """Test compatibility features across Python versions."""
        import sys
        
        # Test features that might differ between Python versions
        if sys.version_info >= (3, 8):
            # Test features available in Python 3.8+
            self.assertTrue(hasattr(self.client, '__dict__'))
        
        if sys.version_info >= (3, 9):
            # Test features available in Python 3.9+
            # Example: dict union operator
            default_headers = {"Content-Type": "application/json"}
            custom_headers = {"User-Agent": "Test"}
            
            # Should work in Python 3.9+
            try:
                combined = default_headers | custom_headers
                self.assertIn("Content-Type", combined)
                self.assertIn("User-Agent", combined)
            except TypeError:
                # Union operator not available, use update instead
                combined = default_headers.copy()
                combined.update(custom_headers)
                self.assertIn("Content-Type", combined)
                self.assertIn("User-Agent", combined)


if __name__ == '__main__':
    # Configure test runner with additional options
    import sys
    import logging
    
    # Set up logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add custom test result class for better reporting
    class DetailedTestResult(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            if self.verbosity > 1:
                self.stream.write(f"✓ {test._testMethodName}\n")
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.stream.write(f"✗ {test._testMethodName} FAILED\n")
        
        def addError(self, test, err):
            super().addError(test, err)
            self.stream.write(f"✗ {test._testMethodName} ERROR\n")
    
    # Custom test runner
    class DetailedTestRunner(unittest.TextTestRunner):
        resultclass = DetailedTestResult
    
    # Run all tests with custom runner
    if len(sys.argv) > 1 and sys.argv[1] == '--detailed':
        unittest.main(testRunner=DetailedTestRunner(verbosity=2), argv=[''])
    else:
        unittest.main(verbosity=2, buffer=True)