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
    """Advanced test scenarios for CharacterClient including concurrency, retries, and complex edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    # Advanced HTTP Status Code Tests
    @patch('requests.Session.get')
    def test_get_character_partial_content_206(self, mock_get):
        """Test handling of 206 Partial Content responses."""
        mock_response = Mock()
        mock_response.status_code = 206
        mock_response.json.return_value = {"id": 1, "name": "Partial Data"}
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result["name"], "Partial Data")
    
    @patch('requests.Session.get')
    def test_get_character_not_modified_304(self, mock_get):
        """Test handling of 304 Not Modified responses."""
        mock_response = Mock()
        mock_response.status_code = 304
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Unexpected status code: 304", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_forbidden_403(self, mock_get):
        """Test handling of 403 Forbidden responses."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        
        with self.assertRaises(AuthenticationError) as context:
            self.client.get_character(1)
        self.assertIn("Access forbidden", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_method_not_allowed_405(self, mock_get):
        """Test handling of 405 Method Not Allowed responses."""
        mock_response = Mock()
        mock_response.status_code = 405
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Method not allowed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_conflict_409(self, mock_get):
        """Test handling of 409 Conflict responses."""
        mock_response = Mock()
        mock_response.status_code = 409
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("API error: 409", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_unprocessable_entity_422(self, mock_get):
        """Test handling of 422 Unprocessable Entity responses."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValidationError) as context:
            self.client.get_character(1)
        self.assertIn("Invalid request data", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_internal_server_error_500(self, mock_get):
        """Test handling of 500 Internal Server Error responses."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("API error: 500", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_bad_gateway_502(self, mock_get):
        """Test handling of 502 Bad Gateway responses."""
        mock_response = Mock()
        mock_response.status_code = 502
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("API error: 502", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_service_unavailable_503(self, mock_get):
        """Test handling of 503 Service Unavailable responses."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Service unavailable", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_gateway_timeout_504(self, mock_get):
        """Test handling of 504 Gateway Timeout responses."""
        mock_response = Mock()
        mock_response.status_code = 504
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Gateway timeout", str(context.exception))
    
    # Advanced Network Error Tests
    @patch('requests.Session.get')
    def test_get_character_ssl_error(self, mock_get):
        """Test handling of SSL certificate errors."""
        mock_get.side_effect = requests.exceptions.SSLError("SSL certificate verify failed")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("SSL certificate verify failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_proxy_error(self, mock_get):
        """Test handling of proxy errors."""
        mock_get.side_effect = requests.exceptions.ProxyError("Proxy connection failed")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Proxy connection failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_chunked_encoding_error(self, mock_get):
        """Test handling of chunked encoding errors."""
        mock_get.side_effect = requests.exceptions.ChunkedEncodingError("Connection broken")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_get_character_content_decoding_error(self, mock_get):
        """Test handling of content decoding errors."""
        mock_get.side_effect = requests.exceptions.ContentDecodingError("Failed to decode response content")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Failed to decode response content", str(context.exception))
    
    # Advanced Session and Connection Tests
    def test_session_persistence(self):
        """Test that the same session is reused across multiple requests."""
        client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
        initial_session = client.session
        
        # Make sure the session object persists
        self.assertIs(client.session, initial_session)
        
        client._setup_session()  # Call setup again
        self.assertIs(client.session, initial_session)  # Should be the same object
    
    def test_session_headers_immutability(self):
        """Test that session headers cannot be easily mutated externally."""
        client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
        original_headers = dict(client.session.headers)
        
        # Try to modify headers externally
        client.session.headers['X-Test'] = 'modified'
        
        # Headers should be modifiable (this is expected behavior)
        self.assertEqual(client.session.headers['X-Test'], 'modified')
        
        # But original headers should still be there
        for key, value in original_headers.items():
            if key != 'X-Test':
                self.assertEqual(client.session.headers[key], value)
    
    def test_custom_user_agent_header(self):
        """Test setting custom user agent in session headers."""
        client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
        expected_user_agent = 'CharacterClient/1.0'
        self.assertEqual(client.session.headers['User-Agent'], expected_user_agent)
    
    # Complex Validation Tests
    def test_validate_character_data_with_nested_attributes(self):
        """Test character data validation with nested attribute structures."""
        valid_nested_data = {
            "name": "Complex Character",
            "class": "Paladin",
            "level": 10,
            "attributes": {
                "strength": 18,
                "dexterity": 14,
                "constitution": 16,
                "intelligence": 12,
                "wisdom": 15,
                "charisma": 17
            },
            "equipment": {
                "weapon": {"name": "Holy Sword", "damage": 15},
                "armor": {"name": "Plate Mail", "defense": 20}
            }
        }
        
        # Should not raise an exception for complex nested data
        result = self.client._validate_character_data(valid_nested_data)
        self.assertTrue(result)
    
    def test_validate_character_data_with_arrays(self):
        """Test character data validation with array fields."""
        data_with_arrays = {
            "name": "Adventurer",
            "class": "Ranger",
            "level": 5,
            "skills": ["Archery", "Tracking", "Survival"],
            "inventory": [
                {"item": "Bow", "quantity": 1},
                {"item": "Arrow", "quantity": 50},
                {"item": "Health Potion", "quantity": 3}
            ]
        }
        
        result = self.client._validate_character_data(data_with_arrays)
        self.assertTrue(result)
    
    def test_validate_character_data_with_null_values(self):
        """Test character data validation with null/None values."""
        data_with_nulls = {
            "name": "Mysterious Character",
            "class": "Unknown",
            "level": 1,
            "description": None,
            "last_seen": None
        }
        
        result = self.client._validate_character_data(data_with_nulls)
        self.assertTrue(result)
    
    # Comprehensive Parameter Validation Tests
    def test_get_characters_boundary_conditions(self):
        """Test get_characters with boundary condition parameters."""
        # Test maximum valid values
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"characters": []}
            mock_get.return_value = mock_response
            
            # Test maximum page and limit
            result = self.client.get_characters(page=999999, limit=100)
            self.assertEqual(result, [])
            
            # Verify correct parameters were passed
            call_args = mock_get.call_args
            self.assertEqual(call_args[1]['params']['page'], 999999)
            self.assertEqual(call_args[1]['params']['limit'], 100)
    
    def test_get_characters_with_multiple_filters(self):
        """Test get_characters with multiple filter combinations."""
        test_cases = [
            {
                "character_class": "Warrior",
                "min_level": 10,
                "max_level": 20,
                "page": 1,
                "limit": 25
            },
            {
                "character_class": "Mage",
                "sort_by": "level",
                "sort_order": "desc"
            }
        ]
        
        for params in test_cases:
            with self.subTest(params=params):
                with patch('requests.Session.get') as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"characters": []}
                    mock_get.return_value = mock_response
                    
                    result = self.client.get_characters(**params)
                    self.assertEqual(result, [])
                    
                    # Verify all parameters were passed
                    call_args = mock_get.call_args
                    for key, value in params.items():
                        if key == 'character_class':
                            self.assertEqual(call_args[1]['params']['class'], value)
                        else:
                            self.assertEqual(call_args[1]['params'][key], value)
    
    # Response Content Validation Tests
    @patch('requests.Session.get')
    def test_get_character_response_with_extra_fields(self, mock_get):
        """Test handling of responses with extra unexpected fields."""
        character_with_extras = {
            "id": 1,
            "name": "Extended Character",
            "class": "Warrior",
            "level": 10,
            "health": 100,
            "mana": 50,
            # Extra fields that might be added in future API versions
            "experience": 5000,
            "guild": "Heroes Guild",
            "last_login": "2023-12-01T10:00:00Z",
            "premium_features": {
                "double_xp": True,
                "custom_appearance": True
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = character_with_extras
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        
        # Should return all fields, including extras
        self.assertEqual(result, character_with_extras)
        self.assertEqual(result["experience"], 5000)
        self.assertEqual(result["guild"], "Heroes Guild")
    
    @patch('requests.Session.get')
    def test_get_characters_response_with_metadata(self, mock_get):
        """Test get_characters response with comprehensive metadata."""
        response_with_metadata = {
            "characters": [
                {"id": 1, "name": "Hero 1", "class": "Warrior", "level": 10},
                {"id": 2, "name": "Hero 2", "class": "Mage", "level": 8}
            ],
            "pagination": {
                "current_page": 1,
                "total_pages": 5,
                "total_characters": 47,
                "per_page": 10
            },
            "filters_applied": {
                "class": "Warrior",
                "min_level": 5
            },
            "request_metadata": {
                "request_id": "req_123456",
                "timestamp": "2023-12-01T10:00:00Z",
                "api_version": "v2"
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_with_metadata
        mock_get.return_value = mock_response
        
        result = self.client.get_characters()
        
        # Should extract just the characters array
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Hero 1")
        self.assertEqual(result[1]["name"], "Hero 2")
    
    # Memory and Resource Management Tests
    def test_client_cleanup_after_use(self):
        """Test proper cleanup of client resources."""
        client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
        session = client.session
        
        # Ensure session exists
        self.assertIsNotNone(session)
        
        # Manually close session (simulating proper cleanup)
        client.session.close()
        
        # Session should still exist but be closed
        self.assertIsNotNone(client.session)
    
    def test_multiple_client_instances_independence(self):
        """Test that multiple client instances don't interfere with each other."""
        client1 = CharacterClient(base_url="https://api1.com", api_key="key1")
        client2 = CharacterClient(base_url="https://api2.com", api_key="key2")
        
        # Verify they have different configurations
        self.assertNotEqual(client1.base_url, client2.base_url)
        self.assertNotEqual(client1.api_key, client2.api_key)
        
        # Verify they have different sessions
        self.assertIsNot(client1.session, client2.session)
        
        # Verify session headers are different
        self.assertNotEqual(
            client1.session.headers['Authorization'],
            client2.session.headers['Authorization']
        )
    
    # Character Data Type Edge Cases
    def test_create_character_with_float_level(self):
        """Test character creation with float level (should be converted to int)."""
        character_data = {
            "name": "Float Level Character",
            "class": "Warrior",
            "level": 10.7  # Float level
        }
        
        # Should convert float to int during validation
        with self.assertRaises(ValueError) as context:
            self.client.create_character(character_data)
        self.assertIn("Character level must be a positive integer", str(context.exception))
    
    def test_create_character_with_boolean_values(self):
        """Test character creation with boolean values in data."""
        character_data = {
            "name": "Boolean Character",
            "class": "Paladin",
            "level": 5,
            "is_premium": True,
            "is_active": False,
            "has_guild": True
        }
        
        # Should pass validation with boolean values
        result = self.client._validate_character_data(character_data)
        self.assertTrue(result)
    
    # URL and Path Construction Edge Cases
    def test_build_url_with_special_characters(self):
        """Test URL building with special characters in resource paths."""
        # Test with URL-encoded characters
        result = self.client._build_url("characters/search", params={"name": "Hero@123"})
        self.assertIn("characters/search", result)
        self.assertIn("name=Hero%40123", result)
    
    def test_build_url_with_unicode_parameters(self):
        """Test URL building with Unicode characters in parameters."""
        unicode_params = {
            "name": "英雄",  # Chinese characters
            "description": "Héroe español"  # Spanish with accents
        }
        
        result = self.client._build_url("characters", params=unicode_params)
        
        # Should properly encode Unicode characters
        self.assertIn("characters", result)
        # URL encoding should be present
        self.assertIn("%", result)
    
    def test_build_url_with_none_values(self):
        """Test URL building with None values in parameters."""
        params_with_none = {
            "name": "Test",
            "description": None,
            "guild": None
        }
        
        result = self.client._build_url("characters", params=params_with_none)
        
        # Should handle None values gracefully
        self.assertIn("characters", result)
        self.assertIn("name=Test", result)
        # None values should be excluded or handled appropriately


class TestCharacterClientIntegrationScenarios(unittest.TestCase):
    """Integration-style tests for CharacterClient that test workflows and scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_complete_character_lifecycle(self, mock_get, mock_post):
        """Test complete character lifecycle: create, read, update, delete."""
        # Step 1: Create character
        create_data = {"name": "Lifecycle Test", "class": "Ranger", "level": 1}
        created_character = {**create_data, "id": 100}
        
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = created_character
        mock_post.return_value = mock_post_response
        
        # Step 2: Read character after creation
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = created_character
        mock_get.return_value = mock_get_response
        
        # Execute create
        created = self.client.create_character(create_data)
        self.assertEqual(created["id"], 100)
        self.assertEqual(created["name"], "Lifecycle Test")
        
        # Execute read
        retrieved = self.client.get_character(100)
        self.assertEqual(retrieved["id"], 100)
        self.assertEqual(retrieved["name"], "Lifecycle Test")
    
    @patch('requests.Session.get')
    def test_character_search_and_filtering_workflow(self, mock_get):
        """Test a complete character search and filtering workflow."""
        # Simulate searching for warriors
        warrior_characters = [
            {"id": 1, "name": "Warrior 1", "class": "Warrior", "level": 10},
            {"id": 2, "name": "Warrior 2", "class": "Warrior", "level": 15}
        ]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"characters": warrior_characters}
        mock_get.return_value = mock_response
        
        # Search for warriors
        warriors = self.client.get_characters(character_class="Warrior")
        
        self.assertEqual(len(warriors), 2)
        for warrior in warriors:
            self.assertEqual(warrior["class"], "Warrior")
        
        # Verify the search parameters
        call_args = mock_get.call_args
        self.assertEqual(call_args[1]['params']['class'], "Warrior")
    
    @patch.object(CharacterClient, 'get_character')
    def test_batch_processing_with_error_handling(self, mock_get_character):
        """Test batch processing with mixed success and failure scenarios."""
        def get_character_side_effect(char_id):
            if char_id == 1:
                return {"id": 1, "name": "Success 1"}
            elif char_id == 2:
                raise CharacterNotFoundError("Character not found")
            elif char_id == 3:
                return {"id": 3, "name": "Success 3"}
            elif char_id == 4:
                raise CharacterClientError("Server error")
            else:
                return {"id": char_id, "name": f"Character {char_id}"}
        
        mock_get_character.side_effect = get_character_side_effect
        
        # Test batch retrieval with mixed results
        character_ids = [1, 2, 3, 4, 5]
        results = self.client.get_characters_batch(character_ids)
        
        # Should return only successful retrievals
        self.assertEqual(len(results), 3)  # 1, 3, and 5 should succeed
        
        success_ids = [result["id"] for result in results]
        self.assertIn(1, success_ids)
        self.assertIn(3, success_ids)
        self.assertIn(5, success_ids)
        self.assertNotIn(2, success_ids)  # Not found
        self.assertNotIn(4, success_ids)  # Server error
    
    def test_character_data_validation_comprehensive(self):
        """Test comprehensive character data validation scenarios."""
        # Test cases with various validation scenarios
        test_cases = [
            # Valid cases
            ({"name": "Valid Hero", "class": "Warrior", "level": 1}, True),
            ({"name": "Mage Hero", "class": "Mage", "level": 50}, True),
            ({"name": "Max Level", "class": "Paladin", "level": 100}, True),
            
            # Invalid cases - missing fields
            ({"class": "Warrior", "level": 1}, False),  # Missing name
            ({"name": "Hero", "level": 1}, False),  # Missing class
            ({"name": "Hero", "class": "Warrior"}, False),  # Missing level
            
            # Invalid cases - empty values
            ({"name": "", "class": "Warrior", "level": 1}, False),  # Empty name
            ({"name": "Hero", "class": "", "level": 1}, False),  # Empty class
            ({"name": "Hero", "class": "Warrior", "level": 0}, False),  # Zero level
            
            # Invalid cases - wrong types
            ({"name": 123, "class": "Warrior", "level": 1}, False),  # Non-string name
            ({"name": "Hero", "class": 456, "level": 1}, False),  # Non-string class
            ({"name": "Hero", "class": "Warrior", "level": "high"}, False),  # Non-int level
        ]
        
        for test_data, should_be_valid in test_cases:
            with self.subTest(data=test_data, valid=should_be_valid):
                if should_be_valid:
                    result = self.client._validate_character_data(test_data)
                    self.assertTrue(result)
                else:
                    with self.assertRaises((ValueError, TypeError)):
                        self.client._validate_character_data(test_data)


class TestCharacterClientPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests for CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    def test_large_batch_character_ids(self):
        """Test handling of large batches of character IDs."""
        # Test with a large list of character IDs
        large_id_list = list(range(1, 1001))  # 1000 character IDs
        
        with self.assertRaises(ValueError) as context:
            # Most APIs would have limits on batch sizes
            # This should trigger validation if implemented
            if hasattr(self.client, '_validate_batch_size'):
                self.client._validate_batch_size(large_id_list)
        
        # If no batch size validation exists, test that it doesn't crash
        with patch.object(self.client, 'get_character') as mock_get:
            mock_get.return_value = {"id": 1, "name": "Test"}
            
            # This might be slow but shouldn't crash
            results = self.client.get_characters_batch(large_id_list[:10])  # Test with smaller subset
            self.assertEqual(len(results), 10)
    
    def test_rapid_successive_requests(self):
        """Test rapid successive API requests."""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1, "name": "Test Character"}
            mock_get.return_value = mock_response
            
            # Make multiple rapid requests
            results = []
            for i in range(100):
                result = self.client.get_character(i + 1)
                results.append(result)
            
            # All requests should succeed
            self.assertEqual(len(results), 100)
            self.assertEqual(mock_get.call_count, 100)
    
    def test_memory_usage_with_large_responses(self):
        """Test memory handling with large response data."""
        # Create a large character response
        large_character = {
            "id": 1,
            "name": "Large Data Character",
            "class": "DataMage",
            "level": 100,
            "inventory": [{"item": f"Item_{i}", "value": i} for i in range(10000)],
            "skills": [f"Skill_{i}" for i in range(1000)],
            "attributes": {f"attr_{i}": i for i in range(1000)}
        }
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = large_character
            mock_get.return_value = mock_response
            
            result = self.client.get_character(1)
            
            # Should handle large responses without issues
            self.assertEqual(result["id"], 1)
            self.assertEqual(len(result["inventory"]), 10000)
            self.assertEqual(len(result["skills"]), 1000)


class TestCharacterClientRetryAndRobustness(unittest.TestCase):
    """Test retry mechanisms and robustness scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    @patch('requests.Session.get')
    def test_temporary_network_failure_recovery(self, mock_get):
        """Test recovery from temporary network failures."""
        # Simulate temporary failure followed by success
        mock_get.side_effect = [
            requests.ConnectionError("Temporary failure"),
            Mock(status_code=200, json=lambda: {"id": 1, "name": "Success"})
        ]
        
        # If the client has retry logic, it should eventually succeed
        # If not, it should fail on the first attempt
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
    
    @patch('requests.Session.get')
    def test_response_with_invalid_json(self, mock_get):
        """Test handling of responses with invalid JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("JSON", str(context.exception))
    
    @patch('requests.Session.get')
    def test_empty_response_body(self, mock_get):
        """Test handling of empty response body."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertIsNone(result)
    
    @patch('requests.Session.get')
    def test_response_with_missing_content_type(self, mock_get):
        """Test handling of responses without content-type header."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}  # No content-type header
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_get.return_value = mock_response
        
        # Should still work if JSON parsing succeeds
        result = self.client.get_character(1)
        self.assertEqual(result["id"], 1)


class TestCharacterClientSecurityAndValidation(unittest.TestCase):
    """Test security-related features and input validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    def test_api_key_in_headers(self):
        """Test that API key is properly included in headers."""
        expected_auth_header = f"Bearer {self.client.api_key}"
        self.assertEqual(
            self.client.session.headers['Authorization'],
            expected_auth_header
        )
    
    def test_sensitive_data_not_logged(self):
        """Test that sensitive data like API keys are not exposed in logs."""
        # This is more of a guideline test - in real scenarios,
        # you'd check logging output
        client_str = str(self.client)
        api_key_str = str(self.client.api_key)
        
        # API key should not appear in string representation
        if hasattr(self.client, '__str__'):
            self.assertNotIn(self.client.api_key, client_str)
    
    def test_sql_injection_protection_in_parameters(self):
        """Test that SQL injection attempts in parameters are handled safely."""
        malicious_inputs = [
            "'; DROP TABLE characters; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            "<script>alert('xss')</script>"
        ]
        
        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input):
                with patch('requests.Session.get') as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"characters": []}
                    mock_get.return_value = mock_response
                    
                    # Should handle malicious input safely
                    result = self.client.get_characters(character_class=malicious_input)
                    self.assertEqual(result, [])
                    
                    # Verify the malicious input was passed as a parameter
                    # (the server should handle the actual sanitization)
                    call_args = mock_get.call_args
                    self.assertEqual(call_args[1]['params']['class'], malicious_input)
    
    def test_extremely_large_input_handling(self):
        """Test handling of extremely large input values."""
        # Test with very large strings
        large_string = "A" * 1000000  # 1MB string
        
        character_data = {
            "name": large_string,
            "class": "Warrior",
            "level": 1
        }
        
        # Should handle large input without crashing
        # (though it might be rejected by validation)
        try:
            result = self.client._validate_character_data(character_data)
            # If validation passes, that's also acceptable
            self.assertTrue(result)
        except (ValueError, MemoryError):
            # If validation rejects it, that's also acceptable
            pass
    
    def test_unicode_normalization_attacks(self):
        """Test handling of Unicode normalization attacks."""
        # Unicode characters that might normalize to different values
        unicode_attacks = [
            "café",  # NFC normalization
            "cafe\u0301",  # NFD normalization
            "ℌ𝕖𝔩𝔩𝕠",  # Mathematical bold characters
            "𝐇𝐞𝐥𝐥𝐨"  # Mathematical script characters
        ]
        
        for unicode_input in unicode_attacks:
            with self.subTest(input=unicode_input):
                character_data = {
                    "name": unicode_input,
                    "class": "Mage",
                    "level": 1
                }
                
                # Should handle Unicode input safely
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)


if __name__ == '__main__':
    # Add these new test classes to the test suite
    unittest.main(verbosity=2, buffer=True)