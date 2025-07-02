import unittest
import sys
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
    """Advanced test scenarios for comprehensive CharacterClient coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
        self.maxDiff = None  # Allow full diff output for large comparisons
    
    # Authentication and Authorization Edge Cases
    @patch('requests.Session.get')
    def test_expired_token_scenario(self, mock_get):
        """Test handling of expired authentication tokens."""
        # First call succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"id": 1, "name": "Test"}
        
        # Second call fails with expired token
        mock_response_expired = Mock()
        mock_response_expired.status_code = 401
        mock_response_expired.json.return_value = {"error": "Token expired"}
        
        mock_get.side_effect = [mock_response_success, mock_response_expired]
        
        # First call should succeed
        result1 = self.client.get_character(1)
        self.assertEqual(result1["id"], 1)
        
        # Second call should raise authentication error
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(2)
        self.assertIn("Authentication failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_rate_limit_with_retry_after_header(self, mock_get):
        """Test rate limit handling with Retry-After header."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    @patch('requests.Session.get')
    def test_forbidden_access_error(self, mock_get):
        """Test handling of 403 Forbidden responses."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Insufficient permissions"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Authentication failed", str(context.exception))
    
    # Network and Connection Edge Cases
    @patch('requests.Session.get')
    def test_ssl_certificate_error(self, mock_get):
        """Test handling of SSL certificate errors."""
        mock_get.side_effect = requests.exceptions.SSLError("SSL certificate verification failed")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_dns_resolution_error(self, mock_get):
        """Test handling of DNS resolution failures."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Name or service not known")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_proxy_connection_error(self, mock_get):
        """Test handling of proxy connection errors."""
        mock_get.side_effect = requests.exceptions.ProxyError("Proxy connection failed")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_chunked_encoding_error(self, mock_get):
        """Test handling of chunked encoding errors."""
        mock_get.side_effect = requests.exceptions.ChunkedEncodingError("Connection broken")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    # Response Handling Edge Cases
    @patch('requests.Session.get')
    def test_empty_response_body(self, mock_get):
        """Test handling of empty response bodies."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_response.text = ""
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Invalid JSON response", str(context.exception))
    
    @patch('requests.Session.get')
    def test_response_encoding_issues(self, mock_get):
        """Test handling of response encoding issues."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 1,
            "name": "Test\udcff\udcfe",  # Invalid UTF-8 sequences
            "description": "Character with encoding issues"
        }
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result["id"], 1)
        self.assertIn("encoding issues", result["description"])
    
    @patch('requests.Session.get')
    def test_very_large_response(self, mock_get):
        """Test handling of very large JSON responses."""
        # Create a large character object with many attributes
        large_character = {
            "id": 1,
            "name": "Large Character",
            "attributes": {f"attr_{i}": i for i in range(1000)},
            "inventory": [{"item_id": i, "name": f"Item {i}"} for i in range(500)],
            "large_description": "A" * 10000
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_character
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result["id"], 1)
        self.assertEqual(len(result["attributes"]), 1000)
        self.assertEqual(len(result["inventory"]), 500)
    
    # Concurrent Access and Thread Safety Tests
    @patch('requests.Session.get')
    def test_concurrent_character_retrieval(self, mock_get):
        """Test concurrent character retrieval operations."""
        import threading
        import time
        
        def mock_response_side_effect(*args, **kwargs):
            time.sleep(0.1)  # Simulate network delay
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1, "name": "Test"}
            return mock_response
        
        mock_get.side_effect = mock_response_side_effect
        
        results = []
        exceptions = []
        
        def fetch_character():
            try:
                result = self.client.get_character(1)
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Start multiple threads
        threads = [threading.Thread(target=fetch_character) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(results), 5)
        self.assertEqual(len(exceptions), 0)
        self.assertEqual(mock_get.call_count, 5)
    
    # Advanced Data Validation Tests
    def test_character_data_validation_comprehensive(self):
        """Test comprehensive character data validation scenarios."""
        # Test nested validation scenarios
        test_cases = [
            # Valid cases with complex data
            ({
                "name": "Complex Character",
                "class": "Multi-Class Warrior/Mage",
                "level": 50,
                "attributes": {
                    "strength": 20,
                    "agility": 15,
                    "intelligence": 18,
                    "luck": 12
                },
                "equipment": ["sword", "shield", "robe"],
                "metadata": {
                    "created_by": "player123",
                    "guild": "TestGuild"
                }
            }, True),
            
            # Edge case: maximum values
            ({
                "name": "Max Level Character",
                "class": "Legendary",
                "level": 999999,
                "health": 2**31 - 1,
                "mana": 2**31 - 1
            }, True),
            
            # Edge case: special character combinations
            ({
                "name": "Character with\nnewlines\tand\ttabs",
                "class": "Tester",
                "level": 1
            }, True)
        ]
        
        for data, should_be_valid in test_cases:
            with self.subTest(data=data):
                if should_be_valid:
                    result = self.client._validate_character_data(data)
                    self.assertTrue(result)
                else:
                    with self.assertRaises((ValueError, TypeError)):
                        self.client._validate_character_data(data)
    
    # URL Building and Parameter Handling Tests
    def test_url_building_with_special_characters(self):
        """Test URL building with special characters in parameters."""
        params = {
            "name": "Hero@Domain.com",
            "class": "Warrior/Mage",
            "description": "A character with spaces & symbols",
            "tags": "RPG,Fantasy,Adventure"
        }
        
        result = self.client._build_url("characters", params=params)
        
        # Check that special characters are properly encoded
        self.assertIn("characters", result)
        self.assertIn("%40", result)  # @ symbol encoded
        self.assertIn("%2F", result)  # / symbol encoded
        self.assertIn("%20", result)  # space encoded
    
    def test_url_building_with_unicode_parameters(self):
        """Test URL building with Unicode parameters."""
        params = {
            "name": "キャラクター",
            "description": "測試角色",
            "location": "Москва"
        }
        
        result = self.client._build_url("characters", params=params)
        self.assertIn("characters", result)
        # Unicode should be properly encoded in URL
        self.assertTrue(any(char in result for char in ["%", "+"]))
    
    # Session Management Tests
    def test_session_headers_modification(self):
        """Test that session headers can be modified after initialization."""
        original_user_agent = self.client.session.headers['User-Agent']
        
        # Modify session headers
        self.client.session.headers['User-Agent'] = 'CharacterClient/2.0'
        self.client.session.headers['X-Custom-Header'] = 'test-value'
        
        self.assertEqual(self.client.session.headers['User-Agent'], 'CharacterClient/2.0')
        self.assertEqual(self.client.session.headers['X-Custom-Header'], 'test-value')
        self.assertNotEqual(self.client.session.headers['User-Agent'], original_user_agent)
    
    def test_session_timeout_handling(self):
        """Test that session timeout is properly configured."""
        # Test with custom timeout
        custom_client = CharacterClient(
            base_url="https://api.test.com",
            api_key="test_key",
            timeout=120
        )
        
        self.assertEqual(custom_client.timeout, 120)
        
        # Test timeout edge cases
        with self.assertRaises(ValueError):
            CharacterClient(
                base_url="https://api.test.com",
                api_key="test_key",
                timeout=-1
            )
        
        with self.assertRaises(ValueError):
            CharacterClient(
                base_url="https://api.test.com",
                api_key="test_key",
                timeout=0
            )
    
    # Batch Operations Advanced Tests
    @patch.object(CharacterClient, 'get_character')
    def test_get_characters_batch_with_mixed_results(self, mock_get_character):
        """Test batch operations with mixed success/failure results."""
        character_ids = [1, 2, 3, 4, 5]
        
        def side_effect(char_id):
            if char_id == 2:
                raise CharacterNotFoundError("Character not found")
            elif char_id == 4:
                raise CharacterClientError("API error")
            else:
                return {"id": char_id, "name": f"Character {char_id}"}
        
        mock_get_character.side_effect = side_effect
        
        results = self.client.get_characters_batch(character_ids)
        
        # Should return only successful results
        self.assertEqual(len(results), 3)  # 1, 3, 5 should succeed
        successful_ids = [r["id"] for r in results]
        self.assertIn(1, successful_ids)
        self.assertIn(3, successful_ids)
        self.assertIn(5, successful_ids)
        self.assertNotIn(2, successful_ids)
        self.assertNotIn(4, successful_ids)
    
    @patch.object(CharacterClient, 'get_character')
    def test_get_characters_batch_performance_with_large_dataset(self, mock_get_character):
        """Test batch operations performance with large datasets."""
        # Test with 100 character IDs
        character_ids = list(range(1, 101))
        mock_get_character.side_effect = [
            {"id": i, "name": f"Character {i}"} for i in character_ids
        ]
        
        import time
        start_time = time.time()
        results = self.client.get_characters_batch(character_ids)
        end_time = time.time()
        
        self.assertEqual(len(results), 100)
        self.assertEqual(mock_get_character.call_count, 100)
        
        # Should complete within reasonable time (adjust threshold as needed)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5.0, "Batch operation took too long")
    
    # Error Recovery and Resilience Tests
    @patch('requests.Session.get')
    def test_partial_network_failure_recovery(self, mock_get):
        """Test recovery from partial network failures."""
        # Simulate intermittent network issues
        responses = [
            requests.ConnectionError("Network error"),
            requests.Timeout("Request timeout"),
            Mock(status_code=200, json=lambda: {"id": 1, "name": "Success"})
        ]
        
        mock_get.side_effect = responses
        
        # First two calls should fail
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
        
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
        
        # Third call should succeed
        result = self.client.get_character(1)
        self.assertEqual(result["id"], 1)
    
    # Content Type and Media Type Tests
    @patch('requests.Session.post')
    def test_create_character_with_alternative_content_types(self, mock_post):
        """Test character creation with different content type handling."""
        character_data = {"name": "Test", "class": "Warrior", "level": 1}
        
        # Test response with different content types
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.headers = {'Content-Type': 'application/json; charset=utf-8'}
        mock_response.json.return_value = {**character_data, "id": 1}
        mock_post.return_value = mock_response
        
        result = self.client.create_character(character_data)
        self.assertEqual(result["id"], 1)
        
        # Verify correct content type was sent
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json'], character_data)
    
    # Memory and Resource Management Tests
    def test_client_cleanup_on_deletion(self):
        """Test that client properly cleans up resources on deletion."""
        client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
        session = client.session
        
        # Delete client
        del client
        
        # Session should still be valid (Python GC handles cleanup)
        self.assertIsNotNone(session)
    
    def test_session_reuse_across_operations(self):
        """Test that the same session is reused across multiple operations."""
        initial_session = self.client.session
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1, "name": "Test"}
            mock_get.return_value = mock_response
            
            # Make multiple calls
            self.client.get_character(1)
            self.client.get_character(2)
            
            # Session should remain the same
            self.assertIs(self.client.session, initial_session)
            self.assertEqual(mock_get.call_count, 2)


class TestCharacterClientIntegrationScenarios(unittest.TestCase):
    """Integration-style tests for end-to-end workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
    
    @patch('requests.Session.post')
    @patch('requests.Session.get')
    @patch('requests.Session.put')
    @patch('requests.Session.delete')
    def test_complete_character_lifecycle(self, mock_delete, mock_put, mock_get, mock_post):
        """Test complete character lifecycle: create, read, update, delete."""
        # Mock character creation
        create_data = {"name": "Lifecycle Test", "class": "Warrior", "level": 1}
        created_character = {**create_data, "id": 100, "created_at": "2023-01-01T00:00:00Z"}
        
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = created_character
        mock_post.return_value = mock_post_response
        
        # Mock character retrieval
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = created_character
        mock_get.return_value = mock_get_response
        
        # Mock character update
        updated_character = {**created_character, "level": 5, "health": 150}
        mock_put_response = Mock()
        mock_put_response.status_code = 200
        mock_put_response.json.return_value = updated_character
        mock_put.return_value = mock_put_response
        
        # Mock character deletion
        mock_delete_response = Mock()
        mock_delete_response.status_code = 204
        mock_delete.return_value = mock_delete_response
        
        # Execute complete lifecycle
        # 1. Create character
        created = self.client.create_character(create_data)
        self.assertEqual(created["name"], "Lifecycle Test")
        self.assertEqual(created["id"], 100)
        
        # 2. Read character
        retrieved = self.client.get_character(100)
        self.assertEqual(retrieved["id"], 100)
        
        # 3. Update character
        update_data = {"level": 5, "health": 150}
        updated = self.client.update_character(100, update_data)
        self.assertEqual(updated["level"], 5)
        self.assertEqual(updated["health"], 150)
        
        # 4. Delete character
        deleted = self.client.delete_character(100)
        self.assertTrue(deleted)
        
        # Verify all operations were called
        mock_post.assert_called_once()
        mock_get.assert_called_once()
        mock_put.assert_called_once()
        mock_delete.assert_called_once()
    
    @patch('requests.Session.get')
    def test_character_search_and_filter_workflow(self, mock_get):
        """Test character search and filtering workflow."""
        # Mock paginated character list responses
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "characters": [
                {"id": 1, "name": "Warrior One", "class": "Warrior", "level": 10},
                {"id": 2, "name": "Mage One", "class": "Mage", "level": 8}
            ],
            "page": 1,
            "total_pages": 2
        }
        
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "characters": [
                {"id": 3, "name": "Warrior Two", "class": "Warrior", "level": 12},
                {"id": 4, "name": "Rogue One", "class": "Rogue", "level": 6}
            ],
            "page": 2,
            "total_pages": 2
        }
        
        filtered_response = Mock()
        filtered_response.status_code = 200
        filtered_response.json.return_value = {
            "characters": [
                {"id": 1, "name": "Warrior One", "class": "Warrior", "level": 10},
                {"id": 3, "name": "Warrior Two", "class": "Warrior", "level": 12}
            ]
        }
        
        mock_get.side_effect = [page1_response, page2_response, filtered_response]
        
        # Test paginated retrieval
        page1_chars = self.client.get_characters(page=1, limit=2)
        self.assertEqual(len(page1_chars), 2)
        self.assertEqual(page1_chars[0]["class"], "Warrior")
        
        page2_chars = self.client.get_characters(page=2, limit=2)
        self.assertEqual(len(page2_chars), 2)
        self.assertEqual(page2_chars[1]["class"], "Rogue")
        
        # Test filtered retrieval
        warrior_chars = self.client.get_characters(character_class="Warrior")
        self.assertEqual(len(warrior_chars), 2)
        self.assertTrue(all(char["class"] == "Warrior" for char in warrior_chars))


class TestCharacterClientPerformanceScenarios(unittest.TestCase):
    """Performance and stress testing scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
    
    @patch('requests.Session.get')
    def test_rapid_successive_requests(self, mock_get):
        """Test handling of rapid successive API requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_get.return_value = mock_response
        
        import time
        start_time = time.time()
        
        # Make 50 rapid requests
        for i in range(50):
            result = self.client.get_character(i + 1)
            self.assertEqual(result["id"], 1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete all requests within reasonable time
        self.assertLess(execution_time, 2.0, "Rapid requests took too long")
        self.assertEqual(mock_get.call_count, 50)
    
    @patch('requests.Session.post')
    def test_large_payload_handling(self, mock_post):
        """Test handling of large character data payloads."""
        # Create a large character with extensive data
        large_character_data = {
            "name": "Large Character",
            "class": "Legendary",
            "level": 100,
            "attributes": {f"skill_{i}": i * 10 for i in range(1000)},
            "inventory": [
                {
                    "item_id": i,
                    "name": f"Item {i}",
                    "description": "A" * 500,  # Large description
                    "properties": {f"prop_{j}": j for j in range(50)}
                }
                for i in range(100)
            ],
            "biography": "B" * 50000,  # Very large biography
            "quest_log": [f"Quest {i}" for i in range(500)]
        }
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {**large_character_data, "id": 1}
        mock_post.return_value = mock_response
        
        result = self.client.create_character(large_character_data)
        self.assertEqual(result["id"], 1)
        self.assertEqual(len(result["inventory"]), 100)
        self.assertEqual(len(result["attributes"]), 1000)
    
    def test_memory_usage_stability(self):
        """Test that client doesn't accumulate excessive memory over time."""
        import sys
        
        initial_refcount = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # Perform many operations that create temporary objects
        for i in range(100):
            url = self.client._build_url("characters", resource_id=i, params={"test": i})
            self.assertIn("characters", url)
            
            # Validate data multiple times
            data = {"name": f"Character {i}", "class": "Test", "level": i}
            self.client._validate_character_data(data)
        
        final_refcount = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        
        # Reference count shouldn't grow excessively (allowing for some variance)
        if hasattr(sys, 'gettotalrefcount'):
            refcount_growth = final_refcount - initial_refcount
            self.assertLess(refcount_growth, 1000, "Excessive memory growth detected")


class TestCharacterClientSecurityScenarios(unittest.TestCase):
    """Security-focused test scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
    
    def test_api_key_not_logged_in_url(self):
        """Test that API key is not accidentally logged in URLs or error messages."""
        # Test URL building doesn't expose API key
        url = self.client._build_url("characters")
        self.assertNotIn(self.client.api_key, url)
        
        # Test with parameters
        url_with_params = self.client._build_url("characters", params={"test": "value"})
        self.assertNotIn(self.client.api_key, url_with_params)
    
    def test_malicious_character_data_handling(self):
        """Test handling of potentially malicious character data."""
        malicious_inputs = [
            # XSS-like payloads
            {"name": "<script>alert('xss')</script>", "class": "Warrior", "level": 1},
            # SQL injection-like payloads
            {"name": "'; DROP TABLE characters; --", "class": "Hacker", "level": 1},
            # Path traversal attempts
            {"name": "../../../etc/passwd", "class": "Infiltrator", "level": 1},
            # Very long strings that might cause buffer overflows
            {"name": "A" * 100000, "class": "Buffer", "level": 1},
            # Null bytes
            {"name": "Test\x00Character", "class": "Null", "level": 1}
        ]
        
        for malicious_data in malicious_inputs:
            with self.subTest(data=malicious_data):
                # Should not raise exceptions, just process the data
                result = self.client._validate_character_data(malicious_data)
                self.assertTrue(result)
    
    @patch('requests.Session.get')
    def test_response_header_injection_protection(self, mock_get):
        """Test protection against response header injection."""
        # Mock response with potentially malicious headers
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'Content-Type': 'application/json\r\nX-Injected: malicious',
            'X-Custom-Header': 'value\r\nSet-Cookie: sessionid=stolen'
        }
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_get.return_value = mock_response
        
        # Should handle the response without issues
        result = self.client.get_character(1)
        self.assertEqual(result["id"], 1)


if __name__ == '__main__':
    # Run all test classes with proper test discovery
    test_classes = [
        TestCharacterClient,
        TestCharacterClientEdgeCases,
        TestCharacterClientAdvancedScenarios,
        TestCharacterClientIntegrationScenarios,
        TestCharacterClientPerformanceScenarios,
        TestCharacterClientSecurityScenarios
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Configure detailed test runner
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False,
        stream=sys.stdout
    )
    
    # Run the comprehensive test suite
    result = runner.run(suite)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)