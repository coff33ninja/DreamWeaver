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

class TestCharacterClientAdvanced(unittest.TestCase):
    """Advanced unit tests for CharacterClient class covering additional scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
        self.sample_character = {
            "id": 1,
            "name": "Test Hero",
            "class": "Warrior",
            "level": 10,
            "health": 100,
            "mana": 50
        }
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    # Authentication and Authorization Tests
    @patch('requests.Session.get')
    def test_authentication_with_different_auth_schemes(self, mock_get):
        """Test various authentication schemes."""
        # Test Bearer token (current default)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_character
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        
        # Verify Authorization header format
        call_args = mock_get.call_args
        self.assertTrue(any('Authorization' in str(arg) for arg in call_args))
    
    @patch('requests.Session.get')
    def test_expired_token_handling(self, mock_get):
        """Test handling of expired authentication tokens."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Token expired"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Authentication failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_forbidden_access_handling(self, mock_get):
        """Test handling of forbidden access (403)."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Insufficient permissions"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Access forbidden", str(context.exception))
    
    # Session Management Tests
    def test_session_reuse(self):
        """Test that the same session is reused across requests."""
        initial_session = self.client.session
        
        # Create another client with same parameters
        client2 = CharacterClient(base_url="https://api.test.com", api_key="test_key")
        
        # Sessions should be different instances
        self.assertIsNot(initial_session, client2.session)
        
        # But the original client should keep its session
        self.assertIs(initial_session, self.client.session)
        
        client2.session.close()
    
    def test_session_headers_immutability(self):
        """Test that session headers are properly set and maintained."""
        original_headers = dict(self.client.session.headers)
        
        # Try to modify headers externally
        self.client.session.headers['X-Custom'] = 'test'
        
        # Verify the custom header was added
        self.assertEqual(self.client.session.headers['X-Custom'], 'test')
        
        # But original headers should remain
        for key, value in original_headers.items():
            self.assertEqual(self.client.session.headers[key], value)
    
    @patch('requests.Session.close')
    def test_session_cleanup(self, mock_close):
        """Test that session is properly closed during cleanup."""
        client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
        
        # Manually close the session
        client.session.close()
        
        mock_close.assert_called_once()
    
    # Rate Limiting and Retry Logic Tests
    @patch('requests.Session.get')
    def test_rate_limit_with_retry_after_header(self, mock_get):
        """Test rate limiting with Retry-After header."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        
        # Should include retry information
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    @patch('requests.Session.get')
    def test_multiple_consecutive_rate_limits(self, mock_get):
        """Test handling of multiple consecutive rate limit responses."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        # Multiple calls should all fail with rate limit
        for _ in range(3):
            with self.assertRaises(CharacterClientError):
                self.client.get_character(1)
    
    # Data Integrity and Validation Tests
    def test_character_data_with_null_values(self):
        """Test character data validation with null/None values."""
        character_data_with_nulls = {
            "name": "Test Character",
            "class": "Warrior",
            "level": 1,
            "description": None,
            "attributes": None
        }
        
        # Should pass validation (null values may be acceptable for optional fields)
        result = self.client._validate_character_data(character_data_with_nulls)
        self.assertTrue(result)
    
    def test_character_data_with_nested_objects(self):
        """Test character data validation with complex nested structures."""
        complex_character_data = {
            "name": "Complex Character",
            "class": "Multiclass",
            "level": 15,
            "attributes": {
                "primary": {"strength": 18, "intelligence": 14},
                "secondary": {"charisma": 12, "wisdom": 10},
                "skills": ["combat", "magic", "diplomacy"]
            },
            "equipment": [
                {"type": "weapon", "name": "Excalibur", "enchantments": ["fire", "holy"]},
                {"type": "armor", "name": "Plate Mail", "enchantments": ["protection"]}
            ]
        }
        
        # Should handle complex nested data
        result = self.client._validate_character_data(complex_character_data)
        self.assertTrue(result)
    
    def test_character_data_with_extremely_large_numbers(self):
        """Test character data with very large numeric values."""
        large_number_data = {
            "name": "Overpowered Character",
            "class": "God",
            "level": 999999999,
            "health": 2**63 - 1,
            "mana": 9999999999999999
        }
        
        result = self.client._validate_character_data(large_number_data)
        self.assertTrue(result)
    
    # Error Recovery and Resilience Tests
    @patch('requests.Session.get')
    def test_intermittent_network_failures(self, mock_get):
        """Test handling of intermittent network failures."""
        # Simulate intermittent failures
        mock_get.side_effect = [
            requests.ConnectionError("Temporary network issue"),
            requests.Timeout("Request timeout"),
            requests.ConnectionError("Another network issue")
        ]
        
        # All calls should raise CharacterClientError
        for _ in range(3):
            with self.assertRaises(CharacterClientError):
                self.client.get_character(1)
    
    @patch('requests.Session.get')
    def test_malformed_response_headers(self, mock_get):
        """Test handling of responses with malformed headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_character
        mock_response.headers = {"Content-Type": "application/json; charset=utf-8; boundary=something"}
        mock_get.return_value = mock_response
        
        # Should still work despite unusual headers
        result = self.client.get_character(1)
        self.assertEqual(result, self.sample_character)
    
    @patch('requests.Session.get')
    def test_response_with_bom(self, mock_get):
        """Test handling of responses with Byte Order Mark (BOM)."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Simulate response with BOM that might cause JSON parsing issues
        mock_response.json.return_value = self.sample_character
        mock_response.text = '\ufeff{"id": 1, "name": "Test"}'  # BOM + JSON
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result, self.sample_character)
    
    # Performance and Stress Testing Scenarios
    def test_large_character_list_handling(self):
        """Test handling of very large character lists."""
        # Create a large list of characters
        large_character_list = []
        for i in range(1000):
            large_character_list.append({
                "id": i,
                "name": f"Character {i}",
                "class": "Warrior" if i % 2 == 0 else "Mage",
                "level": i % 100 + 1
            })
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"characters": large_character_list}
            mock_get.return_value = mock_response
            
            result = self.client.get_characters()
            
            self.assertEqual(len(result), 1000)
            self.assertEqual(result[0]["name"], "Character 0")
            self.assertEqual(result[-1]["name"], "Character 999")
    
    def test_batch_operations_with_mixed_results(self):
        """Test batch operations with mixed success/failure results."""
        character_ids = list(range(1, 11))  # IDs 1-10
        
        def mock_get_character(char_id):
            if char_id in [3, 7, 9]:  # Some characters don't exist
                raise CharacterNotFoundError(f"Character {char_id} not found")
            elif char_id == 5:  # One causes a server error
                raise CharacterClientError("Server error")
            else:
                return {"id": char_id, "name": f"Character {char_id}"}
        
        with patch.object(self.client, 'get_character', side_effect=mock_get_character):
            results = self.client.get_characters_batch(character_ids)
            
            # Should return only successful results (6 out of 10)
            self.assertEqual(len(results), 6)
            successful_ids = [r["id"] for r in results]
            self.assertEqual(sorted(successful_ids), [1, 2, 4, 6, 8, 10])
    
    # Input Sanitization and Security Tests
    def test_sql_injection_patterns_in_character_names(self):
        """Test character names with SQL injection patterns."""
        malicious_names = [
            "'; DROP TABLE characters; --",
            "Robert'); DROP TABLE students;--",
            "' OR '1'='1",
            "'; INSERT INTO characters (name) VALUES ('hacked'); --",
            "' UNION SELECT * FROM users --"
        ]
        
        for name in malicious_names:
            with self.subTest(name=name):
                character_data = {"name": name, "class": "Rogue", "level": 1}
                # Should not raise validation errors (sanitization handled by API)
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_xss_patterns_in_character_data(self):
        """Test character data with XSS patterns."""
        xss_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert(String.fromCharCode(88,83,83))//",
            "<svg onload=alert('xss')>"
        ]
        
        for pattern in xss_patterns:
            with self.subTest(pattern=pattern):
                character_data = {"name": pattern, "class": "Hacker", "level": 1}
                # Should not raise validation errors (sanitization handled by API)
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_extremely_long_api_key(self):
        """Test initialization with extremely long API key."""
        long_api_key = "a" * 10000  # Very long API key
        
        # Should not raise errors during initialization
        client = CharacterClient(base_url="https://api.test.com", api_key=long_api_key)
        self.assertEqual(client.api_key, long_api_key)
        client.session.close()
    
    # Concurrent Operations Tests
    @patch('requests.Session.get')
    def test_concurrent_character_requests(self, mock_get):
        """Test concurrent character retrieval requests."""
        import threading
        import time
        
        results = {}
        errors = {}
        
        def get_character_threaded(char_id):
            try:
                # Simulate varying response times
                time.sleep(0.01 * (char_id % 5))
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"id": char_id, "name": f"Character {char_id}"}
                mock_get.return_value = mock_response
                
                result = self.client.get_character(char_id)
                results[char_id] = result
            except Exception as e:
                errors[char_id] = str(e)
        
        # Create multiple threads
        threads = []
        for i in range(1, 6):
            thread = threading.Thread(target=get_character_threaded, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        for i in range(1, 6):
            self.assertIn(i, results)
            self.assertEqual(results[i]["id"], i)
    
    # Memory and Resource Management Tests
    def test_memory_usage_with_large_responses(self):
        """Test memory handling with large API responses."""
        # Create a character with large data
        large_character = {
            "id": 1,
            "name": "Large Character",
            "class": "DataWarrior",
            "level": 1,
            "description": "A" * 100000,  # Large description
            "history": ["Event " + str(i) for i in range(10000)],  # Large history
            "attributes": {f"attr_{i}": i for i in range(1000)}  # Many attributes
        }
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = large_character
            mock_get.return_value = mock_response
            
            result = self.client.get_character(1)
            
            self.assertEqual(result["id"], 1)
            self.assertEqual(len(result["description"]), 100000)
            self.assertEqual(len(result["history"]), 10000)
    
    def test_session_connection_pooling(self):
        """Test session connection pooling behavior."""
        # Multiple clients should have separate sessions
        clients = []
        for i in range(5):
            client = CharacterClient(
                base_url=f"https://api{i}.test.com", 
                api_key=f"key_{i}"
            )
            clients.append(client)
        
        # All sessions should be different
        sessions = [client.session for client in clients]
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                self.assertIsNot(sessions[i], sessions[j])
        
        # Clean up
        for client in clients:
            client.session.close()
    
    # Additional Edge Cases
    def test_url_with_non_standard_ports(self):
        """Test client with non-standard ports in base URL."""
        non_standard_urls = [
            "https://api.test.com:8080",
            "http://localhost:3000",
            "https://api.test.com:443",  # Standard HTTPS port
            "http://api.test.com:80"     # Standard HTTP port
        ]
        
        for url in non_standard_urls:
            with self.subTest(url=url):
                client = CharacterClient(base_url=url, api_key="test_key")
                self.assertEqual(client.base_url, url)
                client.session.close()
    
    def test_character_class_with_special_cases(self):
        """Test character classes with special formatting."""
        special_classes = [
            "Multi-Class",
            "Spellsword/Battlemage",
            "Monk (Way of Shadow)",
            "Paladin - Oath of Vengeance",
            "Artificer: Alchemist",
            "Warlock (Fiend Patron)"
        ]
        
        for char_class in special_classes:
            with self.subTest(char_class=char_class):
                character_data = {
                    "name": "Test Character",
                    "class": char_class,
                    "level": 1
                }
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    @patch('requests.Session.get')
    def test_response_with_extra_fields(self, mock_get):
        """Test handling of API responses with extra unexpected fields."""
        character_with_extras = {
            **self.sample_character,
            "unexpected_field": "unexpected_value",
            "api_version": "2.1.0",
            "server_timestamp": "2023-12-01T10:00:00Z",
            "debug_info": {"query_time": "0.05s", "cache_hit": False}
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = character_with_extras
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        
        # Should return the full response including extra fields
        self.assertEqual(result, character_with_extras)
        self.assertIn("unexpected_field", result)
        self.assertIn("debug_info", result)
    
    def test_api_key_with_special_characters(self):
        """Test API keys containing special characters."""
        special_api_keys = [
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key+with+plus",
            "key=with=equals",
            "key/with/slashes",
            "key@with@symbols"
        ]
        
        for api_key in special_api_keys:
            with self.subTest(api_key=api_key):
                client = CharacterClient(base_url="https://api.test.com", api_key=api_key)
                self.assertEqual(client.api_key, api_key)
                
                # Verify the authorization header is set correctly
                expected_header = f"Bearer {api_key}"
                self.assertEqual(client.session.headers['Authorization'], expected_header)
                
                client.session.close()


class TestCharacterClientIntegration(unittest.TestCase):
    """Integration-style tests that test multiple components working together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    @patch('requests.Session.post')
    @patch('requests.Session.get')
    def test_create_then_retrieve_character_workflow(self, mock_get, mock_post):
        """Test the complete workflow of creating then retrieving a character."""
        # Setup create response
        character_data = {"name": "New Hero", "class": "Paladin", "level": 1}
        created_character = {**character_data, "id": 100}
        
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = created_character
        mock_post.return_value = mock_post_response
        
        # Setup get response
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = created_character
        mock_get.return_value = mock_get_response
        
        # Execute workflow
        created = self.client.create_character(character_data)
        retrieved = self.client.get_character(created["id"])
        
        # Verify results
        self.assertEqual(created["name"], "New Hero")
        self.assertEqual(retrieved["name"], "New Hero")
        self.assertEqual(created["id"], retrieved["id"])
        
        # Verify call sequence
        mock_post.assert_called_once()
        mock_get.assert_called_once_with(
            "https://api.test.com/characters/100",
            timeout=30
        )
    
    @patch('requests.Session.put')
    @patch('requests.Session.get')
    def test_retrieve_update_retrieve_workflow(self, mock_get, mock_put):
        """Test retrieving, updating, then retrieving a character again."""
        original_character = {
            "id": 1,
            "name": "Hero",
            "class": "Warrior",
            "level": 5,
            "health": 50
        }
        
        updated_character = {
            **original_character,
            "level": 6,
            "health": 60
        }
        
        # Setup responses
        mock_get.side_effect = [
            Mock(status_code=200, json=Mock(return_value=original_character)),
            Mock(status_code=200, json=Mock(return_value=updated_character))
        ]
        
        mock_put_response = Mock()
        mock_put_response.status_code = 200
        mock_put_response.json.return_value = updated_character
        mock_put.return_value = mock_put_response
        
        # Execute workflow
        before_update = self.client.get_character(1)
        update_result = self.client.update_character(1, {"level": 6, "health": 60})
        after_update = self.client.get_character(1)
        
        # Verify results
        self.assertEqual(before_update["level"], 5)
        self.assertEqual(update_result["level"], 6)
        self.assertEqual(after_update["level"], 6)
        
        # Verify call counts
        self.assertEqual(mock_get.call_count, 2)
        mock_put.assert_called_once()
    
    @patch('requests.Session.delete')
    @patch('requests.Session.get')
    def test_delete_then_retrieve_workflow(self, mock_get, mock_delete):
        """Test deleting a character then trying to retrieve it."""
        # Setup delete response
        mock_delete_response = Mock()
        mock_delete_response.status_code = 204
        mock_delete.return_value = mock_delete_response
        
        # Setup get response (should return 404 after deletion)
        mock_get_response = Mock()
        mock_get_response.status_code = 404
        mock_get.return_value = mock_get_response
        
        # Execute workflow
        delete_result = self.client.delete_character(1)
        
        # Verify deletion succeeded
        self.assertTrue(delete_result)
        
        # Verify subsequent retrieval fails
        with self.assertRaises(CharacterNotFoundError):
            self.client.get_character(1)
        
        # Verify calls
        mock_delete.assert_called_once()
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_pagination_workflow(self, mock_get):
        """Test paginated character retrieval workflow."""
        # Setup responses for multiple pages
        page1_characters = [{"id": i, "name": f"Character {i}"} for i in range(1, 6)]
        page2_characters = [{"id": i, "name": f"Character {i}"} for i in range(6, 11)]
        
        mock_get.side_effect = [
            Mock(status_code=200, json=Mock(return_value={"characters": page1_characters})),
            Mock(status_code=200, json=Mock(return_value={"characters": page2_characters})),
            Mock(status_code=200, json=Mock(return_value={"characters": []}))  # Empty third page
        ]
        
        # Execute workflow
        page1 = self.client.get_characters(page=1, limit=5)
        page2 = self.client.get_characters(page=2, limit=5)
        page3 = self.client.get_characters(page=3, limit=5)
        
        # Verify results
        self.assertEqual(len(page1), 5)
        self.assertEqual(len(page2), 5)
        self.assertEqual(len(page3), 0)
        
        self.assertEqual(page1[0]["id"], 1)
        self.assertEqual(page2[0]["id"], 6)
        
        # Verify call count
        self.assertEqual(mock_get.call_count, 3)


# Add more stress tests for boundary conditions
class TestCharacterClientStressBoundary(unittest.TestCase):
    """Test boundary conditions and stress scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://api.test.com", api_key="test_key")
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client, 'session') and self.client.session:
            self.client.session.close()
    
    def test_maximum_integer_boundaries(self):
        """Test with maximum integer values."""
        import sys
        
        max_int = sys.maxsize
        character_data = {
            "name": "Max Integer Character",
            "class": "Boundary Tester",
            "level": max_int,
            "health": max_int,
            "mana": max_int
        }
        
        # Should handle very large integers
        result = self.client._validate_character_data(character_data)
        self.assertTrue(result)
    
    def test_float_precision_edge_cases(self):
        """Test floating point precision edge cases."""
        character_data = {
            "name": "Float Precision Test",
            "class": "Mathematician",
            "level": 1,
            "experience_multiplier": 1.7976931348623157e+308,  # Near max float
            "critical_chance": 2.2250738585072014e-308,  # Near min positive float
            "dodge_rate": 0.9999999999999999  # High precision float
        }
        
        result = self.client._validate_character_data(character_data)
        self.assertTrue(result)
    
    def test_deeply_nested_character_data(self):
        """Test with deeply nested character data structures."""
        def create_nested_dict(depth):
            if depth == 0:
                return {"value": "deep_value"}
            return {"level": depth, "nested": create_nested_dict(depth - 1)}
        
        deep_data = {
            "name": "Deeply Nested Character",
            "class": "Recursive",
            "level": 1,
            "deep_attributes": create_nested_dict(100)  # 100 levels deep
        }
        
        # Should handle deeply nested structures
        result = self.client._validate_character_data(deep_data)
        self.assertTrue(result)
    
    @patch('requests.Session.get')
    def test_response_time_variation(self, mock_get):
        """Test handling of responses with varying delays."""
        import time
        
        def delayed_response(*args, **kwargs):
            # Simulate variable response times
            time.sleep(0.1)  # 100ms delay
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1, "name": "Delayed Character"}
            return mock_response
        
        mock_get.side_effect = delayed_response
        
        start_time = time.time()
        result = self.client.get_character(1)
        end_time = time.time()
        
        # Should handle the delay and return correct result
        self.assertEqual(result["id"], 1)
        self.assertGreaterEqual(end_time - start_time, 0.1)  # At least 100ms
    
    def test_character_batch_size_limits(self):
        """Test batch operations with various sizes."""
        # Test small batch
        small_batch = list(range(1, 6))  # 5 characters
        
        # Test medium batch
        medium_batch = list(range(1, 51))  # 50 characters
        
        # Test large batch
        large_batch = list(range(1, 1001))  # 1000 characters
        
        for batch_name, batch_ids in [("small", small_batch), ("medium", medium_batch), ("large", large_batch)]:
            with self.subTest(batch=batch_name):
                with patch.object(self.client, 'get_character') as mock_get:
                    mock_get.side_effect = lambda char_id: {"id": char_id, "name": f"Character {char_id}"}
                    
                    results = self.client.get_characters_batch(batch_ids)
                    
                    self.assertEqual(len(results), len(batch_ids))
                    self.assertEqual(mock_get.call_count, len(batch_ids))


if __name__ == '__main__':
    # Run all test classes
    test_classes = [
        TestCharacterClient,
        TestCharacterClientEdgeCases,
        TestCharacterClientAdvanced,
        TestCharacterClientIntegration,
        TestCharacterClientStressBoundary
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    runner.run(suite)