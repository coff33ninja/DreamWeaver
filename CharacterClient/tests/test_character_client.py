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
    """Advanced test scenarios for CharacterClient including authentication, retries, and performance edge cases."""
    
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
    
    # Authentication and Authorization Tests
    @patch('requests.Session.get')
    def test_authentication_with_invalid_token_format(self, mock_get):
        """Test authentication with malformed token format."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid token format"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(AuthenticationError) as context:
            self.client.get_character(1)
        self.assertIn("Authentication failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_authentication_with_expired_token(self, mock_get):
        """Test authentication with expired token."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Token expired"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(AuthenticationError) as context:
            self.client.get_character(1)
        self.assertIn("Authentication failed", str(context.exception))
    
    @patch('requests.Session.get')
    def test_forbidden_access_error(self, mock_get):
        """Test handling of 403 Forbidden responses."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Insufficient permissions"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Forbidden", str(context.exception))
    
    # Rate Limiting Tests
    @patch('requests.Session.get')
    def test_rate_limit_with_retry_after_header(self, mock_get):
        """Test rate limiting with Retry-After header."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(RateLimitError) as context:
            self.client.get_character(1)
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    @patch('requests.Session.get')
    def test_rate_limit_without_retry_header(self, mock_get):
        """Test rate limiting without Retry-After header."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": "Too many requests"}
        mock_get.return_value = mock_response
        
        with self.assertRaises(RateLimitError) as context:
            self.client.get_character(1)
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    # HTTP Headers and Content-Type Tests
    @patch('requests.Session.get')
    def test_response_with_different_content_types(self, mock_get):
        """Test handling responses with different content types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/plain'}
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_response.text = "Plain text response"
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Invalid JSON response", str(context.exception))
    
    def test_custom_headers_in_session(self):
        """Test that custom headers are properly set in session."""
        custom_client = CharacterClient(
            base_url="https://test.api.com", 
            api_key="test_key",
            custom_headers={'X-Custom-Header': 'custom-value'}
        )
        
        expected_headers = {
            'Authorization': 'Bearer test_key',
            'Content-Type': 'application/json',
            'User-Agent': 'CharacterClient/1.0',
            'X-Custom-Header': 'custom-value'
        }
        
        for key, value in expected_headers.items():
            self.assertEqual(custom_client.session.headers[key], value)
    
    # Connection and Network Error Tests
    @patch('requests.Session.get')
    def test_ssl_error_handling(self, mock_get):
        """Test handling of SSL certificate errors."""
        mock_get.side_effect = requests.exceptions.SSLError("SSL certificate verify failed")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("SSL error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_dns_resolution_error(self, mock_get):
        """Test handling of DNS resolution errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to resolve hostname")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_proxy_error_handling(self, mock_get):
        """Test handling of proxy connection errors."""
        mock_get.side_effect = requests.exceptions.ProxyError("Cannot connect to proxy")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    # Response Size and Performance Tests
    @patch('requests.Session.get')
    def test_large_response_handling(self, mock_get):
        """Test handling of very large responses."""
        # Simulate a large character object
        large_character = self.sample_character.copy()
        large_character['description'] = 'A' * 10000  # Large description
        large_character['inventory'] = [{'item': f'Item{i}'} for i in range(1000)]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_character
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result['id'], 1)
        self.assertEqual(len(result['description']), 10000)
        self.assertEqual(len(result['inventory']), 1000)
    
    @patch('requests.Session.get')
    def test_empty_response_body(self, mock_get):
        """Test handling of empty response body."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result, {})
    
    # Concurrent Request Simulation
    @patch('requests.Session.get')
    def test_concurrent_character_requests_simulation(self, mock_get):
        """Test simulation of concurrent character requests."""
        mock_responses = []
        for i in range(5):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": i+1, "name": f"Character {i+1}"}
            mock_responses.append(mock_response)
        
        mock_get.side_effect = mock_responses
        
        # Simulate concurrent requests
        results = []
        for i in range(5):
            result = self.client.get_character(i+1)
            results.append(result)
        
        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertEqual(result['id'], i+1)
    
    # Advanced Validation Tests
    def test_character_data_with_nested_objects(self):
        """Test character data validation with complex nested structures."""
        complex_character_data = {
            "name": "Complex Character",
            "class": "Paladin",
            "level": 15,
            "attributes": {
                "strength": 18,
                "dexterity": 14,
                "constitution": 16,
                "intelligence": 12,
                "wisdom": 15,
                "charisma": 17
            },
            "equipment": {
                "weapon": {"name": "Holy Sword", "damage": 15, "enchantments": ["holy", "sharp"]},
                "armor": {"name": "Plate Mail", "defense": 20, "weight": 50},
                "accessories": [
                    {"name": "Ring of Protection", "bonus": 2},
                    {"name": "Amulet of Health", "bonus": 5}
                ]
            },
            "spells": [
                {"name": "Heal", "level": 2, "mana_cost": 10},
                {"name": "Smite", "level": 3, "mana_cost": 15}
            ]
        }
        
        result = self.client._validate_character_data(complex_character_data)
        self.assertTrue(result)
    
    def test_character_data_with_null_values(self):
        """Test character data validation with null values."""
        character_with_nulls = {
            "name": "Test Character",
            "class": "Ranger",
            "level": 5,
            "description": None,
            "backstory": None,
            "guild": None
        }
        
        result = self.client._validate_character_data(character_with_nulls)
        self.assertTrue(result)
    
    # Status Code Edge Cases
    @patch('requests.Session.get')
    def test_uncommon_success_status_codes(self, mock_get):
        """Test handling of uncommon but valid success status codes."""
        success_codes = [200, 201, 202, 204]
        
        for status_code in success_codes:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                if status_code == 204:  # No Content
                    mock_response.json.return_value = {}
                else:
                    mock_response.json.return_value = self.sample_character
                mock_get.return_value = mock_response
                
                if status_code == 204:
                    result = self.client.get_character(1)
                    self.assertEqual(result, {})
                else:
                    result = self.client.get_character(1)
                    self.assertIsInstance(result, dict)
    
    @patch('requests.Session.get')
    def test_client_error_status_codes(self, mock_get):
        """Test handling of various client error status codes."""
        error_codes = {
            400: "Bad Request",
            402: "Payment Required",
            405: "Method Not Allowed",
            406: "Not Acceptable",
            408: "Request Timeout",
            409: "Conflict",
            410: "Gone",
            413: "Payload Too Large",
            415: "Unsupported Media Type",
            422: "Unprocessable Entity"
        }
        
        for status_code, error_message in error_codes.items():
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.json.return_value = {"error": error_message}
                mock_get.return_value = mock_response
                
                with self.assertRaises(CharacterClientError):
                    self.client.get_character(1)
    
    @patch('requests.Session.get')
    def test_server_error_status_codes(self, mock_get):
        """Test handling of various server error status codes."""
        server_error_codes = [500, 501, 502, 503, 504, 505]
        
        for status_code in server_error_codes:
            with self.subTest(status_code=status_code):
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.json.return_value = {"error": "Server error"}
                mock_get.return_value = mock_response
                
                with self.assertRaises(CharacterClientError) as context:
                    self.client.get_character(1)
                self.assertIn(f"API error: {status_code}", str(context.exception))
    
    # Session Management Tests
    def test_session_closure_on_context_exit(self):
        """Test that session is properly closed when using context manager."""
        with CharacterClient(base_url="https://test.api.com", api_key="test_key") as client:
            self.assertIsNotNone(client.session)
            session = client.session
        
        # Session should be closed after exiting context
        self.assertTrue(session.adapters == {})  # Adapters are cleared when session is closed
    
    @patch('requests.Session.close')
    def test_explicit_session_close(self, mock_close):
        """Test explicit session closure."""
        client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
        client.close()
        mock_close.assert_called_once()
    
    # URL Building Edge Cases
    def test_build_url_with_special_characters_in_params(self):
        """Test URL building with special characters in parameters."""
        params = {
            "name": "Sir Lancelot & the Knights",
            "class": "Paladin/Warrior",
            "description": "A noble knight with 100% dedication"
        }
        
        result = self.client._build_url("characters", params=params)
        
        # URL should be properly encoded
        self.assertIn("characters", result)
        # Special characters should be URL encoded
        self.assertIn("%26", result)  # & encoded
        self.assertIn("%2F", result)  # / encoded
        self.assertIn("%25", result)  # % encoded
    
    def test_build_url_with_unicode_params(self):
        """Test URL building with Unicode parameters."""
        params = {
            "name": "魔法师",
            "description": "Très puissant magicien"
        }
        
        result = self.client._build_url("characters", params=params)
        self.assertIn("characters", result)
        # Should not raise encoding errors
        self.assertIsInstance(result, str)
    
    # Pagination Edge Cases
    @patch('requests.Session.get')
    def test_get_characters_boundary_pagination(self, mock_get):
        """Test characters retrieval with boundary pagination values."""
        test_cases = [
            (1, 1),      # Minimum values
            (1, 100),    # Maximum limit
            (999, 50),   # High page number
        ]
        
        for page, limit in test_cases:
            with self.subTest(page=page, limit=limit):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "characters": [self.sample_character],
                    "page": page,
                    "limit": limit,
                    "total": 1000
                }
                mock_get.return_value = mock_response
                
                result = self.client.get_characters(page=page, limit=limit)
                self.assertIsInstance(result, list)
                
                # Verify pagination parameters were sent correctly
                call_args = mock_get.call_args
                self.assertEqual(call_args[1]['params']['page'], page)
                self.assertEqual(call_args[1]['params']['limit'], limit)
    
    # Data Type Conversion Tests
    @patch('requests.Session.get')
    def test_character_with_numeric_strings(self, mock_get):
        """Test character data with numeric values as strings."""
        character_with_string_numbers = {
            "id": "123",  # String ID
            "name": "Test Character",
            "level": "15",  # String level
            "health": "100.5",  # String health with decimal
            "created_at": "2023-01-01T00:00:00Z"
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = character_with_string_numbers
        mock_get.return_value = mock_response
        
        result = self.client.get_character(1)
        self.assertEqual(result["id"], "123")
        self.assertEqual(result["level"], "15")
        self.assertEqual(result["health"], "100.5")
    
    # Timeout Configuration Tests
    def test_custom_timeout_in_requests(self):
        """Test that custom timeout is applied to requests."""
        custom_timeout = 45
        client = CharacterClient(
            base_url="https://test.api.com",
            api_key="test_key",
            timeout=custom_timeout
        )
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = self.sample_character
            mock_get.return_value = mock_response
            
            client.get_character(1)
            
            # Verify timeout was passed to the request
            call_args = mock_get.call_args
            self.assertEqual(call_args[1]['timeout'], custom_timeout)
    
    @patch('requests.Session.get')
    def test_request_timeout_with_slow_response(self, mock_get):
        """Test request timeout with intentionally slow response."""
        mock_get.side_effect = requests.exceptions.ReadTimeout("Request timed out")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))


class TestCharacterClientDataValidation(unittest.TestCase):
    """Comprehensive data validation tests for CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    def test_validate_character_class_types(self):
        """Test validation of different character class types."""
        valid_classes = [
            "Warrior", "Mage", "Rogue", "Paladin", "Ranger", "Barbarian",
            "Cleric", "Druid", "Sorcerer", "Warlock", "Fighter", "Wizard",
            "Monk", "Bard", "Artificer"
        ]
        
        for char_class in valid_classes:
            with self.subTest(char_class=char_class):
                character_data = {
                    "name": f"Test {char_class}",
                    "class": char_class,
                    "level": 1
                }
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_validate_character_level_extremes(self):
        """Test validation with extreme level values."""
        extreme_levels = [1, 50, 100, 999, 9999]
        
        for level in extreme_levels:
            with self.subTest(level=level):
                character_data = {
                    "name": "Test Character",
                    "class": "Warrior",
                    "level": level
                }
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_validate_character_with_optional_fields(self):
        """Test validation with various optional fields."""
        optional_fields_data = {
            "name": "Comprehensive Character",
            "class": "Paladin",
            "level": 10,
            "race": "Human",
            "alignment": "Lawful Good",
            "background": "Noble",
            "health": 85,
            "mana": 40,
            "experience": 15000,
            "gold": 250,
            "location": "Capital City",
            "guild": "Heroes Guild"
        }
        
        result = self.client._validate_character_data(optional_fields_data)
        self.assertTrue(result)
    
    def test_validate_character_name_edge_cases(self):
        """Test character name validation with edge cases."""
        edge_case_names = [
            "A",  # Single character
            "AB",  # Two characters
            "X" * 50,  # Long name
            "Character-123",  # With numbers and hyphens
            "Sir John III",  # With roman numerals
            "Lady Éleanor",  # With accented characters
            "Røgüe Wärrîör",  # Multiple accented characters
        ]
        
        for name in edge_case_names:
            with self.subTest(name=name):
                character_data = {
                    "name": name,
                    "class": "Warrior",
                    "level": 1
                }
                result = self.client._validate_character_data(character_data)
                self.assertTrue(result)
    
    def test_validate_character_with_boolean_fields(self):
        """Test validation with boolean fields."""
        character_with_booleans = {
            "name": "Boolean Character",
            "class": "Mage",
            "level": 5,
            "is_npc": False,
            "is_alive": True,
            "has_mount": True,
            "is_guild_leader": False
        }
        
        result = self.client._validate_character_data(character_with_booleans)
        self.assertTrue(result)
    
    def test_validate_character_with_array_fields(self):
        """Test validation with array/list fields."""
        character_with_arrays = {
            "name": "Array Character",
            "class": "Ranger",
            "level": 8,
            "skills": ["Archery", "Tracking", "Survival"],
            "languages": ["Common", "Elvish", "Draconic"],
            "inventory": [
                {"item": "Bow", "quantity": 1},
                {"item": "Arrows", "quantity": 50},
                {"item": "Healing Potion", "quantity": 3}
            ]
        }
        
        result = self.client._validate_character_data(character_with_arrays)
        self.assertTrue(result)


class TestCharacterClientErrorHandling(unittest.TestCase):
    """Comprehensive error handling tests for CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(base_url="https://test.api.com", api_key="test_key")
    
    @patch('requests.Session.get')
    def test_json_decode_error_handling(self, mock_get):
        """Test handling of JSON decode errors with detailed error information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Expecting ',' delimiter", "doc", 10)
        mock_response.text = "Invalid JSON response"
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        
        error_message = str(context.exception)
        self.assertIn("Invalid JSON response", error_message)
    
    @patch('requests.Session.get')
    def test_response_encoding_error(self, mock_get):
        """Test handling of response encoding errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
        mock_get.return_value = mock_response
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Response encoding error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_connection_pool_error(self, mock_get):
        """Test handling of connection pool errors."""
        mock_get.side_effect = requests.exceptions.HTTPError("Connection pool is full")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("HTTP error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_chunk_encoding_error(self, mock_get):
        """Test handling of chunk encoding errors."""
        mock_get.side_effect = requests.exceptions.ChunkedEncodingError("Connection broken: Invalid chunk encoding")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_redirect_error_handling(self, mock_get):
        """Test handling of too many redirects error."""
        mock_get.side_effect = requests.exceptions.TooManyRedirects("Exceeded 30 redirects")
        
        with self.assertRaises(CharacterClientError) as context:
            self.client.get_character(1)
        self.assertIn("Network error", str(context.exception))
    
    def test_invalid_character_data_types(self):
        """Test validation with invalid data types for character fields."""
        invalid_data_cases = [
            # Invalid name types
            {"name": 123, "class": "Warrior", "level": 1},
            {"name": [], "class": "Warrior", "level": 1},
            {"name": {}, "class": "Warrior", "level": 1},
            
            # Invalid class types
            {"name": "Test", "class": 123, "level": 1},
            {"name": "Test", "class": [], "level": 1},
            
            # Invalid level types
            {"name": "Test", "class": "Warrior", "level": "not_a_number"},
            {"name": "Test", "class": "Warrior", "level": []},
            {"name": "Test", "class": "Warrior", "level": {}},
        ]
        
        for invalid_data in invalid_data_cases:
            with self.subTest(invalid_data=invalid_data):
                with self.assertRaises((ValueError, TypeError)):
                    self.client._validate_character_data(invalid_data)


class TestCharacterClientRetryMechanisms(unittest.TestCase):
    """Test retry mechanisms and resilience features of CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(
            base_url="https://test.api.com", 
            api_key="test_key",
            max_retries=3,
            retry_backoff=1.0
        )
    
    @patch('requests.Session.get')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_on_temporary_server_error(self, mock_sleep, mock_get):
        """Test retry mechanism on temporary server errors."""
        # First two calls fail, third succeeds
        mock_responses = [
            Mock(status_code=503),  # Service unavailable
            Mock(status_code=502),  # Bad gateway
            Mock(status_code=200, json=lambda: {"id": 1, "name": "Test"})  # Success
        ]
        mock_get.side_effect = mock_responses
        
        result = self.client.get_character(1)
        
        self.assertEqual(result["id"], 1)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Two retries
    
    @patch('requests.Session.get')
    @patch('time.sleep')
    def test_retry_exhaustion_raises_error(self, mock_sleep, mock_get):
        """Test that retries are exhausted and error is raised."""
        mock_get.side_effect = [Mock(status_code=503)] * 4  # All calls fail
        
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
        
        self.assertEqual(mock_get.call_count, 4)  # Initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)
    
    @patch('requests.Session.get')
    def test_no_retry_on_client_errors(self, mock_get):
        """Test that client errors (4xx) are not retried."""
        mock_get.return_value = Mock(status_code=400)  # Bad request
        
        with self.assertRaises(ValidationError):
            self.client.get_character(1)
        
        self.assertEqual(mock_get.call_count, 1)  # No retries
    
    @patch('requests.Session.get')
    def test_exponential_backoff_calculation(self, mock_get):
        """Test exponential backoff timing calculation."""
        with patch('time.sleep') as mock_sleep:
            mock_get.side_effect = [Mock(status_code=503)] * 4
            
            with self.assertRaises(CharacterClientError):
                self.client.get_character(1)
            
            # Verify exponential backoff: 1s, 2s, 4s
            expected_delays = [1.0, 2.0, 4.0]
            actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
            self.assertEqual(actual_delays, expected_delays)


class TestCharacterClientCaching(unittest.TestCase):
    """Test caching mechanisms in CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(
            base_url="https://test.api.com", 
            api_key="test_key",
            enable_caching=True,
            cache_ttl=300  # 5 minutes
        )
    
    @patch('requests.Session.get')
    def test_character_caching_on_repeated_requests(self, mock_get):
        """Test that repeated character requests use cache."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test Character"}
        mock_get.return_value = mock_response
        
        # First request
        result1 = self.client.get_character(1)
        # Second request (should use cache)
        result2 = self.client.get_character(1)
        
        self.assertEqual(result1, result2)
        self.assertEqual(mock_get.call_count, 1)  # Only one actual HTTP call
    
    @patch('requests.Session.get')
    def test_cache_invalidation_on_update(self, mock_get):
        """Test that cache is invalidated when character is updated."""
        # Setup initial character
        get_response = Mock()
        get_response.status_code = 200
        get_response.json.return_value = {"id": 1, "name": "Original Name"}
        
        # Setup update response
        update_response = Mock()
        update_response.status_code = 200
        update_response.json.return_value = {"id": 1, "name": "Updated Name"}
        
        mock_get.return_value = get_response
        
        # Get character (cached)
        result1 = self.client.get_character(1)
        self.assertEqual(result1["name"], "Original Name")
        
        # Update character (should invalidate cache)
        with patch('requests.Session.put', return_value=update_response):
            self.client.update_character(1, {"name": "Updated Name"})
        
        # Get character again (should fetch fresh data)
        mock_get.return_value = update_response
        result2 = self.client.get_character(1)
        
        self.assertEqual(result2["name"], "Updated Name")
        self.assertEqual(mock_get.call_count, 2)  # Two HTTP calls due to cache invalidation
    
    def test_cache_key_generation(self):
        """Test cache key generation for different requests."""
        key1 = self.client._generate_cache_key("GET", "/characters/1", {})
        key2 = self.client._generate_cache_key("GET", "/characters/2", {})
        key3 = self.client._generate_cache_key("GET", "/characters/1", {"include": "stats"})
        
        self.assertNotEqual(key1, key2)  # Different character IDs
        self.assertNotEqual(key1, key3)  # Different parameters
        self.assertIsInstance(key1, str)
        self.assertTrue(len(key1) > 0)


class TestCharacterClientMetrics(unittest.TestCase):
    """Test metrics and monitoring capabilities of CharacterClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(
            base_url="https://test.api.com", 
            api_key="test_key",
            enable_metrics=True
        )
    
    @patch('requests.Session.get')
    def test_request_timing_metrics(self, mock_get):
        """Test that request timing metrics are collected."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test"}
        mock_get.return_value = mock_response
        
        self.client.get_character(1)
        
        metrics = self.client.get_metrics()
        self.assertIn("request_count", metrics)
        self.assertIn("average_response_time", metrics)
        self.assertIn("total_requests", metrics)
        self.assertEqual(metrics["request_count"]["GET /characters"], 1)
    
    @patch('requests.Session.get')
    def test_error_rate_metrics(self, mock_get):
        """Test that error rate metrics are tracked."""
        # Generate some successful and failed requests
        responses = [
            Mock(status_code=200, json=lambda: {"id": 1}),  # Success
            Mock(status_code=500),  # Error
            Mock(status_code=200, json=lambda: {"id": 2}),  # Success
            Mock(status_code=404),  # Error
        ]
        
        for response in responses:
            mock_get.return_value = response
            try:
                self.client.get_character(1)
            except CharacterClientError:
                pass  # Expected for error responses
        
        metrics = self.client.get_metrics()
        self.assertIn("error_rate", metrics)
        self.assertIn("success_count", metrics)
        self.assertIn("error_count", metrics)
        
        # Should have 50% error rate (2 successes, 2 errors)
        self.assertEqual(metrics["error_rate"], 0.5)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Generate some activity
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: {"id": 1})
            self.client.get_character(1)
        
        metrics_before = self.client.get_metrics()
        self.assertGreater(metrics_before["total_requests"], 0)
        
        self.client.reset_metrics()
        metrics_after = self.client.get_metrics()
        self.assertEqual(metrics_after["total_requests"], 0)


if __name__ == '__main__':
    # Configure test runner with additional options
    import logging
    logging.basicConfig(level=logging.CRITICAL)  # Suppress logs during testing
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCharacterClient,
        TestCharacterClientEdgeCases,
        TestCharacterClientAdvancedScenarios,
        TestCharacterClientDataValidation,
        TestCharacterClientErrorHandling,
        TestCharacterClientRetryMechanisms,
        TestCharacterClientCaching,
        TestCharacterClientMetrics
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTest(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False,
        warnings='ignore'
    )
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")