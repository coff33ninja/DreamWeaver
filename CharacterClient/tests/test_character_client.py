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