import unittest
from unittest.mock import Mock, patch
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.character_client import CharacterClient  # Updated import
from CharacterClient.exceptions import CharacterClientError


class TestCharacterClientPerformance(unittest.TestCase):
    """Performance-related tests for CharacterClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(
            base_url="https://test.api.com", api_key="test_key"
        )

    @patch("requests.Session.get")
    def test_concurrent_requests_thread_safety(self, mock_get):
        """Test that the client handles concurrent requests safely."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test Character"}
        mock_get.return_value = mock_response

        results = []
        errors = []

        def make_request():
            try:
                result = self.client.get_character(1)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads making concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all requests succeeded
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        self.assertEqual(mock_get.call_count, 10)

    @patch("requests.Session.get")
    def test_batch_operations_performance(self, mock_get):
        """Test performance of batch operations."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "Test Character"}
        mock_get.return_value = mock_response

        character_ids = list(range(1, 101))  # 100 character IDs

        start_time = time.time()
        results = self.client.get_characters_batch(character_ids)
        end_time = time.time()

        # Verify results
        self.assertEqual(len(results), 100)
        self.assertEqual(mock_get.call_count, 100)

        # Performance should be reasonable (this is a rough check)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds

    @patch("requests.Session.get")
    def test_memory_usage_large_responses(self, mock_get):
        """Test memory usage with large response data."""
        # Create a large character object
        large_character = {
            "id": 1,
            "name": "Test Character",
            "class": "Warrior",
            "level": 50,
            "inventory": [f"Item {i}" for i in range(1000)],  # Large inventory
            "description": "A" * 10000,  # Large description
            "stats": {f"stat_{i}": i for i in range(100)},  # Many stats
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_character
        mock_get.return_value = mock_response

        # This should handle large responses without issues
        result = self.client.get_character(1)

        self.assertEqual(result["id"], 1)
        self.assertEqual(len(result["inventory"]), 1000)
        self.assertEqual(len(result["description"]), 10000)

    @patch("requests.Session.get")
    def test_timeout_handling(self, mock_get):
        """Test proper handling of request timeouts."""

        # Simulate a slow response that times out
        def slow_response(*args, **kwargs):
            time.sleep(0.1)  # Simulate network delay
            raise requests.Timeout("Request timeout")

        mock_get.side_effect = slow_response

        start_time = time.time()
        with self.assertRaises(CharacterClientError):
            self.client.get_character(1)
        end_time = time.time()

        # Should fail quickly due to timeout
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)


class TestCharacterClientIntegration(unittest.TestCase):
    """Integration-style tests for CharacterClient workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = CharacterClient(
            base_url="https://test.api.com", api_key="test_key"
        )

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    @patch("requests.Session.put")
    @patch("requests.Session.delete")
    def test_full_character_lifecycle(self, mock_delete, mock_put, mock_get, mock_post):
        """Test complete character lifecycle: create, read, update, delete."""
        # Mock character data
        character_data = {"name": "Test Hero", "class": "Paladin", "level": 1}
        created_character = {**character_data, "id": 1}
        updated_character = {**created_character, "level": 5}

        # Mock create response
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = created_character
        mock_post.return_value = mock_post_response

        # Mock get response
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = created_character
        mock_get.return_value = mock_get_response

        # Mock update response
        mock_put_response = Mock()
        mock_put_response.status_code = 200
        mock_put_response.json.return_value = updated_character
        mock_put.return_value = mock_put_response

        # Mock delete response
        mock_delete_response = Mock()
        mock_delete_response.status_code = 204
        mock_delete.return_value = mock_delete_response

        # Test the lifecycle
        # 1. Create character
        created = self.client.create_character(character_data)
        self.assertEqual(created["name"], "Test Hero")
        self.assertEqual(created["id"], 1)

        # 2. Read character
        retrieved = self.client.get_character(1)
        self.assertEqual(retrieved["id"], 1)
        self.assertEqual(retrieved["name"], "Test Hero")

        # 3. Update character
        updated = self.client.update_character(1, {"level": 5})
        self.assertEqual(updated["level"], 5)

        # 4. Delete character
        deleted = self.client.delete_character(1)
        self.assertTrue(deleted)

        # Verify all operations were called
        mock_post.assert_called_once()
        mock_get.assert_called_once()
        mock_put.assert_called_once()
        mock_delete.assert_called_once()

    @patch("requests.Session.get")
    def test_pagination_workflow(self, mock_get):
        """Test pagination workflow for listing characters."""
        # Mock first page
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "characters": [
                {"id": 1, "name": "Character 1"},
                {"id": 2, "name": "Character 2"},
            ],
            "page": 1,
            "total_pages": 3,
        }

        # Mock second page
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "characters": [
                {"id": 3, "name": "Character 3"},
                {"id": 4, "name": "Character 4"},
            ],
            "page": 2,
            "total_pages": 3,
        }

        # Mock third page
        page3_response = Mock()
        page3_response.status_code = 200
        page3_response.json.return_value = {
            "characters": [{"id": 5, "name": "Character 5"}],
            "page": 3,
            "total_pages": 3,
        }

        mock_get.side_effect = [page1_response, page2_response, page3_response]

        # Simulate paginating through all characters
        all_characters = []
        for page in range(1, 4):
            characters = self.client.get_characters(page=page, limit=2)
            all_characters.extend(characters)

        # Verify we got all characters
        self.assertEqual(len(all_characters), 5)
        self.assertEqual(mock_get.call_count, 3)

        # Verify character names
        expected_names = [f"Character {i}" for i in range(1, 6)]
        actual_names = [char["name"] for char in all_characters]
        self.assertEqual(actual_names, expected_names)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
