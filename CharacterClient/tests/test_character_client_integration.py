import pytest
from unittest.mock import Mock, patch
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
import threading
import time
import sys
import os

# Import the modules to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from character_client import CharacterClient


class TestCharacterClientIntegration:
    """Integration tests for CharacterClient with realistic scenarios."""
    
    @pytest.fixture
    def character_client(self):
        """Create CharacterClient instance for integration testing."""
        with patch('character_client.Config') as mock_config_class:
            mock_config_instance = Mock()
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'CHARACTER_API_URL': 'https://test-api.character.com/v1',
                'CHARACTER_API_KEY': 'test-integration-key'
            }.get(key, default)
            mock_config_class.return_value = mock_config_instance
            
            return CharacterClient()
    
    def test_full_character_lifecycle(self, character_client):
        """Test complete character lifecycle: create, read, update, delete."""
        character_data = {"name": "Integration Test Hero", "class": "Paladin"}
        
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Mock responses for each operation
            create_response = Mock()
            create_response.json.return_value = {**character_data, "id": 1, "level": 1}
            create_response.raise_for_status.return_value = None
            
            read_response = Mock()
            read_response.json.return_value = {**character_data, "id": 1, "level": 1}
            read_response.raise_for_status.return_value = None
            
            update_response = Mock()
            update_response.json.return_value = {**character_data, "id": 1, "level": 5}
            update_response.raise_for_status.return_value = None
            
            delete_response = Mock()
            delete_response.raise_for_status.return_value = None
            
            mock_session.post.return_value = create_response
            mock_session.get.return_value = read_response
            mock_session.put.return_value = update_response
            mock_session.delete.return_value = delete_response
            
            client = CharacterClient()
            
            # Test create
            created = client.create_character(character_data)
            assert created["name"] == character_data["name"]
            assert created["id"] == 1
            
            # Test read
            retrieved = client.get_character(1)
            assert retrieved["id"] == 1
            assert retrieved["name"] == character_data["name"]
            
            # Test update
            updated = client.update_character(1, {"level": 5})
            assert updated["level"] == 5
            
            # Test delete
            result = client.delete_character(1)
            assert result is True
    
    def test_concurrent_requests_safety(self, character_client):
        """Test thread safety with concurrent requests."""
        results = []
        errors = []
        
        def make_request(character_id):
            try:
                with patch('requests.Session.get') as mock_get:
                    mock_response = Mock()
                    mock_response.json.return_value = {
                        "id": character_id,
                        "name": f"Character {character_id}",
                        "class": "Warrior"
                    }
                    mock_response.raise_for_status.return_value = None
                    mock_get.return_value = mock_response
                    
                    result = character_client.get_character(character_id)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create 10 concurrent threads
        threads = []
        for i in range(1, 11):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 10
        assert len(errors) == 0
        
        # Verify all character IDs are present
        character_ids = [result["id"] for result in results]
        assert set(character_ids) == set(range(1, 11))
    
    def test_error_recovery_sequence(self, character_client):
        """Test error recovery in a sequence of operations."""
        with patch('requests.Session.get') as mock_get:
            # First call fails, second succeeds
            mock_get.side_effect = [
                ConnectionError("Network error"),
                Mock(**{
                    'json.return_value': {"id": 1, "name": "Test Character"},
                    'raise_for_status.return_value': None
                })
            ]
            
            # First call should fail
            with pytest.raises(ConnectionError):
                character_client.get_character(1)
            
            # Second call should succeed
            result = character_client.get_character(1)
            assert result["id"] == 1
            assert result["name"] == "Test Character"
    
    def test_pagination_workflow(self, character_client):
        """Test paginated character listing workflow."""
        with patch('requests.Session.get') as mock_get:
            # Mock first page
            page1_response = Mock()
            page1_response.json.return_value = {
                "characters": [{"id": i, "name": f"Character {i}"} for i in range(1, 6)],
                "total": 12,
                "limit": 5,
                "offset": 0
            }
            page1_response.raise_for_status.return_value = None
            
            # Mock second page
            page2_response = Mock()
            page2_response.json.return_value = {
                "characters": [{"id": i, "name": f"Character {i}"} for i in range(6, 11)],
                "total": 12,
                "limit": 5,
                "offset": 5
            }
            page2_response.raise_for_status.return_value = None
            
            # Mock third page
            page3_response = Mock()
            page3_response.json.return_value = {
                "characters": [{"id": i, "name": f"Character {i}"} for i in range(11, 13)],
                "total": 12,
                "limit": 5,
                "offset": 10
            }
            page3_response.raise_for_status.return_value = None
            
            mock_get.side_effect = [page1_response, page2_response, page3_response]
            
            # Get all pages
            page1 = character_client.list_characters(limit=5, offset=0)
            page2 = character_client.list_characters(limit=5, offset=5)
            page3 = character_client.list_characters(limit=5, offset=10)
            
            # Verify pagination data
            assert len(page1["characters"]) == 5
            assert len(page2["characters"]) == 5
            assert len(page3["characters"]) == 2
            assert page1["total"] == page2["total"] == page3["total"] == 12
            
            # Verify character IDs are correct
            all_characters = (
                page1["characters"] + 
                page2["characters"] + 
                page3["characters"]
            )
            character_ids = [char["id"] for char in all_characters]
            assert character_ids == list(range(1, 13))
    
    def test_search_and_retrieve_workflow(self, character_client):
        """Test search followed by character retrieval workflow."""
        search_results = [
            {"id": 1, "name": "Fire Warrior", "class": "Warrior"},
            {"id": 5, "name": "Ice Warrior", "class": "Warrior"}
        ]
        
        detailed_character = {
            "id": 1,
            "name": "Fire Warrior",
            "class": "Warrior",
            "level": 25,
            "health": 150,
            "mana": 50,
            "equipment": ["Fire Sword", "Iron Shield"]
        }
        
        with patch('requests.Session.get') as mock_get:
            # Mock search response
            search_response = Mock()
            search_response.json.return_value = {"characters": search_results}
            search_response.raise_for_status.return_value = None
            
            # Mock detailed character response
            detail_response = Mock()
            detail_response.json.return_value = detailed_character
            detail_response.raise_for_status.return_value = None
            
            mock_get.side_effect = [search_response, detail_response]
            
            # Perform search
            found_characters = character_client.search_characters("warrior")
            assert len(found_characters) == 2
            assert found_characters[0]["name"] == "Fire Warrior"
            
            # Get detailed info for first result
            detailed = character_client.get_character(found_characters[0]["id"])
            assert detailed["id"] == 1
            assert detailed["level"] == 25
            assert "equipment" in detailed
    
    def test_bulk_operations_performance(self, character_client):
        """Test performance characteristics of bulk operations."""
        # Create test data for 100 characters
        bulk_characters = [
            {"id": i, "name": f"Bulk Character {i}", "class": "Test"}
            for i in range(1, 101)
        ]
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "characters": bulk_characters,
                "total": 100,
                "limit": 100,
                "offset": 0
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            start_time = time.time()
            result = character_client.list_characters(limit=100)
            end_time = time.time()
            
            # Verify results
            assert len(result["characters"]) == 100
            assert result["total"] == 100
            
            # Performance should be reasonable (less than 1 second for mocked response)
            response_time = end_time - start_time
            assert response_time < 1.0
    
    def test_error_handling_chain(self, character_client):
        """Test handling of various errors in a chain of operations."""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Simulate different types of errors
            mock_session.post.side_effect = HTTPError("422 Validation Error")
            mock_session.get.side_effect = Timeout("Request timeout")
            mock_session.put.side_effect = ConnectionError("Connection lost")
            mock_session.delete.side_effect = HTTPError("404 Not Found")
            
            client = CharacterClient()
            
            # Test create failure
            with pytest.raises(HTTPError):
                client.create_character({"name": "Test", "class": "Warrior"})
            
            # Test read failure
            with pytest.raises(Timeout):
                client.get_character(1)
            
            # Test update failure
            with pytest.raises(ConnectionError):
                client.update_character(1, {"level": 10})
            
            # Test delete failure
            with pytest.raises(HTTPError):
                client.delete_character(1)