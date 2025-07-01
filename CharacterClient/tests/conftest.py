"""Test configuration and fixtures for CharacterClient tests."""

import pytest
from unittest.mock import Mock
from typing import Dict, List, Any

# Test data fixtures
@pytest.fixture
def sample_character():
    """Sample character data for testing."""
    return {
        "id": 1,
        "name": "Test Hero",
        "class": "Paladin",
        "level": 15,
        "health": 150,
        "mana": 75,
        "attributes": {
            "strength": 18,
            "agility": 14,
            "intelligence": 12,
            "wisdom": 16,
            "constitution": 17,
            "charisma": 15
        },
        "equipment": {
            "weapon": "Holy Sword",
            "armor": "Plate Mail",
            "accessories": ["Ring of Protection", "Amulet of Health"]
        },
        "skills": ["Divine Strike", "Heal", "Turn Undead"],
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z"
    }

@pytest.fixture
def character_client():
    """CharacterClient instance for testing."""
    from CharacterClient.character_client import CharacterClient
    return CharacterClient(base_url="https://test.api.com", api_key="test_key")

@pytest.fixture
def mock_api_response():
    """Mock API response helper."""
    def _mock_response(status_code=200, json_data=None, headers=None):
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        mock_response.headers = headers or {}
        mock_response.text = str(json_data) if json_data else ""
        return mock_response
    return _mock_response

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance