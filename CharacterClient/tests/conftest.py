import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_api_key():
    """Session-scoped fixture for test API key."""
    return "test_api_key_12345"

@pytest.fixture(scope="session")
def test_base_url():
    """Session-scoped fixture for test base URL."""
    return "https://api.test-character.com"

@pytest.fixture
def mock_character_data():
    """Fixture providing mock character data for testing."""
    return {
        "id": 42,
        "name": "Mock Character",
        "level": 25,
        "class": "Adventurer",
        "health": 150,
        "mana": 75,
        "experience": 12500,
        "equipment": {
            "weapon": "Magic Sword",
            "armor": "Plate Mail",
            "accessory": "Ring of Power"
        },
        "stats": {
            "strength": 16,
            "dexterity": 14,
            "intelligence": 12,
            "constitution": 15,
            "wisdom": 13,
            "charisma": 11
        },
        "skills": ["Combat", "Magic", "Stealth"],
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-15T12:30:00Z"
    }

@pytest.fixture
def mock_characters_list(mock_character_data):
    """Fixture providing a list of mock characters for testing."""
    characters = []
    for i in range(5):
        character = mock_character_data.copy()
        character["id"] = i + 1
        character["name"] = f"Mock Character {i + 1}"
        character["level"] = 10 + (i * 5)
        characters.append(character)
    return characters