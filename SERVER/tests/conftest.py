"""Shared pytest fixtures and configuration."""

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add the SERVER directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def temp_data_dir():
    """
    Provides a session-scoped temporary directory for use in tests.
    
    Yields:
        str: The path to the temporary directory, which is automatically cleaned up after the test session.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_file_system():
    """
    Temporarily mocks file system and JSON operations for tests.
    
    All file opening, directory creation, existence checks, and JSON serialization/deserialization are patched to prevent actual disk I/O during the test.
    """
    with patch('builtins.open'), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('json.load', return_value={}), \
         patch('json.dump'):
        yield


@pytest.fixture
def sample_character_data():
    """
    Return a dictionary representing a sample character with predefined attributes for use in tests.
    
    Returns:
        dict: A dictionary containing character attributes such as id, name, level, health, mana, experience, class, stats, and inventory.
    """
    return {
        'id': 'sample-char-001',
        'name': 'Sample Hero',
        'level': 3,
        'health': 90,
        'max_health': 100,
        'mana': 40,
        'experience': 750,
        'class': 'warrior',
        'stats': {
            'strength': 14,
            'agility': 11,
            'intelligence': 9,
            'vitality': 16
        },
        'inventory': {
            'gold': 350,
            'items': [
                {'id': 'health_potion', 'name': 'Health Potion', 'quantity': 2}
            ]
        }
    }


@pytest.fixture
def multiple_character_data():
    """
    Return a list of dictionaries representing multiple sample characters for testing.
    
    Returns:
        List[dict]: A list containing character data dictionaries with attributes such as id, name, level, health, mana, class, stats, and inventory.
    """
    return [
        {
            'id': 'multi-char-001',
            'name': 'Multi Character One',
            'level': 2,
            'health': 80,
            'mana': 30,
            'class': 'warrior',
            'stats': {'strength': 12, 'agility': 8, 'intelligence': 6, 'vitality': 14},
            'inventory': {'gold': 200, 'items': []}
        },
        {
            'id': 'multi-char-002',
            'name': 'Multi Character Two',
            'level': 4,
            'health': 70,
            'mana': 80,
            'class': 'mage',
            'stats': {'strength': 6, 'agility': 10, 'intelligence': 18, 'vitality': 8},
            'inventory': {'gold': 300, 'items': []}
        },
        {
            'id': 'multi-char-003',
            'name': 'Multi Character Three',
            'level': 3,
            'health': 85,
            'mana': 45,
            'class': 'rogue',
            'stats': {'strength': 10, 'agility': 16, 'intelligence': 12, 'vitality': 9},
            'inventory': {'gold': 400, 'items': []}
        }
    ]


def pytest_runtest_setup(item):
    """
    Pytest hook called before running each test item.
    
    This function can be used to implement global setup logic for tests.
    """
    # Add any global test setup here if needed
    pass


def pytest_runtest_teardown(item):
    """
    Hook function called after each test for global cleanup.
    
    This function is invoked by pytest after each test is run. Add any necessary global teardown logic here.
    """
    # Add any global test cleanup here if needed
    pass