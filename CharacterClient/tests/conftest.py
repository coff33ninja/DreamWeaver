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
            "charisma": 15,
        },
        "equipment": {
            "weapon": "Holy Sword",
            "armor": "Plate Mail",
            "accessories": ["Ring of Protection", "Amulet of Health"],
        },
        "skills": ["Divine Strike", "Heal", "Turn Undead"],
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
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
"""
Enhanced pytest configuration and fixtures for TTS Manager tests.
"""

import pytest
import tempfile
import os
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def clean_temp_files():
    """Clean up temporary files after each test."""
    temp_files = []

    def track_temp_file(filepath):
        temp_files.append(filepath)
        return filepath

    yield track_temp_file

    # Cleanup
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except OSError:
            pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Set test environment variables
    os.environ["TTS_TEST_MODE"] = "1"
    os.environ["TTS_CACHE_DISABLED"] = "1"

    yield

    # Cleanup
    os.environ.pop("TTS_TEST_MODE", None)
    os.environ.pop("TTS_CACHE_DISABLED", None)


@pytest.fixture
def mock_audio_file():
    """Create a mock audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(b"fake_audio_data")
        temp_path = temp_file.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_tts_config():
    """Provide a mock TTS configuration for testing."""
    return {
        "voice": "en-US-AriaNeural",
        "speed": 1.0,
        "pitch": 1.0,
        "volume": 0.8,
        "format": "wav",
    }


# Performance test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Automatically mark certain tests based on their names."""
    for item in items:
        # Mark performance tests
        if "performance" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)

        # Mark security tests
        if "security" in item.name.lower() or "malicious" in item.name.lower():
            item.add_marker(pytest.mark.security)

        # Mark integration tests
        if "integration" in item.name.lower() or "concurrent" in item.name.lower():
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default)
        if not any(
            marker.name in ["slow", "integration", "performance", "security"]
            for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing logging functionality."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger_instance = Mock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance
