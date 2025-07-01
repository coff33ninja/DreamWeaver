import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def mock_api_response():
    """Provide a standard mock API response for generation."""
    return {
        'choices': [{'text': 'Mock response', 'finish_reason': 'stop'}],
        'usage': {'total_tokens': 10, 'prompt_tokens': 5, 'completion_tokens': 5}
    }

@pytest.fixture
def mock_chat_response():
    """Provide a standard mock chat API response."""
    return {
        'choices': [{'message': {'content': 'Mock chat response'}, 'finish_reason': 'stop'}],
        'usage': {'total_tokens': 12, 'prompt_tokens': 6, 'completion_tokens': 6}
    }

@pytest.fixture
def mock_aiohttp_session():
    """Provide a mock aiohttp session."""
    with patch('aiohttp.ClientSession') as mock_session:
        mock_instance = AsyncMock()
        mock_session.return_value = mock_instance
        yield mock_instance