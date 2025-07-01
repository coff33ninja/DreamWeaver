import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    """
    Creates and yields a default asyncio event loop for the entire pytest session.
    
    Ensures that an event loop is available for asynchronous tests and properly closes it after the session ends.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def mock_api_response():
    """
    Return a mock API response dictionary simulating a typical generation result.
    
    Returns:
        dict: A dictionary containing mock choices and usage statistics for API response testing.
    """
    return {
        'choices': [{'text': 'Mock response', 'finish_reason': 'stop'}],
        'usage': {'total_tokens': 10, 'prompt_tokens': 5, 'completion_tokens': 5}
    }

@pytest.fixture
def mock_chat_response():
    """
    Return a mock dictionary simulating a typical chat API response, including message content, finish reason, and token usage statistics.
    """
    return {
        'choices': [{'message': {'content': 'Mock chat response'}, 'finish_reason': 'stop'}],
        'usage': {'total_tokens': 12, 'prompt_tokens': 6, 'completion_tokens': 6}
    }

@pytest.fixture
def mock_aiohttp_session():
    """
    Yields an asynchronous mock of `aiohttp.ClientSession` for use in tests.
    
    This fixture allows simulation of HTTP client behavior without making real network requests.
    """
    with patch('aiohttp.ClientSession') as mock_session:
        mock_instance = AsyncMock()
        mock_session.return_value = mock_instance
        yield mock_instance