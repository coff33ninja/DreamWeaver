import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    """
    Provides a session-scoped asyncio event loop for use in asynchronous tests.
    
    Yields:
        loop (asyncio.AbstractEventLoop): The event loop instance available for the duration of the test session.
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
    Return a mock API response dictionary simulating a text generation result.
    
    The response includes a list of choices with generated text and finish reason, as well as token usage statistics.
    """
    return {
        'choices': [{'text': 'Mock response', 'finish_reason': 'stop'}],
        'usage': {'total_tokens': 10, 'prompt_tokens': 5, 'completion_tokens': 5}
    }

@pytest.fixture
def mock_chat_response():
    """
    Return a mock dictionary simulating a typical chat API response.
    
    The response includes a list of choices with message content and finish reason, as well as token usage statistics.
    """
    return {
        'choices': [{'message': {'content': 'Mock chat response'}, 'finish_reason': 'stop'}],
        'usage': {'total_tokens': 12, 'prompt_tokens': 6, 'completion_tokens': 6}
    }

@pytest.fixture
def mock_aiohttp_session():
    """
    Yield an asynchronous mock of aiohttp.ClientSession for use in tests.
    
    This fixture replaces aiohttp.ClientSession with an AsyncMock instance, allowing tests to simulate HTTP client behavior without making real network requests.
    """
    with patch('aiohttp.ClientSession') as mock_session:
        mock_instance = AsyncMock()
        mock_session.return_value = mock_instance
        yield mock_instance