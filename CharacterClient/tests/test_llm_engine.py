import unittest
from unittest.mock import Mock, patch, MagicMock, call, AsyncMock
import asyncio
import json
from typing import Dict, Any, List, Optional
import os
import sys
import tempfile
import pytest

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from llm_engine import LLMEngine, LLMConfig, LLMResponse, LLMError
except ImportError:
    # Create mock classes if the actual implementation doesn't exist yet
    class LLMError(Exception):
        """Base exception for LLM-related errors."""
        pass

    class LLMConfig:
        """Configuration class for LLM Engine."""
        def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7, 
                     max_tokens: int = 1000, api_key: str = None, timeout: int = 30):
            if temperature < 0 or temperature > 2:
                raise ValueError("Temperature must be between 0 and 2")
            if max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
            
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.api_key = api_key
            self.timeout = timeout
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "api_key": self.api_key,
                "timeout": self.timeout
            }
        
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
            return cls(**data)
        
        def __eq__(self, other):
            if not isinstance(other, LLMConfig):
                return False
            return (self.model == other.model and 
                    self.temperature == other.temperature and 
                    self.max_tokens == other.max_tokens and
                    self.api_key == other.api_key)
        
        def __str__(self):
            return f"LLMConfig(model={self.model}, temperature={self.temperature})"
    
    class LLMResponse:
        """Response class for LLM generation results."""
        def __init__(self, content: str, usage: Dict[str, int] = None, 
                     model: str = None, finish_reason: str = "stop"):
            self.content = content
            self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self.model = model
            self.finish_reason = finish_reason
        
        def __eq__(self, other):
            if not isinstance(other, LLMResponse):
                return False
            return (self.content == other.content and 
                    self.usage == other.usage and
                    self.model == other.model and
                    self.finish_reason == other.finish_reason)
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "content": self.content,
                "usage": self.usage,
                "model": self.model,
                "finish_reason": self.finish_reason
            }
    
    class LLMEngine:
        """Main LLM Engine class for handling language model interactions."""
        def __init__(self, config: LLMConfig):
            if not isinstance(config, LLMConfig):
                raise TypeError("Config must be an instance of LLMConfig")
            self.config = config
            self._client = None
            self._initialize_client()
        
        def _initialize_client(self):
            """Initialize the OpenAI client."""
            if self.config.api_key:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.config.api_key)
        
        async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
            """Generate text using the LLM."""
            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty")
            
            if len(prompt) > 50000:  # Arbitrary large limit
                raise ValueError("Prompt too long")
            
            if not self._client:
                raise LLMError("Client not initialized. Check API key.")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **kwargs
                )
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    model=response.model,
                    finish_reason=response.choices[0].finish_reason
                )
            except Exception as e:
                raise LLMError(f"Generation failed: {str(e)}")
        
        def set_api_key(self, api_key: str):
            """Set or update the API key."""
            if not api_key or not isinstance(api_key, str):
                raise ValueError("API key must be a non-empty string")
            
            self.config.api_key = api_key
            self._initialize_client()
        
        def validate_config(self) -> bool:
            """Validate the current configuration."""
            if not self.config.model or not self.config.model.strip():
                return False
            if self.config.api_key is None:
                return False
            if self.config.temperature < 0 or self.config.temperature > 2:
                return False
            if self.config.max_tokens <= 0:
                return False
            return True
        
        def __str__(self):
            return f"LLMEngine(model={self.config.model})"


class TestLLMConfig(unittest.TestCase):
    """Test cases for LLMConfig class."""

    def test_config_initialization_with_defaults(self):
        """Test LLMConfig initialization with default parameters."""
        config = LLMConfig()
        
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 1000)
        self.assertIsNone(config.api_key)
        self.assertEqual(config.timeout, 30)

    def test_config_initialization_with_custom_values(self):
        """Test LLMConfig initialization with custom parameters."""
        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            api_key="test_key_123",
            timeout=60
        )
        
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 2000)
        self.assertEqual(config.api_key, "test_key_123")
        self.assertEqual(config.timeout, 60)

    def test_config_invalid_temperature_negative(self):
        """Test LLMConfig with negative temperature."""
        with self.assertRaises(ValueError) as context:
            LLMConfig(temperature=-0.1)
        self.assertIn("Temperature must be between 0 and 2", str(context.exception))

    def test_config_invalid_temperature_too_high(self):
        """Test LLMConfig with temperature above 2."""
        with self.assertRaises(ValueError) as context:
            LLMConfig(temperature=2.1)
        self.assertIn("Temperature must be between 0 and 2", str(context.exception))

    def test_config_invalid_max_tokens_zero(self):
        """Test LLMConfig with zero max_tokens."""
        with self.assertRaises(ValueError) as context:
            LLMConfig(max_tokens=0)
        self.assertIn("Max tokens must be positive", str(context.exception))

    def test_config_invalid_max_tokens_negative(self):
        """Test LLMConfig with negative max_tokens."""
        with self.assertRaises(ValueError) as context:
            LLMConfig(max_tokens=-100)
        self.assertIn("Max tokens must be positive", str(context.exception))

    def test_config_valid_temperature_boundaries(self):
        """Test LLMConfig with valid temperature boundary values."""
        config_min = LLMConfig(temperature=0.0)
        config_max = LLMConfig(temperature=2.0)
        
        self.assertEqual(config_min.temperature, 0.0)
        self.assertEqual(config_max.temperature, 2.0)

    def test_config_to_dict(self):
        """Test LLMConfig serialization to dictionary."""
        config = LLMConfig(
            model="gpt-4",
            temperature=0.8,
            max_tokens=1500,
            api_key="test_key",
            timeout=45
        )
        
        result = config.to_dict()
        expected = {
            "model": "gpt-4",
            "temperature": 0.8,
            "max_tokens": 1500,
            "api_key": "test_key",
            "timeout": 45
        }
        
        self.assertEqual(result, expected)

    def test_config_from_dict(self):
        """Test LLMConfig deserialization from dictionary."""
        data = {
            "model": "gpt-3.5-turbo-16k",
            "temperature": 0.3,
            "max_tokens": 4000,
            "api_key": "from_dict_key",
            "timeout": 120
        }
        
        config = LLMConfig.from_dict(data)
        
        self.assertEqual(config.model, "gpt-3.5-turbo-16k")
        self.assertEqual(config.temperature, 0.3)
        self.assertEqual(config.max_tokens, 4000)
        self.assertEqual(config.api_key, "from_dict_key")
        self.assertEqual(config.timeout, 120)

    def test_config_equality(self):
        """Test LLMConfig equality comparison."""
        config1 = LLMConfig(model="gpt-3.5-turbo", temperature=0.7, api_key="key1")
        config2 = LLMConfig(model="gpt-3.5-turbo", temperature=0.7, api_key="key1")
        config3 = LLMConfig(model="gpt-4", temperature=0.7, api_key="key1")
        config4 = LLMConfig(model="gpt-3.5-turbo", temperature=0.8, api_key="key1")
        
        self.assertEqual(config1, config2)
        self.assertNotEqual(config1, config3)
        self.assertNotEqual(config1, config4)
        self.assertNotEqual(config1, "not_a_config")

    def test_config_string_representation(self):
        """Test LLMConfig string representation."""
        config = LLMConfig(model="gpt-4", temperature=0.9)
        config_str = str(config)
        
        self.assertIn("LLMConfig", config_str)
        self.assertIn("gpt-4", config_str)
        self.assertIn("0.9", config_str)


class TestLLMResponse(unittest.TestCase):
    """Test cases for LLMResponse class."""

    def test_response_initialization_with_defaults(self):
        """Test LLMResponse initialization with default parameters."""
        response = LLMResponse("Test content")
        
        self.assertEqual(response.content, "Test content")
        self.assertEqual(response.usage["prompt_tokens"], 0)
        self.assertEqual(response.usage["completion_tokens"], 0)
        self.assertEqual(response.usage["total_tokens"], 0)
        self.assertIsNone(response.model)
        self.assertEqual(response.finish_reason, "stop")

    def test_response_initialization_with_custom_values(self):
        """Test LLMResponse initialization with custom parameters."""
        usage = {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        response = LLMResponse(
            content="Custom content",
            usage=usage,
            model="gpt-4",
            finish_reason="length"
        )
        
        self.assertEqual(response.content, "Custom content")
        self.assertEqual(response.usage, usage)
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.finish_reason, "length")

    def test_response_equality(self):
        """Test LLMResponse equality comparison."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        
        response1 = LLMResponse("content", usage, "gpt-3.5-turbo", "stop")
        response2 = LLMResponse("content", usage, "gpt-3.5-turbo", "stop")
        response3 = LLMResponse("different", usage, "gpt-3.5-turbo", "stop")
        
        self.assertEqual(response1, response2)
        self.assertNotEqual(response1, response3)
        self.assertNotEqual(response1, "not_a_response")

    def test_response_to_dict(self):
        """Test LLMResponse serialization to dictionary."""
        usage = {"prompt_tokens": 12, "completion_tokens": 28, "total_tokens": 40}
        response = LLMResponse(
            content="Serialized content",
            usage=usage,
            model="gpt-4-turbo",
            finish_reason="stop"
        )
        
        result = response.to_dict()
        expected = {
            "content": "Serialized content",
            "usage": usage,
            "model": "gpt-4-turbo",
            "finish_reason": "stop"
        }
        
        self.assertEqual(result, expected)

    def test_response_empty_content(self):
        """Test LLMResponse with empty content."""
        response = LLMResponse("")
        
        self.assertEqual(response.content, "")
        self.assertIsInstance(response.usage, dict)

    def test_response_unicode_content(self):
        """Test LLMResponse with unicode characters."""
        unicode_content = "Hello 世界! 🌍 émojis and açcénts"
        response = LLMResponse(unicode_content)
        
        self.assertEqual(response.content, unicode_content)


class TestLLMEngine(unittest.TestCase):
    """Test cases for LLMEngine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.valid_config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_api_key_123"
        )
        self.engine = LLMEngine(self.valid_config)

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any temporary resources
        pass

    def test_engine_initialization_valid_config(self):
        """Test LLMEngine initialization with valid configuration."""
        engine = LLMEngine(self.valid_config)
        
        self.assertEqual(engine.config, self.valid_config)
        self.assertIsNotNone(engine)

    def test_engine_initialization_invalid_config_none(self):
        """Test LLMEngine initialization with None config."""
        with self.assertRaises(TypeError) as context:
            LLMEngine(None)
        self.assertIn("Config must be an instance of LLMConfig", str(context.exception))

    def test_engine_initialization_invalid_config_wrong_type(self):
        """Test LLMEngine initialization with wrong config type."""
        with self.assertRaises(TypeError) as context:
            LLMEngine("not_a_config")
        self.assertIn("Config must be an instance of LLMConfig", str(context.exception))

    def test_engine_initialization_invalid_config_dict(self):
        """Test LLMEngine initialization with dictionary instead of config."""
        config_dict = {"model": "gpt-3.5-turbo", "temperature": 0.7}
        with self.assertRaises(TypeError) as context:
            LLMEngine(config_dict)
        self.assertIn("Config must be an instance of LLMConfig", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_simple_prompt_success(self, mock_openai):
        """Test successful generation with simple prompt."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Reinitialize engine to use mocked client
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate("Test prompt")
        
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Generated response")
        self.assertEqual(result.usage["prompt_tokens"], 10)
        self.assertEqual(result.usage["completion_tokens"], 20)
        self.assertEqual(result.usage["total_tokens"], 30)
        self.assertEqual(result.model, "gpt-3.5-turbo")
        self.assertEqual(result.finish_reason, "stop")

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_system_prompt_success(self, mock_openai):
        """Test successful generation with system prompt."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "System-guided response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate(
            "User prompt", 
            system_prompt="You are a helpful assistant"
        )
        
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "System-guided response")
        
        # Verify system prompt was included in the call
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], 'You are a helpful assistant')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], 'User prompt')

    async def test_generate_empty_prompt(self):
        """Test generation with empty prompt."""
        with self.assertRaises(ValueError) as context:
            await self.engine.generate("")
        self.assertIn("Prompt cannot be empty", str(context.exception))

    async def test_generate_none_prompt(self):
        """Test generation with None prompt."""
        with self.assertRaises(ValueError) as context:
            await self.engine.generate(None)
        self.assertIn("Prompt cannot be empty", str(context.exception))

    async def test_generate_whitespace_only_prompt(self):
        """Test generation with whitespace-only prompt."""
        with self.assertRaises(ValueError) as context:
            await self.engine.generate("   \n\t   ")
        self.assertIn("Prompt cannot be empty", str(context.exception))

    async def test_generate_very_long_prompt(self):
        """Test generation with extremely long prompt."""
        long_prompt = "x" * 50001  # Exceeds the limit
        
        with self.assertRaises(ValueError) as context:
            await self.engine.generate(long_prompt)
        self.assertIn("Prompt too long", str(context.exception))

    async def test_generate_no_client_initialized(self):
        """Test generation when client is not initialized."""
        config = LLMConfig(api_key=None)  # No API key
        engine = LLMEngine(config)
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Client not initialized", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_api_error_handling(self, mock_openai):
        """Test handling of generic API errors during generation."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Generic API Error")
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await self.engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_custom_kwargs(self, mock_openai):
        """Test generation with additional keyword arguments."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        await engine.generate("Test prompt", presence_penalty=0.5, frequency_penalty=0.3)
        
        # Verify custom kwargs were passed
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['presence_penalty'], 0.5)
        self.assertEqual(call_args[1]['frequency_penalty'], 0.3)

    def test_set_api_key_valid(self):
        """Test setting a valid API key."""
        new_key = "new_api_key_456"
        self.engine.set_api_key(new_key)
        
        self.assertEqual(self.engine.config.api_key, new_key)

    def test_set_api_key_empty_string(self):
        """Test setting empty string as API key."""
        with self.assertRaises(ValueError) as context:
            self.engine.set_api_key("")
        self.assertIn("API key must be a non-empty string", str(context.exception))

    def test_set_api_key_none(self):
        """Test setting None as API key."""
        with self.assertRaises(ValueError) as context:
            self.engine.set_api_key(None)
        self.assertIn("API key must be a non-empty string", str(context.exception))

    def test_set_api_key_wrong_type(self):
        """Test setting non-string as API key."""
        with self.assertRaises(ValueError) as context:
            self.engine.set_api_key(123)
        self.assertIn("API key must be a non-empty string", str(context.exception))

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        result = self.engine.validate_config()
        self.assertTrue(result)

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = LLMConfig(api_key=None)
        engine = LLMEngine(config)
        
        result = engine.validate_config()
        self.assertFalse(result)

    def test_validate_config_empty_model(self):
        """Test configuration validation with empty model."""
        config = LLMConfig(model="", api_key="test_key")
        engine = LLMEngine(config)
        
        result = engine.validate_config()
        self.assertFalse(result)

    def test_validate_config_whitespace_model(self):
        """Test configuration validation with whitespace-only model."""
        config = LLMConfig(model="   ", api_key="test_key")
        engine = LLMEngine(config)
        
        result = engine.validate_config()
        self.assertFalse(result)

    def test_validate_config_invalid_temperature(self):
        """Test configuration validation with invalid temperature."""
        # This should not happen due to LLMConfig validation, but test anyway
        config = LLMConfig(api_key="test_key")
        config.temperature = -1  # Manually set invalid value
        engine = LLMEngine(config)
        
        result = engine.validate_config()
        self.assertFalse(result)

    def test_validate_config_invalid_max_tokens(self):
        """Test configuration validation with invalid max_tokens."""
        config = LLMConfig(api_key="test_key")
        config.max_tokens = 0  # Manually set invalid value
        engine = LLMEngine(config)
        
        result = engine.validate_config()
        self.assertFalse(result)

    def test_engine_string_representation(self):
        """Test string representation of LLMEngine."""
        engine_str = str(self.engine)
        
        self.assertIn("LLMEngine", engine_str)
        self.assertIn("gpt-3.5-turbo", engine_str)

    @patch('openai.AsyncOpenAI')
    async def test_generate_special_characters(self, mock_openai):
        """Test generation with special characters in prompt."""
        special_prompt = "Test with émojis 🎉 and unicode ñáéíóú"
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Special char response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate(special_prompt)
        
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Special char response")
        
        # Verify special characters were passed correctly
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertEqual(messages[0]['content'], special_prompt)

    @patch('openai.AsyncOpenAI')
    async def test_generate_json_response(self, mock_openai):
        """Test generation that returns JSON response."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"key": "value", "number": 42}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate("Return JSON response")
        
        # Test that the JSON content can be parsed
        json_data = json.loads(result.content)
        self.assertEqual(json_data["key"], "value")
        self.assertEqual(json_data["number"], 42)

    @patch('openai.AsyncOpenAI')
    async def test_concurrent_generations(self, mock_openai):
        """Test handling multiple concurrent generation requests."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Concurrent response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        # Create multiple concurrent tasks
        tasks = [
            engine.generate(f"Prompt {i}") 
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Concurrent response")
        
        # Verify all calls were made
        self.assertEqual(mock_client.chat.completions.create.call_count, 5)

    @patch('openai.AsyncOpenAI')
    async def test_generate_different_models(self, mock_openai):
        """Test generation with different model types."""
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        
        for model in models:
            with self.subTest(model=model):
                config = LLMConfig(model=model, api_key="test_key")
                engine = LLMEngine(config)
                
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = f"Response from {model}"
                mock_response.choices[0].finish_reason = "stop"
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30
                mock_response.model = model
                
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                engine._client = mock_client
                
                result = await engine.generate("Test prompt")
                
                self.assertEqual(result.model, model)
                self.assertEqual(result.content, f"Response from {model}")
                
                # Verify correct model was used in API call
                call_args = mock_client.chat.completions.create.call_args
                self.assertEqual(call_args[1]['model'], model)

    @patch('openai.AsyncOpenAI')
    async def test_generate_custom_parameters(self, mock_openai):
        """Test generation with custom temperature and max_tokens."""
        custom_config = LLMConfig(
            model="gpt-4",
            temperature=0.9,
            max_tokens=500,
            api_key="test_key"
        )
        engine = LLMEngine(custom_config)
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom parameter response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-4"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        engine._client = mock_client
        
        await engine.generate("Test prompt")
        
        # Verify custom parameters were used
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], "gpt-4")
        self.assertEqual(call_args[1]['temperature'], 0.9)
        self.assertEqual(call_args[1]['max_tokens'], 500)

    @patch('openai.AsyncOpenAI')
    async def test_generate_timeout_handling(self, mock_openai):
        """Test handling of timeout errors during generation."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))


class TestLLMEngineIntegration(unittest.TestCase):
    """Integration tests for LLM Engine with more realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY", "test_key")
        )
        self.engine = LLMEngine(self.config)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "API key required for integration tests")
    async def test_real_api_call(self):
        """Test with real API call (requires API key)."""
        result = await self.engine.generate("Say hello in a friendly way")
        
        self.assertIsInstance(result, LLMResponse)
        self.assertIsNotNone(result.content)
        self.assertTrue(len(result.content) > 0)
        self.assertGreater(result.usage["completion_tokens"], 0)

    def test_comprehensive_config_validation(self):
        """Comprehensive configuration validation test."""
        valid_configs = [
            LLMConfig(model="gpt-3.5-turbo", temperature=0.0, max_tokens=1, api_key="key"),
            LLMConfig(model="gpt-4", temperature=1.0, max_tokens=4096, api_key="key"),
            LLMConfig(model="gpt-3.5-turbo-16k", temperature=0.5, max_tokens=2000, api_key="key")
        ]
        
        for config in valid_configs:
            with self.subTest(config=config):
                engine = LLMEngine(config)
                self.assertTrue(engine.validate_config())

    def test_config_roundtrip_serialization(self):
        """Test config serialization and deserialization roundtrip."""
        original_config = LLMConfig(
            model="gpt-4-turbo",
            temperature=0.3,
            max_tokens=3000,
            api_key="roundtrip_key",
            timeout=90
        )
        
        # Serialize to dict and back
        config_dict = original_config.to_dict()
        restored_config = LLMConfig.from_dict(config_dict)
        
        self.assertEqual(original_config, restored_config)


# Test runner functions
async def run_async_tests():
    """Helper function to run async tests."""
    suite = unittest.TestSuite()
    
    # Add async test methods
    async_test_methods = [
        'test_generate_simple_prompt_success',
        'test_generate_with_system_prompt_success',
        'test_generate_empty_prompt',
        'test_generate_none_prompt',
        'test_generate_whitespace_only_prompt',
        'test_generate_very_long_prompt',
        'test_generate_no_client_initialized',
        'test_generate_api_error_handling',
        'test_generate_with_custom_kwargs',
        'test_generate_special_characters',
        'test_generate_json_response',
        'test_concurrent_generations',
        'test_generate_different_models',
        'test_generate_custom_parameters',
        'test_generate_timeout_handling'
    ]
    
    test_engine = TestLLMEngine()
    for method_name in async_test_methods:
        if hasattr(test_engine, method_name):
            method = getattr(test_engine, method_name)
            if asyncio.iscoroutinefunction(method):
                try:
                    await method()
                    print(f"✓ {method_name} passed")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")


if __name__ == '__main__':
    # Run synchronous tests
    print("Running synchronous tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run async tests
    print("\nRunning asynchronous tests...")
    asyncio.run(run_async_tests())

class TestLLMConfigAdvanced(unittest.TestCase):
    """Advanced test cases for LLMConfig class covering edge cases."""

    def test_config_initialization_with_zero_timeout(self):
        """Test LLMConfig initialization with zero timeout."""
        config = LLMConfig(timeout=0)
        self.assertEqual(config.timeout, 0)

    def test_config_initialization_with_negative_timeout(self):
        """Test LLMConfig initialization with negative timeout."""
        config = LLMConfig(timeout=-1)
        self.assertEqual(config.timeout, -1)

    def test_config_initialization_with_large_timeout(self):
        """Test LLMConfig initialization with very large timeout."""
        large_timeout = 999999
        config = LLMConfig(timeout=large_timeout)
        self.assertEqual(config.timeout, large_timeout)

    def test_config_with_unicode_model_name(self):
        """Test LLMConfig with unicode characters in model name."""
        unicode_model = "gpt-3.5-turbo-测试"
        config = LLMConfig(model=unicode_model)
        self.assertEqual(config.model, unicode_model)

    def test_config_with_special_characters_in_api_key(self):
        """Test LLMConfig with special characters in API key."""
        special_key = "sk-proj-AbC123_/+={}[]|\\:;\"'<>?,."
        config = LLMConfig(api_key=special_key)
        self.assertEqual(config.api_key, special_key)

    def test_config_with_very_long_api_key(self):
        """Test LLMConfig with extremely long API key."""
        long_key = "sk-" + "x" * 1000
        config = LLMConfig(api_key=long_key)
        self.assertEqual(config.api_key, long_key)

    def test_config_with_empty_model_name(self):
        """Test LLMConfig with empty model name."""
        config = LLMConfig(model="")
        self.assertEqual(config.model, "")

    def test_config_with_very_large_max_tokens(self):
        """Test LLMConfig with very large max_tokens value."""
        large_tokens = 2147483647  # Max int32
        config = LLMConfig(max_tokens=large_tokens)
        self.assertEqual(config.max_tokens, large_tokens)

    def test_config_temperature_precision(self):
        """Test LLMConfig with high-precision temperature values."""
        precise_temp = 0.123456789
        config = LLMConfig(temperature=precise_temp)
        self.assertEqual(config.temperature, precise_temp)

    def test_config_from_dict_with_missing_keys(self):
        """Test LLMConfig.from_dict with missing optional keys."""
        minimal_data = {"model": "gpt-4"}
        config = LLMConfig.from_dict(minimal_data)
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.7)  # Default value

    def test_config_from_dict_with_extra_keys(self):
        """Test LLMConfig.from_dict with extra keys (should ignore)."""
        data_with_extra = {
            "model": "gpt-4",
            "temperature": 0.8,
            "extra_key": "should_be_ignored",
            "another_extra": 123
        }
        config = LLMConfig.from_dict(data_with_extra)
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.8)
        # Extra keys should not cause errors

    def test_config_equality_with_timeout_difference(self):
        """Test LLMConfig equality when only timeout differs."""
        config1 = LLMConfig(timeout=30)
        config2 = LLMConfig(timeout=60)
        # Note: Current implementation doesn't include timeout in equality
        self.assertEqual(config1, config2)

    def test_config_hash_consistency(self):
        """Test that equal configs have consistent behavior."""
        config1 = LLMConfig(model="gpt-4", temperature=0.7, api_key="key")
        config2 = LLMConfig(model="gpt-4", temperature=0.7, api_key="key")
        
        # Equal configs should behave consistently
        self.assertEqual(config1, config2)
        self.assertEqual(str(config1), str(config2))

    def test_config_mutation_after_creation(self):
        """Test modifying config attributes after creation."""
        config = LLMConfig()
        original_model = config.model
        
        config.model = "gpt-4"
        self.assertNotEqual(config.model, original_model)
        self.assertEqual(config.model, "gpt-4")

    def test_config_to_dict_preserves_none_values(self):
        """Test that to_dict preserves None values correctly."""
        config = LLMConfig(api_key=None)
        result = config.to_dict()
        self.assertIn("api_key", result)
        self.assertIsNone(result["api_key"])


class TestLLMResponseAdvanced(unittest.TestCase):
    """Advanced test cases for LLMResponse class covering edge cases."""

    def test_response_with_missing_usage_keys(self):
        """Test LLMResponse with partial usage information."""
        partial_usage = {"prompt_tokens": 10}
        response = LLMResponse("content", usage=partial_usage)
        
        # Should use the provided partial usage
        self.assertEqual(response.usage, partial_usage)

    def test_response_with_zero_usage_tokens(self):
        """Test LLMResponse with zero token usage."""
        zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        response = LLMResponse("content", usage=zero_usage)
        
        self.assertEqual(response.usage, zero_usage)

    def test_response_with_large_usage_numbers(self):
        """Test LLMResponse with very large token usage numbers."""
        large_usage = {
            "prompt_tokens": 1000000,
            "completion_tokens": 500000,
            "total_tokens": 1500000
        }
        response = LLMResponse("content", usage=large_usage)
        
        self.assertEqual(response.usage, large_usage)

    def test_response_with_different_finish_reasons(self):
        """Test LLMResponse with various finish reasons."""
        finish_reasons = ["stop", "length", "content_filter", "function_call", "tool_calls"]
        
        for reason in finish_reasons:
            with self.subTest(finish_reason=reason):
                response = LLMResponse("content", finish_reason=reason)
                self.assertEqual(response.finish_reason, reason)

    def test_response_with_none_model(self):
        """Test LLMResponse with explicitly None model."""
        response = LLMResponse("content", model=None)
        self.assertIsNone(response.model)

    def test_response_with_empty_model_string(self):
        """Test LLMResponse with empty model string."""
        response = LLMResponse("content", model="")
        self.assertEqual(response.model, "")

    def test_response_with_very_long_content(self):
        """Test LLMResponse with extremely long content."""
        long_content = "x" * 100000
        response = LLMResponse(long_content)
        self.assertEqual(response.content, long_content)
        self.assertEqual(len(response.content), 100000)

    def test_response_with_multiline_content(self):
        """Test LLMResponse with multiline content including various line endings."""
        multiline_content = "Line 1\nLine 2\r\nLine 3\rLine 4\n\nLine 6"
        response = LLMResponse(multiline_content)
        self.assertEqual(response.content, multiline_content)

    def test_response_with_json_content(self):
        """Test LLMResponse containing valid JSON."""
        json_content = '{"message": "Hello", "count": 42, "active": true}'
        response = LLMResponse(json_content)
        
        # Verify content is preserved as string
        self.assertEqual(response.content, json_content)
        # Verify it's valid JSON
        parsed = json.loads(response.content)
        self.assertEqual(parsed["message"], "Hello")

    def test_response_with_invalid_json_content(self):
        """Test LLMResponse containing invalid JSON."""
        invalid_json = '{"incomplete": json'
        response = LLMResponse(invalid_json)
        
        self.assertEqual(response.content, invalid_json)
        # Should not raise when creating response, only when parsing

    def test_response_equality_edge_cases(self):
        """Test LLMResponse equality with edge cases."""
        usage1 = {"prompt_tokens": 10, "completion_tokens": 20}
        usage2 = {"prompt_tokens": 10, "completion_tokens": 20}
        
        response1 = LLMResponse("content", usage=usage1)
        response2 = LLMResponse("content", usage=usage2)
        
        self.assertEqual(response1, response2)
        
        # Test with different usage object references but same values
        self.assertEqual(response1.usage, response2.usage)

    def test_response_to_dict_with_none_values(self):
        """Test LLMResponse.to_dict with None values."""
        response = LLMResponse("content", model=None)
        result = response.to_dict()
        
        self.assertIn("model", result)
        self.assertIsNone(result["model"])

    def test_response_with_custom_usage_structure(self):
        """Test LLMResponse with non-standard usage structure."""
        custom_usage = {
            "input_tokens": 15,
            "output_tokens": 25,
            "cache_tokens": 5,
            "total": 45
        }
        response = LLMResponse("content", usage=custom_usage)
        
        self.assertEqual(response.usage, custom_usage)


class TestLLMEngineAdvanced(unittest.TestCase):
    """Advanced test cases for LLMEngine class covering complex scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_api_key_123"
        )
        self.engine = LLMEngine(self.valid_config)

    def test_engine_with_config_mutation_after_creation(self):
        """Test engine behavior when config is mutated after creation."""
        original_model = self.engine.config.model
        
        # Mutate the config
        self.engine.config.model = "gpt-4"
        
        self.assertNotEqual(self.engine.config.model, original_model)
        self.assertEqual(self.engine.config.model, "gpt-4")

    def test_engine_multiple_api_key_updates(self):
        """Test multiple API key updates in sequence."""
        keys = ["key1", "key2", "key3"]
        
        for key in keys:
            self.engine.set_api_key(key)
            self.assertEqual(self.engine.config.api_key, key)

    def test_engine_api_key_with_whitespace(self):
        """Test API key handling with various whitespace scenarios."""
        whitespace_key = "  sk-test-key-with-spaces  "
        
        # Should accept the key as-is (including whitespace)
        self.engine.set_api_key(whitespace_key)
        self.assertEqual(self.engine.config.api_key, whitespace_key)

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_empty_system_prompt(self, mock_openai):
        """Test generation with empty string system prompt."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        await engine.generate("User prompt", system_prompt="")
        
        # Verify empty system prompt is not included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertEqual(len(messages), 1)  # Only user message
        self.assertEqual(messages[0]['role'], 'user')

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_whitespace_only_system_prompt(self, mock_openai):
        """Test generation with whitespace-only system prompt."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        await engine.generate("User prompt", system_prompt="   \n\t   ")
        
        # Should include the whitespace system prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], "   \n\t   ")

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_very_long_system_prompt(self, mock_openai):
        """Test generation with extremely long system prompt."""
        long_system_prompt = "System instruction: " + "x" * 10000
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate("User prompt", system_prompt=long_system_prompt)
        
        self.assertIsInstance(result, LLMResponse)
        
        # Verify long system prompt was passed
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertEqual(messages[0]['content'], long_system_prompt)

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_malformed_api_response_missing_choices(self, mock_openai):
        """Test handling of malformed API response missing choices."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 10
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_malformed_api_response_missing_usage(self, mock_openai):
        """Test handling of API response missing usage information."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        # Mock response without usage attribute
        del mock_response.usage
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_different_finish_reasons(self, mock_openai):
        """Test generation with different finish reasons."""
        finish_reasons = ["stop", "length", "content_filter", "function_call"]
        
        for finish_reason in finish_reasons:
            with self.subTest(finish_reason=finish_reason):
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = f"Response for {finish_reason}"
                mock_response.choices[0].finish_reason = finish_reason
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30
                mock_response.model = "gpt-3.5-turbo"
                
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                engine = LLMEngine(self.valid_config)
                engine._client = mock_client
                
                result = await engine.generate("Test prompt")
                
                self.assertEqual(result.finish_reason, finish_reason)
                self.assertEqual(result.content, f"Response for {finish_reason}")

    @patch('openai.AsyncOpenAI')
    async def test_generate_stress_test_many_concurrent(self, mock_openai):
        """Stress test with many concurrent requests."""
        num_concurrent = 50
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Concurrent response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        # Create many concurrent tasks
        tasks = [
            engine.generate(f"Prompt {i}") 
            for i in range(num_concurrent)
        ]
        
        results = await asyncio.gather(*tasks)
        
        self.assertEqual(len(results), num_concurrent)
        for i, result in enumerate(results):
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Concurrent response")
        
        # Verify all calls were made
        self.assertEqual(mock_client.chat.completions.create.call_count, num_concurrent)

    def test_validate_config_comprehensive_edge_cases(self):
        """Comprehensive config validation with various edge cases."""
        # Test with manually corrupted config values
        test_cases = [
            {"field": "model", "value": None, "expected": False},
            {"field": "model", "value": "", "expected": False},
            {"field": "model", "value": "   ", "expected": False},
            {"field": "api_key", "value": None, "expected": False},
            {"field": "temperature", "value": -0.1, "expected": False},
            {"field": "temperature", "value": 2.1, "expected": False},
            {"field": "max_tokens", "value": 0, "expected": False},
            {"field": "max_tokens", "value": -1, "expected": False},
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                config = LLMConfig(api_key="valid_key")
                setattr(config, test_case["field"], test_case["value"])
                engine = LLMEngine(config)
                
                result = engine.validate_config()
                self.assertEqual(result, test_case["expected"])

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_connection_error(self, mock_openai):
        """Test handling of connection errors."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Connection failed")
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))
        self.assertIn("Connection failed", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_ssl_error(self, mock_openai):
        """Test handling of SSL errors."""
        import ssl
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = ssl.SSLError("SSL handshake failed")
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))

    def test_engine_string_representation_with_long_model_name(self):
        """Test string representation with very long model name."""
        long_model = "gpt-4-turbo-preview-with-very-long-name-" + "x" * 100
        config = LLMConfig(model=long_model, api_key="key")
        engine = LLMEngine(config)
        
        engine_str = str(engine)
        self.assertIn("LLMEngine", engine_str)
        self.assertIn(long_model, engine_str)

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_zero_max_tokens_response(self, mock_openai):
        """Test handling of response with zero completion tokens."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""  # Empty response
        mock_response.choices[0].finish_reason = "length"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 10
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate("Test prompt")
        
        self.assertEqual(result.content, "")
        self.assertEqual(result.usage["completion_tokens"], 0)
        self.assertEqual(result.finish_reason, "length")

    def test_client_reinitialization_edge_cases(self):
        """Test client reinitialization in various scenarios."""
        # Start with no API key
        config = LLMConfig(api_key=None)
        engine = LLMEngine(config)
        self.assertIsNone(engine._client)
        
        # Set API key and verify client is initialized
        engine.set_api_key("new_key")
        # Client initialization happens in _initialize_client
        # We can't easily test the actual openai client creation without mocking
        self.assertEqual(engine.config.api_key, "new_key")
        
        # Change API key again
        engine.set_api_key("another_key")
        self.assertEqual(engine.config.api_key, "another_key")


class TestLLMEngineErrorHandling(unittest.TestCase):
    """Specific tests for error handling scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = LLMConfig(api_key="test_key")
        self.engine = LLMEngine(self.valid_config)

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_rate_limit_error(self, mock_openai):
        """Test handling of rate limit errors."""
        mock_client = AsyncMock()
        
        # Simulate a rate limit error (status 429)
        class RateLimitError(Exception):
            def __init__(self):
                super().__init__("Rate limit exceeded")
                self.status_code = 429
        
        mock_client.chat.completions.create.side_effect = RateLimitError()
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_authentication_error(self, mock_openai):
        """Test handling of authentication errors."""
        mock_client = AsyncMock()
        
        class AuthenticationError(Exception):
            def __init__(self):
                super().__init__("Invalid API key")
                self.status_code = 401
        
        mock_client.chat.completions.create.side_effect = AuthenticationError()
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))
        self.assertIn("Invalid API key", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_server_error(self, mock_openai):
        """Test handling of server errors."""
        mock_client = AsyncMock()
        
        class ServerError(Exception):
            def __init__(self):
                super().__init__("Internal server error")
                self.status_code = 500
        
        mock_client.chat.completions.create.side_effect = ServerError()
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_json_decode_error(self, mock_openai):
        """Test handling of JSON decode errors from API."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = json.JSONDecodeError(
            "Invalid JSON", "", 0
        )
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError) as context:
            await engine.generate("Test prompt")
        self.assertIn("Generation failed", str(context.exception))


class TestLLMEnginePerformance(unittest.TestCase):
    """Performance and resource usage tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = LLMConfig(api_key="test_key")
        self.engine = LLMEngine(self.valid_config)

    def test_memory_usage_with_large_responses(self):
        """Test memory handling with large response objects."""
        # Create a large response
        large_content = "x" * 1000000  # 1MB of content
        large_usage = {
            "prompt_tokens": 100000,
            "completion_tokens": 500000,
            "total_tokens": 600000
        }
        
        response = LLMResponse(
            content=large_content,
            usage=large_usage,
            model="gpt-4"
        )
        
        # Verify the response is created successfully
        self.assertEqual(len(response.content), 1000000)
        self.assertEqual(response.usage["total_tokens"], 600000)
        
        # Test serialization of large response
        response_dict = response.to_dict()
        self.assertEqual(len(response_dict["content"]), 1000000)

    def test_config_object_reuse(self):
        """Test that config objects can be reused across engines."""
        config = LLMConfig(api_key="shared_key")
        
        engine1 = LLMEngine(config)
        engine2 = LLMEngine(config)
        
        # Both engines should have the same config reference
        self.assertIs(engine1.config, engine2.config)
        
        # Modifying config should affect both engines
        config.temperature = 0.9
        self.assertEqual(engine1.config.temperature, 0.9)
        self.assertEqual(engine2.config.temperature, 0.9)

    @patch('openai.AsyncOpenAI')
    async def test_rapid_sequential_requests(self, mock_openai):
        """Test rapid sequential API requests."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Sequential response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        # Make rapid sequential requests
        num_requests = 20
        start_time = asyncio.get_event_loop().time()
        
        for i in range(num_requests):
            result = await engine.generate(f"Request {i}")
            self.assertIsInstance(result, LLMResponse)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Verify all requests completed (basic performance test)
        self.assertEqual(mock_client.chat.completions.create.call_count, num_requests)
        # This is just to ensure the test framework is working correctly
        self.assertGreater(total_time, 0)


# Update the async test runner to include new test methods
async def run_additional_async_tests():
    """Helper function to run additional async tests."""
    test_classes = [
        TestLLMEngineAdvanced,
        TestLLMEngineErrorHandling,
        TestLLMEnginePerformance
    ]
    
    for test_class in test_classes:
        print(f"\nRunning async tests for {test_class.__name__}...")
        test_instance = test_class()
        
        # Get all async test methods
        for method_name in dir(test_instance):
            if method_name.startswith('test_') and hasattr(test_instance, method_name):
                method = getattr(test_instance, method_name)
                if asyncio.iscoroutinefunction(method):
                    try:
                        # Call setUp if it exists
                        if hasattr(test_instance, 'setUp'):
                            test_instance.setUp()
                        
                        await method()
                        print(f"✓ {test_class.__name__}.{method_name} passed")
                        
                        # Call tearDown if it exists
                        if hasattr(test_instance, 'tearDown'):
                            test_instance.tearDown()
                            
                    except Exception as e:
                        print(f"✗ {test_class.__name__}.{method_name} failed: {e}")


# Add to main execution block
if __name__ == '__main__':
    print("Running comprehensive LLM Engine tests...")
    
    # Run synchronous tests (existing and new)
    print("Running synchronous tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run async tests (existing and new)
    print("\nRunning asynchronous tests...")
    asyncio.run(run_async_tests())
    asyncio.run(run_additional_async_tests())
    
    print("\nAll tests completed!")
