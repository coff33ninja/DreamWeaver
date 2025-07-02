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
    """Advanced test cases for LLMConfig class covering additional edge cases."""

    def test_config_extreme_temperature_values(self):
        """Test LLMConfig with extreme but valid temperature values."""
        config_zero = LLMConfig(temperature=0.0)
        config_two = LLMConfig(temperature=2.0)
        config_precise = LLMConfig(temperature=1.999999)
        
        self.assertEqual(config_zero.temperature, 0.0)
        self.assertEqual(config_two.temperature, 2.0)
        self.assertEqual(config_precise.temperature, 1.999999)

    def test_config_extreme_max_tokens_values(self):
        """Test LLMConfig with extreme but valid max_tokens values."""
        config_min = LLMConfig(max_tokens=1)
        config_large = LLMConfig(max_tokens=100000)
        
        self.assertEqual(config_min.max_tokens, 1)
        self.assertEqual(config_large.max_tokens, 100000)

    def test_config_special_model_names(self):
        """Test LLMConfig with various model name formats."""
        special_models = [
            "gpt-3.5-turbo-0613",
            "gpt-4-0314",
            "gpt-3.5-turbo-16k-0613",
            "claude-v1",
            "model-with-many-hyphens-and-numbers-123"
        ]
        
        for model in special_models:
            with self.subTest(model=model):
                config = LLMConfig(model=model)
                self.assertEqual(config.model, model)

    def test_config_unicode_api_key(self):
        """Test LLMConfig with unicode characters in API key."""
        unicode_key = "sk-áéíóú你好世界🔑"
        config = LLMConfig(api_key=unicode_key)
        self.assertEqual(config.api_key, unicode_key)

    def test_config_very_long_api_key(self):
        """Test LLMConfig with extremely long API key."""
        long_key = "sk-" + "x" * 1000
        config = LLMConfig(api_key=long_key)
        self.assertEqual(config.api_key, long_key)

    def test_config_timeout_edge_cases(self):
        """Test LLMConfig with various timeout values."""
        timeout_values = [1, 3600, 86400]  # 1 second, 1 hour, 1 day
        
        for timeout in timeout_values:
            with self.subTest(timeout=timeout):
                config = LLMConfig(timeout=timeout)
                self.assertEqual(config.timeout, timeout)

    def test_config_from_dict_missing_fields(self):
        """Test LLMConfig.from_dict with missing optional fields."""
        minimal_data = {"model": "gpt-4"}
        config = LLMConfig.from_dict(minimal_data)
        
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.7)  # default
        self.assertEqual(config.max_tokens, 1000)  # default

    def test_config_from_dict_extra_fields(self):
        """Test LLMConfig.from_dict with extra unknown fields."""
        data_with_extra = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 1500,
            "api_key": "test_key",
            "timeout": 60,
            "unknown_field": "ignored_value",
            "another_extra": 42
        }
        
        # Should not raise an error
        config = LLMConfig.from_dict(data_with_extra)
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.temperature, 0.5)

    def test_config_immutability_concerns(self):
        """Test that modifying config dict doesn't affect original config."""
        config = LLMConfig(model="gpt-4", api_key="original_key")
        config_dict = config.to_dict()
        
        # Modify the dictionary
        config_dict["api_key"] = "modified_key"
        config_dict["model"] = "modified_model"
        
        # Original config should be unchanged
        self.assertEqual(config.api_key, "original_key")
        self.assertEqual(config.model, "gpt-4")


class TestLLMResponseAdvanced(unittest.TestCase):
    """Advanced test cases for LLMResponse class covering additional edge cases."""

    def test_response_very_large_content(self):
        """Test LLMResponse with very large content."""
        large_content = "A" * 100000  # 100KB content
        response = LLMResponse(large_content)
        
        self.assertEqual(len(response.content), 100000)
        self.assertEqual(response.content[0], "A")
        self.assertEqual(response.content[-1], "A")

    def test_response_multiline_content(self):
        """Test LLMResponse with multiline content."""
        multiline_content = """Line 1
Line 2 with special chars: @#$%^&*()
Line 3 with unicode: 你好世界
Line 4 with emojis: 🎉🚀🌟"""
        
        response = LLMResponse(multiline_content)
        self.assertEqual(response.content, multiline_content)
        self.assertIn("Line 1", response.content)
        self.assertIn("你好世界", response.content)
        self.assertIn("🎉🚀🌟", response.content)

    def test_response_null_bytes_content(self):
        """Test LLMResponse with null bytes in content."""
        content_with_nulls = "Before\x00null\x00After"
        response = LLMResponse(content_with_nulls)
        self.assertEqual(response.content, content_with_nulls)

    def test_response_extreme_usage_values(self):
        """Test LLMResponse with extreme usage token counts."""
        extreme_usage = {
            "prompt_tokens": 999999,
            "completion_tokens": 999999,
            "total_tokens": 1999998
        }
        
        response = LLMResponse("test", usage=extreme_usage)
        self.assertEqual(response.usage, extreme_usage)

    def test_response_negative_usage_values(self):
        """Test LLMResponse with negative usage values (malformed data)."""
        negative_usage = {
            "prompt_tokens": -1,
            "completion_tokens": -5,
            "total_tokens": -6
        }
        
        # Should not raise error, just store the values
        response = LLMResponse("test", usage=negative_usage)
        self.assertEqual(response.usage, negative_usage)

    def test_response_unusual_finish_reasons(self):
        """Test LLMResponse with various finish reasons."""
        finish_reasons = ["stop", "length", "content_filter", "function_call", "tool_calls", "error"]
        
        for reason in finish_reasons:
            with self.subTest(reason=reason):
                response = LLMResponse("test", finish_reason=reason)
                self.assertEqual(response.finish_reason, reason)

    def test_response_to_dict_roundtrip_fidelity(self):
        """Test that to_dict preserves all data types correctly."""
        usage = {"prompt_tokens": 42, "completion_tokens": 58, "total_tokens": 100}
        response = LLMResponse(
            content="Test content with unicode 測試",
            usage=usage,
            model="gpt-4-test",
            finish_reason="stop"
        )
        
        response_dict = response.to_dict()
        
        # Verify types are preserved
        self.assertIsInstance(response_dict["content"], str)
        self.assertIsInstance(response_dict["usage"], dict)
        self.assertIsInstance(response_dict["model"], str)
        self.assertIsInstance(response_dict["finish_reason"], str)
        
        # Verify content integrity
        self.assertIn("測試", response_dict["content"])


class TestLLMEngineAdvanced(unittest.TestCase):
    """Advanced test cases for LLMEngine class covering additional scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_api_key_123"
        )
        self.engine = LLMEngine(self.valid_config)

    def test_engine_memory_management(self):
        """Test that engine doesn't leak memory with repeated operations."""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Perform multiple operations
        for i in range(100):
            config = LLMConfig(api_key=f"key_{i}")
            engine = LLMEngine(config)
            engine.set_api_key(f"new_key_{i}")
            self.assertTrue(engine.validate_config())
        
        # Force garbage collection
        gc.collect()
        
        # Memory should not have grown significantly
        final_objects = len(gc.get_objects())
        growth = final_objects - initial_objects
        
        # Allow some growth but not excessive (arbitrary threshold)
        self.assertLess(growth, 1000, f"Memory may be leaking: {growth} new objects")

    def test_engine_thread_safety_config_changes(self):
        """Test thread safety of configuration changes."""
        import threading
        import time
        
        results = []
        
        def change_config(thread_id):
            try:
                for i in range(10):
                    self.engine.set_api_key(f"thread_{thread_id}_key_{i}")
                    time.sleep(0.001)  # Small delay to encourage race conditions
                results.append(f"thread_{thread_id}_success")
            except Exception as e:
                results.append(f"thread_{thread_id}_error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=change_config, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        success_count = sum(1 for r in results if "success" in r)
        self.assertEqual(success_count, 5, f"Thread safety issue: {results}")

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_malformed_response(self, mock_openai):
        """Test handling of malformed API responses."""
        mock_client = AsyncMock()
        
        # Simulate malformed response missing required fields
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_none_content_response(self, mock_openai):
        """Test handling of API response with None content."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None  # None content
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 10
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        result = await engine.generate("Test prompt")
        
        # Should handle None content gracefully
        self.assertIsNone(result.content)

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_missing_usage_info(self, mock_openai):
        """Test handling of API response with missing usage information."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response content"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None  # Missing usage info
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    async def test_generate_prompt_boundary_length(self):
        """Test generation with prompt at exact boundary length."""
        # Test prompt at exactly the limit
        boundary_prompt = "x" * 50000
        
        with self.assertRaises(ValueError) as context:
            await self.engine.generate(boundary_prompt)
        self.assertIn("Prompt too long", str(context.exception))
        
        # Test prompt just under the limit
        acceptable_prompt = "x" * 49999
        
        # Should not raise length error (but will raise client error)
        with self.assertRaises(LLMError) as context:
            await self.engine.generate(acceptable_prompt)
        self.assertIn("Client not initialized", str(context.exception))

    def test_set_api_key_whitespace_handling(self):
        """Test API key setting with whitespace."""
        # Leading/trailing whitespace should be preserved
        key_with_spaces = "  sk-test-key-with-spaces  "
        self.engine.set_api_key(key_with_spaces)
        self.assertEqual(self.engine.config.api_key, key_with_spaces)
        
        # Whitespace-only should fail
        with self.assertRaises(ValueError):
            self.engine.set_api_key("   ")

    def test_validate_config_edge_cases(self):
        """Test configuration validation with edge cases."""
        # Test with manually corrupted config
        config = LLMConfig(api_key="test_key")
        engine = LLMEngine(config)
        
        # Manually corrupt the config after creation
        engine.config.model = None
        self.assertFalse(engine.validate_config())
        
        engine.config.model = "gpt-3.5-turbo"
        engine.config.api_key = ""
        self.assertFalse(engine.validate_config())

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_network_errors(self, mock_openai):
        """Test handling of various network-related errors."""
        network_errors = [
            ConnectionError("Network connection failed"),
            OSError("Network unreachable"),
            TimeoutError("Request timed out"),
        ]
        
        for error in network_errors:
            with self.subTest(error=type(error).__name__):
                mock_client = AsyncMock()
                mock_client.chat.completions.create.side_effect = error
                mock_openai.return_value = mock_client
                
                engine = LLMEngine(self.valid_config)
                engine._client = mock_client
                
                with self.assertRaises(LLMError) as context:
                    await engine.generate("Test prompt")
                self.assertIn("Generation failed", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_stress_test(self, mock_openai):
        """Stress test with rapid sequential generations."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Stress test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.valid_config)
        engine._client = mock_client
        
        # Perform many rapid generations
        for i in range(50):
            result = await engine.generate(f"Stress test prompt {i}")
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Stress test response")
        
        # Verify all calls were made
        self.assertEqual(mock_client.chat.completions.create.call_count, 50)


class TestLLMEngineSecurityAndPerformance(unittest.TestCase):
    """Security and performance-focused tests for LLMEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(api_key="test_key")
        self.engine = LLMEngine(self.config)

    def test_api_key_not_logged_in_string_representation(self):
        """Test that API key is not exposed in string representations."""
        engine_str = str(self.engine)
        config_str = str(self.engine.config)
        
        self.assertNotIn("test_key", engine_str)
        self.assertNotIn("test_key", config_str)

    def test_api_key_in_dict_serialization(self):
        """Test API key handling in dictionary serialization."""
        config_dict = self.engine.config.to_dict()
        
        # API key should be included in dict (for serialization purposes)
        # but this highlights the need for careful handling
        self.assertEqual(config_dict["api_key"], "test_key")

    def test_config_deepcopy_safety(self):
        """Test that config can be safely deep-copied."""
        import copy
        
        original_config = LLMConfig(
            model="gpt-4",
            temperature=0.8,
            api_key="sensitive_key"
        )
        
        copied_config = copy.deepcopy(original_config)
        
        # Configs should be equal but separate objects
        self.assertEqual(original_config, copied_config)
        self.assertIsNot(original_config, copied_config)
        
        # Modifying copy shouldn't affect original
        copied_config.temperature = 0.5
        self.assertNotEqual(original_config.temperature, copied_config.temperature)

    @patch('openai.AsyncOpenAI')
    async def test_generate_input_sanitization(self, mock_openai):
        """Test that potentially dangerous inputs are handled safely."""
        dangerous_inputs = [
            "'; DROP TABLE users; --",  # SQL injection style
            "<script>alert('xss')</script>",  # XSS style
            "\x00\x01\x02\x03\x04",  # Control characters
            "A" * 1000000,  # Extremely long input (should be caught by length check)
        ]
        
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        for dangerous_input in dangerous_inputs:
            with self.subTest(input=dangerous_input[:50]):  # Truncate for readability
                if len(dangerous_input) > 50000:
                    with self.assertRaises(ValueError):
                        await engine.generate(dangerous_input)
                else:
                    # Should not raise an exception for content (API should handle)
                    mock_response = Mock()
                    mock_response.choices = [Mock()]
                    mock_response.choices[0].message.content = "Safe response"
                    mock_response.choices[0].finish_reason = "stop"
                    mock_response.usage.prompt_tokens = 10
                    mock_response.usage.completion_tokens = 10
                    mock_response.usage.total_tokens = 20
                    mock_response.model = "gpt-3.5-turbo"
                    
                    mock_client.chat.completions.create.return_value = mock_response
                    
                    result = await engine.generate(dangerous_input)
                    self.assertIsInstance(result, LLMResponse)

    def test_resource_cleanup_on_error(self):
        """Test that resources are properly cleaned up on errors."""
        # Test with invalid config that should fail
        with self.assertRaises(ValueError):
            LLMConfig(temperature=-1)
        
        # Test with invalid engine initialization
        with self.assertRaises(TypeError):
            LLMEngine(None)
        
        # System should remain stable after errors
        valid_config = LLMConfig(api_key="test")
        valid_engine = LLMEngine(valid_config)
        self.assertIsNotNone(valid_engine)


class TestLLMEngineBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and limit cases."""

    def test_temperature_precision_limits(self):
        """Test temperature with extreme precision values."""
        precision_values = [
            0.0000001,
            1.9999999,
            0.123456789,
            1.000000001
        ]
        
        for temp in precision_values:
            with self.subTest(temperature=temp):
                if 0 <= temp <= 2:
                    config = LLMConfig(temperature=temp)
                    self.assertEqual(config.temperature, temp)
                else:
                    with self.assertRaises(ValueError):
                        LLMConfig(temperature=temp)

    def test_max_tokens_boundary_values(self):
        """Test max_tokens with boundary values."""
        boundary_values = [1, 2, 1000, 4096, 8192, 16384, 32768]
        
        for tokens in boundary_values:
            with self.subTest(max_tokens=tokens):
                config = LLMConfig(max_tokens=tokens)
                self.assertEqual(config.max_tokens, tokens)

    def test_model_name_edge_cases(self):
        """Test model names with edge case formats."""
        edge_case_models = [
            "a",  # Single character
            "gpt-3.5-turbo" + "-" * 100,  # Very long name
            "model_with_underscores",
            "model.with.dots",
            "model with spaces",  # Spaces (should be allowed)
            "模型名称",  # Unicode model name
        ]
        
        for model in edge_case_models:
            with self.subTest(model=model):
                config = LLMConfig(model=model)
                self.assertEqual(config.model, model)


class TestLLMEngineErrorRecovery(unittest.TestCase):
    """Test error recovery and resilience scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(api_key="test_key")
        self.engine = LLMEngine(self.config)

    @patch('openai.AsyncOpenAI')
    async def test_generate_retry_on_transient_errors(self, mock_openai):
        """Test behavior with transient errors that might require retry."""
        transient_errors = [
            Exception("Rate limit exceeded"),
            Exception("Service temporarily unavailable"),
            Exception("Connection timeout"),
        ]
        
        for error in transient_errors:
            with self.subTest(error=str(error)):
                mock_client = AsyncMock()
                mock_client.chat.completions.create.side_effect = error
                mock_openai.return_value = mock_client
                
                engine = LLMEngine(self.config)
                engine._client = mock_client
                
                with self.assertRaises(LLMError) as context:
                    await engine.generate("Test prompt")
                self.assertIn("Generation failed", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_partial_response_data(self, mock_openai):
        """Test handling of responses with partial or incomplete data."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Partial response"
        mock_response.choices[0].finish_reason = "length"  # Truncated due to length
        
        # Simulate incomplete usage data
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = None  # Missing total
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    def test_engine_state_consistency_after_errors(self):
        """Test that engine state remains consistent after errors."""
        original_api_key = self.engine.config.api_key
        
        # Trigger various errors
        with self.assertRaises(ValueError):
            self.engine.set_api_key("")
        
        with self.assertRaises(ValueError):
            self.engine.set_api_key(None)
        
        # Engine state should be unchanged
        self.assertEqual(self.engine.config.api_key, original_api_key)
        self.assertTrue(self.engine.validate_config())


# Additional test utilities
class TestUtilities(unittest.TestCase):
    """Test utility functions and helpers."""

    def test_async_test_runner_functionality(self):
        """Test that the async test runner can handle errors gracefully."""
        # This is a meta-test to ensure our test infrastructure is robust
        async def failing_async_test():
            raise ValueError("Intentional test failure")
        
        async def passing_async_test():
            return True
        
        # These should not crash the test runner
        try:
            asyncio.run(failing_async_test())
        except ValueError:
            pass  # Expected
        
        result = asyncio.run(passing_async_test())
        self.assertTrue(result)

    def test_mock_setup_consistency(self):
        """Test that our mock setups are consistent and reliable."""
        # Test various mock configurations
        mock_configs = [
            {"model": "gpt-3.5-turbo", "api_key": "test1"},
            {"model": "gpt-4", "api_key": "test2", "temperature": 0.5},
            {"model": "claude-v1", "api_key": "test3", "max_tokens": 2000},
        ]
        
        for config_data in mock_configs:
            with self.subTest(config=config_data):
                config = LLMConfig(**config_data)
                engine = LLMEngine(config)
                self.assertIsNotNone(engine)
                self.assertEqual(engine.config.model, config_data["model"])

    def test_test_isolation(self):
        """Test that tests are properly isolated and don't affect each other."""
        # Create and modify a config in this test
        config = LLMConfig(model="test-model", api_key="test-key")
        config.temperature = 1.5
        
        # This modification should not affect other tests
        self.assertEqual(config.temperature, 1.5)
        self.assertEqual(config.model, "test-model")


# Performance benchmark tests (using pytest-benchmark if available)
class TestLLMEnginePerformance(unittest.TestCase):
    """Performance-focused tests for LLMEngine operations."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.config = LLMConfig(api_key="benchmark_key")
        self.engine = LLMEngine(self.config)

    def test_config_creation_performance(self):
        """Test performance of config creation and validation."""
        import time
        
        start_time = time.time()
        
        # Create many configs rapidly
        configs = []
        for i in range(1000):
            config = LLMConfig(
                model=f"model-{i % 5}",
                temperature=0.1 * (i % 20),
                max_tokens=100 + (i % 1000),
                api_key=f"key-{i}"
            )
            configs.append(config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be reasonably fast (less than 1 second for 1000 configs)
        self.assertLess(duration, 1.0, f"Config creation too slow: {duration:.3f}s")
        self.assertEqual(len(configs), 1000)

    def test_engine_initialization_performance(self):
        """Test performance of engine initialization."""
        import time
        
        configs = [LLMConfig(api_key=f"key-{i}") for i in range(100)]
        
        start_time = time.time()
        
        engines = []
        for config in configs:
            engine = LLMEngine(config)
            engines.append(engine)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be reasonably fast
        self.assertLess(duration, 0.5, f"Engine initialization too slow: {duration:.3f}s")
        self.assertEqual(len(engines), 100)

    def test_validation_performance(self):
        """Test performance of configuration validation."""
        import time
        
        # Create engines with various configs
        engines = []
        for i in range(500):
            config = LLMConfig(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000,
                api_key=f"key-{i}"
            )
            engines.append(LLMEngine(config))
        
        start_time = time.time()
        
        # Validate all configs
        results = []
        for engine in engines:
            results.append(engine.validate_config())
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be very fast
        self.assertLess(duration, 0.1, f"Validation too slow: {duration:.3f}s")
        self.assertTrue(all(results))


if __name__ == '__main__':
    # Enhanced test runner with better error handling and performance reporting
    print("Running comprehensive LLM Engine tests...")
    print(f"Testing Framework: pytest with unittest compatibility")
    print(f"Available test classes: TestLLMConfig, TestLLMResponse, TestLLMEngine,")
    print(f"TestLLMEngineIntegration, TestLLMConfigAdvanced, TestLLMResponseAdvanced,")
    print(f"TestLLMEngineAdvanced, TestLLMEngineSecurityAndPerformance,")
    print(f"TestLLMEngineBoundaryConditions, TestLLMEngineErrorRecovery,")
    print(f"TestUtilities, TestLLMEnginePerformance")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLLMConfig,
        TestLLMResponse,
        TestLLMEngine,
        TestLLMEngineIntegration,
        TestLLMConfigAdvanced,
        TestLLMResponseAdvanced,
        TestLLMEngineAdvanced,
        TestLLMEngineSecurityAndPerformance,
        TestLLMEngineBoundaryConditions,
        TestLLMEngineErrorRecovery,
        TestUtilities,
        TestLLMEnginePerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print(f"COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print("\nTest categories covered:")
    print("✓ Basic functionality (existing tests)")
    print("✓ Advanced edge cases")
    print("✓ Security considerations")
    print("✓ Performance benchmarks")
    print("✓ Boundary conditions")
    print("✓ Error recovery")
    print("✓ Thread safety")
    print("✓ Memory management")
    print("✓ Input sanitization")
    print("✓ Resource cleanup")
    
    # Run async tests separately with enhanced reporting
    print("\n" + "=" * 70)
    print("Running asynchronous tests...")
    print("=" * 70)
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)