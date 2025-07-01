import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import json
import time
import aiohttp

# Import the LLM engine components
from CharacterClient.src.llm_engine import LLMEngine, LLMConfig, LLMResponse


class TestLLMConfig:
    """Test suite for LLM configuration validation."""
    
    def test_config_creation_with_required_params(self):
        """
        Verify that an LLMConfig instance can be created with only the required parameters and that default values are set for optional fields.
        """
        config = LLMConfig(
            model_name="gpt-3.5-turbo",
            api_key="sk-test123"
        )
        assert config.model_name == "gpt-3.5-turbo"
        assert config.api_key == "sk-test123"
        assert config.base_url == "https://api.openai.com/v1"  # Default value
        assert config.max_tokens == 1000  # Default value
        assert config.temperature == 0.7  # Default value
    
    def test_config_creation_with_all_params(self):
        """
        Verifies that an LLMConfig instance is correctly created when all parameters are specified.
        """
        config = LLMConfig(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
            max_tokens=2000,
            temperature=0.9,
            timeout=60.0,
            retries=5
        )
        assert config.model_name == "test-model"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.test.com"
        assert config.max_tokens == 2000
        assert config.temperature == 0.9
        assert config.timeout == 60.0
        assert config.retries == 5
    
    def test_config_invalid_temperature_too_high(self):
        """
        Test that creating an LLMConfig with a temperature above the allowed maximum raises a ValueError.
        """
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            LLMConfig(
                model_name="test-model",
                api_key="test-key",
                temperature=2.5
            )
    
    def test_config_invalid_temperature_negative(self):
        """
        Test that creating an LLMConfig with a negative temperature raises a ValueError.
        """
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            LLMConfig(
                model_name="test-model",
                api_key="test-key",
                temperature=-0.1
            )
    
    def test_config_invalid_max_tokens_zero(self):
        """
        Test that creating an LLMConfig with max_tokens set to zero raises a ValueError.
        """
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            LLMConfig(
                model_name="test-model",
                api_key="test-key",
                max_tokens=0
            )
    
    def test_config_invalid_max_tokens_negative(self):
        """
        Test that creating an LLMConfig with a negative max_tokens value raises a ValueError.
        """
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            LLMConfig(
                model_name="test-model",
                api_key="test-key",
                max_tokens=-100
            )
    
    def test_config_boundary_temperature_values(self):
        """
        Test that LLMConfig accepts temperature values at the minimum (0.0) and maximum (2.0) boundaries.
        """
        # Test minimum boundary
        config1 = LLMConfig(model_name="test", api_key="key", temperature=0.0)
        assert config1.temperature == 0.0
        
        # Test maximum boundary
        config2 = LLMConfig(model_name="test", api_key="key", temperature=2.0)
        assert config2.temperature == 2.0


class TestLLMResponse:
    """Test suite for LLM response objects."""
    
    def test_response_creation(self):
        """
        Test that an LLMResponse object is correctly created with specified text and metadata.
        """
        response = LLMResponse(
            text="Hello, world!",
            metadata={"tokens": 15, "finish_reason": "stop"}
        )
        assert response.text == "Hello, world!"
        assert response.metadata["tokens"] == 15
        assert response.metadata["finish_reason"] == "stop"
    
    def test_response_empty_metadata(self):
        """
        Test that an LLMResponse object is correctly created when provided with empty metadata.
        """
        response = LLMResponse(text="Test", metadata={})
        assert response.text == "Test"
        assert response.metadata == {}


class TestLLMEngine:
    """Comprehensive test suite for LLM Engine functionality."""
    
    @pytest.fixture
    def default_config(self):
        """
        Returns a default `LLMConfig` instance with preset values for use in tests.
        """
        return LLMConfig(
            model_name="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
            max_tokens=1000,
            temperature=0.7,
            timeout=30.0,
            retries=3
        )
    
    @pytest.fixture
    def llm_engine(self, default_config):
        """
        Creates an instance of LLMEngine using the provided default configuration for use in tests.
        """
        return LLMEngine(default_config)
    
    @pytest.fixture
    async def initialized_engine(self, llm_engine):
        """
        Asynchronously creates and initializes an LLM engine instance for use in tests, yielding the initialized engine and ensuring cleanup after use.
        """
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value = AsyncMock()
            await llm_engine.initialize()
            yield llm_engine
            llm_engine.shutdown()
    
    # Engine Creation and Initialization Tests
    def test_engine_creation_valid_config(self, default_config):
        """
        Test that an LLMEngine instance is correctly created with a valid configuration.
        
        Verifies that the engine's configuration is set, it is not initialized, and no session exists upon creation.
        """
        engine = LLMEngine(default_config)
        assert engine.config == default_config
        assert not engine.is_initialized
        assert engine.session is None
    
    def test_engine_creation_none_config(self):
        """
        Test that creating an LLMEngine with a None configuration raises a ValueError.
        """
        with pytest.raises(ValueError, match="Config cannot be None"):
            LLMEngine(None)
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, llm_engine):
        """
        Test that the LLM engine initializes successfully, creating a session and setting the initialized state.
        """
        assert not llm_engine.is_initialized
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value = AsyncMock()
            await llm_engine.initialize()
            
            assert llm_engine.is_initialized
            assert llm_engine.session is not None
            mock_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_engine_double_initialization(self, llm_engine):
        """
        Test that calling initialize twice on the engine does not create multiple client sessions.
        
        Ensures that repeated initialization reuses the same session and does not result in duplicate resource allocation.
        """
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value = AsyncMock()
            
            await llm_engine.initialize()
            first_session = llm_engine.session
            
            await llm_engine.initialize()  # Second initialization
            second_session = llm_engine.session
            
            assert first_session is second_session
            mock_session.assert_called_once()  # Should only be called once
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, default_config):
        """
        Tests that the LLMEngine can be used as an async context manager, ensuring it initializes on entry and shuts down on exit.
        """
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value = AsyncMock()
            
            async with LLMEngine(default_config) as engine:
                assert engine.is_initialized
                assert engine.session is not None
            
            # After exiting context, shutdown should be called
            # Note: The actual session close happens asynchronously
    
    # Text Generation Tests
    @pytest.mark.asyncio
    async def test_generate_simple_prompt(self, initialized_engine):
        """
        Test that the engine generates text correctly for a simple prompt and returns a valid LLMResponse with expected metadata.
        """
        prompt = "Hello, world!"
        mock_response = {
            'choices': [{'text': 'Hello! How can I help you?', 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 15}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.generate(prompt)
            
            assert isinstance(response, LLMResponse)
            assert response.text == 'Hello! How can I help you?'
            assert response.metadata['tokens'] == 15
            assert response.metadata['finish_reason'] == 'stop'
            
            # Verify API call was made with correct parameters
            mock_call.assert_called_once()
            call_args = mock_call.call_args[0][0]
            assert call_args['model'] == 'test-model'
            assert call_args['prompt'] == prompt
    
    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, initialized_engine):
        """
        Test that generating text with an empty prompt raises a ValueError.
        """
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await initialized_engine.generate("")
    
    @pytest.mark.asyncio
    async def test_generate_whitespace_only_prompt(self, initialized_engine):
        """
        Test that generating text with a whitespace-only prompt raises a ValueError.
        """
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await initialized_engine.generate("   \n\t  ")
    
    @pytest.mark.asyncio
    async def test_generate_none_prompt(self, initialized_engine):
        """
        Test that calling `generate` with a None prompt raises a ValueError.
        """
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await initialized_engine.generate(None)
    
    @pytest.mark.asyncio
    async def test_generate_uninitialized_engine(self, llm_engine):
        """
        Test that calling `generate` on an uninitialized engine raises a RuntimeError.
        """
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            await llm_engine.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_parameters(self, initialized_engine):
        """
        Tests that the generate method correctly handles custom parameter values and passes them to the API call.
        
        Verifies that the generated response matches the mock output and that the specified max_tokens, temperature, and top_p values are included in the API request.
        """
        prompt = "Tell me a story"
        mock_response = {
            'choices': [{'text': 'Once upon a time...', 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 25}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.generate(
                prompt,
                max_tokens=500,
                temperature=0.9,
                top_p=0.95
            )
            
            assert response.text == 'Once upon a time...'
            
            # Verify custom parameters were passed
            call_args = mock_call.call_args[0][0]
            assert call_args['max_tokens'] == 500
            assert call_args['temperature'] == 0.9
            assert call_args['top_p'] == 0.95
    
    @pytest.mark.asyncio
    async def test_generate_very_long_prompt(self, initialized_engine):
        """
        Tests that the engine can generate a response for an extremely long prompt without errors.
        
        Verifies that the generated response contains the expected text and metadata when provided with a prompt of 10,000 characters.
        """
        long_prompt = "A" * 10000
        mock_response = {
            'choices': [{'text': 'Response to long prompt', 'finish_reason': 'length'}],
            'usage': {'total_tokens': 500}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.generate(long_prompt)
            assert response.text == 'Response to long prompt'
            assert response.metadata['finish_reason'] == 'length'
    
    @pytest.mark.asyncio
    async def test_generate_special_characters(self, initialized_engine):
        """
        Tests that the generate method correctly handles prompts containing special characters and Unicode, ensuring the response includes expected Unicode content.
        """
        special_prompt = "Hello 🌍! Test émojis and spéciål chars: ñáéíóú"
        mock_response = {
            'choices': [{'text': 'Response with émojis 🚀', 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 20}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.generate(special_prompt)
            assert '🚀' in response.text
            assert 'émojis' in response.text
    
    # Chat Functionality Tests
    @pytest.mark.asyncio
    async def test_chat_single_message(self, initialized_engine):
        """
        Tests that the chat method returns the expected response and metadata when given a single user message.
        
        Verifies that the engine correctly processes a single-message conversation, returns the assistant's reply, and includes token usage in the metadata.
        """
        messages = [{"role": "user", "content": "Hello!"}]
        mock_response = {
            'choices': [{'message': {'content': 'Hi there!'}, 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 12}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.chat(messages)
            
            assert response.text == 'Hi there!'
            assert response.metadata['tokens'] == 12
            
            # Verify API call was made with correct format
            call_args = mock_call.call_args[0][0]
            assert call_args['messages'] == messages
    
    @pytest.mark.asyncio
    async def test_chat_conversation_history(self, initialized_engine):
        """
        Tests that the chat method correctly handles a conversation history, sending all messages and returning the expected assistant response.
        """
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]
        mock_response = {
            'choices': [{'message': {'content': '3+3 equals 6.'}, 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 30}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.chat(messages)
            assert response.text == '3+3 equals 6.'
            assert len(mock_call.call_args[0][0]['messages']) == 3
    
    @pytest.mark.asyncio
    async def test_chat_system_message(self, initialized_engine):
        """
        Tests that the chat method correctly handles a conversation including a system message and returns the expected assistant response.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        mock_response = {
            'choices': [{'message': {'content': 'Hello! How can I assist you?'}, 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 18}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.chat(messages)
            assert 'assist' in response.text.lower()
    
    @pytest.mark.asyncio
    async def test_chat_empty_messages(self, initialized_engine):
        """
        Test that calling chat with an empty messages list raises a ValueError.
        """
        with pytest.raises(ValueError, match="Messages cannot be empty"):
            await initialized_engine.chat([])
    
    @pytest.mark.asyncio
    async def test_chat_uninitialized_engine(self, llm_engine):
        """
        Test that calling `chat` on an uninitialized engine raises a RuntimeError.
        """
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            await llm_engine.chat(messages)
    
    @pytest.mark.asyncio
    async def test_chat_invalid_message_missing_role(self, initialized_engine):
        """
        Test that the chat method raises a ValueError when a message is missing the 'role' field.
        """
        messages = [{"content": "Missing role"}]
        
        with pytest.raises(ValueError, match="Each message must have 'role' and 'content' fields"):
            await initialized_engine.chat(messages)
    
    @pytest.mark.asyncio
    async def test_chat_invalid_message_missing_content(self, initialized_engine):
        """
        Test that the chat method raises a ValueError when a message is missing the 'content' field.
        """
        messages = [{"role": "user"}]
        
        with pytest.raises(ValueError, match="Each message must have 'role' and 'content' fields"):
            await initialized_engine.chat(messages)
    
    @pytest.mark.asyncio
    async def test_chat_invalid_role(self, initialized_engine):
        """
        Test that providing a message with an invalid role to the chat method raises a ValueError.
        
        Parameters:
            initialized_engine: An initialized LLMEngine instance used for testing.
        """
        messages = [{"role": "invalid_role", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="Invalid role: invalid_role"):
            await initialized_engine.chat(messages)
    
    # API Call and Error Handling Tests
    @pytest.mark.asyncio
    async def test_make_api_call_success(self, initialized_engine):
        """
        Tests that a successful API call returns the expected JSON response.
        
        Verifies that the engine's internal API call method correctly processes a valid response and that the HTTP POST request is made once.
        """
        payload = {"model": "test", "prompt": "test"}
        expected_response = {"choices": [{"text": "response"}]}
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=expected_response)
        mock_response.raise_for_status = Mock()
        
        with patch.object(initialized_engine.session, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            response = await initialized_engine._make_api_call(payload)
            
            assert response == expected_response
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_api_call_retry_on_client_error(self, initialized_engine):
        """
        Test that the API call method retries on client errors and succeeds after multiple failures.
        
        Simulates two consecutive client errors followed by a successful response, verifying that the retry logic is triggered and the final response is returned as expected.
        """
        payload = {"model": "test", "prompt": "test"}
        
        with patch.object(initialized_engine.session, 'post') as mock_post:
            # First two calls fail, third succeeds
            mock_post.side_effect = [
                aiohttp.ClientError("Network error"),
                aiohttp.ClientError("Another error"),
                AsyncMock()
            ]
            
            # Configure the successful response
            success_response = AsyncMock()
            success_response.status = 200
            success_response.json = AsyncMock(return_value={"choices": [{"text": "success"}]})
            success_response.raise_for_status = Mock()
            mock_post.side_effect[2].__aenter__.return_value = success_response
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                response = await initialized_engine._make_api_call(payload)
                
                assert response == {"choices": [{"text": "success"}]}
                assert mock_post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_make_api_call_max_retries_exceeded(self, initialized_engine):
        """
        Test that the API call retries the maximum allowed times and raises an error if all attempts fail.
        
        Verifies that when persistent client errors occur, the engine retries the API call up to the configured maximum and then raises the last exception.
        """
        payload = {"model": "test", "prompt": "test"}
        
        with patch.object(initialized_engine.session, 'post') as mock_post:
            mock_post.side_effect = aiohttp.ClientError("Persistent failure")
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(aiohttp.ClientError, match="Persistent failure"):
                    await initialized_engine._make_api_call(payload)
                
                # Should retry 3 times + initial attempt = 4 calls
                assert mock_post.call_count == 4
    
    @pytest.mark.asyncio
    async def test_make_api_call_timeout_error(self, initialized_engine):
        """
        Test that a timeout error during an API call raises an asyncio.TimeoutError.
        
        Verifies that when the engine's session post method raises a timeout, the exception is propagated by _make_api_call.
        """
        payload = {"model": "test", "prompt": "test"}
        
        with patch.object(initialized_engine.session, 'post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Request timeout")
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(asyncio.TimeoutError):
                    await initialized_engine._make_api_call(payload)
    
    @pytest.mark.asyncio
    async def test_make_api_call_rate_limit_handling(self, initialized_engine):
        """
        Test that the API call method correctly handles HTTP 429 rate limit errors by applying exponential backoff and retrying the request.
        
        Verifies that after receiving a rate limit response, the method waits for the appropriate backoff period before retrying, and ultimately returns the successful response.
        """
        payload = {"model": "test", "prompt": "test"}
        
        # Mock rate limit response
        rate_limit_response = AsyncMock()
        rate_limit_response.status = 429
        
        # Mock successful response
        success_response = AsyncMock()
        success_response.status = 200
        success_response.json = AsyncMock(return_value={"choices": [{"text": "success"}]})
        success_response.raise_for_status = Mock()
        
        with patch.object(initialized_engine.session, 'post') as mock_post:
            mock_post.side_effect = [
                AsyncMock(__aenter__=AsyncMock(return_value=rate_limit_response)),
                AsyncMock(__aenter__=AsyncMock(return_value=success_response))
            ]
            
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                response = await initialized_engine._make_api_call(payload)
                
                assert response == {"choices": [{"text": "success"}]}
                mock_sleep.assert_called_once_with(1)  # 2^0 = 1 second backoff
    
    # Response Validation Tests
    @pytest.mark.asyncio
    async def test_generate_no_choices_in_response(self, initialized_engine):
        """
        Test that the generate method raises a ValueError when the API response contains no choices.
        """
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"usage": {"total_tokens": 0}}
            
            with pytest.raises(ValueError, match="No response choices received"):
                await initialized_engine.generate("test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_empty_choices_in_response(self, initialized_engine):
        """
        Test that the generate method raises a ValueError when the API response contains an empty choices list.
        """
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"choices": [], "usage": {"total_tokens": 0}}
            
            with pytest.raises(ValueError, match="No response choices received"):
                await initialized_engine.generate("test prompt")
    
    @pytest.mark.asyncio
    async def test_chat_no_choices_in_response(self, initialized_engine):
        """
        Test that the chat method raises a ValueError when the API response contains no choices.
        
        Verifies that the engine correctly detects and handles responses missing the 'choices' field by raising an appropriate error.
        """
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"usage": {"total_tokens": 0}}
            
            with pytest.raises(ValueError, match="No response choices received"):
                await initialized_engine.chat(messages)
    
    @pytest.mark.asyncio
    async def test_generate_missing_text_in_choice(self, initialized_engine):
        """
        Test that the generate method returns an empty string when the API response choice lacks a text field.
        """
        mock_response = {
            'choices': [{'finish_reason': 'stop'}],  # Missing 'text' field
            'usage': {'total_tokens': 10}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.generate("test")
            assert response.text == ""  # Should default to empty string
    
    @pytest.mark.asyncio
    async def test_chat_missing_message_content(self, initialized_engine):
        """
        Test that the chat method returns an empty string when the API response message lacks content.
        
        Verifies that if the API response contains a choice with a message missing the 'content' field, the returned LLMResponse has an empty text value.
        """
        messages = [{"role": "user", "content": "Hello"}]
        mock_response = {
            'choices': [{'message': {}, 'finish_reason': 'stop'}],  # Missing 'content'
            'usage': {'total_tokens': 10}
        }
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await initialized_engine.chat(messages)
            assert response.text == ""  # Should default to empty string
    
    # Performance and Concurrency Tests
    @pytest.mark.asyncio
    async def test_concurrent_generate_requests(self, initialized_engine):
        """
        Test that the engine can handle multiple concurrent text generation requests and returns correct responses for each prompt.
        """
        prompts = [f"Prompt {i}" for i in range(5)]
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [
                {
                    'choices': [{'text': f'Response {i}', 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 10 + i}
                }
                for i in range(5)
            ]
            
            # Execute requests concurrently
            tasks = [initialized_engine.generate(prompt) for prompt in prompts]
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 5
            for i, response in enumerate(responses):
                assert response.text == f'Response {i}'
                assert response.metadata['tokens'] == 10 + i
            
            assert mock_call.call_count == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self, initialized_engine):
        """
        Test that the engine can handle multiple concurrent chat requests and returns correct responses for each.
        
        This test simulates three simultaneous chat calls with different user messages and verifies that each receives the expected answer.
        """
        message_sets = [
            [{"role": "user", "content": f"Question {i}"}]
            for i in range(3)
        ]
        
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [
                {
                    'choices': [{'message': {'content': f'Answer {i}'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 15 + i}
                }
                for i in range(3)
            ]
            
            tasks = [initialized_engine.chat(messages) for messages in message_sets]
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 3
            for i, response in enumerate(responses):
                assert response.text == f'Answer {i}'
    
    # Resource Management and Cleanup Tests  
    def test_engine_shutdown(self, llm_engine):
        """
        Test that the engine's shutdown method resets initialization state and schedules session closure asynchronously.
        """
        # Mock a session
        mock_session = AsyncMock()
        llm_engine.session = mock_session
        llm_engine.is_initialized = True
        
        with patch('asyncio.create_task') as mock_create_task:
            llm_engine.shutdown()
            
            assert not llm_engine.is_initialized
            mock_create_task.assert_called_once()
    
    def test_engine_shutdown_when_not_initialized(self, llm_engine):
        """
        Test that shutting down the engine when it was never initialized does not raise an error and leaves the engine uninitialized.
        """
        assert not llm_engine.is_initialized
        assert llm_engine.session is None
        
        # Should not raise error
        llm_engine.shutdown()
        assert not llm_engine.is_initialized
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, initialized_engine):
        """
        Simulates a multi-turn chat conversation using the engine and verifies correct responses and API call sequencing.
        
        This test mocks API responses to emulate a user-assistant conversation, updating the message history at each turn and asserting that the returned responses and call counts match expectations.
        """
        with patch.object(initialized_engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            # Setup responses for a conversation
            mock_call.side_effect = [
                {
                    'choices': [{'message': {'content': 'Hello! How can I help you?'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 15}
                },
                {
                    'choices': [{'message': {'content': 'I can help with math problems.'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 20}
                },
                {
                    'choices': [{'message': {'content': '2 + 2 = 4'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 25}
                }
            ]
            
            # Start conversation
            messages = [{"role": "user", "content": "Hello"}]
            response1 = await initialized_engine.chat(messages)
            
            # Continue conversation
            messages.extend([
                {"role": "assistant", "content": response1.text},
                {"role": "user", "content": "Can you help with math?"}
            ])
            response2 = await initialized_engine.chat(messages)
            
            # Ask math question
            messages.extend([
                {"role": "assistant", "content": response2.text},
                {"role": "user", "content": "What's 2+2?"}
            ])
            response3 = await initialized_engine.chat(messages)
            
            assert "Hello! How can I help you?" in response1.text
            assert "math" in response2.text.lower()
            assert "4" in response3.text
            assert mock_call.call_count == 3


# Parametrized Tests for Different Configurations
class TestLLMEngineParametrized:
    """Parametrized tests for different model configurations."""
    
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_config_temperature_boundary_values(self, temperature):
        """
        Test that LLMConfig correctly accepts and stores valid boundary values for the temperature parameter.
        
        Parameters:
            temperature (float): A valid temperature value to test.
        """
        config = LLMConfig(
            model_name="test-model",
            api_key="test-key",
            temperature=temperature
        )
        assert config.temperature == temperature
    
    @pytest.mark.parametrize("max_tokens", [1, 10, 100, 1000, 4000])
    def test_config_max_tokens_values(self, max_tokens):
        """
        Test that LLMConfig correctly sets the max_tokens parameter for various values.
        
        Parameters:
            max_tokens (int): The maximum number of tokens to set in the configuration.
        """
        config = LLMConfig(
            model_name="test-model",
            api_key="test-key",
            max_tokens=max_tokens
        )
        assert config.max_tokens == max_tokens
    
    @pytest.mark.parametrize("retries", [0, 1, 3, 5, 10])
    def test_config_retry_values(self, retries):
        """
        Test that `LLMConfig` correctly sets the `retries` parameter for various input values.
        """
        config = LLMConfig(
            model_name="test-model",
            api_key="test-key",
            retries=retries
        )
        assert config.retries == retries
    
    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    @pytest.mark.asyncio
    async def test_chat_valid_roles(self, role):
        """
        Asynchronously tests the chat method of LLMEngine with a single message using a valid role.
        
        Parameters:
            role (str): The role to use in the chat message (e.g., 'user', 'assistant', 'system').
        """
        config = LLMConfig(model_name="test", api_key="key")
        engine = LLMEngine(config)
        
        with patch('aiohttp.ClientSession'):
            await engine.initialize()
            
            messages = [{"role": role, "content": "Test message"}]
            mock_response = {
                'choices': [{'message': {'content': 'Response'}, 'finish_reason': 'stop'}],
                'usage': {'total_tokens': 10}
            }
            
            with patch.object(engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = mock_response
                
                response = await engine.chat(messages)
                assert response.text == 'Response'
            
            engine.shutdown()
    
    @pytest.mark.parametrize("invalid_role", ["moderator", "admin", "bot", ""])
    @pytest.mark.asyncio
    async def test_chat_invalid_roles(self, invalid_role):
        """
        Test that providing an invalid role in chat messages raises a ValueError.
        
        Parameters:
        	invalid_role (str): The role value to test, expected to be invalid.
        """
        config = LLMConfig(model_name="test", api_key="key")
        engine = LLMEngine(config)
        
        with patch('aiohttp.ClientSession'):
            await engine.initialize()
            
            messages = [{"role": invalid_role, "content": "Test message"}]
            
            with pytest.raises(ValueError, match=f"Invalid role: {invalid_role}"):
                await engine.chat(messages)
            
            engine.shutdown()


# Edge Case and Stress Tests
class TestLLMEngineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    async def engine(self):
        """
        Asynchronous fixture that yields an initialized LLMEngine instance for edge case testing.
        
        The engine is created with a test configuration and initialized with a mocked aiohttp.ClientSession. After yielding, the engine is shut down.
        """
        config = LLMConfig(model_name="test", api_key="key")
        engine = LLMEngine(config)
        
        with patch('aiohttp.ClientSession'):
            await engine.initialize()
            yield engine
            engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_generate_with_newlines_and_tabs(self, engine):
        """
        Tests that the engine can generate responses when the prompt contains newlines and tab characters, and that the response preserves newlines in the output.
        """
        prompt = "Line 1\nLine 2\tTabbed content\n\nEmpty line above"
        mock_response = {
            'choices': [{'text': 'Response\nwith\nnewlines', 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 20}
        }
        
        with patch.object(engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await engine.generate(prompt)
            assert '\n' in response.text
    
    @pytest.mark.asyncio
    async def test_chat_with_very_long_conversation(self, engine):
        """
        Tests the chat functionality with a conversation history of 50 alternating user and assistant messages.
        
        Verifies that the engine correctly handles long message histories, passes all messages to the API call, and returns the expected response.
        """
        # Create a conversation with many messages
        messages = []
        for i in range(50):  # Long conversation
            messages.append({"role": "user" if i % 2 == 0 else "assistant", 
                           "content": f"Message {i}"})
        
        mock_response = {
            'choices': [{'message': {'content': 'Final response'}, 'finish_reason': 'stop'}],
            'usage': {'total_tokens': 1000}
        }
        
        with patch.object(engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await engine.chat(messages)
            assert response.text == 'Final response'
            
            # Verify all messages were passed
            call_args = mock_call.call_args[0][0]
            assert len(call_args['messages']) == 50
    
    @pytest.mark.asyncio
    async def test_metadata_with_missing_usage_info(self, engine):
        """
        Test that response metadata defaults token count to 0 when the API response lacks usage information.
        
        Verifies that the 'tokens' field in the response metadata is set to 0 and 'finish_reason' is correctly populated when the 'usage' field is missing from the API response.
        """
        mock_response = {
            'choices': [{'text': 'Response without usage', 'finish_reason': 'stop'}]
            # Missing 'usage' field
        }
        
        with patch.object(engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await engine.generate("test")
            assert response.metadata['tokens'] == 0  # Should default to 0
            assert response.metadata['finish_reason'] == 'stop'
    
    @pytest.mark.asyncio
    async def test_metadata_with_partial_usage_info(self, engine):
        """
        Test that the engine defaults token count to 0 in response metadata when 'total_tokens' is missing from usage information.
        """
        mock_response = {
            'choices': [{'text': 'Response', 'finish_reason': 'length'}],
            'usage': {'prompt_tokens': 10}  # Missing 'total_tokens'
        }
        
        with patch.object(engine, '_make_api_call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            
            response = await engine.generate("test")
            assert response.metadata['tokens'] == 0  # Should default when total_tokens missing
            assert response.metadata['finish_reason'] == 'length'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])