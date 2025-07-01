import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from typing import Dict, Any, List, Optional
import tempfile
import os
import threading
import time

# Testing Framework: pytest
# This file contains comprehensive unit tests for the LLM Engine module


class TestLLMEngineConfiguration:
    """Test configuration and initialization scenarios."""
    
    def test_engine_initialization_with_valid_config(self):
        """Test engine initializes correctly with valid configuration."""
        config = {
            'model_name': 'test-model',
            'api_key': 'test-key',
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            engine = mock_class(config)
            mock_class.assert_called_once_with(config)
            assert engine is not None
    
    def test_engine_initialization_with_missing_required_config(self):
        """Test engine handles missing required configuration gracefully."""
        incomplete_config = {'temperature': 0.7}
        
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_class.side_effect = ValueError("Missing required configuration")
            
            with pytest.raises(ValueError, match="Missing required configuration"):
                mock_class(incomplete_config)
    
    def test_engine_initialization_with_invalid_config_types(self):
        """Test engine validates configuration parameter types."""
        invalid_config = {
            'model_name': 123,  # Should be string
            'max_tokens': 'invalid',  # Should be int
            'temperature': 'invalid'  # Should be float
        }
        
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_class.side_effect = TypeError("Invalid configuration types")
            
            with pytest.raises(TypeError, match="Invalid configuration types"):
                mock_class(invalid_config)
    
    @pytest.mark.parametrize("temperature,max_tokens,should_raise", [
        (0.0, 1, False),        # Valid boundary values
        (2.0, 4096, False),     # Valid boundary values
        (-0.1, 1, True),        # Invalid temperature
        (2.1, 1, True),         # Invalid temperature
        (0.5, 0, True),         # Invalid max_tokens
        (0.5, -1, True),        # Invalid max_tokens
    ])
    def test_engine_initialization_with_boundary_values(self, temperature, max_tokens, should_raise):
        """Test engine handles boundary values in configuration."""
        config = {
            'model_name': 'test',
            'api_key': 'test',
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            if should_raise:
                mock_class.side_effect = ValueError("Invalid parameter values")
                with pytest.raises(ValueError):
                    mock_class(config)
            else:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                engine = mock_class(config)
                assert engine is not None


class TestLLMEngineTextGeneration:
    """Test text generation functionality with various scenarios."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock LLM engine for testing."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance
    
    def test_generate_text_happy_path(self, mock_engine):
        """Test successful text generation with standard input."""
        mock_engine.generate_text.return_value = "Generated response text"
        
        result = mock_engine.generate_text("Test prompt")
        
        assert result == "Generated response text"
        mock_engine.generate_text.assert_called_once_with("Test prompt")
    
    def test_generate_text_with_empty_prompt(self, mock_engine):
        """Test text generation with empty prompt."""
        mock_engine.generate_text.side_effect = ValueError("Prompt cannot be empty")
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            mock_engine.generate_text("")
    
    def test_generate_text_with_none_prompt(self, mock_engine):
        """Test text generation with None prompt."""
        mock_engine.generate_text.side_effect = TypeError("Prompt cannot be None")
        
        with pytest.raises(TypeError, match="Prompt cannot be None"):
            mock_engine.generate_text(None)
    
    def test_generate_text_with_very_long_prompt(self, mock_engine):
        """Test text generation with extremely long prompt."""
        long_prompt = "A" * 10000
        mock_engine.generate_text.side_effect = ValueError("Prompt too long")
        
        with pytest.raises(ValueError, match="Prompt too long"):
            mock_engine.generate_text(long_prompt)
    
    def test_generate_text_with_special_characters(self, mock_engine):
        """Test text generation with special characters and unicode."""
        special_prompt = "Test with émojis 🚀 and symbols !@#$%^&*()[]{}|;':\",./<>?"
        mock_engine.generate_text.return_value = "Response with special chars"
        
        result = mock_engine.generate_text(special_prompt)
        
        assert result == "Response with special chars"
        mock_engine.generate_text.assert_called_once_with(special_prompt)
    
    def test_generate_text_with_multiline_prompt(self, mock_engine):
        """Test text generation with multiline prompt."""
        multiline_prompt = """This is a
        multiline prompt
        with various indentations
            and formatting"""
        mock_engine.generate_text.return_value = "Multiline response"
        
        result = mock_engine.generate_text(multiline_prompt)
        
        assert result == "Multiline response"
        mock_engine.generate_text.assert_called_once_with(multiline_prompt)
    
    def test_generate_text_with_parameters(self, mock_engine):
        """Test text generation with additional parameters."""
        mock_engine.generate_text.return_value = "Parameterized response"
        
        result = mock_engine.generate_text(
            "Test prompt",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9
        )
        
        assert result == "Parameterized response"
        mock_engine.generate_text.assert_called_once_with(
            "Test prompt",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9
        )


class TestLLMEngineAsyncOperations:
    """Test asynchronous operations if the engine supports them."""
    
    @pytest.fixture
    def mock_async_engine(self):
        """Create a mock async LLM engine for testing."""
        mock_engine = AsyncMock()
        return mock_engine
    
    @pytest.mark.asyncio
    async def test_async_generate_text_success(self, mock_async_engine):
        """Test successful asynchronous text generation."""
        mock_async_engine.generate_text_async.return_value = "Async generated text"
        
        result = await mock_async_engine.generate_text_async("Test prompt")
        
        assert result == "Async generated text"
        mock_async_engine.generate_text_async.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_async_generate_text_timeout(self, mock_async_engine):
        """Test async text generation with timeout."""
        mock_async_engine.generate_text_async.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(asyncio.TimeoutError):
            await mock_async_engine.generate_text_async("Test prompt")
    
    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, mock_async_engine):
        """Test handling multiple concurrent async requests."""
        mock_async_engine.generate_text_async.side_effect = [
            "Response 1", "Response 2", "Response 3"
        ]
        
        tasks = [
            mock_async_engine.generate_text_async(f"Prompt {i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert results == ["Response 1", "Response 2", "Response 3"]
        assert mock_async_engine.generate_text_async.call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_cancellation(self, mock_async_engine):
        """Test async request cancellation."""
        mock_async_engine.generate_text_async.side_effect = asyncio.CancelledError()
        
        with pytest.raises(asyncio.CancelledError):
            await mock_async_engine.generate_text_async("Test prompt")


class TestLLMEngineErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for error testing."""
        mock_engine = Mock()
        return mock_engine
    
    def test_api_connection_error(self, mock_engine):
        """Test handling of API connection errors."""
        mock_engine.generate_text.side_effect = ConnectionError("Failed to connect to API")
        
        with pytest.raises(ConnectionError, match="Failed to connect to API"):
            mock_engine.generate_text("Test prompt")
    
    def test_api_authentication_error(self, mock_engine):
        """Test handling of authentication errors."""
        mock_engine.generate_text.side_effect = PermissionError("Invalid API key")
        
        with pytest.raises(PermissionError, match="Invalid API key"):
            mock_engine.generate_text("Test prompt")
    
    def test_api_rate_limit_error(self, mock_engine):
        """Test handling of rate limit errors."""
        mock_engine.generate_text.side_effect = Exception("Rate limit exceeded")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            mock_engine.generate_text("Test prompt")
    
    def test_malformed_response_handling(self, mock_engine):
        """Test handling of malformed API responses."""
        mock_engine.generate_text.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with pytest.raises(json.JSONDecodeError):
            mock_engine.generate_text("Test prompt")
    
    def test_none_response_handling(self, mock_engine):
        """Test handling when API returns None."""
        mock_engine.generate_text.return_value = None
        
        result = mock_engine.generate_text("Test prompt")
        
        assert result is None
    
    def test_empty_response_handling(self, mock_engine):
        """Test handling when API returns empty response."""
        mock_engine.generate_text.return_value = ""
        
        result = mock_engine.generate_text("Test prompt")
        
        assert result == ""
    
    def test_network_timeout_error(self, mock_engine):
        """Test handling of network timeout errors."""
        mock_engine.generate_text.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(TimeoutError, match="Request timed out"):
            mock_engine.generate_text("Test prompt")


class TestLLMEngineParameterValidation:
    """Test parameter validation and sanitization."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock engine for parameter testing."""
        return Mock()
    
    @pytest.mark.parametrize("temperature,expected", [
        (0.0, True),
        (0.5, True),
        (1.0, True),
        (1.5, True),
        (2.0, True),
        (-0.1, False),
        (2.1, False),
        ('invalid', False),
        (None, False),
    ])
    def test_temperature_validation(self, mock_engine, temperature, expected):
        """Test temperature parameter validation."""
        if expected:
            mock_engine.set_temperature.return_value = True
            result = mock_engine.set_temperature(temperature)
            assert result is True
        else:
            mock_engine.set_temperature.side_effect = ValueError(f"Invalid temperature: {temperature}")
            with pytest.raises(ValueError):
                mock_engine.set_temperature(temperature)
    
    @pytest.mark.parametrize("max_tokens,expected", [
        (1, True),
        (100, True),
        (1000, True),
        (4096, True),
        (0, False),
        (-1, False),
        ('invalid', False),
        (None, False),
        (999999, False),
    ])
    def test_max_tokens_validation(self, mock_engine, max_tokens, expected):
        """Test max_tokens parameter validation."""
        if expected:
            mock_engine.set_max_tokens.return_value = True
            result = mock_engine.set_max_tokens(max_tokens)
            assert result is True
        else:
            mock_engine.set_max_tokens.side_effect = ValueError(f"Invalid max_tokens: {max_tokens}")
            with pytest.raises(ValueError):
                mock_engine.set_max_tokens(max_tokens)
    
    @pytest.mark.parametrize("top_p,expected", [
        (0.1, True),
        (0.5, True),
        (0.9, True),
        (1.0, True),
        (0.0, False),
        (1.1, False),
        (-0.1, False),
        ('invalid', False),
        (None, False),
    ])
    def test_top_p_validation(self, mock_engine, top_p, expected):
        """Test top_p parameter validation."""
        if expected:
            mock_engine.set_top_p.return_value = True
            result = mock_engine.set_top_p(top_p)
            assert result is True
        else:
            mock_engine.set_top_p.side_effect = ValueError(f"Invalid top_p: {top_p}")
            with pytest.raises(ValueError):
                mock_engine.set_top_p(top_p)


class TestLLMEngineStateManagement:
    """Test state management and persistence."""
    
    def test_save_and_load_state(self):
        """Test saving and loading engine state."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Mock state data
            test_state = {'model': 'test-model', 'config': {'temp': 0.7}}
            mock_instance.get_state.return_value = test_state
            mock_instance.load_state.return_value = True
            
            # Test save state
            state = mock_instance.get_state()
            assert state == test_state
            
            # Test load state
            result = mock_instance.load_state(test_state)
            assert result is True
            mock_instance.load_state.assert_called_once_with(test_state)
    
    def test_state_persistence_to_file(self):
        """Test persisting state to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with patch('SERVER.llm_engine.LLMEngine') as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                
                test_state = {'model': 'test', 'settings': {}}
                mock_instance.save_state_to_file.return_value = True
                mock_instance.load_state_from_file.return_value = test_state
                
                # Test save to file
                result = mock_instance.save_state_to_file(tmp_path)
                assert result is True
                
                # Test load from file
                loaded_state = mock_instance.load_state_from_file(tmp_path)
                assert loaded_state == test_state
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_state_validation(self):
        """Test state validation during load operations."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Test invalid state
            invalid_state = {'invalid': 'data'}
            mock_instance.load_state.side_effect = ValueError("Invalid state format")
            
            with pytest.raises(ValueError, match="Invalid state format"):
                mock_instance.load_state(invalid_state)


class TestLLMEngineMemoryManagement:
    """Test memory management and resource cleanup."""
    
    def test_engine_cleanup_on_destruction(self):
        """Test that engine properly cleans up resources."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_instance.cleanup = Mock()
            mock_class.return_value = mock_instance
            
            engine = mock_class()
            engine.cleanup()
            
            mock_instance.cleanup.assert_called_once()
    
    def test_context_manager_support(self):
        """Test context manager support for resource management."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_instance.__enter__ = Mock(return_value=mock_instance)
            mock_instance.__exit__ = Mock(return_value=None)
            mock_class.return_value = mock_instance
            
            with mock_class() as engine:
                engine.generate_text("Test")
            
            mock_instance.__enter__.assert_called_once()
            mock_instance.__exit__.assert_called_once()
    
    def test_memory_leak_prevention(self):
        """Test prevention of memory leaks in long-running operations."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_instance.clear_cache = Mock()
            mock_class.return_value = mock_instance
            
            # Simulate clearing cache after operations
            mock_instance.clear_cache()
            
            mock_instance.clear_cache.assert_called_once()


class TestLLMEnginePerformanceMetrics:
    """Test performance monitoring and metrics collection."""
    
    @pytest.fixture
    def mock_engine_with_metrics(self):
        """Create mock engine with performance tracking."""
        mock_engine = Mock()
        mock_engine.get_metrics.return_value = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'total_tokens_generated': 0
        }
        return mock_engine
    
    def test_metrics_collection(self, mock_engine_with_metrics):
        """Test that performance metrics are collected correctly."""
        # Simulate some requests
        mock_engine_with_metrics.generate_text.return_value = "Response"
        mock_engine_with_metrics.get_metrics.return_value = {
            'total_requests': 5,
            'successful_requests': 4,
            'failed_requests': 1,
            'average_response_time': 1.25,
            'total_tokens_generated': 150
        }
        
        # Make some mock requests
        for i in range(4):
            mock_engine_with_metrics.generate_text(f"Prompt {i}")
        
        # Check metrics
        metrics = mock_engine_with_metrics.get_metrics()
        
        assert metrics['total_requests'] == 5
        assert metrics['successful_requests'] == 4
        assert metrics['failed_requests'] == 1
        assert metrics['average_response_time'] > 0
        assert metrics['total_tokens_generated'] > 0
    
    def test_metrics_reset(self, mock_engine_with_metrics):
        """Test metrics can be reset."""
        mock_engine_with_metrics.reset_metrics.return_value = True
        
        result = mock_engine_with_metrics.reset_metrics()
        assert result is True
        mock_engine_with_metrics.reset_metrics.assert_called_once()
    
    def test_response_time_tracking(self, mock_engine_with_metrics):
        """Test response time tracking functionality."""
        mock_engine_with_metrics.get_last_response_time.return_value = 2.5
        
        response_time = mock_engine_with_metrics.get_last_response_time()
        
        assert response_time == 2.5
        assert isinstance(response_time, (int, float))


class TestLLMEngineIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def integration_engine(self):
        """Create engine for integration testing."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance
    
    def test_complete_generation_workflow(self, integration_engine):
        """Test complete text generation workflow from start to finish."""
        # Setup mock responses
        integration_engine.initialize.return_value = True
        integration_engine.validate_prompt.return_value = True
        integration_engine.generate_text.return_value = "Generated response"
        integration_engine.post_process.return_value = "Final response"
        
        # Execute complete workflow
        assert integration_engine.initialize() is True
        assert integration_engine.validate_prompt("Test prompt") is True
        raw_response = integration_engine.generate_text("Test prompt")
        final_response = integration_engine.post_process(raw_response)
        
        assert raw_response == "Generated response"
        assert final_response == "Final response"
        
        # Verify all methods were called
        integration_engine.initialize.assert_called_once()
        integration_engine.validate_prompt.assert_called_once_with("Test prompt")
        integration_engine.generate_text.assert_called_once_with("Test prompt")
        integration_engine.post_process.assert_called_once_with("Generated response")
    
    def test_workflow_with_preprocessing(self, integration_engine):
        """Test workflow that includes prompt preprocessing."""
        integration_engine.preprocess_prompt.return_value = "Processed prompt"
        integration_engine.generate_text.return_value = "Response"
        
        original_prompt = "Raw prompt"
        processed_prompt = integration_engine.preprocess_prompt(original_prompt)
        response = integration_engine.generate_text(processed_prompt)
        
        assert processed_prompt == "Processed prompt"
        assert response == "Response"
        
        integration_engine.preprocess_prompt.assert_called_once_with(original_prompt)
        integration_engine.generate_text.assert_called_once_with("Processed prompt")
    
    def test_error_recovery_workflow(self, integration_engine):
        """Test workflow with error recovery mechanisms."""
        # First call fails, second succeeds
        integration_engine.generate_text.side_effect = [
            ConnectionError("Network error"),
            "Successful response"
        ]
        integration_engine.retry_request.return_value = "Successful response"
        
        # Simulate retry logic
        try:
            result = integration_engine.generate_text("Test prompt")
        except ConnectionError:
            result = integration_engine.retry_request("Test prompt")
        
        assert result == "Successful response"
        assert integration_engine.generate_text.call_count == 1
        integration_engine.retry_request.assert_called_once_with("Test prompt")
    
    def test_batch_processing_workflow(self, integration_engine):
        """Test batch processing of multiple prompts."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        expected_responses = ["Response 1", "Response 2", "Response 3"]
        
        integration_engine.generate_batch.return_value = expected_responses
        
        results = integration_engine.generate_batch(prompts)
        
        assert results == expected_responses
        integration_engine.generate_batch.assert_called_once_with(prompts)


class TestLLMEngineEdgeCases:
    """Test various edge cases and boundary conditions."""
    
    def test_unicode_handling(self):
        """Test proper handling of Unicode characters."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_instance.generate_text.return_value = "Unicode response: 测试 🎉"
            
            unicode_prompt = "Test Unicode: 你好世界 émojis 🚀🌟"
            result = mock_instance.generate_text(unicode_prompt)
            
            assert "Unicode response" in result
            assert "测试" in result
            mock_instance.generate_text.assert_called_once_with(unicode_prompt)
    
    def test_memory_intensive_operations(self):
        """Test handling of memory-intensive operations."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Mock memory error
            mock_instance.generate_text.side_effect = MemoryError("Insufficient memory")
            
            with pytest.raises(MemoryError, match="Insufficient memory"):
                mock_instance.generate_text("Large prompt" * 1000)
    
    def test_concurrent_access_safety(self):
        """Test thread safety for concurrent access."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_instance.generate_text.return_value = "Thread safe response"
            
            results = []
            
            def worker():
                result = mock_instance.generate_text("Concurrent prompt")
                results.append(result)
            
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            assert len(results) == 5
            assert all(r == "Thread safe response" for r in results)
    
    def test_large_response_handling(self):
        """Test handling of very large responses."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Mock large response
            large_response = "A" * 100000
            mock_instance.generate_text.return_value = large_response
            
            result = mock_instance.generate_text("Generate large text")
            
            assert len(result) == 100000
            assert result == large_response
    
    def test_special_token_handling(self):
        """Test handling of special tokens in prompts and responses."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            special_tokens = ["<|endoftext|>", "[MASK]", "<|im_start|>", "<|im_end|>"]
            mock_instance.generate_text.return_value = "Response with special tokens"
            
            for token in special_tokens:
                prompt = f"Process this token: {token}"
                result = mock_instance.generate_text(prompt)
                assert result == "Response with special tokens"
    
    def test_encoding_edge_cases(self):
        """Test various text encoding edge cases."""
        with patch('SERVER.llm_engine.LLMEngine') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_instance.generate_text.return_value = "Encoding handled correctly"
            
            # Test various encodings
            test_cases = [
                "ASCII text",
                "UTF-8 with accents: café naïve",
                "Emoji: 👍🎉🚀",
                "Math symbols: ∑∞∆∇",
                "Mixed: Hello 世界 🌍"
            ]
            
            for test_case in test_cases:
                result = mock_instance.generate_text(test_case)
                assert result == "Encoding handled correctly"


class TestLLMEngineStreamingOperations:
    """Test streaming text generation capabilities."""
    
    @pytest.fixture
    def mock_streaming_engine(self):
        """Create mock engine with streaming capabilities."""
        mock_engine = Mock()
        return mock_engine
    
    def test_streaming_text_generation(self, mock_streaming_engine):
        """Test streaming text generation functionality."""
        # Mock streaming response
        stream_chunks = ["Hello", " world", "!", " How", " are", " you?"]
        mock_streaming_engine.generate_text_stream.return_value = iter(stream_chunks)
        
        result_chunks = list(mock_streaming_engine.generate_text_stream("Test prompt"))
        
        assert result_chunks == stream_chunks
        mock_streaming_engine.generate_text_stream.assert_called_once_with("Test prompt")
    
    def test_streaming_with_interruption(self, mock_streaming_engine):
        """Test streaming with interruption handling."""
        def interrupted_stream():
            yield "Hello"
            yield " world"
            raise KeyboardInterrupt("Stream interrupted")
        
        mock_streaming_engine.generate_text_stream.return_value = interrupted_stream()
        
        chunks = []
        try:
            for chunk in mock_streaming_engine.generate_text_stream("Test prompt"):
                chunks.append(chunk)
        except KeyboardInterrupt:
            pass
        
        assert chunks == ["Hello", " world"]
    
    @pytest.mark.asyncio
    async def test_async_streaming(self, mock_streaming_engine):
        """Test asynchronous streaming functionality."""
        async def async_stream():
            for chunk in ["Async", " streaming", " test"]:
                yield chunk
        
        mock_streaming_engine.generate_text_stream_async.return_value = async_stream()
        
        chunks = []
        async for chunk in mock_streaming_engine.generate_text_stream_async("Test prompt"):
            chunks.append(chunk)
        
        assert chunks == ["Async", " streaming", " test"]


class TestLLMEngineLogging:
    """Test logging and debugging capabilities."""
    
    @pytest.fixture
    def mock_engine_with_logging(self):
        """Create mock engine with logging capabilities."""
        mock_engine = Mock()
        return mock_engine
    
    def test_request_logging(self, mock_engine_with_logging):
        """Test that requests are properly logged."""
        mock_engine_with_logging.enable_logging.return_value = True
        mock_engine_with_logging.get_logs.return_value = [
            {"timestamp": "2023-01-01T00:00:00", "level": "INFO", "message": "Request started"},
            {"timestamp": "2023-01-01T00:00:01", "level": "INFO", "message": "Request completed"}
        ]
        
        # Enable logging
        mock_engine_with_logging.enable_logging()
        
        # Check logs
        logs = mock_engine_with_logging.get_logs()
        
        assert len(logs) == 2
        assert logs[0]["level"] == "INFO"
        assert "Request started" in logs[0]["message"]
    
    def test_error_logging(self, mock_engine_with_logging):
        """Test that errors are properly logged."""
        mock_engine_with_logging.get_error_logs.return_value = [
            {"timestamp": "2023-01-01T00:00:00", "level": "ERROR", "message": "API connection failed"}
        ]
        
        error_logs = mock_engine_with_logging.get_error_logs()
        
        assert len(error_logs) == 1
        assert error_logs[0]["level"] == "ERROR"
        assert "API connection failed" in error_logs[0]["message"]
    
    def test_debug_mode(self, mock_engine_with_logging):
        """Test debug mode functionality."""
        mock_engine_with_logging.set_debug_mode.return_value = True
        mock_engine_with_logging.is_debug_enabled.return_value = True
        
        # Enable debug mode
        result = mock_engine_with_logging.set_debug_mode(True)
        assert result is True
        
        # Check debug status
        debug_status = mock_engine_with_logging.is_debug_enabled()
        assert debug_status is True


class TestLLMEngineConfiguration:
    """Test configuration management and updates."""
    
    @pytest.fixture
    def mock_configurable_engine(self):
        """Create mock engine with configuration capabilities."""
        mock_engine = Mock()
        return mock_engine
    
    def test_runtime_configuration_update(self, mock_configurable_engine):
        """Test updating configuration at runtime."""
        new_config = {"temperature": 0.8, "max_tokens": 2000}
        mock_configurable_engine.update_config.return_value = True
        
        result = mock_configurable_engine.update_config(new_config)
        
        assert result is True
        mock_configurable_engine.update_config.assert_called_once_with(new_config)
    
    def test_configuration_validation(self, mock_configurable_engine):
        """Test configuration validation."""
        invalid_config = {"temperature": 3.0, "max_tokens": -100}
        mock_configurable_engine.validate_config.return_value = False
        
        result = mock_configurable_engine.validate_config(invalid_config)
        
        assert result is False
        mock_configurable_engine.validate_config.assert_called_once_with(invalid_config)
    
    def test_configuration_persistence(self, mock_configurable_engine):
        """Test configuration persistence across sessions."""
        config = {"model": "test-model", "temperature": 0.7}
        mock_configurable_engine.save_config.return_value = True
        mock_configurable_engine.load_config.return_value = config
        
        # Save config
        save_result = mock_configurable_engine.save_config(config)
        assert save_result is True
        
        # Load config
        loaded_config = mock_configurable_engine.load_config()
        assert loaded_config == config