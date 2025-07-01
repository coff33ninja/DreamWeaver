# LLM Engine Tests

This directory contains comprehensive unit tests for the LLM Engine module.

## Test Framework

The tests use **pytest** as the primary testing framework, with the following key features:
- Async test support with `pytest-asyncio`
- Comprehensive mocking with `unittest.mock`
- Parametrized tests for different configurations
- Fixtures for setup and teardown

## Test Structure

### Test Classes

1. **TestLLMConfig**: Tests for configuration validation
2. **TestLLMResponse**: Tests for response object functionality  
3. **TestLLMEngine**: Main test suite for engine functionality
4. **TestLLMEngineParametrized**: Parametrized tests for different configurations
5. **TestLLMEngineEdgeCases**: Edge cases and boundary condition tests

### Test Coverage

The test suite covers:

#### Configuration Tests
- Valid configuration creation
- Invalid parameter validation (temperature, max_tokens)
- Boundary value testing

#### Engine Lifecycle Tests
- Engine initialization and shutdown
- Double initialization handling
- Async context manager usage

#### Text Generation Tests
- Basic prompt generation
- Custom parameter handling
- Empty/invalid prompt validation
- Long prompt handling
- Special character support

#### Chat Functionality Tests
- Single message chat
- Conversation history handling
- System message support
- Message validation (role/content)
- Invalid role handling

#### Error Handling Tests
- Network error retry logic
- Rate limiting with exponential backoff
- Timeout handling
- Max retries exceeded scenarios
- Malformed API response handling

#### Performance Tests
- Concurrent request handling
- Resource cleanup
- Performance metrics tracking

#### Edge Cases
- Unicode and special characters
- Very long conversations
- Missing metadata handling
- Partial response handling

## Running Tests

### Run All Tests

git add .