.PHONY: test test-unit test-performance test-integration test-coverage clean install-test-deps

# Default test target
test:
	cd tests && python run_tests.py

# Run unit tests only
test-unit:
	cd tests && python run_tests.py --type unit

# Run performance tests
test-performance:
	cd tests && python run_tests.py --type performance

# Run integration tests
test-integration:
	cd tests && python run_tests.py --type integration

# Run tests with coverage
test-coverage:
	cd tests && python run_tests.py --framework pytest --coverage

# Install test dependencies
install-test-deps:
	pip install -r tests/requirements-test.txt

# Clean test artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

# Lint code
lint:
	flake8 character_client.py exceptions.py utils.py
	black --check character_client.py exceptions.py utils.py
	isort --check-only character_client.py exceptions.py utils.py

# Format code
format:
	black character_client.py exceptions.py utils.py
	isort character_client.py exceptions.py utils.py

# Type check
typecheck:
	mypy character_client.py exceptions.py utils.py