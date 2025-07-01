#!/bin/bash
"""
Test runner script for CharacterClient tests.
Runs all test suites with coverage reporting.
"""

echo "Running CharacterClient tests..."
echo "================================="

# Change to the test directory
cd "$(dirname "$0")"

# Run tests with pytest
python -m pytest -v --tb=short --cov=../src --cov-report=term-missing --cov-report=html

echo "================================="
echo "Test run completed!"