#!/usr/bin/env python3
"""
Test runner script for character server tests.
Provides different test execution modes and reporting options.
"""

import pytest
import sys
import argparse
from pathlib import Path


def run_basic_tests():
    """Run basic functionality tests."""
    return pytest.main([
        'test_character_server.py::TestCharacterCreation',
        'test_character_server.py::TestCharacterRetrieval',
        'test_character_server.py::TestCharacterUpdate',
        'test_character_server.py::TestCharacterDeletion',
        'test_character_server.py::TestCharacterListing',
        '-v'
    ])


def run_validation_tests():
    """Run validation and security tests."""
    return pytest.main([
        'test_character_server.py::TestCharacterValidation',
        'test_character_server.py::TestCharacterServerSecurityAndValidation',
        'test_character_server.py::TestCharacterServerAdvancedValidation',
        '-v',
        '-m', 'not slow'
    ])


def run_performance_tests():
    """Run performance and benchmark tests."""
    return pytest.main([
        'test_character_server.py::TestCharacterServerPerformance',
        'test_character_server.py::TestCharacterServerPerformanceBenchmarks',
        'test_character_server.py::TestCharacterServerStressAndPerformance',
        '-v',
        '-m', 'performance',
        '--durations=10'
    ])


def run_stress_tests():
    """Run stress and load tests."""
    return pytest.main([
        'test_character_server.py::TestCharacterServerStressAndPerformance',
        'test_character_server.py::TestCharacterServerMemoryAndResourceManagement',
        'test_character_server.py::TestCharacterServerRobustnessAndStability',
        '-v',
        '-m', 'stress',
        '--durations=20'
    ])


def run_comprehensive_tests():
    """Run all tests with comprehensive reporting."""
    return pytest.main([
        'test_character_server.py',
        '-v',
        '--tb=short',
        '--durations=20',
        '--strict-markers',
        '--cov=character_server',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Character Server Test Runner')
    parser.add_argument(
        'test_type',
        choices=['basic', 'validation', 'performance', 'stress', 'comprehensive', 'all'],
        help='Type of tests to run'
    )
    
    args = parser.parse_args()
    
    test_runners = {
        'basic': run_basic_tests,
        'validation': run_validation_tests,
        'performance': run_performance_tests,
        'stress': run_stress_tests,
        'comprehensive': run_comprehensive_tests,
        'all': run_comprehensive_tests
    }
    
    runner = test_runners.get(args.test_type)
    if runner:
        return runner()
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1


if __name__ == '__main__':
    sys.exit(main())