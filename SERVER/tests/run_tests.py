#!/usr/bin/env python3
"""
Test runner script for client_manager tests.
Usage: python run_tests.py
"""

import sys
import os
import unittest

# Add the SERVER directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if __name__ == '__main__':
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    # Run tests
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)