#!/usr/bin/env python3
"""
Test runner specifically for configuration tests.
"""
import unittest
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if __name__ == '__main__':
    # Discover and run config tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_config*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)