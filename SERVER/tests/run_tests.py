#!/usr/bin/env python3
"""
Test runner for CSM tests.
Usage: python run_tests.py [test_class_name]
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all test classes
from test_csm import (
    TestCSMInitialization,
    TestCSMProcessStory,
    TestCSMErrorHandling,
    TestCSMShutdown,
    TestCSMEdgeCases,
    TestCSMPerformance,
    TestCSMIntegration
)

def run_all_tests():
    """Run all CSM tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCSMInitialization,
        TestCSMProcessStory,
        TestCSMErrorHandling,
        TestCSMShutdown,
        TestCSMEdgeCases,
        TestCSMPerformance,
        TestCSMIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_class_name):
    """Run a specific test class."""
    test_classes = {
        'initialization': TestCSMInitialization,
        'process_story': TestCSMProcessStory,
        'error_handling': TestCSMErrorHandling,
        'shutdown': TestCSMShutdown,
        'edge_cases': TestCSMEdgeCases,
        'performance': TestCSMPerformance,
        'integration': TestCSMIntegration
    }
    
    if test_class_name.lower() in test_classes:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_classes[test_class_name.lower()])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    else:
        print(f"Unknown test class: {test_class_name}")
        print("Available test classes:")
        for name in test_classes.keys():
            print(f"  - {name}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        success = run_specific_test(sys.argv[1])
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)