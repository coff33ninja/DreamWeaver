#!/usr/bin/env python3
"""
Test runner script for CharacterClient tests.
Usage: python run_tests.py [options]
"""

import sys
import pytest
import os

def main():
    """Run the test suite with appropriate configuration."""
    # Change to the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)
    
    # Default pytest arguments
    args = [
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--strict-markers',  # Strict marker checking
        '--disable-warnings',  # Disable warnings
        'test_character_client.py'  # Specific test file
    ]
    
    # Add any command line arguments
    args.extend(sys.argv[1:])
    
    # Run pytest
    exit_code = pytest.main(args)
    
    print(f"\nTests completed with exit code: {exit_code}")
    sys.exit(exit_code)

if __name__ == '__main__':
    main()