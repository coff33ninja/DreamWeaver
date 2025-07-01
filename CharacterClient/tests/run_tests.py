#!/usr/bin/env python3
"""
Test runner script for LLM Engine tests.
Run with: python run_tests.py
"""

import sys
import subprocess
import os

def run_tests():
    """
    Runs the LLM Engine test suite using pytest with coverage and reporting options.
    
    Returns:
        int: The exit code from the pytest process, or 1 if an error occurs during execution.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            os.path.join(test_dir, "test_llm_engine.py"),
            "-v", 
            "--tb=short",
            "--durations=10",
            "--cov=CharacterClient.src.llm_engine",
            "--cov-report=term-missing"
        ], cwd=os.path.dirname(test_dir))
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    print(f"\nTest run completed with exit code: {exit_code}")
    sys.exit(exit_code)