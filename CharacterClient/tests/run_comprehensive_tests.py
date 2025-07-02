#!/usr/bin/env python3
"""
Comprehensive test runner for TTS Manager tests.
Run with different test categories and configurations.
"""

import subprocess
import sys
import argparse


def run_tests(test_type="all", verbose=True, coverage=False):
    """Run tests with specified configuration."""
    
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=tts_manager", "--cov-report=html", "--cov-report=term"])
    
    # Test type selection
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "security":
        cmd.extend(["-m", "security"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "all":
        pass  # Run all tests
    
    cmd.append("test_tts_manager.py")
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd="CharacterClient/tests")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TTS Manager tests")
    parser.add_argument("--type", choices=["all", "unit", "integration", "performance", "security", "fast"], 
                       default="fast", help="Type of tests to run")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--quiet", action="store_true", help="Run with minimal output")
    
    args = parser.parse_args()
    
    result = run_tests(
        test_type=args.type, 
        verbose=not args.quiet, 
        coverage=args.coverage
    )
    
    sys.exit(result.returncode)