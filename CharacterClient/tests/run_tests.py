#!/usr/bin/env python3
"""
Test runner script for CharacterClient tests.

This script provides multiple ways to run the test suite:
- All tests with coverage
- Specific test categories
- Performance tests
- Integration tests
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unittest_suite(pattern="test_*.py", verbosity=2):
    """Run tests using unittest framework."""
    print(f"Running tests with unittest (pattern: {pattern})...")

    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    suite = loader.discover(str(test_dir), pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True, failfast=False)

    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    return result.wasSuccessful()


def run_pytest_suite(args=None):
    """Run tests using pytest framework."""
    try:
        import pytest
    except ImportError:
        print("pytest not available, please install it: pip install pytest")
        return False

    print("Running tests with pytest...")

    pytest_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Treat unknown markers as errors
        str(Path(__file__).parent),  # Test directory
    ]

    # Add coverage if available
    try:
        import pytest_cov

        pytest_args.extend(["--cov=CharacterClient", "--cov-report=term-missing"])
    except ImportError:
        print("Coverage not available, install with: pip install pytest-cov")

    # Add custom arguments
    if args:
        pytest_args.extend(args)

    try:
        exit_code = pytest.main(pytest_args)
        return exit_code == 0
    except SystemExit as e:
        return e.code == 0


def run_performance_tests():
    """Run only performance tests."""
    print("Running performance tests...")
    return run_unittest_suite(pattern="test_*_performance.py")


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    # Integration tests are included in the performance test file
    return run_unittest_suite(pattern="test_*_performance.py")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="CharacterClient Test Runner")
    parser.add_argument(
        "--framework",
        choices=["unittest", "pytest", "auto"],
        default="auto",
        help="Testing framework to use",
    )
    parser.add_argument(
        "--type",
        choices=["all", "unit", "performance", "integration"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report (pytest only)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("CharacterClient Test Suite")
    print("=" * 50)

    success = False

    # Determine test type
    if args.type == "performance":
        success = run_performance_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    else:
        # Run all tests or unit tests
        if args.framework == "pytest":
            pytest_args = []
            if args.coverage:
                pytest_args.extend(["--cov=CharacterClient", "--cov-report=html"])
            success = run_pytest_suite(pytest_args)
        elif args.framework == "unittest":
            verbosity = 2 if args.verbose else 1
            success = run_unittest_suite(verbosity=verbosity)
        else:  # auto
            # Try pytest first, fall back to unittest
            try:
                import pytest

                success = run_pytest_suite()
            except ImportError:
                print("pytest not available, falling back to unittest...")
                success = run_unittest_suite()

    # Print final result
    if success:
        print(f"\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
