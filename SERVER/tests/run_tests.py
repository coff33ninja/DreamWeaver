#!/usr/bin/env python3
"""
Test runner script for Character Server tests
Testing Framework: pytest

This script provides various test execution options:
- Run all tests
- Run specific test categories
- Generate coverage reports
- Run performance benchmarks
"""

import sys
import subprocess
import argparse
import os

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ Command completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description="Character Server Test Runner")
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    
    args = parser.parse_args()
    
    # Change to tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)
    
    base_cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        base_cmd.extend(['-v', '--tb=long'])
    
    success = True
    
    if args.all or not any([args.unit, args.integration, args.performance]):
        # Run all tests
        cmd = base_cmd + ['test_character_server.py', 'test_character_server_integration.py']
        if args.fast:
            cmd.extend(['-m', 'not slow'])
        if args.coverage:
            cmd.extend(['--cov=character_server', '--cov-report=html', '--cov-report=term-missing'])
        
        success &= run_command(cmd, "All Character Server Tests")
    
    if args.unit:
        # Run unit tests only
        cmd = base_cmd + ['test_character_server.py']
        if args.fast:
            cmd.extend(['-m', 'not slow'])
        
        success &= run_command(cmd, "Unit Tests")
    
    if args.integration:
        # Run integration tests only
        cmd = base_cmd + ['test_character_server_integration.py', '-m', 'integration']
        
        success &= run_command(cmd, "Integration Tests")
    
    if args.performance:
        # Run performance tests
        cmd = base_cmd + ['-m', 'performance', '--durations=20']
        
        success &= run_command(cmd, "Performance Tests")
    
    if args.coverage and not args.all:
        # Generate coverage report separately
        cmd = base_cmd + [
            '--cov=character_server',
            '--cov-report=html:htmlcov',
            '--cov-report=term-missing',
            '--cov-report=xml'
        ]
        if args.unit:
            cmd.append('test_character_server.py')
        elif args.integration:
            cmd.append('test_character_server_integration.py')
        else:
            cmd.extend(['test_character_server.py', 'test_character_server_integration.py'])
        
        success &= run_command(cmd, "Coverage Report Generation")
    
    if success:
        print(f"\n🎉 All requested tests completed successfully!")
        return 0
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())