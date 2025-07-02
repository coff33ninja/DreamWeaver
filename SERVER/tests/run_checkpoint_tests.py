#!/usr/bin/env python3
"""
Comprehensive test runner for CheckpointManager tests.
Supports multiple execution modes and detailed reporting.
"""

import unittest
import sys
import time
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test classes
from test_checkpoint_manager import (
    TestCheckpointManager,
    TestCheckpointManagerEdgeCases,
    TestCheckpointManagerRealImplementation,
    TestCheckpointManagerAdditionalEdgeCases,
    TestCheckpointManagerMarkers,
    TestCheckpointManagerSecurityAndValidation,
    TestCheckpointManagerStressAndReliability,
    TestCheckpointManagerAdvancedScenarios
)


class ComprehensiveTestRunner:
    """Enhanced test runner with categorized execution and reporting."""
    
    def __init__(self):
        self.test_categories = {
            'basic': [TestCheckpointManager],
            'edge_cases': [TestCheckpointManagerEdgeCases, TestCheckpointManagerAdditionalEdgeCases],
            'real_implementation': [TestCheckpointManagerRealImplementation],
            'security': [TestCheckpointManagerSecurityAndValidation],
            'stress': [TestCheckpointManagerStressAndReliability],
            'advanced': [TestCheckpointManagerAdvancedScenarios],
            'markers': [TestCheckpointManagerMarkers]
        }
    
    def run_category(self, category_name, verbosity=2):
        """Run tests for a specific category."""
        if category_name not in self.test_categories:
            print(f"Unknown category: {category_name}")
            return False
        
        print(f"\n{'='*60}")
        print(f"RUNNING {category_name.upper()} TESTS")
        print(f"{'='*60}")
        
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        
        for test_class in self.test_categories[category_name]:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            buffer=True,
            failfast=False
        )
        
        start_time = time.time()
        result = runner.run(suite)
        duration = time.time() - start_time
        
        print(f"\n{category_name.upper()} TESTS COMPLETED in {duration:.2f}s")
        print(f"Tests: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
        
        return len(result.failures) == 0 and len(result.errors) == 0
    
    def run_all(self, verbosity=2):
        """Run all test categories."""
        print("="*70)
        print("COMPREHENSIVE CHECKPOINTMANAGER TEST SUITE")
        print("="*70)
        print(f"Testing Framework: unittest")
        print(f"Total Categories: {len(self.test_categories)}")
        print("="*70)
        
        overall_start = time.time()
        results = {}
        
        for category in self.test_categories.keys():
            success = self.run_category(category, verbosity)
            results[category] = success
        
        overall_duration = time.time() - overall_start
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL COMPREHENSIVE TEST SUMMARY")
        print("="*70)
        
        total_success = sum(results.values())
        total_categories = len(results)
        
        for category, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{category.upper():.<30} {status}")
        
        print(f"\nOverall Success Rate: {total_success}/{total_categories} ({total_success/total_categories*100:.1f}%)")
        print(f"Total Execution Time: {overall_duration:.2f}s")
        print("="*70)
        
        return total_success == total_categories
    
    def run_quick(self):
        """Run a quick subset of tests for development."""
        print("Running quick test subset...")
        return self.run_category('basic', verbosity=1)
    
    def run_security_only(self):
        """Run only security-focused tests."""
        return self.run_category('security', verbosity=2)
    
    def run_stress_only(self):
        """Run only stress and reliability tests."""
        return self.run_category('stress', verbosity=2)


def main():
    """Main entry point with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive CheckpointManager Test Runner')
    parser.add_argument('--category', choices=['basic', 'edge_cases', 'real_implementation', 
                                             'security', 'stress', 'advanced', 'markers'],
                       help='Run tests for specific category only')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of tests')
    parser.add_argument('--security', action='store_true', help='Run security tests only')
    parser.add_argument('--stress', action='store_true', help='Run stress tests only')
    parser.add_argument('--verbose', '-v', action='count', default=2, help='Increase verbosity')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner()
    
    try:
        if args.quick:
            success = runner.run_quick()
        elif args.security:
            success = runner.run_security_only()
        elif args.stress:
            success = runner.run_stress_only()
        elif args.category:
            success = runner.run_category(args.category, args.verbose)
        else:
            success = runner.run_all(args.verbose)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()