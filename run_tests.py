#!/usr/bin/env python3
"""Unified Automated Test Runner for FlyBrain Snake.

Supports both standard library `unittest` and `pytest`.
Exits with 0 on pass, 1 on fail.

Usage:
    python3 run_tests.py                 # Run all unit and E2E tests
    python3 run_tests.py -v              # Verbose mode
    python3 run_tests.py --unittest      # Use unittest runner
    python3 run_tests.py --pytest        # Use pytest runner
    python3 run_tests.py --unit          # Run only unit tests (tests/test_*.py)
    python3 run_tests.py --e2e           # Run only E2E tests (tests/e2e/)
    python3 run_tests.py --tier 1        # Run specific E2E tier (1, 2, 3, or 4)
    python3 run_tests.py tests/test_connectome.py  # Run specific test file
"""

import argparse
import os
import sys
import time
import unittest


def run_with_unittest(targets: list[str], verbose: bool = False) -> int:
    """Run tests using standard library unittest."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if not targets:
        # Discover all tests in tests/
        discovered = loader.discover(
            start_dir=os.path.join(root_dir, "tests"),
            pattern="test_*.py",
            top_level_dir=root_dir,
        )
        suite.addTests(discovered)
    else:
        for target in targets:
            abs_target = os.path.abspath(target)
            if os.path.isdir(abs_target):
                discovered = loader.discover(
                    start_dir=abs_target,
                    pattern="test_*.py",
                    top_level_dir=root_dir,
                )
                suite.addTests(discovered)
            elif os.path.isfile(abs_target):
                rel_path = os.path.relpath(abs_target, root_dir)
                module_name = rel_path.replace(os.path.sep, ".").removesuffix(".py")
                discovered = loader.loadTestsFromName(module_name)
                suite.addTests(discovered)
            else:
                # Try loading by name
                discovered = loader.loadTestsFromName(target)
                suite.addTests(discovered)

    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - errors - skipped

    print("\n" + "=" * 70)
    print(f"TEST RUN SUMMARY (unittest runner)")
    print(f"Elapsed time : {elapsed:.2f}s")
    print(f"Total tests  : {total}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failures}")
    print(f"Errors       : {errors}")
    print(f"Skipped      : {skipped}")
    print("=" * 70)

    if result.wasSuccessful():
        print("RESULT: ALL TESTS PASSED (EXIT CODE 0)\n")
        return 0
    else:
        print("RESULT: TESTS FAILED (EXIT CODE 1)\n")
        return 1


def run_with_pytest(targets: list[str], verbose: bool = False, extra_args: list[str] | None = None) -> int:
    """Run tests using pytest if available."""
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Falling back to unittest runner...")
        return run_with_unittest(targets, verbose=verbose)

    args = []
    if verbose:
        args.append("-v")
    else:
        args.append("-q")

    if extra_args:
        args.extend(extra_args)

    if targets:
        args.extend(targets)
    else:
        args.append("tests")

    start_time = time.time()
    retcode = pytest.main(args)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"TEST RUN SUMMARY (pytest runner)")
    print(f"Elapsed time : {elapsed:.2f}s")
    print(f"pytest exit  : {retcode}")
    print("=" * 70)

    if retcode == 0:
        print("RESULT: ALL TESTS PASSED (EXIT CODE 0)\n")
        return 0
    elif retcode == 5:
        # pytest code 5 = no tests collected
        print("RESULT: NO TESTS COLLECTED (EXIT CODE 0)\n")
        return 0
    else:
        print(f"RESULT: TESTS FAILED WITH CODE {retcode} (EXIT CODE 1)\n")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyBrain Snake Automated Test Runner")
    parser.add_argument("targets", nargs="*", help="Optional test files, directories, or modules")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet output")
    parser.add_argument("--unittest", action="store_true", help="Force using unittest runner")
    parser.add_argument("--pytest", action="store_true", help="Force using pytest runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests in tests/")
    parser.add_argument("--e2e", action="store_true", help="Run only E2E tests in tests/e2e/")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run specific E2E tier (1-4)")

    args = parser.parse_args()

    import glob
    targets = list(args.targets)

    if args.tier:
        targets.extend(glob.glob(f"tests/e2e/test_tier{args.tier}_*.py"))
    elif args.unit:
        targets.extend(glob.glob("tests/test_*.py"))
    elif args.e2e:
        targets.append("tests/e2e")

    # Determine runner
    if args.unittest:
        return run_with_unittest(targets, verbose=args.verbose)
    elif args.pytest:
        return run_with_pytest(targets, verbose=args.verbose)
    else:
        # Default: try pytest first, fallback to unittest
        try:
            import pytest
            return run_with_pytest(targets, verbose=args.verbose)
        except ImportError:
            return run_with_unittest(targets, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
