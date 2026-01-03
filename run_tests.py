#!/usr/bin/env python
"""Quick test runner for uncertainty estimation functions."""
import pytest
import sys

if __name__ == "__main__":
    # Run tests with verbose output
    exit_code = pytest.main([
        "tests/",
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-x",  # Stop on first failure
        "--disable-warnings",  # Cleaner output
    ])
    sys.exit(exit_code)

