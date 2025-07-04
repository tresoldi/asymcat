#!/usr/bin/env python3
"""
DEPRECATED: Legacy ASymCat Test File
===================================

This test file has been replaced by a modernized test suite with better
organization, documentation, and coverage.

⚠️  DEPRECATION NOTICE ⚠️
This file is maintained for backward compatibility but will be removed
in a future version. Please use the new test structure instead.

NEW TEST STRUCTURE:
==================
The tests have been reorganized into:

📁 tests/
├── 📁 unit/                          # Unit tests for individual components
│   ├── test_data_loading.py          # Data file reading and preprocessing
│   ├── test_scoring_measures.py      # Individual scoring methods
│   ├── test_score_transformations.py # Scaling, inversion, matrices
│   └── test_legacy_compatibility.py  # Validation against this legacy file
├── 📁 integration/                   # Integration tests for workflows
│   └── test_end_to_end_workflows.py  # Complete analysis pipelines
├── 📁 fixtures/                      # Shared test data and utilities
│   ├── data.py                       # Test datasets and resources
│   └── assertions.py                 # Domain-specific assertions
├── conftest.py                       # Pytest configuration and fixtures
└── test_asymcat_modern.py           # Modern test entry point

BENEFITS OF NEW STRUCTURE:
=========================
✅ Proper separation of concerns
✅ Extensive parametrization for comprehensive coverage
✅ Tests serve as documentation examples
✅ Better error handling and edge case testing
✅ Performance and scalability testing
✅ Validation against legacy results (this file)
✅ Modern pytest practices and fixtures

HOW TO RUN TESTS:
================
# Run all tests
pytest

# Run specific categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only

# Run specific modules
pytest tests/unit/test_scoring_measures.py
pytest tests/unit/test_data_loading.py

# Run tests matching patterns
pytest -k "test_mle"                  # Tests with "mle" in name
pytest -k "test_tresoldi"             # Tests with "tresoldi" in name

# Run with different verbosity
pytest -v                             # Verbose output
pytest -s                             # Show print statements

# Run slow/performance tests
pytest --run-slow                     # Include slow tests
pytest --run-large-data              # Include tests needing large datasets

MIGRATION GUIDE:
===============
If you have custom tests that import from this file, update them to use:

OLD: from tests.test_asymcat import test_compute
NEW: from tests.unit.test_legacy_compatibility import TestLegacyDataCompatibility

The new tests provide the same validation with better organization and
more comprehensive coverage.
"""

import warnings
from pathlib import Path

# Import the modern test equivalents for backward compatibility
from .test_asymcat_modern import test_compute, test_readers, test_scorers, test_utils

# Show deprecation warning when this file is imported
warnings.warn(
    "tests/test_asymcat.py is deprecated. "
    "Please use the modernized test structure in tests/unit/ and tests/integration/. "
    "See tests/test_asymcat_modern.py for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy constants for backward compatibility
RESOURCE_DIR = Path(__file__).parent.parent / "resources"

# The large test functions from the original file have been moved to:
# - test_new_scoring_methods() -> tests/unit/test_scoring_measures.py
# - test_new_methods_with_known_values() -> tests/unit/test_scoring_measures.py
# - test_scoring_methods_consistency() -> tests/unit/test_legacy_compatibility.py
# - test_edge_cases_and_error_handling() -> tests/unit/test_scoring_measures.py
# - test_comprehensive_datasets() -> tests/integration/test_end_to_end_workflows.py
# - test_performance_and_scalability() -> tests/integration/test_end_to_end_workflows.py
# - test_mathematical_properties() -> tests/unit/test_scoring_measures.py


def test_new_scoring_methods():
    """Deprecated: Use tests/unit/test_scoring_measures.py instead."""
    warnings.warn("Use tests/unit/test_scoring_measures.py instead", DeprecationWarning)
    from .unit.test_legacy_compatibility import TestLegacyNewMethods

    test_instance = TestLegacyNewMethods()
    test_instance.test_new_methods_basic_validation("toy.tsv", 10)


def test_new_methods_with_known_values():
    """Deprecated: Use tests/unit/test_scoring_measures.py instead."""
    warnings.warn("Use tests/unit/test_scoring_measures.py instead", DeprecationWarning)
    from .unit.test_scoring_measures import TestMeasureProperties

    test_instance = TestMeasureProperties()
    test_instance.test_perfect_correlation_measures()


def test_scoring_methods_consistency():
    """Deprecated: Use tests/unit/test_legacy_compatibility.py instead."""
    warnings.warn("Use tests/unit/test_legacy_compatibility.py instead", DeprecationWarning)
    from .unit.test_legacy_compatibility import TestLegacyNewMethods

    test_instance = TestLegacyNewMethods()
    test_instance.test_new_methods_mathematical_relationships()


def test_edge_cases_and_error_handling():
    """Deprecated: Use tests/unit/test_scoring_measures.py instead."""
    warnings.warn("Use tests/unit/test_scoring_measures.py instead", DeprecationWarning)
    from .unit.test_legacy_compatibility import TestLegacyNewMethods

    test_instance = TestLegacyNewMethods()
    test_instance.test_new_methods_edge_cases()


def test_comprehensive_datasets():
    """Deprecated: Use tests/integration/test_end_to_end_workflows.py instead."""
    warnings.warn("Use tests/integration/test_end_to_end_workflows.py instead", DeprecationWarning)
    print("This test has been moved to tests/integration/test_end_to_end_workflows.py")
    print("Run: pytest tests/integration/test_end_to_end_workflows.py::TestCompleteWorkflows")


def test_performance_and_scalability():
    """Deprecated: Use tests/integration/test_end_to_end_workflows.py instead."""
    warnings.warn("Use tests/integration/test_end_to_end_workflows.py instead", DeprecationWarning)
    print("This test has been moved to tests/integration/test_end_to_end_workflows.py")
    print("Run: pytest tests/integration/test_end_to_end_workflows.py::TestLargeDatasetWorkflows --run-slow")


def test_mathematical_properties():
    """Deprecated: Use tests/unit/test_scoring_measures.py instead."""
    warnings.warn("Use tests/unit/test_scoring_measures.py instead", DeprecationWarning)
    print("This test has been moved to tests/unit/test_scoring_measures.py")
    print("Run: pytest tests/unit/test_scoring_measures.py::TestMeasureProperties")


if __name__ == "__main__":
    print(__doc__)
