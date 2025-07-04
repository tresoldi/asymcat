#!/usr/bin/env python3
"""
Modern ASymCat Test Suite
========================

This is the modernized test suite for ASymCat that replaces the legacy
monolithic test file. The tests are now organized into logical modules:

Unit Tests (tests/unit/):
- test_data_loading.py: Tests for reading and processing data files
- test_scoring_measures.py: Tests for individual scoring methods
- test_score_transformations.py: Tests for scaling, inversion, matrix generation
- test_legacy_compatibility.py: Validation against original test values

Integration Tests (tests/integration/):
- test_end_to_end_workflows.py: Complete analysis workflows

Test Fixtures (tests/fixtures/):
- data.py: Shared test data and resources
- assertions.py: Custom assertions for domain-specific validation

To run specific test categories:
    pytest tests/unit/                    # Run all unit tests
    pytest tests/integration/             # Run all integration tests
    pytest tests/unit/test_scoring_measures.py  # Run specific module
    pytest -k "test_mle"                  # Run tests matching pattern
    pytest -m "slow"                      # Run slow/performance tests

The modernized tests provide:
- Proper separation of concerns
- Extensive parametrization for comprehensive coverage
- Clear documentation as side examples
- Performance and scalability testing
- Better error handling validation
- Compatibility validation with legacy results
"""

# For backward compatibility, import key legacy test functions
from .unit.test_legacy_compatibility import (
    TestLegacyDataCompatibility,
    TestLegacyFileReading,
    TestLegacyNewMethods,
    TestLegacyUtilityFunctions,
)


# Re-export the main legacy test functions with modern names
def test_compute():
    """Legacy test - now handled by TestLegacyDataCompatibility.test_original_cmu_sample_data"""
    test_instance = TestLegacyDataCompatibility()
    test_instance.test_original_cmu_sample_data()


def test_scorers():
    """Legacy test - now handled by TestLegacyDataCompatibility.test_original_scorer_validation"""
    test_instance = TestLegacyDataCompatibility()
    test_instance.test_original_scorer_validation()
    test_instance.test_original_matrix_generation()
    test_instance.test_original_scaling_operations()


def test_readers():
    """Legacy test - now handled by TestLegacyFileReading tests"""
    test_instance = TestLegacyFileReading()
    test_instance.test_original_sequence_reading_results()
    test_instance.test_original_pa_matrix_reading_results()


def test_utils():
    """Legacy test - now handled by TestLegacyUtilityFunctions tests"""
    test_instance = TestLegacyUtilityFunctions()
    test_instance.test_original_ngram_collection()
    test_instance.test_original_cooccurrence_collection_with_ngrams()


# Note: The extensive new method tests from the legacy file are now
# properly organized in the test_scoring_measures.py and other modules
# with better structure, parametrization, and documentation.

if __name__ == "__main__":
    print(__doc__)
