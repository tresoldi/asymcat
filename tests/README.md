# ASymCat Test Suite

This directory contains the modernized test suite for ASymCat, organized following current best practices with proper separation of concerns, extensive parametrization, and comprehensive documentation.

## Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── test_data_loading.py        # Data file reading and preprocessing
│   ├── test_scoring_measures.py    # Individual scoring methods
│   ├── test_score_transformations.py # Scaling, inversion, matrices
│   └── test_legacy_compatibility.py # Validation against legacy results
├── integration/                    # Integration tests for complete workflows
│   └── test_end_to_end_workflows.py # Complete analysis pipelines
├── fixtures/                       # Shared test data and utilities
│   ├── data.py                     # Test datasets and resources
│   └── assertions.py               # Domain-specific assertions
├── conftest.py                     # Pytest configuration and fixtures
├── pytest.ini                     # Test runner configuration
├── test_asymcat.py                 # Legacy test file (deprecated)
├── test_asymcat_modern.py          # Modern test entry point
└── README.md                       # This file
```

## Key Improvements

### 🎯 **Separation of Concerns**
- **Unit tests**: Test individual functions and methods in isolation
- **Integration tests**: Test complete workflows from data loading to final results
- **Fixtures**: Shared test data and custom assertions for reusability

### 📊 **Extensive Parametrization**
- Tests run against multiple datasets automatically
- Different scoring methods tested with consistent parameters
- Edge cases and error conditions systematically covered

### 📚 **Documentation as Tests**
- Each test serves as a usage example
- Clear docstrings explain the purpose and demonstrate typical usage
- Tests validate both functionality and provide learning materials

### 🚀 **Performance Testing**
- Scalability tests with different dataset sizes
- Performance benchmarks with reasonable time limits
- Caching behavior validation

### 🛡️ **Robust Error Handling**
- Edge cases with minimal data
- Invalid input validation
- Graceful handling of missing files or corrupt data

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only

# Run specific modules
pytest tests/unit/test_scoring_measures.py
pytest tests/unit/test_data_loading.py
```

### Advanced Usage

```bash
# Run tests matching patterns
pytest -k "test_mle"                  # Tests with "mle" in name
pytest -k "test_tresoldi"             # Tests with "tresoldi" in name
pytest -k "test_mutual_information"   # Tests for mutual information

# Run with different verbosity
pytest -v                             # Verbose output
pytest -s                             # Show print statements
pytest -q                             # Quiet mode

# Run performance and slow tests
pytest --run-slow                     # Include slow tests
pytest --run-large-data              # Include tests needing large datasets
pytest -m "performance"              # Run only performance tests

# Run specific test categories
pytest -m "unit"                      # Run only unit tests
pytest -m "integration"              # Run only integration tests
pytest -m "slow"                     # Run only slow tests
```

### Development Workflow

```bash
# Quick validation during development
pytest tests/unit/test_scoring_measures.py::TestProbabilisticMeasures::test_maximum_likelihood_estimation

# Test a specific scoring method across all test types
pytest -k "mle"

# Validate changes don't break legacy compatibility
pytest tests/unit/test_legacy_compatibility.py

# Full test suite including performance tests
pytest --run-slow --run-large-data
```

## Test Categories

### Unit Tests

#### `test_data_loading.py`
- **File reading**: TSV sequence files, presence-absence matrices
- **Data validation**: Format checking, error handling
- **Co-occurrence collection**: N-gram generation, padding handling
- **Utility functions**: Helper methods for data processing

**Example usage patterns:**
```python
# Load sequence data
data = asymcat.read_sequences("data.tsv")
cooccs = asymcat.collect_cooccs(data)

# Load presence-absence matrix
combinations = asymcat.read_pa_matrix("species.tsv")

# Generate n-grams
ngrams = list(asymcat.collect_ngrams("hello", 2, "#"))
```

#### `test_scoring_measures.py`
- **Probabilistic measures**: MLE, Goodman-Kruskal Lambda, Jaccard Index
- **Information-theoretic measures**: Mutual Information, PMI, Conditional Entropy
- **Statistical measures**: Chi-square, Cramér's V, Fisher's Exact Test
- **Specialized measures**: Tresoldi measure, custom combinations

**Example usage patterns:**
```python
scorer = asymcat.scorer.CatScorer(cooccs)

# Probabilistic measures [0,1]
mle_scores = scorer.mle()
jaccard_scores = scorer.jaccard_index()

# Information-theoretic measures
mi_scores = scorer.mutual_information()
pmi_scores = scorer.pmi()

# Statistical measures
chi2_scores = scorer.chi2()
cramers_scores = scorer.cramers_v()
```

#### `test_score_transformations.py`
- **Scaling methods**: Min-max normalization, standardization, mean centering
- **Score inversion**: 1 - score transformations
- **Matrix generation**: Convert scores to visualization-ready matrices
- **Transformation chaining**: Combined operations

**Example usage patterns:**
```python
# Scale scores to [0,1] range
scaled = asymcat.scorer.scale_scorer(scores, method="minmax")

# Standardize for comparison between methods
standardized = asymcat.scorer.scale_scorer(scores, method="stdev")

# Invert scores (high becomes low)
inverted = asymcat.scorer.invert_scorer(scaled)

# Generate matrices for heatmaps
xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scores)
```

#### `test_legacy_compatibility.py`
- **Exact value validation**: Ensures modernized code produces identical results
- **Original test reproduction**: Validates against the legacy test suite
- **Regression testing**: Prevents accidental changes to established behavior

### Integration Tests

#### `test_end_to_end_workflows.py`
- **Complete analysis pipelines**: From data loading to final visualization
- **Performance testing**: Scalability with different dataset sizes
- **Error handling workflows**: Graceful degradation with invalid inputs
- **Comparative analysis**: Multi-method comparisons and correlations

**Example workflows:**
```python
# Complete sequence analysis
data = asymcat.read_sequences("linguistic_data.tsv")
cooccs = asymcat.collect_cooccs(data)
scorer = asymcat.scorer.CatScorer(cooccs)

# Compute multiple measures
measures = {
    'mle': scorer.mle(),
    'tresoldi': scorer.tresoldi(),
    'mutual_info': scorer.mutual_information()
}

# Apply transformations
scaled = asymcat.scorer.scale_scorer(measures['tresoldi'], method="minmax")

# Generate visualization matrices
xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scaled)
```

## Test Data and Fixtures

### Shared Test Data (`fixtures/data.py`)
- **Sample datasets**: Small, predictable data for unit testing
- **Resource management**: Paths to larger test files
- **Expected results**: Known values for validation
- **Test case generation**: Parametrized test data

### Custom Assertions (`fixtures/assertions.py`)
- **Domain-specific validation**: Score format, range, and property checking
- **Performance assertions**: Execution time limits
- **Mathematical property validation**: Symmetry, non-negativity, range constraints
- **Matrix consistency**: Dimension and label validation

### Configuration (`conftest.py`)
- **Pytest fixtures**: Shared setup and teardown
- **Custom markers**: Categorizing tests (slow, integration, etc.)
- **Test collection**: Automatic test discovery and organization
- **Performance monitoring**: Timing and resource usage tracking

## Migration from Legacy Tests

The original monolithic `test_asymcat.py` has been replaced with this structured approach. Key changes:

### What Moved Where

| Original Function | New Location | Improvement |
|------------------|--------------|-------------|
| `test_compute()` | `test_legacy_compatibility.py::TestLegacyDataCompatibility::test_original_cmu_sample_data()` | Proper class organization |
| `test_scorers()` | `test_legacy_compatibility.py::TestLegacyDataCompatibility::test_original_scorer_validation()` | Split into focused tests |
| `test_readers()` | `test_data_loading.py::TestSequenceReading`, `TestPresenceAbsenceMatrixReading` | Parametrized across datasets |
| `test_new_scoring_methods()` | `test_scoring_measures.py::TestProbabilisticMeasures`, etc. | Organized by measure type |
| `test_performance_and_scalability()` | `test_end_to_end_workflows.py::TestLargeDatasetWorkflows` | Proper performance testing |

### Benefits of Migration

1. **Better Organization**: Related tests are grouped together
2. **Comprehensive Coverage**: Parametrization tests more scenarios
3. **Clear Documentation**: Each test explains its purpose and usage
4. **Easier Maintenance**: Focused tests are easier to debug and extend
5. **Performance Awareness**: Explicit performance testing and monitoring
6. **Error Handling**: Systematic testing of edge cases and error conditions

## Contributing

When adding new tests:

1. **Choose the right location**: Unit tests for individual functions, integration tests for workflows
2. **Use parametrization**: Test multiple scenarios with `@pytest.mark.parametrize`
3. **Document with examples**: Include usage examples in docstrings
4. **Test error conditions**: Include tests for invalid inputs and edge cases
5. **Validate performance**: Use performance assertions for slow operations
6. **Update fixtures**: Add new test data to `fixtures/data.py` if needed

### Example New Test

```python
class TestNewScoringMethod:
    """Test a new scoring method."""

    @pytest.mark.parametrize("dataset,expected_properties", [
        ("toy.tsv", {"min_pairs": 10, "max_time": 1.0}),
        ("mushroom-small.tsv", {"min_pairs": 50, "max_time": 5.0}),
    ])
    def test_new_method(self, dataset: str, expected_properties: dict):
        """
        Test new scoring method with multiple datasets.

        Example usage:
            scores = scorer.new_method()
            # Returns {(x, y): (score_xy, score_yx)}
        """
        # Load data
        data = asymcat.read_sequences(RESOURCE_DIR / dataset)
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Test method
        start_time = time.time()
        scores = scorer.new_method()
        execution_time = time.time() - start_time

        # Validate results
        assert_valid_scores(scores)
        assert len(scores) >= expected_properties["min_pairs"]
        assert_performance_acceptable(execution_time, expected_properties["max_time"])
```

This structure ensures that ASymCat's test suite remains maintainable, comprehensive, and serves as effective documentation for users and developers.
