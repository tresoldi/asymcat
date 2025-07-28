# ASymCat Developer Guide

This comprehensive guide provides everything you need to contribute to ASymCat, from environment setup to advanced development workflows.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/tresoldi/asymcat.git
cd asymcat
make install  # Creates venv and installs dev dependencies

# Verify installation
make test
make format-check
```

## 📋 Table of Contents

1. [Development Environment](#-development-environment)
2. [Code Architecture](#-code-architecture)
3. [Testing Strategy](#-testing-strategy)
4. [Code Quality Standards](#-code-quality-standards)
5. [Contributing Workflow](#-contributing-workflow)
6. [Adding New Features](#-adding-new-features)
7. [Performance Considerations](#-performance-considerations)
8. [Documentation Standards](#-documentation-standards)
9. [Release Process](#-release-process)
10. [Troubleshooting](#-troubleshooting)

## 🛠️ Development Environment

### Prerequisites

- Python 3.10+ (recommended: Python 3.11+)
- Git
- Make (for automated workflows)
- Optional: pyenv for Python version management

### Environment Setup

#### Option 1: Makefile (Recommended)
```bash
# Automatic setup with virtual environment
make install

# What this does:
# - Creates python venv in .venv/
# - Installs package in development mode with [dev] extras
# - Sets up pre-commit hooks
# - Validates installation with basic tests
```

#### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev,all]"

# Verify installation
python -c "import asymcat; print(asymcat.__version__)"
```

### Development Dependencies

The project uses several categories of dependencies:

**Core Runtime** (`install_requires`):
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Statistical functions
- `matplotlib`, `seaborn` - Visualization
- `tabulate` - Table formatting
- `freqprob` - Probability estimation and smoothing

**Development** (`[dev]` extra):
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting
- `mypy` - Type checking

**Documentation** (`[docs]` extra):
- `sphinx` - Documentation generation
- `sphinx-rtd-theme` - ReadTheDocs theme
- `myst-parser` - Markdown support in Sphinx

**Optional Features** (`[all]` extra):
- `jupyter` - Notebook support
- `plotly`, `bokeh`, `altair` - Advanced visualization

### IDE Configuration

#### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm
- Set interpreter to `.venv/bin/python`
- Enable Black as external formatter
- Configure isort for import organization
- Enable mypy for type checking

## 🏗️ Code Architecture

### Project Structure

```
asymcat/
├── asymcat/                    # Main package
│   ├── __init__.py            # Package initialization and exports
│   ├── __main__.py            # CLI entry point
│   ├── common.py              # Data loading and preprocessing
│   ├── scorer.py              # Core scoring algorithms
│   └── correlation.py         # Correlation analysis utilities
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── fixtures/              # Test data and utilities
│   └── conftest.py           # Pytest configuration
├── resources/                 # Example datasets
├── docs/                      # Documentation source
├── Makefile                   # Development workflow automation
├── pyproject.toml            # Modern Python packaging
├── pytest.ini               # Test configuration
└── README.md                 # User documentation
```

### Core Modules

#### `asymcat/common.py` - Data Processing

**Purpose**: Handle all data input/output and preprocessing operations.

**Key Functions**:
```python
def read_sequences(filename: str) -> List[Tuple[List[str], List[str]]]
    """Load aligned sequence data from TSV files."""

def read_pa_matrix(filename: str) -> List[Tuple[str, str]]
    """Load presence-absence matrices."""

def collect_cooccs(data: Sequence, order: int = 1, pad: str = "#") -> List[Tuple[str, str, int]]
    """Extract co-occurrence pairs with counts."""

def collect_ngrams(sequence: Sequence[str], n: int, pad: str = "#") -> Iterator[Tuple[str, ...]]
    """Generate n-grams from sequences."""
```

**Design Principles**:
- **Format Agnostic**: Support multiple input formats transparently
- **Memory Efficient**: Stream processing for large datasets
- **Error Resilient**: Graceful handling of malformed data
- **Type Safe**: Comprehensive type annotations

#### `asymcat/scorer.py` - Association Measures

**Purpose**: Implement all association measures and score transformations.

**Core Class**:
```python
class CatScorer:
    """Main scoring engine for categorical associations."""
    
    def __init__(self, cooccs: List[Tuple[str, str, int]], 
                 smoothing_method: str = "mle", 
                 smoothing_alpha: float = 1.0):
        """Initialize with co-occurrence data and smoothing parameters."""
```

**Scoring Methods Categories**:

1. **Probabilistic Measures** (Range: [0, 1]):
   - `mle()` - Maximum Likelihood Estimation
   - `jaccard_index()` - Set overlap measure

2. **Information-Theoretic Measures**:
   - `pmi()` - Pointwise Mutual Information
   - `pmi_smoothed()` - Numerically stable PMI
   - `mutual_information()` - Average mutual information
   - `conditional_entropy()` - Information remaining

3. **Statistical Measures**:
   - `chi2()` - Pearson's Chi-square
   - `cramers_v()` - Normalized association strength
   - `fisher()` - Fisher's exact test
   - `log_likelihood()` - G² statistic

4. **Specialized Measures**:
   - `tresoldi()` - Custom measure for sequence alignment
   - `theil_u()` - Uncertainty coefficient
   - `goodman_kruskal_lambda()` - Proportional error reduction

**Return Format**: All methods return `Dict[Tuple[str, str], Tuple[float, float]]`
- Key: `(x, y)` category pair
- Value: `(x→y_score, y→x_score)` directional scores

#### `asymcat/__main__.py` - Command Line Interface

**Purpose**: Provide production-ready CLI access to all functionality.

**Key Features**:
- Multiple input format support
- Comprehensive output options (JSON, CSV, tables)
- Parameter validation and error handling
- Performance monitoring and progress reporting

### Data Flow Architecture

```
Input Data → Data Loading → Co-occurrence Collection → Scoring → Output
     ↓             ↓                   ↓               ↓         ↓
TSV Files → read_sequences() → collect_cooccs() → CatScorer → Results
PA Matrix → read_pa_matrix() →      ↓               ↓         ↓
Sequences →       ↓          → collect_ngrams() →   ↓    → JSON/CSV
              Error Check                        Score     Tables
                  ↓                           Transform      ↓
              Validation                         ↓      Visualization
```

### Design Patterns

#### Composition over Inheritance
- `CatScorer` aggregates functionality rather than inheriting
- Scoring methods are independent and composable
- Utility functions are stateless and pure where possible

#### Strategy Pattern for Smoothing
```python
# Smoothing strategies handled via freqprob integration
scorer = CatScorer(cooccs, smoothing_method="laplace", smoothing_alpha=1.0)
```

#### Factory Pattern for Data Loading
```python
# Automatic format detection and appropriate loader selection
def load_data(filename: str, format: Optional[str] = None):
    if format is None:
        format = detect_format(filename)
    return FORMAT_LOADERS[format](filename)
```

## 🧪 Testing Strategy

### Test Organization

ASymCat follows a modern, hierarchical testing approach:

```
tests/
├── unit/                      # Test individual components
│   ├── test_data_loading.py   # File I/O and preprocessing
│   ├── test_scoring_measures.py # Individual scoring methods
│   ├── test_score_transformations.py # Utilities and transformations
│   └── test_legacy_compatibility.py # Backward compatibility
├── integration/               # Test complete workflows
│   └── test_end_to_end_workflows.py # Full analysis pipelines
└── fixtures/                  # Shared test infrastructure
    ├── data.py               # Test datasets
    └── assertions.py         # Custom validation functions
```

### Testing Principles

#### 1. **Comprehensive Parametrization**
Tests run against multiple datasets and parameter combinations:

```python
@pytest.mark.parametrize("dataset,min_pairs", [
    ("toy.tsv", 10),
    ("mushroom-small.tsv", 50),
    ("cmu_sample.tsv", 100),
])
def test_scorer_method(self, dataset: str, min_pairs: int):
    """Test scoring method across different datasets."""
```

#### 2. **Property-Based Testing**
Validate mathematical properties rather than exact values:

```python
def test_mle_probability_properties(self, sample_cooccs):
    """MLE scores should be valid probabilities [0,1]."""
    scores = CatScorer(sample_cooccs).mle()
    for (x, y), (xy_score, yx_score) in scores.items():
        assert 0 <= xy_score <= 1
        assert 0 <= yx_score <= 1
```

#### 3. **Performance Validation**
Ensure scalability with explicit performance tests:

```python
@pytest.mark.slow
def test_large_dataset_performance(self, large_dataset):
    """Scoring should complete within reasonable time limits."""
    start = time.time()
    scores = CatScorer(large_dataset).pmi()
    duration = time.time() - start
    assert duration < 10.0  # 10 second limit
```

#### 4. **Error Condition Testing**
Systematic testing of edge cases and invalid inputs:

```python
def test_empty_data_handling(self):
    """Empty datasets should be handled gracefully."""
    with pytest.raises(ValueError, match="No co-occurrences found"):
        CatScorer([]).mle()
```

### Running Tests

```bash
# Full test suite
make test

# Specific categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest -m slow                       # Performance tests

# Coverage reporting
make coverage
# Opens HTML report in coverage_html_report/index.html

# Quick development testing
make quick-test  # Skip slow tests

# Test specific functionality
pytest -k "test_mle"                 # All MLE-related tests
pytest -k "test_data_loading"        # Data loading tests
```

### Writing New Tests

#### Unit Test Example
```python
class TestNewScoringMethod:
    """Test suite for a new scoring method."""

    @pytest.mark.parametrize("dataset", ["toy.tsv", "mushroom-small.tsv"])
    def test_new_method_basic(self, dataset: str):
        """Test basic functionality of new scoring method."""
        # Load test data
        data = asymcat.read_sequences(RESOURCE_DIR / dataset)
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.CatScorer(cooccs)
        
        # Execute method
        scores = scorer.new_method()
        
        # Validate results
        assert_valid_scores(scores)  # Custom assertion
        assert len(scores) > 0
        
    def test_new_method_properties(self, sample_cooccs):
        """Test mathematical properties of new method."""
        scores = CatScorer(sample_cooccs).new_method()
        
        # Test range constraints
        for (x, y), (xy_score, yx_score) in scores.items():
            assert xy_score >= 0  # Non-negative
            assert yx_score >= 0
            # Add other property tests
            
    def test_new_method_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with minimal data
        minimal_cooccs = [("A", "B", 1)]
        scores = CatScorer(minimal_cooccs).new_method()
        assert len(scores) == 1
        
        # Test error conditions
        with pytest.raises(ValueError):
            CatScorer([]).new_method()
```

#### Integration Test Example
```python
class TestNewMethodWorkflow:
    """Integration tests for new method in complete workflows."""
    
    def test_end_to_end_analysis(self, tmp_path):
        """Test complete analysis workflow with new method."""
        # Create test data file
        test_file = tmp_path / "test_data.tsv"
        test_file.write_text("A B C\tX Y Z\nD E F\tU V W\n")
        
        # Complete workflow
        data = asymcat.read_sequences(str(test_file))
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.CatScorer(cooccs)
        
        # Multiple methods for comparison
        results = {
            'new_method': scorer.new_method(),
            'mle': scorer.mle(),
            'pmi': scorer.pmi(),
        }
        
        # Validate consistency
        for method, scores in results.items():
            assert_valid_scores(scores)
            
        # Test transformations
        scaled = asymcat.scorer.scale_scorer(results['new_method'])
        matrices = asymcat.scorer.scorer2matrices(scaled)
        
        assert len(matrices) == 4  # xy_matrix, yx_matrix, x_labels, y_labels
```

### Custom Assertions

Located in `tests/fixtures/assertions.py`:

```python
def assert_valid_scores(scores: Dict, allow_infinite: bool = False):
    """Validate score dictionary format and values."""
    assert isinstance(scores, dict)
    for pair, (xy, yx) in scores.items():
        assert isinstance(pair, tuple) and len(pair) == 2
        assert isinstance(xy, (int, float))
        assert isinstance(yx, (int, float))
        if not allow_infinite:
            assert np.isfinite(xy)
            assert np.isfinite(yx)

def assert_scores_symmetric(scores: Dict):
    """Validate symmetric measure properties."""
    for (x, y), (xy_score, yx_score) in scores.items():
        # For symmetric measures, xy should equal yx
        assert abs(xy_score - yx_score) < 1e-10

def assert_performance_acceptable(duration: float, limit: float):
    """Validate performance within acceptable limits."""
    assert duration < limit, f"Execution took {duration:.2f}s, limit {limit:.2f}s"
```

## 🎯 Code Quality Standards

### Code Formatting

ASymCat uses **Black** for consistent code formatting:

```bash
# Auto-format all code
make format

# Check formatting without changes
make black-check

# Manual formatting
black asymcat/ tests/
```

**Black Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
```

### Import Organization

**isort** organizes imports consistently:

```bash
# Auto-organize imports
make isort

# Check import organization
make isort-check
```

**Import Order**:
1. Standard library imports
2. Third-party imports  
3. Local application imports

**Example**:
```python
# Standard library
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Third-party
import numpy as np
import pandas as pd
from scipy import stats

# Local
from asymcat.common import read_sequences
from asymcat.scorer import CatScorer
```

### Linting

**Flake8** enforces code quality rules:

```bash
# Run linter
make lint

# Check specific files
flake8 asymcat/scorer.py
```

**Configuration** (in `.flake8`):
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503  # Black compatibility
exclude = .git,__pycache__,build,dist,.venv
```

### Type Checking

**MyPy** provides static type checking:

```bash
# Run type checker
make mypy

# Check specific module
mypy asymcat/scorer.py
```

**Type Annotation Standards**:
```python
# Function signatures
def collect_cooccs(
    data: List[Tuple[List[str], List[str]]], 
    order: int = 1, 
    pad: str = "#"
) -> List[Tuple[str, str, int]]:
    """Collect co-occurrences with type annotations."""

# Class methods
class CatScorer:
    def __init__(
        self, 
        cooccs: List[Tuple[str, str, int]], 
        smoothing_method: str = "mle",
        smoothing_alpha: float = 1.0
    ) -> None:
        """Type-annotated constructor."""
        
    def mle(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return type clearly specified."""
```

### Documentation Standards

#### Docstring Format

Use **Google-style docstrings**:

```python
def new_scoring_method(self, parameter: float = 1.0) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Compute association scores using new method.
    
    This method implements a novel approach to measuring categorical
    associations based on [theoretical foundation].
    
    Args:
        parameter: Tuning parameter controlling method behavior.
            Higher values increase sensitivity. Default: 1.0.
            
    Returns:
        Dictionary mapping category pairs to directional scores.
        Keys are (category_x, category_y) tuples.
        Values are (x→y_score, y→x_score) tuples.
        
    Raises:
        ValueError: If parameter is negative or data is empty.
        
    Example:
        >>> scorer = CatScorer(cooccs)
        >>> scores = scorer.new_scoring_method(parameter=2.0)
        >>> xy_score, yx_score = scores[('A', 'B')]
    """
```

#### Code Comments

```python
# Use comments for complex algorithmic decisions
def complex_calculation(self, data: np.ndarray) -> float:
    # Apply log-space computation to prevent numerical underflow
    # when dealing with very small probabilities
    log_values = np.log(data + self.smoothing_alpha)
    
    # Normalize using log-sum-exp trick for numerical stability
    max_log = np.max(log_values)
    normalized = np.exp(log_values - max_log)
    
    return np.sum(normalized)
```

### Quality Automation

Run all quality checks together:

```bash
# Complete quality check pipeline
make format-check

# Individual components
make black-check    # Formatting
make isort-check    # Import organization  
make lint          # Code quality
make mypy          # Type checking
```

## 🔄 Contributing Workflow

### Getting Started

1. **Fork and Clone**:
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/asymcat.git
cd asymcat

# Add upstream remote
git remote add upstream https://github.com/tresoldi/asymcat.git
```

2. **Set Up Development Environment**:
```bash
make install
make test  # Verify everything works
```

3. **Create Feature Branch**:
```bash
git checkout -b feature/new-scoring-method
# or
git checkout -b fix/data-loading-bug
```

### Development Process

#### 1. **Implement Changes**
- Write code following established patterns
- Add comprehensive tests
- Update documentation
- Follow type annotation standards

#### 2. **Validate Changes**
```bash
# Run full validation pipeline
make test               # All tests pass
make format-check       # Code style compliance
make coverage          # Maintain test coverage

# Quick development checks
make quick-test        # Fast validation during development
```

#### 3. **Update Documentation**
- Add docstrings to new functions/methods
- Update README.md if adding user-facing features
- Create examples for new functionality
- Update DEVELOPER.md for new development patterns

#### 4. **Commit Changes**
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add new probabilistic scoring method

- Implement Bayesian association measure
- Add comprehensive test suite with 15+ test cases  
- Include performance benchmarks
- Update CLI to support new method"
```

### Pull Request Process

#### 1. **Prepare for PR**
```bash
# Sync with upstream
git fetch upstream
git rebase upstream/master

# Final validation
make test
make format-check
```

#### 2. **Submit Pull Request**
- Use descriptive title summarizing the change
- Include detailed description explaining:
  - What was changed and why
  - How to test the changes
  - Any breaking changes or migration notes
  - References to relevant issues

**PR Template**:
```markdown
## Summary
Brief description of changes and motivation.

## Changes Made
- [ ] Added new scoring method: `bayesian_association()`
- [ ] Implemented comprehensive test suite (15 tests)
- [ ] Updated CLI to support new method
- [ ] Added performance benchmarks

## Testing
- [ ] All existing tests pass
- [ ] New tests cover edge cases and performance
- [ ] Manual testing completed on sample datasets

## Documentation
- [ ] Docstrings added to new functions
- [ ] README updated with usage examples
- [ ] Type annotations complete

## Breaking Changes
None / List any breaking changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Documentation updated
```

#### 3. **Address Review Feedback**
- Respond to comments constructively
- Make requested changes promptly
- Update tests if functionality changes
- Maintain clean commit history (squash if needed)

### Code Review Guidelines

#### For Authors
- Self-review before submitting
- Provide context in PR description
- Keep PRs focused and reasonably sized
- Respond to feedback professionally

#### For Reviewers
- Focus on code correctness and maintainability
- Check test coverage and quality
- Validate documentation updates
- Consider performance implications
- Be constructive and educational in feedback

## ➕ Adding New Features

### Adding New Scoring Methods

#### 1. **Method Implementation**

Add to `asymcat/scorer.py` in the `CatScorer` class:

```python
def new_method(self, parameter: float = 1.0) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Implement new association scoring method.
    
    [Detailed docstring with mathematical foundation]
    """
    # Validate input
    if parameter <= 0:
        raise ValueError("Parameter must be positive")
    
    # Get probability estimates (handles smoothing automatically)
    p_xy, p_x, p_y = self.get_smoothed_probabilities()
    
    # Implement scoring logic
    scores = {}
    for (x, y) in self.pairs:
        # Calculate directional scores
        xy_score = self._compute_xy_score(x, y, p_xy, p_x, p_y, parameter)
        yx_score = self._compute_yx_score(x, y, p_xy, p_x, p_y, parameter)
        
        scores[(x, y)] = (xy_score, yx_score)
    
    return scores

def _compute_xy_score(self, x: str, y: str, p_xy: Dict, p_x: Dict, p_y: Dict, parameter: float) -> float:
    """Helper method for X→Y score calculation."""
    # Implement core mathematical formula
    pass
```

#### 2. **Add to CLI Interface**

Update `asymcat/__main__.py`:

```python
# Add to AVAILABLE_SCORERS
AVAILABLE_SCORERS = [
    "mle", "pmi", "chi2", "fisher", "theil_u",
    "new_method",  # Add here
    "all"
]

# Add to scorer method mapping
def get_scorer_methods(scorers: List[str]) -> List[str]:
    method_mapping = {
        "mle": "mle",
        "pmi": "pmi", 
        # ... existing mappings
        "new_method": "new_method",  # Add mapping
    }
```

#### 3. **Comprehensive Testing**

Create tests in `tests/unit/test_scoring_measures.py`:

```python
class TestNewMethod:
    """Test suite for new scoring method."""
    
    @pytest.mark.parametrize("dataset", STANDARD_DATASETS)
    def test_new_method_basic(self, dataset):
        """Basic functionality test."""
        scores = self.get_scorer(dataset).new_method()
        assert_valid_scores(scores)
        
    @pytest.mark.parametrize("parameter", [0.1, 1.0, 2.0, 5.0])
    def test_parameter_sensitivity(self, sample_cooccs, parameter):
        """Test parameter sensitivity."""
        scores = CatScorer(sample_cooccs).new_method(parameter=parameter)
        assert_valid_scores(scores)
        
    def test_mathematical_properties(self, sample_cooccs):
        """Test mathematical properties specific to method."""
        scores = CatScorer(sample_cooccs).new_method()
        
        # Test method-specific properties
        for (x, y), (xy_score, yx_score) in scores.items():
            # Example: test range constraints
            assert 0 <= xy_score <= 10  # Based on method range
            assert 0 <= yx_score <= 10
            
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        scorer = CatScorer([("A", "B", 1)])
        
        # Test valid edge case
        scores = scorer.new_method()
        assert len(scores) == 1
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            scorer.new_method(parameter=-1.0)
```

#### 4. **Performance Benchmarking**

Add performance tests in `tests/integration/test_end_to_end_workflows.py`:

```python
@pytest.mark.slow
def test_new_method_performance(self, large_dataset):
    """Benchmark new method performance."""
    scorer = CatScorer(large_dataset)
    
    start_time = time.time()
    scores = scorer.new_method()
    execution_time = time.time() - start_time
    
    # Performance assertions
    assert execution_time < 10.0  # 10 second limit
    assert len(scores) > 100  # Ensure meaningful output
```

#### 5. **Documentation Updates**

Update README.md:
```markdown
### Specialized Measures
- **New Method**: Description of new method and its applications
  - Range: [specify range]
  - Best for: [use cases]
  - Parameters: [parameter descriptions]
```

Add usage example:
```python
# New method example
scorer = asymcat.CatScorer(cooccs)
scores = scorer.new_method(parameter=2.0)
```

### Adding New Data Formats

#### 1. **Implement Reader Function**

Add to `asymcat/common.py`:

```python
def read_new_format(filename: str, **kwargs) -> List[Tuple[str, str]]:
    """Read data from new format.
    
    Args:
        filename: Path to data file
        **kwargs: Format-specific parameters
        
    Returns:
        List of (category_x, category_y) pairs
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Implement format-specific parsing
            data = []
            for line in f:
                # Parse line according to format
                parsed = parse_new_format_line(line, **kwargs)
                data.extend(parsed)
        return data
    except Exception as e:
        raise ValueError(f"Error reading new format: {e}")
```

#### 2. **Add Format Detection**

```python
def detect_format(filename: str) -> str:
    """Automatically detect data format."""
    with open(filename, 'r') as f:
        first_line = f.readline()
        # Add detection logic for new format
        if new_format_pattern.match(first_line):
            return "new_format"
        # ... existing detection logic
```

#### 3. **Update CLI Support**

Add format option to CLI:
```python
parser.add_argument(
    "--format",
    choices=["sequences", "pa-matrix", "new-format"],  # Add new format
    help="Input data format"
)
```

### Adding New Utilities

#### Score Transformation Functions

Add to `asymcat/scorer.py`:

```python
def new_transformation(
    scores: Dict[Tuple[str, str], Tuple[float, float]], 
    **kwargs
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Apply new transformation to scores.
    
    Args:
        scores: Input score dictionary
        **kwargs: Transformation parameters
        
    Returns:
        Transformed scores in same format
    """
    transformed = {}
    for pair, (xy, yx) in scores.items():
        # Apply transformation logic
        new_xy = transform_value(xy, **kwargs)
        new_yx = transform_value(yx, **kwargs)
        transformed[pair] = (new_xy, new_yx)
    
    return transformed
```

## ⚡ Performance Considerations

### Optimization Strategies

#### 1. **Algorithmic Efficiency**

**Use Vectorized Operations**:
```python
# Inefficient: Python loops
scores = {}
for x, y in pairs:
    scores[(x, y)] = slow_calculation(x, y)

# Efficient: NumPy vectorization
xy_array = np.array([(x, y) for x, y in pairs])
score_array = vectorized_calculation(xy_array)
scores = dict(zip(pairs, score_array))
```

**Leverage Scipy Functions**:
```python
# Use optimized statistical functions
from scipy import stats, special

# Efficient implementations for common operations
chi2_stat = stats.chi2_contingency(contingency_table)[0]
log_gamma = special.loggamma(values)  # More stable than log(gamma())
```

#### 2. **Memory Management**

**Stream Processing for Large Datasets**:
```python
def process_large_dataset(filename: str) -> Iterator[Tuple[str, str, int]]:
    """Process data in chunks to manage memory."""
    with open(filename, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(parse_line(line))
            if len(chunk) >= CHUNK_SIZE:
                yield from process_chunk(chunk)
                chunk = []
        if chunk:
            yield from process_chunk(chunk)
```

**Efficient Data Structures**:
```python
# Use appropriate data structures for access patterns
from collections import defaultdict, Counter

# Fast counting
co_occurrence_counts = Counter(pairs)

# Efficient lookups with defaults
probability_cache = defaultdict(float)
```

#### 3. **Caching Strategies**

**Memoization for Expensive Computations**:
```python
from functools import lru_cache

class CatScorer:
    @lru_cache(maxsize=1000)
    def _compute_expensive_statistic(self, x: str, y: str) -> float:
        """Cache expensive computations."""
        return expensive_calculation(x, y)
```

**Property-Based Caching**:
```python
class CatScorer:
    def __init__(self, cooccs):
        self.cooccs = cooccs
        self._probability_cache = None
        
    @property
    def probabilities(self):
        """Lazy computation with caching."""
        if self._probability_cache is None:
            self._probability_cache = self._compute_probabilities()
        return self._probability_cache
```

### Performance Testing

#### Benchmarking Framework

```python
import time
import psutil
from typing import Dict, Any

def benchmark_scorer_method(method_name: str, datasets: List[str]) -> Dict[str, Any]:
    """Comprehensive benchmarking of scoring methods."""
    results = {}
    
    for dataset in datasets:
        # Load data
        data = asymcat.read_sequences(f"resources/{dataset}")
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.CatScorer(cooccs)
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Time execution
        start_time = time.perf_counter()
        scores = getattr(scorer, method_name)()
        end_time = time.perf_counter()
        
        # Memory after
        memory_after = process.memory_info().rss
        
        results[dataset] = {
            'execution_time': end_time - start_time,
            'memory_used': memory_after - memory_before,
            'num_pairs': len(scores),
            'pairs_per_second': len(scores) / (end_time - start_time)
        }
    
    return results
```

#### Performance Regression Testing

```python
@pytest.mark.performance
def test_scoring_performance_regression():
    """Ensure performance doesn't regress between versions."""
    baseline_times = {
        'mle': 0.1,      # seconds for standard dataset
        'pmi': 0.15,
        'chi2': 0.2,
    }
    
    for method, baseline in baseline_times.items():
        start = time.time()
        getattr(standard_scorer, method)()
        duration = time.time() - start
        
        # Allow 50% performance degradation before failing
        assert duration < baseline * 1.5, f"{method} slower than baseline"
```

### Scalability Guidelines

#### Dataset Size Recommendations

| Dataset Size | Expected Performance | Recommended Approach |
|--------------|---------------------|----------------------|
| < 1K pairs | < 1 second | Standard processing |
| 1K - 10K pairs | 1-10 seconds | Standard with progress |
| 10K - 100K pairs | 10s - 2 minutes | Chunked processing |
| > 100K pairs | > 2 minutes | Streaming + parallel |

#### Memory Usage Optimization

```python
# Monitor memory usage during development
import tracemalloc

def trace_memory_usage(func):
    """Decorator to trace memory usage."""
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Current memory: {current / 1024 / 1024:.1f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
        return result
    return wrapper
```

## 📚 Documentation Standards

### API Documentation

#### Function Documentation Template

```python
def example_function(
    required_param: str,
    optional_param: int = 10,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """One-line summary of function purpose.
    
    Longer description explaining the function's behavior, use cases,
    and any important implementation details. Reference related functions
    or methods when helpful.
    
    Args:
        required_param: Description of required parameter.
            Include type information if not obvious from annotation.
        optional_param: Description of optional parameter.
            Include default value explanation. Default: 10.
        *args: Variable positional arguments passed to internal function.
        **kwargs: Additional keyword arguments:
            - special_option (bool): Enable special processing mode.
            - timeout (float): Operation timeout in seconds.
    
    Returns:
        Dictionary containing processed results with keys:
        - 'status': Processing status ('success' or 'error')
        - 'data': Main result data
        - 'metadata': Additional information about processing
    
    Raises:
        ValueError: If required_param is empty or invalid format.
        FileNotFoundError: If referenced files don't exist.
        TimeoutError: If processing exceeds timeout limit.
    
    Example:
        Basic usage:
        >>> result = example_function("input_data")
        >>> print(result['status'])
        'success'
        
        With optional parameters:
        >>> result = example_function(
        ...     "input_data", 
        ...     optional_param=20,
        ...     special_option=True
        ... )
        
    Note:
        This function is optimized for datasets < 10K items.
        For larger datasets, consider using streaming_function().
        
    See Also:
        related_function: Similar functionality with different approach.
        SomeClass.method: Alternative implementation in class context.
    """
```

### User Guides

#### Writing Effective Examples

**Progressive Complexity**:
```python
# Start with minimal example
import asymcat
data = asymcat.read_sequences("data.tsv")
scores = asymcat.CatScorer(asymcat.collect_cooccs(data)).mle()

# Then show realistic usage
import asymcat

# Load linguistic data
data = asymcat.read_sequences("phoneme_alignments.tsv")
cooccs = asymcat.collect_cooccs(data)

# Create scorer with smoothing for sparse data
scorer = asymcat.CatScorer(
    cooccs, 
    smoothing_method="laplace", 
    smoothing_alpha=1.0
)

# Compute multiple measures for comparison
results = {
    'mle': scorer.mle(),
    'pmi': scorer.pmi(), 
    'tresoldi': scorer.tresoldi()
}

# Finally show advanced workflow
import asymcat
import numpy as np
import matplotlib.pyplot as plt

# Complete analysis pipeline
data = asymcat.read_sequences("linguistic_data.tsv")
cooccs = asymcat.collect_cooccs(data, order=2, pad="#")
scorer = asymcat.CatScorer(cooccs, smoothing_method="lidstone", smoothing_alpha=0.5)

# Multiple scoring methods
measures = ['mle', 'pmi', 'chi2', 'tresoldi']
results = {measure: getattr(scorer, measure)() for measure in measures}

# Score transformations and visualization
for measure, scores in results.items():
    scaled = asymcat.scorer.scale_scorer(scores, method="minmax")
    xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scaled)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(xy_matrix, aspect='auto')
    plt.title(f"{measure.upper()} Scores (X→Y)")
    plt.colorbar()
    plt.show()
```

### Mathematical Documentation

#### Formula Documentation Standards

For mathematical functions, include LaTeX formulas:

```python
def pointwise_mutual_information(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Compute Pointwise Mutual Information (PMI) scores.
    
    PMI measures the information gained about one variable by observing another.
    Higher PMI indicates stronger positive association.
    
    Mathematical Definition:
        PMI(X,Y) = log(P(X,Y) / (P(X) * P(Y)))
        
        Where:
        - P(X,Y): Joint probability of X and Y co-occurring
        - P(X): Marginal probability of X
        - P(Y): Marginal probability of Y
    
    Range: (-∞, +∞)
        - PMI > 0: Positive association (more likely to co-occur)
        - PMI = 0: Independence (no association)
        - PMI < 0: Negative association (less likely to co-occur)
    
    Implementation Notes:
        Uses smoothed probability estimates to handle zero counts.
        Applies log in natural base for consistency with information theory.
    
    Returns:
        Dictionary mapping (x, y) pairs to (PMI(x,y), PMI(y,x)) tuples.
        
    Example:
        >>> scorer = CatScorer(cooccs)
        >>> pmi_scores = scorer.pmi()
        >>> xy_pmi, yx_pmi = pmi_scores[('cat', 'dog')]
        >>> print(f"PMI(cat,dog) = {xy_pmi:.3f}")
    """
```

### Jupyter Notebook Documentation

#### Notebook Structure Template

```python
# Cell 1: Introduction and Setup
"""
# ASymCat Tutorial: [Specific Topic]

This notebook demonstrates [specific functionality] using real-world data
from [domain]. We'll cover:

1. Data loading and preprocessing
2. Core analysis techniques  
3. Advanced applications
4. Interpretation and visualization
"""

import asymcat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Data Loading with Explanation
"""
## Data Loading

We'll use [dataset description] which contains [data characteristics].
This dataset is particularly suitable for demonstrating [specific concepts].
"""

# Load data with detailed explanation
data = asymcat.read_sequences("examples/linguistic_data.tsv")
print(f"Loaded {len(data)} sequence pairs")

# Show sample data
for i, (seq1, seq2) in enumerate(data[:3]):
    print(f"Pair {i+1}: {' '.join(seq1)} → {' '.join(seq2)}")

# Cell 3: Method Application with Theory
"""
## Association Measure: [Method Name]

[Theoretical background paragraph explaining the measure's foundation,
when to use it, and what it reveals about the data]

### Mathematical Foundation
[LaTeX formulas and interpretation]

### Implementation
"""

# Code with detailed comments
cooccs = asymcat.collect_cooccs(data)
scorer = asymcat.CatScorer(cooccs, smoothing_method="laplace")
scores = scorer.method_name()

# Analysis with interpretation
print(f"Computed scores for {len(scores)} category pairs")

# Cell 4: Results Interpretation
"""
## Results Analysis

Let's examine the strongest associations and what they reveal about our data.
"""

# Sort and display top results with interpretation
sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 X→Y associations:")
for (x, y), (xy_score, yx_score) in sorted_scores[:10]:
    print(f"{x}→{y}: {xy_score:.3f} (reverse: {yx_score:.3f})")
    
# Cell 5: Visualization with Discussion
"""
## Visualization and Patterns

[Discussion of patterns revealed by visualization]
"""

# Create informative visualizations
xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# X→Y heatmap
im1 = ax1.imshow(xy_matrix, aspect='auto', cmap='viridis')
ax1.set_title('X→Y Associations')
ax1.set_xlabel('Y Categories')
ax1.set_ylabel('X Categories')
plt.colorbar(im1, ax=ax1)

# Y→X heatmap  
im2 = ax2.imshow(yx_matrix, aspect='auto', cmap='viridis')
ax2.set_title('Y→X Associations')
ax2.set_xlabel('X Categories') 
ax2.set_ylabel('Y Categories')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

# Cell 6: Advanced Analysis
"""
## Advanced Applications

[Demonstrate sophisticated usage patterns and interpretation techniques]
"""

# Cell 7: Conclusions and Next Steps
"""
## Summary and Extensions

### Key Findings
- [Summarize main discoveries]
- [Highlight methodological insights]

### Further Exploration
- [Suggest related analyses]
- [Point to additional resources]
"""
```

## 🚀 Release Process

### Version Management

ASymCat follows **Semantic Versioning** (SemVer):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

#### Version Update Process

1. **Update Version Numbers**:
```bash
# Update pyproject.toml
[project]
version = "0.4.0"

# Update asymcat/__init__.py
__version__ = "0.4.0"

# Update citation in README.md
version = {0.4.0}
```

2. **Generate Changelog**:
```markdown
## [0.4.0] - 2024-MM-DD

### Added
- New Bayesian association measure
- CLI support for batch processing
- Performance optimizations for large datasets

### Changed  
- Improved numerical stability in PMI calculations
- Enhanced error messages for invalid inputs

### Fixed
- Memory leak in large dataset processing
- Incorrect handling of edge cases in Fisher exact test

### Deprecated
- `old_function()` will be removed in v0.5.0, use `new_function()`

### Removed
- Deprecated `legacy_scorer()` method (use `CatScorer` instead)

### Security
- Updated dependencies to address security vulnerabilities
```

### Pre-Release Checklist

```bash
# 1. Update version numbers
git checkout develop
# Update version in pyproject.toml and __init__.py

# 2. Run comprehensive testing
make test
make coverage
make format-check

# 3. Test CLI functionality
make cli-test

# 4. Performance benchmarking
pytest -m performance --benchmark

# 5. Documentation updates
# Update README.md, CHANGELOG.md, docs/

# 6. Build and test package
make build
python -m pip install dist/asymcat-*.whl
python -c "import asymcat; print(asymcat.__version__)"

# 7. Create release branch
git checkout -b release/v0.4.0
git commit -am "Prepare release v0.4.0"
git push origin release/v0.4.0
```

### Release Workflow

#### 1. **Create Release PR**:
- Merge `develop` → `release/vX.Y.Z`
- Update version numbers and documentation
- Run full test suite
- Review changes with maintainers

#### 2. **Tag and Release**:
```bash
# Merge to master
git checkout master
git merge release/v0.4.0

# Create annotated tag
git tag -a v0.4.0 -m "Release version 0.4.0

New features:
- Bayesian association measure
- Enhanced CLI batch processing
- Performance optimizations

See CHANGELOG.md for complete details."

# Push tag
git push origin v0.4.0
```

#### 3. **Publish Package**:
```bash
# Build distributions
make build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*

# Verify upload
pip install asymcat==0.4.0
```

#### 4. **Post-Release Tasks**:
- Update development version in `develop` branch
- Create GitHub release with changelog
- Update documentation website
- Announce release in relevant channels

### Hotfix Process

For critical bug fixes:

```bash
# Create hotfix branch from master
git checkout master
git checkout -b hotfix/v0.3.1

# Fix critical issue
# ... make changes ...

# Test thoroughly
make test
make cli-test

# Update version (patch level)
# Update pyproject.toml: version = "0.3.1"

# Commit and merge
git commit -am "Fix critical bug in data loading"
git checkout master
git merge hotfix/v0.3.1
git tag v0.3.1
git push origin master --tags

# Merge back to develop
git checkout develop
git merge master
```

## 🔧 Troubleshooting

### Common Development Issues

#### 1. **Import Errors**

**Problem**: `ModuleNotFoundError: No module named 'asymcat'`

**Solutions**:
```bash
# Ensure development installation
pip install -e ".[dev]"

# Check virtual environment
which python  # Should point to .venv/bin/python

# Verify installation
python -c "import asymcat; print(asymcat.__file__)"
```

#### 2. **Test Failures**

**Problem**: Tests fail with numerical precision errors

**Solutions**:
```python
# Use appropriate tolerance in assertions
assert abs(result - expected) < 1e-10

# Use numpy testing utilities
np.testing.assert_allclose(result, expected, rtol=1e-15)

# Check for platform-specific differences
import sys
if sys.platform == "win32":
    # Windows-specific tolerance
    tolerance = 1e-8
else:
    tolerance = 1e-12
```

#### 3. **Memory Issues**

**Problem**: Out of memory errors with large datasets

**Solutions**:
```python
# Use streaming processing
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield process_chunk(chunk)

# Clear intermediate results
del large_intermediate_result
import gc; gc.collect()

# Use memory-efficient data structures
from collections import Counter, defaultdict
counts = Counter(pairs)  # More efficient than dict
```

#### 4. **Performance Issues**

**Problem**: Slow execution on large datasets

**Diagnosis**:
```python
# Profile code to identify bottlenecks
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... run slow code ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

**Optimization**:
```python
# Use vectorized operations
# Slow: Python loop
result = [expensive_func(x) for x in data]

# Fast: NumPy vectorization
data_array = np.array(data)
result = np.vectorize(expensive_func)(data_array)

# Use appropriate algorithms
# O(n²) → O(n log n) improvements
sorted_data = sorted(data)  # Use for binary search
lookup_dict = {k: v for k, v in pairs}  # O(1) lookups
```

### Environment Issues

#### Python Version Compatibility

**Problem**: Code fails on different Python versions

**Solutions**:
```python
# Use version-compatible imports
try:
    from typing import TypedDict  # Python 3.10+
except ImportError:
    from typing_extensions import TypedDict

# Check version requirements
import sys
if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10+ required")

# Use compatible syntax
# Python 3.10+: walrus operator (available since 3.8, but we require 3.10+)
if (n := len(data)) > 100:
    process_large_data(data)

# Compatible alternative:
n = len(data)
if n > 100:
    process_large_data(data)
```

#### Dependency Conflicts

**Problem**: Package version conflicts

**Solutions**:
```bash
# Check dependency tree
pip list
pip show asymcat

# Resolve conflicts
pip install --upgrade package_name

# Clean environment
pip uninstall asymcat
pip install -e ".[dev]"

# Use dependency constraints
pip install "numpy>=1.20,<2.0"
```

### Debugging Strategies

#### Logging Configuration

```python
import logging

# Configure detailed logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asymcat_debug.log'),
        logging.StreamHandler()
    ]
)

# Add logging to functions
logger = logging.getLogger(__name__)

def debug_scorer_method(self, data):
    logger.debug(f"Processing {len(data)} co-occurrences")
    result = self.compute_scores(data)
    logger.debug(f"Generated {len(result)} scores")
    return result
```

#### Interactive Debugging

```python
# Use debugger for complex issues
import pdb; pdb.set_trace()  # Breakpoint

# IPython enhanced debugging
import IPython; IPython.embed()

# Jupyter notebook debugging
from IPython.core.debugger import set_trace
set_trace()
```

### Getting Help

#### Internal Resources
1. Check existing tests for usage examples
2. Review `tests/fixtures/` for test data patterns
3. Examine similar functions in the codebase
4. Look at `resources/` for sample datasets

#### External Resources
1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check ReadTheDocs for API details
3. **Community**: Engage with users and contributors
4. **Scientific Literature**: Reference mathematical foundations

#### Contribution Guidelines
1. Always add tests for new functionality
2. Follow established code patterns
3. Update documentation with changes
4. Consider backward compatibility
5. Validate performance impact

---

This developer guide provides comprehensive information for contributing to ASymCat. For additional questions or clarifications, please open an issue on GitHub or consult the project documentation.