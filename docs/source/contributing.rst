Contributing to ASymCat
=======================

We welcome contributions to ASymCat! This guide will help you get started with contributing to the project, whether you're fixing bugs, adding features, improving documentation, or enhancing tests.

.. contents:: Table of Contents
   :local:
   :depth: 2

Getting Started
---------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

.. code-block:: bash

    # Fork the repository on GitHub, then clone your fork
    git clone https://github.com/YOUR_USERNAME/asymcat.git
    cd asymcat
    
    # Add upstream remote
    git remote add upstream https://github.com/tresoldi/asymcat.git

2. **Set Up Development Environment**

.. code-block:: bash

    # Create and activate virtual environment
    make install  # Uses make for automated setup
    
    # Or manually:
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\\Scripts\\activate  # Windows
    
    # Install in development mode with all dependencies
    pip install -e ".[dev,all]"

3. **Verify Installation**

.. code-block:: bash

    # Run tests to ensure everything works
    make test
    
    # Check code quality
    make format-check

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Create Feature Branch**

.. code-block:: bash

    git checkout -b feature/new-scoring-method
    # or
    git checkout -b fix/data-loading-bug

2. **Make Changes**
   - Write code following established patterns
   - Add comprehensive tests
   - Update documentation
   - Follow type annotation standards

3. **Test Your Changes**

.. code-block:: bash

    # Run full test suite
    make test
    
    # Check code formatting and style
    make format-check
    
    # Run specific test categories
    pytest tests/unit/           # Unit tests only
    pytest tests/integration/    # Integration tests only
    pytest -m slow              # Performance tests

4. **Commit and Push**

.. code-block:: bash

    # Stage your changes
    git add .
    
    # Commit with descriptive message
    git commit -m "Add Bayesian association measure
    
    - Implement Bayesian posterior probability estimation
    - Add comprehensive test suite
    - Include performance benchmarks
    - Update CLI to support new method"
    
    # Push to your fork
    git push origin feature/new-scoring-method

5. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Use the pull request template
   - Include detailed description and testing information

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment information** (Python version, OS, ASymCat version)
- **Minimal example** that demonstrates the bug

**Example Bug Report:**

.. code-block:: text

    **Bug Description**
    MLE scores return NaN for empty co-occurrence lists
    
    **Steps to Reproduce**
    ```python
    import asymcat
    scorer = asymcat.CatScorer([])  # Empty co-occurrences
    scores = scorer.mle()  # Returns NaN values
    ```
    
    **Expected Behavior**
    Should raise informative ValueError about empty data
    
    **Environment**
    - ASymCat version: 0.3.0
    - Python version: 3.11.0
    - OS: Ubuntu 22.04

Feature Requests
~~~~~~~~~~~~~~~

For new features, please provide:

- **Clear use case** and motivation
- **Detailed description** of proposed functionality
- **Example usage** showing how it would work
- **References** to relevant literature (for new measures)

Code Contributions
~~~~~~~~~~~~~~~~~

Areas where contributions are especially welcome:

1. **New Association Measures**
   - Novel asymmetric measures from literature
   - Domain-specific measures
   - Improved computational efficiency

2. **Performance Improvements**
   - Algorithmic optimizations
   - Memory usage reduction
   - Parallel processing support

3. **New Data Formats**
   - Additional input format support
   - Export format extensions
   - Integration with other libraries

4. **Visualization Enhancements**
   - Interactive plotting capabilities
   - Additional chart types
   - Export format options

5. **Documentation**
   - Tutorial improvements
   - Example notebooks
   - API documentation enhancements

Adding New Association Measures
-------------------------------

If you want to add a new association measure, follow this process:

1. **Research and Planning**
   - Review relevant literature
   - Understand mathematical properties
   - Consider computational complexity
   - Plan test cases

2. **Implementation**

.. code-block:: python

    # Add to asymcat/scorer.py in CatScorer class
    def new_measure(self, parameter: float = 1.0) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Implement new association measure.
        
        Description of the measure, its mathematical foundation,
        and when to use it.
        
        Args:
            parameter: Method-specific parameter with clear description.
                      Default value should be theoretically motivated.
        
        Returns:
            Dictionary mapping category pairs to directional scores.
            Keys are (category_x, category_y) tuples.
            Values are (x→y_score, y→x_score) tuples.
        
        Raises:
            ValueError: If parameter is invalid or data is empty.
            
        Example:
            >>> scorer = CatScorer(cooccs)
            >>> scores = scorer.new_measure(parameter=2.0)
            >>> xy_score, yx_score = scores[('A', 'B')]
            
        References:
            Author, A. (Year). Title. Journal, Volume(Issue), pages.
        """
        # Input validation
        if parameter <= 0:
            raise ValueError("Parameter must be positive")
        
        if not self.pairs:
            raise ValueError("No co-occurrences available")
        
        # Get probability estimates (handles smoothing automatically)
        p_xy, p_x, p_y = self.get_smoothed_probabilities()
        
        # Implement core algorithm
        scores = {}
        for (x, y) in self.pairs:
            # Calculate directional scores
            xy_score = self._compute_xy_score(x, y, p_xy, p_x, p_y, parameter)
            yx_score = self._compute_yx_score(x, y, p_xy, p_x, p_y, parameter)
            
            scores[(x, y)] = (xy_score, yx_score)
        
        return scores

3. **Add to CLI Interface**

.. code-block:: python

    # In asymcat/__main__.py, add to AVAILABLE_SCORERS
    AVAILABLE_SCORERS = [
        "mle", "pmi", "chi2", "fisher", "theil_u",
        "new_measure",  # Add your measure here
        "all"
    ]

4. **Write Comprehensive Tests**

.. code-block:: python

    # In tests/unit/test_scoring_measures.py
    class TestNewMeasure:
        """Test suite for new association measure."""
        
        @pytest.mark.parametrize("dataset", STANDARD_DATASETS)
        def test_new_measure_basic(self, dataset):
            """Test basic functionality."""
            scorer = self.get_scorer(dataset)
            scores = scorer.new_measure()
            
            # Validate output format
            assert_valid_scores(scores)
            assert len(scores) > 0
            
        @pytest.mark.parametrize("parameter", [0.1, 1.0, 2.0, 5.0])
        def test_parameter_sensitivity(self, sample_cooccs, parameter):
            """Test parameter effects."""
            scorer = CatScorer(sample_cooccs)
            scores = scorer.new_measure(parameter=parameter)
            assert_valid_scores(scores)
            
        def test_mathematical_properties(self, sample_cooccs):
            """Test measure-specific mathematical properties."""
            scorer = CatScorer(sample_cooccs)
            scores = scorer.new_measure()
            
            # Test specific properties of your measure
            for (x, y), (xy_score, yx_score) in scores.items():
                # Example: test range constraints
                assert 0 <= xy_score <= 1  # If measure is bounded [0,1]
                assert 0 <= yx_score <= 1
                
        def test_edge_cases(self):
            """Test edge cases and error conditions."""
            # Test with minimal data
            minimal_cooccs = [("A", "B", 1)]
            scorer = CatScorer(minimal_cooccs)
            scores = scorer.new_measure()
            assert len(scores) == 1
            
            # Test invalid parameters
            with pytest.raises(ValueError):
                scorer.new_measure(parameter=-1.0)

5. **Update Documentation**

Add to the appropriate documentation files:

- Mathematical formulation in ``mathematical-foundations.rst``
- Usage examples in ``tutorial.rst``
- API documentation (automatically generated from docstrings)

Code Style Guidelines
--------------------

Python Code Style
~~~~~~~~~~~~~~~~~

ASymCat follows PEP 8 with these specific guidelines:

1. **Formatting**: Use Black for automatic formatting

.. code-block:: bash

    make black  # Auto-format code

2. **Import Organization**: Use isort

.. code-block:: bash

    make isort  # Organize imports

3. **Linting**: Use flake8 for code quality

.. code-block:: bash

    make lint  # Check code quality

4. **Type Annotations**: Include comprehensive type hints

.. code-block:: python

    from typing import Dict, List, Tuple, Optional, Union

    def example_function(
        data: List[Tuple[str, str]], 
        parameter: float = 1.0,
        optional_arg: Optional[str] = None
    ) -> Dict[str, float]:
        """Function with proper type annotations."""
        pass

Documentation Style
~~~~~~~~~~~~~~~~~~

1. **Docstring Format**: Use Google-style docstrings

.. code-block:: python

    def example_function(param1: str, param2: int = 10) -> bool:
        """One-line summary of function.
        
        Longer description explaining the function's behavior,
        use cases, and important implementation details.
        
        Args:
            param1: Description of first parameter.
            param2: Description with default value info. Default: 10.
            
        Returns:
            Description of return value and its format.
            
        Raises:
            ValueError: When parameter is invalid.
            RuntimeError: When operation fails.
            
        Example:
            >>> result = example_function("input", param2=20)
            >>> print(result)
            True
            
        Note:
            Additional notes about usage or performance.
        """

2. **Comments**: Add comments for complex algorithmic decisions

.. code-block:: python

    # Use log-space computation to prevent numerical underflow
    # when dealing with very small probabilities
    log_values = np.log(data + self.smoothing_alpha)

3. **README Updates**: Update README.md for user-facing changes

Testing Standards
-----------------

Test Organization
~~~~~~~~~~~~~~~~

Tests are organized in a hierarchical structure:

.. code-block:: text

    tests/
    ├── unit/                      # Test individual components
    │   ├── test_data_loading.py   # Data I/O and preprocessing
    │   ├── test_scoring_measures.py # Individual scoring methods
    │   └── test_score_transformations.py # Utilities
    ├── integration/               # Test complete workflows
    │   └── test_end_to_end_workflows.py # Full analysis pipelines
    └── fixtures/                  # Shared test infrastructure
        ├── data.py               # Test datasets
        └── assertions.py         # Custom validation functions

Test Writing Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Comprehensive Coverage**: Test normal cases, edge cases, and error conditions

.. code-block:: python

    def test_comprehensive_coverage():
        # Normal case
        result = function_under_test(valid_input)
        assert result == expected_output
        
        # Edge case: empty input
        result = function_under_test([])
        assert result == expected_empty_result
        
        # Error case: invalid input
        with pytest.raises(ValueError):
            function_under_test(invalid_input)

2. **Parameterized Tests**: Test multiple scenarios efficiently

.. code-block:: python

    @pytest.mark.parametrize("dataset,expected_pairs", [
        ("toy.tsv", 10),
        ("mushroom-small.tsv", 50),
        ("cmu_sample.tsv", 100),
    ])
    def test_multiple_datasets(self, dataset: str, expected_pairs: int):
        """Test across different datasets."""
        data = asymcat.read_sequences(RESOURCE_DIR / dataset)
        cooccs = asymcat.collect_cooccs(data)
        assert len(cooccs) >= expected_pairs

3. **Property-Based Testing**: Validate mathematical properties

.. code-block:: python

    def test_mathematical_properties(self, sample_cooccs):
        """Test mathematical properties rather than exact values."""
        scores = CatScorer(sample_cooccs).mle()
        
        for (x, y), (xy_score, yx_score) in scores.items():
            # MLE scores should be valid probabilities
            assert 0 <= xy_score <= 1
            assert 0 <= yx_score <= 1

4. **Performance Testing**: Include timing constraints for slow operations

.. code-block:: python

    @pytest.mark.slow
    def test_performance_constraint(self, large_dataset):
        """Ensure performance remains acceptable."""
        start_time = time.time()
        result = expensive_operation(large_dataset)
        duration = time.time() - start_time
        
        assert duration < 10.0  # 10 second limit
        assert len(result) > 0  # Meaningful output

Documentation Contributions
---------------------------

Types of Documentation
~~~~~~~~~~~~~~~~~~~~~~

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Tutorial and how-to documentation
3. **Examples**: Jupyter notebooks with real-world applications
4. **Reference**: Mathematical foundations and technical details

Writing Guidelines
~~~~~~~~~~~~~~~~~

1. **Clarity**: Write for your target audience (users vs developers)
2. **Examples**: Include working code examples
3. **Completeness**: Cover common use cases and edge cases
4. **Accuracy**: Ensure technical accuracy and test code examples

Building Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Build Sphinx documentation
    make docs
    
    # Clean and rebuild
    make docs-clean
    make docs

The built documentation will be available in ``docs/_build/html/index.html``.

Pull Request Process
--------------------

Pull Request Template
~~~~~~~~~~~~~~~~~~~~~

When creating a pull request, include:

.. code-block:: text

    ## Summary
    Brief description of changes and motivation.
    
    ## Changes Made
    - [ ] Added new feature: `new_association_measure()`
    - [ ] Implemented comprehensive test suite
    - [ ] Updated CLI to support new measure
    - [ ] Added performance benchmarks
    
    ## Testing
    - [ ] All existing tests pass
    - [ ] New tests cover edge cases and performance
    - [ ] Manual testing completed
    
    ## Documentation
    - [ ] Docstrings added to new functions
    - [ ] Tutorial updated with usage examples
    - [ ] Mathematical foundations documented
    
    ## Breaking Changes
    None / List any breaking changes
    
    ## Checklist
    - [ ] Code follows project style guidelines
    - [ ] Self-review completed
    - [ ] Tests added for new functionality
    - [ ] Documentation updated

Review Process
~~~~~~~~~~~~~

1. **Automated Checks**: CI runs tests and style checks
2. **Code Review**: Maintainers review for:
   - Code correctness and efficiency
   - Test coverage and quality
   - Documentation completeness
   - API design consistency
3. **Feedback**: Address review comments promptly
4. **Approval**: Once approved, maintainers will merge

Getting Help
-----------

Community Resources
~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Check existing docs before asking
- **Code Examples**: Look at tests and existing implementations

Communication Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

- **Be respectful**: Follow the code of conduct
- **Be specific**: Provide clear problem descriptions
- **Be patient**: Maintainers are volunteers
- **Be helpful**: Help others when you can

Recognition
-----------

Contributors are recognized in:

- **CHANGELOG.md**: Major contributions noted in release notes
- **AUTHORS.md**: All contributors listed
- **GitHub**: Contributor graphs and commit history
- **Releases**: Acknowledgment in release announcements

Thank you for contributing to ASymCat! Your contributions help make asymmetric categorical association analysis accessible to researchers across disciplines.