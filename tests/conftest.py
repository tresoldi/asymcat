"""
Pytest configuration for ASymCat test suite.

Provides shared fixtures, markers, and configuration for the modernized test structure.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take several seconds)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests that measure performance")
    config.addinivalue_line("markers", "large_data: marks tests that require large datasets")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and name."""
    for item in items:
        # Mark tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark slow tests based on name patterns
        if any(keyword in item.name.lower() for keyword in ["performance", "large", "comprehensive"]):
            item.add_marker(pytest.mark.slow)

        # Mark performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)

        # Mark tests that require large datasets
        if any(keyword in item.name.lower() for keyword in ["large_data", "scalability", "cmudict.tsv"]):
            item.add_marker(pytest.mark.large_data)


@pytest.fixture(scope="session")
def resource_dir():
    """Provide the path to test resource directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture(scope="session")
def sample_datasets(resource_dir):
    """Provide information about available sample datasets."""
    datasets = {}

    # Check which datasets are available
    potential_datasets = [
        ("toy", "toy.tsv", "Small toy dataset for basic testing"),
        ("mushroom_small", "mushroom-small.tsv", "Small mushroom classification dataset"),
        ("cmu_sample_100", "cmudict.sample100.tsv", "100-word CMU dictionary sample"),
        ("cmu_sample_1000", "cmudict.sample1000.tsv", "1000-word CMU dictionary sample"),
        ("cmu_full", "cmudict.tsv", "Full CMU pronunciation dictionary"),
        ("galapagos", "galapagos.tsv", "Galapagos finches presence-absence matrix"),
        ("wiktionary", "wiktionary.tsv", "Wiktionary multilingual data"),
    ]

    for name, filename, description in potential_datasets:
        file_path = resource_dir / filename
        if file_path.exists():
            datasets[name] = {
                "path": file_path,
                "filename": filename,
                "description": description,
                "size": file_path.stat().st_size,
            }

    return datasets


@pytest.fixture(scope="function")
def performance_timer():
    """Provide a context manager for timing test execution."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.duration = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time

        def elapsed(self):
            if self.start_time is None:
                return 0.0
            return time.time() - self.start_time

    return Timer


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings that don't affect test validity."""
    # Suppress numpy warnings about invalid values (common in statistical measures)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")

    # Suppress pandas warnings if pandas is used
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


@pytest.fixture(scope="function")
def clean_scorer_cache():
    """Ensure each test starts with a clean scorer cache."""
    # This fixture can be used to clear any global caches
    # that might interfere between tests
    yield
    # Cleanup code here if needed


# Pytest command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")

    parser.addoption(
        "--run-large-data", action="store_true", default=False, help="run tests that require large datasets"
    )

    parser.addoption("--performance-only", action="store_true", default=False, help="run only performance tests")


def pytest_runtest_setup(item):
    """Skip tests based on command line options."""
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")

    if "large_data" in item.keywords and not item.config.getoption("--run-large-data"):
        pytest.skip("need --run-large-data option to run")

    if item.config.getoption("--performance-only") and "performance" not in item.keywords:
        pytest.skip("running only performance tests")


# Custom assertions for better test output
class ASymCatAssertions:
    """Custom assertions for ASymCat testing."""

    @staticmethod
    def assert_scoring_method_exists(scorer, method_name: str):
        """Assert that a scoring method exists on the scorer."""
        assert hasattr(scorer, method_name), f"Scorer missing expected method: {method_name}"

        method = getattr(scorer, method_name)
        assert callable(method), f"Scorer attribute {method_name} is not callable"

    @staticmethod
    def assert_reasonable_execution_time(duration: float, max_seconds: float = 30.0):
        """Assert that execution completed within reasonable time."""
        assert duration < max_seconds, f"Execution took too long: {duration:.2f}s (max: {max_seconds}s)"

    @staticmethod
    def assert_score_properties(
        scores: Dict[Any, Any], min_pairs: int = 1, allow_negative: bool = True, require_finite: bool = True
    ):
        """Assert general properties of scoring results."""
        assert isinstance(scores, dict), "Scores must be a dictionary"
        assert len(scores) >= min_pairs, f"Too few pairs: {len(scores)} < {min_pairs}"

        for pair, (xy, yx) in scores.items():
            if require_finite:
                assert all(
                    float('-inf') < val < float('inf') for val in [xy, yx] if val != float('inf')
                ), f"Non-finite values in scores for {pair}: {xy}, {yx}"

            if not allow_negative:
                assert xy >= 0 and yx >= 0, f"Negative values not allowed for {pair}: {xy}, {yx}"


@pytest.fixture(scope="session")
def asymcat_assertions():
    """Provide custom ASymCat assertions."""
    return ASymCatAssertions
