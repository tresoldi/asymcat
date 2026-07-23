"""
Performance benchmarks for the :class:`asymcat.scorer.CatScorer` measures.

These benchmarks use ``pytest-benchmark`` to record the wall-clock cost of the
main scoring measures on a realistic (phoneme-level) dataset. They are intended
to:

* validate the "scalable" claim by giving concrete, reproducible numbers;
* highlight the relative cost of the different measures (in particular the
  entropy-based ``theil_u`` / ``cond_entropy``, which iterate over the full
  ``alphabet_x`` x ``alphabet_y`` grid and are quadratic in alphabet size); and
* provide a baseline for spotting performance regressions over time.

Every test name contains ``performance``, so the shared ``conftest`` marks them
as ``slow`` (and ``performance``) and they are skipped by default. To run them::

    pytest tests/performance --run-slow --no-cov --benchmark-only

The ``--no-cov`` flag is recommended because coverage instrumentation
significantly skews the recorded timings.
"""

import pytest

import asymcat
from asymcat.scorer import CatScorer

from ..fixtures.data import RESOURCE_DIR

# Dataset small enough that expensive measures still complete within a few
# benchmark rounds, while being large enough for the timings to be meaningful.
BENCHMARK_DATASET = "cmudict.sample100.tsv"


@pytest.fixture(scope="module")
def benchmark_cooccs():
    """Collect co-occurrences from the benchmark dataset once per module."""
    file_path = RESOURCE_DIR / BENCHMARK_DATASET
    if not file_path.exists():
        pytest.skip(f"Benchmark dataset {BENCHMARK_DATASET} not available")

    data = asymcat.read_sequences(str(file_path))
    return asymcat.collect_cooccs(data)


def _benchmark_measure(benchmark, cooccs, method_name: str, cache_attr: str):
    """Benchmark a single scorer measure with a cold cache on each round.

    The scorer memoizes results, so the private cache attribute is reset before
    each invocation to ensure the benchmark measures the computation itself
    rather than a dictionary lookup.
    """
    scorer = CatScorer(cooccs)

    def run():
        setattr(scorer, cache_attr, None)
        return getattr(scorer, method_name)()

    result = benchmark(run)

    # Sanity check: the measure produced a non-empty score mapping.
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.benchmark(group="probabilistic")
def test_mle_performance(benchmark, benchmark_cooccs):
    """Benchmark MLE scoring (cheap probabilistic baseline)."""
    _benchmark_measure(benchmark, benchmark_cooccs, "mle", "_mle")


@pytest.mark.benchmark(group="information-theoretic")
def test_pmi_performance(benchmark, benchmark_cooccs):
    """Benchmark PMI scoring."""
    _benchmark_measure(benchmark, benchmark_cooccs, "pmi", "_pmi")


@pytest.mark.benchmark(group="information-theoretic")
def test_mutual_information_performance(benchmark, benchmark_cooccs):
    """Benchmark mutual information scoring."""
    _benchmark_measure(benchmark, benchmark_cooccs, "mutual_information", "_mutual_information")


@pytest.mark.benchmark(group="entropy")
def test_theil_u_performance(benchmark, benchmark_cooccs):
    """Benchmark Theil's U (quadratic in alphabet size; flagged as a hot path)."""
    _benchmark_measure(benchmark, benchmark_cooccs, "theil_u", "_theil_u")


@pytest.mark.benchmark(group="entropy")
def test_cond_entropy_performance(benchmark, benchmark_cooccs):
    """Benchmark conditional entropy scoring."""
    _benchmark_measure(benchmark, benchmark_cooccs, "cond_entropy", "_cond_entropy")


@pytest.mark.benchmark(group="specialized")
def test_tresoldi_performance(benchmark, benchmark_cooccs):
    """Benchmark the custom Tresoldi measure."""
    _benchmark_measure(benchmark, benchmark_cooccs, "tresoldi", "_tresoldi")
