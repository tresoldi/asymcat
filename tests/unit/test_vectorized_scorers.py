"""
Regression tests for the vectorized ``CatScorer`` measures.

``theil_u``, ``cond_entropy``, ``mutual_information``,
``normalized_mutual_information`` and ``goodman_kruskal_lambda`` are all
computed with vectorized formulations that evaluate every symbol pair at once
from the global joint-count matrix. These tests pin each one to a
straightforward reference implementation of the original per-pair algorithm
(re-filtering the co-occurrence list for each pair and calling the
corresponding scalar ``compute_*`` helper), guaranteeing the fast paths stay
faithful to the definitions -- including the degenerate, zero-entropy cases.
"""

import math

import pytest

import asymcat
from asymcat.scorer import (
    CatScorer,
    compute_goodman_kruskal_lambda,
    compute_mutual_information,
    compute_normalized_mutual_information,
    compute_theil_u,
    conditional_entropy,
)

from ..fixtures.data import RESOURCE_DIR


def _reference_per_pair(cooccs, alphabet_x, alphabet_y, first, second):
    """Build the original per-pair result from two subset-level scalar functions."""
    result = {}
    for x in alphabet_x:
        for y in alphabet_y:
            subset = [pair for pair in cooccs if pair[0] == x or pair[1] == y]
            xs = [pair[0] for pair in subset]
            ys = [pair[1] for pair in subset]
            result[(x, y)] = (first(xs, ys), second(xs, ys))
    return result


def _reference_theil_u(cooccs, alphabet_x, alphabet_y):
    """Original per-pair Theil's U algorithm, used as the correctness oracle."""
    return _reference_per_pair(
        cooccs, alphabet_x, alphabet_y, lambda xs, ys: compute_theil_u(ys, xs), lambda xs, ys: compute_theil_u(xs, ys)
    )


def _reference_cond_entropy(cooccs, alphabet_x, alphabet_y):
    """Original per-pair conditional-entropy algorithm, used as the oracle."""
    return _reference_per_pair(
        cooccs,
        alphabet_x,
        alphabet_y,
        lambda xs, ys: conditional_entropy(ys, xs),
        lambda xs, ys: conditional_entropy(xs, ys),
    )


def _reference_mutual_information(cooccs, alphabet_x, alphabet_y):
    return _reference_per_pair(
        cooccs,
        alphabet_x,
        alphabet_y,
        lambda xs, ys: compute_mutual_information(xs, ys),
        lambda xs, ys: compute_mutual_information(ys, xs),
    )


def _reference_normalized_mutual_information(cooccs, alphabet_x, alphabet_y):
    return _reference_per_pair(
        cooccs,
        alphabet_x,
        alphabet_y,
        lambda xs, ys: compute_normalized_mutual_information(xs, ys),
        lambda xs, ys: compute_normalized_mutual_information(ys, xs),
    )


def _reference_goodman_kruskal_lambda(cooccs, alphabet_x, alphabet_y):
    return _reference_per_pair(
        cooccs,
        alphabet_x,
        alphabet_y,
        lambda xs, ys: compute_goodman_kruskal_lambda(xs, ys, "y_given_x"),
        lambda xs, ys: compute_goodman_kruskal_lambda(xs, ys, "x_given_y"),
    )


def _assert_pairs_match(actual, reference):
    assert set(actual) == set(reference), "Pair keys differ from reference"
    for pair, (ref_first, ref_second) in reference.items():
        act_first, act_second = actual[pair]
        for ref_val, act_val in ((ref_first, act_first), (ref_second, act_second)):
            if math.isnan(ref_val):
                assert math.isnan(act_val), f"{pair}: expected NaN, got {act_val}"
            else:
                assert act_val == pytest.approx(ref_val, rel=1e-9, abs=1e-12), (
                    f"{pair}: expected {ref_val}, got {act_val}"
                )


# -- datasets: factories so file-based ones can skip when a resource is absent --
def _cooccs_simple():
    return asymcat.collect_cooccs([[["a", "b"], ["x", "y"]], [["a", "c"], ["x", "z"]]])


def _cooccs_single_pair():
    """A single co-occurrence yields degenerate, zero-entropy subsets."""
    return asymcat.collect_cooccs([[["a"], ["x"]]])


def _cooccs_degenerate_column():
    """Many x symbols mapping to one y symbol exercises the H(X)==0 branch."""
    return asymcat.collect_cooccs([[["a", "b", "c"], ["x", "x", "x"]]])


def _cooccs_ties():
    """Tied cell counts exercise the argmax / max-excluding logic of lambda."""
    return asymcat.collect_cooccs([[["a", "a", "b", "b"], ["x", "y", "x", "y"]]])


def _cooccs_toy_file():
    toy = RESOURCE_DIR / "toy.tsv"
    if not toy.exists():
        pytest.skip("toy.tsv not available")
    return asymcat.collect_cooccs(asymcat.read_sequences(str(toy)))


def _cooccs_galapagos():
    galapagos = RESOURCE_DIR / "galapagos.tsv"
    if not galapagos.exists():
        pytest.skip("galapagos.tsv not available")
    return asymcat.read_pa_matrix(str(galapagos))


DATASETS = [
    ("simple", _cooccs_simple),
    ("single_pair", _cooccs_single_pair),
    ("degenerate_column", _cooccs_degenerate_column),
    ("ties", _cooccs_ties),
    ("toy_file", _cooccs_toy_file),
    ("galapagos", _cooccs_galapagos),
]

# (scorer method name, reference builder)
MEASURES = [
    ("theil_u", _reference_theil_u),
    ("cond_entropy", _reference_cond_entropy),
    ("mutual_information", _reference_mutual_information),
    ("normalized_mutual_information", _reference_normalized_mutual_information),
    ("goodman_kruskal_lambda", _reference_goodman_kruskal_lambda),
]


@pytest.mark.parametrize("dataset_name, cooccs_factory", DATASETS, ids=[d[0] for d in DATASETS])
@pytest.mark.parametrize("measure_name, reference_builder", MEASURES, ids=[m[0] for m in MEASURES])
def test_vectorized_matches_reference(measure_name, reference_builder, dataset_name, cooccs_factory):
    """Each vectorized scorer reproduces the original per-pair algorithm."""
    cooccs = cooccs_factory()
    scorer = CatScorer(cooccs)
    actual = getattr(scorer, measure_name)()
    reference = reference_builder(cooccs, scorer.alphabet_x, scorer.alphabet_y)
    _assert_pairs_match(actual, reference)


@pytest.mark.parametrize("measure_name, _reference", MEASURES, ids=[m[0] for m in MEASURES])
def test_result_is_cached(measure_name, _reference):
    """Repeated calls return the same memoized object."""
    scorer = CatScorer(_cooccs_simple())
    method = getattr(scorer, measure_name)
    assert method() is method()


class TestVectorizedValueRanges:
    """Sanity bounds for the vectorized measures."""

    @pytest.fixture
    def scorer(self):
        cooccs = asymcat.collect_cooccs([[["a", "b", "c"], ["x", "y", "z"]], [["a", "b"], ["x", "y"]]])
        return CatScorer(cooccs)

    @pytest.mark.parametrize("measure_name", ["theil_u", "normalized_mutual_information", "goodman_kruskal_lambda"])
    def test_within_unit_range(self, scorer, measure_name):
        for first, second in getattr(scorer, measure_name)().values():
            assert 0.0 <= first <= 1.0
            assert 0.0 <= second <= 1.0

    @pytest.mark.parametrize("measure_name", ["cond_entropy", "mutual_information"])
    def test_non_negative(self, scorer, measure_name):
        for first, second in getattr(scorer, measure_name)().values():
            assert first >= 0.0
            assert second >= 0.0

    def test_mutual_information_is_symmetric(self, scorer):
        """MI(X;Y) == MI(Y;X): both tuple entries must be equal."""
        for first, second in scorer.mutual_information().values():
            assert first == second
