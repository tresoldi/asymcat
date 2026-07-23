"""
Regression tests for the vectorized ``CatScorer.theil_u`` and
``CatScorer.cond_entropy`` implementations.

Both scorers are computed with a shared vectorized formulation that evaluates
all symbol pairs at once from the global joint-count matrix. These tests pin
their output to straightforward reference implementations of the original,
per-pair algorithms (re-filtering the co-occurrence list for each pair and
calling the scalar ``compute_theil_u`` / ``conditional_entropy``), guaranteeing
the fast path stays faithful to the definition -- including the degenerate,
zero-entropy cases.
"""

import math

import pytest

import asymcat
from asymcat.scorer import CatScorer, compute_theil_u, conditional_entropy

from ..fixtures.data import RESOURCE_DIR


def _reference_theil_u(cooccs, alphabet_x, alphabet_y):
    """Original per-pair Theil's U algorithm, used as the correctness oracle."""
    result = {}
    for x in alphabet_x:
        for y in alphabet_y:
            subset = [pair for pair in cooccs if pair[0] == x or pair[1] == y]
            xs = [pair[0] for pair in subset]
            ys = [pair[1] for pair in subset]
            result[(x, y)] = (compute_theil_u(ys, xs), compute_theil_u(xs, ys))
    return result


def _reference_cond_entropy(cooccs, alphabet_x, alphabet_y):
    """Original per-pair conditional-entropy algorithm, used as the oracle."""
    result = {}
    for x in alphabet_x:
        for y in alphabet_y:
            subset = [pair for pair in cooccs if pair[0] == x or pair[1] == y]
            xs = [pair[0] for pair in subset]
            ys = [pair[1] for pair in subset]
            result[(x, y)] = (conditional_entropy(ys, xs), conditional_entropy(xs, ys))
    return result


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


def _assert_matches_reference(cooccs):
    scorer = CatScorer(cooccs)
    _assert_pairs_match(scorer.theil_u(), _reference_theil_u(cooccs, scorer.alphabet_x, scorer.alphabet_y))


def _assert_cond_entropy_matches_reference(cooccs):
    scorer = CatScorer(cooccs)
    _assert_pairs_match(scorer.cond_entropy(), _reference_cond_entropy(cooccs, scorer.alphabet_x, scorer.alphabet_y))


class TestTheilUVectorization:
    """The vectorized result must match the per-pair reference implementation."""

    def test_matches_reference_simple(self):
        cooccs = asymcat.collect_cooccs([[["a", "b"], ["x", "y"]], [["a", "c"], ["x", "z"]]])
        _assert_matches_reference(cooccs)

    def test_matches_reference_single_pair(self):
        """A single co-occurrence yields degenerate (zero-entropy) subsets."""
        cooccs = asymcat.collect_cooccs([[["a"], ["x"]]])
        _assert_matches_reference(cooccs)

    def test_matches_reference_degenerate_column(self):
        """Many x symbols mapping to one y symbol exercises the H(X)==0 branch."""
        cooccs = asymcat.collect_cooccs([[["a", "b", "c"], ["x", "x", "x"]]])
        _assert_matches_reference(cooccs)

    def test_matches_reference_toy_file(self):
        toy = RESOURCE_DIR / "toy.tsv"
        if not toy.exists():
            pytest.skip("toy.tsv not available")
        cooccs = asymcat.collect_cooccs(asymcat.read_sequences(str(toy)))
        _assert_matches_reference(cooccs)

    def test_matches_reference_presence_absence(self):
        galapagos = RESOURCE_DIR / "galapagos.tsv"
        if not galapagos.exists():
            pytest.skip("galapagos.tsv not available")
        cooccs = asymcat.read_pa_matrix(str(galapagos))
        _assert_matches_reference(cooccs)

    def test_values_within_unit_range(self):
        """Theil's U is a normalized coefficient in [0, 1]."""
        cooccs = asymcat.collect_cooccs([[["a", "b", "c"], ["x", "y", "z"]], [["a", "b"], ["x", "y"]]])
        for u_yx, u_xy in CatScorer(cooccs).theil_u().values():
            assert 0.0 <= u_yx <= 1.0
            assert 0.0 <= u_xy <= 1.0

    def test_result_is_cached(self):
        """Repeated calls return the same memoized object."""
        cooccs = asymcat.collect_cooccs([[["a", "b"], ["x", "y"]]])
        scorer = CatScorer(cooccs)
        assert scorer.theil_u() is scorer.theil_u()


class TestCondEntropyVectorization:
    """The vectorized conditional entropy must match the per-pair reference."""

    def test_matches_reference_simple(self):
        cooccs = asymcat.collect_cooccs([[["a", "b"], ["x", "y"]], [["a", "c"], ["x", "z"]]])
        _assert_cond_entropy_matches_reference(cooccs)

    def test_matches_reference_single_pair(self):
        cooccs = asymcat.collect_cooccs([[["a"], ["x"]]])
        _assert_cond_entropy_matches_reference(cooccs)

    def test_matches_reference_degenerate_column(self):
        cooccs = asymcat.collect_cooccs([[["a", "b", "c"], ["x", "x", "x"]]])
        _assert_cond_entropy_matches_reference(cooccs)

    def test_matches_reference_toy_file(self):
        toy = RESOURCE_DIR / "toy.tsv"
        if not toy.exists():
            pytest.skip("toy.tsv not available")
        cooccs = asymcat.collect_cooccs(asymcat.read_sequences(str(toy)))
        _assert_cond_entropy_matches_reference(cooccs)

    def test_matches_reference_presence_absence(self):
        galapagos = RESOURCE_DIR / "galapagos.tsv"
        if not galapagos.exists():
            pytest.skip("galapagos.tsv not available")
        cooccs = asymcat.read_pa_matrix(str(galapagos))
        _assert_cond_entropy_matches_reference(cooccs)

    def test_values_non_negative(self):
        """Conditional entropy is always non-negative."""
        cooccs = asymcat.collect_cooccs([[["a", "b", "c"], ["x", "y", "z"]], [["a", "b"], ["x", "y"]]])
        for h_yx, h_xy in CatScorer(cooccs).cond_entropy().values():
            assert h_yx >= 0.0
            assert h_xy >= 0.0

    def test_result_is_cached(self):
        cooccs = asymcat.collect_cooccs([[["a", "b"], ["x", "y"]]])
        scorer = CatScorer(cooccs)
        assert scorer.cond_entropy() is scorer.cond_entropy()
