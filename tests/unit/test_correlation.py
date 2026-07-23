"""
Tests for the :mod:`asymcat.correlation` convenience wrappers.

These functions operate on two aligned series and return a single summary
statistic, delegating the actual computation to :mod:`asymcat.scorer`. The
tests here validate that each wrapper computes the intended measure (in
particular guarding against a regression where ``theil_u`` accidentally
returned the raw conditional entropy) and that the delegation stays in sync
with the underlying scorer functions.
"""

import math

import pytest

from asymcat import correlation, scorer


class TestConditionalEntropy:
    """Tests for the ``conditional_entropy`` wrapper."""

    def test_matches_scorer(self):
        """The wrapper should return exactly what the scorer computes."""
        series_x = ["a", "a", "b", "b", "c"]
        series_y = ["x", "x", "y", "y", "z"]

        assert correlation.conditional_entropy(series_x, series_y) == pytest.approx(
            scorer.conditional_entropy(series_x, series_y)
        )

    def test_zero_when_fully_determined(self):
        """If ``y`` fully determines ``x``, no uncertainty remains."""
        series_x = ["a", "a", "b", "b"]
        series_y = ["x", "x", "y", "y"]

        assert correlation.conditional_entropy(series_x, series_y) == pytest.approx(0.0)

    def test_non_negative(self):
        """Conditional entropy is always non-negative."""
        series_x = ["a", "b", "a", "c", "b", "c"]
        series_y = ["x", "y", "y", "x", "x", "z"]

        assert correlation.conditional_entropy(series_x, series_y) >= 0.0


class TestTheilU:
    """Tests for the ``theil_u`` wrapper."""

    def test_matches_scorer(self):
        """The wrapper must compute Theil's U, not conditional entropy.

        Regression test: an earlier implementation delegated to
        ``scorer.conditional_entropy`` by mistake, silently returning an
        unbounded entropy value under the name ``theil_u``.
        """
        series_x = ["a", "a", "b", "b", "c"]
        series_y = ["x", "x", "y", "y", "z"]

        assert correlation.theil_u(series_x, series_y) == pytest.approx(scorer.compute_theil_u(series_x, series_y))

    def test_is_normalized(self):
        """Theil's U lies within the [0, 1] range for arbitrary input."""
        series_x = ["a", "b", "a", "c", "b", "c", "a"]
        series_y = ["x", "y", "y", "x", "x", "z", "z"]

        value = correlation.theil_u(series_x, series_y)
        assert 0.0 <= value <= 1.0

    def test_perfect_prediction_is_one(self):
        """When ``y`` perfectly predicts ``x``, Theil's U equals 1.0."""
        series_x = ["a", "a", "b", "b", "c", "c"]
        series_y = ["x", "x", "y", "y", "z", "z"]

        assert correlation.theil_u(series_x, series_y) == pytest.approx(1.0)

    def test_differs_from_conditional_entropy(self):
        """Guard against the two wrappers collapsing to the same value.

        For data where conditional entropy is non-zero, Theil's U (a
        normalized quantity) should differ from the raw conditional entropy.
        """
        series_x = ["a", "b", "c", "a", "b", "c"]
        series_y = ["x", "x", "y", "y", "z", "z"]

        theil = correlation.theil_u(series_x, series_y)
        cond_ent = correlation.conditional_entropy(series_x, series_y)

        assert cond_ent > 0.0, "test data should leave residual uncertainty"
        assert not math.isclose(theil, cond_ent)


class TestCramersV:
    """Tests for the ``cramers_v`` wrapper."""

    def test_symmetric(self):
        """Cramér's V is symmetric in its two arguments."""
        series_x = ["a", "a", "b", "b", "c", "c"]
        series_y = ["x", "y", "x", "y", "x", "y"]

        forward = correlation.cramers_v(series_x, series_y)
        backward = correlation.cramers_v(series_y, series_x)

        assert forward == pytest.approx(backward)

    def test_within_unit_range(self):
        """Cramér's V is bounded to the [0, 1] range."""
        series_x = ["a", "b", "a", "c", "b", "c"]
        series_y = ["x", "y", "y", "x", "x", "z"]

        value = correlation.cramers_v(series_x, series_y)
        assert 0.0 <= value <= 1.0
