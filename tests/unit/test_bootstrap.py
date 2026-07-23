"""
Tests for the bootstrap confidence-interval scorer.

``CatScorer.bootstrap_ci`` quantifies the sampling uncertainty around an
association score by resampling the co-occurrences with replacement and taking
the percentile interval of the recomputed measure, per pair and direction.
These tests cover reproducibility, the ``((low, high), (low, high))`` shape,
ordering of the bounds, coverage of the observed point estimate, the expected
narrowing with more data / widening with a higher confidence level, and input
validation.
"""

import statistics

import pytest

import asymcat
from asymcat.scorer import CatScorer

DATA = [[["a", "b"], ["x", "y"]]] * 10 + [[["a"], ["x"]]] * 20


def _scorer(data=DATA):
    return CatScorer(asymcat.collect_cooccs(data))


def _mean_interval_width(ci):
    widths = [high - low for directions in ci.values() for (low, high) in directions]
    return statistics.mean(widths)


class TestBootstrapStructure:
    """Reproducibility, output shape, and bound ordering."""

    def test_is_reproducible_with_seed(self):
        first = _scorer().bootstrap_ci("theil_u", n_bootstrap=100, seed=42)
        second = _scorer().bootstrap_ci("theil_u", n_bootstrap=100, seed=42)
        assert first == second

    def test_output_shape(self):
        scorer = _scorer()
        ci = scorer.bootstrap_ci("mle", n_bootstrap=100, seed=0)
        assert set(ci) == set(scorer.mle())
        for directions in ci.values():
            assert len(directions) == 2  # one interval per direction
            for low, high in directions:
                assert isinstance(low, float) and isinstance(high, float)

    @pytest.mark.parametrize("measure", ["mle", "pmi", "theil_u", "mutual_information", "goodman_kruskal_lambda"])
    def test_lower_bound_not_above_upper(self, measure):
        ci = _scorer().bootstrap_ci(measure, n_bootstrap=100, seed=0)
        for directions in ci.values():
            for low, high in directions:
                assert low <= high


class TestBootstrapBehavior:
    """Statistical sanity of the intervals."""

    def test_observed_estimate_within_interval(self):
        # The point estimate should sit inside a wide (99%) bootstrap interval.
        scorer = _scorer()
        observed = scorer.theil_u()
        ci = scorer.bootstrap_ci("theil_u", n_bootstrap=300, confidence_level=0.99, seed=1)
        for pair, (obs_xy, obs_yx) in observed.items():
            (xy_low, xy_high), (yx_low, yx_high) = ci[pair]
            assert xy_low <= obs_xy <= xy_high
            assert yx_low <= obs_yx <= yx_high

    def test_interval_narrows_with_more_data(self):
        small = CatScorer(asymcat.collect_cooccs([[["a", "b"], ["x", "y"]]] * 3))
        large = CatScorer(asymcat.collect_cooccs([[["a", "b"], ["x", "y"]]] * 300))
        width_small = _mean_interval_width(small.bootstrap_ci("mle", n_bootstrap=150, seed=2))
        width_large = _mean_interval_width(large.bootstrap_ci("mle", n_bootstrap=150, seed=2))
        assert width_large < width_small

    def test_higher_confidence_is_wider(self):
        scorer = _scorer()
        width_90 = _mean_interval_width(scorer.bootstrap_ci("mle", n_bootstrap=200, confidence_level=0.90, seed=5))
        width_99 = _mean_interval_width(scorer.bootstrap_ci("mle", n_bootstrap=200, confidence_level=0.99, seed=5))
        assert width_99 >= width_90


class TestBootstrapValidation:
    """Input validation."""

    def test_rejects_unknown_measure(self):
        with pytest.raises(ValueError, match="Unsupported measure"):
            _scorer().bootstrap_ci("not_a_measure")

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_confidence_level_out_of_range(self, level):
        with pytest.raises(ValueError, match="confidence_level must be in the open interval"):
            _scorer().bootstrap_ci("mle", confidence_level=level)

    def test_rejects_non_positive_bootstrap(self):
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            _scorer().bootstrap_ci("mle", n_bootstrap=0)
