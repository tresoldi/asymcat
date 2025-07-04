"""
Tests for score transformation and manipulation functions.

Demonstrates how to scale, normalize, and manipulate scoring results.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest

import asymcat
from asymcat.scorer import CatScorer, invert_scorer, scale_scorer, scorer2matrices

from ..fixtures.assertions import (
    assert_matrices_consistent,
    assert_scores_in_range,
    assert_valid_scores,
)
from ..fixtures.data import get_sample_cmu_processed


class TestScoreScaling:
    """Test score scaling and normalization methods."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for testing transformations."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)
        return scorer.tresoldi()  # Use Tresoldi scores (good range for testing)

    @pytest.mark.parametrize(
        "method,expected_range",
        [
            ("minmax", (0.0, 1.0)),
            ("mean", (None, None)),  # Mean scaling doesn't guarantee range
            ("stdev", (None, None)),  # Standardization doesn't guarantee range
        ],
    )
    def test_scale_scorer_methods(self, sample_scores, method: str, expected_range):
        """
        Test different score scaling methods.

        Example usage:
            scaled = scale_scorer(scores, method="minmax")  # Scale to [0,1]
            scaled = scale_scorer(scores, method="mean")    # Mean centering
            scaled = scale_scorer(scores, method="stdev")   # Standardization
        """
        scaled_scores = scale_scorer(sample_scores, method=method)

        # Validate structure
        assert_valid_scores(scaled_scores)
        assert len(scaled_scores) == len(sample_scores), "Should preserve number of pairs"

        # Check that all original pairs are present
        assert set(scaled_scores.keys()) == set(sample_scores.keys()), "Should preserve all symbol pairs"

        # Check range if specified
        if expected_range[0] is not None and expected_range[1] is not None:
            assert_scores_in_range(scaled_scores, expected_range[0], expected_range[1])

        # Verify scaling properties
        if method == "minmax":
            # Min-max scaling should have minimum 0 and maximum 1
            all_values = [val for pair_scores in scaled_scores.values() for val in pair_scores]
            assert min(all_values) == pytest.approx(0.0, abs=1e-10), "Min-max scaling should have minimum 0"
            assert max(all_values) == pytest.approx(1.0, abs=1e-10), "Min-max scaling should have maximum 1"

        elif method == "mean":
            # Mean centering should have mean approximately 0
            all_values = [val for pair_scores in scaled_scores.values() for val in pair_scores]
            assert abs(np.mean(all_values)) < 1e-10, "Mean centering should result in mean ≈ 0"

        elif method == "stdev":
            # Standardization should have mean ≈ 0 and std ≈ 1
            all_values = [val for pair_scores in scaled_scores.values() for val in pair_scores]
            assert abs(np.mean(all_values)) < 1e-10, "Standardization should result in mean ≈ 0"
            assert abs(np.std(all_values) - 1.0) < 1e-10, "Standardization should result in std ≈ 1"

    def test_scale_scorer_preservation(self, sample_scores):
        """Test that scaling preserves relative ordering."""
        scaled_scores = scale_scorer(sample_scores, method="minmax")

        # Get a few pairs to test ordering
        pairs = list(sample_scores.keys())[:5]

        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                pair_i, pair_j = pairs[i], pairs[j]

                # Check X→Y ordering preservation
                orig_xy_i, orig_xy_j = sample_scores[pair_i][0], sample_scores[pair_j][0]
                scaled_xy_i, scaled_xy_j = scaled_scores[pair_i][0], scaled_scores[pair_j][0]

                if orig_xy_i < orig_xy_j:
                    assert scaled_xy_i <= scaled_xy_j, f"Scaling should preserve ordering: {pair_i} vs {pair_j}"
                elif orig_xy_i > orig_xy_j:
                    assert scaled_xy_i >= scaled_xy_j, f"Scaling should preserve ordering: {pair_i} vs {pair_j}"

    def test_scale_scorer_error_handling(self, sample_scores):
        """Test error handling for score scaling."""
        # Invalid method
        with pytest.raises(ValueError):
            scale_scorer(sample_scores, method="invalid")

        # Empty scores
        with pytest.raises((ValueError, ZeroDivisionError)):
            scale_scorer({}, method="minmax")


class TestScoreInversion:
    """Test score inversion operations."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for testing inversion."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)
        # Use scaled scores for predictable inversion
        scores = scorer.mle()
        return scale_scorer(scores, method="minmax")

    def test_invert_scorer(self, sample_scores):
        """
        Test score inversion (1 - score).

        Example usage:
            inverted = invert_scorer(scores)  # Inverts all scores
        """
        inverted_scores = invert_scorer(sample_scores)

        # Validate structure
        assert_valid_scores(inverted_scores)
        assert len(inverted_scores) == len(sample_scores), "Should preserve number of pairs"
        assert set(inverted_scores.keys()) == set(sample_scores.keys()), "Should preserve all symbol pairs"

        # Check inversion property: inverted + original = 1
        for pair in sample_scores:
            orig_xy, orig_yx = sample_scores[pair]
            inv_xy, inv_yx = inverted_scores[pair]

            assert abs((orig_xy + inv_xy) - 1.0) < 1e-10, f"Inversion should satisfy: orig + inv = 1 for {pair} X→Y"
            assert abs((orig_yx + inv_yx) - 1.0) < 1e-10, f"Inversion should satisfy: orig + inv = 1 for {pair} Y→X"

    def test_double_inversion(self, sample_scores):
        """Test that double inversion returns to original values."""
        inverted_once = invert_scorer(sample_scores)
        inverted_twice = invert_scorer(inverted_once)

        # Double inversion should return to original values
        for pair in sample_scores:
            orig_xy, orig_yx = sample_scores[pair]
            double_inv_xy, double_inv_yx = inverted_twice[pair]

            assert abs(orig_xy - double_inv_xy) < 1e-10, f"Double inversion should restore original for {pair} X→Y"
            assert abs(orig_yx - double_inv_yx) < 1e-10, f"Double inversion should restore original for {pair} Y→X"


class TestMatrixConversion:
    """Test conversion of scores to matrix format."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for matrix conversion."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)
        return scorer.mle()

    def test_scorer2matrices(self, sample_scores):
        """
        Test conversion of scores to matrix format.

        Example usage:
            xy_matrix, yx_matrix, x_labels, y_labels = scorer2matrices(scores)
            # Use matrices with matplotlib, seaborn, etc.
        """
        xy_matrix, yx_matrix, x_labels, y_labels = scorer2matrices(sample_scores)

        # Validate matrix properties
        assert_matrices_consistent(xy_matrix, yx_matrix, x_labels, y_labels)

        # Check matrix content
        assert isinstance(xy_matrix, np.ndarray), "XY matrix should be numpy array"
        assert isinstance(yx_matrix, np.ndarray), "YX matrix should be numpy array"
        assert len(x_labels) > 0, "Should have X labels"
        assert len(y_labels) > 0, "Should have Y labels"

        # Check that matrices contain finite values
        assert np.all(np.isfinite(xy_matrix)), "XY matrix should contain finite values"
        assert np.all(np.isfinite(yx_matrix)), "YX matrix should contain finite values"

        # Verify matrix-score correspondence
        for i, x_label in enumerate(x_labels):
            for j, y_label in enumerate(y_labels):
                pair = (x_label, y_label)
                if pair in sample_scores:
                    score_xy, score_yx = sample_scores[pair]

                    # Check that matrix values match score values
                    assert abs(xy_matrix[i, j] - score_xy) < 1e-10, f"XY matrix value should match score for {pair}"
                    assert abs(yx_matrix[j, i] - score_yx) < 1e-10, f"YX matrix value should match score for {pair}"

    def test_matrices_for_visualization(self, sample_scores):
        """Test that matrices are suitable for visualization."""
        xy_matrix, yx_matrix, x_labels, y_labels = scorer2matrices(sample_scores)

        # Check dimensions are reasonable for visualization
        assert xy_matrix.shape[0] <= 100, "X dimension should be reasonable for visualization"
        assert xy_matrix.shape[1] <= 100, "Y dimension should be reasonable for visualization"

        # Labels should be strings
        assert all(isinstance(label, str) for label in x_labels), "X labels should be strings"
        assert all(isinstance(label, str) for label in y_labels), "Y labels should be strings"

        # Labels should be unique
        assert len(set(x_labels)) == len(x_labels), "X labels should be unique"
        assert len(set(y_labels)) == len(y_labels), "Y labels should be unique"


class TestTransformationChaining:
    """Test chaining multiple transformations."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for transformation chaining."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)
        return scorer.tresoldi()

    def test_scale_then_invert(self, sample_scores):
        """
        Test chaining scaling followed by inversion.

        Example usage:
            scaled = scale_scorer(scores, method="minmax")
            scaled_inverted = invert_scorer(scaled)
        """
        # Scale to [0,1] then invert
        scaled = scale_scorer(sample_scores, method="minmax")
        scaled_inverted = invert_scorer(scaled)

        # Validate final result
        assert_valid_scores(scaled_inverted)
        assert_scores_in_range(scaled_inverted, 0.0, 1.0)

        # Check that high original scores become low inverted scores
        original_high_pairs = [
            pair
            for pair, (xy, yx) in sample_scores.items()
            if max(xy, yx) > np.percentile([max(s) for s in sample_scores.values()], 75)
        ]

        for pair in original_high_pairs[:3]:  # Check a few high-scoring pairs
            inv_xy, inv_yx = scaled_inverted[pair]
            # Inverted scores should be relatively low for originally high scores
            assert min(inv_xy, inv_yx) < 0.5, f"High original scores should become low inverted scores for {pair}"

    def test_standardize_for_comparison(self, sample_scores):
        """Test standardization for comparing different measures."""
        # Get another measure for comparison
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)
        other_scores = scorer.mle()

        # Standardize both measures
        std_tresoldi = scale_scorer(sample_scores, method="stdev")
        std_mle = scale_scorer(other_scores, method="stdev")

        # Both should have similar statistical properties after standardization
        tresoldi_values = [val for scores in std_tresoldi.values() for val in scores]
        mle_values = [val for scores in std_mle.values() for val in scores]

        # Both should have mean ≈ 0 and std ≈ 1
        assert abs(np.mean(tresoldi_values)) < 1e-10, "Standardized Tresoldi should have mean ≈ 0"
        assert abs(np.std(tresoldi_values) - 1.0) < 1e-10, "Standardized Tresoldi should have std ≈ 1"
        assert abs(np.mean(mle_values)) < 1e-10, "Standardized MLE should have mean ≈ 0"
        assert abs(np.std(mle_values) - 1.0) < 1e-10, "Standardized MLE should have std ≈ 1"


class TestTransformationEdgeCases:
    """Test edge cases and error conditions in transformations."""

    def test_uniform_scores_scaling(self):
        """Test scaling with uniform (constant) scores."""
        # Create uniform scores (all the same value)
        uniform_scores = {
            ('a', 'x'): (0.5, 0.5),
            ('b', 'y'): (0.5, 0.5),
            ('c', 'z'): (0.5, 0.5),
        }

        # Min-max scaling of uniform data should handle gracefully
        try:
            scaled = scale_scorer(uniform_scores, method="minmax")
            # Should either work (returning zeros) or raise appropriate error
            assert_valid_scores(scaled)
        except (ValueError, ZeroDivisionError):
            # This is acceptable for uniform data
            pass

    def test_extreme_values_handling(self):
        """Test handling of extreme score values."""
        extreme_scores = {
            ('a', 'x'): (0.0, 1.0),  # Extreme values
            ('b', 'y'): (float('inf'), 0.0),  # Infinite value
            ('c', 'z'): (0.5, 0.5),  # Normal values
        }

        # Should handle infinite values gracefully
        try:
            scaled = scale_scorer(extreme_scores, method="minmax")
            # If it works, should produce finite results
            for pair, (xy, yx) in scaled.items():
                if np.isfinite(extreme_scores[pair][0]) and np.isfinite(extreme_scores[pair][1]):
                    assert np.isfinite(xy), f"Should produce finite result for {pair} X→Y"
                    assert np.isfinite(yx), f"Should produce finite result for {pair} Y→X"
        except (ValueError, OverflowError):
            # This is acceptable for extreme values
            pass
