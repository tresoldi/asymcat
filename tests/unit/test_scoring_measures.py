"""
Tests for individual scoring measures.

Demonstrates usage of each scoring method and validates their mathematical properties.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest

import asymcat
from asymcat.scorer import CatScorer

from ..fixtures.assertions import (
    assert_expected_score_values,
    assert_information_theoretic_scores,
    assert_probabilistic_scores,
    assert_scores_asymmetric,
    assert_scores_in_range,
    assert_scores_symmetric,
    assert_valid_scores,
)
from ..fixtures.data import (
    ASYMMETRIC_DATA,
    EXPECTED_SCORING_RESULTS,
    INDEPENDENT_DATA,
    PERFECT_CORRELATION_DATA,
    get_sample_cmu_processed,
)


class TestScorerInitialization:
    """Test scorer initialization and basic properties."""

    def test_scorer_creation(self):
        """
        Test creating a CatScorer instance.

        Example usage:
            cooccs = [('a', 'x'), ('b', 'y')]
            scorer = CatScorer(cooccs)
        """
        # Create simple test data
        test_data = [[["a", "b"], ["x", "y"]]]
        cooccs = asymcat.collect_cooccs(test_data)

        # Create scorer
        scorer = CatScorer(cooccs)

        # Validate basic properties
        assert hasattr(scorer, 'cooccs'), "Scorer should have cooccs attribute"
        assert hasattr(scorer, 'obs'), "Scorer should have observations attribute"
        assert hasattr(scorer, 'alphabet_x'), "Scorer should have X alphabet"
        assert hasattr(scorer, 'alphabet_y'), "Scorer should have Y alphabet"

        # Check alphabets
        assert len(scorer.alphabet_x) > 0, "X alphabet should not be empty"
        assert len(scorer.alphabet_y) > 0, "Y alphabet should not be empty"

    def test_scorer_with_sample_data(self):
        """Test scorer with standard sample data."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)

        # Check expected properties from original tests
        assert len(cooccs) == 879, "Should have 879 co-occurrences"
        assert len(scorer.alphabet_x) == 23, "Should have 23 X symbols"
        assert len(scorer.alphabet_y) == 28, "Should have 28 Y symbols"


class TestProbabilisticMeasures:
    """Test measures that produce probability-like values [0,1]."""

    @pytest.fixture
    def sample_scorer(self):
        """Create a scorer with sample data for testing."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        return CatScorer(cooccs)

    def test_maximum_likelihood_estimation(self, sample_scorer):
        """
        Test Maximum Likelihood Estimation scores.

        Example usage:
            mle_scores = scorer.mle()
            # Returns {(x, y): (P(y|x), P(x|y))}
        """
        scores = sample_scorer.mle()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_probabilistic_scores(scores)

        # Check specific expected values
        if ("ONE", "1") in scores:
            xy, yx = scores[("ONE", "1")]
            assert abs(xy - 1.0) < 1e-10, "Perfect correspondence should give probability 1.0"
            assert abs(yx - 1.0) < 1e-10, "Perfect correspondence should give probability 1.0"

    def test_goodman_kruskal_lambda(self, sample_scorer):
        """
        Test Goodman-Kruskal Lambda (proportional reduction in error).

        Example usage:
            lambda_scores = scorer.goodman_kruskal_lambda()
            # Returns {(x, y): (λ(y|x), λ(x|y))}
        """
        scores = sample_scorer.goodman_kruskal_lambda()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_probabilistic_scores(scores)  # Lambda is in [0,1]

    def test_jaccard_index(self, sample_scorer):
        """
        Test Jaccard Index (set similarity).

        Example usage:
            jaccard_scores = scorer.jaccard_index()
            # Returns {(x, y): (J(x,y), J(y,x))} - typically symmetric
        """
        scores = sample_scorer.jaccard_index()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_probabilistic_scores(scores)  # Jaccard is in [0,1]

        # Jaccard should be symmetric
        assert_scores_symmetric(scores, tolerance=1e-10)


class TestInformationTheoreticMeasures:
    """Test information-theoretic measures."""

    @pytest.fixture
    def sample_scorer(self):
        """Create a scorer with sample data for testing."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        return CatScorer(cooccs)

    def test_mutual_information(self, sample_scorer):
        """
        Test Mutual Information.

        Example usage:
            mi_scores = scorer.mutual_information()
            # Returns {(x, y): (MI(x,y), MI(y,x))} - symmetric
        """
        scores = sample_scorer.mutual_information()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_information_theoretic_scores(scores)  # MI >= 0

        # MI should be symmetric
        assert_scores_symmetric(scores, tolerance=1e-10)

    def test_normalized_mutual_information(self, sample_scorer):
        """
        Test Normalized Mutual Information.

        Example usage:
            nmi_scores = scorer.normalized_mutual_information()
            # Returns {(x, y): (NMI(x,y), NMI(y,x))} in [0,1]
        """
        scores = sample_scorer.normalized_mutual_information()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_probabilistic_scores(scores)  # NMI in [0,1]

        # NMI should be symmetric
        assert_scores_symmetric(scores, tolerance=1e-10)

    def test_pointwise_mutual_information(self, sample_scorer):
        """
        Test Pointwise Mutual Information.

        Example usage:
            pmi_scores = scorer.pmi()           # PMI
            npmi_scores = scorer.pmi(True)      # Normalized PMI
        """
        # Test regular PMI
        pmi_scores = sample_scorer.pmi()
        assert_valid_scores(pmi_scores)
        # PMI can be negative, no range restrictions

        # Test normalized PMI
        npmi_scores = sample_scorer.pmi(True)
        assert_valid_scores(npmi_scores)
        # NPMI should be in [-1, 1] range
        for pair, (xy, yx) in npmi_scores.items():
            assert -1.0 <= xy <= 1.0, f"NPMI X→Y for {pair} out of range: {xy}"
            assert -1.0 <= yx <= 1.0, f"NPMI Y→X for {pair} out of range: {yx}"

    def test_conditional_entropy(self, sample_scorer):
        """
        Test Conditional Entropy.

        Example usage:
            entropy_scores = scorer.cond_entropy()
            # Returns {(x, y): (H(y|x), H(x|y))}
        """
        scores = sample_scorer.cond_entropy()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_information_theoretic_scores(scores)  # Entropy >= 0

        # Conditional entropy is typically asymmetric
        assert_scores_asymmetric(scores)

    def test_theil_u(self, sample_scorer):
        """
        Test Theil's U (uncertainty coefficient).

        Example usage:
            theil_scores = scorer.theil_u()
            # Returns {(x, y): (U(y|x), U(x|y))} in [0,1]
        """
        scores = sample_scorer.theil_u()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_probabilistic_scores(scores)  # Theil's U in [0,1]

        # Theil's U is typically asymmetric
        assert_scores_asymmetric(scores)


class TestStatisticalMeasures:
    """Test statistical association measures."""

    @pytest.fixture
    def sample_scorer(self):
        """Create a scorer with sample data for testing."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        return CatScorer(cooccs)

    @pytest.mark.parametrize("square", [True, False])
    def test_chi_square(self, sample_scorer, square: bool):
        """
        Test Chi-square test statistics.

        Example usage:
            chi2_scores = scorer.chi2(square=True)   # 2x2 contingency table
            chi2_ns_scores = scorer.chi2(square=False) # 3x2 contingency table
        """
        scores = sample_scorer.chi2(square)

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_information_theoretic_scores(scores)  # Chi-square >= 0

        # Chi-square should be symmetric
        assert_scores_symmetric(scores, tolerance=1e-10)

    @pytest.mark.parametrize("square", [True, False])
    def test_cramers_v(self, sample_scorer, square: bool):
        """
        Test Cramér's V correlation coefficient.

        Example usage:
            cv_scores = scorer.cramers_v(square=True)    # Based on 2x2 tables
            cv_ns_scores = scorer.cramers_v(square=False) # Based on 3x2 tables
        """
        scores = sample_scorer.cramers_v(square)

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_probabilistic_scores(scores)  # Cramér's V in [0,1]

        # Cramér's V should be symmetric
        assert_scores_symmetric(scores, tolerance=1e-10)

    def test_fisher_exact(self, sample_scorer):
        """
        Test Fisher's Exact Test odds ratios.

        Example usage:
            fisher_scores = scorer.fisher()
            # Returns {(x, y): (odds_ratio_xy, odds_ratio_yx)}
        """
        scores = sample_scorer.fisher()

        # Validate structure (Fisher can produce infinite values for perfect associations)
        assert_valid_scores(scores, allow_infinite=True)
        # Fisher's exact can produce infinite values, so no range check

        # Fisher should be symmetric for odds ratios
        assert_scores_symmetric(scores, tolerance=1e-10)

    def test_log_likelihood_ratio(self, sample_scorer):
        """
        Test Log-Likelihood Ratio (G²).

        Example usage:
            g2_scores = scorer.log_likelihood_ratio(square_ct=True)
        """
        scores = sample_scorer.log_likelihood_ratio()

        # Validate structure and properties
        assert_valid_scores(scores)
        assert_information_theoretic_scores(scores)  # G² >= 0

        # G² should be symmetric
        assert_scores_symmetric(scores, tolerance=1e-10)


class TestSpecializedMeasures:
    """Test specialized and custom measures."""

    @pytest.fixture
    def sample_scorer(self):
        """Create a scorer with sample data for testing."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        return CatScorer(cooccs)

    def test_tresoldi_measure(self, sample_scorer):
        """
        Test Tresoldi measure (custom combination of MLE and PMI).

        Example usage:
            tresoldi_scores = scorer.tresoldi()
            # Returns custom scoring optimized for historical linguistics
        """
        scores = sample_scorer.tresoldi()

        # Validate structure
        assert_valid_scores(scores)
        # Tresoldi can have negative values (incorporates PMI)

        # Should typically show asymmetric patterns
        assert_scores_asymmetric(scores)


class TestMeasureProperties:
    """Test mathematical properties of measures with controlled data."""

    def test_perfect_correlation_measures(self):
        """Test measures with perfectly correlated data."""
        cooccs = asymcat.collect_cooccs(PERFECT_CORRELATION_DATA)
        scorer = CatScorer(cooccs)

        # Test MLE - should show perfect prediction for correlated pairs
        mle = scorer.mle()
        # Only check pairs that are actually perfectly correlated
        perfect_pairs = [('A', 'B'), ('C', 'D')]
        for pair in perfect_pairs:
            if pair in mle:
                xy, yx = mle[pair]
                assert max(xy, yx) >= 0.9, f"Perfect correlation should show high MLE for {pair}"

    def test_independent_data_measures(self):
        """Test measures with independent data."""
        cooccs = asymcat.collect_cooccs(INDEPENDENT_DATA)
        scorer = CatScorer(cooccs)

        # Test mutual information - should be low for independent data
        mi = scorer.mutual_information()
        for pair, (xy, yx) in mi.items():
            # MI should be relatively low for independent variables
            assert xy < 2.0, f"MI should be low for independent data: {pair}"
            assert yx < 2.0, f"MI should be low for independent data: {pair}"

    def test_asymmetric_data_measures(self):
        """Test measures with asymmetric relationships."""
        cooccs = asymcat.collect_cooccs(ASYMMETRIC_DATA)
        scorer = CatScorer(cooccs)

        # Test Theil's U - should show clear asymmetry
        theil = scorer.theil_u()

        # Should have some asymmetric relationships
        assert_scores_asymmetric(theil)


class TestMeasureValidation:
    """Test measures against known expected values."""

    def test_expected_values_validation(self):
        """Test that measures produce expected values for known data."""
        data = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data)
        scorer = CatScorer(cooccs)

        # Test specific pairs with known expected values
        for pair, expected_scores in EXPECTED_SCORING_RESULTS.items():
            if pair not in scorer.obs:
                continue  # Skip if pair not in this dataset

            for measure_name, expected_values in expected_scores.items():
                if hasattr(scorer, measure_name):
                    method = getattr(scorer, measure_name)
                    actual_scores = method()

                    if pair in actual_scores:
                        actual_values = actual_scores[pair]
                        assert_expected_score_values(
                            {pair: actual_values},
                            {pair: expected_values},
                            tolerance=1e-3,  # Allow some numerical tolerance
                        )
