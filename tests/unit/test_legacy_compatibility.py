"""
Tests for legacy compatibility and validation against known values.

This ensures that the modernized test suite maintains compatibility
with the original test results and expected behaviors.
"""

from typing import Dict, Tuple

import numpy as np
import pytest

import asymcat

from ..fixtures.assertions import (
    assert_expected_score_values,
    assert_valid_cooccurrences,
    assert_valid_scores,
)
from ..fixtures.data import RESOURCE_DIR, get_sample_cmu_processed


class TestLegacyDataCompatibility:
    """Test that modernized code produces same results as legacy tests."""

    def test_original_cmu_sample_data(self):
        """
        Test with the exact CMU sample data from original tests.

        This ensures we get identical results to the original implementation.
        """
        # Use the exact same data as the original test
        sample_cmu = [
            ["ONE", "1 1 1"],
            ["A B D O L L A H", "æ b d ɑ l ʌ"],
            ["A M A S S", "ʌ m æ s"],
            ["A N G L O P H I L E", "æ n ɡ l ʌ f aɪ l"],
            ["A N T I C", "æ n t ɪ k"],
            ["A R J O", "ɑ ɹ j oʊ"],
            ["A S T R A D D L E", "ʌ s t ɹ æ d ʌ l"],
            ["B A H L S", "b ɑ l z"],
            ["B L O W E D", "b l oʊ d"],
            ["B O N V I L L A I N", "b ɑ n v ɪ l eɪ n"],
            ["B R A G A", "b ɹ ɑ ɡ ʌ"],
            ["B U R D I", "b ʊ ɹ d i"],
            ["B U R K E R T", "b ɝ k ɝ t"],
            ["B U R R E S S", "b ɝ ʌ s"],
            ["C A E T A N O", "k ʌ t ɑ n oʊ"],
            ["C H E R Y L", "ʃ ɛ ɹ ʌ l"],
            ["C L E M E N C E", "k l ɛ m ʌ n s"],
            ["C O L V I N", "k oʊ l v ɪ n"],
            ["C O N V E N T I O N S", "k ʌ n v ɛ n ʃ ʌ n z"],
            ["C R E A S Y", "k ɹ i s i"],
            ["C R E T I E N", "k ɹ i ʃ j ʌ n"],
            ["C R O C E", "k ɹ oʊ tʃ i"],
        ]

        # Process data the same way
        data_cmu = [[entry[0].split(), entry[1].split()] for entry in sample_cmu]

        # Test co-occurrence collection
        cooccs = asymcat.collect_cooccs(data_cmu)
        assert_valid_cooccurrences(cooccs)

        # Verify exact counts from original test
        assert len(cooccs) == 879, f"Expected 879 co-occurrences, got {len(cooccs)}"

        cooccs_A = [coocc for coocc in cooccs if coocc[0] == "A"]
        cooccs_Ll = [coocc for coocc in cooccs if coocc[0] == "L" and coocc[1] == "l"]

        assert len(cooccs_A) == 92, f"Expected 92 'A' co-occurrences, got {len(cooccs_A)}"
        assert len(cooccs_Ll) == 14, f"Expected 14 'L,l' co-occurrences, got {len(cooccs_Ll)}"

    def test_original_scorer_validation(self):
        """
        Test that all scorers produce the exact values from the original test.

        This is the most critical compatibility test - validates all measures
        against the precise expected values from the legacy test suite.
        """
        # Use the exact same test data
        data_cmu = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data_cmu)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Verify scorer properties match original test
        assert len(cooccs) == 879, "Co-occurrence count mismatch"
        assert len(scorer.alphabet_x) == 23, f"X alphabet size mismatch: {len(scorer.alphabet_x)}"
        assert len(scorer.alphabet_y) == 28, f"Y alphabet size mismatch: {len(scorer.alphabet_y)}"

        # Get all scores using original method calls
        mle = scorer.mle()
        pmi = scorer.pmi()
        npmi = scorer.pmi(True)
        chi2 = scorer.chi2()
        chi2_ns = scorer.chi2(False)
        cramersv = scorer.cramers_v()
        cramersv_ns = scorer.cramers_v(False)
        fisher = scorer.fisher()
        theil_u = scorer.theil_u()
        cond_entropy = scorer.cond_entropy()
        tresoldi = scorer.tresoldi()

        # Validate structure for all measures
        measures = {
            'mle': mle,
            'pmi': pmi,
            'npmi': npmi,
            'chi2': chi2,
            'chi2_ns': chi2_ns,
            'cramersv': cramersv,
            'cramersv_ns': cramersv_ns,
            'fisher': fisher,
            'theil_u': theil_u,
            'cond_entropy': cond_entropy,
            'tresoldi': tresoldi,
        }

        for measure_name, scores in measures.items():
            # Fisher exact test can produce infinite values for perfect associations
            allow_inf = (measure_name == 'fisher')
            assert_valid_scores(scores, allow_infinite=allow_inf)

        # Test exact values against original test expectations
        expected_values = {
            ("ONE", "1"): {
                'mle': (1.0, 1.0),
                'pmi': (5.680172609017068, 5.680172609017068),
                'npmi': (1.0, 1.0),
                'chi2': (609.5807658175393, 609.5807658175393),
                'chi2_ns': (879.0, 879.0),
                'cramersv': (0.8325526903114843, 0.8325526903114843),
                'cramersv_ns': (0.7065025023855139, 0.7065025023855139),
                'fisher': (np.inf, np.inf),
                'theil_u': (1.0, 1.0),
                'cond_entropy': (0.0, 0.0),
                'tresoldi': (1.0, 1.0),
            },
            ("A", "b"): {
                'mle': (0.11320754716981132, 0.06521739130434782),
                'pmi': (0.07846387631207004, 0.07846387631207004),
                'npmi': (0.015733602612959818, 0.015733602612959818),
                'chi2': (0.0, 0.0),
                'chi2_ns': (0.043927505580845905, 0.043927505580845905),
                'cramersv': (0.0, 0.0),
                'cramersv_ns': (0.0, 0.0),
                'fisher': (1.0984661058881742, 1.0984661058881742),
                'theil_u': (0.21299752425693524, 0.3356184612000498),
                'cond_entropy': (1.86638224482290279, 0.9999327965500219),
                'tresoldi': (0.10466500171366895, 0.0926310345228265),
            },
            ("S", "s"): {
                'mle': (0.17142857142857143, 0.13953488372093023),
                'pmi': (1.2539961897302558, 1.2539961897302558),
                'npmi': (0.2514517336476095, 0.2514517336476095),
                'chi2': (9.176175043924879, 9.176175043924879),
                'chi2_ns': (11.758608367318205, 11.758608367318205),
                'cramersv': (0.09649345638896019, 0.09649345638896019),
                'cramersv_ns': (0.07452170854897658, 0.07452170854897658),
                'fisher': (4.512581547064306, 4.512581547064306),
                'theil_u': (0.22071631715993364, 0.2841291022637977),
                'cond_entropy': (1.5938047875022765, 1.137346966185816),
                'tresoldi': (1.2062725270739942, 1.2150117159149825),
            },
            ("H", "i"): {
                'mle': (0.0, 0.0),
                'pmi': (-6.502790045915624, -6.502790045915624),
                'npmi': (-0.4796427489634758, -0.4796427489634758),
                'chi2': (0.09374177071030182, 0.09374177071030182),
                'chi2_ns': (0.8057902693787795, 0.8057902693787795),
                'cramersv': (0.0, 0.0),
                'cramersv_ns': (0.0, 0.0),
                'fisher': (0.0, 0.0),
                'theil_u': (0.3866999152200347, 0.34435838354803283),
                'cond_entropy': (1.0887395664391526, 1.3070160180503212),
                'tresoldi': (-6.502790045915624, -6.502790045915624),
            },
        }

        # Validate all expected values with high precision
        for pair, pair_expectations in expected_values.items():
            for measure_name, expected_scores in pair_expectations.items():
                actual_scores = measures[measure_name]

                if pair in actual_scores:
                    actual_xy, actual_yx = actual_scores[pair]
                    expected_xy, expected_yx = expected_scores

                    # Handle infinite values specially
                    if np.isinf(expected_xy):
                        assert np.isinf(actual_xy), f"{measure_name}[{pair}] X→Y: expected inf, got {actual_xy}"
                    else:
                        assert np.allclose(
                            [actual_xy], [expected_xy], rtol=1e-05, atol=1e-08
                        ), f"{measure_name}[{pair}] X→Y: expected {expected_xy}, got {actual_xy}"

                    if np.isinf(expected_yx):
                        assert np.isinf(actual_yx), f"{measure_name}[{pair}] Y→X: expected inf, got {actual_yx}"
                    else:
                        assert np.allclose(
                            [actual_yx], [expected_yx], rtol=1e-05, atol=1e-08
                        ), f"{measure_name}[{pair}] Y→X: expected {expected_yx}, got {actual_yx}"

    def test_original_matrix_generation(self):
        """Test matrix generation produces original expected dimensions."""
        data_cmu = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data_cmu)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Generate matrices using tresoldi scores (as in original)
        score_dict = scorer.tresoldi()
        xy, yx, alpha_x, alpha_y = asymcat.scorer.scorer2matrices(score_dict)

        # Verify exact dimensions from original test
        assert len(xy) == 28, f"XY matrix size mismatch: expected 28, got {len(xy)}"
        assert len(yx) == 23, f"YX matrix size mismatch: expected 23, got {len(yx)}"
        assert len(alpha_x) == 23, f"X alphabet size mismatch: expected 23, got {len(alpha_x)}"
        assert len(alpha_y) == 28, f"Y alphabet size mismatch: expected 28, got {len(alpha_y)}"

        # Verify alphabet content as in original
        assert "A" in alpha_x and "s" not in alpha_x, "X alphabet content mismatch"
        assert "s" in alpha_y and "A" not in alpha_y, "Y alphabet content mismatch"

    def test_original_scaling_operations(self):
        """Test that scaling operations produce original expected values."""
        data_cmu = get_sample_cmu_processed()
        cooccs = asymcat.collect_cooccs(data_cmu)
        scorer = asymcat.scorer.CatScorer(cooccs)
        score_dict = scorer.tresoldi()

        # Test minmax scaling with original expected values
        scaled_minmax = asymcat.scorer.scale_scorer(score_dict, method="minmax")
        expected_minmax = (0.15476857281060225, 0.15476857281060225)
        actual_minmax = scaled_minmax["H", "i"]

        assert np.allclose(
            actual_minmax, expected_minmax, rtol=1e-05, atol=1e-08
        ), f"Minmax scaling mismatch: expected {expected_minmax}, got {actual_minmax}"

        # Test mean scaling
        scaled_mean = asymcat.scorer.scale_scorer(score_dict, method="mean")
        expected_mean = (-0.36871270234063913, -0.36871270234063913)
        actual_mean = scaled_mean["H", "i"]

        assert np.allclose(
            actual_mean, expected_mean, rtol=1e-05, atol=1e-08
        ), f"Mean scaling mismatch: expected {expected_mean}, got {actual_mean}"

        # Test standard deviation scaling
        scaled_stdev = asymcat.scorer.scale_scorer(score_dict, method="stdev")
        expected_stdev = (-1.3465717087048406, -1.3465717087048406)
        actual_stdev = scaled_stdev["H", "i"]

        assert np.allclose(
            actual_stdev, expected_stdev, rtol=1e-05, atol=1e-08
        ), f"Stdev scaling mismatch: expected {expected_stdev}, got {actual_stdev}"

        # Test inversion
        inverted = asymcat.scorer.invert_scorer(scaled_minmax)
        expected_inverted = (0.8452314271893977, 0.8452314271893977)
        actual_inverted = inverted["H", "i"]

        assert np.allclose(
            actual_inverted, expected_inverted, rtol=1e-05, atol=1e-08
        ), f"Inversion mismatch: expected {expected_inverted}, got {actual_inverted}"


class TestLegacyFileReading:
    """Test file reading functions produce original expected results."""

    def test_original_sequence_reading_results(self):
        """Test sequence reading produces original expected counts."""
        # Test CMU dictionary reading
        cmu_file = RESOURCE_DIR / "cmudict.tsv"
        if cmu_file.exists():
            cmu = asymcat.read_sequences(str(cmu_file))
            assert len(cmu) == 134373, f"CMU dict size mismatch: expected 134373, got {len(cmu)}"

    def test_original_pa_matrix_reading_results(self):
        """Test presence-absence matrix reading produces original expected counts."""
        # Test Galapagos finches reading
        finches_file = RESOURCE_DIR / "galapagos.tsv"
        if finches_file.exists():
            finches = asymcat.read_pa_matrix(str(finches_file))
            assert len(finches) == 447, f"Finches count mismatch: expected 447, got {len(finches)}"


class TestLegacyUtilityFunctions:
    """Test utility functions produce original expected results."""

    def test_original_ngram_collection(self):
        """Test n-gram collection produces original expected results."""
        # Test exact n-gram generation from original test
        ngrams = tuple(asymcat.collect_ngrams("abcde", 2, "#"))
        expected_ngrams = (("#", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "#"))

        assert ngrams == expected_ngrams, f"N-gram mismatch: expected {expected_ngrams}, got {ngrams}"

    def test_original_cooccurrence_collection_with_ngrams(self):
        """Test co-occurrence collection with n-grams produces original results."""
        # Use exact same test case from original
        seqs = [("abcde", "ABCDE"), ("fgh", "FGH"), ("i", "I"), ("jkl", "JKL")]
        cooccs = asymcat.collect_cooccs(seqs, order=3)

        # Verify exact count from original test
        assert len(cooccs) == 78, f"N-gram co-occurrence count mismatch: expected 78, got {len(cooccs)}"

        # Verify specific expected pairs
        assert ("a", "B") in cooccs, "Missing expected co-occurrence ('a', 'B')"
        assert ("l", "L") in cooccs, "Missing expected co-occurrence ('l', 'L')"


class TestLegacyNewMethods:
    """Test compatibility of the new scoring methods from the original expansion."""

    @pytest.mark.parametrize(
        "dataset_file,min_pairs",
        [
            ("toy.tsv", 10),
            ("mushroom-small.tsv", 8),
            ("cmudict.sample100.tsv", 100),
        ],
    )
    def test_new_methods_basic_validation(self, dataset_file: str, min_pairs: int):
        """
        Test new scoring methods with basic validation from original tests.

        This reproduces the validation logic from test_new_scoring_methods().
        """
        file_path = RESOURCE_DIR / dataset_file
        if not file_path.exists():
            pytest.skip(f"Dataset {dataset_file} not available")

        # Load and process data
        data = asymcat.read_sequences(str(file_path))
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Test new methods with original validation logic
        new_methods = [
            ('mutual_information', scorer.mutual_information),
            ('normalized_mutual_information', scorer.normalized_mutual_information),
            ('jaccard_index', scorer.jaccard_index),
            ('goodman_kruskal_lambda', scorer.goodman_kruskal_lambda),
            ('log_likelihood_ratio', scorer.log_likelihood_ratio),
        ]

        for method_name, method_func in new_methods:
            scores = method_func()
            assert_valid_scores(scores)
            assert len(scores) >= min_pairs, f"Too few pairs in {method_name}: {len(scores)}"

            # Validate specific properties as in original tests
            for pair, (xy, yx) in scores.items():
                if method_name == 'mutual_information':
                    assert xy >= 0 and yx >= 0, f"MI should be non-negative for {pair}"
                    assert not np.isnan(xy) and not np.isnan(yx), f"MI should not be NaN for {pair}"

                elif method_name == 'normalized_mutual_information':
                    assert 0.0 <= xy <= 1.0, f"NMI should be in [0,1] for {pair}: {xy}"
                    assert 0.0 <= yx <= 1.0, f"NMI should be in [0,1] for {pair}: {yx}"

                elif method_name == 'jaccard_index':
                    assert 0.0 <= xy <= 1.0, f"Jaccard should be in [0,1] for {pair}: {xy}"
                    assert 0.0 <= yx <= 1.0, f"Jaccard should be in [0,1] for {pair}: {yx}"
                    # Jaccard should be symmetric
                    assert abs(xy - yx) < 1e-10, f"Jaccard should be symmetric for {pair}"

                elif method_name == 'goodman_kruskal_lambda':
                    assert 0.0 <= xy <= 1.0, f"Lambda should be in [0,1] for {pair}: {xy}"
                    assert 0.0 <= yx <= 1.0, f"Lambda should be in [0,1] for {pair}: {yx}"

                elif method_name == 'log_likelihood_ratio':
                    assert xy >= 0 and yx >= 0, f"G² should be non-negative for {pair}"
                    assert np.isfinite(xy) and np.isfinite(yx), f"G² should be finite for {pair}"

    def test_new_methods_mathematical_relationships(self):
        """
        Test mathematical relationships between methods from original tests.

        This reproduces test_scoring_methods_consistency() validation.
        """
        file_path = RESOURCE_DIR / "cmudict.sample100.tsv"
        if not file_path.exists():
            pytest.skip("CMU sample not available")

        data = asymcat.read_sequences(str(file_path))
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Get results for relationship testing
        mi = scorer.mutual_information()
        g2 = scorer.log_likelihood_ratio()
        chi2 = scorer.chi2()

        # Test correlation between G² and Chi² (as in original)
        g2_values = [g2[p][0] for p in g2.keys()]
        chi2_values = [chi2[p][0] for p in chi2.keys() if p in g2]

        if len(g2_values) > 2 and len(chi2_values) > 2:
            correlation = np.corrcoef(g2_values, chi2_values)[0, 1]
            assert correlation > 0.3, f"G² and Chi² should be correlated: {correlation}"

        # Test that all methods return same number of pairs
        assert len(mi) == len(g2), "MI and G² should have same number of pairs"

        # Test deterministic behavior (same results on repeated calls)
        mi2 = scorer.mutual_information()
        for pair in mi:
            assert np.allclose(mi[pair], mi2[pair], rtol=1e-10), f"MI not deterministic for {pair}"

    def test_new_methods_edge_cases(self):
        """
        Test edge cases for new methods from original tests.

        This reproduces test_edge_cases_and_error_handling() validation.
        """
        # Test with minimal data (as in original)
        minimal_data = [["A", "B"]]
        minimal_data = [[entry[0].split(), entry[1].split()] for entry in minimal_data]
        minimal_cooccs = asymcat.collect_cooccs(minimal_data)
        minimal_scorer = asymcat.scorer.CatScorer(minimal_cooccs)

        # Test all new methods work with minimal data
        new_methods = [
            minimal_scorer.mutual_information,
            minimal_scorer.normalized_mutual_information,
            minimal_scorer.jaccard_index,
            minimal_scorer.goodman_kruskal_lambda,
            minimal_scorer.log_likelihood_ratio,
        ]

        for method in new_methods:
            result = method()
            assert_valid_scores(result)
            assert len(result) >= 1, "Should have at least one pair"

            # Check no NaN or infinite values
            for scores in result.values():
                assert all(np.isfinite(s) for s in scores), "Should produce finite values"

        # Test caching behavior (as in original)
        mi1 = minimal_scorer.mutual_information()
        mi2 = minimal_scorer.mutual_information()
        assert mi1 is mi2, "Caching should work"
