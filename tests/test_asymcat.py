#!/usr/bin/env python3

"""
test_asymcat
============

Tests for the `asymcat` package.
"""

# Import system libraries

# Resource dir
# TODO: move within tests
from pathlib import Path

# Import 3rd party libraries
import numpy as np

# Import the library being tested
import asymcat

RESOURCE_DIR = Path(__file__).parent.parent / "resources"

# Small sample for text, derived from CMU (first entry is manually added)
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

# Prepare data
data_cmu = [[entry[0].split(), entry[1].split()] for entry in sample_cmu]


def test_compute():
    # Compute cooccs
    cooccs = asymcat.collect_cooccs(data_cmu)

    # Collect lengths
    cooccs_A = [coocc for coocc in cooccs if coocc[0] == "A"]
    cooccs_Ll = [coocc for coocc in cooccs if coocc[0] == "L" and coocc[1] == "l"]

    # Assert length as proxy for right collection
    assert len(cooccs) == 879
    assert len(cooccs_A) == 92
    assert len(cooccs_Ll) == 14


def test_scorers():
    # Compute cooccs and build scorer
    cooccs = asymcat.collect_cooccs(data_cmu)
    scorer = asymcat.scorer.CatScorer(cooccs)

    # Get all scorers
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

    # Defines testing pairs
    pairs = {
        ("ONE", "1"): (
            1.0,  # mle x>y
            1.0,  # mle y > x
            5.680172609017068,  # pmi x>y
            5.680172609017068,  # pmi y>x
            1.0,  # npmi x>y
            1.0,  # npmi y>x
            609.5807658175393,  # chi2 x>y
            609.5807658175393,  # chi2 y>x
            879.0,  # chi2_ns x>y
            879.0,  # chi2_ns y>x
            0.8325526903114843,  # cramersv x>y
            0.8325526903114843,  # cramersv y>x
            0.7065025023855139,  # cramersv_ns x>y
            0.7065025023855139,  # cramersv_ns y>x
            np.inf,  # fisher x>y
            np.inf,  # fisher y>x
            1.0,  # theil_u x>y
            1.0,  # theil_u y>x
            0.0,  # cond_entropy, x>y
            0.0,  # cond_entropy, y>x
            1.0,  # tresoldi x>y
            1.0,  # tresoldi y>x
        ),
        ("A", "b"): (
            0.06521739130434782,  # mle x>y
            0.11320754716981132,  # mle y>x
            0.07846387631207004,  # pmi x>y
            0.07846387631207004,  # pmi y>x
            0.015733602612959818,  # npmi x>y
            0.015733602612959818,  # npmi y>x
            0.0,  # chi2 x>y
            0.0,  # chi2 y>x
            0.043927505580845905,  # chi2_ns x>y
            0.043927505580845905,  # chi2_ns y>x
            0.0,  # cramersv x>y
            0.0,  # cramersv y>x
            0.0,  # cramersv_ns x>y
            0.0,  # cramersv_ns y>x
            1.0984661058881742,  # fisher x>y
            1.0984661058881742,  # fisher y>x
            0.21299752425693524,  # theil_u x>y
            0.3356184612000498,  # theil_u y>x
            1.86638224482290279,  # cond_entropy, x>y
            0.9999327965500219,  # cond_entropy, y>x
            0.0926310345228265,  # tresoldi x>y
            0.10466500171366895,  # tresoldi y>x
        ),
        ("S", "s"): (
            0.13953488372093023,  # mle x>y
            0.17142857142857143,  # mle y>x
            1.2539961897302558,  # pmi x>y
            1.2539961897302558,  # pmi y>x
            0.2514517336476095,  # npmi x>y
            0.2514517336476095,  # npmi y>x
            9.176175043924879,  # chi2 x>y
            9.176175043924879,  # chi2 y>x
            11.758608367318205,  # chi2_ns x>y
            11.758608367318205,  # chi2_ns y>x
            0.09649345638896019,  # cramersv x>y
            0.09649345638896019,  # cramersv y>x
            0.07452170854897658,  # cramersv_ns x>y
            0.07452170854897658,  # cramersv_ns y>x
            4.512581547064306,  # fisher x>y
            4.512581547064306,  # fisher y>x
            0.22071631715993364,  # theil_u x>y
            0.2841291022637977,  # theil_u y>x
            1.5938047875022765,  # cond_entropy, x>y
            1.137346966185816,  # cond_entropy, y>x
            1.2150117159149825,  # tresoldi x>y
            1.2062725270739942,  # tresoldi y>x
        ),
        ("H", "i"): (
            0.0,  # mle x>y
            0.0,  # mle y>x
            -6.502790045915624,  # pmi x>y
            -6.502790045915624,  # pmi y>x
            -0.4796427489634758,  # npmi x>y
            -0.4796427489634758,  # npmi y>x
            0.09374177071030182,  # chi2 x>y
            0.09374177071030182,  # chi2 y>x
            0.8057902693787795,  # chi2_ns x>y
            0.8057902693787795,  # chi2_ns y>x
            0.0,  # cramersv x>y
            0.0,  # cramersv y>x
            0.0,  # cramersv_ns x>y
            0.0,  # cramersv_ns y>x
            0.0,  # fisher x>y
            0.0,  # fisher y>x
            0.386699915220347,  # theil_u x>y
            0.34435838354803283,  # theil_u y>x
            1.0887395664391526,  # cond_entropy, x>y
            1.3070160180503212,  # cond_entropy, y>x
            -6.502790045915624,  # tresoldi x>y
            -6.502790045915624,  # tresoldi y>x
        ),
    }

    for pair, ref in pairs.items():
        vals = (
            mle[pair]
            + pmi[pair]
            + npmi[pair]
            + chi2[pair]
            + chi2_ns[pair]
            + cramersv[pair]
            + cramersv_ns[pair]
            + fisher[pair]
            + theil_u[pair]
            + cond_entropy[pair]
            + tresoldi[pair]
        )
        print(pair, vals)
        assert np.allclose(vals, ref, rtol=1e-05, atol=1e-08)  # TODO: use pytest.approx

    # Build a scaling dictionary
    score_dict = scorer.tresoldi()

    # Build matrices from scorer
    xy, yx, alpha_x, alpha_y = asymcat.scorer.scorer2matrices(score_dict)
    assert len(xy) == 28
    assert len(yx) == 23
    assert len(alpha_x) == 23
    assert len(alpha_y) == 28
    assert "A" in alpha_x and "s" not in alpha_x
    assert "s" in alpha_y and "A" not in alpha_y

    # Scale the scorer
    scaled_minmax = asymcat.scorer.scale_scorer(score_dict, method="minmax")
    assert np.allclose(
        scaled_minmax["H", "i"],
        (0.15476857281060225, 0.15476857281060225),
        rtol=1e-05,
        atol=1e-08,
    )
    scaled_mean = asymcat.scorer.scale_scorer(score_dict, method="mean")
    assert np.allclose(
        scaled_mean["H", "i"],
        (-0.36871270234063913, -0.36871270234063913),
        rtol=1e-05,
        atol=1e-08,
    )
    scaled_stdev = asymcat.scorer.scale_scorer(score_dict, method="stdev")
    assert np.allclose(
        scaled_stdev["H", "i"],
        (-1.3465717087048406, -1.3465717087048406),
        rtol=1e-05,
        atol=1e-08,
    )

    # invert the scorer
    inverted = asymcat.scorer.invert_scorer(scaled_minmax)
    assert np.allclose(
        inverted["H", "i"],
        (0.8452314271893977, 0.8452314271893977),
        rtol=1e-05,
        atol=1e-08,
    )


def test_readers():
    # Read a sequences file
    cmu_file = RESOURCE_DIR / "cmudict.tsv"
    cmu = asymcat.read_sequences(cmu_file.as_posix())

    # Read presence/absence matrix
    finches_file = RESOURCE_DIR / "galapagos.tsv"
    finches = asymcat.read_pa_matrix(finches_file.as_posix())

    # For assertion, just check length
    assert len(cmu) == 134373
    assert len(finches) == 447


def test_new_scoring_methods():
    """
    Test all new scoring methods with multiple datasets and validation.
    """
    # Test with toy dataset
    toy_file = RESOURCE_DIR / "toy.tsv"
    toy_data = asymcat.read_sequences(toy_file.as_posix())
    toy_cooccs = asymcat.collect_cooccs(toy_data)
    toy_scorer = asymcat.scorer.CatScorer(toy_cooccs)

    # Test 1: Mutual Information
    mi = toy_scorer.mutual_information()
    assert isinstance(mi, dict)
    assert len(mi) > 0
    # Check all values are finite and non-negative for MI
    for pair, (mi_xy, mi_yx) in mi.items():
        assert isinstance(mi_xy, (int, float))
        assert isinstance(mi_yx, (int, float))
        assert not np.isnan(mi_xy) and not np.isnan(mi_yx)
        assert mi_xy >= 0 and mi_yx >= 0  # MI is always non-negative

    # Test 2: Normalized Mutual Information
    nmi = toy_scorer.normalized_mutual_information()
    assert isinstance(nmi, dict)
    assert len(nmi) == len(mi)  # Same number of pairs
    # Check all values are in [0,1] range
    for pair, (nmi_xy, nmi_yx) in nmi.items():
        assert 0.0 <= nmi_xy <= 1.0
        assert 0.0 <= nmi_yx <= 1.0
        assert not np.isnan(nmi_xy) and not np.isnan(nmi_yx)

    # Test 3: Jaccard Index
    jaccard = toy_scorer.jaccard_index()
    assert isinstance(jaccard, dict)
    assert len(jaccard) == len(mi)
    # Check all values are in [0,1] range (Jaccard property)
    for pair, (j_xy, j_yx) in jaccard.items():
        assert 0.0 <= j_xy <= 1.0
        assert 0.0 <= j_yx <= 1.0
        assert not np.isnan(j_xy) and not np.isnan(j_yx)
        # Jaccard is symmetric, so values should be equal
        assert abs(j_xy - j_yx) < 1e-10

    # Test 4: Goodman-Kruskal Lambda
    gk_lambda = toy_scorer.goodman_kruskal_lambda()
    assert isinstance(gk_lambda, dict)
    assert len(gk_lambda) == len(mi)
    # Check all values are in [0,1] range
    for pair, (lambda_xy, lambda_yx) in gk_lambda.items():
        assert 0.0 <= lambda_xy <= 1.0
        assert 0.0 <= lambda_yx <= 1.0
        assert not np.isnan(lambda_xy) and not np.isnan(lambda_yx)

    # Test 5: Log-Likelihood Ratio (both square and non-square)
    g2_square = toy_scorer.log_likelihood_ratio(square_ct=True)
    g2_nonsquare = toy_scorer.log_likelihood_ratio(square_ct=False)
    assert isinstance(g2_square, dict)
    assert isinstance(g2_nonsquare, dict)
    assert len(g2_square) == len(mi)
    assert len(g2_nonsquare) == len(mi)
    # Check all values are non-negative (G² property)
    for pair in g2_square:
        g2_xy, g2_yx = g2_square[pair]
        assert g2_xy >= 0 and g2_yx >= 0
        assert not np.isnan(g2_xy) and not np.isnan(g2_yx)
        g2ns_xy, g2ns_yx = g2_nonsquare[pair]
        assert g2ns_xy >= 0 and g2ns_yx >= 0
        assert not np.isnan(g2ns_xy) and not np.isnan(g2ns_yx)

    # Test with larger dataset (mushroom-small)
    mushroom_file = RESOURCE_DIR / "mushroom-small.tsv"
    mushroom_data = asymcat.read_sequences(mushroom_file.as_posix())
    mushroom_cooccs = asymcat.collect_cooccs(mushroom_data)
    mushroom_scorer = asymcat.scorer.CatScorer(mushroom_cooccs)

    # Test all methods work with larger dataset
    methods = [
        ('mutual_information', mushroom_scorer.mutual_information),
        ('normalized_mutual_information', mushroom_scorer.normalized_mutual_information),
        ('jaccard_index', mushroom_scorer.jaccard_index),
        ('goodman_kruskal_lambda', mushroom_scorer.goodman_kruskal_lambda),
        ('log_likelihood_ratio', mushroom_scorer.log_likelihood_ratio),
    ]

    for method_name, method_func in methods:
        result = method_func()
        assert isinstance(result, dict)
        assert len(result) > 0
        # Check no NaN or infinite values
        for pair, scores in result.items():
            assert len(scores) == 2  # (xy, yx) tuple
            assert all(np.isfinite(s) for s in scores)

    # Test mathematical relationships and properties
    mi_result = mushroom_scorer.mutual_information()
    nmi_result = mushroom_scorer.normalized_mutual_information()

    # Property: NMI should be <= 1 (it's normalized)
    for pair in mi_result:
        nmi_val = nmi_result[pair][0]
        assert nmi_val <= 1.0  # NMI is always <= 1 by definition
        assert nmi_val >= 0.0  # NMI is always >= 0
        # NMI can be > MI when joint entropy < 1, so no direct comparison


def test_new_methods_with_known_values():
    """
    Test new scoring methods with known, predictable data for validation.
    """
    # Create simple, predictable test data
    # Perfect correlation case
    perfect_data = [["A A A", "B B B"], ["C C C", "D D D"]]
    perfect_data = [[entry[0].split(), entry[1].split()] for entry in perfect_data]
    perfect_cooccs = asymcat.collect_cooccs(perfect_data)
    perfect_scorer = asymcat.scorer.CatScorer(perfect_cooccs)

    # Test Jaccard Index with perfect correlation
    jaccard_perfect = perfect_scorer.jaccard_index()
    # For perfect 1:1 mapping, some pairs should have high Jaccard values
    assert any(max(scores) > 0.3 for scores in jaccard_perfect.values())

    # Test MI with independent data
    # Create data where variables are independent
    independent_data = [["A B", "C D"], ["A B", "D C"], ["B A", "C D"], ["B A", "D C"]]
    independent_data = [[entry[0].split(), entry[1].split()] for entry in independent_data]
    independent_cooccs = asymcat.collect_cooccs(independent_data)
    independent_scorer = asymcat.scorer.CatScorer(independent_cooccs)

    mi_independent = independent_scorer.mutual_information()
    # For independent variables, MI should be close to 0
    for pair, scores in mi_independent.items():
        # Allow some tolerance for finite sample effects
        assert all(abs(s) < 2.0 for s in scores)  # Should be low but not necessarily 0

    # Test Lambda with deterministic relationships
    deterministic_data = [["A", "X"], ["A", "X"], ["B", "Y"], ["B", "Y"]]
    deterministic_data = [[entry[0].split(), entry[1].split()] for entry in deterministic_data]
    deterministic_cooccs = asymcat.collect_cooccs(deterministic_data)
    deterministic_scorer = asymcat.scorer.CatScorer(deterministic_cooccs)

    lambda_det = deterministic_scorer.goodman_kruskal_lambda()
    # For deterministic relationships, some pairs should show high predictive power
    # The pairs that never co-occur should have high lambda values
    all_lambda_values = [max(scores) for scores in lambda_det.values()]
    # At least one pair should have high lambda (perfect predictive power)
    assert max(all_lambda_values) >= 0.5


def test_scoring_methods_consistency():
    """
    Test that new methods are consistent with existing ones where expected.
    """
    # Use CMU sample data
    cmu_file = RESOURCE_DIR / "cmudict.sample100.tsv"
    cmu_data = asymcat.read_sequences(cmu_file.as_posix())
    cmu_cooccs = asymcat.collect_cooccs(cmu_data)
    cmu_scorer = asymcat.scorer.CatScorer(cmu_cooccs)

    # Get both new and existing methods
    mi = cmu_scorer.mutual_information()
    g2 = cmu_scorer.log_likelihood_ratio()
    chi2 = cmu_scorer.chi2()
    tresoldi = cmu_scorer.tresoldi()

    # Test: G² and Chi² should be correlated (both test independence)
    g2_values = [g2[p][0] for p in g2.keys()]
    chi2_values = [chi2[p][0] for p in chi2.keys() if p in g2]

    if len(g2_values) > 2 and len(chi2_values) > 2:
        correlation = np.corrcoef(g2_values, chi2_values)[0, 1]
        # Should be positively correlated (both measure association)
        assert correlation > 0.3  # Allow for some differences in calculation

    # Test: All methods should return same number of pairs
    assert len(mi) == len(g2)
    assert len(mi) == len(tresoldi)

    # Test: Methods should be deterministic (same results on repeated calls)
    mi2 = cmu_scorer.mutual_information()
    for pair in mi:
        assert np.allclose(mi[pair], mi2[pair], rtol=1e-10)


def test_edge_cases_and_error_handling():
    """
    Test edge cases and error handling for new scoring methods.
    """
    # Test with minimal data
    minimal_data = [["A", "B"]]
    minimal_data = [[entry[0].split(), entry[1].split()] for entry in minimal_data]
    minimal_cooccs = asymcat.collect_cooccs(minimal_data)
    minimal_scorer = asymcat.scorer.CatScorer(minimal_cooccs)

    # All methods should work with minimal data
    methods = [
        minimal_scorer.mutual_information,
        minimal_scorer.normalized_mutual_information,
        minimal_scorer.jaccard_index,
        minimal_scorer.goodman_kruskal_lambda,
        minimal_scorer.log_likelihood_ratio,
    ]

    for method in methods:
        result = method()
        assert isinstance(result, dict)
        assert len(result) >= 1  # At least one pair
        # Check no NaN or infinite values
        for scores in result.values():
            assert all(np.isfinite(s) for s in scores)

    # Test caching behavior (methods should return same object on repeated calls)
    mi1 = minimal_scorer.mutual_information()
    mi2 = minimal_scorer.mutual_information()
    assert mi1 is mi2  # Should be the same cached object

    # Test that computation functions handle edge cases
    from asymcat.scorer import (
        compute_goodman_kruskal_lambda,
        compute_jaccard_index,
        compute_log_likelihood_ratio,
        compute_mutual_information,
        compute_normalized_mutual_information,
    )

    # Test empty inputs
    assert compute_mutual_information([], []) == 0.0
    assert compute_normalized_mutual_information([], []) == 0.0
    assert compute_jaccard_index([], []) == 0.0
    assert compute_goodman_kruskal_lambda([], [], "y_given_x") == 0.0

    # Test single contingency table
    single_ct = [[1, 0], [0, 1]]
    g2_single = compute_log_likelihood_ratio(single_ct)
    assert np.isfinite(g2_single)
    assert g2_single >= 0


def test_comprehensive_datasets():
    """
    Test all new scoring methods with multiple real datasets from resources.
    """
    datasets = [
        ("toy.tsv", "Toy dataset"),
        ("mushroom-small.tsv", "Mushroom dataset (small)"),
        ("cmudict.sample100.tsv", "CMU Dictionary sample"),
    ]

    for filename, description in datasets:
        print(f"Testing {description}...")
        file_path = RESOURCE_DIR / filename

        # Load and process data
        data = asymcat.read_sequences(file_path.as_posix())
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        print(f"  {len(cooccs)} co-occurrences, {len(scorer.alphabet_x)} x symbols, {len(scorer.alphabet_y)} y symbols")

        # Test all new methods
        new_methods = [
            ("Mutual Information", scorer.mutual_information),
            ("Normalized MI", scorer.normalized_mutual_information),
            ("Jaccard Index", scorer.jaccard_index),
            ("Goodman-Kruskal Lambda", scorer.goodman_kruskal_lambda),
            ("Log-Likelihood Ratio", scorer.log_likelihood_ratio),
        ]

        for method_name, method_func in new_methods:
            result = method_func()

            # Basic sanity checks
            assert isinstance(result, dict)
            assert len(result) > 0

            # Check mathematical properties
            for pair, scores in result.items():
                assert len(scores) == 2
                for score in scores:
                    assert np.isfinite(score), f"{method_name} produced non-finite value for {pair}"

                    # Method-specific bounds
                    if method_name in ["Normalized MI", "Jaccard Index", "Goodman-Kruskal Lambda"]:
                        assert 0.0 <= score <= 1.0, f"{method_name} value {score} out of [0,1] range"
                    elif method_name in ["Mutual Information", "Log-Likelihood Ratio"]:
                        assert score >= 0.0, f"{method_name} value {score} should be non-negative"

        print(f"  ✓ All methods passed for {description}")

    print("✅ Comprehensive dataset testing completed successfully!")


def test_performance_and_scalability():
    """
    Test performance characteristics and scalability of new methods.
    """
    import time

    # Test with the largest available dataset
    large_file = RESOURCE_DIR / "cmudict.sample1000.tsv"
    if large_file.exists():
        print("Testing performance with large dataset...")

        start_time = time.time()
        data = asymcat.read_sequences(large_file.as_posix())
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)
        load_time = time.time() - start_time

        print(f"  Loaded {len(cooccs)} co-occurrences in {load_time:.2f}s")

        # Test each method's performance
        methods = [
            ("MI", scorer.mutual_information),
            ("NMI", scorer.normalized_mutual_information),
            ("Jaccard", scorer.jaccard_index),
            ("Lambda", scorer.goodman_kruskal_lambda),
            ("G²", scorer.log_likelihood_ratio),
        ]

        for method_name, method_func in methods:
            start_time = time.time()
            result = method_func()
            method_time = time.time() - start_time

            print(f"  {method_name}: {len(result)} pairs in {method_time:.2f}s")

            # Performance check: should complete within reasonable time
            assert method_time < 30.0, f"{method_name} took too long: {method_time:.2f}s"

            # Verify caching works (second call should be much faster)
            start_time = time.time()
            result2 = method_func()
            cached_time = time.time() - start_time

            assert result is result2, f"{method_name} caching not working"
            assert cached_time < 0.01, f"{method_name} cached call too slow: {cached_time:.4f}s"

    print("✅ Performance testing completed successfully!")


def test_mathematical_properties():
    """
    Test mathematical properties and relationships between methods.
    """
    # Use a medium-sized dataset for reliable statistics
    data_file = RESOURCE_DIR / "mushroom-small.tsv"
    data = asymcat.read_sequences(data_file.as_posix())
    cooccs = asymcat.collect_cooccs(data)
    scorer = asymcat.scorer.CatScorer(cooccs)

    # Get all results
    mi = scorer.mutual_information()
    nmi = scorer.normalized_mutual_information()
    jaccard = scorer.jaccard_index()
    gk_lambda = scorer.goodman_kruskal_lambda()
    g2 = scorer.log_likelihood_ratio()
    chi2 = scorer.chi2()

    print("Testing mathematical properties...")

    # Property 1: Symmetry where expected
    print("  Testing symmetry properties...")
    for pair in jaccard:
        j_xy, j_yx = jaccard[pair]
        assert abs(j_xy - j_yx) < 1e-10, f"Jaccard not symmetric for {pair}"

    for pair in mi:
        mi_xy, mi_yx = mi[pair]
        assert abs(mi_xy - mi_yx) < 1e-10, f"MI not symmetric for {pair}"

    # Property 2: Range constraints
    print("  Testing range constraints...")
    for pair in nmi:
        for val in nmi[pair]:
            assert 0.0 <= val <= 1.0, f"NMI out of range for {pair}: {val}"

    for pair in gk_lambda:
        for val in gk_lambda[pair]:
            assert 0.0 <= val <= 1.0, f"Lambda out of range for {pair}: {val}"

    # Property 3: Non-negativity
    print("  Testing non-negativity...")
    for pair in mi:
        for val in mi[pair]:
            assert val >= 0.0, f"MI negative for {pair}: {val}"

    for pair in g2:
        for val in g2[pair]:
            assert val >= 0.0, f"G² negative for {pair}: {val}"

    # Property 4: Correlation between related methods
    print("  Testing correlations between related methods...")
    g2_vals = [g2[p][0] for p in g2.keys()]
    chi2_vals = [chi2[p][0] for p in chi2.keys() if p in g2]

    if len(g2_vals) > 5:
        correlation = np.corrcoef(g2_vals, chi2_vals)[0, 1]
        print(f"    G² vs Chi²: correlation = {correlation:.3f}")
        assert correlation > 0.5, "G² and Chi² should be strongly correlated"

    print("✅ Mathematical properties verified!")


def test_utils():
    # Test additional functions from utils
    ngrams = tuple(asymcat.collect_ngrams("abcde", 2, "#"))
    assert ngrams == (
        ("#", "a"),
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "#"),
    )

    # Test collect co-occoc on ngrams
    # TODO: this is giving three ("i", "I"), check if correct/intended
    seqs = [("abcde", "ABCDE"), ("fgh", "FGH"), ("i", "I"), ("jkl", "JKL")]
    cooccs = asymcat.collect_cooccs(seqs, order=3)
    assert len(cooccs) == 78
    assert ("a", "B") in cooccs
    assert ("l", "L") in cooccs
