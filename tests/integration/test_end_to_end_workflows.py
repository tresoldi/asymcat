"""
End-to-end integration tests for complete ASymCat workflows.

Tests complete pipelines from data loading to final scoring results.
"""

import time

import numpy as np
import pytest

import asymcat

from ..fixtures.assertions import (
    assert_performance_acceptable,
    assert_valid_cooccurrences,
    assert_valid_scores,
)
from ..fixtures.data import RESOURCE_DIR


class TestCompleteWorkflows:
    """Test complete analysis workflows from start to finish."""

    @pytest.mark.parametrize(
        "dataset,expected_properties",
        [
            ("toy.tsv", {"min_cooccs": 30, "min_pairs": 20, "max_time": 5.0}),
            ("mushroom-small.tsv", {"min_cooccs": 15, "min_pairs": 8, "max_time": 10.0}),
            ("cmudict.sample100.tsv", {"min_cooccs": 500, "min_pairs": 200, "max_time": 15.0}),
        ],
    )
    def test_sequence_analysis_workflow(self, dataset: str, expected_properties: dict):
        """
        Test complete sequence analysis workflow.

        Example workflow:
            1. Load sequence data from file
            2. Collect co-occurrences
            3. Create scorer
            4. Compute multiple measures
            5. Apply transformations
            6. Generate matrices for visualization
        """
        file_path = RESOURCE_DIR / dataset
        if not file_path.exists():
            pytest.skip(f"Dataset {dataset} not available")

        start_time = time.time()

        # Step 1: Load data
        data = asymcat.read_sequences(str(file_path))
        assert isinstance(data, list), "Data loading failed"
        assert len(data) > 0, "No data loaded"

        # Step 2: Collect co-occurrences
        cooccs = asymcat.collect_cooccs(data)
        assert_valid_cooccurrences(cooccs)
        assert len(cooccs) >= expected_properties["min_cooccs"], f"Too few co-occurrences: {len(cooccs)}"

        # Step 3: Create scorer
        scorer = asymcat.scorer.CatScorer(cooccs)
        assert hasattr(scorer, 'obs'), "Scorer creation failed"

        # Step 4: Compute multiple measures
        measures = {
            'mle': scorer.mle(),
            'tresoldi': scorer.tresoldi(),
            'mutual_info': scorer.mutual_information(),
            'jaccard': scorer.jaccard_index(),
            'chi2': scorer.chi2(),
        }

        for measure_name, scores in measures.items():
            assert_valid_scores(scores)
            assert len(scores) >= expected_properties["min_pairs"], f"Too few pairs in {measure_name}: {len(scores)}"

        # Step 5: Apply transformations
        scaled_scores = asymcat.scorer.scale_scorer(measures['tresoldi'], method="minmax")
        inverted_scores = asymcat.scorer.invert_scorer(scaled_scores)

        assert_valid_scores(scaled_scores)
        assert_valid_scores(inverted_scores)

        # Step 6: Generate matrices
        xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(measures['mle'])

        assert isinstance(xy_matrix, np.ndarray), "Matrix generation failed"
        assert len(x_labels) > 0 and len(y_labels) > 0, "Labels missing"

        # Performance check
        execution_time = time.time() - start_time
        assert_performance_acceptable(execution_time, expected_properties["max_time"])

    def test_presence_absence_workflow(self):
        """
        Test presence-absence matrix analysis workflow.

        Example workflow:
            1. Load presence-absence matrix
            2. Create scorer from co-occurrences
            3. Compute association measures
            4. Compare different measures
        """
        file_path = RESOURCE_DIR / "galapagos.tsv"
        if not file_path.exists():
            pytest.skip("Galapagos dataset not available")

        start_time = time.time()

        # Step 1: Load presence-absence data
        combinations = asymcat.read_pa_matrix(str(file_path))
        assert isinstance(combinations, list), "PA matrix loading failed"
        assert len(combinations) > 0, "No combinations loaded"

        # Step 2: Create scorer from combinations
        scorer = asymcat.scorer.CatScorer(combinations)

        # Step 3: Compute different association measures
        measures = {
            'chi2': scorer.chi2(),
            'cramers_v': scorer.cramers_v(),
            'fisher': scorer.fisher(),
            'jaccard': scorer.jaccard_index(),
        }

        for measure_name, scores in measures.items():
            # Fisher exact test can produce infinite values for perfect associations
            allow_inf = measure_name == 'fisher'
            assert_valid_scores(scores, allow_infinite=allow_inf)
            assert len(scores) > 0, f"No scores for {measure_name}"

        # Step 4: Compare measures (all should analyze same pairs)
        measure_names = list(measures.keys())
        for i in range(len(measure_names)):
            for j in range(i + 1, len(measure_names)):
                name1, name2 = measure_names[i], measure_names[j]
                # All measures should have some overlapping pairs
                common_pairs = set(measures[name1].keys()) & set(measures[name2].keys())
                assert len(common_pairs) > 0, f"No common pairs between {name1} and {name2}"

        execution_time = time.time() - start_time
        assert_performance_acceptable(execution_time, max_time=10.0)

    def test_comparative_analysis_workflow(self):
        """
        Test workflow for comparing different scoring methods.

        Example workflow:
            1. Load data and create scorer
            2. Compute all available measures
            3. Standardize measures for comparison
            4. Analyze correlations between measures
            5. Identify best-performing pairs across measures
        """
        file_path = RESOURCE_DIR / "cmudict.sample100.tsv"
        if not file_path.exists():
            pytest.skip("CMU sample not available")

        # Step 1: Setup
        data = asymcat.read_sequences(str(file_path))
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Step 2: Compute all measures
        all_measures = {
            'mle': scorer.mle(),
            'pmi': scorer.pmi(),
            'npmi': scorer.pmi(True),
            'chi2': scorer.chi2(),
            'cramers_v': scorer.cramers_v(),
            'fisher': scorer.fisher(),
            'theil_u': scorer.theil_u(),
            'tresoldi': scorer.tresoldi(),
            'mutual_info': scorer.mutual_information(),
            'jaccard': scorer.jaccard_index(),
            'gk_lambda': scorer.goodman_kruskal_lambda(),
        }

        # Step 3: Standardize measures
        standardized = {}
        for name, scores in all_measures.items():
            try:
                standardized[name] = asymcat.scorer.scale_scorer(scores, method="stdev")
                assert_valid_scores(standardized[name])
            except (ValueError, ZeroDivisionError):
                # Some measures might not be suitable for standardization
                pass

        assert len(standardized) > 0, "No measures could be standardized"

        # Step 4: Analyze correlations (test a few pairs)
        measure_pairs = [
            ('chi2', 'cramers_v'),  # Should be highly correlated
            ('pmi', 'npmi'),  # Should be correlated
            ('mle', 'theil_u'),  # Should show some correlation
        ]

        for measure1, measure2 in measure_pairs:
            if measure1 in standardized and measure2 in standardized:
                scores1 = standardized[measure1]
                scores2 = standardized[measure2]

                common_pairs = set(scores1.keys()) & set(scores2.keys())
                if len(common_pairs) > 10:  # Need enough pairs for correlation
                    values1 = [scores1[p][0] for p in common_pairs]  # X→Y direction
                    values2 = [scores2[p][0] for p in common_pairs]

                    correlation = np.corrcoef(values1, values2)[0, 1]
                    # Just check that correlation is computable
                    assert np.isfinite(correlation), f"Invalid correlation for {measure1} vs {measure2}"

        # Step 5: Identify top pairs across measures
        top_pairs_per_measure = {}
        for name, scores in all_measures.items():
            # Get top 5 pairs by maximum score in either direction
            sorted_pairs = sorted(scores.items(), key=lambda x: max(x[1]), reverse=True)
            top_pairs_per_measure[name] = sorted_pairs[:5]

        # Verify we have identified top pairs for each measure
        for name, top_pairs in top_pairs_per_measure.items():
            assert len(top_pairs) > 0, f"No top pairs identified for {name}"
            assert all(len(pair_data) == 2 for _, pair_data in top_pairs), f"Invalid top pairs format for {name}"


class TestLargeDatasetWorkflows:
    """Test workflows with larger datasets for performance and scalability."""

    @pytest.mark.slow
    def test_large_dataset_workflow(self):
        """
        Test complete workflow with large dataset (if available).

        This test demonstrates scalability and performance characteristics.
        """
        large_files = [
            "cmudict.sample1000.tsv",
            "cmudict.tsv",  # Full dataset if available
        ]

        test_file = None
        for filename in large_files:
            file_path = RESOURCE_DIR / filename
            if file_path.exists():
                test_file = file_path
                break

        if test_file is None:
            pytest.skip("No large dataset available for testing")

        start_time = time.time()

        # Load and process data
        data = asymcat.read_sequences(str(test_file))
        load_time = time.time() - start_time

        # Should handle large datasets reasonably quickly
        assert_performance_acceptable(load_time, max_time=30.0)

        # Process co-occurrences
        start_time = time.time()
        cooccs = asymcat.collect_cooccs(data)
        process_time = time.time() - start_time
        assert_performance_acceptable(process_time, max_time=60.0)

        # Create scorer and test key measures
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Test a few key measures for performance
        measures_to_test = [
            ('MLE', scorer.mle),
            ('Tresoldi', scorer.tresoldi),
            ('Mutual Info', scorer.mutual_information),
        ]

        for measure_name, method in measures_to_test:
            start_time = time.time()
            scores = method()
            method_time = time.time() - start_time

            assert_valid_scores(scores)
            assert_performance_acceptable(method_time, max_time=30.0)

            # Test caching works
            start_time = time.time()
            cached_scores = method()
            cache_time = time.time() - start_time

            assert scores is cached_scores, f"Caching not working for {measure_name}"
            assert_performance_acceptable(cache_time, max_time=0.1)


class TestErrorHandlingWorkflows:
    """Test error handling in complete workflows."""

    def test_invalid_file_workflow(self):
        """Test workflow behavior with invalid input files."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            asymcat.read_sequences("nonexistent.tsv")

        # Directory instead of file
        with pytest.raises((ValueError, IOError)):
            asymcat.read_sequences(str(RESOURCE_DIR))

    def test_empty_data_workflow(self):
        """Test workflow behavior with empty or minimal data."""
        # Empty data
        with pytest.raises(ValueError):
            asymcat.collect_cooccs([])

        # Minimal valid data
        minimal_data = [["A", "B"]]
        minimal_data = [[entry[0].split(), entry[1].split()] for entry in minimal_data]

        # Should work but produce minimal results
        cooccs = asymcat.collect_cooccs(minimal_data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # All methods should work with minimal data
        measures = [
            scorer.mle,
            scorer.mutual_information,
            scorer.jaccard_index,
        ]

        for method in measures:
            scores = method()
            assert_valid_scores(scores)
            assert len(scores) >= 1, "Should have at least one pair"

    def test_malformed_data_workflow(self):
        """Test workflow behavior with malformed input data."""
        # Wrong sequence structure
        with pytest.raises((ValueError, TypeError)):
            asymcat.collect_cooccs([["not", "a", "pair"]])  # Wrong length

        # Mismatched sequence lengths for n-grams
        with pytest.raises(ValueError):
            asymcat.collect_cooccs([["A B", "X"]], order=2)  # Different lengths


class TestVisualizationWorkflows:
    """Test workflows that prepare data for visualization."""

    def test_matrix_generation_workflow(self):
        """
        Test workflow for generating matrices suitable for heatmaps.

        Example workflow:
            1. Compute scores
            2. Convert to matrices
            3. Verify matrices are suitable for visualization libraries
        """
        # Use toy dataset for predictable results
        file_path = RESOURCE_DIR / "toy.tsv"
        if not file_path.exists():
            pytest.skip("Toy dataset not available")

        data = asymcat.read_sequences(str(file_path))
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Test matrix generation for different measures
        measures = ['mle', 'tresoldi', 'chi2', 'jaccard_index']

        for measure_name in measures:
            if hasattr(scorer, measure_name):
                method = getattr(scorer, measure_name)
                scores = method()

                # Convert to matrices
                xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scores)

                # Verify matrix properties for visualization
                assert isinstance(xy_matrix, np.ndarray), f"XY matrix not numpy array for {measure_name}"
                assert isinstance(yx_matrix, np.ndarray), f"YX matrix not numpy array for {measure_name}"

                # Check dimensions are reasonable
                assert xy_matrix.shape[0] <= 100, f"X dimension too large for visualization: {xy_matrix.shape[0]}"
                assert xy_matrix.shape[1] <= 100, f"Y dimension too large for visualization: {xy_matrix.shape[1]}"

                # Check labels are strings
                assert all(isinstance(label, str) for label in x_labels), f"X labels not strings for {measure_name}"
                assert all(isinstance(label, str) for label in y_labels), f"Y labels not strings for {measure_name}"

                # Check values are finite
                assert np.all(np.isfinite(xy_matrix)), f"Non-finite values in XY matrix for {measure_name}"
                assert np.all(np.isfinite(yx_matrix)), f"Non-finite values in YX matrix for {measure_name}"

    def test_score_transformation_workflow(self):
        """
        Test workflow for transforming scores for visualization.

        Example workflow:
            1. Compute raw scores
            2. Apply scaling/normalization
            3. Apply inversion if needed
            4. Prepare for plotting
        """
        file_path = RESOURCE_DIR / "toy.tsv"
        if not file_path.exists():
            pytest.skip("Toy dataset not available")

        data = asymcat.read_sequences(str(file_path))
        cooccs = asymcat.collect_cooccs(data)
        scorer = asymcat.scorer.CatScorer(cooccs)

        # Get raw scores
        raw_scores = scorer.tresoldi()

        # Apply different transformations
        transformations = [
            ("minmax", asymcat.scorer.scale_scorer, {"method": "minmax"}),
            ("stdev", asymcat.scorer.scale_scorer, {"method": "stdev"}),
            ("mean", asymcat.scorer.scale_scorer, {"method": "mean"}),
        ]

        for transform_name, transform_func, kwargs in transformations:
            try:
                transformed = transform_func(raw_scores, **kwargs)
                assert_valid_scores(transformed)

                # Test that transformation preserves structure
                assert set(transformed.keys()) == set(
                    raw_scores.keys()
                ), f"Transformation {transform_name} changed pairs"

                # Test inversion after transformation
                if transform_name == "minmax":  # Only invert bounded scores
                    inverted = asymcat.scorer.invert_scorer(transformed)
                    assert_valid_scores(inverted)

                    # Check inversion property
                    for pair in transformed:
                        orig_xy, orig_yx = transformed[pair]
                        inv_xy, inv_yx = inverted[pair]
                        assert abs((orig_xy + inv_xy) - 1.0) < 1e-10, f"Inversion failed for {pair} in {transform_name}"
                        assert abs((orig_yx + inv_yx) - 1.0) < 1e-10, f"Inversion failed for {pair} in {transform_name}"

            except (ValueError, ZeroDivisionError):
                # Some transformations might fail with certain data
                # This is acceptable behavior
                pass
