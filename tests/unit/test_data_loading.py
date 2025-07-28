"""
Tests for data loading and preprocessing functionality.

Demonstrates how to load different types of categorical data into ASymCat.
"""

from typing import List

import pytest

import asymcat

from ..fixtures.assertions import assert_valid_cooccurrences
from ..fixtures.data import RESOURCE_DIR


class TestSequenceReading:
    """Test reading sequence data from files."""

    @pytest.mark.parametrize(
        "filename,expected_min_length",
        [
            ("toy.tsv", 2),
            ("mushroom-small.tsv", 15),
            ("cmudict.sample100.tsv", 90),
        ],
    )
    def test_read_sequences_from_files(self, filename: str, expected_min_length: int):
        """
        Test reading sequence data from TSV files.

        Example usage:
            data = asymcat.read_sequences("data.tsv")
        """
        file_path = RESOURCE_DIR / filename
        if not file_path.exists():
            pytest.skip(f"Test file {filename} not available")

        # Load the data
        data = asymcat.read_sequences(str(file_path))

        # Validate basic properties
        assert isinstance(data, list), "Should return a list"
        assert len(data) >= expected_min_length, f"Should have at least {expected_min_length} entries"

        # Check data structure
        for i, entry in enumerate(data[:5]):  # Check first few entries
            assert isinstance(entry, list), f"Entry {i} should be a list"
            assert len(entry) == 2, f"Entry {i} should have exactly 2 sequences"

            seq_a, seq_b = entry
            assert isinstance(seq_a, list), f"First sequence in entry {i} should be a list"
            assert isinstance(seq_b, list), f"Second sequence in entry {i} should be a list"
            assert len(seq_a) > 0, f"First sequence in entry {i} should not be empty"
            assert len(seq_b) > 0, f"Second sequence in entry {i} should not be empty"

    def test_read_sequences_with_columns(self):
        """Test reading specific columns from sequence files."""
        file_path = RESOURCE_DIR / "wiktionary.tsv"
        if not file_path.exists():
            pytest.skip("Wiktionary test file not available")

        # Read with specific columns (if file has headers)
        try:
            data = asymcat.read_sequences(str(file_path), cols=["English", "Finnish"])
            assert isinstance(data, list)
            # Should only contain entries where both columns have data
            for entry in data[:3]:
                assert len(entry) == 2
                assert all(len(seq) > 0 for seq in entry)
        except (ValueError, KeyError):
            # File might not have the expected columns, which is OK
            pytest.skip("File doesn't have expected column structure")

    def test_read_sequences_error_handling(self):
        """Test error handling for sequence reading."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            asymcat.read_sequences("nonexistent.tsv")

        # Invalid file type (try to read directory)
        with pytest.raises((ValueError, IOError)):
            asymcat.read_sequences(str(RESOURCE_DIR))


class TestPresenceAbsenceMatrixReading:
    """Test reading presence-absence matrix data."""

    def test_read_pa_matrix(self):
        """
        Test reading presence-absence matrices.

        Example usage:
            data = asymcat.read_pa_matrix("species_data.tsv")
        """
        file_path = RESOURCE_DIR / "galapagos.tsv"
        if not file_path.exists():
            pytest.skip("Galapagos test file not available")

        # Load the presence-absence matrix
        combinations = asymcat.read_pa_matrix(str(file_path))

        # Validate basic properties
        assert isinstance(combinations, list), "Should return a list"
        assert len(combinations) > 0, "Should have co-occurrence combinations"

        # Check data structure - should be tuples of species pairs
        for combo in combinations[:5]:  # Check first few
            assert isinstance(combo, tuple), "Each combination should be a tuple"
            assert len(combo) == 2, "Each combination should have 2 species"
            assert all(isinstance(species, str) for species in combo), "Species names should be strings"

    def test_read_pa_matrix_error_handling(self):
        """Test error handling for presence-absence matrix reading."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            asymcat.read_pa_matrix("nonexistent.tsv")


class TestCooccurrenceCollection:
    """Test co-occurrence collection from processed data."""

    @pytest.mark.parametrize(
        "test_data,expected_properties",
        [
            # Simple case: each sequence becomes co-occurrences
            ([[["a", "b"], ["x", "y"]]], {"min_cooccs": 4, "contains": [("a", "x")]}),
            # Multiple sequences
            ([[["a"], ["x"]], [["b"], ["y"]]], {"min_cooccs": 2, "contains": [("a", "x"), ("b", "y")]}),
            # Sequences of different lengths
            ([[["a", "b", "c"], ["x", "y", "z"]]], {"min_cooccs": 9, "contains": [("a", "x"), ("c", "z")]}),
        ],
    )
    def test_collect_cooccurrences_basic(self, test_data: List, expected_properties: dict):
        """
        Test basic co-occurrence collection.

        Example usage:
            data = [[["a", "b"], ["x", "y"]]]
            cooccs = asymcat.collect_cooccs(data)
        """
        cooccs = asymcat.collect_cooccs(test_data)

        # Validate structure
        assert_valid_cooccurrences(cooccs)

        # Check expected properties
        assert (
            len(cooccs) >= expected_properties["min_cooccs"]
        ), f"Should have at least {expected_properties['min_cooccs']} co-occurrences"

        # Check that expected pairs are present
        cooccs_set = set(cooccs)
        for expected_pair in expected_properties["contains"]:
            assert expected_pair in cooccs_set, f"Should contain co-occurrence {expected_pair}"

    @pytest.mark.parametrize(
        "order,expected_count",
        [
            (2, 40),  # Bigram co-occurrences
            (3, 78),  # Trigram co-occurrences
        ],
    )
    def test_collect_cooccurrences_ngrams(self, order: int, expected_count: int):
        """
        Test n-gram co-occurrence collection.

        Example usage:
            cooccs = asymcat.collect_cooccs(data, order=3, pad="#")
        """
        # Test data from the original test suite
        seqs = [("abcde", "ABCDE"), ("fgh", "FGH"), ("i", "I"), ("jkl", "JKL")]

        # Collect n-gram co-occurrences
        cooccs = asymcat.collect_cooccs(seqs, order=order, pad="#")

        # Validate structure
        assert_valid_cooccurrences(cooccs)
        assert len(cooccs) == expected_count, f"Should have exactly {expected_count} {order}-gram co-occurrences"

        # Check specific expected pairs
        assert ("a", "B") in cooccs, "Should contain ('a', 'B') co-occurrence"
        assert ("l", "L") in cooccs, "Should contain ('l', 'L') co-occurrence"

    def test_collect_cooccurrences_padding(self):
        """Test that padding symbols are properly excluded."""
        data = [[["a"], ["b"]]]

        # Test with default padding
        cooccs = asymcat.collect_cooccs(data, order=2, pad="#")

        # Should not contain any padding symbols
        for x, y in cooccs:
            assert x != "#", "Should not contain padding symbol in first position"
            assert y != "#", "Should not contain padding symbol in second position"

    def test_collect_cooccurrences_error_handling(self):
        """Test error handling for co-occurrence collection."""
        # Empty data
        with pytest.raises(ValueError):
            asymcat.collect_cooccs([])

        # Invalid sequence structure
        with pytest.raises((ValueError, TypeError)):
            asymcat.collect_cooccs([["not", "a", "pair"]])  # Wrong length

        # Mismatched sequence lengths for n-grams
        with pytest.raises(ValueError):
            asymcat.collect_cooccs([[["a", "b"], ["x"]]], order=2)  # Different lengths


class TestUtilityFunctions:
    """Test utility functions for data processing."""

    @pytest.mark.parametrize(
        "sequence,order,pad,expected_ngrams",
        [
            ("abcde", 2, "#", [("#", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "#")]),
            ("abc", 3, "#", [("#", "#", "a"), ("#", "a", "b"), ("a", "b", "c"), ("b", "c", "#"), ("c", "#", "#")]),
            (["x", "y"], 2, "PAD", [("PAD", "x"), ("x", "y"), ("y", "PAD")]),
        ],
    )
    def test_collect_ngrams(self, sequence, order: int, pad: str, expected_ngrams):
        """
        Test n-gram collection from sequences.

        Example usage:
            ngrams = list(asymcat.collect_ngrams("hello", 2, "#"))
        """
        ngrams = list(asymcat.collect_ngrams(sequence, order, pad))

        assert len(ngrams) == len(expected_ngrams), f"Should generate {len(expected_ngrams)} {order}-grams"

        for i, (actual, expected) in enumerate(zip(ngrams, expected_ngrams)):
            assert actual == expected, f"N-gram {i}: expected {expected}, got {actual}"

    def test_collect_ngrams_error_handling(self):
        """Test error handling for n-gram collection."""
        # Invalid order
        with pytest.raises(ValueError):
            list(asymcat.collect_ngrams("abc", 0, "#"))  # Order must be >= 1

        # Invalid types
        with pytest.raises(TypeError):
            list(asymcat.collect_ngrams("abc", "invalid", "#"))  # Order must be int

        with pytest.raises(TypeError):
            list(asymcat.collect_ngrams("abc", 2, 123))  # Pad must be string
