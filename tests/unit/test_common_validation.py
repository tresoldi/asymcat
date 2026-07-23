"""
Tests for input validation and error handling in :mod:`asymcat.common`.

These tests exercise the defensive validation paths of the data-loading and
pre-computation helpers, ensuring that malformed input raises clear, typed
errors rather than failing obscurely deeper in the call stack.
"""

import pytest

from asymcat import common


class TestCollectAlphabetsValidation:
    """Validation paths for ``collect_alphabets``."""

    def test_rejects_non_list(self):
        with pytest.raises(TypeError, match="Expected list"):
            common.collect_alphabets("not a list")  # type: ignore[arg-type]

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Empty co-occurrence list"):
            common.collect_alphabets([])

    def test_rejects_malformed_pair(self):
        with pytest.raises(ValueError, match="Invalid co-occurrence"):
            common.collect_alphabets([("a", "x"), ("only-one",)])


class TestCollectNgramsValidation:
    """Validation paths for ``collect_ngrams``."""

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError, match="Expected list, tuple, or string"):
            list(common.collect_ngrams(123, 2, "#"))  # type: ignore[arg-type]


class TestCollectCooccsValidation:
    """Validation paths for ``collect_cooccs``."""

    def test_rejects_non_list(self):
        with pytest.raises(TypeError, match="Expected list"):
            common.collect_cooccs("not a list")  # type: ignore[arg-type]

    def test_rejects_non_string_pad(self):
        with pytest.raises(TypeError, match="Pad symbol must be a string"):
            common.collect_cooccs([[["a"], ["x"]]], pad=123)  # type: ignore[arg-type]

    def test_rejects_invalid_order(self):
        with pytest.raises(ValueError, match="Order must be a positive integer"):
            common.collect_cooccs([[["a"], ["x"]]], order=0)

    def test_rejects_non_sequence_element(self):
        with pytest.raises(TypeError, match="must be lists, tuples, or strings"):
            common.collect_cooccs([[123, 456]])  # type: ignore[list-item]


class TestCollectObservationsValidation:
    """Validation paths for ``collect_observations``."""

    def test_rejects_non_list(self):
        with pytest.raises(TypeError, match="Expected list"):
            common.collect_observations("not a list")  # type: ignore[arg-type]

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Empty co-occurrence list"):
            common.collect_observations([])

    def test_rejects_malformed_pair(self):
        with pytest.raises(ValueError, match="Invalid co-occurrence"):
            common.collect_observations([("a", "x"), "bad"])


class TestReadSequencesValidation:
    """Validation paths for ``read_sequences``."""

    def test_rejects_non_string_filename(self):
        with pytest.raises(TypeError, match="Filename must be a string"):
            common.read_sequences(123)  # type: ignore[arg-type]

    def test_rejects_non_list_cols(self, tmp_path):
        data_file = tmp_path / "data.tsv"
        data_file.write_text("A\tB\na b\tc d\n", encoding="utf-8")
        with pytest.raises(TypeError, match="Columns must be a list"):
            common.read_sequences(str(data_file), cols="A")  # type: ignore[arg-type]

    def test_rejects_non_string_delimiter(self, tmp_path):
        data_file = tmp_path / "data.tsv"
        data_file.write_text("A\tB\na b\tc d\n", encoding="utf-8")
        with pytest.raises(TypeError, match="Delimiters must be strings"):
            common.read_sequences(str(data_file), col_delim=123)  # type: ignore[arg-type]

    def test_missing_column_raises(self, tmp_path):
        data_file = tmp_path / "data.tsv"
        data_file.write_text("A\tB\na b\tc d\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Column not found"):
            common.read_sequences(str(data_file), cols=["Nonexistent"])

    def test_no_complete_pairs_raises(self, tmp_path):
        # Header plus single-column rows -> no row has two sequences.
        data_file = tmp_path / "single.tsv"
        data_file.write_text("Header\nonly_one_column\nanother\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No complete sequence pairs"):
            common.read_sequences(str(data_file))


class TestReadPaMatrixValidation:
    """Validation paths for ``read_pa_matrix``."""

    def test_rejects_non_string_filename(self):
        with pytest.raises(TypeError, match="Filename must be a string"):
            common.read_pa_matrix(123)  # type: ignore[arg-type]

    def test_rejects_non_string_delimiter(self, tmp_path):
        data_file = tmp_path / "pa.tsv"
        data_file.write_text("ID\tsp\nloc\t1\n", encoding="utf-8")
        with pytest.raises(TypeError, match="Delimiter must be a string"):
            common.read_pa_matrix(str(data_file), delimiter=123)  # type: ignore[arg-type]

    def test_missing_id_column_raises(self, tmp_path):
        data_file = tmp_path / "no_id.tsv"
        data_file.write_text("sp_a\tsp_b\n1\t0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required 'ID' column"):
            common.read_pa_matrix(str(data_file))

    def test_empty_id_value_raises(self, tmp_path):
        data_file = tmp_path / "empty_id.tsv"
        data_file.write_text("ID\tsp_a\n\t1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty ID value"):
            common.read_pa_matrix(str(data_file))

    def test_invalid_presence_value_raises(self, tmp_path):
        data_file = tmp_path / "bad_value.tsv"
        data_file.write_text("ID\tsp_a\nloc1\t5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid presence-absence value"):
            common.read_pa_matrix(str(data_file))
