"""
Tests for the entropy estimators behind asymcat's information-theoretic measures.

``CatScorer(..., entropy_estimator=...)`` and the scalar helpers in
:mod:`asymcat.scorer` accept ``"plugin"`` (the default, unchanged) and
freqprob's bias-corrected ``"chao_shen"`` and ``"nsb"`` estimators. These tests
pin the contract: the default is untouched, the corrected path delegates to
freqprob, each pair's subset is reconstructed exactly, and the configuration
survives resampling.
"""

import math
from collections import Counter

import freqprob
import pytest
import scipy.stats as ss

import asymcat
from asymcat import correlation, scorer

ESTIMATED = ["chao_shen", "nsb"]
MEASURES = ["theil_u", "cond_entropy", "mutual_information", "normalized_mutual_information"]
SCALAR_HELPERS = [
    scorer.conditional_entropy,
    scorer.compute_theil_u,
    scorer.compute_mutual_information,
    scorer.compute_normalized_mutual_information,
]

# Sparse co-occurrences with several singletons, as in small aligned datasets.
SPARSE_COOCCS = [
    ("a", "x"),
    ("a", "x"),
    ("a", "y"),
    ("b", "y"),
    ("b", "z"),
    ("c", "z"),
    ("c", "x"),
    ("d", "w"),
    ("a", "w"),
    ("b", "x"),
]


def _subset(cooccs, x, y):
    """The co-occurrences that ``CatScorer`` scores the pair ``(x, y)`` over."""
    return [pair for pair in cooccs if pair[0] == x or pair[1] == y]


class TestValidation:
    """Estimator names are validated up front."""

    def test_unknown_estimator_is_rejected_by_the_scorer(self):
        with pytest.raises(ValueError, match="chao_shen"):
            asymcat.CatScorer(SPARSE_COOCCS, entropy_estimator="miller-madow")

    def test_estimator_name_is_case_insensitive(self):
        assert asymcat.CatScorer(SPARSE_COOCCS, entropy_estimator="Chao_Shen").entropy_estimator == "chao_shen"

    @pytest.mark.parametrize("helper", SCALAR_HELPERS, ids=lambda f: f.__name__)
    def test_unknown_estimator_is_rejected_by_scalar_helpers(self, helper):
        """Validation happens before any early return, even for empty input."""
        with pytest.raises(ValueError, match="Unsupported entropy estimator"):
            helper([], [], estimator="miller-madow")


class TestPluginDefault:
    """The default keeps the existing maximum-likelihood behavior."""

    def test_default_estimator_is_plugin(self):
        assert asymcat.CatScorer(SPARSE_COOCCS).entropy_estimator == "plugin"

    @pytest.mark.parametrize("measure", MEASURES)
    def test_explicit_plugin_matches_the_default(self, measure):
        default = getattr(asymcat.CatScorer(SPARSE_COOCCS), measure)()
        explicit = getattr(asymcat.CatScorer(SPARSE_COOCCS, entropy_estimator="plugin"), measure)()
        assert default == explicit

    def test_plugin_entropy_matches_scipy(self):
        counts = Counter({"a": 5, "b": 3, "c": 1, "d": 1})
        assert scorer._entropy(counts) == pytest.approx(ss.entropy(list(counts.values())))


class TestDelegation:
    """The corrected estimates come from freqprob."""

    COUNTS = {"a": 4, "b": 1, "c": 1, "d": 2}

    def test_chao_shen_delegates_to_freqprob(self):
        assert scorer._entropy(self.COUNTS, "chao_shen") == pytest.approx(freqprob.chao_shen_entropy(self.COUNTS))

    def test_nsb_delegates_to_freqprob_with_the_alphabet_size(self):
        expected = freqprob.nsb_entropy(self.COUNTS, bins=7)
        assert scorer._entropy(self.COUNTS, "nsb", bins=7) == pytest.approx(expected)

    @pytest.mark.parametrize("estimator", ["plugin", *ESTIMATED])
    def test_entropy_ignores_labels_order_and_zero_counts(self, estimator):
        """Only the multiset of positive counts matters, which the cache relies on."""
        original = scorer._entropy({"a": 3, "b": 1, "c": 2}, estimator)
        relabeled = scorer._entropy({"r": 2, "q": 3, "s": 1, "t": 0}, estimator)
        assert relabeled == pytest.approx(original)

    def test_chao_shen_corrects_an_all_singleton_sample(self):
        """Requires the freqprob fix: zero coverage no longer collapses to 0."""
        assert scorer._entropy({"a": 1, "b": 1, "c": 1}, "chao_shen") > math.log(3)


class TestScorerSubsets:
    """``CatScorer`` reconstructs each pair's subset exactly."""

    @pytest.mark.parametrize("measure", MEASURES)
    def test_chao_shen_matches_scalar_helpers_on_each_subset(self, measure):
        scored = getattr(asymcat.CatScorer(SPARSE_COOCCS, entropy_estimator="chao_shen"), measure)()

        for (x, y), (value_yx, value_xy) in scored.items():
            subset = _subset(SPARSE_COOCCS, x, y)
            xs, ys = [pair[0] for pair in subset], [pair[1] for pair in subset]
            if measure == "theil_u":
                expected = (scorer.compute_theil_u(ys, xs, "chao_shen"), scorer.compute_theil_u(xs, ys, "chao_shen"))
            elif measure == "cond_entropy":
                expected = (
                    scorer.conditional_entropy(ys, xs, "chao_shen"),
                    scorer.conditional_entropy(xs, ys, "chao_shen"),
                )
            elif measure == "mutual_information":
                mi = scorer.compute_mutual_information(xs, ys, "chao_shen")
                expected = (mi, mi)
            else:
                nmi = scorer.compute_normalized_mutual_information(xs, ys, "chao_shen")
                expected = (nmi, nmi)

            assert (value_yx, value_xy) == pytest.approx(expected), (x, y)

    def test_nsb_uses_the_alphabet_sizes_of_the_data(self):
        """Marginals use |X| and |Y| symbols; the joint uses the |X| + |Y| - 1 subset cells."""
        cond_entropy = asymcat.CatScorer(SPARSE_COOCCS, entropy_estimator="nsb").cond_entropy()
        n_x = len({pair[0] for pair in SPARSE_COOCCS})
        n_y = len({pair[1] for pair in SPARSE_COOCCS})

        subset = _subset(SPARSE_COOCCS, "a", "x")
        h_xy = freqprob.nsb_entropy(Counter(subset), bins=n_x + n_y - 1)
        h_x = freqprob.nsb_entropy(Counter(pair[0] for pair in subset), bins=n_x)
        h_y = freqprob.nsb_entropy(Counter(pair[1] for pair in subset), bins=n_y)

        assert cond_entropy[("a", "x")] == pytest.approx((h_xy - h_x, h_xy - h_y))


class TestConventions:
    """Degenerate subsets keep the plug-in conventions under every estimator."""

    @pytest.mark.parametrize("estimator", ESTIMATED)
    def test_single_symbol_subset(self, estimator):
        """A pair whose subset holds a single cell: U is 1.0 and NMI is 0.0."""
        cooccs = [("a", "b")] * 3 + [("c", "d")] * 2
        scored = asymcat.CatScorer(cooccs, entropy_estimator=estimator)

        assert scored.theil_u()[("a", "b")] == (1.0, 1.0)
        assert scored.normalized_mutual_information()[("a", "b")] == (0.0, 0.0)


class TestResampling:
    """Significance replicates are scored with the same estimator."""

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [("bootstrap_ci", {"n_bootstrap": 3}), ("permutation_pvalue", {"n_permutations": 3})],
    )
    def test_estimator_propagates_to_replicates(self, mocker, method, kwargs):
        scored = asymcat.CatScorer(SPARSE_COOCCS, entropy_estimator="chao_shen")
        spy = mocker.spy(scorer.CatScorer, "__init__")

        getattr(scored, method)("theil_u", seed=0, **kwargs)

        estimators = [call.kwargs.get("entropy_estimator") for call in spy.call_args_list]
        assert estimators == ["chao_shen"] * 3


class TestCorrelationWrappers:
    """The two-series wrappers forward the estimator."""

    SERIES_X = ["a", "a", "b", "b", "c", "a", "d"]
    SERIES_Y = ["x", "y", "y", "z", "z", "x", "w"]

    @pytest.mark.parametrize("estimator", ESTIMATED)
    def test_theil_u_forwards_the_estimator(self, estimator):
        expected = scorer.compute_theil_u(self.SERIES_X, self.SERIES_Y, estimator)
        assert correlation.theil_u(self.SERIES_X, self.SERIES_Y, estimator) == pytest.approx(expected)

    @pytest.mark.parametrize("estimator", ESTIMATED)
    def test_conditional_entropy_forwards_the_estimator(self, estimator):
        expected = scorer.conditional_entropy(self.SERIES_X, self.SERIES_Y, estimator)
        assert correlation.conditional_entropy(self.SERIES_X, self.SERIES_Y, estimator) == pytest.approx(expected)
