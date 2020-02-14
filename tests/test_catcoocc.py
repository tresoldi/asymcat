#!/usr/bin/env python3
# encoding: utf-8

"""
test_catcoocc
=============

Tests for the `catcoocc` package.
"""

# Import system libraries
import unittest

# Import 3rd party libraries
import numpy as np

# Import the library being test
import catcoocc


class TestCoocc(unittest.TestCase):
    """
    Class for `cooccurran` tests.
    """

    # Small sample for text, derived from CMU (first entry is manual)
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

    def test_compute(self):
        # Compute cooccs
        cooccs = catcoocc.collect_cooccs(self.data_cmu)

        # Collect lengths
        cooccs_A = [coocc for coocc in cooccs if coocc[0] == "A"]
        cooccs_Ll = [
            coocc for coocc in cooccs if coocc[0] == "L" and coocc[1] == "l"
        ]

        # Assert length as proxy for right collection
        assert len(cooccs) == 879
        assert len(cooccs_A) == 92
        assert len(cooccs_Ll) == 14

    def test_scorers(self):
        # Compute cooccs and build scorer
        cooccs = catcoocc.collect_cooccs(self.data_cmu)
        scorer = catcoocc.scorer.CatScorer(cooccs)

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
        catcoocc_i = scorer.catcoocc_i()
        catcoocc_ii = scorer.catcoocc_ii()

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
                5.680172609017068,  # catcoocc_i x>y
                5.680172609017068,  # catcoocc_i y>x
                3462.523968980434,  # catcoocc_ii x>y
                3462.523968980434,  # catcoocc_ii y>x
            ),
            ("A", "b"): (
                0.06521739130434782,  # mle x>y
                0.11320754716981132,  # mle y>x
                0.07846387631207004,  # pmi x>y
                0.07846387631207004,  # pmi y>x
                0.015733602612959818,  # npmi x>y
                0.015733602612959818,  # npmi y>x
                0.0004776025004836434,  # chi2 x>y
                0.0004776025004836434,  # chi2 y>x
                0.043927505580845905,  # chi2_ns x>y
                0.043927505580845905,  # chi2_ns y>x
                0.0,  # cramersv x>y
                0.0,  # cramersv y>x
                0.0,  # cramersv_ns x>y
                0.0,  # cramersv_ns y>x
                1.0984661058881742,  # fisher x>y
                1.0984661058881742,  # fisher y>x
                0.6593343352835512,  # theil_u x>y
                0.5997312299898745,  # theil_u y>x
                0.05173392773198948,  # catcoocc_i x>y
                0.04705723705041114,  # catcoocc_i y>x
                3.7474543524283964e-05,  # catcoocc_ii x>y
                3.7474543524283964e-05,  # catcoocc_ii y>x
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
                0.6169022357243095,  # theil_u x>y
                0.54447314456517,  # theil_u y>x
                0.7735930530343602,  # catcoocc_i x_y
                0.682767248695174,  # catcoocc_i y>x
                11.506888541379661,  # catcoocc_ii x>y
                11.506888541379661,  # catcoocc_ii y>x
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
                0.7381300186001525,  # theil_u x>y
                0.6937287111206101,  # theil_u y>x
                -4.799904537544586,  # catcoocc_i x>y
                -4.511172157240979,  # catcoocc_i y>x
                -0.6095830534614555,  # catcoocc_ii x>y
                -0.6095830534614555,  # catcoocc_ii y>x
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
                + catcoocc_i[pair]
                + catcoocc_ii[pair]
            )

            assert np.allclose(vals, ref, rtol=1e-05, atol=1e-08)

    def test_readers(self):
        # Read a sequences file
        cmu_file = catcoocc.RESOURCE_DIR / "cmudict.tsv"
        cmu = catcoocc.read_sequences(cmu_file.as_posix())

        # Read presence/absence matrix
        finches_file = catcoocc.RESOURCE_DIR / "galapagos.tsv"
        finches = catcoocc.read_pa_matrix(finches_file.as_posix())

        # For assertion, just check length
        assert len(cmu) == 134373
        assert len(finches) == 447


if __name__ == "__main__":
    # Explicitly creating and running test suite allows to profile
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoocc)
    unittest.TextTestRunner(verbosity=2).run(suite)
