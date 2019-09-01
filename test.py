# encoding: utf-8

import cooccurran

from collections import Counter
import math
import scipy.stats as ss



def test():
    cmu = cooccurran.dataio.read_sequences("docs/cmudict.tsv")
    toy = cooccurran.dataio.read_sequences("docs/toy.tsv")
    finches = cooccurran.dataio.read_pa_matrix("docs/galapagos.tsv")

    print('cmu', len(cmu))
    print('toy', len(toy))
    print('finches', len(finches))


    print(toy)

    toy_co = cooccurran.measures.collect_cooccs(toy)
    toy_obs = cooccurran.measures.collect_observations(toy_co)

    print('toy_co', len(toy_co))
    print(toy_co[:5])
    print('toy_obs', len(toy_obs))
    print(toy_obs)


    toy_fs = cooccurran.measures.frequency_scorer(toy_obs)
    toy_chi2ss = cooccurran.measures.chi2_scorer(toy_obs, True)
    toy_chi2ns = cooccurran.measures.chi2_scorer(toy_obs, False)
    toy_cVss = cooccurran.measures.cramers_v_scorer(toy_obs, True)
    toy_cVns = cooccurran.measures.cramers_v_scorer(toy_obs, False)
    toy_fe = cooccurran.measures.fisher_exact_scorer(toy_obs)
    toy_pmis = cooccurran.measures.pmi_scorer(toy_obs, False)
    toy_pmins = cooccurran.measures.pmi_scorer(toy_obs, True)
    toy_theilus = cooccurran.measures.theil_u_scorer(toy_co)

    for pair in sorted(toy_obs):
        buf = toy_fs[pair] + \
                toy_chi2ss[pair] + \
                toy_chi2ns[pair] + \
                toy_cVss[pair] + \
                toy_cVns[pair] + \
                toy_fe[pair] + \
                toy_pmis[pair] + \
                toy_pmins[pair] + \
                toy_theilus[pair]
       
        buf = ["%0.2f" % v if v is not None else "None" for v in buf]

        print(pair, buf)

test()
