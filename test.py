# encoding: utf-8

# Import 3rd party libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Import own libraries
import catcoocc

#toy_data = catcoocc.read_sequences("docs/cmudict.tsv")
#toy_data = catcoocc.read_sequences("docs/toy.tsv")
toy_data = catcoocc.read_sequences("docs/mushroom-small.tsv")
toy_co = catcoocc.collect_cooccs(toy_data)

print(len(toy_data), len(toy_co))

sc = catcoocc.scorer.CatScorer(toy_co)
print("mle")
mle = sc.mle()
print("pmi")
pmi = sc.pmi()
print("npmi")
npmi = sc.pmi(True)
print("chi2")
chi2 = sc.chi2()
print("chi2_ns")
chi2_ns = sc.chi2(False)
print("cramersv")
cramersv = sc.cramers_v()
print("cramersv_ns")
cramersv_ns = sc.cramers_v(False)
print("fisher")
fisher = sc.fisher()
print("theil_u")
theil_u = sc.theil_u()
print("catcoocc_i")
catcoocc_i = sc.catcoocc_i()
print("catcoocc_ii")
catcoocc_ii = sc.catcoocc_ii()

for pair in sorted(sc.obs):
    print("--", pair)
    print("  mle        \t%0.4f %0.4f" % mle[pair])
    print("  pmi        \t%0.4f %0.4f" % pmi[pair])
    print("  npmi       \t%0.4f %0.4f" % npmi[pair])
    print("  chi2       \t%0.4f %0.4f" % chi2[pair])
    print("  chi2_ns    \t%0.4f %0.4f" % chi2_ns[pair])
    print("  cramersv   \t%0.4f %0.4f" % cramersv[pair])
    print("  cramersv_ns\t%0.4f %0.4f" % cramersv_ns[pair])
    print("  fisher     \t%0.4f %0.4f" % fisher[pair])
    print("  theil_u    \t%0.4f %0.4f" % theil_u[pair])
    print("  catcoocc_i \t%0.4f %0.4f" % catcoocc_i[pair])
    print("  catcoocc_ii\t%0.4f %0.4f" % catcoocc_ii[pair])

xy, yx, alpha_x, alpha_y = catcoocc.scorer.scorer2matrices(catcoocc_i)
print(xy)
print(yx)


# scaling
#mle_minmax = catcoocc.scorer.scale_scorer(mle, method="minmax")
#mle_mean = catcoocc.scorer.scale_scorer(mle, method="mean")
#mle_stdev = catcoocc.scorer.scale_scorer(mle, method="stdev")
#for pair in sorted(sc.obs):
#    print("==", pair, mle[pair], mle_minmax[pair], mle_mean[pair], mle_stdev[pair])
