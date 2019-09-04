# Categorical Co-occurrences

(introduction)

```python
# Import 3rd party libraries
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Import own libraries
import catcoocc
from catcoocc.scorer import CatScorer
```

Lorem ipsum

```python

toy_data = catcoocc.read_sequences("docs/cmudict.tsv")
toy_cooccs = catcoocc.collect_cooccs(toy_data)

print(len(toy_data), len(toy_cooccs))

scorer = catcoocc.scorer.CatScorer(toy_cooccs)
mle = scorer.mle()
pmi = scorer.pmi()
npmi = scorer.pmi(True)
chi2 = scorer.chi2()
chi2_ns = scorer.chi2(False)
cramersv = scorer.cramers_v()
cramersv_ns = scorer.cramers_v(False)
#fisher = scorer.fisher()
theil_u = scorer.theil_u()
catcoocc_i = scorer.catcoocc_i()
catcoocc_ii = scorer.catcoocc_ii()


for pair in sorted(scorer.obs):
    print("--", pair)
    print("  mle        \t%0.4f %0.4f" % mle[pair])
    print("  pmi        \t%0.4f %0.4f" % pmi[pair])
    print("  npmi       \t%0.4f %0.4f" % npmi[pair])
    print("  chi2       \t%0.4f %0.4f" % chi2[pair])
    print("  chi2_ns    \t%0.4f %0.4f" % chi2_ns[pair])
    print("  cramersv   \t%0.4f %0.4f" % cramersv[pair])
    print("  cramersv_ns\t%0.4f %0.4f" % cramersv_ns[pair])
#    print("  fisher     \t%0.4f %0.4f" % fisher[pair])
    print("  theil_u    \t%0.4f %0.4f" % theil_u[pair])
    print("  catcoocc_i \t%0.4f %0.4f" % catcoocc_i[pair])
    print("  catcoocc_ii\t%0.4f %0.4f" % catcoocc_ii[pair])
```

And now more

```python

def plot_scorer(scorer, alpha_x, alpha_y, title=None, figsize=(25, 25)):
    if not title:
        title = ""

    matrix = pd.DataFrame(scorer, index=alpha_y, columns=alpha_x)

    sns.set(font_scale=2, font="FreeMono")
    plt.figure(figsize=figsize)

    ax = plt.subplot(111)
    sns.heatmap(matrix, annot=True, fmt='.2f', linewidths=.5, center=0, ax=ax).set_title(title, fontsize=100)

xy, yx, alpha_x, alpha_y = catcoocc.scorer.scorer2matrices(catcoocc_ii)
plot_scorer(xy, alpha_x, alpha_y, "x->y", (50, 50))


```

another

```python
plot_scorer(yx, alpha_y, alpha_x, "y->x", (50, 50))
```

leftover

```python

print("ok")
print(dir(catcoocc))



np.random.seed(0)
sns.set(font_scale=2.5)
uniform_data = np.random.rand(10, 12)

plt.figure(figsize=(25, 25))
ax = plt.subplot(111)
sns.heatmap(uniform_data,
annot=True,
linewidths=.5, center=0, ax=ax)
```
