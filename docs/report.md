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
```

Lorem ipsum

```python

toy_data = catcoocc.dataio.read_sequences("docs/cmudict.tsv")
toy_co = catcoocc.dataio.get_cooccs(toy_data)
toy_obs = catcoocc.dataio.get_observations(toy_co)

print(len(toy_data), len(toy_co), len(toy_obs))

mle = catcoocc.scorers.mle_scorer(toy_co)
xy, yx, alpha_x, alpha_y = catcoocc.scorers.scorer2matrix(mle)

#for pair in mle:
#    print(pair, mle[pair])

print('x', alpha_x)
print('y', alpha_y)



def plot_scorer(scorer, alpha_x, alpha_y, title=None):
    if not title:
        title = ""

    matrix = pd.DataFrame(scorer, index=alpha_y, columns=alpha_x)

    sns.set(font_scale=2.5)
    plt.figure(figsize=(25, 25))

    ax = plt.subplot(111)
    sns.heatmap(matrix, annot=True, linewidths=.5, center=0, ax=ax).set_title(title, fontsize=100)

plot_scorer(xy, alpha_x, alpha_y, "x->y")


```

another

```python
plot_scorer(yx, alpha_y, alpha_x, "y->x")
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
