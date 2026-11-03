# Work with sparse data

Categorical data is usually sparse: a few pairs are frequent and most are rare
or unobserved. Raw relative frequencies are unreliable for the rare pairs, and
the plug-in entropy is biased on small samples. A `CatScorer` has two options
for this, one for probabilities and one for entropies.

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
```

## Smooth the probabilities

`smoothing_method` controls the probabilities behind
[`mle`](../measures/mle.md), [`tresoldi`](../measures/tresoldi.md) and
[`pmi_smoothed`](../measures/pmi.md). The estimates come from
[FreqProb](https://freqprob.tresoldi.org/):

| `smoothing_method` | Estimate of $P(y \mid x)$ |
|--------------------|---------------------------|
| `"mle"` (default) | $c(x, y) / c(x)$ |
| `"laplace"` | $\big(c(x, y) + 1\big) / \big(c(x) + 2\big)$ |
| `"lidstone"` | $\big(c(x, y) + \gamma\big) / \big(c(x) + 2\gamma\big)$, $\gamma$ = `smoothing_alpha` |

```pycon
>>> for method in ("mle", "laplace", "lidstone"):
...     mle = CatScorer(pairs, smoothing_method=method, smoothing_alpha=0.5).mle()
...     print(method, [round(p, 3) for p in mle[("q", "k")]])
mle [1.0, 0.182]
laplace [0.75, 0.231]
lidstone [0.833, 0.208]
```

With only two observations of `q`, Laplace smoothing pulls $P(k \mid q)$ from 1
to 0.75; Lidstone with $\gamma = 0.5$ pulls it less. Smoothing matters most for
rare symbols and leaves frequent ones nearly unchanged.

`pmi_smoothed` uses the same setting for the joint and marginal probabilities:

```pycon
>>> laplace = CatScorer(pairs, smoothing_method="laplace")
>>> [round(v, 3) for v in laplace.pmi_smoothed()[("c", "s")]]
[0.302, 0.302]
```

## Correct the entropies

`entropy_estimator` controls the entropies behind
[`theil_u`](../measures/theil-u.md),
[`cond_entropy`](../measures/conditional-entropy.md) and
[mutual information](../measures/mutual-information.md). The plug-in estimate
underestimates entropy on small samples, because it ignores values that were
possible but not observed. FreqProb provides two corrections:

- **`"chao_shen"`** (Chao & Shen, 2003) corrects for unseen values using the
  number of values seen once.
- **`"nsb"`** (Nemenman, Shafee & Bialek, 2002) is a Bayesian estimator that
  takes the number of possible values into account; ASymCat gives it the
  alphabet sizes of the data.

```pycon
>>> for estimator in ("plugin", "chao_shen", "nsb"):
...     theil_u = CatScorer(pairs, entropy_estimator=estimator).theil_u()
...     print(estimator, [round(u, 3) for u in theil_u[("c", "k")]])
plugin [0.226, 0.152]
chao_shen [0.225, 0.144]
nsb [0.301, 0.197]
```

The corrected estimates are not confined to $[0, 1]$; a slightly negative
value on a small subset means no detectable association. The NSB estimator is
slower than the other two, which matters with the
[resampling tools](significance.md).

## Which to use

- With plenty of data for every symbol, the defaults are fine.
- Use `"laplace"` or `"lidstone"` when conditional probabilities of rare
  symbols matter, and always with `pmi_smoothed`.
- Use `"chao_shen"` or `"nsb"` when the information-theoretic measures are
  computed on small pair subsets.
- Whatever the setting, check rare pairs with
  [`bootstrap_ci`](significance.md#confidence-intervals): smoothing makes
  estimates more stable, not more certain.
