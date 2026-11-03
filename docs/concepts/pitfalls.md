# Pitfalls

Association measures are easy to compute and easy to misread. This page
collects the surprises that come up most often, with the
[example data](notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
```

## Symmetric measures return two equal numbers

Every measure returns a tuple $(x \to y,\ y \to x)$, but only some measures are
directional. PMI, mutual information, the statistical tests and the Jaccard
index repeat one value:

```pycon
>>> [round(v, 3) for v in scorer.pmi()[("c", "s")]]
[0.405, 0.405]
```

Equal values from a symmetric measure say nothing about the symmetry of the
relationship. [Directional association](directional-association.md) lists which
measures are which.

## High scores can mean avoidance

The information-theoretic measures and Goodman–Kruskal λ are computed on the
[pair subset](notation.md#the-pair-subset) and measure **predictability**,
whatever its source. `k` and /s/ never occur together, yet:

```pycon
>>> scorer.theil_u()[("k", "s")]
(1.0, 1.0)
>>> scorer.mle()[("k", "s")]
(0.0, 0.0)
```

In their subset, `k` is always /k/ and /s/ is always `c`, so each variable
predicts the other perfectly: the pair is perfectly predictable because the two
symbols exclude each other. Before reading a high Theil's U, mutual information
or λ as attraction, check the sign of the [PMI](../measures/pmi.md) or the
[conditional probability](../measures/mle.md). The test statistics (chi-square,
G²) are unsigned in the same way.

## Degenerate subsets fall back on conventions

When one variable takes a single value in the pair subset, there is nothing
left to predict, and the measures fall back on a convention: Theil's U and λ
score that direction as 1.

```pycon
>>> [round(u, 3) for u in scorer.theil_u()[("q", "k")]]
[1.0, 0.0]
>>> scorer.goodman_kruskal_lambda()[("q", "k")]
(1.0, 0.0)
```

In the subset of `q` and /k/, the sound is always /k/, so knowing the letter
predicts it trivially. The 1 is not evidence about `q` in particular. Pairs whose subset is small or degenerate are
best judged with the conditional probability and its count.

## Unobserved pairs are scored too

Each measure returns every pair in $\mathcal{X} \times \mathcal{Y}$, including
pairs that never co-occur. Their scores come from conventions rather than
evidence: 0 for the conditional probability, a floor of $(1/n)^2$ for the joint
probability in PMI, and 0 or $\infty$ for the odds ratio. Filter them with
`scorer.obs[pair]["11"] > 0` when only observed pairs matter.

```pycon
>>> observed = [pair for pair in scorer.obs if scorer.obs[pair]["11"] > 0]
>>> sorted(observed)
[('c', 'k'), ('c', 's'), ('k', 'k'), ('q', 'k')]
```

## Infinite values break scaling

Fisher's odds ratio is $\infty$ or 0 whenever a cell of the 2×2 table is empty,
which is common with categorical data. Scaling a scorer that holds an infinite
value produces `nan`:

```pycon
>>> from asymcat.scorer import scale_scorer
>>> scorer.fisher()[("c", "s")]
(inf, inf)
>>> scale_scorer(scorer.fisher())[("c", "s")]
(nan, nan)
```

Rank such measures directly, or use their p-values.

## Small samples and the chi-square approximation

The $\chi^2$ and $G^2$ p-values rely on a large-sample approximation. With few
co-occurrences they disagree with each other and with the exact test:

```pycon
>>> round(scorer.chi2_pvalue()[("c", "k")][0], 2)
0.3
>>> round(scorer.log_likelihood_ratio_pvalue()[("c", "k")][0], 2)
0.05
>>> round(scorer.fisher_pvalue()[("c", "k")][0], 2)
0.23
```

`chi2` uses Yates' continuity correction, which makes it conservative; G² has
no correction. Use Fisher's exact test, or a permutation test, when counts are
small. The bias-corrected [Cramér's V](../measures/chi2.md) is 0 here for every
pair: on 15 co-occurrences, no association exceeds what chance would produce.

## Negative zeros and small negative values

Entropy differences computed in floating point can come out as `-0.0` or as a
tiny negative number where the exact value is 0. With the bias-corrected
entropy estimators (`entropy_estimator="chao_shen"` or `"nsb"`), Theil's U and
mutual information can also be slightly negative on small subsets; this means
no detectable association, not a negative one.

## Collecting from sequences multiplies the evidence

[`collect_cooccs`](../how-to/prepare-data.md#aligned-sequences) without an
`order` pairs **every** element of one sequence with **every** element of the
other, so a pair of words of length 5 contributes 25 co-occurrences. This is
appropriate when position does not matter, as in two lists of attributes of
the same item, but for aligned sequences use `order=1`, which pairs elements
position by position.

In both cases, the co-occurrences from one sequence pair are not independent
observations. The statistical tests and the resampling tools treat them as
exchangeable, so their p-values are somewhat too small, and their confidence
intervals too narrow, when sequences are long.

## `pmi_smoothed` with the default smoothing

`pmi_smoothed` estimates the probabilities with the scorer's
`smoothing_method`. With the default, `"mle"`, there is no smoothing, and
unobserved pairs get a near-zero probability from FreqProb and a very large
negative score:

```pycon
>>> round(scorer.pmi_smoothed()[("q", "s")][0])
-20
```

Create the scorer with `smoothing_method="laplace"` or `"lidstone"` to use it
(see [Work with sparse data](../how-to/sparse-data.md)).

## Direction is not causation

"$x \to y$" means that $x$ is used to predict $y$. Both directions are computed
from the same co-occurrence counts, and neither says anything about which
variable comes first or influences the other.
