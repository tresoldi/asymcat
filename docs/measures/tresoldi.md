# The Tresoldi measure

**`CatScorer.tresoldi()`** · directional · signed · range $(-\infty, \infty)$ ·
specialized

The Tresoldi measure is a directional association score specific to ASymCat. It was designed for scoring alignments,
such as grapheme–phoneme correspondences, where the question is how strongly one
symbol is associated with another *in a given direction*. It combines two
quantities: the pointwise mutual information (PMI) of the pair, which says how
far their co-occurrence departs from independence, and the conditional
probability of one symbol given the other.

## Definition

For a pair of symbols $(x, y)$, the score in the $x \to y$ direction is

$$
T(x \to y) = \operatorname{sign}\big(\mathrm{PMI}(x, y)\big)\,
\big|\mathrm{PMI}(x, y)\big|^{\,1 - P(y \mid x)}
$$

and the $y \to x$ score uses $P(x \mid y)$ in the exponent instead. Here

$$
\mathrm{PMI}(x, y) = \ln \frac{P(x, y)}{P(x)\,P(y)}, \qquad
P(y \mid x) = \frac{P(x, y)}{P(x)}.
$$

PMI is symmetric, so the two directions differ only through the exponent: the
scores coincide exactly when $P(y \mid x) = P(x \mid y)$, that is, when $x$ and
$y$ occur equally often.

In ASymCat:

- **PMI** uses the natural logarithm and the relative frequencies of the
  co-occurrences, without smoothing. A pair that never co-occurs is given the
  floor $P(x, y) = (1/n)^2$, with $n$ the number of co-occurrences, which makes
  its PMI strongly negative.
- **The conditional probability** is the one returned by `CatScorer.mle()`, so
  `smoothing_method="laplace"` or `"lidstone"` affects this term only.
- **The sign** is applied outside the power, so negative PMI values never meet a
  fractional exponent.

## How the exponent behaves

The exponent $1 - P(y \mid x)$ falls from 1 to 0 as $x$ becomes a better
predictor of $y$, and the score moves from $\mathrm{PMI}$ towards $\pm 1$:

| PMI | $P(y \mid x) = 0$ | $0.5$ | $0.9$ | $1$ |
|----:|------:|------:|------:|------:|
| 5.0 | 5.000 | 2.236 | 1.175 | 1.000 |
| 2.0 | 2.000 | 1.414 | 1.072 | 1.000 |
| 0.5 | 0.500 | 0.707 | 0.933 | 1.000 |
| −2.0 | −2.000 | −1.414 | −1.072 | −1.000 |

So:

- **Weak prediction** ($P(y \mid x)$ near 0): the score is close to the PMI.
- **Deterministic prediction** ($P(y \mid x) = 1$): the score is $\pm 1$ whatever
  the PMI. (If the PMI is exactly 0 as well, which requires $y$ to be the only
  symbol observed, the result is $0^0 = 1$.)
- **In between**, $|T|$ always lies between $|\mathrm{PMI}|$ and 1. For
  $|\mathrm{PMI}| > 1$ the score *decreases* towards 1 as the conditional
  probability rises; for $|\mathrm{PMI}| < 1$ it *increases* towards 1.
- **The sign** is that of the PMI: positive when the pair co-occurs more often
  than independence predicts, negative when less often.

A score above 1 in absolute value therefore requires both $|\mathrm{PMI}| > 1$
and an imperfect conditional probability, and values near $\pm 1$ can come from
a near-deterministic association or from a PMI near $\pm 1$. The score is not a
monotone measure of predictive strength. To rank pairs by how well one symbol
predicts the other, use `mle()` or `theil_u()` alongside it.

## Examples

These values come from ASymCat itself:

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("x", "a")] * 3 + [("y", "a")] + [("y", "b")] * 20 + [("z", "b")] * 10 + [("z", "c")] * 10
>>> scorer = CatScorer(pairs)
>>> scorer.mle()[("x", "a")]  # x always gives a; a comes from x 3 times in 4
(1.0, 0.75)
>>> [round(v, 3) for v in scorer.pmi()[("x", "a")]]
[2.398, 2.398]
>>> [round(v, 3) for v in scorer.tresoldi()[("x", "a")]]
[1.0, 1.244]
>>> [round(v, 3) for v in scorer.tresoldi()[("y", "b")]]
[0.949, 0.694]
>>> [round(v, 3) for v in scorer.tresoldi()[("y", "a")]]
[-0.66, -0.721]
```

- **`("x", "a")`**: $x$ always occurs with $a$, so $T(x \to a) = 1$. In the other
  direction, $P(x \mid a) = 0.75$ leaves part of the PMI:
  $2.398^{0.25} = 1.244$. The deterministic direction gets the *lower* score,
  because the PMI exceeds 1.
- **`("y", "b")`**: $P(b \mid y) = 0.952$, $P(y \mid b) = 0.667$ and
  $\mathrm{PMI} = 0.334$. Here the PMI is below 1, so the more predictive
  direction scores higher: $(0.949, 0.694)$.
- **`("y", "a")`**: the pair co-occurs less often than independence predicts
  ($\mathrm{PMI} = -0.647$), and both scores are negative: $(-0.660, -0.721)$.

## Significance and uncertainty

The resampling tools accept `"tresoldi"` like any other measure:

```pycon
>>> pvalues = scorer.permutation_pvalue("tresoldi", n_permutations=999, seed=0)
>>> round(pvalues[("x", "a")][0], 2)
0.26
>>> intervals = scorer.bootstrap_ci("tresoldi", n_bootstrap=999, seed=0)
>>> len(intervals[("x", "a")])  # (low, high) for each direction
2
```

The saturation at $\pm 1$ carries over to these tests. In the example above,
the deterministic direction $x \to a$ has a permutation p-value of 0.26.
Shuffled data often produce scores of 1 or more, so a perfect association does
not stand out. Test the conditional probability or Theil's U when the question
is whether one symbol predicts another.

## Relation to other measures

| Measure | Direction | Range | What it reports |
|---------|-----------|-------|-----------------|
| `pmi` | symmetric | $(-\infty, \infty)$ | departure of the pair from independence |
| `mle` | directional | $[0, 1]$ | conditional probability of one symbol given the other |
| `theil_u` | directional | $[0, 1]$ | share of the uncertainty about one variable removed by the other |
| `tresoldi` | directional | $(-\infty, \infty)$ | PMI pulled towards $\pm 1$ by the conditional probability |

## Limitations

- **Not normalized.** The scale depends on the base rates in the data. Compare
  scores within a dataset rather than across datasets.
- **Rare pairs.** PMI is unstable for low counts, and the exponent does not
  account for sample size. Use the resampling tools above to judge reliability.
- **Association, not causation.** The direction refers to which conditional
  probability is used, not to temporal or causal order.

## API

::: asymcat.scorer.CatScorer.tresoldi
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Church, K. W., & Hanks, P. (1990). Word association norms, mutual
  information, and lexicography. *Computational Linguistics*, 16(1), 22–29.
  [ACL Anthology J90-1003](https://aclanthology.org/J90-1003/)

The [References](../references.md) page lists the sources for every measure. To
cite ASymCat, and with it this measure, see [How to cite](../citing.md).
