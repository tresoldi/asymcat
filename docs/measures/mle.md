# Conditional probability (`mle`)

**`CatScorer.mle()`** · directional · range $[0, 1]$ · probabilistic

The conditional probability is the most direct directional measure: how often
$y$ occurs when $x$ does, and how often $x$ occurs when $y$ does. It is the
natural first measure to look at, and the one the others are best compared
with.

## Definition

$$
\texttt{mle}(x, y) = \big(P(y \mid x),\ P(x \mid y)\big)
= \left(\frac{c(x, y)}{c(x)},\ \frac{c(x, y)}{c(y)}\right)
$$

The name refers to the maximum-likelihood estimate, which is what these
relative frequencies are. With a smoothing method, the estimates are instead
computed by [FreqProb](https://freqprob.tresoldi.org/) on the two-outcome
distribution "$y$ or not $y$" among the co-occurrences of $x$ (and likewise in
the other direction):

| `smoothing_method` | $P(y \mid x)$ |
|--------------------|---------------|
| `"mle"` (default) | $c(x, y) / c(x)$ |
| `"laplace"` | $\big(c(x, y) + 1\big) / \big(c(x) + 2\big)$ |
| `"lidstone"` | $\big(c(x, y) + \gamma\big) / \big(c(x) + 2\gamma\big)$, with $\gamma$ = `smoothing_alpha` |

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> mle = CatScorer(pairs).mle()
>>> mle[("q", "k")]
(1.0, 0.18181818181818182)
>>> [round(p, 3) for p in mle[("c", "k")]]
[0.6, 0.545]
>>> mle[("q", "s")]
(0.0, 0.0)
```

`q` is always /k/, but only 2 of the 11 /k/ are spelled `q`. `c` is /k/ six
times in ten, and 6 of the 11 /k/ are `c`. The pair `q`–/s/ is never observed,
so both probabilities are 0.

Smoothing moves mass towards the unobserved outcome, most visibly where counts
are small:

```pycon
>>> laplace = CatScorer(pairs, smoothing_method="laplace").mle()
>>> [round(p, 3) for p in laplace[("q", "k")]]
[0.75, 0.231]
>>> [round(p, 3) for p in laplace[("q", "s")]]
[0.25, 0.167]
```

## Properties

- **Directional.** $P(y \mid x) \neq P(x \mid y)$ unless $c(x) = c(y)$.
- **Bounded and interpretable.** Each score is a probability.
- **Not corrected for chance.** A frequent $y$ has a high $P(y \mid x)$ for
  most $x$, whether or not $x$ and $y$ are associated. Compare with the base
  rate $P(y)$, or use [PMI](pmi.md) or [Theil's U](theil-u.md), which account
  for it.
- **Unstable for rare symbols.** With $c(x) = 2$, the only possible values are
  0, 0.5 and 1. Smoothing, or the [resampling tools](../how-to/significance.md),
  help judge how much to trust them.

## API

::: asymcat.scorer.CatScorer.mle
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Lidstone (1920) for additive smoothing; see
  [References](../references.md#probabilistic-measures).
