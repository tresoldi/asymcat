# Fisher's exact test

**`CatScorer.fisher()`**, **`CatScorer.fisher_pvalue()`** · symmetric ·
statistical

Fisher's exact test (Fisher, 1922) tests the independence of the two rows of a
[2×2 table](../concepts/notation.md#the-22-contingency-table) without the
large-sample approximation of [chi-square](chi2.md). It is the test to use when
counts are small. `fisher` returns the sample odds ratio of the table and
`fisher_pvalue` the two-sided p-value of the test, both from SciPy's
`fisher_exact`.

## Definition

For the table

|         | $y$ | not $y$ |
|---------|-----|---------|
| $x$     | $a$ | $b$ |
| not $x$ | $c$ | $d$ |

the odds ratio is

$$
\mathrm{OR} = \frac{a\,d}{b\,c}:
$$

the odds of $y$ when $x$ is present, divided by the odds of $y$ when it is
not. $\mathrm{OR} > 1$ means attraction and $\mathrm{OR} < 1$ avoidance. The
p-value is the probability, with the margins of the table fixed, of a table at
least as unlikely as the one observed. Both are symmetric, so both tuple
entries are equal.

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> scorer.fisher()[("c", "s")], scorer.fisher()[("c", "k")]
((inf, inf), (0.0, 0.0))
>>> [round(p, 3) for p in scorer.fisher_pvalue()[("c", "s")]]
[0.231, 0.231]
```

Every /s/ is spelled `c`, so the table of `c` and /s/ has $c = 0$ and an
infinite odds ratio; the table of `c` and /k/ has $d = 0$ and an odds ratio of
0. Even so, with 15 observations the association is not significant
($p = 0.23$).

## Properties

- **Exact.** The p-value does not rely on large counts, which makes it the
  safest of the statistical tests for sparse data.
- **Signed through the odds ratio**, around 1 rather than 0. Its logarithm is
  symmetric around 0.
- **Infinite and zero odds ratios.** Any empty cell gives $\mathrm{OR} = 0$ or
  $\infty$, which is common with categorical data. Such values break
  [`scale_scorer`](../how-to/compare-scores.md); rank by the p-value or use the
  smoothed [conditional probability](mle.md) instead.
- **Many tests.** See
  [Test significance](../how-to/significance.md#many-pairs-many-tests).

## API

::: asymcat.scorer.CatScorer.fisher
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: asymcat.scorer.CatScorer.fisher_pvalue
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Fisher (1922). See
  [References](../references.md#statistical-tests-and-contingency-table-measures).
