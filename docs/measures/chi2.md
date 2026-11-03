# Chi-square and Cramér's V

**`CatScorer.chi2()`**, **`CatScorer.chi2_pvalue()`**,
**`CatScorer.cramers_v()`** · symmetric · statistical

Pearson's chi-square statistic (Pearson, 1900) measures how far the
[2×2 contingency table](../concepts/notation.md#the-22-contingency-table) of a
pair departs from what independence predicts, and its p-value tests whether the
departure could be due to chance. Cramér's V (Cramér, 1946) rescales the
statistic to $[0, 1]$, so that it reports the strength of the association
rather than the amount of evidence for it.

## Definition

With $O_{ij}$ the observed counts of the 2×2 table and
$E_{ij} = (\text{row total}_i \times \text{column total}_j) / n$ the counts
expected under independence,

$$
\chi^2 = \sum_{i, j} \frac{\big(|O_{ij} - E_{ij}| - \tfrac{1}{2}\big)^2}{E_{ij}}.
$$

The $\tfrac{1}{2}$ is Yates' (1934) continuity correction, which SciPy's
`chi2_contingency` applies to 2×2 tables and ASymCat keeps. `chi2_pvalue`
returns the p-value of the test, from the $\chi^2$ distribution with one degree
of freedom.

`cramers_v` uses the bias-corrected form of Bergsma (2013), which subtracts the
value expected under independence before taking the root. With $\varphi^2 =
\chi^2 / n$ and an $r \times k$ table:

$$
V = \sqrt{\frac{\tilde\varphi^2}{\min(\tilde k - 1, \tilde r - 1)}},\quad
\tilde\varphi^2 = \max\!\Big(0,\ \varphi^2 - \frac{(k-1)(r-1)}{n-1}\Big),\quad
\tilde r = r - \frac{(r-1)^2}{n-1},\quad \tilde k = k - \frac{(k-1)^2}{n-1}.
$$

All three are symmetric, so both tuple entries are equal.

## Example

The [example data](../concepts/notation.md#the-example-data) has only 15
co-occurrences. Repeating it ten times keeps the same proportions with ten
times the evidence:

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> small, large = CatScorer(pairs), CatScorer(pairs * 10)
>>> round(small.chi2()[("c", "k")][0], 3), round(large.chi2()[("c", "k")][0], 3)
(1.065, 25.266)
>>> round(small.chi2_pvalue()[("c", "k")][0], 3), large.chi2_pvalue()[("c", "k")][0] < 0.001
(0.302, True)
>>> round(small.cramers_v()[("c", "k")][0], 3), round(large.cramers_v()[("c", "k")][0], 3)
(0.0, 0.404)
```

The statistic grows with the sample, and the same association becomes
significant. Cramér's V does not grow with $n$ in the same way, but the bias
correction sets it to 0 when, as in the small sample, the association is no
larger than independence would produce by chance.

## Properties

- **Symmetric.** Both entries are equal, and with only two values of $y$, the
  tables of $(x, y_1)$ and $(x, y_2)$ are mirror images and get the same score.
- **Unsigned.** The statistic is the same for attraction and avoidance; check
  the sign of [PMI](pmi.md) for the direction of the departure.
- **Small expected counts.** The $\chi^2$ approximation is unreliable when
  expected counts are small (a common rule of thumb is below 5). Yates'
  correction makes the test conservative; [Fisher's exact test](fisher.md)
  avoids the approximation.
- **Many tests.** Scoring every pair runs one test per pair; see
  [Test significance](../how-to/significance.md#many-pairs-many-tests).

## API

::: asymcat.scorer.CatScorer.chi2
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: asymcat.scorer.CatScorer.chi2_pvalue
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: asymcat.scorer.CatScorer.cramers_v
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

The whole-variable version of Cramér's V is
[`asymcat.correlation.cramers_v`](../how-to/whole-variables.md).

## References

- Pearson (1900); Yates (1934); Cramér (1946); Bergsma (2013). See
  [References](../references.md#statistical-tests-and-contingency-table-measures).
