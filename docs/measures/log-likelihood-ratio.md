# Log-likelihood ratio (G²)

**`CatScorer.log_likelihood_ratio()`**,
**`CatScorer.log_likelihood_ratio_pvalue()`** · symmetric · statistical

The log-likelihood ratio statistic $G^2$ compares the likelihood of the
[2×2 table](../concepts/notation.md#the-22-contingency-table) under
independence with its likelihood without that constraint. Dunning (1993)
introduced it to computational linguistics as an alternative to
[chi-square](chi2.md) for rare events, and it is widely used for collocation
extraction.

## Definition

With $O_{ij}$ the observed and $E_{ij}$ the expected counts of the table, as for
chi-square,

$$
G^2 = 2 \sum_{i, j} O_{ij} \ln \frac{O_{ij}}{E_{ij}},
$$

where empty cells contribute 0. ASymCat applies no continuity correction.
`log_likelihood_ratio_pvalue` returns the p-value from the $\chi^2$
distribution with one degree of freedom, which $G^2$ follows asymptotically
under independence. Both are symmetric, so both tuple entries are equal.

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> round(scorer.log_likelihood_ratio()[("c", "k")][0], 3)
3.937
>>> round(scorer.log_likelihood_ratio_pvalue()[("c", "k")][0], 3)
0.047
```

On the same table, the Yates-corrected chi-square gives $p = 0.30$ and
Fisher's exact test $p = 0.23$. With so few observations, the asymptotic tests
disagree, and the exact test is the one to trust.

## Properties

- **Additive.** $G^2$ decomposes over nested tables, which makes it convenient
  for model comparison.
- **Asymptotically equivalent to chi-square**, and neither is reliable when
  expected counts are small: without a continuity correction, $G^2$ tends to
  be anti-conservative there, as above. Prefer
  [Fisher's exact test](fisher.md) for sparse tables.
- **Unsigned.** Check the sign of [PMI](pmi.md) or the
  [odds ratio](fisher.md) for the direction of the departure.

## API

::: asymcat.scorer.CatScorer.log_likelihood_ratio
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: asymcat.scorer.CatScorer.log_likelihood_ratio_pvalue
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Dunning (1993). See
  [References](../references.md#statistical-tests-and-contingency-table-measures).
