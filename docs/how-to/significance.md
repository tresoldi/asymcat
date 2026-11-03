# Test significance and uncertainty

A score computed from a few co-occurrences can be large by chance. ASymCat
offers three kinds of tools to judge the evidence: the p-values of the
classical tests, permutation tests for any measure, and bootstrap confidence
intervals.

The examples use the [example data](../concepts/notation.md#the-example-data)
repeated four times, so that there is some evidence to test:

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs * 4)
```

## Parametric tests

Three measures have a classical test of independence for the
[2×2 table](../concepts/notation.md#the-22-contingency-table) of each pair:

```pycon
>>> round(scorer.chi2_pvalue()[("c", "s")][0], 4)
0.0028
>>> round(scorer.log_likelihood_ratio_pvalue()[("c", "s")][0], 4)
0.0001
>>> round(scorer.fisher_pvalue()[("c", "s")][0], 4)
0.0005
```

[Fisher's exact test](../measures/fisher.md) is the safest with small counts;
the [chi-square](../measures/chi2.md) and
[G²](../measures/log-likelihood-ratio.md) tests rely on a large-sample
approximation. All three are symmetric: both tuple entries hold the same
p-value.

## Permutation tests for any measure

`permutation_pvalue` tests any measure, directional ones included, against the
hypothesis that the two variables are independent. It shuffles which $y$ goes
with which $x$, which keeps both sets of frequencies and destroys the
association, recomputes the measure, and reports how often the shuffled score
is at least as extreme as the observed one:

```pycon
>>> pvalues = scorer.permutation_pvalue("theil_u", n_permutations=999, seed=0)
>>> [round(p, 3) for p in pvalues[("c", "k")]]
[0.003, 0.003]
```

With four times the data, the association of `c` with /k/ is now significant
in both directions. The result has one p-value per direction. `alternative`
selects the tail: `"greater"` (the default) when larger scores mean stronger
association,
`"less"` for [conditional entropy](../measures/conditional-entropy.md), and
`"two-sided"` for either. The smallest possible p-value is
$1/(\texttt{n_permutations} + 1)$, so use more permutations to resolve small
p-values. Use `seed` to make the result reproducible.

## Confidence intervals

`bootstrap_ci` gives a percentile confidence interval for each score, by
recomputing the measure on resamples of the co-occurrences drawn with
replacement:

```pycon
>>> intervals = scorer.bootstrap_ci("mle", n_bootstrap=999, confidence_level=0.95, seed=0)
>>> (low_xy, high_xy), (low_yx, high_yx) = intervals[("c", "k")]
>>> round(low_xy, 2), round(high_xy, 2)
(0.45, 0.75)
```

With 60 co-occurrences, $P(k \mid c) = 0.6$ could plausibly lie anywhere
between 0.45 and 0.75. The point estimate is the measure itself
(`scorer.mle()`). Wide intervals mark scores that the data does not pin
down, which is common for rare symbols even after [smoothing](sparse-data.md).

Both resampling methods accept the measures in
`CatScorer.RESAMPLABLE_MEASURES`, computed with their default parameters, and
use the scorer's smoothing and entropy settings.

## Many pairs, many tests

A scorer tests every pair, so some p-values will be small by chance. Control
the false discovery rate (Benjamini & Hochberg, 1995) across the pairs with
SciPy:

```pycon
>>> from scipy.stats import false_discovery_control
>>> pvalues = scorer.fisher_pvalue()
>>> pairs_tested = sorted(pvalues)
>>> adjusted = false_discovery_control([pvalues[pair][0] for pair in pairs_tested])
>>> [pair for pair, p in zip(pairs_tested, adjusted) if p < 0.05]
[('c', 'k'), ('c', 's'), ('k', 'k'), ('k', 's')]
```

The pairs of `q` are not significant after the correction: two
observations of `q`, even repeated four times, are little evidence.

## Caveats

- The tests treat co-occurrences as independent observations. Co-occurrences
  [collected from sequences](prepare-data.md) are not, so p-values are somewhat
  too small and intervals too narrow when sequences are long.
- A significant association can be an avoidance; check the sign of
  [PMI](../measures/pmi.md) or the [odds ratio](../measures/fisher.md).
- Significance is not strength: with enough data, a negligible association is
  significant. Report a measure of strength with every p-value.
