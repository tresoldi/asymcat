# Measures

Every measure is a method of [`CatScorer`](../api/catscorer.md) and returns, for
each pair $(x, y)$, a tuple $(x \to y,\ y \to x)$ (see
[Notation](../concepts/notation.md)). Each page below gives the definition as
ASymCat computes it, a worked example on the same
[example data](../concepts/notation.md#the-example-data), its properties and
caveats, the API entry and the references.

| Measure | Method | Direction | Range | Computed on |
|---------|--------|-----------|-------|-------------|
| [Conditional probability](mle.md) | `mle` | directional | $[0, 1]$ | counts |
| [Pointwise mutual information](pmi.md) | `pmi`, `pmi_smoothed` | symmetric | $(-\infty, \infty)$; NPMI $[-1, 1]$ | counts |
| [Theil's U](theil-u.md) | `theil_u` | directional | $[0, 1]$ | pair subset |
| [Conditional entropy](conditional-entropy.md) | `cond_entropy` | directional | $[0, \infty)$, lower is stronger | pair subset |
| [Mutual information](mutual-information.md) | `mutual_information`, `normalized_mutual_information` | symmetric | $[0, \infty)$; NMI $[0, 1]$ | pair subset |
| [Chi-square and Cramér's V](chi2.md) | `chi2`, `chi2_pvalue`, `cramers_v` | symmetric | $[0, \infty)$; V $[0, 1]$ | 2×2 table |
| [Fisher's exact test](fisher.md) | `fisher`, `fisher_pvalue` | symmetric | odds ratio $[0, \infty]$ | 2×2 table |
| [Log-likelihood ratio](log-likelihood-ratio.md) | `log_likelihood_ratio`, `log_likelihood_ratio_pvalue` | symmetric | $[0, \infty)$ | 2×2 table |
| [Goodman–Kruskal λ](goodman-kruskal-lambda.md) | `goodman_kruskal_lambda` | directional | $[0, 1]$ | pair subset |
| [Jaccard index](jaccard.md) | `jaccard_index` | symmetric | $[0, 1]$ | counts |
| [The Tresoldi measure](tresoldi.md) | `tresoldi` | directional | $(-\infty, \infty)$ | counts |

"Computed on" says which view of the pair a measure uses: the counts
$c(x, y)$, $c(x)$ and $c(y)$ directly; the
[2×2 contingency table](../concepts/notation.md#the-22-contingency-table) over
all co-occurrences; or the
[pair subset](../concepts/notation.md#the-pair-subset) of co-occurrences that
contain $x$ or $y$.

To decide which measure answers your question, see
[Choosing a measure](../concepts/choosing-a-measure.md). For p-values and
confidence intervals for any of them, see
[Test significance and uncertainty](../how-to/significance.md).
