# Choosing a measure

Start from the question you want to answer. The table gives the measures that
answer it, and the notes below help choose among them.

| Question | Measures |
|----------|----------|
| How likely is $y$ when $x$ occurs? | [`mle`](../measures/mle.md) |
| How much does knowing $x$ tell me about $y$, beyond its base rate? | [`theil_u`](../measures/theil-u.md), [`cond_entropy`](../measures/conditional-entropy.md) |
| How many fewer prediction errors does knowing $x$ give? | [`goodman_kruskal_lambda`](../measures/goodman-kruskal-lambda.md) |
| Do $x$ and $y$ co-occur more, or less, than chance predicts? | [`pmi`](../measures/pmi.md) (signed), [`fisher`](../measures/fisher.md) odds ratio |
| Is the association statistically significant? | [`fisher_pvalue`](../measures/fisher.md), [`chi2_pvalue`](../measures/chi2.md), [`log_likelihood_ratio_pvalue`](../measures/log-likelihood-ratio.md); `permutation_pvalue` for any measure |
| How strong is the association, on a fixed scale? | [`cramers_v`](../measures/chi2.md), [`normalized_mutual_information`](../measures/mutual-information.md), NPMI |
| How much do the occurrences of $x$ and $y$ overlap? | [`jaccard_index`](../measures/jaccard.md) |
| How should symbols in an alignment be scored? | [`tresoldi`](../measures/tresoldi.md) |

## Notes

**Start with `mle` and `theil_u`.** The conditional probability is the easiest
to read and to check by hand; Theil's U adds a correction for the base rate,
so a frequent $y$ does not look predictable from every $x$. Disagreement
between them usually means that $y$ is frequent.

**Use a signed measure to tell attraction from avoidance.** The measures on the
pair subset (Theil's U, conditional entropy, mutual information, λ) and the
test statistics are unsigned: a pair that never co-occurs can score as high as
one that always does. Check the sign of PMI, or the conditional probability,
before interpreting a high score. See
[Pitfalls](pitfalls.md#high-scores-can-mean-avoidance).

**Prefer exact or resampling tests for sparse data.** With small counts,
chi-square and G² rely on an approximation that does not hold; use Fisher's
exact test for the 2×2 table, or `permutation_pvalue` for any measure. See
[Test significance and uncertainty](../how-to/significance.md).

**Smooth or correct when counts are small.** Laplace or Lidstone smoothing
stabilizes the conditional probabilities and `pmi_smoothed`; the Chao–Shen
and NSB estimators reduce the bias of the information-theoretic measures. See
[Work with sparse data](../how-to/sparse-data.md).

**Compare scores within a dataset.** PMI, G² and the conditional entropy depend
on the sample size or the number of symbols. To compare across datasets, prefer
a bounded measure (Theil's U, Cramér's V, NMI) and report the sample sizes.
