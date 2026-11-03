# Directional association

## One number hides the direction

Most measures of association report one number for a pair of variables, as if
the relationship between $X$ and $Y$ were the same as between $Y$ and $X$. Many
real relationships are not. In English spelling, the letter `q` is always
pronounced /k/, but /k/ is usually spelled some other way: `q` predicts /k/
perfectly, while /k/ hardly predicts `q`. A symmetric measure averages the two
and reports a moderate association, which describes neither direction.

ASymCat therefore scores every pair in both directions. For a pair $(x, y)$,
each measure returns a tuple

$$
\big(\text{score}(x \to y),\ \text{score}(y \to x)\big),
$$

where $x \to y$ reads "$x$ as a predictor of $y$".

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> k_given_q, q_given_k = CatScorer(pairs).mle()[("q", "k")]
>>> round(k_given_q, 2), round(q_given_k, 2)
(1.0, 0.18)
```

## Directional and symmetric measures

The measures fall into two groups. **Directional** measures can give different
values in the two positions of the tuple. **Symmetric** measures are the same
in both directions by definition, so the tuple repeats one value; ASymCat
returns them in the same format so that every measure can be used the same way.

| Directional | Symmetric |
|-------------|-----------|
| [`mle`](../measures/mle.md), [`theil_u`](../measures/theil-u.md), [`cond_entropy`](../measures/conditional-entropy.md), [`goodman_kruskal_lambda`](../measures/goodman-kruskal-lambda.md), [`tresoldi`](../measures/tresoldi.md) | [`pmi`](../measures/pmi.md), [`pmi_smoothed`](../measures/pmi.md), [`mutual_information`](../measures/mutual-information.md), [`normalized_mutual_information`](../measures/mutual-information.md), [`chi2`](../measures/chi2.md), [`cramers_v`](../measures/chi2.md), [`fisher`](../measures/fisher.md), [`log_likelihood_ratio`](../measures/log-likelihood-ratio.md), [`jaccard_index`](../measures/jaccard.md) |

The symmetric measures remain useful alongside the directional ones: they say
whether a pair departs from independence at all (PMI, the statistical tests),
and the directional measures then say which way the prediction runs.

## Where asymmetry comes from

For the conditional probability, the two directions share a numerator and
differ in the denominator:

$$
P(y \mid x) = \frac{c(x, y)}{c(x)}, \qquad P(x \mid y) = \frac{c(x, y)}{c(y)}.
$$

They differ whenever $x$ and $y$ have different frequencies. Theil's U behaves
in the same way: both directions share the mutual information and divide it by
the entropy of a different variable. Asymmetry is therefore not a property of a
pair alone, but of the pair within its data: a symbol that occurs with many
partners is a poor predictor of any one of them, even when each of them
predicts it well.

## What the direction does not mean

The direction of a score refers to which variable is conditioned on, not to
time or cause. "$x$ predicts $y$" means that knowing $x$ narrows down $y$, as
the letter narrows down the sound. Whether one causes the other is a question
about how the data came about, which association measures cannot answer.

## Pairs and whole variables

`CatScorer` scores pairs of values: `q` with /k/, `c` with /s/. To summarise the
association between two whole variables (all letters against all sounds) with
one number in each direction, use the functions in `asymcat.correlation`
(see [Summarise whole variables](../how-to/whole-variables.md)).
