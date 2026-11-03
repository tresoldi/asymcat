# Theil's U

**`CatScorer.theil_u()`** · directional · range $[0, 1]$ · information-theoretic

Theil's U, or the uncertainty coefficient (Theil, 1970), is the share of the
uncertainty about one variable that knowing the other removes. 0 means that
knowing $x$ says nothing about $y$; 1 means that it determines $y$ completely.
It is the information-theoretic counterpart of the
[conditional probability](mle.md), and corrects for the base rate.

## Definition

For two variables $X$ and $Y$,

$$
U(Y \mid X) = \frac{H(Y) - H(Y \mid X)}{H(Y)} = \frac{I(X; Y)}{H(Y)},
$$

with $U(Y \mid X) = 1$ by convention when $H(Y) = 0$. In ASymCat, the variables
are the first and second elements of the
[pair subset](../concepts/notation.md#the-pair-subset) $S(x, y)$, the
co-occurrences whose first element is $x$ or whose second element is $y$:

$$
\texttt{theil_u}(x, y) = \big(U(Y_S \mid X_S),\ U(X_S \mid Y_S)\big)
$$

The entropies follow the scorer's `entropy_estimator`: the plug-in
(maximum-likelihood) estimate by default, or the bias-corrected `"chao_shen"`
and `"nsb"` estimators from FreqProb (see
[Work with sparse data](../how-to/sparse-data.md)).

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> theil_u = CatScorer(pairs).theil_u()
>>> [round(u, 3) for u in theil_u[("c", "k")]]
[0.226, 0.152]
>>> [round(u, 3) for u in theil_u[("q", "k")]]
[1.0, 0.0]
>>> theil_u[("k", "s")]
(1.0, 1.0)
```

- **`c` and /k/**: the subset is the whole dataset. Knowing the letter removes
  23% of the uncertainty about the sound, and knowing the sound removes 15% of
  the uncertainty about the letter.
- **`q` and /k/**: the subset is the column of /k/, where the sound is always
  /k/. Its entropy is zero, so $U(Y_S \mid X_S) = 1$ by convention, and knowing
  the sound says nothing about which of `c`, `k` or `q` spelled it.
- **`k` and /s/**: the subset holds `k`–/k/ and `c`–/s/. Each letter determines
  the sound and each sound the letter, so both scores are 1, although `k` and
  /s/ never occur together. Theil's U measures predictability, not attraction:
  see [Pitfalls](../concepts/pitfalls.md#high-scores-can-mean-avoidance).

## Properties

- **Directional.** $U(Y \mid X) \neq U(X \mid Y)$ in general; both share the
  numerator $I(X; Y)$ and differ in the entropy they normalize by.
- **Corrected for the base rate.** A frequent $y$ does not by itself raise the
  score, unlike $P(y \mid x)$.
- **Unsigned.** Avoidance is as predictive as attraction.
- **A convention on degenerate subsets.** When one variable takes a single
  value in the subset, its score is 1 (no uncertainty left to remove).
  [Goodman–Kruskal λ](goodman-kruskal-lambda.md) uses 0 in the same case.
- **Bias-corrected estimates leave $[0, 1]$.** With `"chao_shen"` or `"nsb"`,
  a slightly negative score on a small subset means no detectable association.
  The plug-in estimate can also print as `-0.0`.

## API

::: asymcat.scorer.CatScorer.theil_u
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

The whole-variable version, for two aligned series rather than a pair of
symbols, is [`asymcat.correlation.theil_u`](../how-to/whole-variables.md).

## References

- Theil (1970); Shannon (1948); Cover & Thomas (2006). See
  [References](../references.md#information-theoretic-measures).
