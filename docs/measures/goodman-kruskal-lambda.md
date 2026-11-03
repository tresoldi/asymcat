# Goodman–Kruskal λ

**`CatScorer.goodman_kruskal_lambda()`** · directional · range $[0, 1]$ ·
statistical

Goodman and Kruskal's λ (Goodman & Kruskal, 1954) is the proportional reduction
in prediction errors: if you must guess $Y$, how many fewer mistakes do you make
by knowing $X$? Without $X$, the best guess is always the most frequent $y$;
with it, the most frequent $y$ for each $x$.

## Definition

With $n(x, y)$ the counts of a table, $n(y)$ its column totals and $N$ its
total,

$$
\lambda(Y \mid X) = \frac{\sum_x \max_y n(x, y) - \max_y n(y)}{N - \max_y n(y)},
$$

with $\lambda = 1$ by convention when the denominator is 0 (when $Y$ takes a
single value, so that there are no errors to reduce), as for
[Theil's U](theil-u.md). In ASymCat, the table is the
[pair subset](../concepts/notation.md#the-pair-subset) $S(x, y)$:

$$
\texttt{goodman_kruskal_lambda}(x, y) = \big(\lambda(Y_S \mid X_S),\ \lambda(X_S \mid Y_S)\big)
$$

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> gk_lambda = CatScorer(pairs).goodman_kruskal_lambda()
>>> gk_lambda[("c", "k")]
(0.0, 0.0)
>>> gk_lambda[("k", "s")]
(1.0, 1.0)
```

For `c` and /k/, the subset is the whole dataset. Without the letter, the best
guess for the sound is /k/, right 11 times in 15. Knowing the letter does not
change any guess, since /k/ is also the most frequent sound for each letter, so
λ is 0, although [Theil's U](theil-u.md) finds that the letter carries
information about the sound (0.23). In the subset of `k` and /s/, each variable
determines the other.

## Properties

- **Directional.** $\lambda(Y \mid X) \neq \lambda(X \mid Y)$ in general.
- **Zero despite association.** λ only registers associations that change the
  modal guess. When one value dominates every row, λ is 0 whatever the
  association; Theil's U is more sensitive.
- **A convention on degenerate subsets.** When the predicted variable takes a
  single value in the subset, λ is 1, as is Theil's U: for `("q", "k")`, both
  give 1 and 0.
- **Unsigned**, as for the other measures on the pair subset: see
  [Pitfalls](../concepts/pitfalls.md#high-scores-can-mean-avoidance).

## API

::: asymcat.scorer.CatScorer.goodman_kruskal_lambda
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Goodman & Kruskal (1954). See
  [References](../references.md#statistical-tests-and-contingency-table-measures).
