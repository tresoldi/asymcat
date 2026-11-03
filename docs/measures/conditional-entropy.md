# Conditional entropy

**`CatScorer.cond_entropy()`** · directional · range $[0, \infty)$ nats ·
information-theoretic

The conditional entropy $H(Y \mid X)$ is the uncertainty about $Y$ that remains
once $X$ is known. Unlike most measures, **lower means stronger**: 0 means that
$X$ determines $Y$. It is the unnormalized quantity behind
[Theil's U](theil-u.md).

## Definition

$$
H(Y \mid X) = -\sum_{x, y} P(x, y) \ln P(y \mid x) = H(X, Y) - H(X)
$$

As for Theil's U, the variables are the first and second elements of the
[pair subset](../concepts/notation.md#the-pair-subset) $S(x, y)$:

$$
\texttt{cond_entropy}(x, y) = \big(H(Y_S \mid X_S),\ H(X_S \mid Y_S)\big)
$$

The entropies follow the scorer's `entropy_estimator`; with `"chao_shen"` or
`"nsb"`, the two terms of $H(X, Y) - H(X)$ are corrected separately.

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> cond_entropy = CatScorer(pairs).cond_entropy()
>>> [round(h, 3) for h in cond_entropy[("c", "k")]]
[0.449, 0.73]
>>> [round(h, 3) for h in cond_entropy[("q", "k")]]
[0.0, 0.995]
```

Once the letter is known, 0.449 nats of uncertainty about the sound remain
across the whole dataset (the subset of `c` and /k/). In the subset of `q` and
/k/, the sound is always /k/, so nothing remains to predict ($0$), while the
letter keeps almost all its uncertainty (0.995 nats, close to
$\ln 3 \approx 1.10$ for three letters).

## Properties

- **Directional**, and **lower is stronger**: rank pairs in ascending order,
  and use `alternative="less"` with
  [`permutation_pvalue`](../how-to/significance.md).
- **Not normalized.** Its scale depends on the number of values in the subset;
  Theil's U divides it out, which makes pairs comparable.
- **Unsigned**, as for the other measures on the pair subset: see
  [Pitfalls](../concepts/pitfalls.md#high-scores-can-mean-avoidance).

## API

::: asymcat.scorer.CatScorer.cond_entropy
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

The whole-variable version is
[`asymcat.correlation.conditional_entropy`](../how-to/whole-variables.md).

## References

- Shannon (1948); Cover & Thomas (2006). See
  [References](../references.md#information-theoretic-measures).
