# Mutual information

**`CatScorer.mutual_information()`**,
**`CatScorer.normalized_mutual_information()`** · symmetric · range
$[0, \infty)$ nats, or $[0, 1]$ normalized · information-theoretic

Mutual information (MI) is the amount of information two variables share: how
much knowing one reduces the uncertainty about the other, on average. It is
zero when they are independent. It is the numerator of
[Theil's U](theil-u.md), before division by the entropy of one variable.

## Definition

$$
I(X; Y) = \sum_{x, y} P(x, y) \ln \frac{P(x, y)}{P(x)\,P(y)}
= H(Y) - H(Y \mid X) = H(X) - H(X \mid Y)
$$

The normalized mutual information divides by the joint entropy, which bounds
it to $[0, 1]$; it is the complement of Rajski's (1961) distance:

$$
\mathrm{NMI}(X; Y) = \frac{I(X; Y)}{H(X, Y)}
$$

In ASymCat, both are computed on the
[pair subset](../concepts/notation.md#the-pair-subset) $S(x, y)$, and both
tuple entries are equal. When the subset is a single cell (one $x$ and one
$y$), the joint entropy is zero and NMI is 0. The entropies follow the
scorer's `entropy_estimator`.

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> [round(i, 3) for i in scorer.mutual_information()[("c", "k")]]
[0.131, 0.131]
>>> [round(i, 3) for i in scorer.normalized_mutual_information()[("c", "k")]]
[0.1, 0.1]
>>> [round(i, 3) for i in scorer.normalized_mutual_information()[("k", "s")]]
[1.0, 1.0]
```

In the subset of `k` and /s/ (`k`–/k/ and `c`–/s/), each variable determines the
other, so NMI is 1 even though `k` and /s/ never co-occur.

## Properties

- **Symmetric.** For the direction, divide by one entropy instead of the joint
  one: that is [Theil's U](theil-u.md).
- **Unsigned.** Avoidance shares as much information as attraction; see
  [Pitfalls](../concepts/pitfalls.md#high-scores-can-mean-avoidance). For a
  signed, per-pair quantity, use [PMI](pmi.md), whose average over the cells
  of a table is the mutual information.
- **Biased upwards on small samples** with the plug-in estimator. The
  bias-corrected estimators reduce this, and can return slightly negative
  values.

## API

::: asymcat.scorer.CatScorer.mutual_information
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: asymcat.scorer.CatScorer.normalized_mutual_information
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Shannon (1948); Rajski (1961); Cover & Thomas (2006). See
  [References](../references.md#information-theoretic-measures).
