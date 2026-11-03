# Pointwise mutual information

**`CatScorer.pmi()`**, **`CatScorer.pmi(normalized=True)`**,
**`CatScorer.pmi_smoothed()`** · symmetric · signed · information-theoretic

Pointwise mutual information (PMI) compares how often $x$ and $y$ co-occur with
how often they would if they were independent. It is positive when the pair is
more frequent than chance, negative when it is less frequent, and zero at
independence. It is widely used for collocations and word association
(Church & Hanks, 1990).

## Definition

$$
\mathrm{PMI}(x, y) = \ln \frac{P(x, y)}{P(x)\,P(y)}
= \ln \frac{P(y \mid x)}{P(y)} = \ln \frac{P(x \mid y)}{P(x)}
$$

The normalized PMI (Bouma, 2009) divides by $-\ln P(x, y)$, which bounds it to
$[-1, 1]$: 1 when $x$ and $y$ only ever occur together, 0 at independence.

$$
\mathrm{NPMI}(x, y) = \frac{\mathrm{PMI}(x, y)}{-\ln P(x, y)}
$$

PMI is symmetric, so both tuple entries are equal. In ASymCat:

- **`pmi()`** uses relative frequencies. A pair never observed together would
  have $\mathrm{PMI} = -\infty$; it gets the floor $P(x, y) = (1/n)^2$
  instead, which gives a large negative value.
- **`pmi(normalized=True)`** returns NPMI, with the same floor.
- **`pmi_smoothed()`** smooths the joint distribution $P(x, y)$ over all pairs
  of observed symbols with the scorer's `smoothing_method`, estimated by
  FreqProb, and takes $P(x)$ and $P(y)$ as its sums. It is meant for use with
  `"laplace"` or `"lidstone"`; see the properties below.

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> pmi = scorer.pmi()
>>> [round(v, 3) for v in pmi[("c", "s")]]
[0.405, 0.405]
>>> [round(v, 3) for v in pmi[("c", "k")]]
[-0.201, -0.201]
>>> [round(v, 3) for v in pmi[("q", "s")]]  # never observed
[-2.079, -2.079]
>>> [round(v, 3) for v in scorer.pmi(normalized=True)[("c", "s")]]
[0.307, 0.307]
```

`c` occurs with /s/ more often than independence predicts
($\mathrm{PMI} > 0$), and with /k/ slightly less often: most /k/ come from `c`,
but `c` is frequent, so $P(k \mid c) = 0.6$ is below the base rate
$P(k) = 11/15 \approx 0.73$.

## Properties

- **Symmetric.** Both tuple entries are always equal. For a directional
  version, see [the Tresoldi measure](tresoldi.md), which tempers the PMI with
  the conditional probability.
- **Signed.** Unlike the information-theoretic measures on the pair subset, PMI
  separates attraction (positive) from repulsion (negative).
- **Biased towards rare pairs.** A pair seen once, made of two rare symbols,
  can have a higher PMI than a frequent, reliable pair. NPMI reduces this
  effect; testing with [`permutation_pvalue`](../how-to/significance.md) helps
  decide which scores to trust.
- **Unobserved pairs depend on the floor.** Their score is determined by
  $(1/n)^2$ and the marginals, not by evidence, so they rank below every
  observed pair but are not comparable across datasets of different sizes.
- **`pmi_smoothed` needs a smoothing method.** With the default
  `smoothing_method="mle"`, unobserved pairs get a near-zero probability from
  FreqProb and a very large negative score. With `"laplace"` or `"lidstone"`,
  every pair receives the same pseudo-count, so unobserved pairs score below
  independence but by a finite amount.

## API

::: asymcat.scorer.CatScorer.pmi
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: asymcat.scorer.CatScorer.pmi_smoothed
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Church & Hanks (1990); Bouma (2009). See
  [References](../references.md#pointwise-mutual-information).
