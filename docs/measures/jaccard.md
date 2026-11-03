# Jaccard index

**`CatScorer.jaccard_index()`** · symmetric · range $[0, 1]$ · set-based

The Jaccard index (Jaccard, 1912) is the overlap of two sets divided by their
union. Here, the sets are the co-occurrences that contain $x$ and those that
contain $y$: the index is the share of the co-occurrences involving either
symbol that involve both.

## Definition

$$
J(x, y) = \frac{|A_x \cap A_y|}{|A_x \cup A_y|} = \frac{c(x, y)}{c(x) + c(y) - c(x, y)}
$$

where $A_x$ is the set of co-occurrences whose first element is $x$, and $A_y$
the set whose second element is $y$. It is 0 when $x$ and $y$ never co-occur,
and 1 when they only ever occur together. The index is symmetric, so both
tuple entries are equal.

## Example

With the [example data](../concepts/notation.md#the-example-data):

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> jaccard = CatScorer(pairs).jaccard_index()
>>> jaccard[("c", "k")], jaccard[("c", "s")]
((0.4, 0.4), (0.4, 0.4))
>>> [round(j, 3) for j in jaccard[("q", "k")]]
[0.182, 0.182]
```

`c` and /k/ share 6 of the $10 + 11 - 6 = 15$ co-occurrences that involve
either; `c` and /s/ share 4 of 10.

## Properties

- **Symmetric** and **bounded**, with a simple reading as overlap.
- **Not corrected for chance.** Frequent symbols overlap more by chance, and
  the index does not distinguish attraction from mere frequency. Use
  [PMI](pmi.md) for that.
- **Natural for presence–absence data**, where co-occurrences are joint
  presences at a site (see
  [Prepare co-occurrence data](../how-to/prepare-data.md#presenceabsence-matrices)).
- **Related to the conditional probabilities.** $1/J = 1/P(y \mid x) +
  1/P(x \mid y) - 1$, so $J$ is high only when both are high.

## API

::: asymcat.scorer.CatScorer.jaccard_index
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## References

- Jaccard (1912). See [References](../references.md#set-based-measures).
