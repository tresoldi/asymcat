# Rank, scale and compare scores

A measure returns a dictionary from pairs to $(x \to y,\ y \to x)$ tuples.
This guide shows common ways to work with that dictionary: ranking pairs,
finding the most asymmetric ones, and rescaling scores.

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> mle = scorer.mle()
```

## Rank pairs in one direction

Sort the pairs by the score in the direction you care about: index `0` for
$x \to y$, index `1` for $y \to x$.

```pycon
>>> by_letter = sorted(mle, key=lambda pair: mle[pair][0], reverse=True)
>>> by_letter[:3]
[('k', 'k'), ('q', 'k'), ('c', 'k')]
>>> by_sound = sorted(mle, key=lambda pair: mle[pair][1], reverse=True)
>>> by_sound[:3]
[('c', 's'), ('c', 'k'), ('k', 'k')]
```

Letters are the best predictors of sounds for `k` and `q` (always /k/); sounds
are the best predictors of letters for /s/ (always `c`). Sort in ascending
order for [conditional entropy](../measures/conditional-entropy.md), where
lower is stronger.

## Find the most asymmetric pairs

The difference between the two directions shows where prediction runs one way:

```pycon
>>> asymmetry = {pair: xy - yx for pair, (xy, yx) in mle.items()}
>>> for pair in sorted(asymmetry, key=lambda p: abs(asymmetry[p]), reverse=True)[:3]:
...     print(pair, round(asymmetry[pair], 2))
('q', 'k') 0.82
('k', 'k') 0.73
('c', 's') -0.6
```

Positive values mean that $x$ predicts $y$ better than the reverse. Only
directional measures give non-zero differences; see
[Directional association](../concepts/directional-association.md).

## Keep only observed pairs

Measures score every combination of values, including pairs never observed.
The observation counts are in `scorer.obs`, where `"11"` is $c(x, y)$:

```pycon
>>> observed = {pair: score for pair, score in mle.items() if scorer.obs[pair]["11"] > 0}
>>> sorted(observed)
[('c', 'k'), ('c', 's'), ('k', 'k'), ('q', 'k')]
```

## Rescale scores

`scale_scorer` rescales both directions together, so that scores from
different measures can be compared or combined. `"minmax"` maps them to a
range (by default $[0, 1]$), `"mean"` centres them on the mean and divides by
the range, and `"stdev"` standardizes them:

```pycon
>>> from asymcat.scorer import scale_scorer
>>> pmi = scorer.pmi()
>>> scaled = scale_scorer(pmi, method="minmax")
>>> [round(v, 3) for v in scaled[("c", "s")]]
[1.0, 1.0]
>>> [round(v, 3) for v in scaled[("k", "s")]]
[0.0, 0.0]
```

Scaling cannot handle infinite values, such as
[Fisher's odds ratio](../measures/fisher.md) for tables with an empty cell.

## Invert scores

`invert_scorer` subtracts every score from the largest one, turning a measure
where lower is stronger into one where higher is stronger:

```pycon
>>> from asymcat.scorer import invert_scorer
>>> inverted = invert_scorer(scorer.cond_entropy())
>>> [round(v, 3) for v in inverted[("q", "k")]]
[0.995, 0.0]
```

## Combine measures

To combine measures with different scales, rank or scale each one first. For
example, the average rank of each pair under two measures, for $x \to y$:

```pycon
>>> def ranks(scores, direction=0):
...     order = sorted(scores, key=lambda pair: scores[pair][direction], reverse=True)
...     return {pair: rank for rank, pair in enumerate(order, start=1)}
>>> mle_ranks, theil_ranks = ranks(mle), ranks(scorer.theil_u())
>>> average = {pair: (mle_ranks[pair] + theil_ranks[pair]) / 2 for pair in observed}
>>> sorted(average, key=average.get)
[('k', 'k'), ('q', 'k'), ('c', 'k'), ('c', 's')]
```

Ties share the order in which the scorer lists the pairs; break them
explicitly, for example by the count `scorer.obs[pair]["11"]`, when the order
matters.
