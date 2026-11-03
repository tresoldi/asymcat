# Summarise whole variables

`CatScorer` scores pairs of values. To summarise the association between two
whole variables (all letters against all sounds) with a single number, use the
functions in `asymcat.correlation`. They take two aligned series of the same
length, one value per observation:

```pycon
>>> from asymcat import correlation
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> letters = [x for x, _ in pairs]
>>> sounds = [y for _, y in pairs]
```

## Theil's U

`correlation.theil_u(series_x, series_y)` is $U(X \mid Y)$: the share of the
uncertainty about the **first** series that knowing the second removes.

```pycon
>>> round(correlation.theil_u(sounds, letters), 3)  # sounds predicted from letters
0.226
>>> round(correlation.theil_u(letters, sounds), 3)  # letters predicted from sounds
0.152
```

Knowing the letter removes 23% of the uncertainty about the sound; knowing the
sound removes 15% of the uncertainty about the letter. On this dataset, the
pair subset of `c` and /k/ is the whole dataset, so
[`CatScorer.theil_u`](../measures/theil-u.md) gives the same numbers for that
pair.

## Conditional entropy

`correlation.conditional_entropy(series_x, series_y)` is $H(X \mid Y)$, in
nats:

```pycon
>>> round(correlation.conditional_entropy(sounds, letters), 3)
0.449
>>> round(correlation.conditional_entropy(letters, sounds), 3)
0.73
```

Both functions accept `estimator="chao_shen"` or `"nsb"` for bias-corrected
entropies (see [Work with sparse data](sparse-data.md)).

## Cramér's V

`correlation.cramers_v(series_x, series_y)` is the bias-corrected Cramér's V of
the full contingency table of the two series. It is symmetric:

```pycon
>>> round(correlation.cramers_v(letters, sounds), 3)
0.205
```

Over the whole 3×2 table, the association is weak. For each pair's own 2×2
table, the bias correction finds no association beyond chance on 15
observations; see [Chi-square and Cramér's V](../measures/chi2.md).
