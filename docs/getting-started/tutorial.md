# Tutorial

This tutorial goes from a list of observations to directional scores, tests
and matrices ready to plot. It takes about ten minutes, and every example can
be pasted into a Python session. The examples use English spelling and
pronunciation, but nothing in ASymCat is specific to language: any two
categorical variables observed together will do.

## Score pairs of observations

ASymCat starts from **co-occurrences**: pairs $(x, y)$ of values observed
together. Here, each pair is a letter and the sound it spells:

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
```

A `CatScorer` computes every measure from these pairs, on demand. The simplest
is the conditional probability:

```pycon
>>> mle = scorer.mle()
>>> k_given_q, q_given_k = mle[("q", "k")]
>>> round(k_given_q, 2), round(q_given_k, 2)
(1.0, 0.18)
```

Every measure returns a dictionary from pairs $(x, y)$ to tuples
$(x \to y,\ y \to x)$. Here, `q` predicts /k/ perfectly, while /k/ points back
to `q` only 18% of the time: the association is **directional**. A symmetric
measure would report one number and hide that difference.

The dictionary covers every combination of the observed values, including
pairs never seen together:

```pycon
>>> sorted(mle)
[('c', 'k'), ('c', 's'), ('k', 'k'), ('k', 's'), ('q', 'k'), ('q', 's')]
>>> mle[("q", "s")]
(0.0, 0.0)
```

## Compare measures

Measures answer different questions (see
[Choosing a measure](../concepts/choosing-a-measure.md)). For `c` and /k/:

```pycon
>>> [round(v, 3) for v in scorer.mle()[("c", "k")]]
[0.6, 0.545]
>>> [round(v, 3) for v in scorer.theil_u()[("c", "k")]]
[0.226, 0.152]
>>> [round(v, 3) for v in scorer.pmi()[("c", "k")]]
[-0.201, -0.201]
```

`c` is /k/ six times in ten, and knowing the letter removes 23% of the
uncertainty about the sound (Theil's U). Yet the PMI is negative: /k/ is so
frequent that `c` is pronounced /k/ slightly *less* often than chance would
predict, because `c` is also the only spelling of /s/. Each
[measure page](../measures/index.md) explains what its numbers mean.

## Collect co-occurrences from sequences

Data often comes as pairs of sequences, such as words and their
pronunciations. `collect_cooccs` turns aligned sequences into co-occurrences,
position by position with `order=1`:

```pycon
>>> import asymcat
>>> words = [("cat", "kat"), ("cot", "kot"), ("city", "siti"), ("kit", "kit"), ("cent", "sent")]
>>> cooccs = asymcat.collect_cooccs(words, order=1)
>>> cooccs[:3]
[('c', 'k'), ('a', 'a'), ('t', 't')]
>>> len(cooccs)
17
```

The [guide to preparing data](../how-to/prepare-data.md) covers the other
options: whole sequences, n-grams, files and presence–absence matrices.

## Handle sparse data

Real data has many rare pairs, for which raw relative frequencies are
unreliable. A scorer can smooth its probabilities, here with Laplace smoothing:

```pycon
>>> smoothed = asymcat.CatScorer(cooccs, smoothing_method="laplace")
>>> [round(v, 2) for v in smoothed.mle()[("c", "k")]]
[0.5, 0.6]
```

The information-theoretic measures can also use bias-corrected entropy
estimators; see [Work with sparse data](../how-to/sparse-data.md).

## Test significance

A score is only as good as the evidence behind it. `permutation_pvalue` tests
any measure against independence by shuffling the data:

```pycon
>>> pvalues = smoothed.permutation_pvalue("theil_u", n_permutations=499, seed=0)
>>> [round(p, 3) for p in pvalues[("c", "k")]]
[0.872, 1.0]
```

With five words, the data cannot tell the association of `c` with /k/ from
chance: shuffled data score at least as high most of the time. [Test significance and
uncertainty](../how-to/significance.md) covers the parametric tests, bootstrap
confidence intervals and testing many pairs at once.

## Turn scores into matrices

To plot or analyse the scores of all pairs at once, `scorer2matrices` arranges
a measure as two matrices, one for each direction:

```pycon
>>> from asymcat.scorer import scorer2matrices
>>> xy, yx, letters, sounds = scorer2matrices(smoothed.theil_u())
>>> xy.shape, yx.shape
((9, 8), (8, 9))
>>> letters[:4], sounds[:4]
(['a', 'c', 'e', 'i'], ['a', 'e', 'i', 'k'])
```

`xy[i, j]` is the score of `letters[i]` predicting `sounds[j]`, and `yx[j, i]`
that of `sounds[j]` predicting `letters[i]`.
[Plot score matrices](../how-to/plot.md) draws them as heatmaps.

## Next steps

- [Directional association](../concepts/directional-association.md) and
  [Notation](../concepts/notation.md) explain what the scores mean.
- [Pitfalls](../concepts/pitfalls.md) lists the surprises to watch for.
- The [measure pages](../measures/index.md) define each measure.
- The [API reference](../api/index.md) documents every function.
