# Notation

This page fixes the notation used across the documentation and the example
data that the measure pages share.

## Co-occurrences

ASymCat works on a list of **co-occurrences**: pairs $(x, y)$ in which $x$ comes
from one categorical variable and $y$ from another. The two variables need not
share values. In the example below, $x$ is a letter and $y$ is a sound.

| Symbol | Meaning |
|--------|---------|
| $\mathcal{X}$, $\mathcal{Y}$ | the **alphabets**: the distinct values observed for each variable |
| $n$ | the number of co-occurrences |
| $c(x, y)$ | the number of co-occurrences of $x$ with $y$ |
| $c(x)$, $c(y)$ | the number of co-occurrences whose first element is $x$, or whose second element is $y$ |
| $P(x, y)$, $P(x)$, $P(y)$ | relative frequencies: $c(x, y)/n$, $c(x)/n$ and $c(y)/n$ |
| $P(y \mid x)$ | the conditional probability $c(x, y)/c(x)$, and likewise $P(x \mid y) = c(x, y)/c(y)$ |
| $H(\cdot)$, $I(\cdot\,;\cdot)$ | entropy and mutual information, in **nats** (natural logarithms) |

## The score tuple

Every measure is a method of [`CatScorer`](../api/catscorer.md) that returns a
dictionary. It has an entry for every pair in $\mathcal{X} \times \mathcal{Y}$,
including pairs that were never observed together, and each value is a tuple

$$
\big(\text{score}(x \to y),\ \text{score}(y \to x)\big).
$$

The first number scores $x$ as a predictor of $y$; the second scores $y$ as a
predictor of $x$. For the conditional probability, the tuple is
$\big(P(y \mid x),\ P(x \mid y)\big)$. Several measures are symmetric by
definition and repeat one value in both positions; see
[Directional association](directional-association.md).

## The example data

The measure pages use the same small dataset: how English spellings map to
sounds. The letter `q` is always pronounced /k/, while /k/ is spelled `c`, `k`
or `q`, and `c` can also be /s/.

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> scorer.alphabet_x, scorer.alphabet_y
(['c', 'k', 'q'], ['k', 's'])
>>> len(pairs)
15
```

Arranged as a table of counts $c(x, y)$:

|         | /k/ | /s/ | $c(x)$ |
|---------|----:|----:|-------:|
| `c`     |   6 |   4 |     10 |
| `k`     |   3 |   0 |      3 |
| `q`     |   2 |   0 |      2 |
| $c(y)$  |  11 |   4 | $n = 15$ |

## Two views of a pair

Measures look at a pair $(x, y)$ in one of two ways.

### The 2×2 contingency table

The statistical measures ([chi-square](../measures/chi2.md),
[Cramér's V](../measures/chi2.md), [Fisher's exact test](../measures/fisher.md)
and the [log-likelihood ratio](../measures/log-likelihood-ratio.md)) classify
**all** $n$ co-occurrences by whether their first element is $x$ and whether
their second element is $y$:

|              | $y$ | not $y$ |
|--------------|-----|---------|
| $x$          | $c(x, y)$ | $c(x) - c(x, y)$ |
| not $x$      | $c(y) - c(x, y)$ | $n - c(x) - c(y) + c(x, y)$ |

For `c` and /k/:

```pycon
>>> from asymcat.common import build_ct
>>> build_ct(scorer.obs[("c", "k")])
[[6, 4], [5, 0]]
```

These measures also accept `square_ct=False`, which is deprecated: it uses the
2×3 table that `build_ct(..., square=False)` returns, which adds the row totals
$c(x)$ and $n - c(x)$ as a first column and is not a contingency table of the
pair.

### The pair subset

The information-theoretic measures ([Theil's U](../measures/theil-u.md),
[conditional entropy](../measures/conditional-entropy.md),
[mutual information](../measures/mutual-information.md)) and
[Goodman–Kruskal λ](../measures/goodman-kruskal-lambda.md) are computed on the
**pair subset** $S(x, y)$: the co-occurrences whose first element is $x$ **or**
whose second element is $y$. In the table of counts, this is the row of $x$ and
the column of $y$. Within the subset, the first and second elements are treated
as two random variables, $X_S$ and $Y_S$, and the measure is computed for them.

For `c` and /k/, the row of `c` (10 pairs) and the column of /k/ (11 pairs)
share the 6 pairs of `c` with /k/, so $|S| = 10 + 11 - 6 = 15$: here the subset
is the whole dataset. For `q` and /k/, the row of `q` lies inside the column of
/k/, so the subset is that column: 11 pairs, all with $y =$ /k/.

The pair subset makes these measures local to the pair, but it also has
consequences that the [Pitfalls](pitfalls.md) page explains: in a subset where
one variable takes a single value, its entropy is zero, and measures fall back
on a convention.
