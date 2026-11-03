# API reference

The reference is generated from the docstrings in the source, so it always
matches the installed version. It is organised by module:

| Module | Contents |
|--------|----------|
| [`asymcat.scorer.CatScorer`](catscorer.md) | The scorer: every association measure, smoothing, entropy estimation, permutation tests and bootstrap intervals |
| [`asymcat.scorer` functions](scorer-functions.md) | Measures on raw series or tables, and helpers to scale, invert and reshape scores |
| [`asymcat.common`](common.md) | Reading data and collecting co-occurrences, n-grams, observations and contingency tables |
| [`asymcat.correlation`](correlation.md) | Association between two whole variables |

The top-level package re-exports the names most often needed:

```pycon
>>> import asymcat
>>> sorted(asymcat.__all__)
['CatScorer', 'build_ct', 'collect_alphabets', 'collect_cooccs', 'collect_ngrams', 'collect_observations', 'correlation', 'read_pa_matrix', 'read_sequences', 'scorer']
```

Each [measure page](../measures/index.md) also includes the reference entry
of its methods, next to their definition and examples.
