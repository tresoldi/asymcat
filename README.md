# ASymCat

[![CI](https://github.com/tresoldi/asymcat/actions/workflows/quality.yml/badge.svg)](https://github.com/tresoldi/asymcat/actions/workflows/quality.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://asymcat.tresoldi.org/)
[![PyPI version](https://badge.fury.io/py/asymcat.svg)](https://badge.fury.io/py/asymcat)
[![Python versions](https://img.shields.io/pypi/pyversions/asymcat.svg)](https://pypi.org/project/asymcat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Measure association between categorical variables — separately in each direction.**

Most association measures are *symmetric*: they report one number for a pair of
variables, as if the relationship between `X` and `Y` were the same as between
`Y` and `X`. Many real relationships are not. ASymCat scores association in both
directions across a dozen probabilistic, information-theoretic, and statistical
measures, behind one consistent interface.

```python
import asymcat

data = asymcat.read_sequences("data.tsv")
coocs = asymcat.collect_cooccs(data)
scorer = asymcat.CatScorer(coocs)

scorer.mle()[("a", "x")]  # (0.83, 0.20)  — P(x|a) vs P(a|x)
```

That tuple is the whole point: a symmetric summary would collapse the two numbers
into one and hide exactly the structure ASymCat is built to reveal — which
variable predicts which, and how strongly, in each direction.

## Install

```bash
pip install asymcat
```

## The interface

The workflow is always the same: read data into paired sequences (or a
presence–absence matrix), collect the co-occurrences, build a `CatScorer`, then
call a measure. Every measure returns a mapping from category pairs to a
`(x→y, y→x)` tuple.

```python
import asymcat

data = asymcat.read_sequences("data.tsv")  # or asymcat.read_pa_matrix(...)
coocs = asymcat.collect_cooccs(data)  # order=2, pad="#" for n-grams
scorer = asymcat.CatScorer(coocs, smoothing_method="laplace", smoothing_alpha=1.0)

scorer.mle()  # P(y|x), P(x|y)
scorer.theil_u()  # uncertainty coefficient in each direction
scorer.pmi()  # pointwise mutual information
scorer.fisher()  # exact odds ratios

# Turn any scored measure into matrices for plotting
xy, yx, x_labels, y_labels = asymcat.scorer.scorer2matrices(scorer.pmi())
```

## Choosing a measure

| Measure | Use it for | Family |
|---------|------------|--------|
| `mle` | conditional probabilities `P(y\|x)`, `P(x\|y)` | probabilistic |
| `pmi` / `pmi_smoothed` | co-occurrence strength vs. independence | information-theoretic |
| `theil_u` | directional predictability (uncertainty coefficient) | information-theoretic |
| `cond_entropy` / `mutual_information` | information remaining / shared | information-theoretic |
| `chi2` / `cramers_v` | strength of statistical association | statistical |
| `fisher` | exact odds ratios for small samples | statistical |
| `log_likelihood_ratio` | G² association statistic | statistical |
| `goodman_kruskal_lambda` | proportional reduction in prediction error | statistical |
| `jaccard_index` | directional set overlap | set-based |
| `tresoldi` | smoothed measure tuned for sequence alignment | specialized |

**Significance and uncertainty.** The statistical tests expose matching p-value
scorers (`chi2_pvalue`, `fisher_pvalue`, `log_likelihood_ratio_pvalue`). For any
measure — including the information-theoretic ones with no closed-form null —
`permutation_pvalue(measure, ...)` estimates significance by shuffling the `x`↔`y`
pairing, and `bootstrap_ci(measure, ...)` returns percentile confidence intervals
by resampling the co-occurrences.

```python
scorer.chi2_pvalue()  # closed-form p-values
scorer.permutation_pvalue("theil_u", n_permutations=1000, seed=0)
scorer.bootstrap_ci("theil_u", n_bootstrap=1000, confidence_level=0.95, seed=0)
```

## Why ASymCat

- **Directional by construction** — every measure reports `x→y` and `y→x`
  separately, surfacing asymmetries symmetric measures average away.
- **One consistent API** across a dozen measures — swap `scorer.mle()` for
  `scorer.theil_u()` without changing anything else.
- **Robust smoothing** via [FreqProb](https://github.com/tresoldi/freqprob) for
  numerically stable probability estimates on sparse data.
- **Typed and tested** — full type hints (`py.typed`), strict linting and
  type-checking, and a test suite run across Python 3.10–3.12.

## Documentation

- **[Documentation site](https://asymcat.tresoldi.org/)** — user guide and full
  API reference.
- **[User Guide](docs/USER_GUIDE.md)** — concepts, measure selection, data
  preparation, and worked examples.
- **[The Tresoldi Measure](docs/TRESOLDI_MEASURE.md)** — motivation and
  definition of the specialized `tresoldi` measure.
- **[API Reference](https://asymcat.tresoldi.org/reference/)** — every public
  class and function, generated from the source.

## Applications

Directional association between categories recurs across fields: grapheme–phoneme
correspondence and sound-change directionality in **linguistics**; asymmetric
species co-occurrence in **ecology**; feature screening in **machine learning**;
and dependency analysis in **categorical analytics**. Sample datasets for several
of these live in [`resources/`](resources/).

## Citation

If you use ASymCat in academic research, please cite:

```bibtex
@software{tresoldi_asymcat,
  author  = {Tresoldi, Tiago},
  title   = {ASymCat: Asymmetric measures of association between categorical variables},
  url     = {https://github.com/tresoldi/asymcat}
}
```

## License

MIT — see [LICENSE](LICENSE).
