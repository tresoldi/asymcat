# ASymCat

[![PyPI version](https://img.shields.io/pypi/v/asymcat.svg)](https://pypi.org/project/asymcat/)
[![Python versions](https://img.shields.io/pypi/pyversions/asymcat.svg)](https://pypi.org/project/asymcat/)
[![CI](https://github.com/tresoldi/asymcat/actions/workflows/quality.yml/badge.svg)](https://github.com/tresoldi/asymcat/actions/workflows/quality.yml)
[![Docs](https://img.shields.io/badge/docs-asymcat.tresoldi.org-blue.svg)](https://asymcat.tresoldi.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23917211.svg)](https://doi.org/10.5281/zenodo.23917211)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/tresoldi/asymcat/blob/master/LICENSE)

**Measure association between categorical variables — separately in each direction.**

Most association measures are *symmetric*: they report one number for a pair of
variables, as if the relationship between `X` and `Y` were the same as between
`Y` and `X`. Many real relationships are not. ASymCat scores association in both
directions across a dozen probabilistic, information-theoretic and statistical
measures, behind one consistent interface.

If you use ASymCat in research, please cite it:
[doi:10.5281/zenodo.23917211](https://doi.org/10.5281/zenodo.23917211)
(version 0.9.0; see [Citation](#citation)).

## Install

```bash
pip install asymcat
```

ASymCat needs only FreqProb, NumPy and SciPy. For the plotting examples in the
documentation, `pip install "asymcat[viz]"` adds matplotlib, seaborn and pandas.

## Quick start

How English spellings map to sounds: the letter `q` is always pronounced /k/,
but /k/ is spelled in several ways.

```pycon
>>> from asymcat import CatScorer
>>> pairs = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4
>>> scorer = CatScorer(pairs)
>>> p_k_given_q, p_q_given_k = scorer.mle()[("q", "k")]
>>> round(p_k_given_q, 2), round(p_q_given_k, 2)
(1.0, 0.18)
```

Every measure returns, for each pair `(x, y)`, a tuple `(x→y, y→x)`. Here `q`
predicts /k/ perfectly, while /k/ points back to `q` only 18% of the time. A
symmetric measure would collapse the two numbers into one and hide exactly that
difference.

## From sequences to scores

ASymCat collects co-occurrences from paired sequences, such as aligned spellings
and pronunciations, and from presence–absence matrices (`read_sequences` and
`read_pa_matrix` load them from files). Any measure can then be scored, tested
and plotted the same way:

```pycon
>>> import asymcat
>>> words = [("cat", "kat"), ("cot", "kot"), ("city", "siti"), ("kit", "kit"), ("cent", "sent")]
>>> cooccs = asymcat.collect_cooccs(words, order=1)  # align position by position
>>> scorer = asymcat.CatScorer(cooccs, smoothing_method="laplace")
>>> [round(v, 2) for v in scorer.theil_u()[("c", "k")]]  # uncertainty coefficient
[0.18, 0.24]
>>> pvalues = scorer.permutation_pvalue("mle", n_permutations=999, seed=0)
>>> [round(float(p), 3) for p in pvalues[("c", "k")]]
[0.137, 0.137]
>>> xy, yx, x_labels, y_labels = asymcat.scorer.scorer2matrices(scorer.pmi())
>>> xy.shape  # one row per spelling symbol, one column per sound, ready to plot
(9, 8)
```

For sparse data, `CatScorer(..., entropy_estimator="chao_shen")` or `"nsb"`
replaces the plug-in entropy behind the information-theoretic measures with a
bias-corrected estimate.

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
| `jaccard_index` | set overlap | set-based |
| `tresoldi` | PMI tempered by conditional probability, for alignments | specialized |

The statistical tests also expose p-values (`chi2_pvalue`, `fisher_pvalue`,
`log_likelihood_ratio_pvalue`). For any measure, `permutation_pvalue` and
`bootstrap_ci` give resampling-based p-values and confidence intervals.
[Choosing a measure](https://asymcat.tresoldi.org/concepts/choosing-a-measure/)
helps pick one, and each has a
[page of its own](https://asymcat.tresoldi.org/measures/) with its definition
and a worked example.

## Highlights

- **Directional by construction:** every measure reports `x→y` and `y→x`
  separately.
- **One interface for every measure:** swap `scorer.mle()` for
  `scorer.theil_u()` without changing anything else.
- **Built for sparse data:** smoothing and bias-corrected entropy estimation
  from [FreqProb](https://freqprob.tresoldi.org/).
- **Domain-neutral:** linguistics, ecology, machine learning and categorical
  analytics; the repository ships
  [sample datasets](https://github.com/tresoldi/asymcat/tree/master/resources)
  with their sources and licences.
- **Typed and tested:** full type hints (`py.typed`), with tests on Python
  3.10–3.12 across Linux, macOS and Windows.

## Documentation

- **[Tutorial](https://asymcat.tresoldi.org/getting-started/tutorial/):** from
  pairs of observations to directional scores, tests and matrices.
- **[Concepts](https://asymcat.tresoldi.org/concepts/directional-association/):**
  directional association, notation, choosing a measure and common pitfalls.
- **[Measures](https://asymcat.tresoldi.org/measures/):** one page per measure,
  with its definition, a worked example, its properties and references.
- **[How-to guides](https://asymcat.tresoldi.org/how-to/):** preparing data,
  sparse data, significance testing, ranking scores and plotting.
- **[API Reference](https://asymcat.tresoldi.org/api/):** every public class
  and function, generated from the source.
- **[References](https://asymcat.tresoldi.org/references/):** the published
  source of each measure, with a BibTeX file.
- **[Changelog](https://github.com/tresoldi/asymcat/blob/master/CHANGELOG.md)**
  and **[Migration guide](https://github.com/tresoldi/asymcat/blob/master/MIGRATION.md)**.

## Related projects

- **[FreqProb](https://freqprob.tresoldi.org/)** provides ASymCat's smoothed
  probabilities and bias-corrected entropy estimators.
- **SciPy's
  [`scipy.stats.contingency.association`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html)**
  computes symmetric association for a whole contingency table (Cramér's V,
  Tschuprow's T, Pearson's contingency coefficient). ASymCat adds directional
  measures and scores every pair of categories.

## Citation

If you use ASymCat in published work, please cite the version you used:

```bibtex
@software{tresoldi_asymcat,
  author  = {Tresoldi, Tiago},
  title   = {{ASymCat}: Asymmetric measures of association between
             categorical variables},
  year    = {2026},
  version = {0.9.0},
  doi     = {10.5281/zenodo.23917211},
  url     = {https://asymcat.tresoldi.org/}
}
```

The DOI [10.5281/zenodo.23917211](https://doi.org/10.5281/zenodo.23917211) always resolves to the latest
version, and the Zenodo record lists the DOI of each version. See
[How to cite](https://asymcat.tresoldi.org/citing/) for APA and classic BibTeX
forms, and [References](https://asymcat.tresoldi.org/references/) for the papers
behind each measure. Citation metadata is also in
[`CITATION.cff`](https://github.com/tresoldi/asymcat/blob/master/CITATION.cff).

## License

MIT; see [LICENSE](https://github.com/tresoldi/asymcat/blob/master/LICENSE).
