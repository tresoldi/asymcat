# catcoocc

[![Build Status](https://travis-ci.org/tresoldi/catcoocc.svg?branch=master)](https://travis-ci.org/tresoldi/catcoocc)
[![codecov](https://codecov.io/gh/tresoldi/catcoocc/branch/master/graph/badge.svg)](https://codecov.io/gh/tresoldi/catcoocc)
[![Codacy
Badge](https://api.codacy.com/project/badge/Grade/0f820951c6374be29717a02471a3fd45)](https://www.codacy.com/manual/tresoldi/catcoocc?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=tresoldi/catcoocc&amp;utm_campaign=Badge_Grade)

The `catcoocc` library is designed for the study of co-occurrence association
between categorical variables by implementing a number of symmetric and
asymmetric measures of association.
Given a series of
co-occurrence observations, starting from data such as records,
alignments, and matrices of presence-absence, it allows to compute
dictionaries with the association score between categories, offering
methods focused on strength of association, direction of association, or
both. It is primarily developed for linguistic research, but can be
applied to any kind of data exploration and description based on
categorical data; besides the main methods for numeric computation of
measures of
association, it includes auxiliary ones for dealing with relational data,
n-grams from sequences, alignments, and binary matrices of
presence/absence.


## Background

A measure of association is a factor or coefficient used to quantify
the relationship between two or more variables. Various measures exist to
determine the strength and relationship of such associations, the most
common being measures of *correlation* which, in a sense stricter than
*association*, refers to linear correlation. Among the most common measures,
are Pearson's **rho** coefficient of product-moment correlation for
continuous values, Spearman **rho** coefficient for measuring the strenght
of monotonic ordinal or ranked variables, and Chi-square measure for
association between categorical values. Each measure is usually indicated
to investigate either strength (such as Pearson's **rho**) or significance
(such as Chi-square), and most are **symmetric**, meaning that, when
measuring the relationship between series `X` and series `Y`,
 the association between any `x` and `y` value is equal to that
between `y` and `x`.

While symmetric measures are the natural measure for numeric variables,
the analyses arising from many studies and applications for categorical
variables can in most cases benefit from asymmetric measures, as
the fraction of variability in `x` that is explainable by variations in `y`
(Pearson, 2016). Such property can be easily demonstrated by modifying the
example given by (Zychlinski, 2018) while introducing his `dython`
library

  |               | X | Y |
  |---------------|---|---|
  | Observation 1 | A | c |
  | Observation 2 | A | d |
  | Observation 3 | A | c |
  | Observation 4 | B | g |
  | Observation 5 | B | g |
  | Observation 6 | B | f |

In this example, the categorical value of `y` cannot be determined with full
certainty given `x`, but `x` can be determined with certainty from `y`. In
a symmetric version Maximum-Likelihood estimation (MLE), which just divides
the number of cases for the total number of observations (i.e., Cxy/Cx and
Cxy/Cy, where C is the overall count), the tables the XY and YX
are the transposed version of each other:

  | X given Y | `A`  | `B`  |
  |-----------|------|------|
  | **`c`**   | 0.75 | 0.75 |
  | **`d`**   | 0.00 | 0.00 |
  | **`f`**   | 0.00 | 0.00 |
  | **`g`**   | 0.75 | 0.75 |

  | Y given X | `c`  | `d`  | `f`  | `g`  |
  |-----------|------|------|------|------|
  | **`A`**   | 0.75 | 0.00 | 0.00 | 0.75 |
  | **`B`**   | 0.75 | 0.00 | 0.00 | 0.75 |

With the same MLE scorer, asymmetric tables are
able to capture the difference in information expressing that, if we know
`y` in this simple dataset, we can predict `x` with certainty.

  | X given Y  | `A`  | `B`  |
  |------------|------|------|
  | **`c`**    | 1.00 | 0.00 |
  | **`d`**    | 1.00 | 0.00 |
  | **`f`**    | 0.00 | 1.00 |
  | **`g`**    | 0.00 | 1.00 |

  | Y given X | `c`  | `d`  | `f`  | `g`  |
  |-----------|------|------|------|------|
  | **`A`**   | 0.67 | 0.33 | 0.00 | 0.00 |
  | **`B`**   | 0.00 | 0.00 | 0.33 | 0.67 |

The most popular methods for measure of categorical association are the
aforementioned Chi-square and Cramer's V, defined as the square root of a
normalized chi-square value. Both are symmetric values. Among the best
known asymmetric measures are Theil's U and Goodman and Kruskal's tau.
The former is particularly useful for domains of the humanities such as
lingustic research, as it is ultimately based on the conditional entropy
between `x` and `y`, that is, how many possible states of `y` are observed
given `x` and how often they occur.

The following scorers are implemented:

- Maximum-Likelihood Estimation
- Pointwise Mutual Information
- Normalized Pointwise Mutual Information
- Chi-square (over both 2x2 and 3x2 contingency tables)
- Cramér's V (over both 2x2 and 3x2 contingency tables)
- Fisher Exact Odds Ratio (over unconditional MLE)
- Theil's U ("uncertainty score")
- Conditional Entropy
- A new scorer `tresoldi`, for the study of linguistic alignment
  (combining information from MLE and PMI)

The library also offers functions for scaling scores with user-determined
ranges using different methods (`minmax`, `mean`, and `stdev`) as well
as functions for plotting heatmaps of the scorers. The same dataset of
above plotted with the `tresoldi` scorer, where positive numbers indicate
co-occurrence and negative numbers indicate no co-occurrence (with the
larger the number, the higher the degree of confidence), results in the
following heatmaps:

![Table 1, tresoldi, xy](https://raw.githubusercontent.com/tresoldi/catcoocc/master/docs/zychlinski_tresoldi_xy.png)

![Table 1, tresoldi, yx](https://raw.githubusercontent.com/tresoldi/catcoocc/master/docs/zychlinski_tresoldi_yx.png)

## Installation and usage

The library can be installed as any standard Python library with `pip`:

```bash
pip install catcoocc
```

Detailed instructions on how to use the library can be found in
the [official documentation]().

A show-case example with a subset of the `mushroom` dataset is shown here:

```python
import tabulate
import catcoocc
from catcoocc.scorer import CatScorer

mushroom_data = catcoocc.read_sequences("resources/mushroom-small.tsv")
mushroom_cooccs = catcoocc.collect_cooccs(mushroom_data)
scorer = catcoocc.scorer.CatScorer(mushroom_cooccs)

mle = scorer.mle()
pmi = scorer.pmi()
npmi = scorer.pmi(True)
chi2 = scorer.chi2()
chi2_ns = scorer.chi2(False)
cramersv = scorer.cramers_v()
cramersv_ns = scorer.cramers_v(False)
fisher = scorer.fisher()
theil_u = scorer.theil_u()
cond_entropy = scorer.cond_entropy()
tresoldi = scorer.tresoldi()

headers = [
    'pair',
    'mle_xy',          'mle_yx',
    'pmi_xy',          'pmi_yx',
    'npmi_xy',         'npmi_yx',
    'chi2_xy',         'chi2_yx',
    'chi2ns_xy',       'chi2ns_yx',
    'cremersv_xy',     'cremersv_yx',
    'cremersvns_xy',   'cremersvns_yx',
    'fisher_xy',       'fisher_yx',
    'theilu_xy',       'theilu_yx',
    'cond_entropy_xy', 'cond_entropy_yx',
    'tresoldi_xy',     'tresoldi_yx'
]

table = []
for pair in sorted(scorer.obs):
    buf = [
        pair,
        "%0.4f" % mle[pair][0],          "%0.4f" % mle[pair][1],
        "%0.4f" % pmi[pair][0],          "%0.4f" % pmi[pair][1],
        "%0.4f" % npmi[pair][0],         "%0.4f" % npmi[pair][1],
        "%0.4f" % chi2[pair][0],         "%0.4f" % chi2[pair][1],
        "%0.4f" % chi2_ns[pair][0],      "%0.4f" % chi2_ns[pair][1],
        "%0.4f" % cramersv[pair][0],     "%0.4f" % cramersv[pair][1],
        "%0.4f" % cramersv_ns[pair][0],  "%0.4f" % cramersv_ns[pair][1],
        "%0.4f" % fisher[pair][0],       "%0.4f" % fisher[pair][1],
        "%0.4f" % theil_u[pair][0],      "%0.4f" % theil_u[pair][1],
        "%0.4f" % cond_entropy[pair][0], "%0.4f" % cond_entropy[pair][1],
        "%0.4f" % tresoldi[pair][0],     "%0.4f" % tresoldi[pair][1],
    ]
    table.append(buf)


print(tabulate.tabulate(table, headers=headers, tablefmt='markdown'))
```

Which will output:

| pair                    |   mle_xy |   mle_yx |   pmi_xy |   pmi_yx |   npmi_xy |   npmi_yx |   chi2_xy |   chi2_yx |   chi2ns_xy |   chi2ns_yx |   cremersv_xy |   cremersv_yx |   cremersvns_xy |   cremersvns_yx |   fisher_xy |   fisher_yx |   theilu_xy |   theilu_yx |   cond_entropy_xy |   cond_entropy_yx |   tresoldi_xy |   tresoldi_yx |
|-------------------------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|-------------|-------------|---------------|---------------|-----------------|-----------------|-------------|-------------|-------------|-------------|-------------------|-------------------|---------------|---------------|
| ('edible', 'bell')      |   0.3846 |   1      |   0.4308 |   0.4308 |    0.3107 |    0.3107 |    1.8315 |    1.8315 |      3.5897 |      3.5897 |        0.2027 |        0.2027 |          0.1987 |          0.1987 |         inf |         inf |      0      |      1      |            1.119  |            0      |        0.5956 |        1      |
| ('edible', 'convex')    |   0.4615 |   0.4615 |  -0.3424 |  -0.3424 |   -0.2844 |   -0.2844 |    3.6735 |    3.6735 |      5.7988 |      5.7988 |        0.3719 |        0.3719 |          0.3101 |          0.3101 |           0 |           0 |      0.2147 |      0.3071 |            0.7273 |            0.4486 |       -0.5615 |       -0.5615 |
| ('edible', 'flat')      |   0.0769 |   1      |   0.4308 |   0.4308 |    0.1438 |    0.1438 |    0.1041 |    0.1041 |      0.5668 |      0.5668 |        0      |        0      |          0      |          0      |         inf |         inf |      0      |      1      |            1.119  |            0      |        0.4596 |        1      |
| ('edible', 'sunken')    |   0.0769 |   1      |   0.4308 |   0.4308 |    0.1438 |    0.1438 |    0.1041 |    0.1041 |      0.5668 |      0.5668 |        0      |        0      |          0      |          0      |         inf |         inf |      0      |      1      |            1.119  |            0      |        0.4596 |        1      |
| ('poisonous', 'bell')   |   0      |   0      |  -3.5553 |  -3.5553 |   -0.5934 |   -0.5934 |    1.8315 |    1.8315 |      3.5897 |      3.5897 |        0.2027 |        0.2027 |          0.1987 |          0.1987 |           0 |           0 |      1      |      1      |            0      |            0      |       -3.5553 |       -3.5553 |
| ('poisonous', 'convex') |   1      |   0.5385 |   0.4308 |   0.4308 |    0.4103 |    0.4103 |    3.6735 |    3.6735 |      5.7988 |      5.7988 |        0.3719 |        0.3719 |          0.3101 |          0.3101 |         inf |         inf |      1      |      0      |            0      |            0.6902 |        1      |        0.6779 |
| ('poisonous', 'flat')   |   0      |   0      |  -1.9459 |  -1.9459 |   -0.3248 |   -0.3248 |    0.1041 |    0.1041 |      0.5668 |      0.5668 |        0      |        0      |          0      |          0      |           0 |           0 |      1      |      1      |            0      |            0      |       -1.9459 |       -1.9459 |
| ('poisonous', 'sunken') |   0      |   0      |  -1.9459 |  -1.9459 |   -0.3248 |   -0.3248 |    0.1041 |    0.1041 |      0.5668 |      0.5668 |        0      |        0      |          0      |          0      |           0 |           0 |      1      |      1      |            0      |            0      |       -1.9459 |       -1.9459 |

## Changelog

Version 0.2.1:

  - Added basic functions for double series correlation

## Similar Projects

https://github.com/pafoster/pyitlib

Griffith, Daniel M.; Veech, Joseph A.; and Marsh, Charles J. (2016)
*cooccur: Probabilistic Species Co-Occurrence Analysis in R*. Journal
of Statistical Software (69). doi: 10.18627/jss.v069.c02

https://cran.r-project.org/web/packages/GoodmanKruskal/vignettes/GoodmanKruskal.html

## Community guidelines

While the author can be contacted directly for support, it is recommended
that third parties use GitHub standard features, such as issues and
pull requests, to contribute, report problems, or seek support.

## Author and citation

The library is developed by Tiago Tresoldi (tresoldi@shh.mpg.de).

The author has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation
programme (grant agreement
No. [ERC Grant #715618](https://cordis.europa.eu/project/rcn/206320/factsheet/en),
[Computer-Assisted Language Comparison](https://digling.org/calc/).

If you use `catcoocc`, please cite it as:

> Tresoldi, Tiago (2020). `catcoocc`, a library for symmetric and asymmetric
analysis of categorical co-occurrences. Version 0.1. Jena. Available at:
> <https://github.com/tresoldi/catcoocc>

In BibTeX:

```bibtex
@misc{Tresoldi2020catcoocc,
  author = {Tresoldi, Tiago},
  title = {catcoocc, a library for symmetric and asymmetric analysis of categorical co-occurrences. Version 0.1.},
  howpublished = {\url{https://github.com/tresoldi/catcoocc}},
  address = {Jena},
  year = {2020},
}
```
