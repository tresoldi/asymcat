# ASymCat

ASymCat is a powerful Python library designed to analyze co-occurrence association between categorical
variables using various symmetric and asymmetric measures of association. Whether you are working with
linguistic data, or any other kind of categorical data, ASymCat can help you explore and describe your
data in new and meaningful ways.

Using ASymCat, you can input co-occurrence observations such as records, alignments, and matrices of
presence-absence, and generate dictionaries that provide association scores between categories. You can also
focus on the strength or direction of association, or both, depending on your research questions.

In addition to the main methods for numeric computation of measures of association, ASymCat includes auxiliary
methods that can help you deal with relational data, n-grams from sequences, alignments, and binary
matrices of presence/absence.

ASymCat is a versatile library that can be used by researchers in various fields, and it can help you gain
new insights into your data.

## Background

Measures of association play a crucial role in statistical analysis as they help to quantify the relationship
between variables. While a variety of measures exist, correlation coefficients are widely used to determine
the strength and direction of the relationship between two variables. Pearson's **rho**, for example,
is frequently used to assess the strength of the linear relationship between continuous variables, whereas
Spearman's **rho** is employed to evaluate the strength of the monotonic relationship between ordinal or
ranked variables. The Chi-square measure, on the other hand, is a useful tool for measuring the
association between categorical variables.

It is important to note that most measures of association are symmetric, indicating that the association
between any `x` and `y` value is equal to that between `y` and `x`. In general, these measures are
used to investigate either the strength or significance of the association.

For researchers in various fields, an understanding of measures of association is essential for
conducting meaningful statistical analyses. By using appropriate measures of association, researchers can
gain valuable insights into the relationships between variables, draw conclusions, and make informed decisions.

While symmetric measures are commonly used to measure the relationship between numeric variables, studies
and applications involving categorical variables would typically benefit from the use of asymmetric measures.
Asymmetric measures are useful in determining the fraction of variability in `x` that can be explained
by variations in `y` (Pearson, 2016). This property is easily demonstrated through a modification
of the example provided by Zychlinski (2018), with the introduction of the `dython` library:

  |               | X | Y |
  |---------------|---|---|
  | Observation 1 | A | c |
  | Observation 2 | A | d |
  | Observation 3 | A | c |
  | Observation 4 | B | g |
  | Observation 5 | B | g |
  | Observation 6 | B | f |

In the given example, it is not possible to determine the categorical value of `y` with complete certainty
based on `x`. However, it is possible to determine `x` with certainty based on `y`. The symmetric version
of Maximum-Likelihood estimation (MLE), which simply divides the number of cases by the total number
of observations (i.e., Cxy/Cx and Cxy/Cy, where C represents the overall count), produces tables that
are transposed versions of each other for XY and YX.

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

In contrast to symmetric tables, asymmetric tables can capture the difference in information between variables,
indicating that if `y` is known in a dataset, `x` can be predicted with certainty. Using the
Maximum-Likelihood estimation scorer, we can see that the asymmetric tables more accurately reflect
this relationship:

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

The assessment of categorical association typically involves the use of measures such as Chi-square
and Cramer's V, which are symmetric in nature. However, there are some widely-used asymmetric
measures that include Theil's U and Goodman and Kruskal's tau. The Theil's U measure is
well-suited for fields in the humanities, including linguistic research, as it considers the conditional
entropy between `x` and `y`, which reflects the number of possible states of `y` given `x` and
their frequency of occurrence.

In the ASymCat library, various scoring methods are available to measure asymmetric associations,
including Maximum-Likelihood Estimation, Pointwise Mutual Information, Normalized Pointwise Mutual
Information, Chi-square (over both 2x2 and 3x2 contingency tables), Cramér's V (over both 2x2
and 3x2 contingency tables), Fisher Exact Odds Ratio (over unconditional MLE), Theil's U
("uncertainty score"), Conditional Entropy, and a new scorer `tresoldi` that combines information
from MLE and PMI, tailored for studies in computational historical linguistics.


The most popular methods for measure of categorical association are the
aforementioned Chi-square and Cramer's V, defined as the square root of a
normalized chi-square value. Both are symmetric values. Among the best
known asymmetric measures are Theil's U and Goodman and Kruskal's tau.
The former is particularly useful for domains of the humanities such as
lingustic research, as it is ultimately based on the conditional entropy
between `x` and `y`, that is, how many possible states of `y` are observed
given `x` and how often they occur.

In addition to the various categorical association scorers, the ASymCat library provides several
useful features. Users can scale the scores with their desired ranges using methods such as
`minmax`, `mean`, and `stdev`. The library also includes functions for generating heatmaps of the
scorers. Moreover, it provides a range of smoothing methods for frequency counts, including Laplace
smoothing and more sophisticated approaches like the Certainty Degree scorer, which was specifically
designed for use with this library.

## Installation and usage

The library can be installed as any standard Python library with `pip`:

```bash
pip install asymcat
```

Detailed instructions on how to use the library can be found in
the [official documentation]().

A show-case example with a subset of the `mushroom` dataset is shown here:

```python
import tabulate
import asymcat
from asymcat.scorer import CatScorer

mushroom_data = asymcat.read_sequences("resources/mushroom-small.tsv")
mushroom_cooccs = asymcat.collect_cooccs(mushroom_data)
scorer = asymcat.scorer.CatScorer(mushroom_cooccs)

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
  'mle_xy', 'mle_yx',
  'pmi_xy', 'pmi_yx',
  'npmi_xy', 'npmi_yx',
  'chi2_xy', 'chi2_yx',
  'chi2ns_xy', 'chi2ns_yx',
  'cremersv_xy', 'cremersv_yx',
  'cremersvns_xy', 'cremersvns_yx',
  'fisher_xy', 'fisher_yx',
  'theilu_xy', 'theilu_yx',
  'cond_entropy_xy', 'cond_entropy_yx',
  'tresoldi_xy', 'tresoldi_yx'
]

table = []
for pair in sorted(scorer.obs):
  buf = [
    pair,
    "%0.4f" % mle[pair][0], "%0.4f" % mle[pair][1],
    "%0.4f" % pmi[pair][0], "%0.4f" % pmi[pair][1],
    "%0.4f" % npmi[pair][0], "%0.4f" % npmi[pair][1],
    "%0.4f" % chi2[pair][0], "%0.4f" % chi2[pair][1],
    "%0.4f" % chi2_ns[pair][0], "%0.4f" % chi2_ns[pair][1],
    "%0.4f" % cramersv[pair][0], "%0.4f" % cramersv[pair][1],
    "%0.4f" % cramersv_ns[pair][0], "%0.4f" % cramersv_ns[pair][1],
    "%0.4f" % fisher[pair][0], "%0.4f" % fisher[pair][1],
    "%0.4f" % theil_u[pair][0], "%0.4f" % theil_u[pair][1],
    "%0.4f" % cond_entropy[pair][0], "%0.4f" % cond_entropy[pair][1],
    "%0.4f" % tresoldi[pair][0], "%0.4f" % tresoldi[pair][1],
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


The library is developed by Tiago Tresoldi (tiago.tresoldi@lingfil.uu.se). The library is developed in the context of
the [Cultural Evolution of Texts](https://github.com/evotext/) project, with funding from the
[Riksbankens Jubileumsfond](https://www.rj.se/) (grant agreement ID:
[MXM19-1087:1](https://www.rj.se/en/anslag/2019/cultural-evolution-of-texts/)).

During the first stages of development, the author received funding from the
[European Research Council](https://erc.europa.eu/) (ERC) under the European Union’s Horizon 2020
research and innovation programme (grant agreement
No. [ERC Grant #715618](https://cordis.europa.eu/project/rcn/206320/factsheet/en),
[Computer-Assisted Language Comparison](https://digling.org/calc/)).

If you use `asymcat`, please cite it as:

> Tresoldi, Tiago (2023). `asymcat`, a library for symmetric and asymmetric
analysis of categorical co-occurrences. Version 0.3. Uppsala. Available at:
> <https://github.com/tresoldi/asymcat>

In BibTeX:

```bibtex
@misc{Tresoldi2023asymcat,
  author = {Tresoldi, Tiago},
  title = {asymcat, a library for symmetric and asymmetric analysis of categorical co-occurrences. Version 0.3.},
  howpublished = {\url{https://github.com/tresoldi/asymcat}},
  address = {Uppsala},
  year = {2023},
}
```
