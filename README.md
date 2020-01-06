# catcoocc

[![Build Status](https://travis-ci.org/tresoldi/catcoocc.svg?branch=master)](https://travis-ci.org/tresoldi/catcoocc)
[![codecov](https://codecov.io/gh/tresoldi/catcoocc/branch/master/graph/badge.svg)](https://codecov.io/gh/tresoldi/catcoocc)
[![Codacy
Badge](https://api.codacy.com/project/badge/Grade/0f820951c6374be29717a02471a3fd45)](https://www.codacy.com/manual/tresoldi/catcoocc?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=tresoldi/catcoocc&amp;utm_campaign=Badge_Grade)

Library for symmetrical and assymetrical analysis of categorical co-occurrences

## Installation and usage

The library can be installed as any standard Python library with `pip`:

```bash
pip install catcoocc
```

Detailed instructions on how to use the library can be found in
the [official documentation]().

A show-case example is shown here:

```python
import tabulate
import catcoocc
from catcoocc.scorer import CatScorer

mushroom_data = catcoocc.read_sequences("docs/mushroom-small.tsv")
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
catcoocc_i = scorer.catcoocc_i()
catcoocc_ii = scorer.catcoocc_ii()

headers = [
    'pair',
    'mle_0',        'mle_1', 
    'pmi_0',        'pmi_1', 
    'npmi_0',       'npmi_1', 
    'chi2_0',       'chi2_1', 
    'chi2ns_0',     'chi2ns_1', 
    'cremersv_0',   'cremersv_1', 
    'cremersvns_0', 'cremersvns_1', 
    'fisher_0',     'fisher_1', 
    'theilu_0',     'theilu_1', 
    'catcoocci_0',  'catcoocci_1', 
    'catcooccii_0', 'catcooccii_1', 
]

table = []
for pair in sorted(scorer.obs):
    buf = [
        pair,
        "%0.4f" % mle[pair][0],         "%0.4f" % mle[pair][1],
        "%0.4f" % pmi[pair][0],         "%0.4f" % pmi[pair][1],
        "%0.4f" % npmi[pair][0],        "%0.4f" % npmi[pair][1],
        "%0.4f" % chi2[pair][0],        "%0.4f" % chi2[pair][1],
        "%0.4f" % chi2_ns[pair][0],     "%0.4f" % chi2_ns[pair][1],
        "%0.4f" % cramersv[pair][0],    "%0.4f" % cramersv[pair][1],
        "%0.4f" % cramersv_ns[pair][0], "%0.4f" % cramersv_ns[pair][1],
        "%0.4f" % fisher[pair][0],      "%0.4f" % fisher[pair][1],
        "%0.4f" % theil_u[pair][0],     "%0.4f" % theil_u[pair][1],
        "%0.4f" % catcoocc_i[pair][0],  "%0.4f" % catcoocc_i[pair][1],
        "%0.4f" % catcoocc_ii[pair][0], "%0.4f" % catcoocc_ii[pair][1],
    ]
    table.append(buf)

    
print(tabulate.tabulate(table, headers=headers, tablefmt='markdown'))
```

Which will output:

```bash
pair                       mle_0    mle_1    pmi_0    pmi_1    npmi_0    npmi_1    chi2_0    chi2_1    chi2ns_0    chi2ns_1    cremersv_0    cremersv_1    cremersvns_0    cremersvns_1    fisher_0    fisher_1    theilu_0    theilu_1    catcoocci_0    catcoocci_1    catcooccii_0    catcooccii_1
-----------------------  -------  -------  -------  -------  --------  --------  --------  --------  ----------  ----------  ------------  ------------  --------------  --------------  ----------  ----------  ----------  ----------  -------------  -------------  --------------  --------------
('edible', 'bell')        0.3846   1        0.4308   0.4308    0.3107    0.3107    1.8315    1.8315      3.5897      3.5897        0.2027        0.2027          0.1987          0.1987         inf         inf      1           0.3985         0.4308         0.1717          0.789           0.789
('edible', 'convex')      0.4615   0.4615  -0.3424  -0.3424   -0.2844   -0.2844    3.6735    3.6735      5.7988      5.7988        0.3719        0.3719          0.3101          0.3101           0           0      0.2955      0.1823        -0.1012        -0.0624         -1.2578         -1.2578
('edible', 'flat')        0.0769   1        0.4308   0.4308    0.1438    0.1438    0.1041    0.1041      0.5668      0.5668        0             0               0               0              inf         inf      1           1              0.4308         0.4308          0.0448          0.0448
('edible', 'sunken')      0.0769   1        0.4308   0.4308    0.1438    0.1438    0.1041    0.1041      0.5668      0.5668        0             0               0               0              inf         inf      1           1              0.4308         0.4308          0.0448          0.0448
('poisonous', 'bell')     0        0       -3.5553  -3.5553   -0.5934   -0.5934    1.8315    1.8315      3.5897      3.5897        0.2027        0.2027          0.1987          0.1987           0           0      1           1             -3.5553        -3.5553         -6.5116         -6.5116
('poisonous', 'convex')   1        0.5385   0.4308   0.4308    0.4103    0.4103    3.6735    3.6735      5.7988      5.7988        0.3719        0.3719          0.3101          0.3101         inf         inf      0.0105      1              0.0045         0.4308          1.5825          1.5825
('poisonous', 'flat')     0        0       -1.9459  -1.9459   -0.3248   -0.3248    0.1041    0.1041      0.5668      0.5668        0             0               0               0                0           0      1           1             -1.9459        -1.9459         -0.2026         -0.2026
('poisonous', 'sunken')   0        0       -1.9459  -1.9459   -0.3248   -0.3248    0.1041    0.1041      0.5668      0.5668        0             0               0               0                0           0      1           1             -1.9459        -1.9459         -0.2026         -0.2026
```

## Related Projects

https://github.com/pafoster/pyitlib

Griffith, Daniel M.; Veech, Joseph A.; and Marsh, Charles J. (2016)
*cooccur: Probabilistic Species Co-Occurrence Analysis in R*. Journal
of Statistical Software (69). doi: 10.18627/jss.v069.c02

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
