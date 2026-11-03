# Sample datasets

These files are used by ASymCat's documentation, tests and examples. Most come
from third parties and keep their own licences: **the repository's MIT licence
covers ASymCat's code, not these datasets.** If you use one in published work,
cite its original source as given below.

None of these files is included in the packages published on PyPI.

| File | Contents | Source | Licence |
|------|----------|--------|---------|
| `cmudict.tsv` | 134,373 English words, spelled out letter by letter, with IPA transcriptions | CMUdict 0.7b | BSD-style (see below) |
| `cmudict.sample1000.tsv`, `cmudict.sample100.tsv` | 1,000- and 100-word samples of `cmudict.tsv` | CMUdict 0.7b | BSD-style (see below) |
| `english_phonology.tsv` | A handful of English words chosen to illustrate spelling–sound correspondences | Written for ASymCat, in the CMUdict format | MIT, like the code |
| `galapagos.tsv` | Presence (1) or absence (0) of 13 species of Darwin's finches on 17 Galápagos islands | Sanderson (2000) | Published factual data; cite the source |
| `mushrooms.tsv` | 1,000 instances of the UCI Mushroom data, one `feature:value` token list and class per row | UCI Machine Learning Repository | CC BY 4.0 |
| `mushroom-small.tsv` | 20 instances of the same data, cap shape and class only | UCI Machine Learning Repository | CC BY 4.0 |
| `pokemon.tsv` | Attributes of 801 Pokémon (types, abilities, base statistics) | Banik (2017), Kaggle | **Not stated by the source** (see below) |
| `wiktionary.tsv` | Translations of the same words in 13 languages, one character per token | Collected from Wiktionary by T. Tresoldi (2018) for ASymCat | CC BY-SA 3.0, as required by Wiktionary |
| `toy.tsv` | Three invented sequence pairs | Written for ASymCat's tests | MIT, like the code |

## Sources

**CMUdict.** The Carnegie Mellon Pronouncing Dictionary, version 0.7b (2014),
Copyright © 1993–2015 Carnegie Mellon University,
<https://github.com/cmusphinx/cmudict>. ASymCat's copy converts the ARPAbet
transcriptions to IPA and splits spellings into letters. Its licence requires
redistributions to keep the copyright notice and conditions, which are
reproduced in [`LICENSE-cmudict.txt`](LICENSE-cmudict.txt).

**Darwin's finches.** Sanderson, J. G. (2000). Testing ecological patterns: A
well-known algorithm from computer science aids the evaluation of species
distributions. *American Scientist*, 88, 332–339. The same matrix is
distributed with the `cooccur` R package: Griffith, D. M., Veech, J. A., &
Marsh, C. J. (2016). cooccur: Probabilistic species co-occurrence analysis in
R. *Journal of Statistical Software*, 69(Code Snippet 2), 1–17.
<https://doi.org/10.18637/jss.v069.c02>

**Mushrooms.** Mushroom [Dataset]. (1981). UCI Machine Learning Repository.
<https://doi.org/10.24432/C5959T>. Records drawn from *The Audubon Society
Field Guide to North American Mushrooms* (1981), donated by J. Schlimmer in
1987, and obtained here through Kaggle's "Mushroom Classification" copy.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
ASymCat's files are subsets reformatted as described in the table.

**Pokémon.** Banik, R. (2017). *The Complete Pokemon Dataset* [Dataset].
Kaggle. <https://www.kaggle.com/datasets/rounakbanik/pokemon>. The Kaggle page
says the data was scraped from <https://serebii.net/> and gives no licence for
redistribution. Pokémon names are trademarks of Nintendo, Creatures Inc. and
GAME FREAK inc. The file is kept as example data. Its source states no licence,
so check the source's terms before redistributing it or using it in published
work.

**Wiktionary.** Collected by Tiago Tresoldi in 2018 from
[Wiktionary](https://www.wiktionary.org/) for use with ASymCat; it has not been
published separately. Wiktionary's text was then licensed under the [Creative
Commons Attribution-ShareAlike 3.0
licence](https://creativecommons.org/licenses/by-sa/3.0/), so `wiktionary.tsv`,
as a derivative of that content, is distributed under the same licence, with
attribution to the Wiktionary contributors. To cite it, cite ASymCat (see
[How to cite](https://asymcat.tresoldi.org/citing/)) and credit Wiktionary as
its source, for example:

> Tresoldi, T. (2018). *Multilingual word list collected from Wiktionary*
> [Data set]. Distributed with ASymCat. CC BY-SA 3.0. Source: Wiktionary
> contributors, https://www.wiktionary.org/
