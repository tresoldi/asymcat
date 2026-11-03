# How to cite

If ASymCat contributes to work you publish, please cite the software and the
version you used. Results can change between versions (see the
[Changelog](https://github.com/tresoldi/asymcat/blob/master/CHANGELOG.md)), so
the version number is what lets a reader reproduce your numbers. Check it with:

```python
import asymcat

print(asymcat.__version__)
```

## Citing the software

**APA**

> Tresoldi, T. (2026). *ASymCat: Asymmetric measures of association between
> categorical variables* (Version 0.7.0) [Computer software].
> https://asymcat.tresoldi.org/

**BibTeX** (biblatex, or any style that supports `@software`)

```bibtex
@software{tresoldi_asymcat,
  author  = {Tresoldi, Tiago},
  title   = {{ASymCat}: Asymmetric measures of association between
             categorical variables},
  year    = {2026},
  version = {0.7.0},
  url     = {https://asymcat.tresoldi.org/}
}
```

**BibTeX** (classic styles without `@software`, such as `plain` or most journal
templates)

```bibtex
@misc{tresoldi_asymcat,
  author       = {Tresoldi, Tiago},
  title        = {{ASymCat}: Asymmetric measures of association between
                  categorical variables},
  year         = {2026},
  note         = {Version 0.7.0},
  howpublished = {\url{https://asymcat.tresoldi.org/}}
}
```

Replace the version and year with those of the release you used. The citation
metadata also lives in
[`CITATION.cff`](https://github.com/tresoldi/asymcat/blob/master/CITATION.cff);
GitHub's **Cite this repository** button on the repository page builds APA and
BibTeX entries from it.

## Citing the measures and their dependencies

ASymCat implements published measures. When a result depends on a specific one,
such as Theil's U or the Chao–Shen entropy estimator, cite the measure's
original paper as well as the software. The [References](references.md) page
lists the source for each measure and provides them all as a BibTeX file.

ASymCat computes its smoothed probabilities and bias-corrected entropies with
[FreqProb](https://freqprob.tresoldi.org/). If those options matter to your
results, cite FreqProb too; its
[How to cite](https://freqprob.tresoldi.org/citing/) page has the entries.
