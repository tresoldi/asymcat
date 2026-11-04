# How to cite

If ASymCat contributes to work you publish, please cite the software and the
version you used. Results can change between versions (see the
[Changelog](https://github.com/tresoldi/asymcat/blob/master/CHANGELOG.md)), so
the version number is what lets a reader reproduce your numbers. Check it with:

```python
import asymcat

print(asymcat.__version__)
```

## DOIs

Every release of ASymCat is archived on [Zenodo](https://doi.org/10.5281/zenodo.23917211) with its
own DOI:

- **A specific version:** cite the DOI of the release you used, so readers get
  exactly that code. Version 0.9.0 is
  [10.5281/zenodo.23938358](https://doi.org/10.5281/zenodo.23938358), and version 0.8.0 is
  [10.5281/zenodo.23917212](https://doi.org/10.5281/zenodo.23917212). Other versions are listed under "Versions" on the
  Zenodo record.
- **All versions:** [10.5281/zenodo.23917211](https://doi.org/10.5281/zenodo.23917211) always resolves to the latest
  release. Use it when you refer to ASymCat in general rather than to results
  from a particular version.

## Citing the software

**APA**

> Tresoldi, T. (2026). *ASymCat: Asymmetric measures of association between
> categorical variables* (Version 0.9.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.23938358

**BibTeX** (biblatex, or any style that supports `@software`)

```bibtex
@software{tresoldi_asymcat,
  author  = {Tresoldi, Tiago},
  title   = {{ASymCat}: Asymmetric measures of association between
             categorical variables},
  year    = {2026},
  version = {0.9.0},
  doi     = {10.5281/zenodo.23938358},
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
  note         = {Version 0.9.0},
  doi          = {10.5281/zenodo.23938358},
  howpublished = {\url{https://doi.org/10.5281/zenodo.23938358}}
}
```

Replace the version, year and DOI with those of the release you used. The citation
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
