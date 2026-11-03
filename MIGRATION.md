# Migration Guide

## Migrating to 0.8.0

### `mle()` and `tresoldi()` now follow the documented direction

Every measure returns, for a pair `(x, y)`, a tuple whose first element is the
`x → y` direction (`x` predicting `y`) and whose second is `y → x`. `mle()` used
to return the reverse, `(P(x|y), P(y|x))`, unlike the documentation and unlike
`theil_u`, `cond_entropy` and `goodman_kruskal_lambda`. It now returns
`(P(y|x), P(x|y))`.

`tresoldi()` takes its exponent from `mle()`, and PMI is symmetric, so its two
elements swap in the same way. No other measure changes.

```python
# With "q" always followed by "k", but "k" also following other symbols:
scorer.mle()[("q", "k")]
# before 0.8.0: (0.2, 1.0)   i.e. (P(q|k), P(k|q))
# from 0.8.0:   (1.0, 0.2)   i.e. (P(k|q), P(q|k))
```

**What to do:**

- **Results from earlier versions:** if you reported `mle` or `tresoldi` values,
  the two directions in them were swapped relative to their labels. Swap the
  elements to compare them with 0.8.0 results.
- **Code that swapped the tuple:** if your code worked around the old order
  (for example, reading `mle()[pair][1]` as `P(y|x)`), remove the workaround.
- **Everything else:** code that already treated the first element as `x → y`
  is now correct, with no change needed.

`scorer2matrices`, `permutation_pvalue` and `bootstrap_ci` pass the tuples
through unchanged, so their results for these two measures swap too.

## Migrating to 0.7.0

The public API is unchanged and every existing result is identical:
`entropy_estimator` is a new, opt-in argument whose default (`"plugin"`) is
the previous behavior. What changed is what a plain `pip install asymcat`
installs.

### A plain install no longer pulls in plotting and dataframe libraries

ASymCat's runtime dependencies are now exactly what the package imports:
**freqprob, NumPy and SciPy**. It no longer installs `matplotlib`, `pandas`,
`seaborn` or `tabulate`, none of which the package itself used.

If your own code imports any of them and relied on ASymCat to install them,
either declare them in your project or install the plotting extra:

```bash
pip install "asymcat[viz]"  # matplotlib, seaborn and pandas
```

`tabulate` is not in any extra; install it directly if you need it.

### The `viz` extra changed contents

`asymcat[viz]` used to install `plotly`, `bokeh` and `altair`, which nothing in
ASymCat or its documentation used. It now installs the libraries the User
Guide's plotting examples actually use (`matplotlib`, `seaborn`, `pandas`).
Install `plotly`, `bokeh` or `altair` directly if you want them.

### freqprob 0.8.0 or later is required

The bias-corrected estimators come from freqprob. Version 0.8.0 fixes
`chao_shen_entropy` returning zero for samples in which every symbol occurs
once, a case that is common in sparse co-occurrence data.

### For contributors

`scikit-learn`, `types-tabulate` and `types-setuptools` were dropped from the
`test` and `dev` extras (none were imported), and the linters are now pinned at
`ruff==0.17.3` and `mypy==2.4.1`. Reinstall with `pip install -e ".[dev]"`.

## The `src/` layout restructure

This restructure aligns ASymCat's packaging, documentation, and tooling with the
[freqprob](https://github.com/tresoldi/freqprob) project layout. It is a
**packaging- and tooling-only** change: the public `asymcat` API is unchanged.

### For users: nothing changes

If you install ASymCat from PyPI and import from the top-level namespace, your
code keeps working exactly as before:

```python
import asymcat

cooccs = asymcat.collect_cooccs(asymcat.read_sequences("data.tsv"))
scorer = asymcat.CatScorer(cooccs)
scorer.theil_u()
```

Every public name — `CatScorer`, `read_sequences`, `read_pa_matrix`,
`collect_cooccs`, `collect_ngrams`, `collect_observations`, `collect_alphabets`,
`build_ct`, and the `asymcat.scorer` / `asymcat.correlation` submodules — is
re-exported unchanged from `asymcat/__init__.py`.

### For contributors: what changed

#### 1. `src/` layout

The importable package moved from `asymcat/` to `src/asymcat/`:

| Before | After |
|--------|-------|
| `asymcat/__init__.py` | `src/asymcat/__init__.py` |
| `asymcat/common.py` | `src/asymcat/common.py` |
| `asymcat/scorer.py` | `src/asymcat/scorer.py` |
| `asymcat/correlation.py` | `src/asymcat/correlation.py` |

After pulling this change, **reinstall the package** so the editable install
points at the new location:

```bash
pip install -e ".[dev]"
```

Tooling that referenced the old path (for example `mypy asymcat/`) now targets
`src/asymcat/`; this is handled by `pyproject.toml` and the CI workflows.

#### 2. Documentation moved to MkDocs

The Sphinx site under `docs/source/` and the rendered tutorial artifacts were
moved to `docs/archive/` and replaced with a MkDocs-Material site
(`mkdocs.yml`, `docs/index.md`, `docs/USER_GUIDE.md`, `docs/reference.md`, with a
custom `overrides/home.html`). Build it locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The API reference is now autogenerated from docstrings via `mkdocstrings`, so it
no longer needs hand-maintained `.rst` files.

#### 3. Dependency and toolchain

- **freqprob** is pinned to `>=0.6.2`. ASymCat treats freqprob estimators as
  log-probability sources (freqprob defaults to `logprob=True`) and converts
  internally, so smoothing results are unchanged.
- **ruff** and **mypy** are pinned exactly (`ruff==0.16.0`, `mypy==1.19.1`) so
  local, pre-commit, and CI runs use identical versions.
- A `.pre-commit-config.yaml` mirrors the CI quality job. Enable it with:

  ```bash
  pip install pre-commit
  pre-commit install
  ```

- Optional-dependency groups are now `viz`, `test`, `docs`, `dev`, and `all`
  (see `pyproject.toml`).

### New project files

`ARCHITECTURE.md`, `SECURITY.md`, `CITATION.cff`,
`.pre-commit-config.yaml`, and `.github/dependabot.yml` were added to match the
freqprob project structure.
