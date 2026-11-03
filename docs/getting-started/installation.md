# Installation

ASymCat supports Python 3.10 to 3.12 and installs from PyPI:

```bash
pip install asymcat
```

It depends only on [FreqProb](https://freqprob.tresoldi.org/), NumPy and SciPy.
The plotting examples in this documentation also use matplotlib, which the
`viz` extra installs together with seaborn and pandas:

```bash
pip install "asymcat[viz]"
```

To check the installation and the version:

```pycon
>>> import asymcat
>>> isinstance(asymcat.__version__, str)
True
```

Report the version when you [cite ASymCat](../citing.md): results can change
between versions, as the
[Changelog](https://github.com/tresoldi/asymcat/blob/master/CHANGELOG.md) and
the [Migration guide](https://github.com/tresoldi/asymcat/blob/master/MIGRATION.md)
explain.

## From source

To work on ASymCat itself, clone the repository and install it in editable mode
with the development tools:

```bash
git clone https://github.com/tresoldi/asymcat.git
cd asymcat
pip install -e ".[dev]"
pytest
```

The repository also ships sample datasets in
[`resources/`](https://github.com/tresoldi/asymcat/tree/master/resources), with
their sources and licences.

Next: the [Tutorial](tutorial.md).
