# __init__.py for the ASymCat library

"""
ASymCat: a library for symmetric and asymmetric analysis of categorical co-occurrences

ASymCat is a Python library for obtaining symmetric and asymmetric measures of association between categorical
variables in data exploration and description.
"""

# Information on the package
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"
__version__ = "0.3"  # sync with setup.py

# Build the namespace
from asymcat import correlation, scorer
from asymcat.common import *

# Resource dir (mostly for tests)
# TODO: can this be dropped, at least from the namespace?
from pathlib import Path
RESOURCE_DIR = Path(__file__).parent.parent.parent / "resources"

# TODO: build the __all__ for exporting