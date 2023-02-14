# __init__.py

# Information on the package
__version__ = "0.3"  # sync with setup.py
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"

# Build the namespace
from asymcat.utils import *
from asymcat import scorer
from asymcat import correlation

# Resource dir (mostly for tests)
from pathlib import Path

RESOURCE_DIR = Path(__file__).parent.parent.parent / "resources"
