#!/usr/bin/env python3

"""
setup.py for the  `asymcat` package.
"""

# TODO: move the metadata to pyproject.toml

# Import Python standard libraries
from pathlib import Path
import setuptools

ROOT_DIR = Path(__file__).parent

# Get the long description from the README.md file
with open(ROOT_DIR / 'README.md', 'r') as fh:
    long_description = fh.read()

# Load requirements, so they are listed in a single place
with open(ROOT_DIR / "requirements.txt") as fh:
    REQUIREMENTS = [dep.strip() for dep in fh.readlines()]

DEV_REQUIREMENTS = [
    'black == 22.*',
    'build == 0.7.*',
    'flake8 == 4.*',
    'isort == 5.*',
    'mypy >= 0.981',
    'pytest == 7.*',
    'pytest-cov == 4.*',
    'twine == 4.*',
]

setuptools.setup(
    author='Tiago Tresoldi',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    description=(
"A Python library for obtaining asymmetric measures of association between categorical variables in data exploration and description"
    ),
    entry_points={
        'console_scripts': [
            'asymcat=asymcat.my_module:main',
        ]
    },
    extras_require={
        'dev': DEV_REQUIREMENTS,
    },
    install_requires=REQUIREMENTS,
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    name='asymcat',
    package_data={
        'asymcat': [
            'py.typed',
        ]
    },
    packages=setuptools.find_packages(
        exclude=[
            'examples',
            'test',
        ]
    ),
    python_requires='>=3.8, <4',
    url='http://github.com/tresoldi/asymcat',
    version='0.3.0',  # Remember to sync with __init__.py
)

