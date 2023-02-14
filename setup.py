# setup.py file for the `asymcat` package.

"""
Standard setup.py file for the `asymcat` package.
"""

# Standard library imports
from setuptools import setup, find_packages
import pathlib

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_FILE = (LOCAL_PATH / "README.md").read_text()


# Load requirements, so they are listed in a single place
with open(LOCAL_PATH / "requirements.txt", encoding="utf-8") as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

# This call to setup() does all the work
setup(
    author="Tiago Tresoldi",
    author_email="tiago.tresoldi@lingfil.uu.se",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    description="A Python library for obtaining asymmetric measures of association between categorical variables in data exploration and description",
    entry_points={"console_scripts": ["asymcat=asymcat.__main__:main", ]},
    include_package_data=True,
    install_requires=install_requires,
    keywords=[
        "categorical data analysis",
        "measures of association",
        "symmetric and asymmetric measures",
        "categorical variables",
        "co-occurrence association",
        "presence/absence analysis",
        "strength of association",
        "direction of association"
    ],
    license="MIT",
    long_description=README_FILE,
    long_description_content_type="text/markdown",
    name="asymcat",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    test_suite='tests',
    tests_require=[],
    url="https://github.com/tresoldi/catcoocc",  # TODO: change upon PR
    version="0.3",
    zip_safe=False,
)
