import pathlib
from setuptools import setup

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_FILE = (LOCAL_PATH / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="catcoocc",
    version="0.2.2",
    description="Methods for symmetric and asymmetric analysis of categorical co-occurrences",
    long_description=README_FILE,
    long_description_content_type="text/markdown",
    url="https://github.com/tresoldi/catcoocc",
    author="Tiago Tresoldi",
    author_email="tresoldi@shh.mpg.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
    packages=["catcoocc", "docs", "resources"],
    keywords=["co-occurrence", "cooccurrence", "categorical variables", "mutual information", "scorer"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "tabulate",
    ],
    entry_points={
        "console_scripts": [
            "catcoocc=catcoocc.__main__:main",
        ]
    },
    test_suite='tests',
    tests_require=[],
    zip_safe=False,
)
