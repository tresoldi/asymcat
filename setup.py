import pathlib
from setuptools import setup

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_FILE = (LOCAL_PATH / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="cooccurran",
    version="0.1",
    description="Methods for categorical co-occurrence analysis by scorers",
    long_description=README_FILE,
    long_description_content_type="text/markdown",
    url="https://github.com/tresoldi/cooccuran",
    author="Tiago Tresoldi",
    author_email="tresoldi@shh.mpg.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
    packages=["cooccurren"],
    keywords=["co-occurrence", "cooccurrence", "categorical variables", "mutual information", "scorer"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "ngesh=ngesh.__main__:main",
        ]
    },
    test_suite='tests',
    tests_require=[],
)

