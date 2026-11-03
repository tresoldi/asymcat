"""MkDocs hook that builds the documentation's figures from its own examples.

A page that shows a figure produces it in its ``pycon`` examples (for instance
with ``fig.savefig("mle.png")``). At build time, this hook runs the
examples of each page listed in ``FIGURES`` in a temporary directory and
publishes the images they write under ``figures/``, so every figure is exactly
what the code on its page produces. The examples are run, not compared with
their printed output; ``tests/unit/test_docs_examples.py`` checks that.
"""

import doctest
import os
import re
import tempfile
from pathlib import Path

from mkdocs.structure.files import File, Files

# Page (relative to docs/) -> images its examples write.
FIGURES = {"how-to/plot.md": ["mle.png"]}

BLOCK = re.compile(r"^```pycon\n(.*?)^```", flags=re.MULTILINE | re.DOTALL)


def _run_examples(page: Path, workdir: Path) -> None:
    """Execute every pycon example of ``page`` in one namespace, inside ``workdir``."""
    parser = doctest.DocTestParser()
    namespace: dict = {}
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        for block in BLOCK.findall(page.read_text(encoding="utf-8")):
            for example in parser.get_examples(block):
                exec(compile(example.source, str(page), "exec"), namespace)  # noqa: S102  # nosec B102
    finally:
        os.chdir(cwd)


def on_files(files: Files, config) -> Files:
    """Add the figures produced by the documentation's examples to the site."""
    docs_dir = Path(config["docs_dir"])
    for page, images in FIGURES.items():
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _run_examples(docs_dir / page, workdir)
            for image in images:
                content = (workdir / image).read_bytes()
                files.append(File.generated(config, f"figures/{image}", content=content))
    return files
