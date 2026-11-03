"""Run the README's examples, so they stay correct as the code changes.

Every ``pycon`` block in ``README.md`` is executed as a doctest, in its own
namespace, and its printed results must match the README exactly.
"""

import doctest
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"
BLOCKS = re.findall(r"^```pycon\n(.*?)^```", README.read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)


def test_readme_has_examples():
    assert BLOCKS, "README.md should contain at least one ```pycon example"


@pytest.mark.parametrize("index", range(len(BLOCKS)), ids=lambda i: f"block{i + 1}")
def test_readme_example(index):
    test = doctest.DocTestParser().get_doctest(BLOCKS[index], {}, f"README block {index + 1}", str(README), 0)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    runner.run(test)
    assert runner.failures == 0, f"README block {index + 1} failed; see the doctest report above"
