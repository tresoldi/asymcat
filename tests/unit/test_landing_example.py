"""Run the landing page's example, so it stays correct as the code changes.

Every ``<pre class="pycon">`` block in ``overrides/home.html`` is stripped of
its markup and executed as a doctest; its printed results must match the page.
"""

import doctest
import html
import re
from pathlib import Path

import pytest

HOME = Path(__file__).resolve().parents[2] / "overrides" / "home.html"
BLOCKS = [
    html.unescape(re.sub(r"<[^>]+>", "", block))
    for block in re.findall(r'<pre class="pycon">(.*?)</pre>', HOME.read_text(encoding="utf-8"), flags=re.DOTALL)
]


def test_landing_page_has_examples():
    assert BLOCKS, 'overrides/home.html should contain at least one <pre class="pycon"> example'


@pytest.mark.parametrize("index", range(len(BLOCKS)), ids=lambda i: f"block{i + 1}")
def test_landing_page_example(index):
    test = doctest.DocTestParser().get_doctest(BLOCKS[index] + "\n", {}, f"home.html block {index + 1}", str(HOME), 0)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    runner.run(test)
    assert runner.failures == 0, f"home.html block {index + 1} failed; see the doctest report above"
