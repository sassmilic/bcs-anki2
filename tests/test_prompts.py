"""Regression tests for prompt rendering — guards against brace-escape bugs.

Anki cloze syntax requires literal `{{c1::...}}`. Because the prompts use
str.format(), every literal `{` must be doubled to `{{` in source. A wrong
brace count silently produces invalid syntax that the LLM may copy, causing
cards to be dropped by the cloze check in cli._process_word.
"""
from __future__ import annotations

import re

import pytest

from bcs_anki.prompts import (
    DEFINITION_USER,
    EXAMPLES_USER,
    REVIEW_DEFINITION_USER,
    REVIEW_EXAMPLES_USER,
)


CLOZE_USING_PROMPTS = [
    ("DEFINITION_USER", DEFINITION_USER, {"word": "vidjeti"}),
    ("EXAMPLES_USER", EXAMPLES_USER, {"word": "vidjeti"}),
    ("REVIEW_DEFINITION_USER", REVIEW_DEFINITION_USER, {"word": "vidjeti", "definition": "x"}),
    ("REVIEW_EXAMPLES_USER", REVIEW_EXAMPLES_USER, {"word": "vidjeti", "examples": "y"}),
]


@pytest.mark.parametrize("name, template, kwargs", CLOZE_USING_PROMPTS)
def test_renders_with_double_brace_cloze(name, template, kwargs):
    rendered = template.format(**kwargs)
    assert "{{c1::" in rendered, f"{name}: missing valid cloze marker after rendering"


@pytest.mark.parametrize("name, template, kwargs", CLOZE_USING_PROMPTS)
def test_no_broken_single_brace_cloze(name, template, kwargs):
    rendered = template.format(**kwargs)
    # Match `{c1` not preceded by `{` — that's a stray single-brace cloze.
    broken = re.findall(r"(?<!\{)\{c1[:]+[^}]*\}(?!\})", rendered)
    assert not broken, f"{name}: broken cloze markers in rendered output: {broken}"
