from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DialectNormalizationResult:
    original: str
    normalized: str
    changed: bool


def ekavian_to_ijekavian(word: str) -> DialectNormalizationResult:
    """
    Very lightweight, heuristic ekavian -> ijekavian normalizer.
    This is intentionally conservative; it only handles some frequent patterns.
    """
    original = word.strip()
    w = original

    # Common patterns: mleko->mlijeko, reka->rijeka, pesma->pjesma etc.
    replacements = [
        (r"^(m)leko$", r"\1lijeko"),
        (r"^(r)eka$", r"\1ijeka"),
        (r"^(p)esma$", r"\1jesma"),
        (r"^(d)ete$", r"\1ijete"),
        (r"^(n)ebo$", r"\1ebo"),  # placeholder, avoid over-aggressive rules
    ]

    for pattern, repl in replacements:
        if re.search(pattern, w):
            w = re.sub(pattern, repl, w)

    changed = w != original
    return DialectNormalizationResult(original=original, normalized=w, changed=changed)


def normalize_line(line: str) -> Tuple[str, DialectNormalizationResult]:
    word = line.strip()
    res = ekavian_to_ijekavian(word)
    return res.normalized, res

