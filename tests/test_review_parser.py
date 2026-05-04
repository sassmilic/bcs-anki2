"""Tier 3: review-command CSV parser, including missing-row scenarios."""
from __future__ import annotations

from bcs_anki.cli import _parse_review_csv


HEADER = (
    "#separator:Tab\n"
    "#html:true\n"
    "#notetype column:1\n"
    "#tags column:4\n"
)


def _row(note_type: str, field1: str, field2: str = "", tags: str = "test") -> str:
    return f"{note_type}\t{field1}\t{field2}\t{tags}"


def test_parses_full_three_row_word():
    csv = HEADER + "\n".join([
        _row("Cloze", "{{c1::primirje}} — def"),
        _row("Cloze", "Ex {{c1::primirje}}."),
        _row("Basic (and reversed card)", '<img src="primirje_abc.png">', "primirje"),
    ])
    words = _parse_review_csv(csv)
    assert len(words) == 1
    assert words[0]["word"] == "primirje"
    assert "def" in words[0]["definition"]
    assert "Ex" in words[0]["examples"]
    assert words[0]["image_file"] == "primirje_abc.png"


def test_handles_missing_image_then_next_word():
    """Word A has clozes but no Basic (image was filtered); word B is complete.

    Pre-fix, the parser took rows in groups of 3 and misaligned everything
    after the missing row. Post-fix, A is emitted as orphan and B is intact.
    """
    csv = HEADER + "\n".join([
        _row("Cloze", "{{c1::wordA}} — defA"),
        _row("Cloze", "Ex {{c1::wordA}}."),
        # No Basic row for A
        _row("Cloze", "{{c1::wordB}} — defB"),
        _row("Cloze", "Ex {{c1::wordB}}."),
        _row("Basic (and reversed card)", '<img src="wordB.png">', "wordb"),
    ])
    words = _parse_review_csv(csv)
    assert len(words) == 2
    assert words[0]["word"] == "(no image)"
    assert "defA" in words[0]["definition"]
    assert "Ex" in words[0]["examples"]
    assert words[1]["word"] == "wordb"
    assert "defB" in words[1]["definition"]
    assert words[1]["image_file"] == "wordB.png"


def test_handles_missing_definition_row():
    """Word's definition row was dropped (no cloze markers); examples + image survive."""
    csv = HEADER + "\n".join([
        _row("Cloze", "Ex only {{c1::test}}."),
        _row("Basic (and reversed card)", '<img src="test.png">', "test"),
    ])
    words = _parse_review_csv(csv)
    assert len(words) == 1
    assert words[0]["word"] == "test"
    assert "Ex only" in words[0]["definition"]  # promoted to definition slot
    assert words[0]["examples"] == ""


def test_handles_only_basic_row():
    """Both clozes were dropped; only the image card remains."""
    csv = HEADER + _row("Basic (and reversed card)", '<img src="x.png">', "x")
    words = _parse_review_csv(csv)
    assert len(words) == 1
    assert words[0]["word"] == "x"
    assert words[0]["definition"] == ""
    assert words[0]["image_file"] == "x.png"


def test_handles_trailing_orphan_clozes():
    """File ends with cloze rows for a word whose Basic was never written."""
    csv = HEADER + "\n".join([
        _row("Cloze", "{{c1::final}} — def"),
        _row("Cloze", "Ex {{c1::final}}."),
    ])
    words = _parse_review_csv(csv)
    assert len(words) == 1
    assert words[0]["word"] == "(no image)"
    assert "def" in words[0]["definition"]


def test_empty_csv_returns_empty_list():
    assert _parse_review_csv(HEADER) == []
    assert _parse_review_csv("") == []
