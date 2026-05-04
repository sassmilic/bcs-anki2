"""Tier 1: Pure logic tests — no mocks, no I/O."""
from __future__ import annotations

from bcs_anki.csv_writer import _escape_field
from bcs_anki.images import build_image_filename


# --- build_image_filename ---

class TestBuildImageFilename:
    def test_deterministic_hash(self):
        f1 = build_image_filename("primirje")
        f2 = build_image_filename("primirje")
        assert f1 == f2

    def test_different_words_different_hashes(self):
        assert build_image_filename("kuća") != build_image_filename("voda")

    def test_valid_chars_only(self):
        filename = build_image_filename("kuća")
        stem = filename.rsplit(".", 1)[0]
        for ch in stem:
            assert ch.isalnum() or ch in ("_", "-"), f"Invalid char: {ch}"

    def test_unicode_handling(self):
        filename = build_image_filename("čćžšđ")
        assert filename.endswith(".png")
        assert len(filename) > 4

    def test_empty_word_fallback(self):
        filename = build_image_filename("   ")
        assert filename.startswith("image_")

    def test_special_chars_stripped(self):
        filename = build_image_filename("hello world!")
        assert " " not in filename
        assert "!" not in filename


# --- _escape_field ---

class TestEscapeField:
    def test_newlines_replaced(self):
        assert _escape_field("line1\nline2") == "line1<br>line2"

    def test_passthrough_normal(self):
        assert _escape_field("hello world") == "hello world"

    def test_multiple_newlines(self):
        assert _escape_field("a\nb\nc") == "a<br>b<br>c"

    def test_empty_string(self):
        assert _escape_field("") == ""
