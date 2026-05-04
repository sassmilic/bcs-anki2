"""Tier 3: Gemini reviewer tests (mocked SDK)."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from bcs_anki.gemini import review_definition, review_examples


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


class TestReviewDefinition:
    def test_ok_sigil_returns_original(self, mock_cfg):
        original = "{{c1::primirje}} (imenica, sr.) — privremeni prekid"
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response("✓")
            assert review_definition(mock_cfg, "primirje", original) == original

    def test_correction_returns_gemini_text(self, mock_cfg):
        original = "{{c1::primirje}} — pogrešna definicija"
        corrected = "{{c1::primirje}} (imenica, sr.) — privremeni prestanak neprijateljstava"
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(corrected)
            assert review_definition(mock_cfg, "primirje", original) == corrected

    def test_sigil_with_whitespace_still_ok(self, mock_cfg):
        original = "{{c1::test}} — def"
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response("  ✓\n")
            assert review_definition(mock_cfg, "test", original) == original


class TestReviewExamples:
    def test_ok_sigil_returns_original(self, mock_cfg):
        original = "<ul><li>Primjer sa {{c1::primirje}}.</li></ul>"
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response("✓")
            assert review_examples(mock_cfg, "primirje", original) == original

    def test_correction_returns_gemini_text(self, mock_cfg):
        original = "<ul><li>Loše rečenica {{c1::primirja}}.</li></ul>"
        corrected = "<ul><li>Loša rečenica sa {{c1::primirjem}}.</li></ul>"
        with patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = _mock_response(corrected)
            assert review_examples(mock_cfg, "primirje", original) == corrected


class TestMissingApiKey:
    def test_review_raises_without_key(self, mock_cfg):
        no_key_cfg = replace(mock_cfg, gemini_api_key=None)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            review_definition(no_key_cfg, "test", "anything")
