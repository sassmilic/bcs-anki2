"""Tier 3: Gemini reviewer tests (mocked SDK)."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from bcs_anki.errors import MissingApiKeyError
from bcs_anki.gemini import review_definition, review_examples


def _server_error(code: int = 503, status: str = "UNAVAILABLE", message: str = "high demand"):
    return genai_errors.ServerError(
        code,
        {"error": {"code": code, "message": message, "status": status}},
        None,
    )


def _client_error(code: int = 400, status: str = "INVALID_ARGUMENT", message: str = "bad input"):
    return genai_errors.ClientError(
        code,
        {"error": {"code": code, "message": message, "status": status}},
        None,
    )


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
        with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
            review_definition(no_key_cfg, "test", "anything")


class TestRetryOnServerError:
    def test_retries_then_succeeds(self, mock_cfg):
        """One transient 503, second call succeeds — retry yields the right result."""
        original = "{{c1::primirje}} — def"
        with patch("bcs_anki.gemini.time.sleep") as mock_sleep, \
             patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = [
                _server_error(503),
                _mock_response("✓"),
            ]
            assert review_definition(mock_cfg, "primirje", original) == original
            assert mock_sleep.called  # backoff actually slept

    def test_raises_after_max_attempts(self, mock_cfg):
        """All attempts return 503 — final exception bubbles up."""
        with patch("bcs_anki.gemini.time.sleep"), \
             patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = _server_error(503)
            with pytest.raises(genai_errors.ServerError):
                review_definition(mock_cfg, "primirje", "x")
            # 3 attempts × 1 call each = 3 invocations of generate_content.
            assert mock_client.return_value.models.generate_content.call_count == 3

    def test_does_not_retry_client_error(self, mock_cfg):
        """4xx (auth, quota, bad input) is permanent — no retry."""
        with patch("bcs_anki.gemini.time.sleep") as mock_sleep, \
             patch("bcs_anki.gemini._get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = _client_error(400)
            with pytest.raises(genai_errors.ClientError):
                review_definition(mock_cfg, "primirje", "x")
            assert mock_client.return_value.models.generate_content.call_count == 1
            assert not mock_sleep.called
