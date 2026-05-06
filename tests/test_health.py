"""Tier 3: API health-check tests (mocked SDKs + HTTP)."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from bcs_anki.health import check_apis


def _ok_openai_client(image_models: list[str]):
    """Build a mock OpenAI client whose chat call succeeds and models.list returns the given ids."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock()
    client.models.list.return_value = MagicMock(
        data=[MagicMock(id=mid) for mid in image_models]
    )
    return client


def _ok_gemini_client():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="x")
    return client


def _ok_stock_response():
    r = MagicMock()
    r.raise_for_status.return_value = None
    return r


class TestAllPass:
    def test_all_apis_ok(self, mock_cfg):
        with patch("bcs_anki.health.OpenAI") as openai_cls, \
             patch("google.genai.Client") as gemini_cls, \
             patch("bcs_anki.health.requests.get", return_value=_ok_stock_response()):
            openai_cls.return_value = _ok_openai_client([mock_cfg.image_generation_model])
            gemini_cls.return_value = _ok_gemini_client()
            check_apis(mock_cfg)  # must not raise


class TestOpenAIFailure:
    def test_chat_call_fails(self, mock_cfg):
        client = _ok_openai_client([mock_cfg.image_generation_model])
        client.chat.completions.create.side_effect = RuntimeError("invalid api key")
        with patch("bcs_anki.health.OpenAI", return_value=client):
            with pytest.raises(RuntimeError, match="OpenAI"):
                check_apis(replace(mock_cfg, gemini_api_key=None, stock_image_api_key=None))

    def test_image_model_not_listed(self, mock_cfg):
        client = _ok_openai_client(["gpt-4.1-mini", "some-other-model"])
        with patch("bcs_anki.health.OpenAI", return_value=client):
            with pytest.raises(RuntimeError, match="image model"):
                check_apis(replace(mock_cfg, gemini_api_key=None, stock_image_api_key=None))

    def test_missing_api_key(self, mock_cfg):
        no_key = replace(mock_cfg, openai_api_key=None, gemini_api_key=None, stock_image_api_key=None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            check_apis(no_key)


class TestGeminiFailure:
    def test_gemini_call_fails(self, mock_cfg):
        with patch("bcs_anki.health.OpenAI") as openai_cls, \
             patch("google.genai.Client") as gemini_cls:
            openai_cls.return_value = _ok_openai_client([mock_cfg.image_generation_model])
            gemini = MagicMock()
            gemini.models.generate_content.side_effect = RuntimeError("PERMISSION_DENIED")
            gemini_cls.return_value = gemini
            with pytest.raises(RuntimeError, match="Gemini"):
                check_apis(replace(mock_cfg, stock_image_api_key=None))

    def test_gemini_skipped_when_no_key(self, mock_cfg):
        """No Gemini key set: skip check, do not require the SDK."""
        with patch("bcs_anki.health.OpenAI") as openai_cls, \
             patch("bcs_anki.health.requests.get", return_value=_ok_stock_response()):
            openai_cls.return_value = _ok_openai_client([mock_cfg.image_generation_model])
            check_apis(replace(mock_cfg, gemini_api_key=None))


class TestStockImageFailure:
    def test_unsplash_returns_403(self, mock_cfg):
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("403 Forbidden")
        with patch("bcs_anki.health.OpenAI") as openai_cls, \
             patch("google.genai.Client") as gemini_cls, \
             patch("bcs_anki.health.requests.get", return_value=bad):
            openai_cls.return_value = _ok_openai_client([mock_cfg.image_generation_model])
            gemini_cls.return_value = _ok_gemini_client()
            with pytest.raises(RuntimeError, match="Stock images"):
                check_apis(mock_cfg)

    def test_stock_skipped_when_no_key(self, mock_cfg):
        with patch("bcs_anki.health.OpenAI") as openai_cls, \
             patch("google.genai.Client") as gemini_cls:
            openai_cls.return_value = _ok_openai_client([mock_cfg.image_generation_model])
            gemini_cls.return_value = _ok_gemini_client()
            check_apis(replace(mock_cfg, stock_image_api_key=None))


class TestErrorAggregation:
    def test_collects_all_failures_in_one_message(self, mock_cfg):
        """When multiple APIs fail, the error message lists all of them."""
        bad_openai = _ok_openai_client([])  # image model missing
        bad_gemini = MagicMock()
        bad_gemini.models.generate_content.side_effect = RuntimeError("rate limit")
        bad_stock = MagicMock()
        bad_stock.raise_for_status.side_effect = Exception("401")

        with patch("bcs_anki.health.OpenAI", return_value=bad_openai), \
             patch("google.genai.Client", return_value=bad_gemini), \
             patch("bcs_anki.health.requests.get", return_value=bad_stock):
            with pytest.raises(RuntimeError) as excinfo:
                check_apis(mock_cfg)
        msg = str(excinfo.value)
        assert "OpenAI" in msg
        assert "Gemini" in msg
        assert "Stock images" in msg
