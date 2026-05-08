from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bcs_anki.config import AppConfig


@pytest.fixture()
def mock_cfg(tmp_path: Path) -> AppConfig:
    """AppConfig with test values and fake API keys."""
    return AppConfig(
        openai_api_key="test-openai-key",
        gemini_api_key="test-gemini-key",
        stock_image_api_key="test-stock-key",
        image_generation_model="gpt-image-2",
        image_size="1024x1024",
        image_quality="medium",
        stock_image_api="unsplash",
        anki_media_folder=tmp_path / "anki_media",
        output_folder=tmp_path / "output",
        temp_image_folder=tmp_path / "temp_images",
        log_file=tmp_path / "test.log",
        rate_limit_delay_seconds=0.0,
        tags="test",
        llm_model="gpt-5.4-mini",
        gemini_model="gemini-2.5-pro",
        max_workers=2,
    )


@pytest.fixture()
def mock_openai_chat():
    """Returns a factory that builds a mock OpenAI chat response."""
    def _make(content: str) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp
    return _make


@pytest.fixture()
def mock_openai_image():
    """Returns a mock OpenAI images.generate response (legacy dall-e-3 shape, no usage)."""
    import base64

    def _make(image_bytes: bytes = b"AI_IMAGE_DATA") -> MagicMock:
        datum = MagicMock()
        datum.b64_json = base64.b64encode(image_bytes).decode("ascii")
        resp = MagicMock()
        resp.data = [datum]
        resp.usage = None  # legacy models don't return token usage
        return resp
    return _make


@pytest.fixture()
def mock_openai_image_with_usage():
    """OpenAI images.generate response with token usage (gpt-image-* shape)."""
    import base64

    def _make(
        image_bytes: bytes = b"AI_IMAGE_DATA",
        input_tokens: int = 50,
        output_tokens: int = 4096,
        text_tokens: int | None = None,
        image_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> MagicMock:
        datum = MagicMock()
        datum.b64_json = base64.b64encode(image_bytes).decode("ascii")

        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        details = {
            "text_tokens": text_tokens if text_tokens is not None else input_tokens,
            "image_tokens": image_tokens,
            "cached_tokens": cached_tokens,
        }
        usage.input_tokens_details = details

        resp = MagicMock()
        resp.data = [datum]
        resp.usage = usage
        return resp
    return _make


