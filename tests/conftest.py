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
        llm_model="gpt-4.1-mini",
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
    """Returns a mock OpenAI images.generate response (gpt-image-2 base64 shape)."""
    import base64

    def _make(image_bytes: bytes = b"AI_IMAGE_DATA") -> MagicMock:
        datum = MagicMock()
        datum.b64_json = base64.b64encode(image_bytes).decode("ascii")
        resp = MagicMock()
        resp.data = [datum]
        return resp
    return _make


