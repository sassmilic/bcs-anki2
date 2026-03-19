from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bcs_anki.cli import WordEntry
from bcs_anki.config import AppConfig


@pytest.fixture()
def mock_cfg(tmp_path: Path) -> AppConfig:
    """AppConfig with test values and fake API keys."""
    return AppConfig(
        openai_api_key="test-openai-key",
        stock_image_api_key="test-stock-key",
        image_generation_model="dall-e-3",
        image_size="1024x1024",
        stock_image_api="unsplash",
        anki_media_folder=tmp_path / "anki_media",
        output_folder=tmp_path / "output",
        temp_image_folder=tmp_path / "temp_images",
        log_file=tmp_path / "test.log",
        rate_limit_delay_seconds=0.0,
        tags="test",
        llm_model="gpt-4.1-mini",
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
    """Returns a mock OpenAI images.generate response."""
    def _make(url: str = "https://example.com/image.png") -> MagicMock:
        datum = MagicMock()
        datum.url = url
        resp = MagicMock()
        resp.data = [datum]
        return resp
    return _make


@pytest.fixture()
def sample_word_entry() -> WordEntry:
    return WordEntry("primirje", None)


@pytest.fixture()
def sample_word_entry_with_context() -> WordEntry:
    return WordEntry("zamak", "dvorac, ne brava")
