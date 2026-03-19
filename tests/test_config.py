"""Tier 2: Config loading tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from bcs_anki.config import AppConfig, load_config


class TestDefaultConfig:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "k1", "UNSPLASH_API_KEY": "k2"}, clear=False)
    def test_defaults(self):
        cfg = load_config(None)
        assert isinstance(cfg, AppConfig)
        assert cfg.max_workers == 4
        assert cfg.llm_model == "gpt-4.1-mini"
        assert cfg.image_generation_model == "dall-e-3"
        assert cfg.tags == "bcs naski"
        assert cfg.rate_limit_delay_seconds == 2.0
        assert cfg.openai_api_key == "k1"
        assert cfg.stock_image_api == "unsplash"
        assert cfg.stock_image_api_key == "k2"


class TestCustomConfig:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "k1", "UNSPLASH_API_KEY": "k2"}, clear=False)
    def test_yaml_overrides(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "max_workers: 8\nllm_model: gpt-4o\ntags: custom\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.max_workers == 8
        assert cfg.llm_model == "gpt-4o"
        assert cfg.tags == "custom"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestEnvVarOverride:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key", "UNSPLASH_API_KEY": "env-stock"}, clear=False)
    def test_env_vars_used(self):
        cfg = load_config(None)
        assert cfg.openai_api_key == "env-key"
        assert cfg.stock_image_api == "unsplash"
        assert cfg.stock_image_api_key == "env-stock"
