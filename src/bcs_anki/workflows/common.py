from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import AppConfig, load_config
from ..logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_app_config(config_path: Optional[Path | str], verbose: bool = False) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else None
    cfg = load_config(path)
    setup_logging(cfg.log_file, verbose=verbose)
    return cfg


def log_effective_config(cfg: AppConfig) -> None:
    safe_cfg = {
        "openai_api_key": "set" if cfg.openai_api_key else "not set",
        "gemini_api_key": "set" if cfg.gemini_api_key else "not set",
        "image_generation_model": cfg.image_generation_model,
        "image_size": cfg.image_size,
        "stock_image_api": cfg.stock_image_api or "none",
        "stock_image_api_key": "set" if cfg.stock_image_api_key else "not set",
        "anki_media_folder": str(cfg.anki_media_folder),
        "output_folder": str(cfg.output_folder),
        "temp_image_folder": str(cfg.temp_image_folder),
        "log_file": str(cfg.log_file),
        "rate_limit_delay_seconds": cfg.rate_limit_delay_seconds,
        "tags": cfg.tags,
        "llm_model": cfg.llm_model,
        "gemini_model": cfg.gemini_model,
        "max_workers": cfg.max_workers,
    }
    logger.info("Loaded configuration: %s", safe_cfg)
