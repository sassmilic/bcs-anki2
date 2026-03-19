from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class AppConfig:
    # API keys (from .env / environment only)
    openai_api_key: Optional[str]
    stock_image_api_key: Optional[str]

    # App settings
    image_generation_model: str
    image_size: str
    stock_image_api: str
    anki_media_folder: Path
    output_folder: Path
    temp_image_folder: Path
    log_file: Path
    rate_limit_delay_seconds: float
    tags: str
    llm_model: str
    max_workers: int


DEFAULT_CONFIG_YAML = """\
# Image generation
image_generation_model: "dall-e-3"
image_size: "1024x1024"

# Stock image API (unsplash, pexels, or pixabay)
stock_image_api: "unsplash"

# Paths
anki_media_folder: "~/Library/Application Support/Anki2/User 1/collection.media"
output_folder: "./output"
temp_image_folder: "./temp_images"
log_file: "./processing.log"

# Processing
rate_limit_delay_seconds: 2
tags: "bcs naski"
max_workers: 4

# LLM model for definitions/sentences
llm_model: "gpt-4.1-mini"
"""


def _load_api_keys() -> dict[str, Optional[str]]:
    """Load API keys from .env file and/or environment variables."""
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "stock_image_api_key": os.getenv("STOCK_IMAGE_API_KEY"),
    }


def load_config(path: Optional[Path]) -> AppConfig:
    """Load app settings from YAML/JSON, API keys from environment."""
    data: dict
    if path is None:
        data = yaml.safe_load(DEFAULT_CONFIG_YAML)
    else:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".json":
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Config file must be YAML or JSON")

    keys = _load_api_keys()

    return AppConfig(
        openai_api_key=keys["openai_api_key"],
        stock_image_api_key=keys["stock_image_api_key"],
        image_generation_model=data.get("image_generation_model", "dall-e-3"),
        image_size=data.get("image_size", "1024x1024"),
        stock_image_api=data.get("stock_image_api", "unsplash"),
        anki_media_folder=Path(data.get("anki_media_folder", "./collection.media")).expanduser(),
        output_folder=Path(data.get("output_folder", "./output")).expanduser(),
        temp_image_folder=Path(data.get("temp_image_folder", "./temp_images")).expanduser(),
        log_file=Path(data.get("log_file", "./processing.log")).expanduser(),
        rate_limit_delay_seconds=float(data.get("rate_limit_delay_seconds", 2)),
        tags=str(data.get("tags", "bcs naski")),
        llm_model=data.get("llm_model", "gpt-4.1-mini"),
        max_workers=int(data.get("max_workers", 4)),
    )
