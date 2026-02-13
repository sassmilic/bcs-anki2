from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AppConfig:
    openai_api_key: Optional[str]
    image_generation_model: str
    image_size: str

    stock_image_api: str
    stock_image_api_key: Optional[str]

    anki_media_folder: Path
    output_folder: Path
    temp_image_folder: Path
    log_file: Path

    rate_limit_delay_seconds: float
    tags: str
    llm_model: str


DEFAULT_CONFIG_YAML = """\
# API Configuration
openai_api_key: ""  # Or use environment variable OPENAI_API_KEY
image_generation_model: "dall-e-3"
image_size: "1024x1024"

# Stock Image API (choose one)
stock_image_api: "unsplash"  # Options: unsplash, pexels, pixabay
stock_image_api_key: ""

# Paths
anki_media_folder: "/path/to/Anki2/User/collection.media"
output_folder: "./output"
temp_image_folder: "./temp_images"
log_file: "./processing.log"

# Processing
rate_limit_delay_seconds: 2
tags: "bcs naski"

# LLM for definitions/sentences
llm_model: "gpt-4.1-mini"
"""


def load_config(path: Optional[Path]) -> AppConfig:
    """
    Load configuration from YAML or JSON, with env var overrides.
    """
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

    # Env overrides
    openai_key = os.getenv("OPENAI_API_KEY") or data.get("openai_api_key") or None
    stock_key = os.getenv("STOCK_IMAGE_API_KEY") or data.get("stock_image_api_key") or None

    base = {
        "openai_api_key": openai_key,
        "image_generation_model": data.get("image_generation_model", "dall-e-3"),
        "image_size": data.get("image_size", "1024x1024"),
        "stock_image_api": data.get("stock_image_api", "unsplash"),
        "stock_image_api_key": stock_key,
        "anki_media_folder": Path(data.get("anki_media_folder", "./collection.media")).expanduser(),
        "output_folder": Path(data.get("output_folder", "./output")).expanduser(),
        "temp_image_folder": Path(data.get("temp_image_folder", "./temp_images")).expanduser(),
        "log_file": Path(data.get("log_file", "./processing.log")).expanduser(),
        "rate_limit_delay_seconds": float(data.get("rate_limit_delay_seconds", 2)),
        "tags": str(data.get("tags", "bcs naski")),
        "llm_model": data.get("llm_model", "gpt-4.1-mini"),
    }

    return AppConfig(**base)

