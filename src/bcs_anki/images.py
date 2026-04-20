from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

from openai import OpenAI

from .config import AppConfig
from .http import request_with_retries

logger = logging.getLogger(__name__)


ImageSource = Literal["stock", "ai"]



def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def build_image_filename(word: str) -> str:
    base = "".join(ch for ch in word.strip() if ch.isalnum() or ch in ("_", "-")).strip("_-")
    if not base:
        base = "image"
    return f"{base}_{_short_hash(word)}.png"


def fetch_stock_image(cfg: AppConfig, word_en: str, dest: Path, count: int = 3) -> list[Path]:
    """Fetch up to `count` stock images. Returns list of paths actually downloaded."""
    if not cfg.stock_image_api_key:
        raise RuntimeError("Stock image API key is not configured")

    api = cfg.stock_image_api.lower()

    if api == "unsplash":
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {cfg.stock_image_api_key}"}
        params = {"query": word_en, "per_page": count}
        resp = request_with_retries("GET", url, headers=headers, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            raise RuntimeError("No Unsplash results")
        img_urls = [r["urls"]["regular"] for r in results[:count]]
    elif api == "pexels":
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": cfg.stock_image_api_key}
        params = {"query": word_en, "per_page": count}
        resp = request_with_retries("GET", url, headers=headers, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        photos = data.get("photos", [])
        if not photos:
            raise RuntimeError("No Pexels results")
        img_urls = [p["src"]["medium"] for p in photos[:count]]
    elif api == "pixabay":
        url = "https://pixabay.com/api/"
        params = {"key": cfg.stock_image_api_key, "q": word_en, "per_page": count}
        resp = request_with_retries("GET", url, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        hits = data.get("hits", [])
        if not hits:
            raise RuntimeError("No Pixabay results")
        img_urls = [h["webformatURL"] for h in hits[:count]]
    else:
        raise ValueError(f"Unsupported stock_image_api: {cfg.stock_image_api}")

    stem = dest.stem
    suffix = dest.suffix
    paths = []
    for i, img_url in enumerate(img_urls):
        img_resp = request_with_retries("GET", img_url, delay_seconds=cfg.rate_limit_delay_seconds)
        if i == 0:
            p = dest
        else:
            p = dest.with_name(f"{stem}_{i}{suffix}")
        p.write_bytes(img_resp.content)
        paths.append(p)

    return paths


def generate_ai_image(cfg: AppConfig, prompt: str, dest: Path) -> None:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=cfg.openai_api_key)
    response = client.images.generate(
        model=cfg.image_generation_model,
        prompt=prompt,
        size=cfg.image_size,
        n=1,
    )
    img_url = response.data[0].url
    img_resp = request_with_retries("GET", img_url, delay_seconds=cfg.rate_limit_delay_seconds)
    dest.write_bytes(img_resp.content)

