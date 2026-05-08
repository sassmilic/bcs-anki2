from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from typing import Literal

from openai import BadRequestError, OpenAI

from .config import AppConfig
from .costs import COST_TRACKER
from .errors import (
    ImageRejectedError,
    MissingApiKeyError,
    NoStockResultsError,
    UnsupportedStockProviderError,
)
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
        raise MissingApiKeyError("Stock image API key is not configured")

    api = cfg.stock_image_api.lower()

    if api == "unsplash":
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {cfg.stock_image_api_key}"}
        params = {"query": word_en, "per_page": count}
        resp = request_with_retries("GET", url, headers=headers, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            raise NoStockResultsError("No Unsplash results")
        img_urls = [r["urls"]["regular"] for r in results[:count]]
    elif api == "pexels":
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": cfg.stock_image_api_key}
        params = {"query": word_en, "per_page": count}
        resp = request_with_retries("GET", url, headers=headers, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        photos = data.get("photos", [])
        if not photos:
            raise NoStockResultsError("No Pexels results")
        img_urls = [p["src"]["medium"] for p in photos[:count]]
    elif api == "pixabay":
        url = "https://pixabay.com/api/"
        params = {"key": cfg.stock_image_api_key, "q": word_en, "per_page": count}
        resp = request_with_retries("GET", url, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        hits = data.get("hits", [])
        if not hits:
            raise NoStockResultsError("No Pixabay results")
        img_urls = [h["webformatURL"] for h in hits[:count]]
    else:
        raise UnsupportedStockProviderError(
            f"Unsupported stock_image_api: {cfg.stock_image_api}"
        )

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
        raise MissingApiKeyError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=cfg.openai_api_key)
    try:
        response = client.images.generate(
            model=cfg.image_generation_model,
            prompt=prompt,
            size=cfg.image_size,
            quality=cfg.image_quality,
            n=1,
        )
    except BadRequestError as exc:
        raise ImageRejectedError(str(exc)) from exc
    dest.write_bytes(base64.b64decode(response.data[0].b64_json))
    _record_image_cost(cfg, response)


def _record_image_cost(cfg: AppConfig, response) -> None:
    """Route to per-image or per-token tracking based on what the API returned.

    Token-priced models (gpt-image-*) include a `usage` object on the response;
    legacy fixed-price models (dall-e-3) don't.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        COST_TRACKER.add_image(cfg.image_generation_model, cfg.image_size, cfg.image_quality)
        return

    # The OpenAI SDK exposes input_tokens / output_tokens on the usage object,
    # and may break input_tokens down via input_tokens_details. We pull whatever
    # is available; missing fields default to 0.
    details = getattr(usage, "input_tokens_details", None) or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    elif not isinstance(details, dict):
        details = {}

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    text_in = details.get("text_tokens", 0) or 0
    image_in = details.get("image_tokens", 0) or 0
    cached_in = details.get("cached_tokens", 0) or details.get("cached_input_tokens", 0) or 0

    # If the breakdown wasn't provided, treat all input as text — most likely
    # for prompt-only generations (no input images).
    if text_in == 0 and image_in == 0 and cached_in == 0:
        text_in = input_tokens

    COST_TRACKER.add_image_tokens(
        cfg.image_generation_model,
        text_input_tokens=text_in,
        image_input_tokens=image_in,
        cached_image_input_tokens=cached_in,
        image_output_tokens=output_tokens,
    )

