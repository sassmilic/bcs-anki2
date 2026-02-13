from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Literal, Tuple

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)


ImageSource = Literal["stock", "ai"]


def decide_image_source(word: str, pos_hint: str) -> ImageSource:
    """
    Very lightweight heuristic:
    - If explicitly marked as noun and seems concrete -> stock.
    - Verbs, adjectives, idioms/phrases -> AI.
    """
    lower = word.lower()
    if " " in lower:
        return "ai"
    if any(t in pos_hint.lower() for t in ["glagol", "verb", "pridjev", "idiom", "fraza", "izraz"]):
        return "ai"
    # crude heuristic: ends in -a, -o, -e, likely a concrete noun (but not always)
    if lower.endswith(("a", "o", "e")):
        return "stock"
    return "ai"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def build_image_filename(word: str) -> str:
    base = "".join(ch for ch in word.strip() if ch.isalnum() or ch in ("_", "-")).strip("_-")
    if not base:
        base = "image"
    return f"{base}_{_short_hash(word)}.png"


def _request_with_retries(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    json_body: dict | None = None,
    max_retries: int = 3,
    delay_seconds: float = 2.0,
) -> requests.Response:
    delay = delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=60,
            )
            if resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}")
            return resp
        except Exception as exc:  # noqa: BLE001
            logger.error("HTTP request failed (attempt %s): %s", attempt, exc)
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2


def fetch_stock_image(cfg: AppConfig, word_en: str, dest: Path) -> None:
    """
    Simplified Unsplash/Pexels/Pixabay integration.
    We just hit the API search endpoint and download the first result URL.
    """
    if not cfg.stock_image_api_key:
        raise RuntimeError("Stock image API key is not configured")

    api = cfg.stock_image_api.lower()

    if api == "unsplash":
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {cfg.stock_image_api_key}"}
        params = {"query": word_en, "per_page": 1}
        resp = _request_with_retries("GET", url, headers=headers, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            raise RuntimeError("No Unsplash results")
        img_url = results[0]["urls"]["regular"]
    elif api == "pexels":
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": cfg.stock_image_api_key}
        params = {"query": word_en, "per_page": 1}
        resp = _request_with_retries("GET", url, headers=headers, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        photos = data.get("photos", [])
        if not photos:
            raise RuntimeError("No Pexels results")
        img_url = photos[0]["src"]["medium"]
    elif api == "pixabay":
        url = "https://pixabay.com/api/"
        params = {"key": cfg.stock_image_api_key, "q": word_en, "per_page": 1}
        resp = _request_with_retries("GET", url, params=params, delay_seconds=cfg.rate_limit_delay_seconds)
        data = resp.json()
        hits = data.get("hits", [])
        if not hits:
            raise RuntimeError("No Pixabay results")
        img_url = hits[0]["webformatURL"]
    else:
        raise ValueError(f"Unsupported stock_image_api: {cfg.stock_image_api}")

    img_resp = _request_with_retries("GET", img_url, delay_seconds=cfg.rate_limit_delay_seconds)
    dest.write_bytes(img_resp.content)


def generate_ai_image(cfg: AppConfig, prompt: str, dest: Path) -> None:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.image_generation_model,
        "prompt": prompt,
        "size": cfg.image_size,
        "n": 1,
    }
    resp = _request_with_retries("POST", url, headers=headers, json_body=body, delay_seconds=cfg.rate_limit_delay_seconds)
    data = resp.json()
    img_url = data["data"][0]["url"]
    img_resp = _request_with_retries("GET", img_url, delay_seconds=cfg.rate_limit_delay_seconds)
    dest.write_bytes(img_resp.content)

