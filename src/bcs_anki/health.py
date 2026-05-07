"""Startup health check for all APIs the pipeline depends on.

Each check is best-effort cheap: a 1-token chat completion validates auth, model
access, and rate-limit headroom in a single call. Image-model availability is
verified via the (free) models.list endpoint. Errors are accumulated so the
user sees every misconfiguration in one shot rather than one at a time.
"""
from __future__ import annotations

import logging
from typing import Callable

import click
import requests
from openai import OpenAI

from .config import AppConfig

logger = logging.getLogger(__name__)


def _check_openai(cfg: AppConfig) -> None:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=cfg.openai_api_key)
    # 1-token chat: validates auth, chat model, and rate-limit headroom in one call.
    client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": "ping"}],
        max_completion_tokens=64,
    )
    # Image model availability via models.list (free).
    ids = {m.id for m in client.models.list().data}
    if cfg.image_generation_model not in ids:
        raise RuntimeError(
            f"image model '{cfg.image_generation_model}' is not available to this account "
            f"(not in client.models.list())"
        )


def _check_gemini(cfg: AppConfig) -> None:
    # Imported lazily so test environments without the SDK don't fail at import.
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=cfg.gemini_api_key)
    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents="ping",
        config=genai_types.GenerateContentConfig(max_output_tokens=1),
    )
    # Don't require .text — max_output_tokens=1 can yield empty content. The
    # absence of an exception above is what we care about.
    _ = response


def _check_stock_image(cfg: AppConfig) -> None:
    api = (cfg.stock_image_api or "").lower()
    timeout = 10
    if api == "unsplash":
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            headers={"Authorization": f"Client-ID {cfg.stock_image_api_key}"},
            params={"query": "test", "per_page": 1},
            timeout=timeout,
        )
    elif api == "pexels":
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": cfg.stock_image_api_key},
            params={"query": "test", "per_page": 1},
            timeout=timeout,
        )
    elif api == "pixabay":
        r = requests.get(
            "https://pixabay.com/api/",
            params={"key": cfg.stock_image_api_key, "q": "test", "per_page": 3},
            timeout=timeout,
        )
    else:
        raise RuntimeError(f"unknown stock_image_api: {cfg.stock_image_api!r}")
    r.raise_for_status()


def _run_check(label: str, fn: Callable[[], None], errors: list[str]) -> None:
    click.echo(f"  {label} ... ", nl=False)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        click.echo(f"FAILED ({type(exc).__name__}: {exc})")
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        logger.exception("%s health check failed", label)
    else:
        click.echo("OK")


def check_apis(cfg: AppConfig) -> None:
    """Run every applicable API health check. Raise RuntimeError if any fails.

    Stock-image and Gemini checks are skipped when their keys aren't set —
    those APIs are optional. OpenAI is required.
    """
    click.echo("Checking API keys...")
    errors: list[str] = []

    _run_check(
        f"OpenAI ({cfg.llm_model} + {cfg.image_generation_model})",
        lambda: _check_openai(cfg),
        errors,
    )
    if cfg.gemini_api_key:
        _run_check(f"Gemini ({cfg.gemini_model})", lambda: _check_gemini(cfg), errors)
    else:
        click.echo("  Gemini ... skipped (GEMINI_API_KEY not set)")
    if cfg.stock_image_api_key:
        _run_check(
            f"Stock images ({cfg.stock_image_api})",
            lambda: _check_stock_image(cfg),
            errors,
        )
    else:
        click.echo("  Stock images ... skipped (no provider key set)")

    if errors:
        msg = "API health check failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(msg)
    click.echo("All API checks passed.\n")
