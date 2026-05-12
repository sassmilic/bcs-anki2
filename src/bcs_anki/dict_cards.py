"""Generate Anki Basic+reversed flashcards (image ↔ Serbian) from refined dict CSVs.

Per row: try a stock image with the English term; on any stock failure, fall
back to AI image generation with a static prompt template. Write a CSV row
mapping the image filename to the Serbian word, tagged with the subject.

No LLM calls per row — the English term is used directly for both stock
search and (formatted into AI_FALLBACK_PROMPT for) AI generation.
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_LEADING_THE = re.compile(r"^the\s+", re.IGNORECASE)


def _strip_leading_the(text: str) -> str:
    """Drop a leading 'the ' (case-insensitive). Used for queries/prompts only."""
    return _LEADING_THE.sub("", text)

from .config import AppConfig
from .csv_writer import CsvRow, append_rows, ensure_header
from .dict_ocr import read_dict_csv, subject_slug
from .errors import ImageRejectedError
from .images import build_image_filename, fetch_stock_image, generate_ai_image
from .pipeline import RunContext, _append_failed, ensure_failed_header, summarize_exception
from .prompts import AI_FALLBACK_PROMPT

logger = logging.getLogger(__name__)


def _process_row(
    english: str,
    serbian: str,
    subject: str,
    tag_str: str,
    ctx: RunContext,
) -> bool:
    """Fetch/generate image for one row; write CSV row.

    Returns True on success, False on per-row failure (still recorded; run continues).
    """
    cfg = ctx.cfg
    query = _strip_leading_the(english)
    img_filename = build_image_filename(query)
    img_path = cfg.temp_image_folder / img_filename
    cfg.temp_image_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Multi-word entries (e.g. "Pisces (the Fish)", "rotary motions") are
        # almost never well-served by stock photo search — skip straight to AI.
        if len(query.split()) > 1:
            logger.info("Multi-word entry '%s'; skipping stock, generating AI image", english)
            ai_prompt = AI_FALLBACK_PROMPT.format(english=query, subject=subject)
            generate_ai_image(cfg, ai_prompt, img_path)
        else:
            try:
                paths = fetch_stock_image(cfg, query, img_path, count=1)
                if not paths:
                    raise RuntimeError("fetch_stock_image returned no paths")
                logger.info("Stock image for '%s' → %s", english, paths[0].name)
            except Exception as stock_exc:  # noqa: BLE001
                # Any stock failure (no results, HTTP errors like 403/429 from
                # rate limits, network issues, missing key) falls back to AI.
                logger.warning(
                    "Stock image failed for '%s' (%s: %s); falling back to AI",
                    english, type(stock_exc).__name__, stock_exc,
                )
                ai_prompt = AI_FALLBACK_PROMPT.format(english=query, subject=subject)
                generate_ai_image(cfg, ai_prompt, img_path)
                logger.info("AI image generated for '%s'", english)

        row = CsvRow(
            note_type="Basic (and reversed card)",
            field1=f'<img src="{img_filename}">',
            field2=serbian,
            tags=tag_str,
        )
        append_rows(ctx.out_csv, [row])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Row failed for '%s': %s", english, exc)
        _append_failed(ctx, english, summarize_exception(exc))
        return False


def _build_tags(cfg: AppConfig, slug: str) -> str:
    return f"{cfg.tags} {slug}".strip() if cfg.tags else slug


def run_generate_dict(
    cfg: AppConfig,
    csv_path: Path,
    output_csv: Optional[Path] = None,
    *,
    append: bool = False,
) -> tuple[int, int]:
    """Process one refined dict CSV → Anki flashcards CSV.

    Returns (completed, failed) row counts.
    """
    subject, rows = read_dict_csv(csv_path)
    slug = subject_slug(subject)
    out = output_csv or (cfg.output_folder / "cards" / f"{slug}.csv")
    failed_csv = out.with_name(f"{out.stem}_failed.tsv")

    out.parent.mkdir(parents=True, exist_ok=True)

    if not append and out.exists():
        out.unlink()
        logger.info("Removed existing output file: %s", out)
    if not append and failed_csv.exists():
        failed_csv.unlink()
    ensure_header(out)
    ensure_failed_header(failed_csv)

    ctx = RunContext(
        cfg=cfg,
        out_csv=out,
        failed_csv=failed_csv,
    )
    tag_str = _build_tags(cfg, slug)

    if not rows:
        return 0, 0

    logger.info(
        "Processing %d row(s) with %d worker(s)",
        len(rows), cfg.max_workers,
    )

    completed = 0
    failed = 0
    workers = max(1, min(cfg.max_workers, len(rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_row, eng, sr, subject, tag_str, ctx)
            for eng, sr in rows
        ]
        for f in as_completed(futures):
            if f.result():
                completed += 1
            else:
                failed += 1

    return completed, failed
