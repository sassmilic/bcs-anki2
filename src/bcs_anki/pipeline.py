from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig
from .csv_writer import CsvRow, append_rows, ensure_header
from .errors import ImageRejectedError
from .images import (
    ImageSource,
    build_image_filename,
    fetch_stock_image,
    generate_ai_image,
)
from .llm import (
    decide_image_source,
    generate_definition_and_examples,
    generate_image_prompt,
    generate_image_search_term,
    resolve_lemma,
)

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Per-run output paths and locks shared across worker threads."""

    cfg: AppConfig
    out_csv: Path
    failed_csv: Path
    failed_lock: threading.Lock = field(default_factory=threading.Lock)
    lemma_lock: threading.Lock = field(default_factory=threading.Lock)
    lemmas_in_progress: set[str] = field(default_factory=set)


def ensure_failed_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("word\treason\n", encoding="utf-8")


def summarize_exception(exc: BaseException, max_len: int = 160) -> str:
    """Return a short, single-line description of an exception for the failed.tsv.

    Strips verbose API payloads (anything after the first `{`, where Google/OpenAI
    error JSON typically begins) and collapses whitespace. Full traceback still
    goes to the log via logger.exception.
    """
    msg = str(exc)
    brace = msg.find("{")
    if brace > 0:
        msg = msg[:brace].rstrip(" .:,-")
    msg = " ".join(msg.split())
    summary = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    if len(summary) > max_len:
        summary = summary[: max_len - 1] + "…"
    return summary


def _append_failed(ctx: RunContext, word: str, reason: str) -> None:
    safe_reason = reason.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    with ctx.failed_lock:
        with ctx.failed_csv.open("a", encoding="utf-8") as f:
            f.write(f"{word}\t{safe_reason}\n")


def _fetch_image(cfg: AppConfig, word: str) -> list[tuple[str, Path]] | None:
    """Fetch or generate images for a word. Returns list of (filename, path) or None."""
    img_source: ImageSource = decide_image_source(cfg, word)
    img_filename = build_image_filename(word)
    img_path = cfg.temp_image_folder / img_filename
    cfg.temp_image_folder.mkdir(parents=True, exist_ok=True)

    if img_source == "stock":
        logger.info("Using stock image for '%s'", word)
        try:
            search_term_en = generate_image_search_term(cfg, word)
            logger.info("Image search term (EN) for '%s': %s", word, search_term_en)
            paths = fetch_stock_image(cfg, search_term_en, img_path)
            return [(p.name, p) for p in paths]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Stock image failed for '%s', falling back to AI image: %s",
                word,
                exc,
            )
            img_source = "ai"

    if img_source == "ai":
        logger.info("Using AI image for '%s'", word)
        img_prompt = generate_image_prompt(cfg, word)
        logger.info("Image prompt for '%s': %s", word, img_prompt)
        try:
            generate_ai_image(cfg, img_prompt, img_path)
        except ImageRejectedError as exc:
            logger.warning(
                "AI image rejected by safety filter for '%s', regenerating prompt and retrying: %s",
                word, exc,
            )
            img_prompt = generate_image_prompt(cfg, word)
            logger.info("Retry image prompt for '%s': %s", word, img_prompt)
            try:
                generate_ai_image(cfg, img_prompt, img_path)
            except ImageRejectedError:
                logger.warning(
                    "AI image retry also rejected for '%s', skipping image", word,
                )
                return None

    return [(img_filename, img_path)]


def process_word(raw_word: str, ctx: RunContext) -> bool:
    """Process a single word. Resolves the lemma first, then uses it for all downstream work."""
    cfg = ctx.cfg
    logger.info("Processing input: %s", raw_word)
    word = raw_word

    claimed = False
    try:
        word = resolve_lemma(cfg, raw_word)
        if word != raw_word:
            logger.info("Resolved lemma '%s' -> '%s'", raw_word, word)

        with ctx.lemma_lock:
            if word in ctx.lemmas_in_progress:
                logger.info("Skipping '%s': lemma already in progress", word)
                return True
            ctx.lemmas_in_progress.add(word)
            claimed = True

        with ThreadPoolExecutor(max_workers=2) as inner_pool:
            def_future = inner_pool.submit(generate_definition_and_examples, cfg, word)
            img_future = inner_pool.submit(_fetch_image, cfg, word)

            gen = def_future.result()
            img_result = img_future.result()

        rows = []
        if "{{c1::" in gen.definition_html:
            rows.append(CsvRow(
                note_type="Cloze",
                field1=gen.definition_html,
                field2="",
                tags=cfg.tags,
            ))
        else:
            logger.warning("Skipping definition card for '%s': no cloze markers found", word)

        if "{{c1::" in gen.examples_html:
            rows.append(CsvRow(
                note_type="Cloze",
                field1=gen.examples_html,
                field2="",
                tags=cfg.tags,
            ))
        else:
            logger.warning("Skipping examples card for '%s': no cloze markers found", word)

        if img_result is not None:
            img_html = "".join(f'<img src="{fn}">' for fn, _path in img_result)
            rows.append(CsvRow(
                note_type="Basic (and reversed card)",
                field1=img_html,
                field2=word.lower(),
                tags=cfg.tags,
            ))

        append_rows(ctx.out_csv, rows)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process word '%s': %s", word, exc)
        _append_failed(ctx, word, summarize_exception(exc))
        return False
    finally:
        if claimed:
            with ctx.lemma_lock:
                ctx.lemmas_in_progress.discard(word)
