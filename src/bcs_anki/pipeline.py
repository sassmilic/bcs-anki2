from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click

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
from .progress import (
    ProgressState,
    load_progress,
    mark_completed,
    mark_failed,
    progress_path_for,
)

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Per-run state and locks shared across worker threads.

    Bundles the output paths, the in-memory progress state, and the
    synchronization primitives that keep concurrent workers from racing on
    shared resources (failed.tsv writes, lemma deduplication).
    """

    cfg: AppConfig
    state: ProgressState
    out_csv: Path
    progress_file: Path
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
    state = ctx.state
    logger.info("Processing input: %s", raw_word)
    word = raw_word

    claimed = False
    try:
        word = resolve_lemma(cfg, raw_word)
        if word != raw_word:
            logger.info("Resolved lemma '%s' -> '%s'", raw_word, word)

        with ctx.lemma_lock:
            if word in state.completed_words or word in ctx.lemmas_in_progress:
                logger.info("Skipping '%s': lemma already completed or in progress", word)
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
        mark_completed(ctx.progress_file, state, word)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process word '%s': %s", word, exc)
        _append_failed(ctx, word, summarize_exception(exc))
        mark_failed(ctx.progress_file, state, word)
        return False
    finally:
        if claimed:
            with ctx.lemma_lock:
                ctx.lemmas_in_progress.discard(word)


def _edit_ocr_text(cfg: AppConfig, image_path: Path, raw_text: str) -> str:
    """Write OCR text to disk, open $EDITOR for human review, return the saved text.

    Falls back gracefully if the editor cannot be opened (e.g. non-interactive
    environments): logs a warning and returns the unedited text.
    """
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    ocr_path = cfg.output_folder / f"{image_path.stem}.ocr.txt"
    ocr_path.write_text(raw_text, encoding="utf-8")
    click.echo(
        f"\nOCR text for {image_path.name} written to {ocr_path}.\n"
        f"Edit it now — close your editor when done. (Set --no-edit-ocr to skip.)\n"
    )
    try:
        click.edit(filename=str(ocr_path))
    except click.UsageError as exc:
        logger.warning("Could not open editor (%s); proceeding with unedited OCR text.", exc)
        return raw_text
    edited = ocr_path.read_text(encoding="utf-8")
    click.echo(f"Resuming with {len(edited)} chars of OCR text from {ocr_path.name}.")
    return edited


def run_dictionary_pipeline(
    cfg: AppConfig,
    image_paths: list[Path],
    *,
    output_csv: Optional[Path] = None,
    resume: bool = False,
    fresh: bool = False,
    append: bool = False,
    edit_ocr: bool = True,
) -> tuple[int, int]:
    """Process one or more dictionary page images into Basic+reversed flashcards.

    Each image gets its own CSV and progress file (named after the image stem)
    unless `output_csv` is provided, in which case all images write into that one.

    When `edit_ocr` is True (default), pauses after Tesseract on each image to
    let the user review/correct the raw OCR text before parsing — the dominant
    quality lever per real-world testing.

    Returns (total_completed, total_failed) across all images.
    """
    from .dictionary import (
        DictionaryRunContext,
        canonicalize_ijekavian,
        ocr_page,
        parse_page,
        process_dictionary_entry,
        _slugify_tag,
    )

    total_completed = 0
    total_failed = 0

    for image_path in image_paths:
        logger.info("=== Dictionary image: %s ===", image_path)
        raw_text = ocr_page(image_path)
        if edit_ocr:
            raw_text = _edit_ocr_text(cfg, image_path, raw_text)
        page = parse_page(cfg, raw_text)
        if not page.entries:
            logger.warning("No entries parsed from %s; skipping.", image_path)
            continue

        section_slug = _slugify_tag(page.section) if page.section else _slugify_tag(image_path.stem)
        logger.info(
            "Page %s: section=%r (slug=%r), %d entries",
            image_path.name, page.section, section_slug, len(page.entries),
        )

        out_csv = output_csv or (cfg.output_folder / f"{image_path.stem}.csv")
        if not append and out_csv.exists() and output_csv is None:
            out_csv.unlink()
            logger.info("Removed existing output file: %s", out_csv)
        ensure_header(out_csv)

        failed_csv = cfg.output_folder / f"{image_path.stem}_failed.tsv"
        if not append and failed_csv.exists():
            failed_csv.unlink()
        ensure_failed_header(failed_csv)

        progress_file = progress_path_for(image_path, cfg.output_folder)
        if fresh or not resume or not progress_file.exists():
            state = ProgressState(
                input_file=str(image_path),
                total_words=len(page.entries),
                completed_words=[],
                failed_words=[],
                last_updated="",
            )
        else:
            existing = load_progress(progress_file)
            if existing and existing.input_file == str(image_path):
                state = existing
            else:
                state = ProgressState(
                    input_file=str(image_path),
                    total_words=len(page.entries),
                    completed_words=[],
                    failed_words=[],
                    last_updated="",
                )

        ijekavian = canonicalize_ijekavian(cfg, [e.serbian_raw for e in page.entries])
        logger.info("Canonicalized %d entries to ijekavian", len(ijekavian))

        ctx = DictionaryRunContext(
            cfg=cfg,
            state=state,
            out_csv=out_csv,
            progress_file=progress_file,
            failed_csv=failed_csv,
            section_slug=section_slug,
            failed_lock=threading.Lock(),
        )

        effective_workers = max(1, min(cfg.max_workers, len(page.entries)))
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = [
                pool.submit(process_dictionary_entry, entry, ijekavian[i], ctx)
                for i, entry in enumerate(page.entries)
            ]
            for fut in as_completed(futures):
                fut.result()

        completed = len(state.completed_words)
        failed = len(state.failed_words)
        total_completed += completed
        total_failed += failed
        logger.info(
            "Page %s done: %d completed, %d failed (out of %d entries)",
            image_path.name, completed, failed, len(page.entries),
        )

    return total_completed, total_failed
