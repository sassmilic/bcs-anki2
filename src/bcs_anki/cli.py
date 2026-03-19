from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

from .config import AppConfig, load_config
from .csv_writer import CsvRow, append_rows, ensure_header
from .images import (
    ImageSource,
    build_image_filename,
    decide_image_source,
    fetch_stock_image,
    generate_ai_image,
)
from .llm import generate_definition_and_examples, generate_image_prompt, generate_image_search_term
from .logging_utils import setup_logging
from .progress import ProgressState, load_progress, mark_completed, mark_failed, progress_path_for

logger = logging.getLogger(__name__)


@dataclass
class WordEntry:
    word: str
    context: str | None


def parse_word_line(line: str) -> WordEntry:
    if "|" in line:
        word, context = line.split("|", maxsplit=1)
        return WordEntry(word=word.strip(), context=context.strip())
    return WordEntry(word=line.strip(), context=None)


def _load_app_config(config_path: Optional[str]) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else None
    cfg = load_config(path)
    return cfg


def _fetch_image(cfg: AppConfig, entry: WordEntry, word: str) -> tuple[str, Path]:
    """Fetch or generate an image for a word. Returns (img_filename, img_path)."""
    img_source: ImageSource = decide_image_source(word)
    img_filename = build_image_filename(word)
    img_path = cfg.temp_image_folder / img_filename
    cfg.temp_image_folder.mkdir(parents=True, exist_ok=True)

    if img_source == "stock":
        logger.info("Using stock image for '%s'", word)
        try:
            search_term_en = generate_image_search_term(cfg, word, context=entry.context)
            logger.info("Image search term (EN) for '%s': %s", word, search_term_en)
            fetch_stock_image(cfg, search_term_en, img_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Stock image failed for '%s', falling back to AI image: %s",
                word,
                exc,
            )
            img_source = "ai"

    if img_source == "ai":
        logger.info("Using AI image for '%s'", word)
        img_prompt = generate_image_prompt(cfg, word, context=entry.context)
        logger.info("Image prompt for '%s': %s", word, img_prompt)
        generate_ai_image(cfg, img_prompt, img_path)

    return img_filename, img_path


def _process_word(
    entry: WordEntry,
    cfg: AppConfig,
    state: ProgressState,
    out_csv: Path,
    progress_file: Path,
) -> bool:
    """Process a single word with within-word parallelism. Returns True on success."""
    word = entry.word
    logger.info("Processing word: %s", word)

    try:
        # Run definition and image generation in parallel
        with ThreadPoolExecutor(max_workers=2) as inner_pool:
            def_future = inner_pool.submit(
                generate_definition_and_examples, cfg, word, context=entry.context
            )
            img_future = inner_pool.submit(_fetch_image, cfg, entry, word)

            gen = def_future.result()
            img_filename, _img_path = img_future.result()

        def_row = CsvRow(
            note_type="Cloze",
            field1=gen.definition_html,
            field2="",
            tags=cfg.tags,
        )
        ex_row = CsvRow(
            note_type="Cloze",
            field1=gen.examples_html,
            field2="",
            tags=cfg.tags,
        )
        img_html = f'<img src="{img_filename}">'
        img_row = CsvRow(
            note_type="Basic (and reversed card)",
            field1=img_html,
            field2=word,
            tags=cfg.tags,
        )

        append_rows(out_csv, [def_row, ex_row, img_row])
        mark_completed(progress_file, state, word)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process word '%s': %s", word, exc)
        mark_failed(progress_file, state, word)
        return False


@click.group()
@click.version_option(package_name="bcs-anki")
def main() -> None:
    """BCS-to-Anki flashcard generator CLI."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", "output_csv", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--anki-media", type=click.Path(file_okay=False, path_type=Path))
@click.option("--resume", "-r", is_flag=True, help="Resume from checkpoint if available.")
@click.option("--fresh", is_flag=True, help="Ignore checkpoint and start fresh.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without making API calls.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
def generate(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    resume: bool,
    fresh: bool,
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
) -> None:
    """Generate Anki-ready CSV and images from a word list."""
    cfg = _load_app_config(str(config_path) if config_path else None)
    if anki_media:
        cfg.anki_media_folder = anki_media.expanduser()
    if workers is not None:
        cfg.max_workers = workers

    setup_logging(cfg.log_file, verbose=verbose)

    # Log effective (sanitized) configuration
    safe_cfg = {
        "openai_api_key": "set" if cfg.openai_api_key else "not set",
        "image_generation_model": cfg.image_generation_model,
        "image_size": cfg.image_size,
        "stock_image_api": cfg.stock_image_api,
        "stock_image_api_key": "set" if cfg.stock_image_api_key else "not set",
        "anki_media_folder": str(cfg.anki_media_folder),
        "output_folder": str(cfg.output_folder),
        "temp_image_folder": str(cfg.temp_image_folder),
        "log_file": str(cfg.log_file),
        "rate_limit_delay_seconds": cfg.rate_limit_delay_seconds,
        "tags": cfg.tags,
        "llm_model": cfg.llm_model,
        "max_workers": cfg.max_workers,
    }
    logger.info("Loaded configuration: %s", safe_cfg)

    out_csv = output_csv or (cfg.output_folder / (input_file.stem + ".csv"))
    ensure_header(out_csv)

    progress_file = progress_path_for(input_file, cfg.output_folder)
    entries = [parse_word_line(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_words = len(entries)

    state: ProgressState
    if fresh or not resume or not progress_file.exists():
        state = ProgressState(
            input_file=str(input_file),
            total_words=total_words,
            completed_words=[],
            failed_words=[],
            last_updated="",
        )
    else:
        existing = load_progress(progress_file)
        if existing and existing.input_file == str(input_file):
            state = existing
        else:
            state = ProgressState(
                input_file=str(input_file),
                total_words=total_words,
                completed_words=[],
                failed_words=[],
                last_updated="",
            )

    already_done = len(state.completed_words) + len(state.failed_words)
    start_time = time.monotonic()
    processed_since_start = 0

    click.echo(f"Processing {total_words} words from {input_file} (already done: {already_done})...")

    # Filter to words that still need processing
    pending = [e for e in entries if e.word not in state.completed_words]

    if dry_run:
        for entry in pending:
            ctx_info = f" (context: {entry.context})" if entry.context else ""
            click.echo(f"[DRY-RUN] Would process: {entry.word}{ctx_info}")
    else:
        effective_workers = min(cfg.max_workers, len(pending)) if pending else 1
        click.echo(f"Using {effective_workers} workers for {len(pending)} remaining words.")

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_word = {
                pool.submit(_process_word, entry, cfg, state, out_csv, progress_file): entry.word
                for entry in pending
            }

            for future in as_completed(future_to_word):
                word = future_to_word[future]
                success = future.result()
                processed_since_start += 1

                # Periodic progress logging
                if processed_since_start % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_per_word = elapsed / processed_since_start
                    remaining = len(pending) - processed_since_start
                    if remaining < 0:
                        remaining = 0
                    eta_seconds = remaining * avg_per_word
                    minutes, seconds = divmod(int(eta_seconds), 60)
                    eta_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

                    done = already_done + processed_since_start
                    percent = (done / total_words * 100) if total_words else 100.0
                    logger.info(
                        "Progress: %d/%d words processed (%.1f%%). Remaining: %d. Approximate remaining time: %s.",
                        done,
                        total_words,
                        percent,
                        remaining,
                        eta_str,
                    )

    # Final progress/ETA log
    done_total = len(state.completed_words) + len(state.failed_words)
    remaining_total = max(total_words - done_total, 0)
    logger.info(
        "Finished processing. Total: %d, completed/failed: %d, remaining (unprocessed in file): %d.",
        total_words,
        done_total,
        remaining_total,
    )

    click.echo("Done.")
    if state.failed_words:
        click.echo(f"Failed words: {', '.join(state.failed_words)}")


@main.command("copy-media")
@click.option("--from", "src", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--to", "dst", required=True, type=click.Path(file_okay=False, path_type=Path))
def copy_media(src: Path, dst: Path) -> None:
    """Copy generated images to the Anki media folder."""
    if not src.exists():
        raise click.ClickException(f"Source folder does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for file in src.iterdir():
        if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            shutil.copy2(file, dst / file.name)
            count += 1

    click.echo(f"Copied {count} media files from {src} to {dst}.")


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
def status(input_file: Path, config_path: Optional[Path]) -> None:
    """Show progress for a given input file."""
    cfg = _load_app_config(str(config_path) if config_path else None)
    progress_file = progress_path_for(input_file, cfg.output_folder)
    state = load_progress(progress_file)
    if not state:
        click.echo("No progress found.")
        return

    completed = len(state.completed_words)
    failed = len(state.failed_words)
    click.echo(f"Input file: {state.input_file}")
    click.echo(f"Total words: {state.total_words}")
    click.echo(f"Completed: {completed}")
    click.echo(f"Failed: {failed}")
    click.echo(f"Last updated: {state.last_updated}")


@main.command()
@click.argument("csv_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(csv_file: Path) -> None:
    """Validate that CSV file roughly matches Anki import format."""
    text = csv_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 4:
        raise click.ClickException("CSV file too short or missing header.")

    header = "\n".join(lines[:4])
    if "#separator:Tab" not in header or "#notetype column:1" not in header:
        raise click.ClickException("CSV header does not match required Anki format.")

    # Check some rows
    for i, line in enumerate(lines[4:], start=5):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise click.ClickException(f"Line {i} does not have 4 tab-separated columns.")
        note_type = parts[0]
        if note_type not in {"Cloze", "Basic (and reversed card)"}:
            raise click.ClickException(f"Line {i} has unexpected note type: {note_type}")

    click.echo("CSV appears valid for Anki import.")


if __name__ == "__main__":
    main()

