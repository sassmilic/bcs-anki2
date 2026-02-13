from __future__ import annotations

import logging
import shutil
import time
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
from .llm import generate_definition_and_examples, generate_image_prompt
from .logging_utils import setup_logging
from .progress import ProgressState, load_progress, progress_path_for, save_progress

logger = logging.getLogger(__name__)


def _load_app_config(config_path: Optional[str]) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else None
    cfg = load_config(path)
    return cfg


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
def generate(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    resume: bool,
    fresh: bool,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Generate Anki-ready CSV and images from a word list."""
    cfg = _load_app_config(str(config_path) if config_path else None)
    if anki_media:
        cfg.anki_media_folder = anki_media.expanduser()

    setup_logging(cfg.log_file, verbose=verbose)

    out_csv = output_csv or (cfg.output_folder / (input_file.stem + ".csv"))
    ensure_header(out_csv)

    progress_file = progress_path_for(input_file)
    words = [w.strip() for w in input_file.read_text(encoding="utf-8").splitlines() if w.strip()]
    total_words = len(words)

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

    for word in words:
        if word in state.completed_words:
            logger.info("Skipping already completed word: %s", word)
            continue

        # Za sada samo prosljeđujemo riječ direktno; ovdje se može dodati
        # naprednija normalizacija (npr. uklanjanje dijakritika, mala/velika slova).
        normalized = word

        canonical = word  # Placeholder: in a full app, add POS detection & canonicalization.
        canonical_info = "kanonski oblik; gramatičke informacije nisu specificirane"

        logger.info("Processing word: %s (canonical: %s)", normalized, canonical)

        if dry_run:
            click.echo(f"[DRY-RUN] Would process: {normalized}")
            continue

        try:
            # LLM definitions and examples
            gen = generate_definition_and_examples(cfg, normalized, canonical_info)
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

            # Image logic
            img_source: ImageSource = decide_image_source(normalized, canonical_info)
            img_filename = build_image_filename(normalized)
            img_path = cfg.temp_image_folder / img_filename
            cfg.temp_image_folder.mkdir(parents=True, exist_ok=True)

            if img_source == "stock":
                logger.info("Using stock image for '%s'", normalized)
                # In a real implementation, we would translate to English first.
                fetch_stock_image(cfg, normalized, img_path)
            else:
                logger.info("Using AI image for '%s'", normalized)
                img_prompt = generate_image_prompt(cfg, normalized, canonical_info)
                logger.info("Image prompt for '%s': %s", normalized, img_prompt)
                generate_ai_image(cfg, img_prompt, img_path)

            img_html = f'<img src="{img_filename}">'
            img_row = CsvRow(
                note_type="Basic (and reversed card)",
                field1=img_html,
                field2=canonical,
                tags=cfg.tags,
            )

            append_rows(out_csv, [def_row, ex_row, img_row])

            state.completed_words.append(word)
            if word in state.failed_words:
                state.failed_words.remove(word)
            save_progress(progress_file, state)
            processed_since_start += 1

            # Periodic progress logging
            if processed_since_start and processed_since_start % 10 == 0:
                elapsed = time.monotonic() - start_time
                avg_per_word = elapsed / processed_since_start
                remaining = total_words - (already_done + processed_since_start)
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
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process word '%s': %s", word, exc)
            if word not in state.failed_words:
                state.failed_words.append(word)
            save_progress(progress_file, state)

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
def status(input_file: Path) -> None:
    """Show progress for a given input file."""
    progress_file = progress_path_for(input_file)
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

