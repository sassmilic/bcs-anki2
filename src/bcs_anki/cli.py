from __future__ import annotations

import logging
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click

from .config import AppConfig, load_config
from .costs import COST_TRACKER
from .csv_writer import ensure_header
from .health import check_apis
from .logging_utils import setup_logging
from .pipeline import RunContext, ensure_failed_header, process_word, run_dictionary_pipeline
from .progress import ProgressState, load_progress, progress_path_for

logger = logging.getLogger(__name__)


def _load_app_config(config_path: Optional[str], verbose: bool = False) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else None
    cfg = load_config(path)
    setup_logging(cfg.log_file, verbose=verbose)
    return cfg


def _parse_review_csv(text: str) -> list[dict]:
    """Group an Anki-style CSV into one entry per word.

    Each word produces 0-2 Cloze rows followed by 0-1 Basic row. The Basic row
    carries the lemma in column 2; Cloze rows immediately preceding it belong
    to the same word. Cloze rows with no following Basic (image was skipped)
    are emitted as orphans with word="(no image)".
    """
    lines = text.splitlines()
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]

    def _emit(words_list: list, clozes: list, basic_word: str, image_file: str) -> None:
        words_list.append({
            "word": basic_word,
            "definition": clozes[0] if len(clozes) > 0 else "",
            "examples": clozes[1] if len(clozes) > 1 else "",
            "image_file": image_file,
        })

    words: list[dict] = []
    pending_clozes: list[str] = []
    for line in data_lines:
        parts = line.split("\t")
        if not parts:
            continue
        note_type = parts[0]
        if note_type == "Cloze":
            if len(pending_clozes) >= 2:
                _emit(words, pending_clozes, "(no image)", "")
                pending_clozes = []
            pending_clozes.append(parts[1] if len(parts) > 1 else "")
        elif note_type == "Basic (and reversed card)":
            word = parts[2] if len(parts) > 2 else "?"
            img_field = parts[1] if len(parts) > 1 else ""
            img_file = ""
            if 'src="' in img_field:
                img_file = img_field.split('src="')[1].split('"')[0]
            _emit(words, pending_clozes, word, img_file)
            pending_clozes = []

    if pending_clozes:
        _emit(words, pending_clozes, "(no image)", "")

    return words


def _process_word(
    raw_word: str,
    cfg: AppConfig,
    state: ProgressState,
    out_csv: Path,
    progress_file: Path,
    failed_csv: Path,
) -> bool:
    """Process a single word. Resolves the lemma first, then uses it for all downstream work."""
    logger.info("Processing input: %s", raw_word)
    word = raw_word

    claimed = False
    try:
        word = resolve_lemma(cfg, raw_word)
        if word != raw_word:
            logger.info("Resolved lemma '%s' -> '%s'", raw_word, word)

        with _lemma_lock:
            if word in state.completed_words or word in _lemmas_in_progress:
                logger.info("Skipping '%s': lemma already completed or in progress", word)
                return True
            _lemmas_in_progress.add(word)
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

        append_rows(out_csv, rows)
        mark_completed(progress_file, state, word)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process word '%s': %s", word, exc)
        _append_failed(failed_csv, word, _summarize_exception(exc))
        mark_failed(progress_file, state, word)
        return False
    finally:
        if claimed:
            with _lemma_lock:
                _lemmas_in_progress.discard(word)


@click.group()
@click.version_option(package_name="bcs-anki")
def main() -> None:
    """BCS-to-Anki flashcard generator CLI."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--output", "-o", "output_csv", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
@click.option("--anki-media", type=click.Path(file_okay=False, path_type=Path))
@click.option("--resume", "-r", is_flag=True, help="Resume from checkpoint if available.")
@click.option("--fresh", is_flag=True, help="Ignore checkpoint and start fresh.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without making API calls.")
@click.option("--workers", "-w", type=int, default=None, help="Max parallel workers (overrides config).")
@click.option("--append", is_flag=True, help="Append to existing CSV instead of overwriting.")
@click.option(
    "--dictionary-image",
    "dictionary_images",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="Path to a dictionary page image. Repeat for multiple pages. "
         "Skips definition/examples generation; produces image↔ijekavian flashcards only.",
)
@click.option(
    "--edit-ocr/--no-edit-ocr",
    default=True,
    help="With --dictionary-image: pause after OCR to let you edit the raw text before parsing. Default on.",
)
def generate(
    input_file: Optional[Path],
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    resume: bool,
    fresh: bool,
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
    append: bool,
    dictionary_images: tuple[Path, ...],
    edit_ocr: bool,
) -> None:
    """Generate Anki-ready CSV and images from a word list or dictionary page image(s)."""
    if not input_file and not dictionary_images:
        raise click.UsageError("Provide either INPUT_FILE or --dictionary-image PATH (or both).")

    cfg = _load_app_config(str(config_path) if config_path else None, verbose=verbose)
    if anki_media:
        cfg.anki_media_folder = anki_media.expanduser()
    if workers is not None:
        cfg.max_workers = workers

    # Log effective (sanitized) configuration
    safe_cfg = {
        "openai_api_key": "set" if cfg.openai_api_key else "not set",
        "gemini_api_key": "set" if cfg.gemini_api_key else "not set",
        "image_generation_model": cfg.image_generation_model,
        "image_size": cfg.image_size,
        "stock_image_api": cfg.stock_image_api or "none",
        "stock_image_api_key": "set" if cfg.stock_image_api_key else "not set",
        "anki_media_folder": str(cfg.anki_media_folder),
        "output_folder": str(cfg.output_folder),
        "temp_image_folder": str(cfg.temp_image_folder),
        "log_file": str(cfg.log_file),
        "rate_limit_delay_seconds": cfg.rate_limit_delay_seconds,
        "tags": cfg.tags,
        "llm_model": cfg.llm_model,
        "gemini_model": cfg.gemini_model,
        "max_workers": cfg.max_workers,
    }
    logger.info("Loaded configuration: %s", safe_cfg)

    if not dry_run:
        check_apis(cfg)

    if dictionary_images:
        click.echo(f"Processing {len(dictionary_images)} dictionary image(s)...")
        if dry_run:
            for img in dictionary_images:
                click.echo(f"[DRY-RUN] Would process dictionary image: {img}")
        else:
            completed, failed = run_dictionary_pipeline(
                cfg,
                list(dictionary_images),
                output_csv=output_csv,
                resume=resume,
                fresh=fresh,
                append=append,
                edit_ocr=edit_ocr,
            )
            click.echo(f"Dictionary pipeline done: {completed} completed, {failed} failed.")
            cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
            logger.info("Token/cost summary for this run: %s", cost_summary)
            click.echo(f"Token/cost summary: {cost_summary}")
        if input_file is None:
            return

    out_csv = output_csv or (cfg.output_folder / (input_file.stem + ".csv"))
    if not append and out_csv.exists():
        out_csv.unlink()
        logger.info("Removed existing output file: %s", out_csv)
    ensure_header(out_csv)

    failed_csv = cfg.output_folder / f"{input_file.stem}_failed.tsv"
    if not append and failed_csv.exists():
        failed_csv.unlink()
    ensure_failed_header(failed_csv)

    progress_file = progress_path_for(input_file, cfg.output_folder)
    words = [s for line in input_file.read_text(encoding="utf-8").splitlines() if (s := line.strip())]
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

    # Cheap pre-filter on raw input. The lemma may differ from the input, so
    # _process_word does a second dedup check after resolving the lemma.
    pending = [w for w in words if w not in state.completed_words]

    if dry_run:
        for w in pending:
            click.echo(f"[DRY-RUN] Would process: {w}")
    else:
        effective_workers = min(cfg.max_workers, len(pending)) if pending else 1
        click.echo(f"Using {effective_workers} workers for {len(pending)} remaining words.")

        ctx = RunContext(
            cfg=cfg,
            state=state,
            out_csv=out_csv,
            progress_file=progress_file,
            failed_csv=failed_csv,
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_word = {
                pool.submit(process_word, w, ctx): w
                for w in pending
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

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)

    click.echo("Done.")
    click.echo(f"Token/cost summary: {cost_summary}")
    if state.failed_words:
        click.echo(f"Failed words: {', '.join(state.failed_words)}")
        click.echo(f"See {failed_csv} for failure reasons.")


@main.command("copy-media")
@click.option("--from", "src", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Source folder (default: temp_image_folder from config).")
@click.option("--to", "dst", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Destination folder (default: anki_media_folder from config).")
@click.option("--config", "-c", "config_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
def copy_media(src: Optional[Path], dst: Optional[Path], config_path: Optional[Path]) -> None:
    """Copy generated images to the Anki media folder."""
    cfg = _load_app_config(str(config_path) if config_path else None)
    src = src or cfg.temp_image_folder
    dst = dst or cfg.anki_media_folder

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


@main.command()
@click.argument("csv_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--image-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help="Directory containing generated images.")
@click.option("--sample", "-n", type=int, default=None, help="Randomly sample N words to review.")
@click.option("--reject-file", "-r", type=click.Path(dir_okay=False, path_type=Path), default=None, help="File to write rejected words to.")
def review(csv_file: Path, image_dir: Optional[Path], sample: Optional[int], reject_file: Optional[Path]) -> None:
    """Review generated flashcards for subjective quality."""
    text = csv_file.read_text(encoding="utf-8")
    words = _parse_review_csv(text)

    if not words:
        click.echo("No words found in CSV.")
        return

    if sample is not None and sample < len(words):
        words = random.sample(words, sample)

    total = len(words)
    rejected: list[str] = []
    accepted = 0
    reviewed = 0

    for idx, entry in enumerate(words, 1):
        click.echo(f"\n{'─' * 3} Word {idx}/{total}: {entry['word']} {'─' * 40}")
        click.echo(f"Definition: {entry['definition'][:120]}...")
        click.echo(f"Examples:   {entry['examples'][:120]}...")

        img_display = entry["image_file"]
        if image_dir and entry["image_file"]:
            img_display = str(image_dir / entry["image_file"])
        click.echo(f"Image:      {img_display}")

        choice = ""
        while choice not in ("a", "r", "q"):
            click.echo("\n[a]ccept  [r]eject  [o]pen image  [q]uit")
            choice = click.getchar().lower()
            if choice == "o":
                if image_dir and entry["image_file"]:
                    img_path = image_dir / entry["image_file"]
                    if img_path.exists():
                        if sys.platform == "darwin":
                            subprocess.run(["open", str(img_path)], check=False)
                        elif sys.platform == "linux":
                            subprocess.run(["xdg-open", str(img_path)], check=False)
                        else:
                            subprocess.run(["start", str(img_path)], check=False, shell=True)
                    else:
                        click.echo(f"  Image not found: {img_path}")
                else:
                    click.echo("  No image directory specified (use --image-dir).")

        if choice == "a":
            accepted += 1
            reviewed += 1
        elif choice == "r":
            rejected.append(entry["word"])
            reviewed += 1
        else:  # quit
            click.echo("\nQuitting review.")
            break

    # Write rejected words
    out_reject = reject_file or Path("rejected.txt")
    if rejected:
        out_reject.write_text("\n".join(rejected) + "\n", encoding="utf-8")

    click.echo(f"\nReviewed {reviewed} words: {accepted} accepted, {len(rejected)} rejected.")
    if rejected:
        click.echo(f"Rejected words saved to {out_reject}")


if __name__ == "__main__":
    main()

