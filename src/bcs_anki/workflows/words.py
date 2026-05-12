from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click

from ..costs import COST_TRACKER
from ..csv_writer import ensure_header
from ..failures import RunContext, ensure_failed_header
from ..health import check_apis
from ..word_cards import process_word
from .common import load_app_config, log_effective_config

logger = logging.getLogger(__name__)


def run_words_pipeline(
    input_file: Path,
    output_csv: Optional[Path],
    config_path: Optional[Path],
    anki_media: Optional[Path],
    verbose: bool,
    dry_run: bool,
    workers: Optional[int],
    append: bool,
) -> None:
    cfg = load_app_config(config_path, verbose=verbose)
    if anki_media:
        cfg.anki_media_folder = anki_media.expanduser()
    if workers is not None:
        cfg.max_workers = workers

    log_effective_config(cfg)

    if not dry_run:
        try:
            check_apis(cfg)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from None

    out_csv = output_csv or (cfg.output_folder / (input_file.stem + ".csv"))
    if not append and out_csv.exists():
        out_csv.unlink()
        logger.info("Removed existing output file: %s", out_csv)
    ensure_header(out_csv)

    failed_csv = cfg.output_folder / f"{input_file.stem}_failed.tsv"
    if not append and failed_csv.exists():
        failed_csv.unlink()
    ensure_failed_header(failed_csv)

    words = [s for line in input_file.read_text(encoding="utf-8").splitlines() if (s := line.strip())]
    total_words = len(words)

    start_time = time.monotonic()
    processed_since_start = 0
    failed_words: list[str] = []

    click.echo(f"Processing {total_words} words from {input_file}...")

    if dry_run:
        for w in words:
            click.echo(f"[DRY-RUN] Would process: {w}")
    else:
        effective_workers = min(cfg.max_workers, len(words)) if words else 1
        click.echo(f"Using {effective_workers} workers for {len(words)} words.")

        ctx = RunContext(
            cfg=cfg,
            out_csv=out_csv,
            failed_csv=failed_csv,
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_word = {
                pool.submit(process_word, w, ctx): w
                for w in words
            }

            for future in as_completed(future_to_word):
                word = future_to_word[future]
                success = future.result()
                if not success:
                    failed_words.append(word)
                processed_since_start += 1

                # Periodic progress logging.
                if processed_since_start % 10 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_per_word = elapsed / processed_since_start
                    remaining = len(words) - processed_since_start
                    if remaining < 0:
                        remaining = 0
                    eta_seconds = remaining * avg_per_word
                    minutes, seconds = divmod(int(eta_seconds), 60)
                    eta_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

                    percent = (processed_since_start / total_words * 100) if total_words else 100.0
                    logger.info(
                        "Progress: %d/%d words processed (%.1f%%). Remaining: %d. Approximate remaining time: %s.",
                        processed_since_start,
                        total_words,
                        percent,
                        remaining,
                        eta_str,
                    )

    logger.info(
        "Finished processing. Total: %d, failed: %d.",
        total_words,
        len(failed_words),
    )

    cost_summary = COST_TRACKER.summary(cfg.llm_model, cfg.gemini_model)
    logger.info("Token/cost summary for this run: %s", cost_summary)

    click.echo("Done.")
    click.echo(f"Token/cost summary: {cost_summary}")
    if failed_words:
        click.echo(f"Failed words: {', '.join(failed_words)}")
        click.echo(f"See {failed_csv} for failure reasons.")
