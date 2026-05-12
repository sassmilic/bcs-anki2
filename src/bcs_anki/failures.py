from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig


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
    """Return a short, single-line description of an exception for failed.tsv."""
    msg = str(exc)
    brace = msg.find("{")
    if brace > 0:
        msg = msg[:brace].rstrip(" .:,-")
    msg = " ".join(msg.split())
    summary = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    if len(summary) > max_len:
        summary = summary[: max_len - 1] + "…"
    return summary


def append_failed(ctx: RunContext, word: str, reason: str) -> None:
    safe_reason = reason.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    with ctx.failed_lock:
        with ctx.failed_csv.open("a", encoding="utf-8") as f:
            f.write(f"{word}\t{safe_reason}\n")
