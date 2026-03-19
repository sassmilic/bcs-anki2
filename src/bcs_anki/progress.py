from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Module-level lock for thread-safe progress mutations
_progress_lock = threading.Lock()


@dataclass
class ProgressState:
    input_file: str
    total_words: int
    completed_words: List[str]
    failed_words: List[str]
    last_updated: str


def progress_path_for(input_file: Path, output_folder: Path) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder / f".progress_{input_file.stem}.json"


def load_progress(path: Path) -> ProgressState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProgressState(
        input_file=data["input_file"],
        total_words=data.get("total_words", 0),
        completed_words=list(data.get("completed_words", [])),
        failed_words=list(data.get("failed_words", [])),
        last_updated=data.get("last_updated", ""),
    )


def save_progress(path: Path, state: ProgressState) -> None:
    with _progress_lock:
        state.last_updated = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def mark_completed(path: Path, state: ProgressState, word: str) -> None:
    """Thread-safe: mark a word as completed and save."""
    with _progress_lock:
        state.completed_words.append(word)
        if word in state.failed_words:
            state.failed_words.remove(word)
        state.last_updated = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def mark_failed(path: Path, state: ProgressState, word: str) -> None:
    """Thread-safe: mark a word as failed and save."""
    with _progress_lock:
        if word not in state.failed_words:
            state.failed_words.append(word)
        state.last_updated = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

