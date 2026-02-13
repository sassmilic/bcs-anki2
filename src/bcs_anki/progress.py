from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class ProgressState:
    input_file: str
    total_words: int
    completed_words: List[str]
    failed_words: List[str]
    last_updated: str


def progress_path_for(input_file: Path) -> Path:
    return input_file.with_suffix(input_file.suffix + ".progress.json")


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
    state.last_updated = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

