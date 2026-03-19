"""Tier 2: Progress tracking tests with file I/O."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bcs_anki.progress import (
    ProgressState,
    load_progress,
    mark_completed,
    mark_failed,
    progress_path_for,
    save_progress,
)


def _make_state(**overrides) -> ProgressState:
    defaults = dict(
        input_file="words.txt",
        total_words=10,
        completed_words=[],
        failed_words=[],
        last_updated="",
    )
    defaults.update(overrides)
    return ProgressState(**defaults)


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "progress.json"
        state = _make_state(completed_words=["a", "b"], failed_words=["c"])
        save_progress(path, state)
        loaded = load_progress(path)
        assert loaded is not None
        assert loaded.input_file == state.input_file
        assert loaded.total_words == state.total_words
        assert loaded.completed_words == ["a", "b"]
        assert loaded.failed_words == ["c"]
        assert loaded.last_updated != ""

    def test_load_nonexistent(self, tmp_path: Path):
        assert load_progress(tmp_path / "nope.json") is None


class TestMarkCompleted:
    def test_adds_word_and_saves(self, tmp_path: Path):
        path = tmp_path / "progress.json"
        state = _make_state()
        save_progress(path, state)
        mark_completed(path, state, "hello")
        assert "hello" in state.completed_words
        loaded = load_progress(path)
        assert "hello" in loaded.completed_words

    def test_removes_from_failed(self, tmp_path: Path):
        path = tmp_path / "progress.json"
        state = _make_state(failed_words=["hello"])
        save_progress(path, state)
        mark_completed(path, state, "hello")
        assert "hello" not in state.failed_words
        assert "hello" in state.completed_words

    def test_thread_safety(self, tmp_path: Path):
        path = tmp_path / "progress.json"
        state = _make_state()
        save_progress(path, state)

        words = [f"word_{i}" for i in range(20)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(mark_completed, path, state, w) for w in words]
            for f in as_completed(futures):
                f.result()

        assert len(state.completed_words) == 20
        assert set(state.completed_words) == set(words)


class TestMarkFailed:
    def test_idempotent(self, tmp_path: Path):
        path = tmp_path / "progress.json"
        state = _make_state()
        save_progress(path, state)
        mark_failed(path, state, "bad")
        mark_failed(path, state, "bad")
        assert state.failed_words.count("bad") == 1


class TestProgressPathFor:
    def test_expected_path(self, tmp_path: Path):
        result = progress_path_for(Path("my_words.txt"), tmp_path)
        assert result == tmp_path / ".progress_my_words.json"
        assert tmp_path.exists()
