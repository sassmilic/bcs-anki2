"""Tier 3: Full word pipeline integration tests (all mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

from bcs_anki.pipeline import RunContext, process_word
from bcs_anki.csv_writer import ensure_header
from bcs_anki.llm import GeneratedText
from bcs_anki.progress import ProgressState, save_progress, load_progress


def _make_state(input_file: str = "words.txt") -> ProgressState:
    return ProgressState(
        input_file=input_file,
        total_words=5,
        completed_words=[],
        failed_words=[],
        last_updated="",
    )


def _make_ctx(cfg, state, tmp_path: Path) -> RunContext:
    return RunContext(
        cfg=cfg,
        state=state,
        out_csv=tmp_path / "output.csv",
        progress_file=tmp_path / "progress.json",
        failed_csv=tmp_path / "failed.tsv",
    )


class TestProcessWordSuccess:
    @patch("bcs_anki.pipeline.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.pipeline._fetch_image")
    @patch("bcs_anki.pipeline.generate_definition_and_examples")
    def test_returns_true_and_writes_csv(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        state = _make_state()
        ctx = _make_ctx(mock_cfg, state, tmp_path)
        save_progress(ctx.progress_file, state)

        mock_gen.return_value = GeneratedText(
            definition_html="{{c1::primirje}} — def",
            examples_html="1. Ex {{c1::one}}.<br>2. Ex {{c1::two}}.<br>3. Ex {{c1::three}}.",
        )
        mock_img.return_value = [("primirje_abc123.png", tmp_path / "primirje_abc123.png")]

        result = process_word("primirje", ctx)

        assert result is True
        assert "primirje" in state.completed_words

        # CSV should have header (4 lines) + 3 data rows
        lines = ctx.out_csv.read_text(encoding="utf-8").strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 3

    @patch("bcs_anki.pipeline.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.pipeline._fetch_image")
    @patch("bcs_anki.pipeline.generate_definition_and_examples")
    def test_csv_has_correct_note_types(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        state = _make_state()
        ctx = _make_ctx(mock_cfg, state, tmp_path)
        save_progress(ctx.progress_file, state)

        mock_gen.return_value = GeneratedText(
            definition_html="{{c1::test}} — def", examples_html="Ex {{c1::test}}."
        )
        mock_img.return_value = [("img.png", tmp_path / "img.png")]

        process_word("test", ctx)

        lines = ctx.out_csv.read_text(encoding="utf-8").strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("#")]
        assert data_lines[0].startswith("Cloze")
        assert data_lines[1].startswith("Cloze")
        assert data_lines[2].startswith("Basic (and reversed card)")


class TestProcessWordFailure:
    @patch("bcs_anki.pipeline.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.pipeline._fetch_image")
    @patch("bcs_anki.pipeline.generate_definition_and_examples")
    def test_returns_false_on_error(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        state = _make_state()
        ctx = _make_ctx(mock_cfg, state, tmp_path)
        save_progress(ctx.progress_file, state)

        mock_gen.side_effect = RuntimeError("API error")
        mock_img.return_value = [("img.png", tmp_path / "img.png")]

        result = process_word("fail_word", ctx)

        assert result is False
        assert "fail_word" in state.failed_words
        assert ctx.failed_csv.exists()
        rows = ctx.failed_csv.read_text(encoding="utf-8").strip().splitlines()
        assert any(row.startswith("fail_word\t") and "API error" in row for row in rows)
