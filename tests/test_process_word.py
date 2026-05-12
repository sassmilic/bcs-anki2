"""Tier 3: Full word pipeline integration tests (all mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

from bcs_anki.csv_writer import ensure_header
from bcs_anki.failures import RunContext, ensure_failed_header
from bcs_anki.llm import GeneratedText
from bcs_anki.word_cards import process_word


def _make_ctx(cfg, tmp_path: Path) -> RunContext:
    failed_csv = tmp_path / "failed.tsv"
    ensure_failed_header(failed_csv)
    return RunContext(
        cfg=cfg,
        out_csv=tmp_path / "output.csv",
        failed_csv=failed_csv,
    )


class TestProcessWordSuccess:
    @patch("bcs_anki.word_cards.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.word_cards._fetch_image")
    @patch("bcs_anki.word_cards.generate_definition_and_examples")
    def test_returns_true_and_writes_csv(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path)

        mock_gen.return_value = GeneratedText(
            definition_html="{{c1::primirje}} — def",
            examples_html="1. Ex {{c1::one}}.<br>2. Ex {{c1::two}}.<br>3. Ex {{c1::three}}.",
        )
        mock_img.return_value = [("primirje_abc123.png", tmp_path / "primirje_abc123.png")]

        result = process_word("primirje", ctx)

        assert result is True

        # CSV should have header (4 lines) + 3 data rows
        lines = ctx.out_csv.read_text(encoding="utf-8").strip().splitlines()
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 3

    @patch("bcs_anki.word_cards.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.word_cards._fetch_image")
    @patch("bcs_anki.word_cards.generate_definition_and_examples")
    def test_csv_has_correct_note_types(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path)

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
    @patch("bcs_anki.word_cards.resolve_lemma", side_effect=lambda cfg, w: w)
    @patch("bcs_anki.word_cards._fetch_image")
    @patch("bcs_anki.word_cards.generate_definition_and_examples")
    def test_returns_false_on_error(self, mock_gen, mock_img, mock_lemma, mock_cfg, tmp_path):
        ctx = _make_ctx(mock_cfg, tmp_path)

        mock_gen.side_effect = RuntimeError("API error")
        mock_img.return_value = [("img.png", tmp_path / "img.png")]

        result = process_word("fail_word", ctx)

        assert result is False
        assert ctx.failed_csv.exists()
        rows = ctx.failed_csv.read_text(encoding="utf-8").strip().splitlines()
        assert any(row.startswith("fail_word\t") and "API error" in row for row in rows)
