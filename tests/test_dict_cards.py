"""Tests for the generate-dict pipeline (refined CSV → Anki Basic+reversed cards)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from bcs_anki.dict_cards import run_generate_dict
from bcs_anki.errors import ImageRejectedError, NoStockResultsError


def _write_csv(path: Path, subject: str, rows: list[tuple[str, str]]) -> None:
    lines = [f"# Subject: {subject}", "english,serbian"]
    lines.extend(f"{eng},{sr}" for eng, sr in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture()
def cards_cfg(mock_cfg, tmp_path):
    """mock_cfg with output/temp dirs scoped to tmp_path and a known tags string."""
    mock_cfg.output_folder = tmp_path / "output"
    mock_cfg.temp_image_folder = tmp_path / "temp_images"
    mock_cfg.tags = "bcs"
    mock_cfg.max_workers = 1  # serial, deterministic ordering for assertions
    return mock_cfg


def _stock_writes_path(dest: Path):
    """Side-effect: simulate fetch_stock_image by writing to dest and returning [dest]."""
    def _impl(cfg, word_en, dst, count=3):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"STOCK_BYTES")
        return [dst]
    return _impl


class TestStockHappyPath:
    def test_writes_basic_reversed_row_with_subject_tag(self, cards_cfg, tmp_path):
        src = tmp_path / "geo.csv"
        _write_csv(src, "Geografija I", [("river", "rijeka")])

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image") as mock_ai:
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (1, 0)
        assert mock_stock.call_count == 1
        assert mock_ai.call_count == 0

        out = cards_cfg.output_folder / "cards" / "geografija-i.csv"
        assert out.exists()
        lines = _read_csv_lines(out)
        # Anki header lines + one data row
        assert any(line.startswith("Basic (and reversed card)\t") for line in lines)
        data = [line for line in lines if line.startswith("Basic (and reversed card)\t")][0]
        parts = data.split("\t")
        assert '<img src="' in parts[1]
        assert parts[2] == "rijeka"
        assert parts[3] == "bcs geografija-i"


class TestStockFailsFallsBackToAi:
    def test_ai_called_with_formatted_prompt(self, cards_cfg, tmp_path):
        src = tmp_path / "ast.csv"
        _write_csv(src, "Astronomija", [("star", "zvijezda")])

        def _ai_writes(cfg, prompt, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"AI_BYTES")

        with patch(
            "bcs_anki.dict_cards.fetch_stock_image",
            side_effect=NoStockResultsError("no results"),
        ) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image", side_effect=_ai_writes) as mock_ai:
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (1, 0)
        assert mock_stock.call_count == 1
        assert mock_ai.call_count == 1
        # The AI prompt was formatted with both the English term and the subject.
        prompt_arg = mock_ai.call_args.args[1]
        assert "star" in prompt_arg
        assert "Astronomija" in prompt_arg

    def test_http_error_also_triggers_ai_fallback(self, cards_cfg, tmp_path):
        """A 403/429/etc. from the stock provider must fall back to AI, not fail the row."""
        src = tmp_path / "z.csv"
        _write_csv(src, "Zodijak", [("Pisces", "ribe")])

        def _ai_writes(cfg, prompt, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"AI_BYTES")

        http_err = requests.HTTPError("403 Client Error: Forbidden")

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=http_err) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image", side_effect=_ai_writes) as mock_ai:
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (1, 0)
        assert mock_stock.call_count == 1
        assert mock_ai.call_count == 1

    def test_multi_word_entry_skips_stock_and_uses_ai(self, cards_cfg, tmp_path):
        """Compound English terms go straight to AI without burning a stock API call."""
        src = tmp_path / "z.csv"
        _write_csv(src, "Zodijak", [("Pisces (the Fish)", "ribe")])

        def _ai_writes(cfg, prompt, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"AI_BYTES")

        with patch("bcs_anki.dict_cards.fetch_stock_image") as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image", side_effect=_ai_writes) as mock_ai:
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (1, 0)
        assert mock_stock.call_count == 0          # stock skipped entirely
        assert mock_ai.call_count == 1
        prompt_arg = mock_ai.call_args.args[1]
        assert "Pisces (the Fish)" in prompt_arg


class TestLeadingTheStripped:
    def test_leading_the_dropped_for_stock_query(self, cards_cfg, tmp_path):
        """'the moon' is treated as 'moon' — single word, hits stock with stripped query."""
        src = tmp_path / "a.csv"
        _write_csv(src, "Astronomija", [("the moon", "mjesec")])

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image") as mock_ai:
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (1, 0)
        assert mock_stock.call_count == 1
        assert mock_stock.call_args.args[1] == "moon"   # stripped query
        assert mock_ai.call_count == 0

    def test_capitalized_The_also_dropped(self, cards_cfg, tmp_path):
        src = tmp_path / "a.csv"
        _write_csv(src, "Geografija", [("The river", "rijeka")])

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image"):
            run_generate_dict(cards_cfg, src)

        assert mock_stock.call_args.args[1] == "river"

    def test_inner_the_preserved(self, cards_cfg, tmp_path):
        """Only LEADING 'the' is stripped — inner 'the' (e.g. parenthesized) stays."""
        src = tmp_path / "z.csv"
        _write_csv(src, "Zodijak", [("Pisces (the Fish)", "ribe")])

        with patch("bcs_anki.dict_cards.fetch_stock_image") as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image") as mock_ai:
            run_generate_dict(cards_cfg, src)

        # Multi-word, so AI is called; the prompt still contains the inner "the".
        assert mock_stock.call_count == 0
        prompt_arg = mock_ai.call_args.args[1]
        assert "(the Fish)" in prompt_arg

    def test_word_that_starts_with_the_not_stripped(self, cards_cfg, tmp_path):
        """'they' must not be split into 'y' — only the exact word 'the' is dropped."""
        src = tmp_path / "x.csv"
        _write_csv(src, "Subj", [("they", "oni")])

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image"):
            run_generate_dict(cards_cfg, src)

        assert mock_stock.call_args.args[1] == "they"


class TestAiAlsoFailsRecordsFailedRow:
    def test_no_csv_row_failed_tsv_populated(self, cards_cfg, tmp_path):
        src = tmp_path / "x.csv"
        _write_csv(src, "Subj", [("rotary motions", "rotacioni pokreti")])

        with patch(
            "bcs_anki.dict_cards.fetch_stock_image",
            side_effect=NoStockResultsError("no results"),
        ), \
             patch(
            "bcs_anki.dict_cards.generate_ai_image",
            side_effect=ImageRejectedError("safety"),
        ):
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (0, 1)
        out = cards_cfg.output_folder / "cards" / "subj.csv"
        # Only the Anki header survives — no data row.
        assert not any(
            line.startswith("Basic (and reversed card)\t") for line in _read_csv_lines(out)
        )
        failed_tsv = cards_cfg.output_folder / "cards" / "subj_failed.tsv"
        text = failed_tsv.read_text(encoding="utf-8")
        assert "rotary motions" in text
        assert "ImageRejectedError" in text


class TestResume:
    def test_completed_rows_are_not_refetched(self, cards_cfg, tmp_path):
        src = tmp_path / "geo.csv"
        _write_csv(src, "Geografija I", [("river", "rijeka"), ("rock", "stijena")])

        # Pre-populate progress as if "river" already done.
        out_dir = cards_cfg.output_folder / "cards"
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_file = out_dir / ".progress_geo.json"
        progress_file.write_text(
            json.dumps({
                "input_file": str(src),
                "total_words": 2,
                "completed_words": ["river"],
                "failed_words": [],
                "last_updated": "",
            }),
            encoding="utf-8",
        )

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)) as mock_stock, \
             patch("bcs_anki.dict_cards.generate_ai_image"):
            completed, failed = run_generate_dict(
                cards_cfg, src, resume=True, append=True,
            )

        # Only "rock" should have been processed.
        assert mock_stock.call_count == 1
        assert mock_stock.call_args.args[1] == "rock"
        assert completed >= 1
        assert failed == 0


class TestSubjectTag:
    def test_complex_subject_normalized_in_tag(self, cards_cfg, tmp_path):
        src = tmp_path / "g2.csv"
        _write_csv(src, "Geografija II — special", [("rock", "stijena")])

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)), \
             patch("bcs_anki.dict_cards.generate_ai_image"):
            run_generate_dict(cards_cfg, src)

        out = cards_cfg.output_folder / "cards" / "geografija-ii-special.csv"
        assert out.exists()
        data = [line for line in _read_csv_lines(out) if line.startswith("Basic (and reversed card)\t")][0]
        assert data.split("\t")[3] == "bcs geografija-ii-special"


class TestRoundTripTwoRows:
    def test_both_rows_present_in_order(self, cards_cfg, tmp_path):
        src = tmp_path / "two.csv"
        _write_csv(src, "Animals", [("cat", "mačka"), ("dog", "pas")])

        with patch("bcs_anki.dict_cards.fetch_stock_image", side_effect=_stock_writes_path(None)), \
             patch("bcs_anki.dict_cards.generate_ai_image"):
            completed, failed = run_generate_dict(cards_cfg, src)

        assert (completed, failed) == (2, 0)
        out = cards_cfg.output_folder / "cards" / "animals.csv"
        data_rows = [line for line in _read_csv_lines(out) if line.startswith("Basic (and reversed card)\t")]
        assert len(data_rows) == 2
        # Field 2 (Serbian) preserves order
        serbians = [row.split("\t")[2] for row in data_rows]
        assert serbians == ["mačka", "pas"]
