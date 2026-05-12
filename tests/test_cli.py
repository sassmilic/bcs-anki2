from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bcs_anki import cli
from bcs_anki.dictionary_csv import DictEntry, DictPage
from bcs_anki.workflows import dictionary as dictionary_workflow


def test_top_level_help_uses_source_based_commands() -> None:
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "words" in result.output
    assert "dictionary" in result.output
    assert "generate-dict" not in result.output
    assert "ocr-dict" not in result.output


def test_dictionary_help_exposes_full_and_staged_workflows() -> None:
    result = CliRunner().invoke(cli.main, ["dictionary", "--help"])

    assert result.exit_code == 0
    assert "pages" in result.output
    assert "ocr" in result.output
    assert "refine" in result.output
    assert "csv" in result.output


def test_dictionary_pages_runs_full_pipeline(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"not a real image; OCR is mocked")

    calls: dict[str, object] = {}

    def fake_extract_dict_pages(_cfg, image_paths):
        calls["image_paths"] = image_paths
        return DictPage(
            subject="Astronomija",
            entries=[DictEntry(number="1", english="moon", serbian="mjesec")],
        )

    def fake_refine_csv(_cfg, input_path, output_path):
        calls["refine"] = (input_path, output_path)
        return 1

    def fake_run_generate_dict(_cfg, csv_path, output_csv, *, append=False):
        calls["cards"] = (csv_path, output_csv, append)
        return 1, 0

    monkeypatch.setattr(dictionary_workflow, "extract_dict_pages", fake_extract_dict_pages)
    monkeypatch.setattr(dictionary_workflow, "refine_csv", fake_refine_csv)
    monkeypatch.setattr(dictionary_workflow, "run_generate_dict", fake_run_generate_dict)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli.main,
            ["dictionary", "pages", "--skip-health-check", str(image_path)],
        )

        assert result.exit_code == 0, result.output
        assert calls["image_paths"] == [image_path]
        assert calls["refine"] == (
            Path("output/dict/astronomija.csv"),
            Path("output/dict/refined/astronomija.csv"),
        )
        assert calls["cards"] == (
            Path("output/dict/refined/astronomija.csv"),
            None,
            False,
        )
        assert Path("output/dict/astronomija.csv").exists()
