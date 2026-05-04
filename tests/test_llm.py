"""Tier 3: LLM response parsing tests (mocked OpenAI)."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

from bcs_anki.llm import (
    GeneratedText,
    decide_image_source,
    generate_definition_and_examples,
    generate_image_prompt,
    generate_image_search_term,
    resolve_lemma,
)


class TestResolveLemma:
    def test_returns_bare_lemma(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("vidjeti")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            assert resolve_lemma(mock_cfg, "vidim") == "vidjeti"

    def test_strips_quotes_and_whitespace(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat('  "primirje"  \nextra line ignored')
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            assert resolve_lemma(mock_cfg, "primirja") == "primirje"


class TestGenerateDefinitionAndExamples:
    @patch("bcs_anki.llm.review_examples", side_effect=lambda cfg, w, html: html)
    @patch("bcs_anki.llm.review_definition", side_effect=lambda cfg, w, html: html)
    def test_separate_definition_and_examples(self, _mock_def, _mock_ex, mock_cfg, mock_openai_chat):
        definition_resp = mock_openai_chat(
            "{{c1::primirje}} — privremeni prekid"
        )
        examples_resp = mock_openai_chat(
            "Sentence one {{c1::primirje}}.<br>Sentence two {{c1::primirja}}.<br>Sentence three {{c1::primirju}}."
        )

        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                definition_resp, examples_resp,
            ]
            result = generate_definition_and_examples(mock_cfg, "primirje")

        assert isinstance(result, GeneratedText)
        assert "primirje" in result.definition_html
        assert not result.definition_html.startswith("DEFINICIJA:")
        assert "Sentence one" in result.examples_html

    @patch("bcs_anki.llm.review_examples", side_effect=lambda cfg, w, html: html.replace("primirja", "primirje"))
    @patch("bcs_anki.llm.review_definition", side_effect=lambda cfg, w, html: html)
    def test_gemini_correction_applied(self, _mock_def, _mock_ex, mock_cfg, mock_openai_chat):
        """Gemini's correction should replace OpenAI's output when reviewer changed it."""
        definition_resp = mock_openai_chat("{{c1::primirje}} — definicija")
        examples_resp = mock_openai_chat("<ul><li>Greška u {{c1::primirja}}.</li></ul>")

        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                definition_resp, examples_resp,
            ]
            result = generate_definition_and_examples(mock_cfg, "primirje")

        assert "primirja" not in result.examples_html
        assert "primirje" in result.examples_html

    def test_skips_gemini_when_no_api_key(self, mock_cfg, mock_openai_chat):
        """Without GEMINI_API_KEY, OpenAI output is returned untouched, no review call."""
        no_gemini = replace(mock_cfg, gemini_api_key=None)
        definition_resp = mock_openai_chat("{{c1::primirje}} — direct")
        examples_resp = mock_openai_chat("<ul><li>Direct {{c1::primirje}}.</li></ul>")

        with patch("bcs_anki.llm.review_definition") as mock_def, \
             patch("bcs_anki.llm.review_examples") as mock_ex, \
             patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                definition_resp, examples_resp,
            ]
            result = generate_definition_and_examples(no_gemini, "primirje")

        mock_def.assert_not_called()
        mock_ex.assert_not_called()
        assert "direct" in result.definition_html
        assert "Direct" in result.examples_html


class TestDecideImageSource:
    def test_returns_stock(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("stock")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            result = decide_image_source(mock_cfg, "jabuka")
        assert result == "stock"

    def test_returns_ai(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("ai")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            result = decide_image_source(mock_cfg, "nada")
        assert result == "ai"

    def test_defaults_to_ai_on_unexpected(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("maybe stock?")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            result = decide_image_source(mock_cfg, "test")
        assert result == "ai"


class TestGenerateImagePrompt:
    def test_returns_trimmed_string(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("  A peaceful ceasefire scene  ")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            result = generate_image_prompt(mock_cfg, "primirje")
        assert result == "A peaceful ceasefire scene"


class TestGenerateImageSearchTerm:
    def test_returns_first_line_trimmed(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("ceasefire, truce\nextra line")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            result = generate_image_search_term(mock_cfg, "primirje")
        assert result == "ceasefire, truce"
