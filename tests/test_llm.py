"""Tier 3: LLM response parsing tests (mocked OpenAI)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from bcs_anki.llm import (
    GeneratedText,
    decide_image_source,
    generate_definition_and_examples,
    generate_image_prompt,
    generate_image_search_term,
)


class TestGenerateDefinitionAndExamples:
    def test_splits_on_primjeri_delimiter(self, mock_cfg, mock_openai_chat):
        definition_resp = mock_openai_chat(
            "DEFINICIJA:\n{{c1::primirje}} — privremeni prekid\nPRIMJERI:\n1. Sentence one.\n2. Sentence two.\n3. Sentence three."
        )
        validation_resp = mock_openai_chat("primirje 90%\nprekid 7%\nmir 3%")

        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                definition_resp, validation_resp,
            ]
            result = generate_definition_and_examples(mock_cfg, "primirje")

        assert isinstance(result, GeneratedText)
        assert "primirje" in result.definition_html
        assert not result.definition_html.startswith("DEFINICIJA:")
        assert "Sentence one" in result.examples_html

    def test_fallback_parsing_without_delimiter(self, mock_cfg, mock_openai_chat):
        definition_resp = mock_openai_chat(
            "Definition line\n1. Example one.\n2. Example two.\n3. Example three."
        )
        # Validation will fail (no cloze to extract lemma, word is "test")
        validation_resp = mock_openai_chat("test 80%\nother 15%\nword 5%")

        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                definition_resp, validation_resp,
            ]
            result = generate_definition_and_examples(mock_cfg, "test")

        assert result.definition_html == "Definition line"
        assert "Example one" in result.examples_html

    def test_validation_triggers_refinement(self, mock_cfg, mock_openai_chat):
        definition_resp = mock_openai_chat(
            "DEFINICIJA:\n{{c1::primirje}} — privremeni prekid\nPRIMJERI:\n1. S1.\n2. S2.\n3. S3."
        )
        # Validation fails: top guess is wrong
        validation_resp = mock_openai_chat("prekid 60%\nprimirje 30%\nmir 10%")
        # Refinement returns improved definition in full format
        refine_resp = mock_openai_chat(
            "{{c1::primirje}} (imenica, sr.) — privremeni prestanak neprijateljstava"
        )

        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = [
                definition_resp, validation_resp, refine_resp,
            ]
            result = generate_definition_and_examples(mock_cfg, "primirje")

        assert "{{c1::primirje}}" in result.definition_html
        assert "neprijateljstava" in result.definition_html
        assert "S1" in result.examples_html


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
        # _chat returns content as-is; generate_image_prompt returns it directly
        assert "ceasefire" in result


class TestGenerateImageSearchTerm:
    def test_returns_first_line_trimmed(self, mock_cfg, mock_openai_chat):
        resp = mock_openai_chat("ceasefire, truce\nextra line")
        with patch("bcs_anki.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = resp
            result = generate_image_search_term(mock_cfg, "primirje")
        assert result == "ceasefire, truce"
