"""Tier 1: failed.tsv reason-summarization helper.

Without this, a single Gemini 429 dumps ~2KB of nested JSON into the reason
column, making the file impossible to scan.
"""
from __future__ import annotations

from bcs_anki.pipeline import summarize_exception as _summarize_exception


class _GeminiClientError(Exception):
    """Stand-in for google.genai.errors.ClientError so we don't need the SDK in tests."""


def test_strips_gemini_json_payload():
    blob = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
        "current quota...', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
        "'type.googleapis.com/google.rpc.QuotaFailure'}]}}"
    )
    exc = _GeminiClientError(blob)
    summary = _summarize_exception(exc)
    assert summary == "_GeminiClientError: 429 RESOURCE_EXHAUSTED"
    assert "{" not in summary
    assert len(summary) < 100


def test_short_message_passes_through():
    exc = RuntimeError("OPENAI_API_KEY is not configured")
    summary = _summarize_exception(exc)
    assert summary == "RuntimeError: OPENAI_API_KEY is not configured"


def test_collapses_whitespace_and_newlines():
    exc = ValueError("line one\n  line two\t\ttabbed")
    summary = _summarize_exception(exc)
    assert summary == "ValueError: line one line two tabbed"


def test_truncates_long_message_with_ellipsis():
    long_msg = "x" * 500
    exc = RuntimeError(long_msg)
    summary = _summarize_exception(exc, max_len=80)
    assert len(summary) == 80
    assert summary.endswith("…")


def test_empty_message_falls_back_to_class_name():
    exc = RuntimeError()
    assert _summarize_exception(exc) == "RuntimeError"


from bcs_anki.errors import MissingApiKeyError

def test_typed_error_includes_class_name():
    exc = MissingApiKeyError("OPENAI_API_KEY is not configured")
    summary = _summarize_exception(exc)
    assert summary == "MissingApiKeyError: OPENAI_API_KEY is not configured"
