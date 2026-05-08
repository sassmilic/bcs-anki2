"""Tests for the CostTracker summary, including image-generation tracking."""
from __future__ import annotations

from bcs_anki.costs import CostTracker


def test_tokens_only_unchanged():
    t = CostTracker()
    t.add_openai(1000, 500)
    t.add_gemini(2000, 100)
    s = t.summary("gpt-5.4-mini", "gemini-2.5-pro")
    assert "ChatGPT: 1000 in / 500 out tokens, est. $" in s
    assert "Gemini: 2000 in / 100 out tokens, est. $" in s
    assert "Images: 0 generated" in s
    assert "Total est. cost: $" in s


def test_image_count_and_known_pricing():
    t = CostTracker()
    t.add_image("dall-e-3", "1024x1024", "standard")
    t.add_image("dall-e-3", "1024x1024", "standard")
    s = t.summary("gpt-5.4-mini", "gemini-2.5-pro")
    # 2 × $0.040 = $0.08
    assert "Images: 2 generated" in s
    assert "$0.040" in s
    assert "$0.08" in s
    # Total includes image cost (no unknown pieces).
    assert "Total est. cost: $0.0800" in s


def test_token_priced_model_uses_per_million_rates():
    t = CostTracker()
    # Simulate 69 generations at 50 input tokens (text) + 4096 output tokens each.
    for _ in range(69):
        t.add_image_tokens(
            "gpt-image-2",
            text_input_tokens=50,
            image_output_tokens=4096,
        )
    s = t.summary("gpt-5.4-mini", "gemini-2.5-pro")
    assert "69× gpt-image-2" in s
    # Text input: 69*50 = 3450 tokens × $5/1M = $0.01725
    # Image output: 69*4096 = 282624 tokens × $30/1M = $8.47872
    # Total ≈ $8.49597
    assert "Total est. cost: $8." in s
    # Per-image lookup not used.
    assert "(price unknown)" not in s


def test_unknown_token_model_marks_total_partial():
    t = CostTracker()
    t.add_image_tokens("brand-new-model", text_input_tokens=10, image_output_tokens=1000)
    s = t.summary("gpt-5.4-mini", "gemini-2.5-pro")
    assert "1× brand-new-model" in s
    assert "price unknown" in s
    assert "Total est. cost: partial" in s


def test_mixed_legacy_and_token_models():
    t = CostTracker()
    t.add_image("dall-e-3", "1024x1024", "hd")  # legacy: known
    t.add_image_tokens("gpt-image-2", text_input_tokens=100, image_output_tokens=2048)
    s = t.summary("gpt-5.4-mini", "gemini-2.5-pro")
    assert "1× dall-e-3" in s
    assert "1× gpt-image-2" in s
    # Both priced — total should be a clean dollar figure, not partial.
    assert "Total est. cost: $" in s
    assert "partial" not in s
