from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProviderCost:
    input_per_million: float
    output_per_million: float


_OPENAI_PRICING: dict[str, ProviderCost] = {
    "gpt-5.4-mini": ProviderCost(input_per_million=0.75, output_per_million=4.50),
}

_GEMINI_PRICING: dict[str, ProviderCost] = {
    "gemini-2.5-pro": ProviderCost(input_per_million=1.25, output_per_million=10.00),
}

# Per-image pricing in USD, keyed by (model, size, quality). Used only for
# legacy fixed-price models like dall-e-3.
_OPENAI_IMAGE_PRICING: dict[tuple[str, str, str], float] = {
    ("dall-e-3", "1024x1024", "standard"): 0.040,
    ("dall-e-3", "1024x1024", "hd"): 0.080,
    ("dall-e-3", "1024x1792", "standard"): 0.080,
    ("dall-e-3", "1024x1792", "hd"): 0.120,
    ("dall-e-3", "1792x1024", "standard"): 0.080,
    ("dall-e-3", "1792x1024", "hd"): 0.120,
}


@dataclass
class ImageTokenPricing:
    """Per-1M-token rates for token-priced image models (Standard mode)."""
    text_input: float
    image_input: float
    cached_image_input: float
    image_output: float


# https://developers.openai.com/api/docs/pricing — Standard rates per 1M tokens.
_OPENAI_IMAGE_TOKEN_PRICING: dict[str, ImageTokenPricing] = {
    "gpt-image-2": ImageTokenPricing(
        text_input=5.00, image_input=8.00, cached_image_input=2.00, image_output=30.00,
    ),
    "gpt-image-1.5": ImageTokenPricing(
        text_input=5.00, image_input=8.00, cached_image_input=2.00, image_output=32.00,
    ),
    "gpt-image-1-mini": ImageTokenPricing(
        text_input=2.00, image_input=2.50, cached_image_input=0.25, image_output=8.00,
    ),
}


@dataclass
class ImageTokenUsage:
    text_input_tokens: int = 0
    image_input_tokens: int = 0
    cached_image_input_tokens: int = 0
    image_output_tokens: int = 0


def _safe_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    return 0


class CostTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.openai = ProviderUsage()
        self.gemini = ProviderUsage()
        # Legacy per-image counts (dall-e-3 etc.).
        self.images: dict[tuple[str, str, str], int] = {}
        # Per-model token totals + image counts for token-priced models
        # (gpt-image-2, gpt-image-1.5, gpt-image-1-mini, etc.).
        self.image_tokens: dict[str, ImageTokenUsage] = {}
        self.image_token_counts: dict[str, int] = {}

    def add_openai(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.openai.input_tokens += _safe_non_negative_int(input_tokens)
            self.openai.output_tokens += _safe_non_negative_int(output_tokens)

    def add_gemini(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.gemini.input_tokens += _safe_non_negative_int(input_tokens)
            self.gemini.output_tokens += _safe_non_negative_int(output_tokens)

    def add_image(self, model: str, size: str, quality: str) -> None:
        with self._lock:
            key = (model, size, quality)
            self.images[key] = self.images.get(key, 0) + 1

    def add_image_tokens(
        self,
        model: str,
        text_input_tokens: int = 0,
        image_input_tokens: int = 0,
        cached_image_input_tokens: int = 0,
        image_output_tokens: int = 0,
    ) -> None:
        with self._lock:
            usage = self.image_tokens.setdefault(model, ImageTokenUsage())
            usage.text_input_tokens += _safe_non_negative_int(text_input_tokens)
            usage.image_input_tokens += _safe_non_negative_int(image_input_tokens)
            usage.cached_image_input_tokens += _safe_non_negative_int(cached_image_input_tokens)
            usage.image_output_tokens += _safe_non_negative_int(image_output_tokens)
            self.image_token_counts[model] = self.image_token_counts.get(model, 0) + 1

    def _estimate(self, model: str, usage: ProviderUsage, pricing: dict[str, ProviderCost]) -> float | None:
        price = pricing.get(model)
        if not price:
            return None
        return (
            usage.input_tokens / 1_000_000 * price.input_per_million
            + usage.output_tokens / 1_000_000 * price.output_per_million
        )

    def _image_summary(self) -> tuple[str, float | None]:
        """Return (human-readable line, total cost or None if any bucket is unknown)."""
        if not self.images and not self.image_tokens:
            return "Images: 0 generated", 0.0

        bucket_strs: list[str] = []
        total_count = 0
        total_cost = 0.0
        any_unknown = False

        # Legacy per-image-priced models (dall-e-3 etc.)
        for (model, size, quality), count in self.images.items():
            total_count += count
            unit_price = _OPENAI_IMAGE_PRICING.get((model, size, quality))
            if unit_price is None:
                any_unknown = True
                bucket_strs.append(f"{count}× {model} {size} {quality} (price unknown)")
            else:
                bucket_cost = count * unit_price
                total_cost += bucket_cost
                bucket_strs.append(
                    f"{count}× {model} {size} {quality} @ ${unit_price:.3f} = ${bucket_cost:.2f}"
                )

        # Token-priced models (gpt-image-*)
        for model, usage in self.image_tokens.items():
            count = self.image_token_counts.get(model, 0)
            total_count += count
            price = _OPENAI_IMAGE_TOKEN_PRICING.get(model)
            in_t = usage.text_input_tokens + usage.image_input_tokens
            cached_t = usage.cached_image_input_tokens
            out_t = usage.image_output_tokens
            tokens_summary = (
                f"{usage.text_input_tokens} text-in / "
                f"{usage.image_input_tokens} img-in / "
                f"{cached_t} cached-in / "
                f"{out_t} img-out"
            )
            if price is None:
                any_unknown = True
                bucket_strs.append(
                    f"{count}× {model} ({tokens_summary} tokens, price unknown)"
                )
            else:
                bucket_cost = (
                    usage.text_input_tokens / 1_000_000 * price.text_input
                    + usage.image_input_tokens / 1_000_000 * price.image_input
                    + usage.cached_image_input_tokens / 1_000_000 * price.cached_image_input
                    + usage.image_output_tokens / 1_000_000 * price.image_output
                )
                total_cost += bucket_cost
                bucket_strs.append(
                    f"{count}× {model} ({tokens_summary} tokens) = ${bucket_cost:.4f}"
                )

        line = f"Images: {total_count} generated [" + "; ".join(bucket_strs) + "]"
        return line, (None if any_unknown else total_cost)

    def summary(self, openai_model: str, gemini_model: str) -> str:
        openai_cost = self._estimate(openai_model, self.openai, _OPENAI_PRICING)
        gemini_cost = self._estimate(gemini_model, self.gemini, _GEMINI_PRICING)
        image_line, image_cost = self._image_summary()

        openai_part = (
            f"ChatGPT: {self.openai.input_tokens} in / {self.openai.output_tokens} out tokens"
            + (f", est. ${openai_cost:.4f}" if openai_cost is not None else ", est. cost unknown")
        )
        gemini_part = (
            f"Gemini: {self.gemini.input_tokens} in / {self.gemini.output_tokens} out tokens"
            + (f", est. ${gemini_cost:.4f}" if gemini_cost is not None else ", est. cost unknown")
        )

        components = [openai_cost, gemini_cost, image_cost]
        if any(c is None for c in components):
            total_part = "Total est. cost: partial (some prices unknown)"
        else:
            total_part = f"Total est. cost: ${sum(components):.4f}"

        return f"{openai_part}; {gemini_part}; {image_line}; {total_part}."


COST_TRACKER = CostTracker()
