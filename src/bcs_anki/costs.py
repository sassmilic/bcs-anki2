from __future__ import annotations

from dataclasses import dataclass
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

    def add_openai(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.openai.input_tokens += _safe_non_negative_int(input_tokens)
            self.openai.output_tokens += _safe_non_negative_int(output_tokens)

    def add_gemini(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.gemini.input_tokens += _safe_non_negative_int(input_tokens)
            self.gemini.output_tokens += _safe_non_negative_int(output_tokens)

    def _estimate(self, model: str, usage: ProviderUsage, pricing: dict[str, ProviderCost]) -> float | None:
        price = pricing.get(model)
        if not price:
            return None
        return (
            usage.input_tokens / 1_000_000 * price.input_per_million
            + usage.output_tokens / 1_000_000 * price.output_per_million
        )

    def summary(self, openai_model: str, gemini_model: str) -> str:
        openai_cost = self._estimate(openai_model, self.openai, _OPENAI_PRICING)
        gemini_cost = self._estimate(gemini_model, self.gemini, _GEMINI_PRICING)

        openai_part = (
            f"ChatGPT: {self.openai.input_tokens} in / {self.openai.output_tokens} out tokens"
            + (f", est. ${openai_cost:.4f}" if openai_cost is not None else ", est. cost unknown")
        )
        gemini_part = (
            f"Gemini: {self.gemini.input_tokens} in / {self.gemini.output_tokens} out tokens"
            + (f", est. ${gemini_cost:.4f}" if gemini_cost is not None else ", est. cost unknown")
        )

        total = 0.0
        has_total = False
        if openai_cost is not None:
            total += openai_cost
            has_total = True
        if gemini_cost is not None:
            total += gemini_cost
            has_total = True

        total_part = f"Total est. cost: ${total:.4f}" if has_total else "Total est. cost: unknown"
        return f"{openai_part}; {gemini_part}; {total_part}."


COST_TRACKER = CostTracker()
