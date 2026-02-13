from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Tuple

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class GeneratedText:
    definition_html: str
    examples_html: str


def _openai_chat_completion(
    cfg: AppConfig,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    delay = cfg.rate_limit_delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI chat completion failed (attempt %s): %s", attempt, exc)
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2


def generate_definition_and_examples(cfg: AppConfig, word: str, canonical_info: str) -> GeneratedText:
    system_prompt = (
        "Ti si iskusni leksikograf za bosanski/hrvatski/srpski jezik (ijekavski). "
        "Odgovaraj isključivo na BHS jeziku, ijekavski, bez objašnjenja na engleskom."
    )
    user_prompt = f"""
Zadatak:
- Za zadanu riječ ili izraz: "{word}"
- Kanonski oblik i gramatičke informacije: {canonical_info}

0) Pažljivo prepoznaj vjerovatni KANONSKI OBLIK (lemu) i osnovne gramatičke informacije (npr. rod, broj, padežna promjena, vid glagola...). Ako ulazna riječ izgleda POGREŠNO NAPISANO ili bez dijakritičkih znakova (č, ć, ž, š, đ), pretpostavi najvjerovatniji ispravan oblik i to jasno napomeni.
1) Napiši JEDNU cjelovitu definiciju koja obuhvata sve glavne relevantne smislove.
2) Zatim napiši TAČNO TRI primjerne rečenice, numerisane 1–3, u kojima se riječ pojavljuje u različitim oblicima (padeži, vremena, vidovi...), ali:
   - Svako pojavljivanje ciljne riječi/izraza treba biti u formatu {{c1::OBLIK}}.

Format izlaza:
DEFINICIJA:
{{c1::KANONSKI_OBLIK}} (GRAMMATIKA)<br>
- ako je potrebno, kratka napomena o eventualnoj grešci u pravopisu ili nedostajućim dijakritičkim znakovima ulazne riječi (npr. "Ulazna riječ je vjerovatno pogrešno napisana; kanonski oblik je ...").<br>
– definicija u jednom ili dva kraća, jasna iskaza.

PRIMJERI:
1. Rečenica s {{c1::oblik1}}.<br>2. Rečenica s {{c1::oblik2}}.<br>3. Rečenica s {{c1::oblik3}}.
"""
    text = _openai_chat_completion(cfg, system_prompt, user_prompt)

    # Extremely lightweight parsing: split on a blank line or "PRIMJERI:"
    parts = text.split("PRIMJERI:")
    if len(parts) == 2:
        definition_part, examples_part = parts
    else:
        # Fallback: first line vs rest
        lines = text.splitlines()
        definition_part = "\n".join(lines[:1])
        examples_part = "\n".join(lines[1:])

    return GeneratedText(definition_html=definition_part.strip(), examples_html=examples_part.strip())


def generate_image_prompt(cfg: AppConfig, word: str, word_type_hint: str) -> str:
    system_prompt = (
        "Ti si kreativni dizajner koji osmišljava upute za generiranje slika (prompte) za model poput DALL-E. "
        "Ne koristi tekst u samoj slici. Piši na engleskom jeziku."
    )
    user_prompt = f"""
Zadatak:
- Riječ ili izraz (na BHS, ijekavski): "{word}"
- Kratki nagovještaj vrste riječi ili pojma: {word_type_hint}

Napiši detaljan prompt na engleskom jeziku za generiranje slike koja:
- jasno i nedvosmisleno vizualizira pojam,
- stil prilagodi vrsti pojma:
  - glagoli/radnje: strip, sekvenca ili scena s jasnom radnjom,
  - apstraktne imenice: konceptualna ili nadrealna ilustracija,
  - emocije: izražajni, slikarski stil,
  - idiomi/fraze: kreativna vizuelna metafora figurativnog značenja.
Ne pominji riječi "text", "caption" niti upotrebljavaj natpise u slici.
"""
    return _openai_chat_completion(cfg, system_prompt, user_prompt)

