"""
All LLM prompts in one place for easy tweaking.

Each pair is (system_prompt, user_prompt_template).
User prompt templates use {word} for substitution.
"""

# --- Definition & examples ---

DEFINITION_SYSTEM = (
    "Ti si iskusni leksikograf za bosanski/hrvatski/srpski jezik (ijekavski). "
    "Odgovaraj isključivo na BHS jeziku, ijekavski, bez objašnjenja na engleskom."
)

DEFINITION_USER = """\
Zadatak:
- Za zadanu riječ ili izraz: "{word}"

0) Pažljivo prepoznaj vjerovatni KANONSKI OBLIK (lemu) i osnovne gramatičke informacije (npr. rod, broj, padežna promjena, vid glagola...). Ako ulazna riječ izgleda POGREŠNO NAPISANO ili bez dijakritičkih znakova (č, ć, ž, š, đ), pretpostavi najvjerovatniji ispravan oblik i to jasno napomeni.
1) Napiši JEDNU cjelovitu definiciju koja obuhvata sve glavne relevantne smislove.
2) Zatim napiši TAČNO TRI primjerne rečenice, numerisane 1–3, u kojima se riječ pojavljuje u različitim oblicima (padeži, vremena, vidovi...), ali:
   - Svako pojavljivanje ciljne riječi/izraza treba biti u formatu {{{{c1::OBLIK}}}}.

Format izlaza:
DEFINICIJA:
{{{{c1::KANONSKI_OBLIK}}}} (GRAMMATIKA)<br>
- ako je potrebno, kratka napomena o eventualnoj grešci u pravopisu ili nedostajućim dijakritičkim znakovima ulazne riječi (npr. "Ulazna riječ je vjerovatno pogrešno napisana; kanonski oblik je ...").<br>
– definicija u jednom ili dva kraća, jasna iskaza.

PRIMJERI:
1. Rečenica s {{{{c1::oblik1}}}}.<br>2. Rečenica s {{{{c1::oblik2}}}}.<br>3. Rečenica s {{{{c1::oblik3}}}}."""


# --- Image prompt generation (for AI-generated images) ---

IMAGE_PROMPT_SYSTEM = (
    "Ti si kreativni dizajner koji osmišljava upute za generiranje slika (prompte) za model poput DALL-E. "
    "Ne koristi tekst u samoj slici. Piši na engleskom jeziku."
)

IMAGE_PROMPT_USER = """\
Zadatak:
- Riječ ili izraz (na BHS, ijekavski): "{word}"

Napiši detaljan prompt na engleskom jeziku za generiranje slike koja:
- jasno i nedvosmisleno vizualizira pojam,
- stil prilagodi vrsti pojma:
  - glagoli/radnje: strip, sekvenca ili scena s jasnom radnjom,
  - apstraktne imenice: konceptualna ili nadrealna ilustracija,
  - emocije: izražajni, slikarski stil,
  - idiomi/fraze: kreativna vizuelna metafora figurativnog značenja.
Ne pominji riječi "text", "caption" niti upotrebljavaj natpise u slici."""


# --- Image search term translation (for stock images) ---

IMAGE_SEARCH_SYSTEM = (
    "You are a bilingual lexicographer (BHS-English). "
    "Your task is to provide concise English keywords for image search."
)

IMAGE_SEARCH_USER = """\
Task:
- Input word or phrase (Bosnian/Croatian/Serbian, ijekavian): "{word}"

Return:
- 1 to 3 short English keywords suitable for stock image search, comma-separated.
- Do NOT include quotes or any extra explanation, only the keywords."""


# --- Context addendum (appended to any user prompt when context is provided) ---

CONTEXT_ADDENDUM = '\n\nKontekst (primjer u kojem se riječ koristi): "{context}"'
