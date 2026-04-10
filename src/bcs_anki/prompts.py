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

1) Prepoznaj KANONSKI OBLIK (lemu). Ako je riječ pogrešno napisana ili bez dijakritičkih znakova, tiho koristi ispravan oblik bez napomene.
2) Odredi gramatičku kategoriju:
   - imenice: vrsta + rod (npr. "imenica, ž.")
   - glagoli: vrsta + "ja" oblik (npr. "glagol, ja vidim")
   - ostalo: samo vrsta (npr. "pridjev", "prilog", "veznik")
   - Samo ako je riječ NEOBIČNA, dodaj kratku napomenu (npr. "arhaično", "pretežno u hrvatskom").
3) Napiši KRATKU definiciju (~10 riječi) na BHS jeziku. Ako ima više značenja, odvoji ih tačkom-zarezom.

Format izlaza:
DEFINICIJA:
{{{{c1::KANONSKI_OBLIK}}}} (GRAMATIKA) — definicija; drugo značenje

Primjer izlaza:
{{{{c1::viditi}}}} (glagol, ja {{{{c1::vidim}}}}) — opažati očima; shvatiti, razumjeti

PRIMJERI:
1. Rečenica s {{{{c1::oblik1}}}}.<br>2. Rečenica s {{{{c1::oblik2}}}}.<br>3. Rečenica s {{{{c1::oblik3}}}}.

Pravila za primjere:
- Svaka rečenica MAKSIMALNO 10 riječi.
- Zajedno, rečenice trebaju pokriti SVA značenja riječi.
- Koristi različite gramatičke oblike (padeže, vremena, vidove...) gdje je moguće.
- Rečenice trebaju biti UPAMTLJIVE — koristi snažne vizuelne slike ili emocije."""


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


# --- Image source decision (stock vs AI) ---

IMAGE_SOURCE_SYSTEM = (
    "You are a visual content strategist. You decide whether a concept is best "
    "illustrated by a stock photograph or an AI-generated illustration."
)

IMAGE_SOURCE_USER = """\
Word (Bosnian/Croatian/Serbian): "{word}"

Decide whether this word/concept can be clearly and unambiguously represented by a \
simple stock photograph.

Reply ONLY with "stock" or "ai" — nothing else.

Guidelines:
- "stock" — concrete, everyday objects, animals, places, common actions, or professions \
that photograph well and are unlikely to be ambiguous in a photo.\
  Examples: apple, kitchen, ankle, zoo, bicycle, sunset, doctor, bridge, cat, umbrella.
- "ai" — abstract nouns, emotions, idioms, figurative expressions, culturally specific \
concepts, rare/unusual things, or anything where a generic photo would be misleading \
or unhelpful.\
  Examples: hope, irony, wanderlust, "to pull someone's leg", melancholy, redemption.

When in doubt, prefer "ai"."""


# --- Definition validation (reverse-guess) ---

VALIDATE_DEFINITION_SYSTEM = (
    "Ti si iskusni leksikograf za bosanski/hrvatski/srpski jezik (ijekavski)."
)

VALIDATE_DEFINITION_USER = """\
Data ti je definicija jedne BHS riječi (ijekavski). Pogodi koja je riječ.

Definicija: "{definition}"

Vrati TAČNO 3 kandidata, poredana od najvjerovatnijeg ka najmanje vjerovatnom.
Format: jedan kandidat po redu, sa procentom sigurnosti, bez dodatnog objašnjenja.

Primjer formata:
primirje 85%
prekid 10%
mir 5%"""


# --- Definition refinement ---

REFINE_DEFINITION_SYSTEM = DEFINITION_SYSTEM

REFINE_DEFINITION_USER = """\
Sljedeća definicija za riječ "{word}" nije dovoljno precizna — čitaoci je ne mogu \
jednoznačno povezati s pravom riječju.

Trenutna definicija: "{definition}"
Pogrešne pretpostavke čitaoca: {wrong_guesses}

Prepravi definiciju tako da jasnije i preciznije upućuje na "{word}".
Definicija mora biti kratka (~10 riječi). Odvoji značenja tačkom-zarezom.

Vrati SAMO jedan red u ovom formatu (bez oznake "DEFINICIJA:" i bez primjera):
{{{{c1::KANONSKI_OBLIK}}}} (GRAMATIKA) — definicija; drugo značenje"""


# --- Context addendum (appended to any user prompt when context is provided) ---

CONTEXT_ADDENDUM = '\n\nKontekst (primjer u kojem se riječ koristi): "{context}"'
