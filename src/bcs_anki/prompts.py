"""
All LLM prompts in one place for easy tweaking.

Each pair is (system_prompt, user_prompt_template).
User prompt templates use {word} for substitution.
"""

# --- Lemma resolution ---

LEMMA_SYSTEM = (
    "Ti si iskusni leksikograf za bosanski/hrvatski/srpski jezik (ijekavski)."
)

LEMMA_USER = """\
Vrati KANONSKI OBLIK (lemu) zadane riječi ili izraza.

Ulaz: "{word}"

Pravila:
- Ako je riječ pogrešno napisana ili bez dijakritičkih znakova, tiho ispravi.
- Glagol → infinitiv (npr. "vidim" → "vidjeti").
- Imenica/pridjev → nominativ jednine, muški rod (osim ako je riječ samo ženskog ili srednjeg roda).
- Izraz/fraza → ostavi u uobičajenom rječničkom obliku.

Vrati SAMO jednu riječ ili izraz, bez objašnjenja, navodnika ili prefiksa."""


# --- Definition ---

DEFINITION_SYSTEM = (
    "Ti si iskusni leksikograf za bosanski/hrvatski/srpski jezik (ijekavski). "
    "Odgovaraj isključivo na BHS jeziku, ijekavski, bosanski dijalekt, bez objašnjenja na engleskom."
)

DEFINITION_USER = """\
Zadatak:
- Za zadanu riječ ili izraz: "{word}"

1) Prepoznaj KANONSKI OBLIK (lemu). Ako je riječ pogrešno napisana ili bez dijakritičkih znakova, tiho koristi ispravan oblik bez napomene.
2) Odredi gramatičku kategoriju:
   - imenice: vrsta + rod (npr. "imenica, ž.")
   - glagoli: OBAVEZNO vrsta + "ja" oblik u zagradi, omotan u {{{{c1::...}}}} (npr. "glagol, {{{{c1::vidim}}}}"). 
   - ostalo: samo vrsta (npr. "pridjev", "prilog", "veznik")
   - Samo ako je riječ NEOBIČNA, dodaj kratku napomenu (npr. "arhaično", "pretežno u hrvatskom").
3) Napiši KRATKU definiciju (maksimalno 15 riječi) na BHS jeziku. Ako ima više značenja, odvoji ih tačkom-zarezom.

Format:
{{{{c1::KANONSKI_OBLIK}}}} (GRAMATIKA) — definicija; drugo značenje (ako postoji)

Primjer izlaza:
{{{{c1::raspiriti}}}} (glagol, {{{{c1::raspirim}}}}) — podstaknuti vatru da jače gori; pojačati emocije ili interes"""


# --- Examples ---

EXAMPLES_SYSTEM = DEFINITION_SYSTEM

EXAMPLES_USER = """
Zadatak:
- Za zadanu riječ ili izraz: "{word}"

Napiši 3 primjera rečenica koje koriste ovu riječ.

Pravila:
1. KRATKOĆA: Svaka rečenica MAKSIMALNO 15 riječi.
2. SEMANTIKA: Zajedno, rečenice trebaju pokriti što više značenja riječi (npr. doslovno i preneseno).
3. GRAMATIKA: Koristi različite gramatičke oblike (padeže, vremena, vidove) i osiguraj da su povratni glagoli (se) ispravno upotrijebljeni.
4. KVALITET: Rečenice trebaju biti prirodne (native-like) i UPAMTLJIVE — koristi snažne vizuelne slike ili emocije. Izbjegavaj generičke primjere.
5. CLOZE: SVAKO pojavljivanje ciljne riječi u SVAKOJ rečenici MORA biti omotano u {{{{c1::...}}}}.
6. KONTROLA: Ako riječ ima pejorativno (uvredljivo) ili arhaično značenje, osiguraj da rečenica to jasno odražava kroz kontekst.

Format izlaza (isključivo HTML bullet points, bez numeracije, bez uvoda):
<ul>
  <li>Gasiš {{{{c1::vatru}}}} benzinom ako misliš da ćeš tako riješiti taj sukob.</li>
  <li>Nakon dugog uspona, napokon smo naložili {{{{c1::vatre}}}} da se ugrijemo.</li>
  <li>Njezin je nastup bio pun {{{{c1::vatre}}}} i neobuzdane energije.</li>
</ul>
"""


# --- Gemini review: definition ---

REVIEW_DEFINITION_SYSTEM = (
    "Ti si izvorni govornik bosanskog/hrvatskog/srpskog jezika i iskusni leksikograf. "
    "OBAVEZNO koristiš IJEKAVSKI izgovor i pravopis (npr. rječnik, vrijeme, primjer, lijep, "
    "mlijeko, dijete — NIKAD ekavski oblike rečnik, vreme, primer, lep, mleko, dete). "
    "Ne ispravljaj ijekavske oblike u ekavske; ako su u ulazu ekavski oblici, to je greška "
    "koju treba ispraviti u ijekavske. "
    "Tvoj zadatak je validirati definicije rječnika napisane od strane drugog AI modela."
)

REVIEW_DEFINITION_USER = """\
Riječ koja se definira: "{word}"

Predložena definicija (od drugog AI modela):
{definition}

Provjeri:
1. Da li je definicija činjenično tačna i nedvosmisleno opisuje upravo riječ "{word}" (a ne neku sličnu riječ)?
2. Da li je gramatička kategorija (vrsta riječi, rod, "ja"-oblik za glagole) ispravna?
3. Da li je CJELOKUPNI BCS tekst strogo i dosljedno u ijekavici (bez ijednog ekavskog oblika)?
4. Da li je zadržana izvorna struktura: {{{{c1::lema}}}} (gramatika) — definicija; drugo značenje (ako postoji)?

PRAVILA ODGOVORA — IZUZETNO VAŽNO:
- Ako je definicija u potpunosti ispravna, odgovori SAMO znakom: ✓
  (jedan znak, bez ikakvog drugog teksta, bez objašnjenja, bez razmaka)
- Ako je definicija pogrešna, odgovori ISKLJUČIVO ispravljenom verzijom definicije
  (jedan red, isti format kao izvorna, sa minimalnim brojem izmjena potrebnih da bude tačna).
  Ne dodaji komentar, prefiks "Ispravka:", niti bilo kakav drugi tekst."""


# --- Gemini review: examples ---

REVIEW_EXAMPLES_SYSTEM = (
    "Ti si izvorni govornik bosanskog/hrvatskog/srpskog jezika. "
    "OBAVEZNO koristiš IJEKAVSKI izgovor i pravopis (npr. rječnik, vrijeme, primjer, lijep, "
    "mlijeko, dijete — NIKAD ekavski oblike rečnik, vreme, primer, lep, mleko, dete). "
    "Ne ispravljaj ijekavske oblike u ekavske; ako su u ulazu ekavski oblici, to je greška "
    "koju treba ispraviti u ijekavske. "
    "Tvoj zadatak je validirati primjere rečenica koje je napisao drugi AI model."
)

REVIEW_EXAMPLES_USER = """\
Ciljna riječ: "{word}"

Predloženi primjeri (HTML lista):
{examples}

Provjeri svaku rečenicu po sljedećim kriterijumima:
1. Gramatička ispravnost (padeži, vremena, vidovi, povratni glagoli "se").
2. Semantička prirodnost — da li bi izvorni govornik to zaista rekao?
3. Tolerancija za kreativnost: dozvoljene su neobične, ekspresivne rečenice koliko bi ih napisao kreativan pisac. Odbaci samo ono što izvorni govornik NE BI rekao.
4. Da li je CJELOKUPNI BCS tekst strogo i dosljedno u ijekavici (bez ijednog ekavskog oblika)?
5. Da li su sva pojavljivanja ciljne riječi i dalje omotana u {{{{c1::...}}}}?

PRAVILA ODGOVORA — IZUZETNO VAŽNO:
- Ako su SVE rečenice ispravne, odgovori SAMO znakom: ✓
  (jedan znak, bez ikakvog drugog teksta)
- Ako je bar jedna rečenica neispravna, odgovori ISKLJUČIVO ispravljenom verzijom CIJELE HTML liste
  (<ul><li>...</li>...</ul>), sa minimalnim brojem izmjena, zadržavajući sve {{{{c1::...}}}} markere.
  Ne dodaji komentar niti bilo kakav drugi tekst."""


# --- Image prompt generation (for AI-generated images) ---

IMAGE_PROMPT_SYSTEM = (
    "Ti si kreativni dizajner koji osmišljava upute za generiranje slika (prompte) za AI generator slika. "
    "Ne koristi tekst u samoj slici. Piši na engleskom jeziku."
)

IMAGE_PROMPT_USER = """\
Task:
- Target word (BCS): "{word}"

Create an English prompt for an AI image generator that:
1. Visualizes the meaning of the word "{word}" clearly and unambiguously.
2. Adapt the style:
   - Verbs/Actions: Dynamic scene showing the movement or interaction.
   - Abstract Nouns: Conceptual, metaphorical; get creative, but stay true to the meaning of the word.
   - Emotions: Atmospheric, capturing emotion by focusing on lighting, colors, and facial expressions.
   - Idioms: Creative visual metaphors for the figurative meaning.
3. Composition: Focus on a single, clear subject to ensure the card remains readable at a glance.
5. Constraint: NO text, letters, or captions in the image. Do not use the word "{word}" as an instruction for the image content."""


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


# --- Dictionary OCR (Gemini, image input) ---

DICT_OCR_SYSTEM = (
    "OCR Serbian/Eng thematic dictionary pages. Same subject per request. "
    "Each page: heading, numbered Eng list, numbered Serbian/SC list. "
    "Return strict JSON pairing entries by matching number."
)

DICT_OCR_USER = """\
Extract image(s) to JSON only:

{
  "subject": "<first-page heading>",
  "entries": [{"n": "<number or range>", "eng": "<Eng>", "sr": "<Serbian/SC as printed>"}]
}

Rules:
- Pair same-number Eng/Sr terms.
- Sr numbering resets to 1; that marks Sr list start.
- Multi-image request = consecutive pages, same subject; pair within each page/list block even if numbering restarts.
- `n` is a string. For ordinary entries it is the number ("1", "2", ...). For category headers that span a range of sub-entries (e.g. "1-5. layered structure of the earth" labelling rows 1 through 5), use the range itself as `n` (e.g. "1-5"). Emit BOTH the category header AND each numbered sub-entry as separate entries, in reading order.
- Preserve printed forms exactly: commas, parentheses, alternates.
- Join wrapped lines with single spaces.
- Skip an entry (header or row) only if you cannot find a paired Serbian counterpart for it; do not guess.
- No markdown/commentary.
"""


