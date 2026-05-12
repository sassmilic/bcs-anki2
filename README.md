# bcs-anki

Create Anki-ready CSVs and images for learning BCS vocabulary.

## Setup

```sh
pip install -e .
cp .env.example .env
# Edit .env and fill in your API keys
```

The Anki media folder defaults to `~/Library/Application Support/Anki2/User 1/collection.media`. To change it, create a config YAML with `anki_media_folder: /your/path` and pass `--config config.yaml`.

## Workflow

### Words from reading

```sh
bcs-anki words words.txt
bcs-anki copy-media

# Then in Anki desktop: File > Import → output/<input>.csv
```

This path creates rich cards with dictionary-style definitions, example sentences, and images.

### Thematic dictionary pages

```sh
bcs-anki dictionary pages page1.jpg page2.jpg

# Then in Anki desktop: File > Import → output/cards/<subject>.csv
```

This path OCRs the page image(s), refines the extracted Serbian/English pairs, and creates image-based Anki cards.

If you want to inspect or redo one stage:

```sh
bcs-anki dictionary ocr page1.jpg page2.jpg
bcs-anki dictionary refine output/dict/<subject>.csv
bcs-anki dictionary csv output/dict/refined/<subject>.csv
```

## Testing

```sh
pip install -e ".[dev]"
pytest -v
pytest --cov=bcs_anki
```
