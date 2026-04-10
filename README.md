# bcs-anki

Generate Anki flashcards (cloze deletions + images) from BCS word lists.

## Setup

```sh
pip install -e .
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
STOCK_IMAGE_API_KEY=...
```

The Anki media folder defaults to `~/Library/Application Support/Anki2/User 1/collection.media`. To change it, create a config YAML with `anki_media_folder: /your/path` and pass `--config config.yaml`.

## Workflow

```sh
# 1. Generate flashcards + images from a word list
bcs-anki generate words.txt

# 2. Copy generated images to Anki media folder (uses config defaults)
bcs-anki copy-media

# 3. Import output/words.csv in Anki desktop (File > Import)
```

## Other commands

```sh
bcs-anki generate words.txt --resume          # resume interrupted run
bcs-anki review output/words.csv --sample 10  # spot-check generated cards
bcs-anki status words.txt                     # check progress
bcs-anki validate output/words.csv            # validate CSV format
```

## Testing

```sh
pip install -e ".[dev]"
pytest -v
pytest --cov=bcs_anki
```
