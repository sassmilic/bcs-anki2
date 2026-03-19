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
ANKI_MEDIA_FOLDER=~/Library/Application Support/Anki2/User 1/collection.media
```

## Workflow

```sh
# 1. Generate flashcards + images from a word list
bcs-anki generate words.txt

# 2. Copy generated images to Anki media folder
bcs-anki copy-media --from temp_images --to "$ANKI_MEDIA_FOLDER"

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
