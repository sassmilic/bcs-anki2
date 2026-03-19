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

## Usage

```sh
# Generate flashcards
bcs-anki generate words.txt

# Resume interrupted run
bcs-anki generate words.txt --resume

# Review generated cards
bcs-anki review output/words.csv --image-dir temp_images --sample 10

# Copy images to Anki media folder
bcs-anki copy-media --from temp_images --to ~/Library/Application\ Support/Anki2/User\ 1/collection.media

# Check progress
bcs-anki status words.txt

# Validate CSV format
bcs-anki validate output/words.csv
```

## Testing

```sh
pip install -e ".[dev]"
pytest -v
pytest --cov=bcs_anki
```
