# bcs-anki

Generate Anki flashcards (cloze deletions + images) from BCS word lists.

## Setup

```sh
pip install -e .
cp run.sh.example run.sh
chmod +x run.sh
# Edit run.sh and fill in your API keys (OPENAI_API_KEY + one stock-image key)
```

The Anki media folder defaults to `~/Library/Application Support/Anki2/User 1/collection.media`. To change it, create a config YAML with `anki_media_folder: /your/path` and pass `--config config.yaml`.

## Workflow

```sh
./run.sh                # uses words.txt
./run.sh other.txt      # custom word list

# Then in Anki desktop: File > Import → output/<input>.csv
```

`run.sh` runs `bcs-anki generate` and then `bcs-anki copy-media` for you.

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
