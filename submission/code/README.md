# Code – Hindi–Hinglish voice cloning bootcamp

Everything here is rule-based. No model is used anywhere on the path from text to audio except the
pinned IndicF5 baseline. Seed 0 everywhere.

## Layout

| Path | What it does |
|---|---|
| `frontend/normalize.py` | Task 2 `normalize(text)`: Indian-system numbers (लाख, करोड़), ₹ amounts and paise, dd/mm/yyyy dates, years, decimals, %, clock times, abbreviations spelled letter by letter (RBI → आर बी आई), or read as a word when they are pronounceable (4+ letters, 2+ vowels) |
| `frontend/translit.py` | Task 3 idea: rule-based English → Devanagari letter-to-sound rules for Indian English (grapheme units, open/closed syllables, silent final e, r-coloured vowels, unstressed endings, Hindi spelling conventions). No word list |
| `synth_idea.py` | Optional Task 3 extension: up to 4 spelling variants per Hinglish item (ऑ/ओ, ऐ/ए swaps), the one the recognizer reads back best is kept (verifier use only), with a 0.03 speaker-similarity guard |
| `analysis/translit_eval.py`, `analysis/translit_devset.tsv` | Checks the transliteration rules on the validation English words and on a self-written list of 107 common loanwords (no evaluation words) |
| `frontend/syllables.py` | Rule-based Devanagari syllable counter; `speed_for()` turns IndicF5's character-based length into a syllable-based one through the allowed `speed` setting |
| `prep_prompts.py` | Shortens pauses longer than 0.25 s inside the reference clips to 0.15 s (allowed trimming; transcript unchanged) |
| `synth_system.py` | Synthesizes and times any configuration: text CSV, `--syllable-speed`, `--prompts`, `--keep-pace`, decoding overrides |
| `run_experiments.sh`, `experiments.ipynb` | Validation-only grid for Task 3/Task 4 ideas; `analysis/compare.py` prints the table and per-item changes |
| `make_t2_csv.py` | Writes `predictions/t2_normalized.csv` (64 rows) |
| `make_texts.py` | Writes the model input for each system: `texts/t2.csv` (Task 2), `texts/idea.csv` (Task 3) |
| `run_all.sh` | Reproduces every submitted audio folder, the metrics and the timing files (GPU) |
| `run_all.ipynb` | The same in Colab/Kaggle |
| `fill_results.py` | Fills `results.json` from the metrics/timing files |
| `make_zip.py` | Packs `<roll>_tts_bootcamp.zip` |
| `analysis/eda.py` | Exploratory analysis of the kit and the baseline audio |
| `results/` | Metrics outputs (JSON/CSV) and EDA outputs |

## Exact commands

From the submission root (the folder with `code/` and `predictions/`), with the kit at `../kit`:

```bash
pip install -r ../kit/baseline/requirements.txt
huggingface-cli login                      # IndicF5 is gated

# Task 1 (CPU is fine)
python ../kit/metrics.py ../kit/data/baseline_val --json code/results/baseline_val_scores.json --csv code/results/baseline_val_scores.csv
python code/analysis/eda.py --kit ../kit --scores code/results/baseline_val_scores.csv --out code/results

# Tasks 2-4 (GPU): text, synthesis, scoring, timing, results.json
KIT=../kit bash code/run_all.sh

# Package and check
python code/fill_results.py --roll <ROLLNO> --machine "Colab T4"
python code/make_zip.py --roll <ROLLNO>
python ../kit/check_format.py <ROLLNO>_tts_bootcamp.zip
```

`run_all.sh` calls the kit's own `baseline/run_baseline.py` with `--text-csv` for every system, so
the decoding settings are exactly the baseline's unless stated (Task 4 uses `--set nfe_step=16`).

## Rules compliance

- Normalization and transliteration rules were written from the validation items and general
  knowledge of Hindi/English spelling; evaluation text was not used to add or change rules.
- `predictions/t1_val_errors.csv` was tagged from the metrics transcripts (and must be confirmed by listening).
- `analysis/translit_devset.tsv` is a self-written list of common English loanwords used only to check
  that the spelling rules generalize; it contains no evaluation words and no rule looks words up in it.

## AI coding assistant

The code in this folder was written with Claude Code (Anthropic's AI coding assistant).
`idea.md` and `report.md` are written by the participant.
