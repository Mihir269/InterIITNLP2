# TTS Bootcamp Kit – Hindi–Hinglish Voice Cloning

Everything you need for the selection bootcamp. Read the problem statement first; this file only explains the kit.

## What is here

| Path | What it is |
|---|---|
| `baseline/tts.py` | The baseline: `synthesize(text, prompt_wav, prompt_text, seed=0, **overrides)` → 16 kHz float32 audio; `save_wav(path, audio)` writes the required format |
| `baseline/config.yaml` | The pinned baseline settings. Your system may override decoding values per call |
| `baseline/run_baseline.py` | Synthesizes a whole item file and times it (use it for Task 2 audio and Task 4 timing) |
| `baseline/run_baseline.ipynb` | Colab notebook: install, synthesize one item, rerun the baseline, score it |
| `baseline/patch_length.py` | The length fix the baseline ships with (see below). You do not need to call it |
| `data/val_items.csv` | 24 validation items: `id, category, text, prompt_id, spoken` |
| `data/eval_items.csv` | 40 evaluation items: `id, category, text, prompt_id` |
| `data/prompts.csv`, `data/prompts/` | 10 reference clips (24 kHz) and their transcripts: V1–V4 validation, E1–E6 evaluation |
| `data/baseline_val/` | The baseline's audio for the 24 validation items |
| `data/baseline_published.json` | Its validation CER and SIM from `metrics.py` |
| `metrics.py` | Scores a folder of validation audio: CER, SIM and sanity checks |
| `check_format.py` | Checks your submission zip. Run it before every upload |
| `results_template.json` | Copy to `results.json` and fill it in |

`spoken` is what the item should sound like, written out in Devanagari words. CER is computed against it. Where an item has more than one acceptable reading (a train number read as one number or digit by digit), the readings are separated by ` || ` and the closest one counts.

## Setup (Day 1)

1. Make a free Hugging Face account and accept the model terms at https://huggingface.co/ai4bharat/IndicF5. Create an access token.
2. Open `baseline/run_baseline.ipynb` in Colab (or Kaggle) with a T4 GPU, add the token as a secret named `HF_TOKEN`, point `KIT` at this folder, and run all.
3. If you see "the length fix was just applied ... restart", restart the session and run again. This happens once per fresh machine.

On your own machine: `pip install -r baseline/requirements.txt`, `huggingface-cli login`, then run the commands below from this folder.

## Commands by task

```bash
# Task 1 – score the provided baseline audio (works on CPU, slowly)
python metrics.py data/baseline_val --csv baseline_val_scores.csv

# Task 2 – synthesize the evaluation items from your normalized text
python baseline/run_baseline.py --items data/eval_items.csv --out predictions/t2_norm \
    --text-csv predictions/t2_normalized.csv --text-column normalized_text

# Task 3 – your system, scored on validation audio
python metrics.py predictions/val_idea --json val_idea.json --csv val_idea.csv

# Task 4 – time the baseline and your fast system in the SAME session on the SAME machine
python baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val      # -> out/baseline_val_timing.json
# ... time your fast system on the same 24 items, same way (model loaded, one warm-up excluded)

# Before uploading
python check_format.py 21CS10042_tts_bootcamp.zip
```

Using the wrapper from your own code:

```python
from baseline.tts import synthesize, save_wav
audio = synthesize(text, "data/prompts/V1.wav", transcript, seed=0)              # baseline settings
audio = synthesize(text, "data/prompts/V1.wav", transcript, seed=0, nfe_step=16)  # an override
save_wav("predictions/val_idea/val_hi_01.wav", audio)
```

Decoding values you can override: `nfe_step`, `cfg_strength`, `sway_sampling_coef`, `speed`, `cross_fade_duration`, `target_rms`. Splitting long text yourself, editing or trimming the prompt clip or its transcript, and generating up to 4 candidates per item are also allowed; see the Rules in the problem statement.

## Things worth knowing

- **The length fix.** IndicF5 estimates how long to speak from the length of the text. Upstream code counts bytes, which gives Latin-script words about a third of the time Devanagari would get. The baseline counts characters instead. This is part of the baseline, not something you need to fix.
- **metrics.py is a development tool.** Its recognizer is told to write Hindi words only, so English words come out transliterated and numbers come out as words. The final scoring uses different recognizers and a different speaker model with the same text rules, so improvements that only fool this one recognizer will not survive.
- **Failed items.** A missing file, or one with mostly silence, an implausible speaking rate, clipping, or a duration outside 0.5–60 s, counts as CER 1 and SIM 0.
- **Seeds.** Use the same seed for your system and the baseline when comparing on validation, and record it in `results.json`.
- **report.md format.** Start each answer on a line beginning `Q1`, `Q2`, `Q3`; do not paste the question text (it counts toward the 75 words).

## Credits and licences

Reference clips: IndicVoices-R (AI4Bharat), CC BY 4.0. Baseline model: IndicF5 (AI4Bharat), used under its terms; do not use it to clone anyone's voice without consent. Sentences: written by the organizers for this bootcamp.
