#!/usr/bin/env bash
# Reproduces every submitted file. Run from the submission root (the folder holding code/ and
# predictions/) on a GPU machine after `huggingface-cli login`:
#
#     KIT=/path/to/kit bash code/run_all.sh
#
# Steps can be skipped with SKIP="t2 t3 t4" etc. Seed is 0 everywhere.
set -euo pipefail
SUB="$(pwd)"
KIT="${KIT:-$SUB/../kit}"
SEED="${SEED:-0}"
SKIP="${SKIP:-}"
FAST_SET="${FAST_SET:-nfe_step=12}"
PRED="$SUB/predictions"
RES="$SUB/code/results"
mkdir -p "$PRED" "$RES" "$SUB/out"
skip() { [[ " $SKIP " == *" $1 "* ]]; }

# Text (CPU only)
python code/make_t2_csv.py --kit "$KIT" --out "$PRED/t2_normalized.csv"
python code/make_texts.py --kit "$KIT" --out code/texts

RB="python $KIT/baseline/run_baseline.py --prompts $KIT/data/prompts.csv --seed $SEED"
MET="python $KIT/metrics.py"

# Task 1: score the provided baseline audio
if ! skip t1; then
  $MET "$KIT/data/baseline_val" --json "$RES/baseline_val_scores.json" --csv "$RES/baseline_val_scores.csv"
fi

# Task 2: normalized text, baseline otherwise. Eval audio + validation control.
if ! skip t2; then
  $RB --items "$KIT/data/eval_items.csv" --out "$PRED/t2_norm" --text-csv code/texts/t2.csv
  $RB --items "$KIT/data/val_items.csv"  --out "$PRED/val_control" --text-csv code/texts/t2.csv
  $MET "$PRED/val_control" --json "$RES/val_control.json" --csv "$RES/val_control.csv"
fi

# Task 3: the idea = Task 2 text + rule-based transliteration of English words + syllable-based
# output length (through `speed`) for sentences that contain English words
IDEA_OPTS="--text-csv code/texts/idea.csv --syllable-speed --syllable-scope english"
SYN="python code/synth_system.py --kit $KIT --seed $SEED"
if ! skip t3; then
  $SYN --items "$KIT/data/val_items.csv"  --out "$PRED/val_idea" $IDEA_OPTS
  $SYN --items "$KIT/data/eval_items.csv" --out "$PRED/t3_idea"  $IDEA_OPTS
  $MET "$PRED/val_idea" --json "$RES/val_idea.json" --csv "$RES/val_idea.csv"
fi

# Task 4: the Task 3 system with 12 flow-matching steps. Baseline and fast system timed in the
# same session on the same machine (model loaded, one warm-up excluded).
if ! skip t4; then
  $RB  --items "$KIT/data/val_items.csv" --out "$SUB/out/baseline_val"
  $SYN --items "$KIT/data/val_items.csv" --out "$SUB/out/val_fast" $IDEA_OPTS --set $FAST_SET
  $SYN --items "$KIT/data/eval_items.csv" --out "$PRED/t4_fast" $IDEA_OPTS --set $FAST_SET
  $MET "$SUB/out/val_fast" --json "$RES/val_fast.json" --csv "$RES/val_fast.csv"
  cp "$SUB/out/baseline_val_timing.json" "$SUB/out/val_fast_timing.json" code/   # Task 4 needs both in code/
fi
# run_baseline.py writes <folder>_timing.json next to each folder; keep predictions/ to audio + csvs
mv -f "$PRED"/*_timing.json "$RES"/ 2>/dev/null || true

python code/check_outputs.py --kit "$KIT" || echo "WARNING: some outputs fail the sanity checks (see above)"
python code/fill_results.py --sub "$SUB"
echo "Done. Next: python code/make_zip.py --roll <ROLLNO> && python $KIT/check_format.py <ROLLNO>_tts_bootcamp.zip"
