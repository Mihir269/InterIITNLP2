#!/usr/bin/env bash
# Validation-only experiment grid for Task 3 and Task 4. Run from the submission root on a GPU:
#     KIT=../kit PHASES="t3 t4" bash code/run_experiments.sh
# Every config runs the 24 validation items with seed 0, is scored with metrics.py and timed the
# same way as run_baseline.py. Results: code/results/exp/<name>.{json,csv} and <name>_timing.json.
# Configs whose outputs already exist are skipped, so the script can be resumed.
set -uo pipefail
SUB="$(pwd)"; KIT="${KIT:-$SUB/../kit}"; PHASES="${PHASES:-t3 t4}"
OUT="$SUB/out/exp"; RES="$SUB/code/results/exp"; mkdir -p "$OUT" "$RES"
VAL="$KIT/data/val_items.csv"; IDEA="code/texts/idea.csv"; TRIM="code/prompts_trim/prompts.csv"

python code/make_texts.py --kit "$KIT" --out code/texts >/dev/null
python code/prep_prompts.py --kit "$KIT" --out code/prompts_trim

score() {  # name
  [ -f "$RES/$1.json" ] || python "$KIT/metrics.py" "$OUT/$1" --items "$VAL" --prompts "$KIT/data/prompts.csv" \
      --json "$RES/$1.json" --csv "$RES/$1.csv" > "$RES/$1.log" 2>&1
  cp -f "$OUT/$1_timing.json" "$RES/" 2>/dev/null || true
  python -c "import json;d=json.load(open('$RES/$1.json'));print('$1', 'CER', d['cer'], 'SIM', d['sim'], d['cer_by_category'])"
}
run() {  # name, args...
  local name=$1; shift
  if [ ! -f "$OUT/${name}_timing.json" ]; then
    echo "=== $name: $*"
    python code/synth_system.py --kit "$KIT" --items "$VAL" --out "$OUT/$name" --seed 0 "$@" || { echo "FAILED $name"; return; }
  fi
  score "$name"
}

# Reference points, same session: the baseline itself (for the Task 4 speed-up) and the current idea
# the baseline via the kit's own runner (its timing file is the Task 4 reference)
[ -f "$OUT/base_timing.json" ] || python "$KIT/baseline/run_baseline.py" --items "$VAL" --prompts "$KIT/data/prompts.csv" --out "$OUT/base" --seed 0
score base
run idea       --text-csv $IDEA

if [[ " $PHASES " == *" t3 "* ]]; then
  run idea_syl        --text-csv $IDEA --syllable-speed
  run idea_trim       --text-csv $IDEA --prompts $TRIM
  run idea_trim_pace  --text-csv $IDEA --prompts $TRIM --keep-pace
  run idea_syl_trim   --text-csv $IDEA --prompts $TRIM --keep-pace --syllable-speed
fi
if [[ " $PHASES " == *" t4 "* ]]; then
  run fast16          --text-csv $IDEA --set nfe_step=16
  run fast12          --text-csv $IDEA --set nfe_step=12
  run fast8           --text-csv $IDEA --set nfe_step=8
  run fast12_trim     --text-csv $IDEA --prompts $TRIM --keep-pace --set nfe_step=12
  run fast16_nocfg    --text-csv $IDEA --set nfe_step=16 cfg_strength=0
fi
python code/analysis/compare.py --res "$RES" --ref idea --base base
