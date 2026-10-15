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

score() {  # name   (env ITEMS: items csv, default all validation items)
  [ -f "$RES/$1.json" ] || python "$KIT/metrics.py" "$OUT/$1" --items "${ITEMS:-$VAL}" --prompts "$KIT/data/prompts.csv" \
      --json "$RES/$1.json" --csv "$RES/$1.csv" > "$RES/$1.log" 2>&1
  cp -f "$OUT/$1_timing.json" "$RES/" 2>/dev/null || true
  python -c "import json;d=json.load(open('$RES/$1.json'));print('$1', 'CER', d['cer'], 'SIM', d['sim'], d['cer_by_category'])"
}
run() {  # name, args...
  local name=$1; shift
  if [ ! -f "$OUT/${name}_timing.json" ]; then
    echo "=== $name: $*"
    python code/synth_system.py --kit "$KIT" --items "${ITEMS:-$VAL}" --out "$OUT/$name" --seed "${SEED:-0}" "$@" || { echo "FAILED $name"; return; }
  fi
  score "$name"
}

# Reference points, same session: the baseline itself (for the Task 4 speed-up) and the current idea
if [[ "$PHASES" != "t2c" && "$PHASES" != *cfg* && "$PHASES" != *pause* ]]; then
# the baseline via the kit's own runner (its timing file is the Task 4 reference)
[ -f "$OUT/base_timing.json" ] || python "$KIT/baseline/run_baseline.py" --items "$VAL" --prompts "$KIT/data/prompts.csv" --out "$OUT/base" --seed 0
score base
run idea       --text-csv $IDEA
fi

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
if [[ " $PHASES " == *" t3b "* ]]; then
  # syllable-based length only where the sentence has English words
  run idea_sylE       --text-csv $IDEA --syllable-speed --syllable-scope english
  # robustness: a second seed on the 6 Hinglish items (scored on those items only)
  HG="$OUT/hinglish_items.csv"
  python -c "import csv,sys;r=list(csv.DictReader(open('$VAL',encoding='utf-8')));w=csv.DictWriter(open('$HG','w',encoding='utf-8',newline=''),fieldnames=list(r[0]));w.writeheader();w.writerows(x for x in r if x['category']=='hinglish')"
  ITEMS=$HG SEED=1 run s1hg_idea     --text-csv $IDEA
  ITEMS=$HG SEED=1 run s1hg_idea_syl --text-csv $IDEA --syllable-speed
  ITEMS=$HG SEED=2 run s2hg_idea     --text-csv $IDEA
  ITEMS=$HG SEED=2 run s2hg_idea_syl --text-csv $IDEA --syllable-speed
fi
if [[ " $PHASES " == *" t4b "* ]]; then
  run fast10          --text-csv $IDEA --set nfe_step=10
  run fast12_sylE     --text-csv $IDEA --syllable-speed --syllable-scope english --set nfe_step=12
  run fast10_sylE     --text-csv $IDEA --syllable-speed --syllable-scope english --set nfe_step=10
fi
if [[ " $PHASES " == *" t2c "* ]]; then
  # spelled abbreviations with commas between the letters, on the 8 hard items, seeds 0-2
  HD="$OUT/hard_items.csv"
  python -c "import csv;r=list(csv.DictReader(open('$VAL',encoding='utf-8')));w=csv.DictWriter(open('$HD','w',encoding='utf-8',newline=''),fieldnames=list(r[0]));w.writeheader();w.writerows(x for x in r if x['category']=='hard')"
  for S in 0 1 2; do
    ITEMS=$HD SEED=$S run s${S}hd_idea  --text-csv $IDEA
    ITEMS=$HD SEED=$S run s${S}hd_abbr  --text-csv code/texts/idea_abbrcomma.csv
  done
fi
FINAL="--text-csv $IDEA --syllable-speed --syllable-scope english"   # the submitted Task 3 system
# reference for rounds 4-5: the submitted Task 3 system (cfg 2.0, no pause allowance), seed 0.
# It is deterministic and already measured (round 2, idea_sylE), so it is copied, not re-run.
cp code/results/exp_round2/idea_sylE.json "$RES/final.json"; cp code/results/exp_round2/idea_sylE.csv "$RES/final.csv"
if [[ " $PHASES " == *" pause "* ]]; then
  # pause allowance: k extra syllables of time per sentence-internal punctuation mark
  run pause1 $FINAL --pause-syl 1
  run pause2 $FINAL --pause-syl 2
fi
if [[ " $PHASES " == *" cfg "* ]]; then
  # cfg_strength around the default 2.0, on the final Task 3 system, all 24 items, seed 0
  run cfg15  $FINAL --set cfg_strength=1.5
  run cfg25  $FINAL --set cfg_strength=2.5
  run cfg30  $FINAL --set cfg_strength=3.0
fi
if [[ " $PHASES " == *" cfg_seeds "* ]]; then
  # robustness of the best value (set CFGBEST, e.g. 2.5) against 2.0 on seeds 1 and 2
  for S in 1 2; do
    SEED=$S run s${S}_final $FINAL
    SEED=$S run s${S}_cfg${CFGBEST/./} $FINAL --set cfg_strength=$CFGBEST
  done
fi
python code/analysis/compare.py --res "$RES" --ref final --base base
