#!/usr/bin/env python3
"""Fill results.json from the metrics.py JSON files and timing files that run_all.sh wrote."""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=str(HERE.parent))
    ap.add_argument("--roll", default=None)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--targeted-category", default="hinglish")
    a = ap.parse_args()
    sub = Path(a.sub)
    res_dir = HERE / "results"
    path = sub / "results.json"
    r = load(path) or load(HERE.parent.parent / "kit" / "results_template.json")
    r.pop("_help", None)
    if a.roll:
        r["roll_no"] = a.roll
    if a.machine:
        r["machine"] = a.machine
    r["seed"] = 0

    b = load(res_dir / "baseline_val_scores.json")
    if b:
        r["task1"] = {"val_cer": b["cer"], "val_sim": b["sim"]}
    c = load(res_dir / "val_control.json")
    if c:
        r["task2"] = {"val_cer": c["cer"], "val_sim": c["sim"], "val_cer_hard": c["cer_by_category"]["hard"]}
    i = load(res_dir / "val_idea.json")
    t3 = r["task3"]
    if i:
        t3.update({"method_name": "rule-based Indian-English letter-to-sound transliteration + syllable-based length for English-containing sentences",
               "targeted_failure": "english_word_error", "targeted_category": a.targeted_category,
               "builds_on_task2": True, "settings": {"decoding": "baseline config.yaml (nfe_step 32)", "text": "normalize + translit",
                                "speed": "per item: chars/syllables ratio vs prompt, clamped 0.8-1.25, only if the text has English words"}})
        t3.update({"val_cer": i["cer"], "val_sim": i["sim"],
                   "val_cer_targeted": i["cer_by_category"][a.targeted_category],
                   "val_control_cer": c["cer"] if c else None})
    f = load(res_dir / "val_fast.json")
    bt, ft = load(HERE / "baseline_val_timing.json"), load(HERE / "val_fast_timing.json")
    if f and bt and ft:
        r["task4"] = {"method_name": "Task 3 system with 12 flow-matching steps",
                      "settings": ft.get("overrides", {}), "val_cer": f["cer"],
                      "baseline_seconds": bt["total_synth_s"], "fast_seconds": ft["total_synth_s"],
                      "speedup": round(bt["total_synth_s"] / ft["total_synth_s"], 3)}
    path.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
