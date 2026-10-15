#!/usr/bin/env python3
"""
Run the scorer's audio sanity checks (duration, silence share, speaking rate, clipping) on every
submitted audio folder, including the evaluation folders that metrics.py cannot score. The
speaking rate uses the model input text as the reference (evaluation items have no `spoken`).

    python code/check_outputs.py --kit ../kit
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(HERE.parent.parent / "kit"))
    ap.add_argument("--pred", default=str(HERE.parent / "predictions"))
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.kit).resolve()))
    import metrics as M
    texts = {}
    for name in ("t2", "idea"):
        p = HERE / "texts" / f"{name}.csv"
        if p.exists():
            texts[name] = {r["id"]: r["normalized_text"] for r in csv.DictReader(open(p, encoding="utf-8"))}
    folders = {"t2_norm": "t2", "val_control": "t2", "t3_idea": "idea", "val_idea": "idea", "t4_fast": "idea"}
    bad = 0
    for folder, tname in folders.items():
        d = Path(a.pred) / folder
        if not d.exists():
            continue
        rates, worst = [], []
        for w in sorted(d.glob("*.wav")):
            y, _ = M.load_audio(w)
            ok, reasons, st = M.sanity(y, 16000, len(M.canon(texts[tname][w.stem])))
            rates.append(st.get("chars_per_s", 0))
            if not ok:
                worst.append(f"{w.stem}: {'; '.join(reasons)}")
        bad += len(worst)
        print(f"{folder:12} {len(rates)} files, rate {min(rates):.1f}-{max(rates):.1f} chars/s, "
              f"{'all pass' if not worst else str(len(worst)) + ' FAIL'}")
        for x in worst:
            print("    ", x)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
