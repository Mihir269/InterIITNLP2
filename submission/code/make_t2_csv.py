#!/usr/bin/env python3
"""Task 2: write predictions/t2_normalized.csv (id,split,text,normalized_text) for all 64 items."""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frontend"))
from normalize import normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(HERE.parent.parent / "kit"))
    ap.add_argument("--out", default=str(HERE.parent / "predictions" / "t2_normalized.csv"))
    a = ap.parse_args()
    rows = []
    for split, name in (("val", "val_items.csv"), ("eval", "eval_items.csv")):
        with open(Path(a.kit) / "data" / name, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append({"id": r["id"], "split": split, "text": r["text"],
                             "normalized_text": normalize(r["text"])})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "text", "normalized_text"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
