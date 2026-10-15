#!/usr/bin/env python3
"""Write the model-input text for each system as <out>/<system>.csv (id,normalized_text).

  t2   : Task 2 normalizer only (English words stay in Latin script)
  idea : Task 3 = Task 2 normalizer + rule-based English->Devanagari transliteration
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frontend"))
from normalize import normalize  # noqa: E402
from translit import translit_text  # noqa: E402

import normalize as _norm  # noqa: E402


def _with_sep(sep, fn):
    def run(t):
        old, _norm.ABBR_SEP = _norm.ABBR_SEP, sep
        try:
            return fn(t)
        finally:
            _norm.ABBR_SEP = old
    return run


SYSTEMS = {
    "t2": normalize,
    "idea": lambda t: translit_text(normalize(t)),
    # experiment: spelled abbreviations with commas between letters (ए, टी, एम)
    "idea_abbrcomma": _with_sep(", ", lambda t: translit_text(normalize(t))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(HERE.parent.parent / "kit"))
    ap.add_argument("--out", default=str(HERE / "texts"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for name in ("val_items.csv", "eval_items.csv"):
        with open(Path(a.kit) / "data" / name, encoding="utf-8", newline="") as f:
            items += list(csv.DictReader(f))
    for sys_name, fn in SYSTEMS.items():
        with open(out / f"{sys_name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "normalized_text"])
            for it in items:
                w.writerow([it["id"], fn(it["text"])])
        print("wrote", out / f"{sys_name}.csv")


if __name__ == "__main__":
    main()
