#!/usr/bin/env python3
"""Score translit() on the validation English words (from val_items.csv `spoken`) and on a
self-written dev list of common loanwords (no evaluation words). Uses metrics.canon for CER."""
import csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent.parent.parent / "kit"
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HERE.parent / "frontend"))
from metrics import canon, _edit_distance  # noqa: E402
from translit import translit  # noqa: E402

VAL = {"office": "ऑफिस", "internet": "इंटरनेट", "slow": "स्लो", "new": "न्यू", "address": "एड्रेस",
       "message": "मैसेज", "flight": "फ्लाइट", "cancel": "कैंसिल", "hotel": "होटल", "online": "ऑनलाइन",
       "class": "क्लास", "walking": "वॉकिंग", "proper": "प्रॉपर", "sleep": "स्लीप", "weekend": "वीकेंड",
       "movie": "मूवी", "shopping": "शॉपिंग"}


def score(pairs, show=True):
    errs = chars = exact = 0
    for w, ref in pairs:
        hyp = translit(w)
        e = _edit_distance(canon(ref), canon(hyp))
        errs += e; chars += len(canon(ref)); exact += e == 0
        if show and e:
            print(f"   {w:12} {hyp:14} want {ref}")
    return errs / chars, exact / len(pairs)


if __name__ == "__main__":
    dev = [l.split("\t") for l in (HERE / "translit_devset.tsv").read_text(encoding="utf-8").split("\n") if l.strip()]
    show = "-q" not in sys.argv
    print("validation words:"); v = score(VAL.items(), show)
    print("dev words:"); d = score(dev, show)
    print(f"VAL  CER {v[0]:.3f} exact {v[1]:.2f}  ({len(VAL)} words)")
    print(f"DEV  CER {d[0]:.3f} exact {d[1]:.2f}  ({len(dev)} words)")
