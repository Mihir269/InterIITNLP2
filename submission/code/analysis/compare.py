#!/usr/bin/env python3
"""Table of validation CER/SIM/time for every experiment, plus per-item changes against a reference."""
import argparse, csv, json
from pathlib import Path

CATS = ["hindi", "hinglish", "hard", "long"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--ref", default="idea")
    ap.add_argument("--base", default="base")
    a = ap.parse_args()
    res = Path(a.res)
    names = sorted(p.stem for p in res.glob("*.json") if not p.stem.endswith("_timing"))
    timing = {n: json.loads((res / f"{n}_timing.json").read_text())["total_synth_s"]
              for n in names if (res / f"{n}_timing.json").exists()}
    bt = timing.get(a.base)
    ref = json.loads((res / f"{a.ref}.json").read_text()) if (res / f"{a.ref}.json").exists() else None
    print(f"{'config':18}{'CER':>7}{'SIM':>7}" + "".join(f"{c[:5]:>8}" for c in CATS) + f"{'time_s':>9}{'speedup':>9}  guards")
    for n in names:
        d = json.loads((res / f"{n}.json").read_text())
        t = timing.get(n)
        sp = f"{bt / t:8.2f}x" if bt and t else f"{'':9}"
        g = ""
        if ref and n != a.ref and d["n_items"] == ref["n_items"]:
            g = "CER<=1.2*idea" if d["cer"] <= 1.2 * ref["cer"] else "CER>1.2*idea"
        print(f"{n:18}{d['cer']:7.4f}{d['sim']:7.4f}" + "".join(f"{d['cer_by_category'][c]:8.4f}" if c in d["cer_by_category"] else f"{'-':>8}" for c in CATS)
              + (f"{t:9.1f}" if t else f"{'':9}") + sp + "  " + g + (f"  failed={d['failed']}" if d["failed"] else ""))
    if not ref:
        return
    rows = {r["id"]: r for r in csv.DictReader(open(res / f"{a.ref}.csv", encoding="utf-8"))}
    for n in names:
        if n in (a.ref, a.base) or not (res / f"{n}.csv").exists():
            continue
        other = {r["id"]: r for r in csv.DictReader(open(res / f"{n}.csv", encoding="utf-8"))}
        diffs = sorted(((float(other[i]["cer"]) - float(rows[i]["cer"]), i) for i in rows if i in other), reverse=True)
        worse = [f"{i}+{d:.3f}" for d, i in diffs if d > 0.005][:4]
        better = [f"{i}{d:.3f}" for d, i in reversed(diffs) if d < -0.005][:4]
        print(f"{n:18} better: {', '.join(better) or '-'} | worse: {', '.join(worse) or '-'}")


if __name__ == "__main__":
    main()
