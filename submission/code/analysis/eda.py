#!/usr/bin/env python3
"""Exploratory analysis of the kit: item text, prompt clips and the baseline's validation audio.

    python code/analysis/eda.py --kit kit --scores code/results/baseline_val_scores.csv --out code/results
"""
import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audio_stats(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    hop = int(0.025 * sr)
    n = len(y) // hop
    rms = np.sqrt((y[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1))
    voiced = np.where(rms >= 0.003)[0]
    lead = voiced[0] * 0.025 if len(voiced) else 0.0
    trail = (n - 1 - voiced[-1]) * 0.025 if len(voiced) else 0.0
    # longest internal pause
    longest, run = 0, 0
    for v in rms[voiced[0]:voiced[-1] + 1] if len(voiced) else []:
        run = run + 1 if v < 0.003 else 0
        longest = max(longest, run)
    return {"dur_s": len(y) / sr, "sr": sr, "rms": float(np.sqrt((y ** 2).mean())),
            "silence_share": float((rms < 0.003).mean()), "lead_sil_s": lead, "trail_sil_s": trail,
            "longest_pause_s": longest * 0.025, "peak": float(np.abs(y).max())}


def nchars(s):
    return len(re.sub(r"[\s।॥.,!?;:\"'()\-–—]", "", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(HERE.parent.parent.parent / "kit"))
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    a = ap.parse_args()
    kit, out = Path(a.kit), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    val, ev = read_csv(kit / "data/val_items.csv"), read_csv(kit / "data/eval_items.csv")
    prompts = read_csv(kit / "data/prompts.csv")
    rep = {}

    # 1. text
    def text_feats(it):
        t = it["text"]
        return {"words": len(t.split()), "chars": nchars(t), "latin_words": len(re.findall(r"[A-Za-z]+", t)),
                "caps_abbrev": len(re.findall(r"\b[A-Z]{2,}\b", t)), "digit_groups": len(re.findall(r"\d[\d,./]*", t)),
                "rupee": t.count("₹")}
    for name, items in (("val", val), ("eval", ev)):
        cats = {}
        for it in items:
            f = text_feats(it)
            c = cats.setdefault(it["category"], {k: [] for k in f})
            for k, v in f.items():
                c[k].append(v)
        rep[f"text_{name}"] = {c: {k: round(float(np.mean(v)), 2) for k, v in d.items()} | {"n": len(d["words"])}
                               for c, d in cats.items()}
    # 2. prompts
    rep["prompts"] = {}
    for p in prompts:
        s = audio_stats(kit / "data" / p["file"])
        s["chars_per_s"] = nchars(p["transcript"]) / s["dur_s"]
        s["chars_per_voiced_s"] = nchars(p["transcript"]) / max(1e-3, s["dur_s"] - s["lead_sil_s"] - s["trail_sil_s"])
        rep["prompts"][p["prompt_id"]] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} | \
                                          {"gender": p["gender"], "split": p["split"]}
    # 3. baseline audio vs text
    rows = []
    scores = {r["id"]: r for r in read_csv(a.scores)} if a.scores and Path(a.scores).exists() else {}
    for it in val:
        s = audio_stats(kit / "data/baseline_val" / f"{it['id']}.wav")
        ref = it["spoken"].split("||")[0]
        p_rate = rep["prompts"][it["prompt_id"]]["chars_per_s"]
        r = {"id": it["id"], "category": it["category"], "prompt": it["prompt_id"],
             "dur_s": round(s["dur_s"], 2), "ref_chars": nchars(ref), "input_chars": nchars(it["text"]),
             "chars_per_s": round(nchars(ref) / s["dur_s"], 2), "prompt_chars_per_s": round(p_rate, 2),
             "lead_sil_s": round(s["lead_sil_s"], 2), "trail_sil_s": round(s["trail_sil_s"], 2),
             "longest_pause_s": round(s["longest_pause_s"], 2), "silence_share": round(s["silence_share"], 3)}
        if it["id"] in scores:
            r["cer"] = float(scores[it["id"]]["cer"])
            r["sim"] = float(scores[it["id"]]["sim"])
            r["hyp"] = scores[it["id"]]["hyp"]
        rows.append(r)
    with open(out / "eda_baseline_items.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if scores:
        x = np.array([r["chars_per_s"] for r in rows]); y = np.array([r["cer"] for r in rows])
        rep["corr_rate_cer"] = round(float(np.corrcoef(x, y)[0, 1]), 3)
        ratio = np.array([r["input_chars"] / r["ref_chars"] for r in rows])
        rep["corr_inputchars_over_refchars_vs_cer"] = round(float(np.corrcoef(ratio, y)[0, 1]), 3)
        by_prompt = {}
        for r in rows:
            by_prompt.setdefault(r["prompt"], []).append((r["cer"], r["sim"]))
        rep["by_prompt"] = {k: {"cer": round(float(np.mean([a for a, _ in v])), 4),
                                "sim": round(float(np.mean([b for _, b in v])), 4), "n": len(v)}
                            for k, v in sorted(by_prompt.items())}
    (out / "eda_summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if scores:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            colors = {"hindi": "#4C78A8", "hinglish": "#F58518", "hard": "#E45756", "long": "#54A24B"}
            for r in rows:
                ax[0].scatter(r["chars_per_s"], r["cer"], c=colors[r["category"]], s=30)
                ax[1].scatter(r["input_chars"] / r["ref_chars"], r["cer"], c=colors[r["category"]], s=30)
            for c, col in colors.items():
                ax[0].scatter([], [], c=col, label=c)
            ax[0].legend(frameon=False)
            ax[0].set_xlabel("spoken-reference chars per second of output"); ax[0].set_ylabel("CER")
            ax[1].set_xlabel("input text chars / spoken reference chars"); ax[1].set_ylabel("CER")
            for a_ in ax:
                a_.spines[["top", "right"]].set_visible(False)
            fig.suptitle("Baseline validation items")
            fig.tight_layout()
            fig.savefig(out / "eda_baseline.png", dpi=120)
    except Exception as e:
        print("plot skipped:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
