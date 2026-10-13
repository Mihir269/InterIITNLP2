#!/usr/bin/env python3
"""
Shorten long pauses inside the reference clips (allowed: trimming the reference clip).

Every pause longer than --max-pause seconds is cut down to --keep seconds (the middle is
removed, with 10 ms fades); leading/trailing silence is cut to 50 ms. The words and their
order are unchanged, so the transcript still matches. Writes <out>/<id>.wav (same sample rate)
and <out>/prompts.csv with the new durations.

    python code/prep_prompts.py --kit ../kit --out code/prompts_trim
"""
import argparse, csv
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent


def compress_pauses(y, sr, thresh_db=-40.0, max_pause=0.25, keep=0.15, edge=0.05):
    hop = int(0.01 * sr)
    n = len(y) // hop
    rms = np.sqrt((y[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-9
    ref = np.percentile(rms, 95)
    sil = 20 * np.log10(rms / ref) < thresh_db
    voiced = np.where(~sil)[0]
    if len(voiced) == 0:
        return y
    start = max(0, voiced[0] - int(edge / 0.01))
    end = min(n, voiced[-1] + 1 + int(edge / 0.01))
    pieces, k = [], start
    fade = int(0.01 * sr)
    i = start
    while i < end:
        if sil[i]:
            j = i
            while j < end and sil[j]:
                j += 1
            if (j - i) * 0.01 > max_pause and i > start and j < end:
                half = int(keep / 0.01 / 2)
                pieces.append(y[k * hop:(i + half) * hop].copy())
                k = j - half
            i = j
        else:
            i += 1
    pieces.append(y[k * hop:end * hop].copy())
    for p in pieces:  # short fades so the joins do not click
        if len(p) > 2 * fade:
            p[:fade] *= np.linspace(0, 1, fade)
            p[-fade:] *= np.linspace(1, 0, fade)
    return np.concatenate(pieces).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(HERE.parent.parent / "kit"))
    ap.add_argument("--out", default=str(HERE / "prompts_trim"))
    ap.add_argument("--max-pause", type=float, default=0.25)
    ap.add_argument("--keep", type=float, default=0.15)
    a = ap.parse_args()
    kit, out = Path(a.kit), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(kit / "data" / "prompts.csv", encoding="utf-8")))
    for r in rows:
        y, sr = sf.read(str(kit / "data" / r["file"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        z = compress_pauses(y, sr, max_pause=a.max_pause, keep=a.keep)
        sf.write(str(out / f"{r['prompt_id']}.wav"), z, sr, subtype="PCM_16")
        print(f"{r['prompt_id']}: {len(y) / sr:.2f}s -> {len(z) / sr:.2f}s")
        r["file"], r["duration_s"] = f"{r['prompt_id']}.wav", f"{len(z) / sr:.2f}"
    with open(out / "prompts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
