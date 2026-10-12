#!/usr/bin/env python3
"""
Synthesize a set of items with the baseline and time it.

    # baseline on all validation items (reproduces data/baseline_val/)
    python baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val

    # Task 2: same, but read the text from your normalized CSV
    python baseline/run_baseline.py --items data/eval_items.csv --out predictions/t2_norm \
        --text-csv predictions/t2_normalized.csv --text-column normalized_text

    # override decoding settings
    python baseline/run_baseline.py --items data/val_items.csv --out out/fast --set nfe_step=16

Writes <out>/<item_id>.wav (16 kHz mono PCM) and <out>_timing.json next to the folder (kept
outside it, because submission folders must hold only the .wav files). Timing excludes model
loading and one warm-up item, so it is the number to use for the Task 4 speed-up.
"""
import argparse, csv, json, platform, sys, time
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from baseline.tts import get_engine, save_wav  # noqa: E402


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--prompts", default=str(KIT / "data" / "prompts.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", help="comma-separated subset of item ids")
    ap.add_argument("--text-csv", help="CSV with an `id` column and replacement text")
    ap.add_argument("--text-column", default="normalized_text")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="decoding overrides")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()

    items = read_csv(a.items)
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    if a.text_csv:
        repl = {r["id"]: r[a.text_column] for r in read_csv(a.text_csv)}
        missing = [it["id"] for it in items if it["id"] not in repl]
        if missing:
            sys.exit(f"{a.text_csv} lacks ids: {missing[:5]}...")
        for it in items:
            it["text"] = repl[it["id"]]
    pbase = Path(a.prompts).resolve().parent
    prompts = {r["prompt_id"]: dict(r, path=str(pbase / r["file"])) for r in read_csv(a.prompts)}
    overrides = dict(kv.split("=", 1) for kv in a.set)
    overrides = {k: parse_value(v) for k, v in overrides.items()}

    import torch
    t0 = time.time()
    engine = get_engine()
    load_s = time.time() - t0
    if not a.no_warmup and items:
        p = prompts[items[0]["prompt_id"]]
        engine("नमस्ते, यह एक छोटा सा परीक्षण है।", p["path"], p["transcript"], seed=a.seed, **overrides)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_item, total = {}, 0.0
    for i, it in enumerate(items, 1):
        p = prompts[it["prompt_id"]]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        y = engine(it["text"], p["path"], p["transcript"], seed=a.seed, **overrides)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t
        total += dt
        save_wav(out / f"{it['id']}.wav", y)
        per_item[it["id"]] = round(dt, 3)
        print(f"[{i}/{len(items)}] {it['id']}: {dt:.2f}s synth, {len(y) / 16000:.2f}s audio", flush=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing = {"total_synth_s": round(total, 3), "n_items": len(items), "model_load_s": round(load_s, 1),
              "seed": a.seed, "overrides": overrides, "gpu": gpu, "python": platform.python_version(),
              "items_file": str(a.items), "per_item_s": per_item}
    tpath = out.parent / f"{out.name}_timing.json"
    tpath.write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(f"\nDone: {len(items)} items in {total:.1f}s of synthesis on {gpu}. Timing in {tpath}")


if __name__ == "__main__":
    main()
