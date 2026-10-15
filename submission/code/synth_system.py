#!/usr/bin/env python3
"""
Synthesize a set of items with a configurable system and time it the way run_baseline.py does
(model loaded, one warm-up item excluded, CUDA synchronized around each item).

Options (each is an allowed change: model input text, reference clip, decoding settings):
  --text-csv           model input text per item (texts/t2.csv or texts/idea.csv)
  --syllable-speed     per-item `speed` so the output length follows syllable counts instead of
                       character counts (frontend/syllables.py)
  --syllable-scope     all | english: only for items whose original text has an English word
                       (a Latin-script word with a lower-case letter; ALL-CAPS abbreviations do not count)
  --prompts            alternative prompts.csv, e.g. pause-compressed clips from prep_prompts.py
  --keep-pace          with a shortened prompt, rescale `speed` so outputs keep the original
                       clip's pace (saves compute without speaking faster)
  --set k=v ...        decoding overrides (nfe_step, cfg_strength, sway_sampling_coef, ...)

    python code/synth_system.py --items ../kit/data/val_items.csv --out out/exp_syl \
        --text-csv code/texts/idea.csv --syllable-speed
"""
import argparse, csv, json, platform, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frontend"))
from syllables import chars, speed_for  # noqa: E402


def has_english(text):
    """True if the text has an English word in Latin script (not just an ALL-CAPS abbreviation)."""
    return any(re.search(r"[a-z]", w) for w in re.findall(r"[A-Za-z]+", text))


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default=str(HERE.parent.parent / "kit"))
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-csv")
    ap.add_argument("--text-column", default="normalized_text")
    ap.add_argument("--prompts", default=None, help="prompts.csv (default: the kit's)")
    ap.add_argument("--syllable-speed", action="store_true")
    ap.add_argument("--syllable-scope", choices=["all", "english"], default="all")
    ap.add_argument("--ids", help="comma-separated subset of item ids")
    ap.add_argument("--min-rate", type=float, default=5.0, help="chars/s floor when slowing an item down")
    ap.add_argument("--keep-pace", action="store_true")
    ap.add_argument("--speed-range", default="0.8,1.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    kit = Path(a.kit).resolve()
    sys.path.insert(0, str(kit))
    from baseline.tts import get_engine, save_wav
    import torch

    items = read_csv(a.items)
    if a.ids:
        keep = set(a.ids.split(","))
        items = [it for it in items if it["id"] in keep]
    for it in items:
        it["orig_text"] = it["text"]
    if a.text_csv:
        repl = {r["id"]: r[a.text_column] for r in read_csv(a.text_csv)}
        for it in items:
            it["text"] = repl[it["id"]]
    orig = {r["prompt_id"]: r for r in read_csv(kit / "data" / "prompts.csv")}
    pcsv = Path(a.prompts) if a.prompts else kit / "data" / "prompts.csv"
    prompts = {r["prompt_id"]: dict(r, path=str(pcsv.resolve().parent / r["file"])) for r in read_csv(pcsv)}
    base = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    lo, hi = (float(x) for x in a.speed_range.split(","))

    def item_overrides(it):
        p = prompts[it["prompt_id"]]
        sp = base.get("speed", 1.0)
        if a.syllable_speed and (a.syllable_scope == "all" or has_english(it["orig_text"])):
            f = speed_for(it["text"], p["transcript"], lo, hi)
            if f < 1.0:
                # never slow an item below --min-rate chars/s (the scorer fails items under 4 chars/s);
                # IndicF5's own length is clip_seconds * chars(text) / chars(transcript)
                dur = float(orig[it["prompt_id"]]["duration_s"]) * chars(it["text"]) / chars(p["transcript"])
                f = max(f, min(1.0, dur * a.min_rate / max(1, chars(it["text"]))))
            sp *= f
        if a.keep_pace:
            sp *= float(p["duration_s"]) / float(orig[it["prompt_id"]]["duration_s"])
        return {**base, "speed": round(sp, 4)}

    t0 = time.time()
    engine = get_engine()
    load_s = time.time() - t0
    if items:
        p = prompts[items[0]["prompt_id"]]
        engine("नमस्ते, यह एक छोटा सा परीक्षण है।", p["path"], p["transcript"], seed=a.seed, **base)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_item, speeds, total = {}, {}, 0.0
    for i, it in enumerate(items, 1):
        p = prompts[it["prompt_id"]]
        ov = item_overrides(it)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        y = engine(it["text"], p["path"], p["transcript"], seed=a.seed, **ov)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t
        total += dt
        save_wav(out / f"{it['id']}.wav", y)
        per_item[it["id"]], speeds[it["id"]] = round(dt, 3), ov["speed"]
        print(f"[{i}/{len(items)}] {it['id']}: speed {ov['speed']:.3f}, {dt:.2f}s synth, {len(y) / 16000:.2f}s audio", flush=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing = {"total_synth_s": round(total, 3), "n_items": len(items), "model_load_s": round(load_s, 1),
              "seed": a.seed, "overrides": base, "syllable_speed": a.syllable_speed, "syllable_scope": a.syllable_scope, "keep_pace": a.keep_pace,
              "prompts": str(pcsv), "text_csv": a.text_csv, "gpu": gpu, "python": platform.python_version(),
              "items_file": str(a.items), "per_item_s": per_item, "per_item_speed": speeds}
    tpath = out.parent / f"{out.name}_timing.json"
    tpath.write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(f"\nDone: {len(items)} items in {total:.1f}s of synthesis on {gpu}. Timing in {tpath}")


if __name__ == "__main__":
    main()
