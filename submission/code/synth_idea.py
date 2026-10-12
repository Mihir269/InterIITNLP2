#!/usr/bin/env python3
"""
Task 3 system with spelling-variant candidates (optional part of the idea).

For an item whose text had English words, the transliteration rules are least sure about a few
vowel choices (ऑ vs ओ, ऐ vs ए). This script synthesizes up to N spellings of the same sentence
(same seed, baseline decoding) and keeps the one whose recognizer transcript best matches the
spelling it was asked to say, i.e. the rendering the model pronounced most clearly. A candidate
whose speaker similarity is more than --sim-margin below the best is never chosen. Items without
English words get one candidate, so they are identical to the plain idea system.

    python code/synth_idea.py --kit ../kit --items ../kit/data/val_items.csv --out predictions/val_idea --variants 4
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frontend"))
from normalize import normalize  # noqa: E402
from translit import translit  # noqa: E402

# (from, to) swaps applied to the transliterated English words only
SWAPS = [("ॉ", "ो"), ("ै", "े")]


def variant_texts(text, n):
    """Return up to n distinct model inputs: the base rendering, then vowel-swapped spellings."""
    norm = normalize(text)
    words = re.findall(r"[A-Za-z]+", norm)
    base = re.sub(r"[A-Za-z]+", lambda m: translit(m.group(0)), norm)
    if not words or n <= 1:
        return [base]
    out = [base]
    combos = [[0], [1], [0, 1]]
    for combo in combos:
        def sub(m):
            w = translit(m.group(0))
            for i in combo:
                a, b = SWAPS[i]
                w = w.replace(a, "\0").replace(b, a).replace("\0", b)
            return w
        t = re.sub(r"[A-Za-z]+", sub, norm)
        if t not in out:
            out.append(t)
        if len(out) >= n:
            break
    return out


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default=str(HERE.parent.parent / "kit"))
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--sim-margin", type=float, default=0.03)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    kit = Path(a.kit).resolve()
    sys.path.insert(0, str(kit))
    from baseline.tts import get_engine, save_wav
    import metrics as M

    overrides = {k: float(v) if "." in v else int(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    prompts = {r["prompt_id"]: dict(r, path=str(kit / "data" / r["file"])) for r in read_csv(kit / "data" / "prompts.csv")}
    items = read_csv(a.items)
    engine = get_engine()
    models = M.Models()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    pemb = {}
    for it in items:
        p = prompts[it["prompt_id"]]
        cands = variant_texts(it["text"], a.variants)
        waves = [engine(t, p["path"], p["transcript"], seed=a.seed, **overrides) for t in cands]
        best = 0
        if len(waves) > 1:
            if p["prompt_id"] not in pemb:
                pemb[p["prompt_id"]] = models.embed(M.load_audio(p["path"])[0])
            hyps = models.transcribe(waves)
            sims = [float(models.embed(w) @ pemb[p["prompt_id"]]) for w in waves]
            cers = [M.cer(t, h) for t, h in zip(cands, hyps)]
            ok = [i for i in range(len(waves)) if sims[i] >= max(sims) - a.sim_margin]
            best = min(ok, key=lambda i: (cers[i], -sims[i]))
            log.append({"id": it["id"], "cands": cands, "cer_self": cers, "sim": sims, "chosen": best})
        save_wav(out / f"{it['id']}.wav", waves[best])
        print(f"{it['id']}: {len(cands)} candidate(s), chose {best}", flush=True)
    logp = HERE / "results" / f"{out.name}_candidates.json"
    logp.parent.mkdir(exist_ok=True)
    logp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s; selection log in {logp}")


if __name__ == "__main__":
    main()
