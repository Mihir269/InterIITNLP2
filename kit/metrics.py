#!/usr/bin/env python3
"""
Development metrics for the TTS bootcamp: CER and speaker similarity (SIM) on validation audio.

    python metrics.py baseline_val/                      # score a folder of <item_id>.wav files
    python metrics.py val_idea/ --json val_idea.json --csv val_idea.csv

What it does
  * CER: Whisper large-v3-turbo transcribes each file in Hindi. Tokens containing digits, Latin
    letters or symbols such as ₹ are suppressed, so the transcript is always Hindi words in
    Devanagari (English words come out transliterated). It is compared with the `spoken`
    reference in val_items.csv after both are canonicalized (see `canon`). When a reference has
    several accepted readings, separated by " || ", the lowest CER counts. Item CER is capped at 1.
  * SIM: cosine similarity between ECAPA-TDNN embeddings of the output and its prompt clip.
  * Sanity checks: duration, share of silence, speaking rate and clipping. A missing file or one
    that fails a check is a failed item: CER 1 and SIM 0.

The final scoring uses different recognizers and a different speaker model, with the same text
rules and sanity checks. Do not over-fit to this script.
"""
import argparse, csv, json, math, re, sys, unicodedata
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
VERSION = "1.0"
ASR_MODEL = "openai/whisper-large-v3-turbo"
SPK_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# ---- sanity-check thresholds (organizers calibrate these on the baseline before release) ----
MIN_DUR_S, MAX_DUR_S = 0.5, 60.0
MAX_SILENCE_SHARE = 0.60        # share of 25 ms frames quieter than SILENCE_RMS
SILENCE_RMS = 0.003             # about -50 dBFS
MIN_CHARS_PER_S, MAX_CHARS_PER_S = 4.0, 30.0   # reference characters (no spaces) per second
MAX_CLIP_SHARE = 0.001          # share of samples at |x| >= 0.999

# --------------------------------------------------------------------------------------------
# Text rules (shared with the final scorer)
# --------------------------------------------------------------------------------------------
_PUNCT = re.compile(r"[\u0964\u0965.,!?;:\"'“”‘’()\[\]{}<>\-–—…/\\|*_~`#@&+=]")
_NASAL_CONJUNCT = re.compile(r"[ङञणनम]\u094d(?=[\u0915-\u0939])")
_FOLD = str.maketrans({
    "\u093c": None,       # nukta: ज़ -> ज
    "\u0901": "\u0902",   # chandrabindu -> anusvara
    "\u0949": "\u094b",   # ॉ -> ो   (ऑफिस ~ ओफिस)
    "\u0911": "\u0913",   # ऑ -> ओ
    "\u0945": "\u0947",   # ॅ -> े
    "\u090d": "\u090f",   # ऍ -> ए
    "\u200c": None, "\u200d": None,   # zero-width joiners
})


def canon(s: str) -> str:
    """Canonical form used for CER: NFC, fold spelling variants, drop punctuation and spaces."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = unicodedata.normalize("NFD", s).translate(_FOLD)
    s = unicodedata.normalize("NFC", s)
    s = _NASAL_CONJUNCT.sub("\u0902", s)          # हिन्दी -> हिंदी
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", "", s)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def alternatives(spoken: str):
    return [a.strip() for a in (spoken or "").split("||") if a.strip()]


def cer(spoken: str, hyp: str) -> float:
    """CER of `hyp` against the best-matching accepted reading, capped at 1."""
    h = canon(hyp)
    best = 1.0
    for ref in alternatives(spoken):
        r = canon(ref)
        if not r:
            continue
        best = min(best, _edit_distance(r, h) / len(r))
    return min(best, 1.0)


def ref_chars(spoken: str) -> int:
    alts = alternatives(spoken)
    return min(len(canon(a)) for a in alts) if alts else 0


# --------------------------------------------------------------------------------------------
# Audio sanity checks (shared with the final scorer)
# --------------------------------------------------------------------------------------------
def sanity(y: np.ndarray, sr: int, n_ref_chars: int):
    """Return (ok, reasons, stats) for one output."""
    reasons, stats = [], {}
    dur = len(y) / sr if sr else 0.0
    stats["dur_s"] = round(dur, 3)
    if not (MIN_DUR_S <= dur <= MAX_DUR_S):
        reasons.append(f"duration {dur:.2f}s outside [{MIN_DUR_S}, {MAX_DUR_S}]")
    if len(y) == 0:
        return False, reasons or ["empty"], stats
    hop = max(1, int(0.025 * sr))
    n = len(y) // hop
    frames = y[: n * hop].reshape(n, hop) if n else y.reshape(1, -1)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    silence = float((rms < SILENCE_RMS).mean())
    stats["silence_share"] = round(silence, 3)
    if silence > MAX_SILENCE_SHARE:
        reasons.append(f"silence share {silence:.2f} > {MAX_SILENCE_SHARE}")
    rate = n_ref_chars / dur if dur > 0 else math.inf
    stats["chars_per_s"] = round(rate, 2)
    if not (MIN_CHARS_PER_S <= rate <= MAX_CHARS_PER_S):
        reasons.append(f"speaking rate {rate:.1f} chars/s outside [{MIN_CHARS_PER_S}, {MAX_CHARS_PER_S}]")
    clip = float((np.abs(y) >= 0.999).mean())
    stats["clip_share"] = round(clip, 5)
    if clip > MAX_CLIP_SHARE:
        reasons.append(f"clipping share {clip:.4f} > {MAX_CLIP_SHARE}")
    return not reasons, reasons, stats


# --------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_items(items_csv):
    rows = read_csv(items_csv)
    if rows and "spoken" not in rows[0]:
        sys.exit(f"{items_csv} has no `spoken` column; metrics.py scores validation items only.")
    return rows


def load_prompts(prompts_csv):
    base = Path(prompts_csv).resolve().parent
    return {r["prompt_id"]: dict(r, path=str(base / r["file"])) for r in read_csv(prompts_csv)}


def load_audio(path, target_sr=16000):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), sr


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def whisper_suppress_ids(tokenizer, generation_config=None):
    """Token ids whose text contains digits, Latin letters or symbols, plus Whisper's defaults."""
    bad = re.compile(r"[0-9A-Za-z\u0966-\u096f₹%$€£]")
    special = set(tokenizer.all_special_ids)
    first_special = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ids = [i for i in range(first_special) if i not in special and bad.search(tokenizer.decode([i]))]
    default = list(getattr(generation_config, "suppress_tokens", None) or [])
    return sorted(set(ids) | set(default))


def load_whisper(model_id=ASR_MODEL):
    import torch
    from transformers import pipeline
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    try:
        asr = pipeline("automatic-speech-recognition", model=model_id, dtype=dtype, device=dev)
    except TypeError:
        asr = pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=dev)
    asr._bootcamp_suppress = whisper_suppress_ids(asr.tokenizer, asr.model.generation_config)
    return asr


def transcribe_whisper(asr, waves, batch_size=8):
    kw = {"language": "hindi", "task": "transcribe", "suppress_tokens": asr._bootcamp_suppress}
    inputs = [{"raw": y, "sampling_rate": 16000} for y in waves]
    out = asr(inputs, batch_size=batch_size, chunk_length_s=30, generate_kwargs=kw)
    return [o["text"].strip() for o in out]


def load_ecapa(source=SPK_MODEL):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    savedir = Path.home() / ".cache" / "bootcamp_ecapa"
    return EncoderClassifier.from_hparams(source=source, savedir=str(savedir), run_opts={"device": device()})


def ecapa_embed(model, y):
    import torch
    with torch.no_grad():
        e = model.encode_batch(torch.from_numpy(y).unsqueeze(0).to(model.device))
    e = e.squeeze().float().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


class Models:
    """Kit models, loaded once."""
    def __init__(self):
        print(f"Loading {ASR_MODEL} and {SPK_MODEL} on {device()} ...", file=sys.stderr)
        self.asr = load_whisper()
        self.spk = load_ecapa()

    def transcribe(self, waves):
        return transcribe_whisper(self.asr, waves)

    def embed(self, y):
        return ecapa_embed(self.spk, y)


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------
def evaluate_folder(audio_dir, items, prompts, models):
    audio_dir = Path(audio_dir)
    rows, waves, idx = [], [], []
    for it in items:
        r = {"id": it["id"], "category": it["category"], "failed": False, "reasons": "", "hyp": ""}
        path = audio_dir / f"{it['id']}.wav"
        if not path.exists():
            r.update(failed=True, reasons="missing")
            rows.append(r)
            continue
        try:
            y, orig_sr = load_audio(path)
        except Exception as e:
            r.update(failed=True, reasons=f"unreadable: {e}")
            rows.append(r)
            continue
        ok, reasons, stats = sanity(y, 16000, ref_chars(it["spoken"]))
        r.update(stats)
        if orig_sr != 16000:
            reasons.append(f"sample rate {orig_sr} (must be 16000)")
            ok = False
        if not ok:
            r.update(failed=True, reasons="; ".join(reasons))
        rows.append(r)
        waves.append(y)
        idx.append(len(rows) - 1)

    hyps = models.transcribe(waves) if waves else []
    prompt_emb = {}
    for k, y in zip(idx, waves):
        r, it = rows[k], items[k]
        r["hyp"] = hyps[idx.index(k)]
        pid = it["prompt_id"]
        if pid not in prompt_emb:
            prompt_emb[pid] = models.embed(load_audio(prompts[pid]["path"])[0])
        r["cer_raw"] = cer(it["spoken"], r["hyp"])
        r["sim_raw"] = float(np.dot(models.embed(y), prompt_emb[pid]))
    for r in rows:
        r["cer"] = 1.0 if r["failed"] else r.get("cer_raw", 1.0)
        r["sim"] = 0.0 if r["failed"] else r.get("sim_raw", 0.0)
    return rows


def summarize(rows):
    cats = sorted({r["category"] for r in rows})
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    return {
        "metrics_version": VERSION,
        "recognizer": ASR_MODEL,
        "speaker_model": SPK_MODEL,
        "n_items": len(rows),
        "failed": sum(r["failed"] for r in rows),
        "cer": mean([r["cer"] for r in rows]),
        "sim": mean([r["sim"] for r in rows]),
        "cer_by_category": {c: mean([r["cer"] for r in rows if r["category"] == c]) for c in cats},
        "sim_by_category": {c: mean([r["sim"] for r in rows if r["category"] == c]) for c in cats},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir", help="folder with <item_id>.wav for every validation item")
    ap.add_argument("--items", default=str(KIT / "data" / "val_items.csv"))
    ap.add_argument("--prompts", default=str(KIT / "data" / "prompts.csv"))
    ap.add_argument("--json", help="write the summary here")
    ap.add_argument("--csv", help="write per-item scores here")
    ap.add_argument("--threshold", type=float, default=0.20, help="list items above this CER (Task 1)")
    a = ap.parse_args()

    items = load_items(a.items)
    prompts = load_prompts(a.prompts)
    rows = evaluate_folder(a.audio_dir, items, prompts, Models())
    summary = summarize(rows)

    print(f"\n{'id':<12}{'cat':<10}{'CER':>7}{'SIM':>7}  transcript / problem")
    for r in rows:
        note = r["reasons"] if r["failed"] else r["hyp"]
        print(f"{r['id']:<12}{r['category']:<10}{r['cer']:>7.3f}{r['sim']:>7.3f}  {note[:90]}")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    above = [r["id"] for r in rows if r["cer"] > a.threshold]
    print(f"\nItems with CER > {a.threshold} (listen to these for Task 1): {', '.join(above) or 'none'}")

    if a.json:
        Path(a.json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.csv:
        keys = ["id", "category", "cer", "sim", "failed", "reasons", "dur_s", "silence_share",
                "chars_per_s", "clip_share", "hyp"]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
