#!/usr/bin/env python3
"""
Check a TTS bootcamp submission before you upload it.

    python check_format.py 21CS10042_tts_bootcamp.zip
    python check_format.py my_submission_folder/

Checks names, formats, counts and word limits only, not quality. Every file is reported as
OK, MISSING (not scored; fine if you skipped that task) or FAIL (scored 0). Exit code 1 means
at least one FAIL. Run it before every upload.
"""
import argparse, csv, json, re, sys, tempfile, zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
MAX_ZIP_MB = 100
ZIP_NAME = re.compile(r"^[A-Za-z0-9]+_tts_bootcamp\.zip$")
FAILURE_TAGS = {"skipped_words", "repeated_words", "number_error", "english_word_error",
                "speaker_mismatch", "other"}
ITEM_CATEGORIES = {"hindi", "hinglish", "hard", "long"}
EVAL_AUDIO = ["t2_norm", "t3_idea", "t4_fast"]
VAL_AUDIO = ["val_idea", "val_control"]


class Report:
    def __init__(self):
        self.lines, self.failed = [], False

    def add(self, name, status, msgs=()):
        self.failed |= status == "FAIL"
        self.lines.append((name, status, list(msgs)))

    def show(self):
        for name, status, msgs in self.lines:
            print(f"{status:<8} {name}")
            for m in msgs[:12]:
                print(f"           - {m}")
            if len(msgs) > 12:
                print(f"           - ... and {len(msgs) - 12} more")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        rows = list(r)
    return (rows[0] if rows else []), rows[1:]


def words(text):
    return len(re.findall(r"\S+", re.sub(r"[#*_>`|-]+", " ", text)))


def find_root(path: Path):
    """The submission root is the folder that holds predictions/ or idea.md."""
    for cand in [path] + sorted(p for p in path.rglob("*") if p.is_dir()):
        if (cand / "predictions").is_dir() or (cand / "idea.md").exists() or (cand / "results.json").exists():
            return cand
    return path


def load_ids(items_csv):
    header, rows = read_csv(items_csv)
    i = header.index("id")
    return [r[i] for r in rows if r]


def check_audio_dir(rep, d: Path, expected_ids, label):
    if not d.exists():
        rep.add(label, "MISSING")
        return
    import numpy as np
    import soundfile as sf
    msgs, warn = [], []
    files = [p for p in d.iterdir() if not p.name.startswith(".")]
    names = {p.name for p in files}
    want = {f"{i}.wav" for i in expected_ids}
    if missing := sorted(want - names):
        msgs.append(f"missing {len(missing)} file(s): {', '.join(missing[:6])}")
    if extra := sorted(names - want):
        msgs.append(f"unexpected file(s): {', '.join(extra[:6])}")
    for name in sorted(want & names):
        p = d / name
        try:
            info = sf.info(str(p))
        except Exception as e:
            msgs.append(f"{name}: not a readable wav ({e})")
            continue
        problems = []
        if info.format != "WAV":
            problems.append(f"format {info.format}")
        if info.channels != 1:
            problems.append(f"{info.channels} channels")
        if info.samplerate != 16000:
            problems.append(f"{info.samplerate} Hz")
        if info.subtype != "PCM_16":
            problems.append(f"subtype {info.subtype}")
        if not (0.5 <= info.duration <= 60):
            problems.append(f"duration {info.duration:.2f}s")
        if problems:
            msgs.append(f"{name}: " + ", ".join(problems) + " (need mono 16 kHz PCM_16, 0.5-60 s)")
            continue
        y, _ = sf.read(str(p), dtype="float32")
        if y.size == 0 or float(np.abs(y).max()) < 1e-3:
            warn.append(f"{name}: silent (will count as a failed item)")
    status = "FAIL" if msgs else "OK"
    rep.add(f"{label} ({len(want)} files)", status, msgs + [f"warning: {w}" for w in warn])


def check(sub: Path, zip_path=None):
    rep = Report()
    val_ids = load_ids(KIT / "data" / "val_items.csv")
    eval_ids = load_ids(KIT / "data" / "eval_items.csv")
    root = find_root(sub)
    pred = root / "predictions"

    if zip_path:
        msgs = []
        if not ZIP_NAME.match(zip_path.name):
            msgs.append(f"name {zip_path.name!r} should be <rollno>_tts_bootcamp.zip")
        mb = zip_path.stat().st_size / 1e6
        if mb > MAX_ZIP_MB:
            msgs.append(f"{mb:.1f} MB > {MAX_ZIP_MB} MB")
        rep.add("zip file", "FAIL" if msgs else "OK", msgs)

    # code/README.md
    readme = root / "code" / "README.md"
    if not readme.exists():
        rep.add("code/README.md", "MISSING", ["the README with exact commands is worth 8 points"])
    else:
        n = len(readme.read_text(encoding="utf-8", errors="replace").strip())
        rep.add("code/README.md", "OK" if n else "FAIL", [] if n else ["empty"])

    # results.json
    res_path = root / "results.json"
    results = {}
    if not res_path.exists():
        rep.add("results.json", "MISSING")
    else:
        msgs = []
        try:
            results = json.loads(res_path.read_text(encoding="utf-8"))
            template = json.loads((KIT / "results_template.json").read_text(encoding="utf-8"))
            for sec, val in template.items():
                if sec.startswith("_"):
                    continue
                if sec not in results:
                    msgs.append(f"missing key {sec!r}")
                elif isinstance(val, dict):
                    for k in val:
                        if k not in results[sec]:
                            msgs.append(f"missing key {sec}.{k}")
                    for k, v in results[sec].items():
                        if k in val and val[k] is None and v is not None and not isinstance(v, (int, float)):
                            msgs.append(f"{sec}.{k} must be a number or null")
            t3 = results.get("task3", {})
            if t3.get("targeted_failure") and t3["targeted_failure"] not in FAILURE_TAGS:
                msgs.append(f"task3.targeted_failure must be one of {sorted(FAILURE_TAGS)}")
            if t3.get("targeted_category") and t3["targeted_category"] not in ITEM_CATEGORIES:
                msgs.append(f"task3.targeted_category must be one of {sorted(ITEM_CATEGORIES)}")
            if (pred / "t3_idea").exists() and not t3.get("targeted_category"):
                msgs.append("t3_idea/ submitted but task3.targeted_category is empty")
        except json.JSONDecodeError as e:
            msgs.append(f"invalid JSON: {e}")
        rep.add("results.json", "FAIL" if msgs else "OK", msgs)

    # idea.md
    idea = root / "idea.md"
    if not idea.exists():
        rep.add("idea.md", "MISSING")
    else:
        text = idea.read_text(encoding="utf-8", errors="replace")
        n = words(text)
        msgs = [f"{n} words > 200"] if n > 200 else []
        warn = [f"warning: no part ({k}) found" for k in "1234" if not re.search(rf"\(\s*{k}\s*\)|^\s*{k}[.)]", text, re.M)]
        rep.add(f"idea.md ({n} words)", "FAIL" if msgs else "OK", msgs + warn)

    # report.md
    report = root / "report.md"
    if not report.exists():
        rep.add("report.md", "MISSING")
    else:
        text = report.read_text(encoding="utf-8", errors="replace")
        parts = re.split(r"(?im)^[#*>\s]*(Q[123])\b[\s.:)*-]*", text)
        answers = {parts[i].upper(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
        msgs = []
        for q in ("Q1", "Q2", "Q3"):
            if q not in answers:
                msgs.append(f"no {q} heading found (start each answer with a line beginning Q1, Q2, Q3)")
            elif words(answers[q]) > 75:
                msgs.append(f"{q}: {words(answers[q])} words > 75")
        rep.add("report.md", "FAIL" if msgs else "OK", msgs)

    # t1_val_errors.csv
    p = pred / "t1_val_errors.csv"
    if not p.exists():
        rep.add("predictions/t1_val_errors.csv", "MISSING")
    else:
        header, rows = read_csv(p)
        msgs = []
        if header != ["item_id", "category"]:
            msgs.append(f"header must be item_id,category (got {','.join(header)})")
        else:
            seen = set()
            for k, r in enumerate(rows, 2):
                if not r or not any(r):
                    continue
                if len(r) != 2:
                    msgs.append(f"line {k}: expected 2 columns")
                    continue
                iid, cat = r
                if iid not in val_ids:
                    msgs.append(f"line {k}: {iid!r} is not a validation item id")
                if cat not in FAILURE_TAGS:
                    msgs.append(f"line {k}: category {cat!r} not allowed")
                if iid in seen:
                    msgs.append(f"line {k}: {iid} tagged twice")
                seen.add(iid)
            if not seen:
                msgs.append("no rows")
        rep.add("predictions/t1_val_errors.csv", "FAIL" if msgs else "OK", msgs)

    # t2_normalized.csv
    p = pred / "t2_normalized.csv"
    if not p.exists():
        rep.add("predictions/t2_normalized.csv", "MISSING")
    else:
        header, rows = read_csv(p)
        msgs, warn = [], []
        if header != ["id", "split", "text", "normalized_text"]:
            msgs.append(f"header must be id,split,text,normalized_text (got {','.join(header)})")
        else:
            want = {i: "val" for i in val_ids} | {i: "eval" for i in eval_ids}
            got = {}
            for k, r in enumerate(rows, 2):
                if not r or not any(r):
                    continue
                if len(r) != 4:
                    msgs.append(f"line {k}: expected 4 columns (quote fields that contain commas)")
                    continue
                iid, split, _, norm = r
                if iid in got:
                    msgs.append(f"line {k}: duplicate id {iid}")
                got[iid] = split
                if iid in want and split != want[iid]:
                    msgs.append(f"line {k}: {iid} should have split={want[iid]}")
                if not norm.strip():
                    msgs.append(f"line {k}: empty normalized_text")
                elif re.search(r"[0-9\u0966-\u096f₹%]", norm):
                    warn.append(f"{iid}: digits, ₹ or % left in normalized_text")
            if missing := sorted(set(want) - set(got)):
                msgs.append(f"missing {len(missing)} id(s): {', '.join(missing[:6])}")
            if extra := sorted(set(got) - set(want)):
                msgs.append(f"unknown id(s): {', '.join(extra[:6])}")
        rep.add("predictions/t2_normalized.csv (64 rows)", "FAIL" if msgs else "OK",
                msgs + [f"warning: {w}" for w in warn])

    for d in EVAL_AUDIO:
        check_audio_dir(rep, pred / d, eval_ids, f"predictions/{d}/")
    for d in VAL_AUDIO:
        check_audio_dir(rep, pred / d, val_ids, f"predictions/{d}/")

    if (pred / "t3_idea").exists() and not (pred / "val_idea").exists():
        rep.add("task 3 pairing", "FAIL", ["t3_idea/ needs val_idea/ (24 validation files)"])
    if results.get("task3", {}).get("builds_on_task2") and not (pred / "val_control").exists():
        rep.add("task 3 control", "FAIL", ["builds_on_task2 is true, so val_control/ is required"])
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", help="the .zip you will upload, or its unzipped folder")
    a = ap.parse_args()
    path = Path(a.submission)
    if not path.exists():
        sys.exit(f"{path} not found")
    if path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as z:
                bad = [n for n in z.namelist() if n.startswith("/") or ".." in Path(n).parts]
                if bad:
                    sys.exit(f"unsafe paths in zip: {bad[:3]}")
                z.extractall(tmp)
            rep = check(Path(tmp), zip_path=path)
    else:
        rep = check(path)
    rep.show()
    print("\nRESULT:", "FAIL - fix the items marked FAIL before uploading" if rep.failed
          else "OK - every file present passes (MISSING files are simply not scored)")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
