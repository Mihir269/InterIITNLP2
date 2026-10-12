#!/usr/bin/env python3
"""Pack <roll>_tts_bootcamp.zip from the submission folder (code/, predictions/, *.md, results.json)."""
import argparse, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP_TOP = ["idea.md", "report.md", "results.json"]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True)
    ap.add_argument("--sub", default=str(HERE.parent))
    a = ap.parse_args()
    sub = Path(a.sub)
    out = sub / f"{a.roll}_tts_bootcamp.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in KEEP_TOP:
            if (sub / name).exists():
                z.write(sub / name, name)
        for d in ("code", "predictions"):
            for p in sorted((sub / d).rglob("*")):
                if p.is_file() and not (set(p.parts) & SKIP_DIRS):
                    z.write(p, p.relative_to(sub).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
