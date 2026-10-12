"""
Length fix for IndicF5 / F5-TTS.

Upstream F5-TTS estimates how long the output should be from UTF-8 *byte* counts of the text.
Devanagari takes about 3 bytes per character and Latin 1, so English words get roughly a third
of the time they need and are cut off. This rewrites those counts to non-space character counts
in the installed code. It is idempotent and must run before `f5_tts` is imported.
"""
import importlib.util, re, sys
from pathlib import Path

PATTERN = re.compile(
    r"""len\(\s*([A-Za-z_]\w*(?:\(\))?(?:\.[A-Za-z_]\w*(?:\(\))?)*)\.encode\(\s*["']utf-?8["']\s*\)\s*\)""")
HF_MODULES = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"


def _patch_tree(root):
    changed = []
    for p in Path(root).rglob("*.py"):
        try:
            src = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n = PATTERN.subn(lambda m: f'len("".join({m.group(1)}.split()))', src)
        if n:
            p.write_text(new, encoding="utf-8")
            changed.append((str(p), n))
    return changed


def ensure_patched():
    """Patch the installed f5_tts package. Raise if it was already imported unpatched."""
    spec = importlib.util.find_spec("f5_tts")
    if spec is None or spec.origin is None:
        raise ImportError("f5_tts not installed: pip install -r baseline/requirements.txt")
    changed = _patch_tree(Path(spec.origin).parent)
    if changed and "f5_tts" in sys.modules:
        raise RuntimeError("f5_tts was imported before the length fix was applied. "
                           "Restart the Python session (Colab: Runtime > Restart session) and run again.")
    return changed


def patch_hf_modules():
    """Patch model code that transformers downloaded with trust_remote_code."""
    return _patch_tree(HF_MODULES) if HF_MODULES.exists() else []


if __name__ == "__main__":
    print("f5_tts:", ensure_patched() or "already patched")
    print("remote code:", patch_hf_modules() or "nothing to patch")
