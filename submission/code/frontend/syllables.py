"""
Rule-based Devanagari syllable counter (no model), used to estimate how long text takes to say.

A syllable is a vowel nucleus: an independent vowel, a vowel sign (matra), or a consonant that
carries the inherent vowel. Hindi drops the inherent vowel at the end of a word (राम = रा-म्),
before a halant, and before a matra-less consonant + vowel sign in the middle of a word
(schwa deletion, approximated: VCaCV -> VCCV, e.g. समझना = सम-झ-ना).
"""
import re

CONS = "क-हक़-य़"
INDEP_VOWELS = "ऄ-औॠॡ"
MATRAS = "ा-ौॢॣ"
HALANT = "्"
NUKTA = "़"
MARKS = "ऀ-ः"  # candrabindu, anusvara, visarga: not nuclei


def word_syllables(w: str) -> int:
    w = w.replace(NUKTA, "")
    units = []  # one entry per consonant/vowel: 'V' full vowel, 'M' consonant+matra, 'a' consonant+schwa, 'C' consonant+halant
    i = 0
    while i < len(w):
        ch = w[i]
        if re.match(f"[{INDEP_VOWELS}]", ch):
            units.append("V")
        elif re.match(f"[{CONS}]", ch):
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if nxt == HALANT:
                units.append("C"); i += 1
            elif nxt and re.match(f"[{MATRAS}]", nxt):
                units.append("M"); i += 1
            else:
                units.append("a")
        i += 1
    # word-final schwa deletion (not for one-letter words like न, व)
    if len(units) > 1 and units[-1] == "a":
        units[-1] = "C"
    # medial schwa deletion: (V|M) a (M|V-like) -> drop the schwa
    for k in range(1, len(units) - 1):
        if units[k] == "a" and units[k - 1] in ("V", "M", "a") and units[k + 1] in ("M",) and k >= 2:
            units[k] = "C"
    return sum(u in ("V", "M", "a") for u in units)


def syllables(text: str) -> int:
    return sum(word_syllables(w) for w in re.findall(f"[{CONS}{INDEP_VOWELS}{MATRAS}{HALANT}{NUKTA}{MARKS}]+", text))


def chars(text: str) -> int:
    """The quantity IndicF5 uses (after the kit's length fix): non-space characters."""
    return len("".join(text.split()))


def speed_for(gen_text: str, ref_text: str, lo=0.8, hi=1.25) -> float:
    """`speed` that makes IndicF5's character-based length equal a syllable-based length.

    IndicF5: gen_frames = ref_frames * chars(gen)/chars(ref) / speed.
    Wanted:  gen_frames = ref_frames * syl(gen)/syl(ref).
    """
    s_gen, s_ref = syllables(gen_text), syllables(ref_text)
    if not s_gen or not s_ref:
        return 1.0
    sp = (chars(gen_text) / chars(ref_text)) / (s_gen / s_ref)
    return float(min(hi, max(lo, sp)))


PAUSE_MARKS = re.compile(r"[,।?!.;:](?=\s*\S)")   # punctuation followed by more text (not the final mark)


def pauses(text: str) -> int:
    """Number of sentence-internal punctuation marks, i.e. places where a speaker pauses."""
    return len(PAUSE_MARKS.findall(text))


def pause_speed(gen_text: str, ref_text: str, k_syl: float, syllable_based: bool,
                lo=0.8, hi=1.25) -> float:
    """`speed` that adds a pause allowance of k_syl syllables per internal punctuation mark.

    The prompt transcripts have no punctuation, so the prompt's rate already includes its average
    pausing; this gives extra time only where the new text marks a pause. Lengths are expressed in
    'prompt characters' so it works on top of either the character-based (default) or the
    syllable-based length.
    """
    c_gen, c_ref = chars(gen_text), chars(ref_text)
    s_gen, s_ref = syllables(gen_text), syllables(ref_text)
    if not c_gen or not s_ref:
        return 1.0
    chars_per_syl = c_ref / s_ref
    if syllable_based:
        units = min(c_gen / lo, max(c_gen / hi, s_gen * chars_per_syl))   # same clamp as speed_for
    else:
        units = c_gen
    units += k_syl * pauses(gen_text) * chars_per_syl
    return float(c_gen / units)
