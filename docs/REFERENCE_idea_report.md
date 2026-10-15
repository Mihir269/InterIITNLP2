# Reference notes for writing idea.md and report.md

Not for submission. Notes only – turn them into your own sentences (rules: idea.md and report.md
must be in your own words). Lives in `docs/`, so `make_zip.py` does not pack it.

All numbers: validation set, seed 0, kit `metrics.py` (Whisper-large-v3-turbo + ECAPA), Colab T4.

---

## A. Fact sheet (copy numbers from here, not the wording)

| System | CER | Hindi | Hinglish | Hard | Long | SIM |
|---|---|---|---|---|---|---|
| Baseline (Task 1) | 0.242 | 0.034 | 0.302 | 0.458 | 0.032 | 0.830 |
| Task 2 normalizer only (`val_control`) | 0.122 | 0.034 | 0.302 | 0.099 | 0.032 | 0.837 |
| Rules only (translit, before syllable timing) | 0.059 | 0.034 | 0.047 | 0.099 | 0.032 | 0.841 |
| **Final Task 3** (`val_idea`) | **0.053** | 0.034 | **0.026** | 0.099 | 0.032 | **0.841** |
| Task 4 (12 steps) | 0.053 | 0.030 | 0.041 | 0.087 | 0.038 | 0.844 |

- Hinglish relative reduction: (0.302 − 0.026) / 0.302 = **91.5%** (rules alone: 84.5%)
- Task 4: 616.9 s → 245.6 s = **2.51×**; CER 0.053 ≤ 1.2 × 0.053
- Guards: overall CER improved (not >10% worse); SIM +0.011 (not −0.03)

**Evidence for the hypothesis**
- prompt transcripts write English in Devanagari: ऑडर (V1), मार्केट (V3), टाइम/वेट/पिज़्ज़ा (E1)
- `vocab.txt`: 2,546 tokens, **all 52 Latin letters present**, 104 Devanagari → Latin is known but
  under-trained (not missing)
- baseline val_hg_01: office/internet/slow → "अजसे … सर्ण … कोई"; Hindi words in same sentence correct
- Hinglish items had the right length (input/spoken ≈ 1.0–1.1) → not a timing failure
- rules: 82% exact on val words, 78% on 107 other loanwords (generalize; a lookup table would not)

**Length mechanism (for the syllable part)**
- IndicF5 length = prompt sec × chars(new) / chars(transcript) ÷ speed
- transliterated English 2.7 chars/syllable vs Hindi 2.0 → too much time (e.g. स्लो: 4 chars, 1 syllable)
- fix: speed = (C_new/C_ref) ÷ (S_new/S_ref), clamp 0.8–1.25, only if sentence has English words
- example val_hg_01 (V3): 2.55 s → 2.23 s (speed 1.143); actual 2.21 s

**Robustness / negative results**
- syllable timing on ALL items: Hinglish better but Hindi 0.034→0.046, long 0.032→0.060 → restricted
- seeds 1, 2 on Hinglish: 0.025→0.025, 0.030→0.029 (never worse; mean 0.034→0.027); seed noise ≈ effect
- 4 spelling variants + recognizer pick: 0.047→0.043, 4× compute, fits kit ASR → dropped
- pause-trimmed prompts: SIM up (0.850) but CER worse → dropped
- commas in abbreviations (Task 2 side): mean 0.105→0.103, within noise → dropped
- 10 steps: CER 0.067 > 1.2× → fails; 8 steps 0.098; CFG off 0.22

**Per item, final Task 3 vs baseline (Q1)**
- no item worse; Hindi + long unchanged (not touched)
- largest: val_hd_08 0.603→0.064, val_hd_01 0.587→0.127, val_hg_01 0.452→0.032, val_hg_06 0.318→0.000
- smallest Hinglish: val_hg_02 0.216→0.054 (address→अड्रेस, message→मेसेज: vowel rules),
  val_hg_04 0.263→0.026
- still weak: val_hd_03 0.163 (V3 fast speaker; छब्बीस→चब्विस, घोषणा→गोशना, aspiration lost)

---

## B. idea.md (≤ 200 words total, four labelled parts)

Format: lines/headings with `(1)`, `(2)`, `(3)`, `(4)`; check with `check_format.py`.
**Parts 1–3 must match your Day-3 form exactly** if you submitted it. Suggested budget 35/60/40/60.

**(1) Failure** – notes
- tag: english_word_error; category: hinglish
- 2 item IDs: val_hg_01, val_hg_06 (optionally quote one garbled word)
- size: Hinglish CER 0.302 vs Hindi 0.034

**(2) Hypothesis + method** – notes
- why: model trained on transcripts with English in Devanagari; Latin letters in vocab but
  under-trained → poor letter-to-sound for Latin
- what: rewrite English words in Devanagari by general Indian-English spelling rules
  (letter groups sh/ee/igh, open vs closed syllable, silent final e, r after vowel, -tion)
- not a lookup table → works on unseen eval words; rule-based, no model
- (only if it was on your Day-3 form) syllable-based timing

**(3) Prediction** – notes
- which category improves: hinglish; rough size you predicted (keep what you wrote on Day 3)
- risks: wrong vowels on irregular spellings; gain may not carry to other recognizers; SIM change;
  other categories unaffected

**(4) Result + lesson** – notes
- rules alone: 0.302 → 0.047 (−84.5%)
- added after Day 3: length follows characters; English spelled in Devanagari is char-dense
  → syllable-based speed for English sentences → 0.026 (−91.5%); overall 0.053; SIM 0.841
- honesty: applying it everywhere hurt Hindi/long; seed noise ≈ second gain (seeds 1–2 ~no change)
- lesson: failure is about script/representation + length estimate, not "English sounds" as such

Checklist: item IDs ✓ numbers ✓ mechanism ✓ prediction vs result kept separate ✓ ≤200 words ✓

---

## C. report.md (Q1, Q2, Q3 – each ≤ 75 words; line starts with Q1/Q2/Q3; no question text)

**Q1 – two items better, two worse (or smallest gains), and why** – notes
- none worse → give two smallest gains (say so explicitly)
- better (pick 2): val_hg_01 0.452→0.032 (English words now in Devanagari, said correctly);
  val_hd_08 0.603→0.064 (numbers/dates/₹ written out, length now right)
- smallest (pick 2): val_hg_02 0.216→0.054 (vowel rule: address→अड्रेस); Hindi items 0 gain
  (e.g. val_hi_04 0.100→0.100: not touched by the method, model slip on धैर्य)

**Q2 – what it taught you about why the model fails; one thing to do differently** – notes
- model's length = characters × prompt pace; wrong when chars ≠ spoken length (digits too few,
  Devanagari English too many)
- pronunciation failure = script it was trained on, not inability to say the sounds
- seed noise as large as small gains (0.025–0.047 Hinglish across seeds)
- do differently (pick 1): test several seeds from the start / measure rules on held-out word list
  before audio / listen early

**Q3 – single next change, and why** – notes (pick ONE)
- pause allowance at commas/। (prompt transcripts have no punctuation; length rule ignores pauses)
- per-segment synthesis: English stretches with their own speed, joined ≤0.3 s (one speed per call
  is a limit we hit)
- calmer prompt segment / rate clamp for very fast or very slow prompt clips (V3 slips, E2 slow)
- correct prompt transcripts (they guide alignment)

Checklist: each answer ≤ 75 words ✓ starts with Q1/Q2/Q3 ✓ item IDs + numbers in Q1 ✓
