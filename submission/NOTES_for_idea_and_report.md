# Facts for writing idea.md and report.md (write those two files in your own words)

The rules require idea.md and report.md to be in your own words. This page only collects the
evidence. Delete it before zipping (make_zip.py does not include it).

## Task 1 – baseline on validation (metrics.py, CPU)
- CER 0.2426, SIM 0.8303 (published: 0.2421 / 0.8303). By category CER: hard 0.458, hinglish 0.302, hindi 0.034, long 0.034.
- 14 items above CER 0.20: all 6 hinglish (english_word_error), all 8 hard (number_error).
- Example transcripts: val_hg_01 "office/internet/slow" -> "अजसे / सर्ण / कोई"; val_hd_01 "12/03/2026 ₹1,25,000" -> "पेड़म शोर एन्यो उसन जाइट".

## EDA findings
- Output duration = prompt speaking rate × number of input characters (input-rate = prompt-rate for every item, within ~5%).
- Hard items: the input (digits) is 54-83% as long as what must be said, so the model gets too little time:
  effective rate 10-17 chars/s vs the speaker's 9-12.5. corr(CER, input/spoken length ratio) = -0.72.
- Hinglish: length is fine (ratio 1.0-1.1) yet CER is 0.30 -> the failure is pronouncing Latin script, not timing.
  The prompt transcripts write English loanwords in Devanagari (ऑडर, मार्केट, टाइम, वेट, पिज़्ज़ा), as does
  the IndicVoices-R training data, so Latin script is out of distribution for IndicF5.
- Prompt V3 speaks fastest (12.6 chars/s) and has the highest per-prompt CER (0.30); V4 has the lowest SIM (0.77).
- Every baseline output starts with 0.2-0.8 s of silence (allowed to trim).

## Text-level check (model input vs spoken reference, CER)
| category | raw text | Task 2 normalized | + transliteration |
|---|---|---|---|
| hard | 0.460 | 0.000 | 0.000 |
| hinglish | 0.419 | 0.419 | ~0.03 |

Transliteration rules: validation words 82% exact (CER 0.034); self-written 107-word loanword list 78% exact (CER 0.059).

## Idea (Task 3) as implemented
- Target: english_word_error, category hinglish. Example items: val_hg_01, val_hg_06.
- Hypothesis: IndicF5 learned English loanwords in Devanagari; Latin letters are rare in its training text, so it
  guesses. Rewriting English words with general Indian-English letter-to-sound rules (not a lookup table) should
  make the model say them correctly.
- Optional extension: up to 4 spelling variants for the vowels the rules are unsure about, chosen by the recognizer.
- Risks: wrong vowel rules on unseen words; Whisper-kit vs final recognizers; SIM changes.

## Still to fill after the GPU run
Task 2 val_control CER/SIM, Task 3 val_idea CER/SIM and per-item gains/losses (Q1), Task 4 timing.


## Experiments after the first run (validation only; facts for idea.md part 4 and report.md)
Round 1 (seed 0): syllable-based length for every item: Hinglish 0.047 -> 0.026 but Hindi 0.034 -> 0.046 and
long 0.032 -> 0.060. Pause-compressed prompts: SIM up (0.841 -> 0.850) but CER worse. 8 steps / no CFG fail.
Round 2: syllable-based length only for sentences with English words (decided from the input text):
- seed 0: Hinglish 0.0467 -> 0.0257, other categories identical, overall 0.0585 -> 0.0532
- seed 1: 0.0254 -> 0.0254;  seed 2: 0.0303 -> 0.0291  (never worse; mean 0.034 -> 0.027)
- seed-to-seed spread of the plain idea system (0.025-0.047) is as large as the effect.
Task 4: 12 steps on the final Task 3 system: CER 0.0530 (= Task 3), 2.42x faster; 10 steps 0.0669 (> 1.2x limit).

Final Task 3 vs baseline per item (seed 0): no item worse; Hindi/long unchanged (not touched).
Largest gains: val_hd_08 0.603->0.064, val_hd_01 0.587->0.127, val_hg_01 0.452->0.032.
Smallest Hinglish gains: val_hg_02 0.216->0.054 ("address" -> अड्रेस), val_hg_04 0.263->0.026.

Round 3 (validation hard items, seeds 0-2): commas between spelled letters (ए, टी, एम):
hard CER 0.099/0.101/0.114 -> 0.107/0.102/0.101 (mean 0.105 -> 0.103, within seed noise). ATM always
better, RBI worse or equal, GST mixed. Not adopted.
