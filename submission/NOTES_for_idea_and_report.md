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
