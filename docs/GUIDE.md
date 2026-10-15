# TTS bootcamp – background, what we did, and how to write idea.md / report.md

This guide is for you to read, not to submit. The rules say `idea.md` and `report.md` must be in
**your own words**, so it gives you the knowledge, the numbers and a structure, not finished text.

---

## 0. Plan for the last ~12 hours

| When | What | Who |
|---|---|---|
| now | Read sections 1–5 of this guide | you |
| now (15 min GPU) | Run `experiments.ipynb` (round 3, `PHASES = "t2c"`), upload the result | you → me |
| +1 h | I decide on the abbreviation change and update the pipeline | me |
| +1–2 h | Run `run_all.ipynb` (final run, ~60 min), upload `results_bundle.zip` | you |
| meanwhile | Listen to the 14 tagged baseline items; write `idea.md` and `report.md` | you |
| last 2 h | Put idea.md/report.md in the zip, run `check_format.py`, upload | you |

Keep at least 3 hours of margin before the deadline for Colab problems.

---

## 1. Background you need (plain language)

### 1.1 Text-to-speech (TTS) and voice cloning
A TTS system turns text into audio. **Zero-shot voice cloning** means: give the model a short clip
of someone speaking (the *prompt* or *reference clip*, here 8–12 s) plus the text of that clip
(the *transcript*), and it reads **new** text in that voice, without any training on that speaker.

### 1.2 How IndicF5 works (enough to reason about it)
IndicF5 is F5-TTS fine-tuned on ~1,400 hours of Indian-language speech.

- Audio is represented as a **mel spectrogram**: a picture of the sound, one column every ~10 ms.
  A separate small network, the **vocoder** (Vocos), turns the spectrogram back into a waveform.
- The model gets: *prompt spectrogram* + *prompt transcript + new text* (as one character string),
  and must fill in the spectrogram for the new text. It is a **flow-matching** model: it starts
  from random noise and moves it step by step towards a clean spectrogram.
  - **NFE steps** (`nfe_step`, 32 by default) = how many steps. Fewer steps = faster but rougher.
  - **CFG** (`cfg_strength`, classifier-free guidance) = each step runs the model twice (with and
    without the text) and pushes the result towards "follows the text". Turning it off halves the
    work but the speech stops following the text (we measured CER 0.22 – a disaster).
  - **Seed** fixes the random noise. Different seeds give slightly different audio; that is why we
    compare systems with the same seed (and why a gain on one seed can be luck).
- **The model does not decide how long to speak.** The length is set *before* generation:

  `output length = prompt length × (characters in new text ÷ characters in prompt transcript) ÷ speed`

  So it assumes "every character takes the same time, at the prompt speaker's pace". This one formula
  explains a lot of our findings. `speed` is a setting we are allowed to change per item.
- Its text vocabulary is characters. It was trained on Indian speech whose transcripts write
  English loanwords **in Devanagari** (ऑफिस, मार्केट). Latin letters are rare in its training text.
  (Checked: `vocab.txt` has 2,546 tokens and does contain all 52 Latin letters, so Latin is
  *known but under-trained*, not missing.)

### 1.3 How the outputs are scored
- **CER (character error rate)**: a speech recognizer (ASR) transcribes the audio; CER = number of
  character edits (insert/delete/substitute) needed to turn the transcript into the reference
  `spoken` text, divided by the reference length. 0 = perfect, 1 = useless. Before comparing,
  spaces and punctuation are removed and spelling variants are folded (nukta, ॉ→ो, chandrabindu).
  Our kit uses Whisper-large-v3-turbo; **the final scoring uses other recognizers**, so tiny gains
  on the kit recognizer may not transfer.
- **SIM (speaker similarity)**: a speaker-verification network (ECAPA) turns each clip into a
  "voice fingerprint" vector; SIM is the cosine similarity between output and prompt (1 = same voice).
- **Sanity checks**: an item fails (CER 1, SIM 0) if it is mostly silence, clipped, outside 0.5–60 s,
  or spoken slower than 4 or faster than 30 reference characters per second.

### 1.4 Text side
- **Text normalization**: rewriting things that are not words (12/03/2026, ₹1,25,000, 92%, RBI)
  into the words a person would say (बारह मार्च दो हज़ार छब्बीस …). Hindi numbers 0–99 are irregular
  (इक्यावन, उनहत्तर), so they must be listed; above that the Indian system uses हज़ार, लाख, करोड़.
- **Transliteration / G2P (grapheme-to-phoneme)**: writing a word in another script by how it sounds.
  English spelling is irregular, so we use rules: letter groups ("sh", "ee", "igh"), open vs closed
  syllables (ho-tel vs shop), silent final e (late), r after a vowel (car, order), and Indian-English
  vowel habits (o in a closed syllable → ऑ: ऑफिस, शॉपिंग).
- **Schwa deletion**: Hindi drops the inherent "a" at word ends (राम = rām, not rāma). Our syllable
  counter uses this.
- **Aspiration**: छ, घ, झ, ख, ध are breathy versions of च, ग, ज, क, द. The model sometimes loses the
  breath (छब्बीस → चब्बीस), especially for a fast speaker.

---

## 2. The problem and how each task is scored

| Task | What you deliver | How it is scored |
|---|---|---|
| 1 | Baseline scores + error tags (`t1_val_errors.csv`) | tags must be real errors, checked by ear |
| 2 | Rule-based `normalize()`; `t2_normalized.csv`; `t2_norm/` audio | hard-item CER vs baseline; other categories must not get >5% worse |
| 3 | Your own idea; `t3_idea/`, `val_idea/`, `val_control/`, `idea.md` | **relative CER reduction on your targeted category**; overall CER ≤ 10% worse and SIM ≤ 0.03 lower; idea.md read by two seniors |
| 4 (bonus) | Faster version; `t4_fast/` | CER ≤ 1.2 × Task 3; speed-up = baseline time ÷ your time |
| all | `results.json`, `report.md`, `code/README.md` | numbers must match the audio within 0.01 |

The seniors judge idea.md on **grounding** (is the failure real and shown with item IDs?),
**testability** (is the prediction checkable?), **originality** (more than tuning / a lookup table)
and **insight** (did you learn *why*?). A clear negative result counts as fully as a positive one.

---

## 3. What we did, why, and what came out

All numbers: 24 validation items, seed 0, kit recognizer.

### Task 1 – measure the baseline
- Baseline CER **0.242**, SIM **0.830** (matches the published numbers).
- By category: hindi 0.034, long 0.032, **hinglish 0.302**, **hard 0.458**.
- 14 items above CER 0.20 = all 6 hinglish (`english_word_error`) + all 8 hard (`number_error`).
- Example: val_hg_01 "office … internet … slow" was heard as "अजसे … सर्ण … कोई";
  val_hd_01 "12/03/2026 … ₹1,25,000" as "पेड़म शोर एन्यो उसन जाइट".
- **Why** (from our EDA): output length follows the character count. "₹1,25,000" is 9 characters
  but "एक लाख पच्चीस हज़ार रुपये" is ~20 spoken characters, so number items get far too little time
  (correlation between "input shorter than what must be said" and CER: −0.72). Hinglish items have
  the right length, so their problem is different: **Latin script** the model hardly saw in training.

### Task 2 – text normalizer (`code/frontend/normalize.py`)
- Rules for Indian numbers, ₹ and paise, dd/mm/yyyy dates, years, decimals, %, times, abbreviations
  spelled letter by letter (RBI → आर बी आई; pronounceable ones like NASA read as a word).
- Text check: on validation it reproduces the reference `spoken` text of all 8 hard items exactly.
- Audio: **hard CER 0.458 → 0.099** (−78%); other categories unchanged. Overall 0.242 → 0.122.

### Task 3 – the idea
**Target:** `english_word_error`, category **hinglish** (examples val_hg_01, val_hg_06).

**Part A – rule-based transliteration (`translit.py`).** Rewrite every English word in Devanagari
with general Indian-English letter-to-sound rules (no word list, so it can handle unseen evaluation
words). Hypothesis: the model knows English loanwords in Devanagari but not in Latin script.
- Rules get 82% of validation words exactly right and 78% of a 107-word list of other common
  loanwords (built without evaluation words) – this shows they generalize.
- Audio: **hinglish CER 0.302 → 0.047** (−84.5%), SIM 0.830 → 0.841.

**Part B – syllable-based length for English-containing sentences (`syllables.py`).** Added after
the first results. Because length = characters × speaker pace, words with many conjuncts and vowel
signs (typical of transliterated English, e.g. इंटरनेट, प्रॉपर) get the wrong amount of time. We count
syllables instead and pass the correction through `speed` – only for sentences that contain English
words, decided automatically from the input text.
- Applying it to *every* item hurt Hindi (0.034 → 0.046) and long items (0.032 → 0.060): for normal
  Hindi the character count was already fine. Limiting it to English sentences fixed that.
- Result (seed 0): **hinglish 0.047 → 0.026**, other categories identical; overall **0.0532**.
- Robustness: on seeds 1 and 2 the gain was 0.025 → 0.025 and 0.030 → 0.029. So it **never hurt**,
  but the true average gain is modest (mean 0.034 → 0.027); seed-to-seed noise (0.025–0.047) is as
  big as the effect. That is an important honest finding for your write-up.
- Final Task 3: **hinglish 0.302 → 0.026 = 91.5% relative reduction**; overall 0.242 → 0.053;
  SIM 0.830 → 0.841; no validation item got worse than the baseline.

**Tried and rejected (negative results are worth mentioning):**
- 4 spelling variants per sentence chosen by the recognizer: 0.047 → 0.043 for 4× the compute, and
  it risks fitting the kit recognizer – dropped.
- Shortening pauses in the prompt clips: SIM went up (0.841 → 0.850) but CER got worse – dropped.

### Task 4 – faster
- 12 flow-matching steps instead of 32 on the final Task 3 system: CER **0.0530** (same as Task 3),
  **2.42× faster** (668.7 s → 276.4 s for the 24 items, same Colab T4 session).
- 10 steps: CER 0.0669 > 1.2 × 0.0532 – fails. 8 steps: 0.098. No CFG: 0.22. Shorter prompts +
  12 steps: 0.073. So 12 steps is the sweet spot.

---

## 4. Data exploration – what is still wrong (for Q2/Q3 and for ideas)

From the transcripts of the final Task 3 system:
- **Recognizer spelling, not real errors:** रुपये/रुपे, सौ/सो, बारह/बारा, हफ़्ते/हफते, ऊपर/उपर. Different
  recognizers spell these differently; chasing them would be fitting the kit recognizer.
- **Real pronunciation slips:** aspiration lost (छब्बीस→चब्विस, घोषणा→गोशना, झाड़ता→जाडता, लाख→लाग),
  श→स (शांति→सांती, शाम→साम). Concentrated in **prompt V3**, the fastest speaker (6.3 syllables/s,
  others 3.6–4.8): V3 items have the highest CER among the hard and long items (val_hd_03 0.163,
  val_hd_07 0.111, val_lg_03 0.052) and the highest per-prompt average (0.063 vs 0.046–0.053).
- **Spelled abbreviations run together:** ए टी एम → "एटियम", जी एस टी → "एस्टी". Round 3 tests commas
  between the letters (ए, टी, एम). Evaluation has LIC, OTP, UPI, KYC, EMI, so this matters for Task 2.
- **Evaluation speakers are slower** (2.0–4.5 syllables/s; E2 is a list of names, only 2.0). So a
  "slow down fast speakers" rule would only ever fire on validation – we decided not to use it.
  The slow E2 voice plus our slow-down came close to the 4 chars/s fail line (4.5), so we added a
  floor of 5 chars/s; it changes only two E2 items and nothing on validation.
- **Remaining Hinglish errors** are vowel choices: address → अड्रेस (should be एड्रेस), message →
  मेसेज (मैसेज), cancel → कैंसल (कैंसिल). English spelling is ambiguous; rules cannot be perfect.

---

## 5. How to write `idea.md` (≤ 200 words in total)

**Format rules** (from the problem statement and `check_format.py`):
- Four parts, labelled so the checker finds them: `(1)`, `(2)`, `(3)`, `(4)`.
- Total ≤ 200 words. A rough budget: (1) 35, (2) 60, (3) 40, (4) 60.
- Parts 1–3 were due on the **Day-3 form** and must not be edited afterwards. If you submitted
  them, copy them **exactly** as submitted. Part 4 is where everything learned later goes.

**What each part needs (answer these questions in your own sentences):**

**(1) Failure.** Which failure (use the Task 1 tag name: `english_word_error`) and which category
(hinglish)? Give 2 baseline item IDs as evidence (val_hg_01, val_hg_06 are clear ones) and one
number (hinglish CER 0.302 vs 0.034 for pure Hindi).

**(2) Hypothesis and method.** *Why* does the model fail (training transcripts write English in
Devanagari, so Latin script is unfamiliar)? *What* do you do about it (general letter-to-sound rules,
not a lookup table; name 2–3 rule types)? Say why it should work on unseen evaluation words.

**(3) Prediction.** Which category improves, by roughly how much, and what could go wrong
(e.g. wrong vowels on irregular English spellings; gain may not transfer to other recognizers;
could change speaker similarity).

**(4) Result and lesson.** Give the numbers (hinglish 0.302 → 0.047 with rules alone; → 0.026 with
the syllable length). Say what was added after Day 3 and why (the length formula counts characters).
Mention the honest caveats: seed noise as large as the second gain; Hindi got worse when the length
fix was applied everywhere. End with what it taught you about *why* the model fails.

**Graders like:** a mechanism (not just "it works"), numbers with item IDs, and clear separation
between what was predicted and what was found.
**Avoid:** claiming the evaluation result (you cannot measure it); calling the tuning of steps an
idea; vague words like "significantly".

---

## 6. How to write `report.md` (Q1–Q3, ≤ 75 words each)

Start each answer on a line beginning `Q1`, `Q2`, `Q3`. Do not copy the question text.

**Q1. Two items where your system beats the baseline, two where it is worse, and why.**
No validation item is worse, so the rule says: give the two with the **smallest gain**.
- Largest gains (pick two): val_hd_08 0.603 → 0.064, val_hd_01 0.587 → 0.127,
  val_hg_01 0.452 → 0.032 (office/internet/slow said correctly once written in Devanagari).
- Smallest gains: the Hindi and long items are unchanged (0 gain) because the idea does not touch
  them; among Hinglish, val_hg_02 0.216 → 0.054 (address → अड्रेस, vowel rule wrong) and
  val_hg_04 0.263 → 0.026. Pick what you can explain best.

**Q2. What did the result teach you about why the model fails? One thing you would do differently.**
Material: the model's length is set by character count × speaker pace, not by the words; the
pronunciation failure is about script, not about English sounds as such; seed noise is as big as
small effects. Something to do differently: test several seeds from the start; or measure the
rules on a held-out word list before running audio.

**Q3. What single change would you try next, and why?**
Candidates (pick one and justify it): slowing fast prompts / choosing calmer prompt segments to
keep aspiration; commas between spelled letters; a vowel-ambiguity rule set checked on more words;
correcting prompt transcripts.

---

## 7. Final checklist
- [ ] `idea.md`: parts (1)–(4), ≤ 200 words, parts 1–3 identical to the Day-3 form
- [ ] `report.md`: Q1, Q2, Q3, each ≤ 75 words
- [ ] Listened to the 14 tagged items in `kit/data/baseline_val/`; change a tag if your ear disagrees
- [ ] `results.json` from the final run (roll number filled; numbers come from `fill_results.py`)
- [ ] `python kit/check_format.py 24b2443_tts_bootcamp.zip` → RESULT: OK
- [ ] Zip ≤ 100 MB, uploaded before the deadline

---

## 8. Glossary
- **ASR** – automatic speech recognition (speech → text). Whisper is one.
- **CER** – character error rate (see 1.3).
- **CFG** – classifier-free guidance; makes the output follow the text more strongly.
- **ECAPA** – a speaker-recognition network used for SIM.
- **Flow matching** – generation by moving noise towards data in a number of steps (NFE).
- **G2P** – grapheme-to-phoneme: spelling → pronunciation.
- **Hinglish / code-mixing** – Hindi sentences with English words.
- **Mel spectrogram** – time × frequency picture of audio; **vocoder** turns it back into sound.
- **NFE** – number of function evaluations = flow-matching steps.
- **Prompt / reference clip** – the voice sample to clone; its **transcript** must match it.
- **Schwa** – the inherent "a" of a Devanagari consonant; often silent in Hindi.
- **Seed** – fixes randomness so runs are repeatable.
- **SIM** – cosine similarity of speaker embeddings, output vs prompt.
- **Zero-shot** – works for a new speaker without training on them.
