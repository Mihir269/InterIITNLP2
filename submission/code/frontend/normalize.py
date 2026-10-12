"""
Task 2: rule-based Hindi text normalizer (no models).

normalize(text) rewrites numbers, ₹ amounts, dates, percentages, decimals and
letter abbreviations into Hindi words in Devanagari, the way they are spoken.
English words in Latin script are left unchanged (Task 3 handles them), except
all-capital abbreviations, which are spelled out letter by letter.

Every rule here was written from the validation items only.
"""
import re
import unicodedata

# 0..99 are irregular in Hindi, so they are listed in full.
ONES = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(ONES) == 100

MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त",
          "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]

# Letter names as Hindi speakers say them (RBI -> आर बी आई).
LETTERS = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी", "H": "एच",
    "I": "आई", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी",
    "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू",
    "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}
SCALE_WORDS = ("हज़ार", "हजार", "लाख", "करोड़", "करोड", "अरब")
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def num_words(n: int) -> str:
    """Integer to Hindi words with the Indian system (हज़ार, लाख, करोड़)."""
    if n < 100:
        return ONES[n]
    parts = []
    for size, name in ((10 ** 7, "करोड़"), (10 ** 5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        q, n = divmod(n, size)
        if q:
            parts += [num_words(q), name]
    if n:
        parts.append(ONES[n])
    return " ".join(parts)


def year_words(n: int) -> str:
    """Years 1100-1999 are read in hundreds (उन्नीस सौ सैंतालीस); others as numbers."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return f"{ONES[hi]} सौ" + (f" {ONES[lo]}" if lo else "")
    return num_words(n)


def digits_words(s: str) -> str:
    return " ".join(ONES[int(d)] for d in s)


def decimal_words(int_part: str, frac: str) -> str:
    return f"{num_words(int(int_part))} दशमलव {digits_words(frac)}"


def spell_abbrev(tok: str) -> str:
    return " ".join(LETTERS[c] for c in tok)


def read_abbrev(tok: str) -> str:
    """Abbreviations of 4+ letters with 2+ vowels and no 3-consonant run are said as a word
    (NASA, UNICEF); the rest are spelled (RBI, ATM, GST in validation)."""
    if len(tok) >= 4 and sum(c in "AEIOU" for c in tok) >= 2 and not re.search(r"[^AEIOU]{3}", tok):
        from translit import translit
        return translit(tok.lower(), acronym=True)
    return spell_abbrev(tok)


NUM = r"\d(?:[\d,]*\d)?"  # 1,25,000 or 125000 or 4


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _rupees(m: re.Match) -> str:
    amount, frac, scale = m.group("amt"), m.group("frac"), m.group("scale")
    if scale:  # ₹38 लाख -> अड़तीस लाख रुपये ; ₹2.3 करोड़ -> दो दशमलव तीन करोड़ रुपये
        head = decimal_words(amount.replace(",", ""), frac) if frac else num_words(_int(amount))
        return f"{head} {scale} रुपये"
    words = f"{num_words(_int(amount))} रुपये"
    if frac:
        paise = int((frac + "0")[:2])
        if paise:
            words += f" {num_words(paise)} पैसे"
    return words


def _date(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    y = int(y) + (2000 if len(y) == 2 else 0)
    return f"{num_words(d)} {MONTHS[mo - 1]} {year_words(y)}"


def _number(m: re.Match) -> str:
    s = m.group(1) if m.re.groups else m.group(0)
    if "." in s:
        a, b = s.split(".", 1)
        return decimal_words(a.replace(",", ""), b)
    if "," not in s and len(s) == 4:
        return year_words(int(s))
    if s.startswith("0") and len(s) > 1:  # 0471, PIN-like strings: read digit by digit
        return digits_words(s)
    return num_words(_int(s))


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).translate(DEV_DIGITS)
    # dates dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
    t = re.sub(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})\b", _date, t)
    # clock times 3:30 -> तीन बजकर तीस मिनट
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b(?:\s*बजे)?",
               lambda m: f"{num_words(int(m.group(1)))} बजकर {num_words(int(m.group(2)))} मिनट"
               if int(m.group(2)) else f"{num_words(int(m.group(1)))} बजे", t)
    # rupee amounts, with an optional scale word after the number
    scale = "|".join(SCALE_WORDS)
    t = re.sub(rf"(?:₹|\bRs\.?|\bINR)\s*(?P<amt>{NUM})(?:\.(?P<frac>\d+))?(?:\s*(?P<scale>{scale}))?",
               _rupees, t)
    # percentages
    t = re.sub(rf"({NUM}(?:\.\d+)?)\s*%", lambda m: _number(m) + " प्रतिशत", t)
    # word-number joins such as चंद्रयान-3
    t = re.sub(r"(?<=[ऀ-ॿA-Za-z])-(?=\d)", " ", t)
    # remaining numbers (integers, Indian comma groups, decimals)
    t = re.sub(rf"{NUM}(?:\.\d+)?", _number, t)
    # all-capital Latin abbreviations (2-7 letters): spell them out
    t = re.sub(r"\b[A-Z]{2,7}\b", lambda m: read_abbrev(m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    import sys
    for line in sys.argv[1:] or sys.stdin:
        print(normalize(line))
