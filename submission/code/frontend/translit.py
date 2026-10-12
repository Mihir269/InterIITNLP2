"""
Task 3: rule-based English -> Devanagari transliteration with Indian-English pronunciation.

No dictionary of words and no model: the word is split into grapheme units (digraphs such as
"sh", "ee", "igh"), each vowel letter is read from its context (open or closed syllable, a
silent final e, a following r, unstressed endings), and the result is written in Devanagari
with Hindi spelling habits (anusvara before a stop, halant only inside real clusters, bare
consonant at the end of a word).

    >>> translit("internet")
    'इंटरनेट'
"""
import re

# ---- phoneme -> Devanagari --------------------------------------------------------------
CONS = {
    "k": "क", "kh": "ख", "g": "ग", "ch": "च", "j": "ज", "T": "ट", "D": "ड", "t": "त", "d": "द",
    "th": "थ", "n": "न", "p": "प", "f": "फ", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल",
    "v": "व", "sh": "श", "s": "स", "h": "ह", "z": "ज़",
}
# vowel phoneme -> (independent letter, matra)
VOW = {
    "a": ("अ", ""), "aa": ("आ", "ा"), "i": ("इ", "ि"), "ii": ("ई", "ी"), "u": ("उ", "ु"),
    "uu": ("ऊ", "ू"), "e": ("ए", "े"), "ai": ("ऐ", "ै"), "o": ("ओ", "ो"), "au": ("औ", "ौ"),
    "O": ("ऑ", "ॉ"),
}
HALANT, ANUSVARA = "्", "ं"
STOPS = set("k kh g ch j T D t d th p f b s sh z".split())  # n before these -> anusvara
SONORANT_CODA = {"r", "l", "n", "m"}                           # written bare before a consonant

VOWEL_LETTERS = set("aeiouAEIUV")
# Upper-case markers written by the spelling rewrites in _phonemes: they fix a vowel's value.
MARKERS = {"A": "aa", "E": "e", "I": "ii", "U": "u", "V": "a"}


def _units(w):
    """Split a lowercase word into grapheme units, longest match first."""
    multi = ["iew", "tch", "dge", "igh", "sch", "ch", "sh", "ph", "th", "ck", "wh", "qu", "gh", "ng",
             "ee", "ea", "oo", "ou", "ow", "ai", "ay", "oa", "oi", "oy", "au", "aw", "ew", "ue",
             "ie", "ei", "ey"]
    out, i = [], 0
    while i < len(w):
        for m in multi:
            if w.startswith(m, i):
                out.append(m)
                i += len(m)
                break
        else:
            out.append(w[i])
            i += 1
    return out


def _is_vowel_unit(u):
    return u[0] in VOWEL_LETTERS or (u == "y")


def _phonemes(word, acronym=False):
    w = word.lower()
    # whole-word endings and silent letters
    w = re.sub(r"^kn", "n", w)
    w = re.sub(r"^wr", "r", w)
    w = re.sub(r"^ps", "s", w)
    w = re.sub(r"mb$", "m", w)
    w = re.sub(r"gn$", "n", w)
    w = re.sub(r"alk", "awk", w)             # walk, talk
    w = re.sub(r"(?<=.)tion", "shan", w)     # station
    w = re.sub(r"(?<=.)sion", "shan", w)
    w = re.sub(r"(?<=.)ture$", "char", w)    # picture
    w = re.sub(r"(?<=.)ous$", "as", w)
    w = re.sub(r"ou(?=bl|ch|ntr|ng)", "V", w)  # double, touch, country, young
    w = re.sub(r"(?<=[^aeiou])le$", "al", w)  # table -> tebal
    w = re.sub(r"(?<=.)[ct]ial", "shal", w)  # special, partial
    w = re.sub(r"ower", "Aver", w)            # power, flower
    w = re.sub(r"oo(?=[kdt])", "U", w)        # book, good, foot
    w = re.sub(r"ea(?=d|th|lth|v)", "E", w)   # ready, weather, health, heavy
    w = re.sub(r"ou(?=p)", "U", w)            # group
    w = re.sub(r"(?<=.)th(?=er)", "ð", w)     # weather, mother: voiced th
    w = re.sub(r"(?<=[aeiouAEIU])s(?=[iy])", "z", w)  # music, busy, easy
    w = re.sub(r"^ex(?=[aeiou])", "egz", w)    # exam, exact
    w = re.sub(r"(?<=.)ege$", "Ej", w)        # college
    if len(w) > 5:
        w = re.sub(r"^re(?=[bcdfgjklmnpqstvz][aeiou])", "ri", w)  # result, return, repeat
        w = re.sub(r"^a(?=(cc|pp|tt|nn|ss|dd|ff|ll))", "V", w)  # account, attend: unstressed a
    u = _units(w)
    n = len(u)
    ph = []
    syll = sum(1 for k, x in enumerate(u) if x[0] in VOWEL_LETTERS or (x == "y" and k > 0))

    def nxt(k, off=1):
        j = k + off
        return u[j] if 0 <= j < n else ""

    def cons_run_after(k):
        """Number of consonant units after position k before the next vowel (or end)."""
        c, j = 0, k + 1
        while j < n and not (u[j][0] in VOWEL_LETTERS):
            c += 1
            j += 1
        return c, j

    k = 0
    while k < n:
        x = u[k]
        last = k == n - 1
        # ---------------- consonants ----------------
        if x[0] not in VOWEL_LETTERS:
            prev_is_vowel = k > 0 and u[k - 1][0] in VOWEL_LETTERS
            if x == "y":
                if k == 0 or (k > 0 and u[k - 1][0] in VOWEL_LETTERS and not last):
                    ph.append("y")
                elif last:
                    ph.append("ii")              # party, happy
                elif cons_run_after(k)[0] >= 2:
                    ph.append("i")               # system, gym
                else:
                    ph += ["aa", "i"]            # type, cycle
                k += 1
                continue
            m = {
                "tch": ["ch"], "dge": ["j"], "sch": ["s", "k"], "ch": ["ch"], "sh": ["sh"], "ph": ["f"],
                "th": ["th"], "ck": ["k"], "wh": ["v"], "qu": ["k", "v"], "ng": ["n", "g"],
                "b": ["b"], "d": ["D"], "f": ["f"], "h": ["h"], "j": ["j"], "k": ["k"], "l": ["l"],
                "m": ["m"], "n": ["n"], "p": ["p"], "q": ["k"], "r": ["r"], "s": ["s"], "t": ["T"],
                "v": ["v"], "w": ["v"], "x": ["k", "s"], "z": ["z"], "ð": ["d"],
            }.get(x)
            if x == "gh":
                if k == 0:
                    ph.append("g")
                elif last:
                    ph.append("f")               # enough, tough
                k += 1
                continue
            if x == "c":
                m = ["s"] if nxt(k)[:1] in ("e", "i", "y") else ["k"]
            if x == "g":
                m = ["j"] if (nxt(k) == "e" and k + 1 == n - 1) or nxt(k)[:1] in ("y",) else ["g"]
                if nxt(k) in ("e", "i") and k + 2 < n and nxt(k, 2) in ("n", "s") and syll > 1:
                    m = ["j"]                     # agent, logic-like
            if x == "h" and prev_is_vowel and (last or not _is_vowel_unit(nxt(k))):
                k += 1; continue                 # silent h after a vowel
            if x == "s" and last and k > 0 and u[k - 1] in ("e",) and syll > 1:
                m = ["s"]
            # doubled letters are pronounced once
            if k > 0 and u[k - 1] == x:
                k += 1
                continue
            ph.extend(m or [])
            k += 1
            continue
        # ---------------- vowels ----------------
        ncons, j = cons_run_after(k)
        if x in MARKERS:
            ph.append(MARKERS[x])
            k += 1
            continue
        final_e_follows = (ncons == 1 and j == n - 1 and u[j] == "e") or \
                          (ncons == 1 and j == n - 2 and u[j] == "e" and u[-1] in ("s", "d"))
        last_syllable = j >= n or (final_e_follows)
        followed_by_r = nxt(k) == "r"
        r_closed = followed_by_r and (k + 2 >= n or nxt(k, 2)[0] not in VOWEL_LETTERS)

        if acronym and x in ("a", "e", "i", "o", "u") and not (x == "e" and last):
            v = {"a": "aa", "e": "e", "i": "i", "o": "o", "u": "u"}[x]   # NASA -> नासा
        elif x == "igh":
            v = "aai"                             # night, flight
        elif x in ("ee", "ea"):
            v = "ii"
            if x == "ea" and r_closed: v = "ii"  # near, clear -> ii + r
        elif x == "oo":
            v = "uu"
        elif x == "ou":
            v = "aau"
        elif x == "ow":
            v = "o" if last else "aau"
        elif x in ("ai", "ay", "ei", "ey"):
            v = "e" if not (x == "ey" and last) else "ii"
        elif x == "oa":
            v = "o"
        elif x in ("oi", "oy"):
            v = "Oy"
        elif x in ("au", "aw"):
            v = "O"
        elif x in ("ew", "ue", "iew"):
            v = "yuu"
        elif x == "ie":
            v = "ii" if last or (k + 1 < n and u[k + 1] == "s" and k + 2 == n) else "aai"
        elif x == "e" and last:
            if syll == 1 and k > 0:
                v = "ii"                          # me, be, he
            else:
                k += 1
                continue                          # silent final e
        elif x == "e" and k == n - 2 and u[-1] in ("s", "d") and syll > 1 and k > 0 and \
                u[k - 1][0] not in VOWEL_LETTERS and ncons == 1:
            # -es / -ed endings: mostly silent vowel (lives, moved); keep short e after t/d
            v = "i" if u[k - 1] in ("t", "d", "s", "c", "ch", "sh", "z", "x", "g") else None
            if v is None:
                k += 1
                continue
        elif r_closed and x in ("a", "o", "e", "i", "u"):
            v = {"a": "aa", "o": "O", "e": "a", "i": "a", "u": "a"}[x]
            if x == "a" and k > 0 and u[k - 1] in ("w", "qu"):
                v = "O"                           # war, quarter
            if x == "o" and k == n - 2 and syll > 1:
                v = "a"                           # doctor, motor
            if x in ("e", "a") and k == n - 2 and syll > 1:
                v = "a"                           # paper, sugar
        elif final_e_follows:
            unstressed = syll > 2 or (syll == 2 and k > 1 and u[0][0] not in VOWEL_LETTERS and
                                      any(z[0] in VOWEL_LETTERS for z in u[:k]))
            cons_unit = u[k + 1]
            if unstressed and x == "i" and cons_unit in ("c", "v"):
                v = "i"                           # office, service, active
            elif unstressed and x == "a" and cons_unit in ("g", "t"):
                v = "e" if cons_unit == "g" else "e"  # message, private
            else:
                v = {"a": "e", "e": "ii", "i": "aai", "o": "o", "u": "yuu"}[x]
                if x == "u" and k > 0 and u[k - 1] in ("r", "l", "j"):
                    v = "uu"
        else:
            closed = ncons >= 2 or j >= n
            nxt_c = nxt(k)
            if x == "a":
                if k > 0 and u[k - 1] in ("w", "qu"):
                    v = "O"                       # what, want
                elif re.match(r"(ss|st|sk|sp|ff|th|nce|nt)", "".join(u[k + 1:k + 3])) and syll == 1:
                    v = "aa"                      # class, fast, ask
                elif nxt_c == "l" and nxt(k, 2) in ("l", "") and syll == 1:
                    v = "O"                       # all, call
                elif last:
                    v = "aa"                      # data, sofa (final a)
                elif closed:
                    v = "ai"                      # cat, match, cancel
                elif re.match(r"(er|ion|an|al|a|e|y|ey)$", "".join(u[k + 2:])):
                    v = "e"                       # paper, station, table, data, lady
                else:
                    v = "ai"                      # family, salary, manager, camera
                if j >= n and syll > 1 and ncons == 1 and nxt_c in ("l", "n", "m"):
                    v = "a"                       # final, woman: unstressed
            elif x == "e":
                v = "e"
                if j >= n and syll > 1 and ncons == 1 and nxt_c in ("l", "n", "m", "r") and k > 0:
                    v = "a"                       # hotel, chicken: unstressed
                if k == 0 and nxt_c == "x":
                    v = "e"
            elif x == "i":
                v = "i"
                if last:
                    v = "ii"                      # taxi, wifi
                elif nxt_c == "n" and nxt(k, 2) == "d" and k + 3 >= n:
                    v = "aai"                     # find, mind
                elif nxt_c == "l" and nxt(k, 2) == "d":
                    v = "aai"                     # child, wild
            elif x == "o":
                if last:
                    v = "o"                       # go, photo
                elif closed:
                    v = "O"                       # shop, office, online
                    if nxt_c in ("l",) and nxt(k, 2) in ("d", "t"):
                        v = "o"                   # old, bolt
                    if nxt_c == "n" and syll > 1 and j >= n - 1:
                        v = "a"                   # lesson, button (final -on)
                else:
                    v = "o"                       # hotel, total
                    if nxt_c == "v" and nxt(k, 2) in ("e", "ie", "i"):
                        v = "uu"                  # move, movie, prove
                if j >= n and syll > 1 and ncons == 1 and nxt_c in ("n", "m") and k > 0:
                    v = "a"
            elif x == "u":
                if last:
                    v = "uu"
                elif closed:
                    v = "a"                       # cut, bus, submit
                    if k > 0 and u[k - 1] in ("p", "b", "f") and nxt_c in ("l", "sh", "t") \
                            and nxt(k, 2) != nxt_c:
                        v = "u"                   # pull, push, put
                else:
                    v = "yuu" if k == 0 or u[k - 1] in ("m", "f", "c", "p", "h") else "uu"
            else:
                v = "a"
        if v:
            for p in re.findall(r"aai|aau|yuu|Oy|aa|ii|uu|ai|au|[aeiouOy]", v):
                if p == "aai":
                    ph += ["aa", "i"]
                elif p == "aau":
                    ph += ["aa", "u"]
                elif p == "yuu":
                    ph += ["y", "uu"]
                elif p == "Oy":
                    ph += ["O", "y"]
                else:
                    ph.append(p)
        k += 1
    return ph


def _render(ph):
    out = []
    n = len(ph)
    is_v = [p in VOW for p in ph]
    for k, p in enumerate(ph):
        if is_v[k]:
            if k > 0 and not is_v[k - 1]:
                out.append(VOW[p][1])
            elif k > 0 and p == "a":
                out.append("व" if ph[k - 1] in ("u", "uu") else "य")   # glide: player -> प्लेयर
            else:
                # vowel after vowel ("aa"+"i" -> आइ) or word-initial vowel
                out.append(VOW[p][0])
            continue
        nxt = ph[k + 1] if k + 1 < n else None
        prev_v = k > 0 and is_v[k - 1]
        if p == "n" and nxt is not None and not is_v[k + 1] and nxt in STOPS and prev_v:
            out.append(ANUSVARA)
            continue
        if p == "m" and nxt in ("p", "b") and prev_v:
            out.append(ANUSVARA)
            continue
        out.append(CONS[p])
        if nxt is not None and not is_v[k + 1]:
            # consonant cluster: write a halant unless it is a sonorant coda after a vowel
            # a sonorant coda is written bare when a new syllable follows (इंटरनेट, ऑनलाइन),
            # with a halant in a final cluster (कोल्ड); r after a full vowel takes a halant (कार्ड)
            new_syll = k + 2 < n and is_v[k + 2]
            bare = prev_v and p in SONORANT_CODA and new_syll and not (p == "r" and ph[k - 1] != "a")
            if not bare:
                out.append(HALANT)
    s = "".join(out)
    return s


def translit(word: str, acronym: bool = False) -> str:
    if not re.fullmatch(r"[A-Za-z]+", word):
        return word
    return _render(_phonemes(word, acronym))


def translit_text(text: str) -> str:
    """Replace every Latin-script word in `text` with its Devanagari rendering."""
    return re.sub(r"[A-Za-z]+", lambda m: translit(m.group(0)), text)


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        print(w, translit(w))
