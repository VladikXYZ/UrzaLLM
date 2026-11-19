"""Text normalisation and edit distance shared by make_drafts.py and score.py."""
import re
import unicodedata

PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> list[str]:
    """Lowercase words without punctuation. A hyphen inside a word is dropped
    ("anarcho-kapitalismus" = "anarchokapitalismus"); other punctuation becomes a space."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"#\w+", " ", text)              # SeamlessM4T's hesitation marks (#hmm, #err): not words
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    return PUNCT_RE.sub(" ", text).split()


# Colloquial (obecná) and standard Czech forms of the same word count as equal in the lenient score:
# for the RAG corpus a model writing "této" for spoken "týdletý" is fine, even better.
DEMONSTRATIVES = set("""
ten ta to ti ty tu toho tomu tom tím té těch těm těmi
tento tato toto tito tyto tuto tohoto tomuto tomto tímto této těchto těmto těmito
tenhle tahle tohle tihle tyhle tuhle tohohle tomuhle tomhle tímhle týhle těchhle těmhle těmihle
tenhleten tahleta tohleto tihleti tyhlety tuhletu tohohletoho tomhletom tímhletím týhletý týdletý
týdlety tejdle tejhle tadyten tadyta tadyto tadyty tamten tamta tamto tamty
""".split())


def standardize(word: str) -> str:
    """Map a normalised word to a rough canonical form shared by its colloquial and standard variants.
    Applied the same way to reference and hypothesis, so it only has to be consistent, not correct Czech."""
    if word in DEMONSTRATIVES:
        return "<dem>"
    w = word
    if w.startswith("vo") and len(w) >= 3:         # vokno -> okno, von -> on
        w = w[1:]
    w = re.sub(r"(uj|ěj|ij|ej)ou$", r"\1í", w)       # existujou -> existují (before the ej rule below)
    w = re.sub(r"ej(?=[a-zčďěňřšťůýž]|$)", "ý", w)    # dobrej -> dobrý, jinejch -> jiných, mlejn -> mlýn
    w = re.sub(r"ýho$", "ého", w)
    w = re.sub(r"ýmu$", "ému", w)
    w = re.sub(r"ém$", "ým", w)                     # ve stejným (colloquial) = ve stejném
    w = re.sub(r"ýma$", "ými", w)                   # kterýma -> kterými
    w = re.sub(r"ma$", "mi", w) if len(w) > 4 else w  # lidma -> lidmi (not "doma", "ma")
    w = re.sub(r"ří$", "rý", w)                     # kteří, dobří -> který, dobrý
    w = re.sub(r"(?<=[knvlrtdh])[éí]$", "ý", w)      # který = které (colloquial -ý); -á stays: a real gender
    return w


def align(ref: list, hyp: list) -> list[tuple[str, int, int]]:
    """Levenshtein alignment as (op, i, j) steps: op is ok/sub/del/ins, i/j index ref/hyp (-1 if none)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != hyp[j - 1]))
    steps, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            steps.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            steps.append(("del", i - 1, -1))
            i -= 1
        else:
            steps.append(("ins", -1, j - 1))
            j -= 1
    return steps[::-1]


def errors(ref: list, hyp: list) -> dict:
    c = {"sub": 0, "del": 0, "ins": 0}
    for op, _, _ in align(ref, hyp):
        if op != "ok":
            c[op] += 1
    c["n"] = len(ref)
    return c
