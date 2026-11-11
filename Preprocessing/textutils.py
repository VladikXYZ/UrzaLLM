"""Small text helpers shared by all sources."""
import re
import unicodedata

WORD_RE = re.compile(r"\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SENTENCE_END = (".", "!", "?", "…", ":")
CLOSERS = "\"“”»)’'"
# Split after sentence punctuation when the next sentence starts with a capital,
# digit, quote or bracket.
SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?…])\s+(?=[„\"“(\[0-9A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])"
)


def words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def normalize(text: str) -> str:
    """Unicode NFKC, remove soft hyphens and zero-width chars, tidy spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "").replace("​", "").replace("﻿", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def join_paragraphs(paras: list[str]) -> str:
    return "\n\n".join(p for p in paras if p)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def repair_split_lines(lines: list[str]) -> tuple[list[str], int]:
    """Rejoin lines that the scraper split around inline links.

    A line is glued to the previous one when the previous line does not end a
    sentence, or the new line starts in lower case or with punctuation.
    Returns (paragraphs, number_of_joins).
    """
    out: list[str] = []
    joins = 0
    for line in (l.strip() for l in lines):
        if not line:
            continue
        if out:
            prev = out[-1].rstrip(CLOSERS + " ")
            starts_punct = line[0] in ",;.:!?)]»“”"
            starts_lower = line[0].islower()
            if starts_punct:
                out[-1] = out[-1].rstrip() + line
                joins += 1
                continue
            if starts_lower or not prev.endswith(SENTENCE_END):
                out[-1] = out[-1].rstrip() + " " + line
                joins += 1
                continue
        out.append(line)
    # tidy spaces before punctuation created by the joins, e.g. "Polis ;"
    out = [re.sub(r"\s+([,;.:!?)])", r"\1", p) for p in out]
    return out, joins


def norm_title(title: str, prefixes: list[str], noise: list[str] = ()) -> str:
    """Lower-case, strip known series prefixes, noise phrases and punctuation."""
    t = normalize(title).casefold()
    for p in prefixes:
        if re.match(re.escape(p) + r"\s*:", t):
            t = t.split(":", 1)[1].strip()
            break
    for n in noise:
        t = re.sub(n, "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def quantiles(values: list[float], qs=(0, 5, 25, 50, 75, 95, 100)) -> dict:
    if not values:
        return {f"p{q}": None for q in qs}
    v = sorted(values)
    n = len(v)
    return {f"p{q}": v[min(n - 1, round(q / 100 * (n - 1)))] for q in qs}
