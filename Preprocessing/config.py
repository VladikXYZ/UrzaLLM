"""All tunable settings for the preprocessing pipeline live here.

Paths are resolved relative to this file, so the pipeline works from any
working directory.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent   # UrzaLLM repo root

# --- Paths -----------------------------------------------------------------
DATA_DIR = REPO / "jsons"
OUTPUT_DIR = HERE / "output"

ARTICLES_FILE = "urza_articles.json"
VIDEOS_FILE = "channel_metadata.json"

# --- Chunking ----------------------------------------------------------------
# Per source:
#   target   aim for chunks about this long (words)
#   max      never exceed this (unless one sentence is longer); shorter docs stay whole
#   min_tail a shorter final chunk is merged into the previous one (if that stays ≤ 1.2 × max)
#   overlap  carry the previous chunk's last sentence over if it's at most this many words
# Books are dense, so they get smaller chunks (book paragraphs: median 84 words,
# 95% under 200), making each chunk closer to a single argument. Articles and
# videos are already this size.
CHUNKING = {
    "book":    {"target": 150, "max": 250, "min_tail": 50, "overlap": 30},
    "article": {"target": 300, "max": 450, "min_tail": 80, "overlap": 40},
    "video":   {"target": 300, "max": 450, "min_tail": 80, "overlap": 40},
}

# --- Articles ----------------------------------------------------------------
# After cleaning, an article with fewer words than this counts as "no text"
# (it is a video post: linked to its video, or kept as a title-only stand-in).
ARTICLE_MIN_WORDS = 5
# Short posts containing one of these cues are tagged kind="announcement".
ANNOUNCEMENT_MAX_WORDS = 100
ANNOUNCEMENT_CUES = [
    r"\bdnes\b", r"\bzítra\b", r"\bpozítří\b", r"\bpřijďte\b", r"\bzváni\b",
    r"\bbudu (?:besedovat|mluvit|přednášet|povídat)", r"\bvystoupí", r"\bvystoupíme",
    r"\bzrušen", r"\brušíme\b", r"\bruší\b", r"\bse koná\b", r"\bpozvání\b",
    r"\blivestream", r"\bnakupujte\b", r"\bupgrad", r"\bkampaň", r"\bpřednášk",
]
# Lines produced by the site's video player; removed from article text.
VIDEO_EMBED_LINES = [
    "Váš prohlížeč nepodporuje přehrávání videa.",
    "Nejde-li video přehrát, prosím, zkuste to tady.",
    "Video v nižší kvalitě v menším souboru můžete stáhnout tady.",
    "Nefunguje-li předchozí link, prosím, zkuste to tady.",
]
# A title prefix ("Prefix: rest") is treated as a series name if used this often.
SERIES_MIN_COUNT = 2

# --- Article <-> video linking ----------------------------------------------
# Fuzzy title matches need a higher similarity the further apart the dates are:
# (max days apart, min similarity 0–1). None = any date.
# Matches further apart than the last tier are rejected, even with identical
# titles: recurring titles ("Stát proti chudým", "Streamaraton") are different videos.
LINK_TIERS = [(2, 0.65), (30, 0.85)]
# Series prefixes ("Prefix: title") stripped from titles before comparing them.
TITLE_PREFIXES = [
    "kanál svobodného přístavu", "studio svobodného přístavu",
    "livestream svobodného přístavu", "svobodný přístav", "anarchokapitalismus",
]
# Phrases removed from titles before matching ("V 19 hodin živě o …" → "…").
TITLE_NOISE = [
    r"^ve? \d{1,4}(?: hodin)? (?:živě|začíná)\s*", r"^ankap přednáška\s*[–-]?\s*", r"^o\s+",
]

# --- Videos ------------------------------------------------------------------
SHORT_VIDEO_SECONDS = 180     # flag videos shorter than this
THIN_DESCRIPTION_WORDS = 15   # flag videos whose cleaned description is shorter
# Lines in descriptions that are donation/support boilerplate.
DONATION_LINE_PATTERNS = [
    r"^\s*(?:BTC|LTC|XMR|ETH|LN)\s*:", r"číslo účtu", r"variabilní symbol",
    r"pravidelná podpora", r"opristavu\.urza\.cz",
    r"chcete-li podpořit", r"pokud se vám .* líbil", r"pošlete dobrovolný příspěvek",
    r"dobrovolný příspěvek", r"podpořit .* (?:v krypt|korun)",
]
# Tags dropped everywhere (in addition to any tag on > GENERIC_TAG_SHARE of videos).
GENERIC_TAGS = [
    "svoboda", "anarchokapitalismus", "kanál svobodného přístavu",
    "studio svobodného přístavu", "svobodný přístav", "urza",
]
GENERIC_TAG_SHARE = 0.5
# Tag synonyms: variant -> canonical (compared case-insensitively).
TAG_SYNONYMS = {
    "nap": "princip neagrese",
    "kryptopeníze": "kryptoměny",
    "kryptoměna": "kryptoměny",
    "rakouská ekonomie": "rakouská ekonomická škola",
    "svoboda projevu": "svoboda slova",
    "vzdělání": "vzdělávání",
    "anarchoagorism": "anarchoagorismus",
    "covid": "koronavirus",
    "covid-19": "koronavirus",
}

# --- Books -------------------------------------------------------------------
# Books are parsed straight from the EPUBs (see epub_books.py), following the
# book's own table of contents. Top-level TOC entries before `content_from`
# are front matter; from `content_until` on is back matter (publisher's pages,
# partner organisations, sponsors). Labels match by prefix.
BOOKS_DIR = REPO / "books"
BOOKS = {
    "ankap": {
        "epub": "ankap.epub",
        "title": "Anarchokapitalismus",
        "content_from": "Tři příběhy",
        "content_until": "Nakloněni svobodě",
    },
    "anag": {
        "epub": "anag.epub",
        "title": "Anarchoagorismus",
        "content_from": "Předmluva",
        "content_until": "Vydání knihy podpořili",
    },
}
# Sections whose path contains a label starting with one of these get this kind.
BOOK_KINDS = [("AMEN", "book_faq"), ("Tři příběhy", "book_story")]
# A section with fewer words than this (e.g. an aphorism page that holds only
# "– SEK3") is merged into the next section with the same parent.
BOOK_MERGE_MAX_WORDS = 8
