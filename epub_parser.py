"""Parse the EPUB books into topic-level sections.

Output: a list of {"chapter": ..., "name": ..., "text": ...} dictionaries, one per
section, where a section is the text between two consecutive headings (h1-h4).
"name" is the section's own heading, "chapter" the path of headings above it
("" for top-level chapters), e.g.
    {"chapter": "AMEN: analýza mnohých etatistických námitek › Ekonomie", "name": "Kartely"}
Aphorism pages (anag) use the aphorism with its author as the name:
    {"chapter": "AMEN: aforismy manifestující esenci neposlušnosti",
     "name": "„Dobří lidé neposlouchají špatné zákony.“ – neznámý autor"}

Why EPUB and not PDF: the EPUB is generated from structured source (one XHTML file
per section, semantic headings, <p> per paragraph). The PDF has hard line breaks,
hyphenation across lines, page numbers inside the text and no paragraph boundaries.

Front/back matter (cover, imprint, table of contents, supporter lists, adverts) is
dropped: content starts at the first file with <body class="body"> and stops at the
first heading listed in BACK_MATTER_HEADINGS or with a supporter-* class.

Usage:
    python epub_parser.py                 # parses data/*.epub -> mat_czech/<book>.json
    python epub_parser.py --max-depth 1   # one entry per top-level chapter instead
"""

import argparse
import json
import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, Tag

NAME_SEP = " › "
HEADINGS = ["h1", "h2", "h3", "h4"]
BLOCKS = ["p", "li", "pre", "dl"]

# <body class="..."> values that never contain book content
SKIP_BODY_CLASSES = {"cover", "front", "imprint", "titlepage", "contents", "art-page"}
# A heading with one of these texts (or a supporter-* class) ends the book content
BACK_MATTER_HEADINGS = {"vydání knihy podpořili", "nakloněni svobodě", "poděkování"}
# Pure container headings left out of "chapter" ("Dodatek › AMEN: …" -> "AMEN: …")
CONTAINER_HEADINGS = {"dodatek"}

NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
}


def clean_str(text: str) -> str:
    """Normalize Unicode (NBSP -> space), drop soft hyphens/zero-width spaces, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "").replace("​", "")
    return " ".join(text.split())


def clean_heading(text: str) -> str:
    text = clean_str(text).lstrip("•").strip()
    return text[:1].upper() + text[1:]  # "dodatek" -> "Dodatek"


def spine_files(z: zipfile.ZipFile) -> list[str]:
    """XHTML files in reading order, from META-INF/container.xml -> content.opf spine."""
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = container.find(".//c:rootfile", NS).attrib["full-path"]
    opf_dir = posixpath.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))
    manifest = {
        item.attrib["id"]: posixpath.normpath(posixpath.join(opf_dir, item.attrib["href"]))
        for item in opf.findall(".//opf:manifest/opf:item", NS)
    }
    files = []
    for ref in opf.findall(".//opf:spine/opf:itemref", NS):
        path = manifest.get(ref.attrib["idref"])
        if path and path in z.namelist() and ref.attrib.get("linear", "yes") != "no":
            files.append(path)
    return files


def book_title(z: zipfile.ZipFile) -> str:
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf = ET.fromstring(z.read(container.find(".//c:rootfile", NS).attrib["full-path"]))
    title = opf.find(".//{http://purl.org/dc/elements/1.1/}title")
    return clean_str(title.text) if title is not None and title.text else ""


def chapter_number(body: Tag) -> str:
    """Chapter numbers are images: <div class="chapter-number"><img alt="Kapitola 1"/></div>."""
    div = body.find("div", class_="chapter-number")
    if not div:
        return ""
    img = div.find("img")
    m = re.search(r"\d+", (img.get("alt", "") if img else "") + " " + div.get_text(" "))
    return m.group(0) if m else ""


def dl_rows(dl: Tag) -> list[str]:
    """Render a <dl> comparison table as 'term: value – note' lines."""
    rows, term = [], None
    for child in dl.find_all(["dt", "dd"], recursive=False):
        txt = clean_str(" – ".join(child.stripped_strings) if child.name == "dd" else child.get_text(" "))
        if child.name == "dt":
            term = txt
        elif txt:
            rows.append(f"{term}: {txt}" if term else txt)
    return rows


class SectionBuilder:
    """Accumulates paragraphs under a stack of headings and emits leaf sections."""

    def __init__(self, max_depth: int | None):
        self.max_depth = max_depth
        self.stack: list[tuple[int, str]] = []  # (level, heading)
        self.paras: list[str] = []
        self.sections: list[dict] = []

    def heading(self, level: int, text: str):
        if self.max_depth and level > self.max_depth:
            self.para(text)  # too deep: keep the heading as an inline line
            return
        if not self.paras and self.stack and self.stack[-1][0] == level:
            # Same-level headings with nothing in between (e.g. several quotes on
            # one page) share the text that follows.
            self.stack[-1] = (level, f"{self.stack[-1][1]} / {text}")
            return
        self.flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    def para(self, text: str):
        if text:
            self.paras.append(text)

    def flush(self):
        if self.paras:
            path = [h for _, h in self.stack] or ["Úvod"]
            chapter = NAME_SEP.join(h for h in path[:-1] if h.lower() not in CONTAINER_HEADINGS)
            self.sections.append({"chapter": chapter, "name": path[-1], "text": "\n\n".join(self.paras)})
        self.paras = []


def parse_epub(epub_path: str, output_json_path: str | None = None, max_depth: int | None = None) -> list[dict]:
    """Parse an EPUB into a list of {'chapter': parent headings, 'name': heading, 'text': section text}."""
    builder = SectionBuilder(max_depth)
    started = False

    with zipfile.ZipFile(epub_path) as z:
        for path in spine_files(z):
            body = BeautifulSoup(z.read(path), "xml").find("body")
            if body is None:
                continue
            classes = set(body.get("class", "").split()) if isinstance(body.get("class"), str) else set(body.get("class") or [])
            if classes & SKIP_BODY_CLASSES:
                continue
            if not started:
                if "body" not in classes:  # dedication/thanks pages before the first real chapter
                    continue
                started = True

            for tag in body.find_all(["script", "style", "nav", "figure", "svg"]):
                tag.decompose()
            for br in body.find_all("br"):
                br.replace_with(" ")

            if "quote-page" in classes:
                # One page = one aphorism (sometimes several variants); the commentary
                # follows on the next, heading-less page. The section is named after
                # the aphorism(s) with author: „…“ – author; the text is the commentary.
                quotes = []
                for bq in body.find_all("blockquote"):
                    h = bq.find(HEADINGS)
                    if not h:
                        continue
                    attr = bq.find("p", class_="attribution")
                    attr = clean_str(attr.get_text(" ")) if attr else ""
                    quotes.append(f"„{clean_heading(h.get_text(' '))}“ {attr}".strip())
                if quotes:
                    builder.heading(2, " / ".join(quotes))
                continue

            num = chapter_number(body)
            inline_headings = "comparison" in classes  # table sub-headings stay inside the section

            stop = False
            for el in body.find_all(HEADINGS + BLOCKS):
                if el.find_parent(BLOCKS):
                    continue  # nested block (<p> in <li>, <dd> content): parent covers it
                if el.name in HEADINGS:
                    text = clean_heading(el.get_text(" "))
                    cls = " ".join(el.get("class") or [])
                    if text.lower() in BACK_MATTER_HEADINGS or "supporter" in cls:
                        stop = True
                        break
                    if inline_headings:
                        builder.para(text)
                        continue
                    if el.name == "h1" and num:
                        text = f"{num}. {text}"
                    builder.heading(int(el.name[1]), text)
                elif el.name == "dl":
                    for row in dl_rows(el):
                        builder.para(row)
                else:
                    builder.para(clean_str(el.get_text()))
            if stop:
                break

    builder.flush()
    sections = builder.sections

    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=4)
    return sections


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("epubs", nargs="*", help="EPUB files (default: data/*.epub)")
    ap.add_argument("--out-dir", default="mat_czech")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="split only on headings up to this level (1 = chapters); default: all levels")
    args = ap.parse_args()

    epubs = args.epubs or sorted(str(p) for p in Path("data").glob("*.epub"))
    for epub in epubs:
        out = Path(args.out_dir) / f"{Path(epub).stem}.json"
        data = parse_epub(epub, str(out), max_depth=args.max_depth)
        chars = sum(len(s["text"]) for s in data)
        print(f"{epub}: {len(data)} sections, {chars:,} chars -> {out}")


if __name__ == "__main__":
    main()
