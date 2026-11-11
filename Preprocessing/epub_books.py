"""Parse the book EPUBs into sections that follow the book's own table of contents.

Why not use jsons/{ankap,anag}.json? That parser gives each XHTML file
one title. The EPUBs often hold several TOC sections in one file (all six
economics objections of the AMEN section are in one file; "Drogy" shares a file
with "dodatek"; the nine subsections of "19 Praxe" are in one file), so those
sections were merged under the wrong name. It also inserted spaces at inline
tag boundaries, splitting words ("pod</em><em>nikatelé" → "pod nikatelé").

Here the TOC (nav.xhtml) is the authority: every TOC entry starts a section at
its anchor, and its position in the TOC tree gives the section's path, e.g.
["dodatek", "AMEN: …", "Ekonomie", "Kartely"].
"""
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup, NavigableString, Tag

from textutils import normalize

LEAF_BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "pre", "td", "th"}
SKIP_TAGS = {"figure", "img", "script", "style", "svg", "nav", "head"}
SKIP_CLASSES = {"chapter-number", "role-label"}


def _text(el: Tag) -> str:
    # get_text("") keeps the source's own spacing, so words split across
    # inline tags (<em>pod</em><em>nikatelé</em>) stay whole.
    return re.sub(r"\s+", " ", normalize(el.get_text(""))).strip()


def _toc(z: zipfile.ZipFile, opf_dir: str, nav_path: str) -> list[dict]:
    soup = BeautifulSoup(z.read(nav_path), "html.parser")
    navs = soup.find_all("nav")
    toc = next((n for n in navs if "toc" in (n.get("epub:type", "") + " " + str(n.get("id", "")))), navs[0])
    nav_dir = posixpath.dirname(nav_path)
    entries = []

    def walk(ol, path):
        for li in ol.find_all("li", recursive=False):
            a = li.find("a")
            if a is None:
                continue
            label = re.sub(r"\s+", " ", normalize(a.get_text(" "))).strip()
            href = a.get("href", "")
            file, _, anchor = href.partition("#")
            entries.append({
                "label": label,
                "file": posixpath.normpath(posixpath.join(nav_dir, file)),
                "anchor": anchor or None,
                "path": path + [label],
            })
            sub = li.find("ol", recursive=False)
            if sub:
                walk(sub, path + [label])

    walk(toc.find("ol"), [])
    return entries


def _spine(z: zipfile.ZipFile) -> tuple[list[str], str]:
    def tags(root, name):
        return [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == name]

    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = tags(container, "rootfile")[0].attrib["full-path"]
    opf_dir = posixpath.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))
    manifest = {i.attrib["id"]: (posixpath.normpath(posixpath.join(opf_dir, i.attrib["href"])),
                                 i.attrib.get("properties", ""))
                for i in tags(opf, "item")}
    nav = next(p for p, props in manifest.values() if "nav" in props.split())
    spine = [manifest[r.attrib["idref"]][0] for r in tags(opf, "itemref")
             if r.attrib.get("linear", "yes") != "no" and r.attrib["idref"] in manifest]
    return [p for p in spine if p != nav], nav


def _render_comparison(section: Tag) -> str:
    """<section class="money-property"><h3>Vzácnost</h3><dl><dt>Bitcoin</dt><dd>…</dd>…"""
    title = section.find(["h2", "h3", "h4"])
    parts = []
    for dt in section.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        rating = dd.find(class_="rating") if dd else None
        rest = [_text(p) for p in dd.find_all("p") if p is not rating] if dd else []
        item = _text(dt) + (f" – {_text(rating)}" if rating else "")
        if rest:
            item += f" ({'; '.join(rest)})"
        parts.append(item)
    return f"{_text(title) if title else ''}: " + "; ".join(parts)


def parse_epub(path) -> list[dict]:
    """Return sections in reading order: {label, path, depth, text, files}."""
    z = zipfile.ZipFile(path)
    spine, nav = _spine(z)
    entries = _toc(z, "", nav)
    by_file_start = {}
    by_anchor = {}
    for e in entries:
        if e["anchor"]:
            by_anchor[(e["file"], e["anchor"])] = e
        else:
            by_file_start.setdefault(e["file"], e)

    sections: list[dict] = []
    cur = {"label": None, "path": [], "text": [], "files": []}   # text before the first TOC entry
    state = {"skip_heading": False, "after_objection": False, "file": None}

    def start(entry, file):
        nonlocal cur
        sections.append(cur)
        cur = {"label": entry["label"], "path": entry["path"], "text": [], "files": [file]}
        state["skip_heading"] = True

    def emit(text):
        if text:
            cur["text"].append(text)
            if state["file"] not in cur["files"]:
                cur["files"].append(state["file"])

    def walk(el: Tag, file: str):
        if not isinstance(el, Tag) or el.name in SKIP_TAGS:
            return
        classes = set(el.get("class", []))
        if el.get("id") and (file, el["id"]) in by_anchor:
            start(by_anchor[(file, el["id"])], file)
        if classes & SKIP_CLASSES:
            return
        if "money-property" in classes:
            emit(_render_comparison(el))
            return
        if el.name in LEAF_BLOCKS:
            t = _text(el)
            if el.name.startswith("h") and len(el.name) == 2:
                # The heading that opens a TOC section is its title, not body text.
                if state["skip_heading"]:
                    state["skip_heading"] = False
                    return
            else:
                state["skip_heading"] = False
            if "objection" in classes:
                t = f"Námitka: {t}"
                state["after_objection"] = True
            elif state["after_objection"] and t:
                t = f"Odpověď: {t}"
                state["after_objection"] = False
            emit(t)
            return
        for child in el.children:
            if isinstance(child, Tag):
                walk(child, file)
            elif isinstance(child, NavigableString) and child.strip() and el.name in ("div", "section", "blockquote", "body"):
                emit(re.sub(r"\s+", " ", normalize(str(child))).strip())

    for file in spine:
        state["file"] = file
        if file in by_file_start:
            start(by_file_start[file], file)
        soup = BeautifulSoup(z.read(file), "html.parser")
        body = soup.find("body") or soup
        for child in body.children:
            if isinstance(child, Tag):
                walk(child, file)
    sections.append(cur)

    out = []
    for s in sections:
        if s["label"] is None and not s["text"]:
            continue
        out.append({
            "label": s["label"] or "(před obsahem)",
            "path": s["path"] or ["(před obsahem)"],
            "depth": max(len(s["path"]) - 1, 0),
            "text": "\n\n".join(s["text"]),
            "files": [posixpath.basename(f) for f in s["files"]],
        })
    return out
