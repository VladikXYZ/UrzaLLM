import json
import posixpath
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"]


def clean_str(text: str) -> str:
    """Normalizes Unicode (removes NBSP \u00a0), strips soft hyphens, and collapses spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\u200b", "")
    return " ".join(text.split())


def extract_clean_body(xhtml_bytes: bytes, chapter_title: str = "") -> str:
    """Extracts paragraph-level text without breaking inline tags onto new lines."""
    soup = BeautifulSoup(xhtml_bytes, "xml")
    body = soup.find("body") or soup

    for tag in body.find_all(["script", "style", "head", "title", "svg", "nav"]):
        tag.decompose()

    # Convert <br> tags to spaces so words don't glue together
    for br in body.find_all("br"):
        br.replace_with(" ")

    blocks = []
    for elem in body.find_all(BLOCK_TAGS):
        # Skip nested block elements (e.g., <p> inside <li> or <blockquote>) to prevent duplicates
        if elem.find_parent(BLOCK_TAGS):
            continue
        line = clean_str(elem.get_text(" ", strip=True))
        if line:
            blocks.append(line)

    # Fallback if the XHTML file puts text directly inside <div> or <body> without <p> tags
    if not blocks:
        raw_lines = body.get_text(separator="\n", strip=True).splitlines()
        blocks = [clean_str(line) for line in raw_lines if clean_str(line)]

    # Remove the first line if it just repeats the chapter title
    if chapter_title and blocks and blocks[0].lower() == clean_str(chapter_title).lower():
        blocks.pop(0)

    return "\n\n".join(blocks)


def get_toc_map(z: zipfile.ZipFile, opf_root: ET.Element, opf_dir: str, manifest: dict) -> dict:
    """Builds a mapping of {normalized_file_path: chapter_title} from NCX (EPUB 2) or NAV (EPUB 3)."""
    ns = {
        "opf": "http://www.idpf.org/2007/opf",
        "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    }
    toc_map = {}

    # 1. Try EPUB 2 toc.ncx
    for item in opf_root.findall(".//opf:manifest/opf:item", ns):
        if item.attrib.get("media-type") == "application/x-dtbncx+xml":
            ncx_path = posixpath.normpath(posixpath.join(opf_dir, item.attrib["href"]))
            if ncx_path in z.namelist():
                ncx = ET.fromstring(z.read(ncx_path))
                ncx_dir = posixpath.dirname(ncx_path)
                for nav_point in ncx.findall(".//ncx:navPoint", ns):
                    label = nav_point.find(".//ncx:navLabel/ncx:text", ns)
                    content = nav_point.find(".//ncx:content", ns)
                    if label is not None and content is not None and label.text:
                        src = content.attrib["src"].split("#")[0]
                        full_src = posixpath.normpath(posixpath.join(ncx_dir, src))
                        # Keep the top-most title if multiple anchors point to the same file
                        if full_src not in toc_map:
                            toc_map[full_src] = clean_str(label.text)

    # 2. Try EPUB 3 nav.xhtml (fills in any missing entries)
    for item in opf_root.findall(".//opf:manifest/opf:item", ns):
        if "nav" in item.attrib.get("properties", "").split():
            nav_path = posixpath.normpath(posixpath.join(opf_dir, item.attrib["href"]))
            if nav_path in z.namelist():
                nav_dir = posixpath.dirname(nav_path)
                soup = BeautifulSoup(z.read(nav_path), "xml")
                for a in soup.find_all("a", href=True):
                    src = a["href"].split("#")[0]
                    if not src:
                        continue
                    full_src = posixpath.normpath(posixpath.join(nav_dir, src))
                    title = clean_str(a.get_text(" ", strip=True))
                    if title and full_src not in toc_map:
                        toc_map[full_src] = title

    return toc_map


def parse_epub(epub_path: str, output_json_path: str = None) -> list[dict]:
    """Parses an EPUB into a list of {'name': ..., 'text': ...} dictionaries."""
    chapters = []

    with zipfile.ZipFile(epub_path, "r") as z:
        # 1. Locate content.opf via META-INF/container.xml
        container = ET.fromstring(z.read("META-INF/container.xml"))
        rootfile = container.find(
            ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
        )
        opf_path = rootfile.attrib["full-path"]
        opf_dir = posixpath.dirname(opf_path)

        # 2. Parse manifest and spine reading order
        opf_root = ET.fromstring(z.read(opf_path))
        ns = {"opf": "http://www.idpf.org/2007/opf"}

        manifest = {}
        for item in opf_root.findall(".//opf:manifest/opf:item", ns):
            item_id = item.attrib["id"]
            href = posixpath.normpath(posixpath.join(opf_dir, item.attrib["href"]))
            manifest[item_id] = href

        toc_map = get_toc_map(z, opf_root, opf_dir, manifest)

        spine_ids = [
            itemref.attrib["idref"]
            for itemref in opf_root.findall(".//opf:spine/opf:itemref", ns)
            if itemref.attrib.get("linear", "yes") != "no"
        ]

        # 3. Extract text in spine reading order
        current_title = None

        for idx, item_id in enumerate(spine_ids, 1):
            file_path = manifest.get(item_id)
            if not file_path or file_path not in z.namelist():
                continue

            # Skip navigation documents if they are listed in the spine
            if posixpath.basename(file_path).lower() in ("nav.xhtml", "toc.xhtml"):
                continue

            raw_bytes = z.read(file_path)

            # Determine title: 1. TOC Map, 2. First Heading inside file, 3. Previous/Fallback title
            title = toc_map.get(file_path)
            if not title:
                soup = BeautifulSoup(raw_bytes, "xml")
                heading = soup.find(["h1", "h2", "h3"])
                if heading and clean_str(heading.get_text(" ", strip=True)):
                    title = clean_str(heading.get_text(" ", strip=True))
                elif current_title:
                    # Continuation file of the current chapter
                    title = current_title
                else:
                    title = f"Chapter {idx}"

            current_title = title
            text = extract_clean_body(raw_bytes, chapter_title=title)

            if not text:
                continue

            # Merge if this spine file belongs to the same chapter as the previous one
            if chapters and chapters[-1]["name"] == title:
                chapters[-1]["text"] += "\n\n" + text
            else:
                chapters.append({"name": title, "text": text})

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(chapters, f, ensure_ascii=False, indent=4)

    return chapters


if __name__ == "__main__":
    data = parse_epub("books/ankap.epub", "jsons/ankap.json")
    print(f"Extracted {len(data)} chapters.")

    data = parse_epub("books/anag.epub", "jsons/anag.json")
    print(f"Extracted {len(data)} chapters.")