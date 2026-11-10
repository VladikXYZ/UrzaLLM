import json
import re
import time
import unicodedata
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://stoky.urza.cz/autori/urza-2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}

import time
import unicodedata
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}


def clean_text(text):
    """Normalizes unicode (removes NBSP \u00a0) and strips extra whitespace."""
    return unicodedata.normalize("NFKC", text).replace("\u00ad", "").strip()


def scrape_article_text(session, url, article_title=""):
    """Fetches an individual article URL and extracts the main text

    without using any regular expressions.
    """
    resp = session.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main = soup.find("main")
    if not main:
        return ""

    # Remove unwanted metadata wrappers, menus, and side navigation elements
    for unwanted in main.find_all(
        ["table", "h2", "section", "nav", "script", "style", "div"]
    ):
        if unwanted.name == "div" and "menuItem" not in unwanted.get(
            "class", []
        ):
            continue
        unwanted.decompose()

    # Extract clean paragraph text blocks
    paragraphs = [
        clean_text(p.get_text(" ", strip=True))
        for p in main.find_all("p")
        if p.get_text(strip=True)
    ]

    if paragraphs:
        raw_text = "\n\n".join(paragraphs)
    else:
        # Fallback if the layout doesn't use <p> tags
        raw_text = clean_text(main.get_text(separator="\n", strip=True))

    # Split into lines to clean up metadata headers and footers safely
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 1. Remove the top metadata ("Autor:", "Urza", "Čas: ...") if present
    if len(lines) >= 3 and lines[0].startswith("Autor:") and lines[2].startswith("Čas:"):
        lines = lines[3:]

    # 2. Remove the repeated title at the very top if it's there
    if article_title and lines and lines[0] == clean_text(article_title):
        lines.pop(0)

    # 3. Remove the bottom footer stats ("Přečtení: ..." and "» komentáře ... «")
    if lines and "komentáře" in lines[-1]:
        lines.pop()
    if lines and lines[-1].startswith("Přečtení:"):
        lines.pop()

    return "\n".join(lines).strip()


def get_total_pages(session):
    """Finds the highest page number from the pagination row on page 1."""
    resp = session.get(BASE_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="dripsTable")
    if not table:
        return 1

    page_numbers = [
        int(a.get_text(strip=True))
        for a in table.find_all("a")
        if a.get_text(strip=True).isdigit()
    ]
    return max(page_numbers) if page_numbers else 1



def scrape_urza(fetch_full_text=False, output_file="urza_articles.json"):
    session = requests.Session()
    total_pages = get_total_pages(session)
    print(f"Found {total_pages} pages to scrape.")

    articles = []

    for page in range(1, total_pages + 1):
        url = BASE_URL if page == 1 else f"{BASE_URL}-{page}-strana"
        print(f"Scraping page {page}/{total_pages}: {url}")

        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="dripsTable")
        if not table:
            continue

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            # Skip header and pagination rows (which use colspan="2")
            if len(cells) != 2:
                continue

            date_str = clean_text(cells[0].get_text(strip=True))
            link_tag = cells[1].find("a")
            if not link_tag:
                continue

            title = clean_text(link_tag.get_text(strip=True))
            article_url = link_tag["href"]

            item = {
                "date": date_str,
                "title": title,
                "url": article_url,
            }

            if fetch_full_text:
                print(f"  -> Fetching text: {title}")
                item["text"] = scrape_article_text(session, article_url)
                time.sleep(0.3)  # Polite delay between article requests

            articles.append(item)

        time.sleep(0.5)  # Polite delay between index pages

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(articles)} articles to {output_file}")


if __name__ == "__main__":
    # Set fetch_full_text=True if you also want to open every article link and save its body
    scrape_urza(fetch_full_text=True)