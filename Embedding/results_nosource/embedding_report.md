# Embedding report: variant `nosource`

Model **BAAI/bge-m3** on **cuda** (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44), 82.8 s, 2,528 chunks × 1024 dimensions. Created 2026-11-11T21:33:54.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| variant matches | OK | (nosource) |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2528, 1024) |
| finite | OK |  |
| normalised | OK | (norm 1.0000-1.0000) |
| no truncated chunks | OK | (longest 831 tokens) |

## Similarity distribution

Cosine similarity between all pairs of different chunks:

| 1% | 25% | median | 75% | 99% |
|---:|---:|---:|---:|---:|
| 0.299 | 0.398 | 0.444 | 0.494 | 0.639 |

## Do chunks cluster by source?

For each chunk, the share of its 10 nearest neighbours that come from the same source, compared with the share expected if neighbours were picked at random. A large gap means the embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.

| Source | Chunks | Same-source neighbours | Expected at random |
|---|---:|---:|---:|
| article | 956 | 75% | 38% |
| book | 592 | 88% | 23% |
| video | 980 | 84% | 39% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 75% | 75% | 38% |
| book | 88% | 76% | 23% |
| video | 84% | 70% | 39% |

Here the context line no longer names the source, but still carries a chapter path or a title and year that differ by source; the `textonly` variant removes it entirely.

Nearest neighbour is another chunk of the same document for 19% of chunks. All sources (articles, books and videos) have documents split into several chunks; per source, the share is 2% for articles, 78% for books, 0% for videos.

## Near-duplicates (similarity > 0.95, different documents)

67 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 0.999 | Studio Svobodného přístavu: Streamaraton 2025 (13. až 17. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2025 (1. až 12. hodina) (video) |
| 0.999 | Studio Svobodného přístavu: Streamaraton 2024 (13. až 16. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2024 (1. až 12. hodina) (video) |
| 0.996 | Studio Svobodného přístavu: Streamaraton 2026 do konce kampaně (13. až 15. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2026 do konce kampaně (1. až 12. hodina) (video) |
| 0.996 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.994 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.994 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.993 | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) | Devětadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.993 | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.992 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) |
| 0.992 | Gabriela a Urza: Věk je jen číslo aneb přátelství v JbK (article) | Gabriela a Urza: Věk je jen číslo aneb přátelství v JbK (snad už opravdu) (article) |
| 0.992 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) |
| 0.992 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.991 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Devětadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.991 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.991 | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) | Sedmadvacet let od smrti Murrayho Newtona Rothbarda (article) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “Kartely”, “10. Zdravotnictví”, “Drogy”, “Bitcoin”, “Demokracie”, “Školství”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**Kartely** (book)

- 0.770 · book · Monopoly
- 0.677 · book · 12. Silnice a dopravní pravidla
- 0.670 · book · Dumpingové ceny
- 0.664 · article · Konkurenční boj v dlouhodobém měřítku
- 0.656 · book · Drogy

**10. Zdravotnictví** (book)

- 0.776 · video · Anarchokapitalismus, díl čtrnáctý: Zdravotnictví
- 0.720 · book · Nekvalitní soukromé instituce
- 0.690 · book · Drogy
- 0.690 · book · XX. Boření mýtů
- 0.687 · book · 2. Proč selhává centrální plánování?

**Urza v Polis: Drogy** (article)

- 0.883 · article · Urza v Polis: Zbraně
- 0.874 · article · Urza v Polis: Školství
- 0.795 · article · Urza v Polis: Veřejný prostor a svoboda slova
- 0.788 · article · Urza v Polis: Hasiči
- 0.776 · article · Urza v Polis: Demokracie

**Budoucnost bitcoinu?** (article)

- 0.714 · article · Stát zaznamenal BTC
- 0.696 · video · Urza for BTC Prague 2025
- 0.685 · article · Kryptoměny a hospodářská krize
- 0.681 · article · Zrušení hotovosti
- 0.670 · book · 5. Peníze

**Urza v Polis: Demokracie** (article)

- 0.856 · article · Urza v Polis: Kultura
- 0.826 · article · Urza v Polis: Školství
- 0.783 · article · Urza v Polis: Veřejný prostor a svoboda slova
- 0.780 · article · Urza v Polis: Zbraně
- 0.776 · article · Urza v Polis: Drogy

**Petice za svobodnější školství** (article)

- 0.685 · article · Vlastenectví jedné mladé dívenky
- 0.669 · article · Polští studenti na straně svobody
- 0.645 · video · Misesova akademie 2017: Svoboda učení
- 0.642 · article · Dejte státu korunu!
- 0.640 · video · O založení libertariánské strany
