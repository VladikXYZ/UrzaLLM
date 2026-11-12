# Embedding report: variant `withsource`

Model **BAAI/bge-m3** on **cuda** (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44), 121.3 s, 2,528 chunks × 1024 dimensions. Created 2026-11-11T20:33:47.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| variant matches | OK | (withsource) |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2528, 1024) |
| finite | OK |  |
| normalised | OK | (norm 1.0000-1.0000) |
| no truncated chunks | OK | (longest 834 tokens) |

## Similarity distribution

Cosine similarity between all pairs of different chunks:

| 1% | 25% | median | 75% | 99% |
|---:|---:|---:|---:|---:|
| 0.297 | 0.398 | 0.445 | 0.497 | 0.643 |

## Do chunks cluster by source?

For each chunk, the share of its 10 nearest neighbours that come from the same source, compared with the share expected if neighbours were picked at random. A large gap means the embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.

| Source | Chunks | Same-source neighbours | Expected at random |
|---|---:|---:|---:|
| article | 956 | 79% | 38% |
| book | 592 | 90% | 23% |
| video | 980 | 88% | 39% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 79% | 75% | 38% |
| book | 90% | 75% | 23% |
| video | 88% | 70% | 39% |

Part of the remaining clustering likely comes from the context line at the start of each chunk (“Kniha …”, “Video „…“”, “Článek …” or “Oznámení …”), which names the source outright; the `nosource` and `textonly` variants test that.

Nearest neighbour is another chunk of the same document for 19% of chunks. All sources (articles, books and videos) have documents split into several chunks; per source, the share is 1% for articles, 79% for books, 0% for videos.

## Near-duplicates (similarity > 0.95, different documents)

72 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 0.999 | Studio Svobodného přístavu: Streamaraton 2025 (13. až 17. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2025 (1. až 12. hodina) (video) |
| 0.999 | Studio Svobodného přístavu: Streamaraton 2024 (13. až 16. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2024 (1. až 12. hodina) (video) |
| 0.998 | Studio Svobodného přístavu: Streamaraton 2026 do konce kampaně (13. až 15. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2026 do konce kampaně (1. až 12. hodina) (video) |
| 0.996 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.994 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.994 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.994 | Gabriela a Urza: Věk je jen číslo aneb přátelství v JbK (article) | Gabriela a Urza: Věk je jen číslo aneb přátelství v JbK (snad už opravdu) (article) |
| 0.993 | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.993 | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) | Devětadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.993 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) |
| 0.993 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) |
| 0.992 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.991 | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) | Sedmadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.991 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.991 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Devětadvacet let od smrti Murrayho Newtona Rothbarda (article) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “Kartely”, “10. Zdravotnictví”, “Drogy”, “Bitcoin”, “Demokracie”, “Školství”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**Kartely** (book)

- 0.772 · book · Monopoly
- 0.675 · book · Dumpingové ceny
- 0.675 · book · 12. Silnice a dopravní pravidla
- 0.661 · book · Pašeráci, dealeři, děvky a další svobodní psanci
- 0.661 · book · Drogy

**10. Zdravotnictví** (book)

- 0.771 · video · Anarchokapitalismus, díl čtrnáctý: Zdravotnictví
- 0.722 · book · Nekvalitní soukromé instituce
- 0.696 · book · Drogy
- 0.686 · book · XX. Boření mýtů
- 0.683 · book · 2. Proč selhává centrální plánování?

**Urza v Polis: Drogy** (article)

- 0.885 · article · Urza v Polis: Zbraně
- 0.866 · article · Urza v Polis: Školství
- 0.796 · article · Urza v Polis: Veřejný prostor a svoboda slova
- 0.785 · article · Urza v Polis: Hasiči
- 0.778 · article · Urza v Polis: Demokracie

**Budoucnost bitcoinu?** (article)

- 0.716 · article · Stát zaznamenal BTC
- 0.702 · video · Urza for BTC Prague 2025
- 0.684 · article · Kryptoměny a hospodářská krize
- 0.683 · article · Zrušení hotovosti
- 0.667 · book · 5. Peníze

**Urza v Polis: Demokracie** (article)

- 0.865 · article · Urza v Polis: Kultura
- 0.823 · article · Urza v Polis: Školství
- 0.778 · article · Urza v Polis: Drogy
- 0.776 · article · Urza v Polis: Veřejný prostor a svoboda slova
- 0.775 · article · Urza v Polis: Zbraně

**Petice za svobodnější školství** (article)

- 0.686 · article · Vlastenectví jedné mladé dívenky
- 0.669 · article · Polští studenti na straně svobody
- 0.653 · article · Dejte státu korunu!
- 0.641 · video · Misesova akademie 2017: Svoboda učení
- 0.640 · article · Nový web Svobody učení
