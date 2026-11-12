# Embedding report: variant `textonly`

Model **BAAI/bge-m3** on **cuda** (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44), 73.6 s, 2,528 chunks × 1024 dimensions. Created 2026-11-11T22:09:44.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| variant matches | OK | (textonly) |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2528, 1024) |
| finite | OK |  |
| normalised | OK | (norm 1.0000-1.0000) |
| no truncated chunks | OK | (longest 821 tokens) |

## Similarity distribution

Cosine similarity between all pairs of different chunks:

| 1% | 25% | median | 75% | 99% |
|---:|---:|---:|---:|---:|
| 0.299 | 0.396 | 0.442 | 0.492 | 0.635 |

## Do chunks cluster by source?

For each chunk, the share of its 10 nearest neighbours that come from the same source, compared with the share expected if neighbours were picked at random. A large gap means the embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.

| Source | Chunks | Same-source neighbours | Expected at random |
|---|---:|---:|---:|
| article | 956 | 70% | 38% |
| book | 592 | 77% | 23% |
| video | 980 | 78% | 39% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 70% | 74% | 38% |
| book | 77% | 61% | 23% |
| video | 78% | 67% | 39% |

Nothing is put in front of the chunks here, so the remaining clustering comes from the texts themselves: how each source writes (book prose, article, video description), their length and their typical subjects.

Nearest neighbour is another chunk of the same document for 12% of chunks. All sources (articles, books and videos) have documents split into several chunks; per source, the share is 1% for articles, 51% for books, 0% for videos.

## Near-duplicates (similarity > 0.95, different documents)

70 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 1.000 | Studio Svobodného přístavu: Streamaraton 2025 (13. až 17. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2025 (1. až 12. hodina) (video) |
| 1.000 | Studio Svobodného přístavu: Streamaraton 2026 do konce kampaně (13. až 15. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2026 do konce kampaně (1. až 12. hodina) (video) |
| 1.000 | Studio Svobodného přístavu: Streamaraton 2024 (13. až 16. hodina) (video) | Studio Svobodného přístavu: Streamaraton 2024 (1. až 12. hodina) (video) |
| 1.000 | Vypnutí prázdninového módu 2019 (article) | Vypnutí prázdninového módu 2020 (article) |
| 0.999 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Třiadvacet let od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Dvaadvacet let od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Sedmadvacet let od smrti Murrayho Newtona Rothbarda (article) | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Osmadvacet let od smrti Murrayho Newtona Rothbarda (article) | Devětadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.998 | Přepnutí Stok do prázdninového módu 2017 (article) | Přepnutí Stok do prázdninového módu 2019 (article) |
| 0.997 | Čtvrt století od smrti Murrayho Newtona Rothbarda (article) | Šestadvacet let od smrti Murrayho Newtona Rothbarda (article) |
| 0.997 | Přepnutí Stok do prázdninového módu 2017 (article) | Přepnutí Stok do prázdninového módu 2018 (article) |
| 0.997 | Sedmadvacet let od smrti Murrayho Newtona Rothbarda (article) | Devětadvacet let od smrti Murrayho Newtona Rothbarda (article) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “Kartely”, “10. Zdravotnictví”, “Drogy”, “Bitcoin”, “Demokracie”, “Školství”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**Kartely** (book)

- 0.765 · book · Monopoly
- 0.690 · book · 12. Silnice a dopravní pravidla
- 0.669 · article · Konkurenční boj v dlouhodobém měřítku
- 0.651 · book · Dumpingové ceny
- 0.650 · book · Pašeráci, dealeři, děvky a další svobodní psanci

**10. Zdravotnictví** (book)

- 0.774 · video · Anarchokapitalismus, díl čtrnáctý: Zdravotnictví
- 0.749 · book · Nekvalitní soukromé instituce
- 0.721 · book · Nemožnost ekonomické kalkulace v centrálním plánování
- 0.717 · article · Regulace škod z regulací
- 0.712 · book · Drogy

**Urza v Polis: Drogy** (article)

- 0.875 · article · Urza v Polis: Školství
- 0.856 · article · Urza v Polis: Zbraně
- 0.785 · article · Urza v Polis: Veřejný prostor a svoboda slova
- 0.774 · article · Urza v Polis: Hasiči
- 0.746 · article · Urza v Polis: Demokracie

**Budoucnost bitcoinu?** (article)

- 0.710 · article · Stát zaznamenal BTC
- 0.690 · book · 5. Peníze
- 0.687 · article · Kryptoměny a hospodářská krize
- 0.686 · video · Urza for BTC Prague 2025
- 0.678 · book · Kryptopeníze

**Urza v Polis: Demokracie** (article)

- 0.855 · article · Urza v Polis: Kultura
- 0.799 · article · Urza v Polis: Školství
- 0.757 · article · Urza v Polis: Hasiči
- 0.756 · article · Urza v Polis: Zbraně
- 0.751 · article · Urza v Polis: Veřejný prostor a svoboda slova

**Petice za svobodnější školství** (article)

- 0.689 · article · Vlastenectví jedné mladé dívenky
- 0.672 · article · Polští studenti na straně svobody
- 0.656 · video · O založení libertariánské strany
- 0.651 · article · Nový web Svobody učení
- 0.649 · article · Dejte státu korunu!
