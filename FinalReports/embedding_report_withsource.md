# Embedding report: variant `withsource`

Model **BAAI/bge-m3** on **cuda** (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44), 86.7 s, 2,910 chunks × 1024 dimensions. Created 2026-11-12T13:59:48.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| variant matches | OK | (withsource) |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2910, 1024) |
| finite | OK |  |
| normalised | OK | (norm 1.0000-1.0000) |
| no truncated chunks | OK | (longest 1154 tokens) |

## Similarity distribution

Cosine similarity between all pairs of different chunks:

| 1% | 25% | median | 75% | 99% |
|---:|---:|---:|---:|---:|
| 0.280 | 0.384 | 0.433 | 0.488 | 0.641 |

## Do chunks cluster by source?

For each chunk, the share of its 10 nearest neighbours that come from the same source, compared with the share expected if neighbours were picked at random. A large gap means the embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.

| Source | Chunks | Same-source neighbours | Expected at random |
|---|---:|---:|---:|
| article | 956 | 77% | 33% |
| book | 976 | 95% | 34% |
| video | 978 | 86% | 34% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 77% | 73% | 33% |
| book | 95% | 89% | 34% |
| video | 86% | 70% | 34% |

Part of the remaining clustering likely comes from the context line at the start of each chunk (“Kniha …”, “Video „…“”, “Článek …” or “Oznámení …”), which names the source outright; the `nosource` and `textonly` variants test that.

Nearest neighbour is another chunk of the same document for 29% of chunks. Only books have documents split into several chunks; per source, the share is 85% for books.

## Near-duplicates (similarity > 0.95, different documents)

86 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 0.999 | Freeport Studio: Streamarathon 2025 (1pm to 5pm) (video) | Freeport Studio: Streamarathon 2025 (Hour 1 to 12) (video) |
| 0.999 | Freeport Studio: Streamarathon 2024 (1pm to 4pm) (video) | Freeport Studio: Streamarathon 2024 (Hour 1 to 12) (video) |
| 0.998 | Freeport Studio: Streamarathon 2026 until the end of the campaign (1pm to 3pm) (video) | Freeport Studio: Streamarathon 2026 until the end of the campaign (Hour 1 to 12) (video) |
| 0.996 | Twenty-three years since the death of Murray Newton Rothbard (article) | Twenty-two years since the death of Murray Newton Rothbard (article) |
| 0.996 | Synergeticum: The Different Faces of Anarchy II (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 0.994 | Twenty-six years since the death of Murray Newton Rothbard (article) | Twenty-two years since the death of Murray Newton Rothbard (article) |
| 0.994 | Twenty-six years since the death of Murray Newton Rothbard (article) | Twenty-three years since the death of Murray Newton Rothbard (article) |
| 0.994 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 0.993 | Gabriela and Urza: Age is just a number or friendship in JbK (maybe really) (article) | Gabriela and Urza: Age is just a number or friendship in JbK (article) |
| 0.993 | Twenty-six years since the death of Murray Newton Rothbard (article) | A quarter of a century since the death of Murray Newton Rothbard (article) |
| 0.993 | Twenty-nine years since the death of Murray Newton Rothbard (article) | Twenty-eight years since the death of Murray Newton Rothbard (article) |
| 0.993 | A quarter of a century since the death of Murray Newton Rothbard (article) | Twenty-three years since the death of Murray Newton Rothbard (article) |
| 0.993 | A quarter of a century since the death of Murray Newton Rothbard (article) | Twenty-two years since the death of Murray Newton Rothbard (article) |
| 0.992 | Twenty-eight years since the death of Murray Newton Rothbard (article) | Twenty-three years since the death of Murray Newton Rothbard (article) |
| 0.992 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy II (video) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “10. Healthcare”, “Drugs”, “Bitcoin”, “Democracy”, “Education”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**10. Healthcare** (book)

- 0.716 · book · How could free-market education or healthcare work? Just look at private schools or hospitals, they are no better than public ones!
- 0.690 · book · How could free-market education or healthcare work? Just look at private schools or hospitals, they are no better than public ones!
- 0.662 · video · Anarcho-Capitalism, Volume Fourteen: Healthcare
- 0.658 · book · Impossibility of economic calculation in central planning
- 0.651 · book · 9. Social system

**Freeport Conference 2023: Drugs and Guns Yesterday, Today and Tomorrow is happening... tomorrow!** (article)

- 0.877 · article · Freeport Conference 2022: Money Yesterday, Today and Tomorrow is happening... tomorrow!
- 0.866 · article · Freeport Conference 2024: The Technology of Yesterday, Today and Tomorrow is happening already... the day after tomorrow!
- 0.808 · article · Freeport Conference 2025: Culture Wars Yesterday, Today and Tomorrow is happening... tomorrow!
- 0.785 · article · Conference: The Totality of Yesterday, Today and Tomorrow takes place already tomorrow!
- 0.733 · article · Freeport Conference 2023: Drugs and Weapons Yesterday, Today and Tomorrow; even on February 18

**Tonight: Bitcoin (anarchist Urza) versus Fiat (monetary economist Ing. Jan Jonáš, Ph.D.)** (article)

- 0.834 · video · MUNI ECON: Bitcoin (Urza) versus fiat (Jan Jonáš) (volatile sound)
- 0.807 · article · Tonight: Bitcoin (anarchist Urza) versus Fiat (central banker Mojmír Hampl)
- 0.698 · video · Students for Liberty: Bitcoin (Urza) versus fiat (Mojmír Hampl)
- 0.664 · video · Anarcho-Capitalism: The Bitcoinization
- 0.646 · article · Urza Behind the Curtain: The Path to Ankap in Practice

**Freedom? Democracy?** (article)

- 0.762 · book · 3. Indoctrination
- 0.724 · article · Are we living in totality?
- 0.720 · article · There is no evil as evil
- 0.715 · book · 9. Democracy
- 0.702 · article · The limits of democracy

**No toilet, yes education?** (article)

- 0.718 · book · 2. Why does central planning fail?
- 0.713 · book · Education and propaganda
- 0.712 · article · Belief in socialism
- 0.689 · book · 2. Why does central planning fail?
- 0.664 · book · 2. Why does central planning fail?
