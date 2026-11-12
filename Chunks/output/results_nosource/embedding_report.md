# Embedding report: variant `nosource`

Model **BAAI/bge-m3** on **cuda** (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44), 84.8 s, 2,910 chunks × 1024 dimensions. Created 2026-11-12T14:01:13.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| variant matches | OK | (nosource) |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2910, 1024) |
| finite | OK |  |
| normalised | OK | (norm 1.0000-1.0000) |
| no truncated chunks | OK | (longest 1152 tokens) |

## Similarity distribution

Cosine similarity between all pairs of different chunks:

| 1% | 25% | median | 75% | 99% |
|---:|---:|---:|---:|---:|
| 0.281 | 0.383 | 0.431 | 0.485 | 0.636 |

## Do chunks cluster by source?

For each chunk, the share of its 10 nearest neighbours that come from the same source, compared with the share expected if neighbours were picked at random. A large gap means the embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.

| Source | Chunks | Same-source neighbours | Expected at random |
|---|---:|---:|---:|
| article | 956 | 74% | 33% |
| book | 976 | 94% | 34% |
| video | 978 | 80% | 34% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 74% | 73% | 33% |
| book | 94% | 89% | 34% |
| video | 80% | 70% | 34% |

Here the context line no longer names the source, but still carries a chapter path or a title and year that differ by source; the `textonly` variant removes it entirely.

Nearest neighbour is another chunk of the same document for 28% of chunks. Only books have documents split into several chunks; per source, the share is 85% for books.

## Near-duplicates (similarity > 0.95, different documents)

82 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 0.999 | Freeport Studio: Streamarathon 2025 (1pm to 5pm) (video) | Freeport Studio: Streamarathon 2025 (Hour 1 to 12) (video) |
| 0.998 | Freeport Studio: Streamarathon 2024 (1pm to 4pm) (video) | Freeport Studio: Streamarathon 2024 (Hour 1 to 12) (video) |
| 0.998 | Freeport Studio: Streamarathon 2026 until the end of the campaign (1pm to 3pm) (video) | Freeport Studio: Streamarathon 2026 until the end of the campaign (Hour 1 to 12) (video) |
| 0.996 | Twenty-three years since the death of Murray Newton Rothbard (article) | Twenty-two years since the death of Murray Newton Rothbard (article) |
| 0.996 | Ankap as global respect in Cherubín teahouse for the second time (article) | Ankap as global respect in the Cherubín tea house (article) |
| 0.996 | Synergeticum: The Different Faces of Anarchy II (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 0.994 | Twenty-six years since the death of Murray Newton Rothbard (article) | Twenty-two years since the death of Murray Newton Rothbard (article) |
| 0.994 | Twenty-six years since the death of Murray Newton Rothbard (article) | Twenty-three years since the death of Murray Newton Rothbard (article) |
| 0.993 | Twenty-nine years since the death of Murray Newton Rothbard (article) | Twenty-eight years since the death of Murray Newton Rothbard (article) |
| 0.993 | Twenty-six years since the death of Murray Newton Rothbard (article) | A quarter of a century since the death of Murray Newton Rothbard (article) |
| 0.993 | Gabriela and Urza: Age is just a number or friendship in JbK (maybe really) (article) | Gabriela and Urza: Age is just a number or friendship in JbK (article) |
| 0.993 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 0.992 | A quarter of a century since the death of Murray Newton Rothbard (article) | Twenty-three years since the death of Murray Newton Rothbard (article) |
| 0.992 | A quarter of a century since the death of Murray Newton Rothbard (article) | Twenty-two years since the death of Murray Newton Rothbard (article) |
| 0.992 | Twenty-eight years since the death of Murray Newton Rothbard (article) | Twenty-three years since the death of Murray Newton Rothbard (article) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “10. Healthcare”, “Drugs”, “Bitcoin”, “Democracy”, “Education”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**10. Healthcare** (book)

- 0.715 · book · How could free-market education or healthcare work? Just look at private schools or hospitals, they are no better than public ones!
- 0.686 · book · How could free-market education or healthcare work? Just look at private schools or hospitals, they are no better than public ones!
- 0.673 · video · Anarcho-Capitalism, Volume Fourteen: Healthcare
- 0.654 · book · Impossibility of economic calculation in central planning
- 0.646 · book · 9. Social system

**Freeport Conference 2023: Drugs and Guns Yesterday, Today and Tomorrow is happening... tomorrow!** (article)

- 0.863 · article · Freeport Conference 2024: The Technology of Yesterday, Today and Tomorrow is happening already... the day after tomorrow!
- 0.863 · article · Freeport Conference 2022: Money Yesterday, Today and Tomorrow is happening... tomorrow!
- 0.804 · article · Freeport Conference 2025: Culture Wars Yesterday, Today and Tomorrow is happening... tomorrow!
- 0.785 · article · Conference: The Totality of Yesterday, Today and Tomorrow takes place already tomorrow!
- 0.729 · article · Freeport Conference 2023: Drugs and Weapons Yesterday, Today and Tomorrow; even on February 18

**Tonight: Bitcoin (anarchist Urza) versus Fiat (monetary economist Ing. Jan Jonáš, Ph.D.)** (article)

- 0.842 · video · MUNI ECON: Bitcoin (Urza) versus fiat (Jan Jonáš) (volatile sound)
- 0.801 · article · Tonight: Bitcoin (anarchist Urza) versus Fiat (central banker Mojmír Hampl)
- 0.690 · video · Students for Liberty: Bitcoin (Urza) versus fiat (Mojmír Hampl)
- 0.667 · video · Anarcho-Capitalism: The Bitcoinization
- 0.657 · article · Urza Behind the Curtain: The Path to Ankap in Practice

**Freedom? Democracy?** (article)

- 0.764 · book · 3. Indoctrination
- 0.726 · article · Are we living in totality?
- 0.717 · book · 9. Democracy
- 0.712 · article · There is no evil as evil
- 0.702 · book · 9. Democracy

**No toilet, yes education?** (article)

- 0.732 · book · 2. Why does central planning fail?
- 0.713 · book · Education and propaganda
- 0.708 · article · Belief in socialism
- 0.700 · book · 2. Why does central planning fail?
- 0.667 · book · 2. Why does central planning fail?
