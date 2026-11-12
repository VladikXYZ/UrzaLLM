# Embedding report: variant `textonly`

Model **BAAI/bge-m3** on **cuda** (Linux-7.3.3-1-cachyos-x86_64-with-glibc2.44), 78.1 s, 2,910 chunks × 1024 dimensions. Created 2026-11-12T14:02:33.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| variant matches | OK | (textonly) |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2910, 1024) |
| finite | OK |  |
| normalised | OK | (norm 1.0000-1.0000) |
| no truncated chunks | OK | (longest 1141 tokens) |

## Similarity distribution

Cosine similarity between all pairs of different chunks:

| 1% | 25% | median | 75% | 99% |
|---:|---:|---:|---:|---:|
| 0.272 | 0.377 | 0.426 | 0.480 | 0.626 |

## Do chunks cluster by source?

For each chunk, the share of its 10 nearest neighbours that come from the same source, compared with the share expected if neighbours were picked at random. A large gap means the embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.

| Source | Chunks | Same-source neighbours | Expected at random |
|---|---:|---:|---:|
| article | 956 | 69% | 33% |
| book | 976 | 80% | 34% |
| video | 978 | 74% | 34% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 69% | 71% | 33% |
| book | 80% | 76% | 34% |
| video | 74% | 67% | 34% |

Nothing is put in front of the chunks here, so the remaining clustering comes from the texts themselves: how each source writes (book prose, article, video description), their length and their typical subjects.

Nearest neighbour is another chunk of the same document for 17% of chunks. Only books have documents split into several chunks; per source, the share is 52% for books.

## Near-duplicates (similarity > 0.95, different documents)

179 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 1.000 | ???? (article) | ?? (article) |
| 1.000 | Ankap as global respect in Cherubín teahouse for the second time (article) | Ankap as global respect in the Cherubín tea house (article) |
| 1.000 | Christmas party for supporters of Přístav 2023! (article) | Christmas party for Port supporters! (article) |
| 1.000 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy II (video) |
| 1.000 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 1.000 | Synergeticum: The Different Faces of Anarchy II (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 1.000 | #disobedience: Respect (article) | #disobedience: Symbolƴ̇(ka) (article) |
| 1.000 | #disobedience: Respect (article) | #disobedience: Freedom of expression (article) |
| 1.000 | #disobedience: Respect (article) | #disobedience: Why does Urza publicly break the law on camera? (article) |
| 1.000 | #disobedience: Symbolƴ̇(ka) (article) | #disobedience: Freedom of expression (article) |
| 1.000 | #disobedience: Symbolƴ̇(ka) (article) | #disobedience: Why does Urza publicly break the law on camera? (article) |
| 1.000 | #disobedience: Freedom of expression (article) | #disobedience: Why does Urza publicly break the law on camera? (article) |
| 1.000 | Freeport Conference 2024: The Technology of Yesterday, Today and Tomorrow is happening already... the day after tomorrow! (article) | Freeport Conference 2023: Drugs and Guns Yesterday, Today and Tomorrow is happening... tomorrow! (article) |
| 1.000 | Freeport Studio: Streamarathon 2026 until the end of the campaign (1pm to 3pm) (video) | Freeport Studio: Streamarathon 2026 until the end of the campaign (Hour 1 to 12) (video) |
| 1.000 | Freeport Studio: Streamarathon 2025 (1pm to 5pm) (video) | Freeport Studio: Streamarathon 2025 (Hour 1 to 12) (video) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “10. Healthcare”, “Drugs”, “Bitcoin”, “Democracy”, “Education”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**10. Healthcare** (book)

- 0.748 · book · How could free-market education or healthcare work? Just look at private schools or hospitals, they are no better than public ones!
- 0.697 · book · Impossibility of economic calculation in central planning
- 0.677 · article · Reforms
- 0.633 · video · Freeport Studio: Healthcare
- 0.632 · article · Regulation to protect the weak

**Freeport Conference 2023: Drugs and Guns Yesterday, Today and Tomorrow is happening... tomorrow!** (article)

- 1.000 · article · Freeport Conference 2024: The Technology of Yesterday, Today and Tomorrow is happening already... the day after tomorrow!
- 0.961 · article · Freeport Conference 2022: Money Yesterday, Today and Tomorrow is happening... tomorrow!
- 0.944 · article · Conference: The Totality of Yesterday, Today and Tomorrow takes place already tomorrow!
- 0.793 · article · Freeport Conference 2025: Culture Wars Yesterday, Today and Tomorrow is happening... tomorrow!
- 0.567 · article · Anarch evening at Café Kampus

**Tonight: Bitcoin (anarchist Urza) versus Fiat (monetary economist Ing. Jan Jonáš, Ph.D.)** (article)

- 0.775 · article · Tonight: Bitcoin (anarchist Urza) versus Fiat (central banker Mojmír Hampl)
- 0.714 · video · MUNI ECON: Bitcoin (Urza) versus fiat (Jan Jonáš) (volatile sound)
- 0.649 · video · Students for Liberty: Bitcoin (Urza) versus fiat (Mojmír Hampl)
- 0.644 · article · Urza Behind the Curtain: The Path to Ankap in Practice
- 0.640 · article · Pepa Tétek Behind the Curtain: Bitcoin

**Freedom? Democracy?** (article)

- 0.808 · book · 3. Indoctrination
- 0.747 · book · 9. Democracy
- 0.742 · article · Are we living in totality?
- 0.729 · book · 9. Democracy
- 0.725 · book · 10. Statism

**No toilet, yes education?** (article)

- 0.747 · book · 2. Why does central planning fail?
- 0.747 · book · 2. Why does central planning fail?
- 0.720 · book · Education and propaganda
- 0.706 · article · Belief in socialism
- 0.701 · book · 10. Statism
