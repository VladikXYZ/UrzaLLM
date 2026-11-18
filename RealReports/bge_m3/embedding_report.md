# Embedding report: bge-m3

Model **BAAI/bge-m3** on **GPU (friend's machine)** (real_data/embeddings_out_cs), 2,911 chunks × 1024 dimensions. Created 2026-11-18T20:30:47.

## Checks

| Check | Result | Detail |
|---|---|---|
| not a dry run | OK |  |
| embedded text = current chunks.parquet | OK |  |
| chunk ids and order match | OK |  |
| shape | OK | (2911, 1024) |
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
| book | 975 | 80% | 33% |
| video | 980 | 74% | 34% |

**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” and helps topics match across sources. Use them for the 2D map; for scoring a question against chunks, the raw vectors are the ones the model was trained for.

| Source | Same-source neighbours, raw | Centred | Expected at random |
|---|---:|---:|---:|
| article | 69% | 71% | 33% |
| book | 80% | 76% | 33% |
| video | 74% | 67% | 34% |

Nothing is put in front of the chunks here, so the remaining clustering comes from the texts themselves: how each source writes (book prose, article, video description), their length and their typical subjects.

Nearest neighbour is another chunk of the same document for 17% of chunks. Only books have documents split into several chunks; per source, the share is 51% for books.

## Near-duplicates (similarity > 0.95, different documents)

181 pairs.

| Similarity | Chunk A | Chunk B |
|---:|---|---|
| 1.000 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy II (video) |
| 1.000 | Synergeticum: The Different Faces of Anarchy III (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 1.000 | Synergeticum: The Different Faces of Anarchy II (video) | Synergeticum: The Different Faces of Anarchy I (video) |
| 1.000 | Ancap as Global Respect at Cherubín Teahouse: Part Two (article) | Ancap as Global Respect at the Cherubín Teahouse (article) |
| 1.000 | Freeport Studio: Streamarathon 2024 (1pm to 4pm) (video) | Freeport Studio: Streamarathon 2024 (Hour 1 to 12) (video) |
| 1.000 | Urza with Tim the President (video) | Urza with Tim about bans on (not only) abortions (video) |
| 1.000 | Urza with Tim the President (video) | Urza and Tim want to legalize cannibalism (video) |
| 1.000 | Urza with Tim the President (video) | Urza with Tim and sex with animals (video) |
| 1.000 | Urza with Tim the President (video) | Urza and Tim want to legalize incest (video) |
| 1.000 | Urza with Tim about bans on (not only) abortions (video) | Urza and Tim want to legalize cannibalism (video) |
| 1.000 | Urza with Tim about bans on (not only) abortions (video) | Urza with Tim and sex with animals (video) |
| 1.000 | Urza with Tim about bans on (not only) abortions (video) | Urza and Tim want to legalize incest (video) |
| 1.000 | Urza and Tim want to legalize cannibalism (video) | Urza with Tim and sex with animals (video) |
| 1.000 | Urza and Tim want to legalize cannibalism (video) | Urza and Tim want to legalize incest (video) |
| 1.000 | Urza with Tim and sex with animals (video) | Urza and Tim want to legalize incest (video) |

## Example neighbourhoods

Five nearest chunks from *other* documents, for the first chunk whose title contains each of: “Healthcare”, “Drugs”, “Bitcoin”, “Democracy”, “Education”. Ideally they share its topic; announcements and episodes of a recurring series often match other items of the same series instead.

**Healthcare** (book)

- 0.749 · book · How could free-market education or healthcare work? Just look at private schools or hospitals, they are no better than public ones!
- 0.698 · book · Impossibility of economic calculation in central planning
- 0.677 · article · Reforms
- 0.633 · video · Freeport Studio: Healthcare
- 0.632 · article · Regulations to Protect the Weak

**Drugs** (book)

- 0.744 · article · Medicine, Drugs, and Double Standards
- 0.685 · book · Smugglers, dealers, prostitutes and other free outlaws
- 0.681 · article · The War on Drugs
- 0.675 · article · Aggression or Non-Aggression? That Is the Question!
- 0.674 · article · Drugs and Weapons

**MUNI ECON: Bitcoin (Urza) versus fiat (Jan Jonáš) (volatile sound)** (video)

- 0.758 · video · Students for Liberty: Bitcoin (Urza) versus fiat (Mojmír Hampl)
- 0.714 · article · Tonight: Bitcoin (Anarchist Urza) vs. Fiat (Monetary Economist Jan Jonáš, Ph.D.)
- 0.634 · video · Freeport Studio: About (crypto) money with Josef Tětek
- 0.628 · video · Anarcho-Capitalism: The Bitcoinization
- 0.626 · video · Freeport Studio: About the economy with Lukáš Kovanda

**Democracy** (book)

- 0.737 · book · Indoctrination
- 0.704 · article · Freedom? Democracy?
- 0.667 · article · The Problem Lies in the People
- 0.665 · book · Education and propaganda
- 0.664 · book · State

**Education (Anarchokapitalismus)** (book)

- 0.747 · video · 4D for the separation of education from the state!
- 0.700 · book · Education and propaganda
- 0.699 · book · Education and propaganda
- 0.680 · book · Education (Anarchoagorismus)
- 0.674 · article · Mandatory Kindergarten
