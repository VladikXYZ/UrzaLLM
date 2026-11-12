# Final reports

All reports regenerated from the final corpus (2026-11-12): Urza's articles, video descriptions and two books,
chunked by `Chunks/chunk.py` from the preprocessed data in `preprocessed/`, embedded with bge-m3 in three variants.
**The reports are in English:** every chunk is shown through its English translation (text, title, chapter and
section names) and the political-space items through `Questions/political_space_en.tsv`, while everything computed
(the embeddings, the embedded texts) is the Czech original. Book titles (*Anarchokapitalismus*, *Anarchoagorismus*),
people's names and the embedded Czech context lines are kept as they are.
Rebuild everything with `python Chunks/build_reports.py` (see `Chunks/README.md`). The earlier reports, built on
the earlier pipeline's chunks, stay in `Reports/` for comparison.

## Corpus

| Source | Chunk | Chunks |
|---|---|---:|
| Articles | the whole article text | 956 |
| Videos | the whole description, without the recurring donation/signature boilerplate (34% of video words) | 978 |
| Books | one paragraph | 976 |

2,910 chunks in Czech (embedded) and the same 2,910 in English (aligned by `chunk_id`). Compared with the earlier
chunking, articles and videos are the same; the books are finer (976 paragraphs of median 101 words, instead of
592 chunks of about 170), and the opening stories (“Three stories”, mostly dialogue) became 167 short chunks, which
shows up in several reports.

## Embedding variants

| Variant | Text in front of each chunk |
|---|---|
| `withsource` | context line naming the source (Czech, as embedded): “Kniha Anarchokapitalismus › 10. Zdravotnictví” (Book … › 10. Healthcare), “Článek „…“ (2017)” (Article), “Oznámení „…“” (Announcement), “Video „…“ (2026; hosté: …)” (guests) |
| `nosource` | the same without the source word: “Anarchokapitalismus › 10. Zdravotnictví”, “„…“ (2017)” |
| `textonly` | nothing: only the chunk's own text |

## Reports

| Report | What it covers |
|---|---|
| [chunking_report.html](chunking_report.html) | The final chunks: rules, counts and lengths, boilerplate removal, Czech–English alignment, odd chunks, comparison with the earlier chunking (replaces the preprocessing report) |
| [embedding_report_withsource.md](embedding_report_withsource.md), [_nosource](embedding_report_nosource.md), [_textonly](embedding_report_textonly.md) | Checks on the embeddings, similarity distribution, same-source clustering, near-duplicates, example neighbours |
| [projection_compare_withsource.html](projection_compare_withsource.html), [_nosource](projection_compare_nosource.html), [_textonly](projection_compare_textonly.html) | All sources: UMAP maps of raw vs source-centred embeddings; the textonly report compares the three variants |
| [book_map_nosource.html](book_map_nosource.html), [book_map_textonly.html](book_map_textonly.html) | Books only: UMAP and t-SNE maps labelled by chapter, neighbour shares, how tightly each chapter clusters |
| [book_pca_ica_textonly.html](book_pca_ica_textonly.html) | Books, linear projections: PCA (variance, cumulative variance, PC maps) and ICA (16 independent themes) |
| [book_trails_textonly.html](book_trails_textonly.html) | Link graph of the 65 book sections, its backbone and a path explorer |
| [book_nearest_textonly.html](book_nearest_textonly.html) | Each of the 56 book topics points to its nearest topic; the families this forms |
| [all_pca_ica_textonly.html](all_pca_ica_textonly.html) | All chunks, linear projections: PCA axes with source η², ICA themes vs single-source components |
| [timeline_map_textonly.html](timeline_map_textonly.html) | Evolution over time: map with a year slider, books as reference, shift between the two books' themes |
| [topic_distribution_textonly.html](topic_distribution_textonly.html) | Articles and videos assigned to book sections: shares, split between the books, fit, robustness |
| [political_space_textonly.html](political_space_textonly.html) | Urza in the general political space (326 items): coverage, depth, blind spots, map with three placements |

`figures/` holds static PNGs of the maps; the HTML files are self-contained.

## Findings

**Source clustering (all chunks).** Same-source share of each chunk's 10 nearest neighbours, full vectors (raw / source-centred):

| Source | withsource | nosource | textonly | Random |
|---|---:|---:|---:|---:|
| Articles | 77% / 73% | 74% / 73% | 69% / 71% | 33% |
| Books | 95% / 89% | 94% / 89% | 80% / 76% | 34% |
| Videos | 86% / 70% | 80% / 70% | 74% / 67% | 34% |

The source word itself changes little; the rest of the context line (chapter path, title, year) matters more,
especially for books (94% → 80% raw). What remains in `textonly` comes from the texts: how each source writes and
what it writes about.

**Chapters as ideas (books only).** Same-chapter share of a book chunk's 10 nearest neighbours: 73% with the chapter
path in front (`nosource`), 48% on content alone (`textonly`), against 5% by chance; the label accounts for about 37%
of the above-chance clustering (38% in the earlier data). On content alone, chapters are about 6× tighter than a
random set on the UMAP map, and 35 of 56 chapter groups form compact islands (spread below 0.20). t-SNE agrees on
which chapters are compact (rank correlation 0.92 for `textonly`, 0.82 for `nosource`). Regions of the map do
correspond to ideas.

**Linear structure.** No direction dominates. In the books, PC1 explains 8.0% and half the variance needs 40
components. With paragraph chunks, PC1 is no longer the difference between the books (η² 0.08); they separate
along PC2 (η² 0.33). ICA (16 components) finds 12 themes spread over several chapters, among them school, money,
non-aggression and guns. Over all chunks, PC1 (6.3%) is partly the kind of text (source η² 0.36); from PC2 on, and
in 17 of 20 ICA components, the source explains little. 8 ICA components are themes across sources; 12 are tied to
one source (recurring formats and series, or particular book chapters).

**Links between book topics.** 237 links between the 65 sections (lift ≥ 2), backbone of 63; trails such as
2. Education → 3. Indoctrination → 9. Democracy, and Drugs → Smugglers, dealers, prostitutes and other free
outlaws → Liberation Travel → 22. Love. Hubs: Myth busting, 8. State, 20. Participation, 21. The future; 13
backbone links join the two books. Only “Three stories” has no strong link. Nearest-topic arrows form 10 families
(e.g. 2. Why does central planning fail? ↔ AMEN: Economics, 8. Education ↔ 2. Education, 3. The principle of
non-aggression ↔ 11. Crime); the largest has 17 topics, almost all from Anarchoagorismus.

**Evolution over time.** The share of article and video chunks closest to *Anarchoagorismus* sections rises from
22% (2015–2018) to 54% (2023–2026), with the change around 2021 (no year before above 27%, none from 2021 below
37%), in articles (23% → 49%) and videos (20% → 49%) alike. Growing: the free school Urza co-founded (“Ježek bez
klece”, shown as “Hedgehog without a cage”, 0% → 9%), Free port, crypto-anarchy; shrinking: work, roads, guns,
education as an institution and the stories.

**Topic distribution.** Book chunks held out from their section land back in it 68% of the time (chance 1.5%).
Articles and videos reach all 58 topical sections; articles lean to *Anarchokapitalismus* (61%), videos to
*Anarchoagorismus* (56%). They fit the books less tightly than the books fit themselves (median 0.26 / 0.25 vs
0.42). The exact ranking of the most covered sections depends on the size of the space (2 of the top 5 kept at 50
dimensions, 4 at 184), the broad picture does not.

**Political space.** 93% of 168 political questions have at least a relevant Urza passage (≥ 0.55) and 54% a
strongly matching one (≥ 0.60); questions on his core subjects match his own texts at 0.64–0.72. The 326 items are
shown in English translation. By volume he
writes most about the state and public administration, law and justice, and education; his answers are least
direct (lowest depth) in agriculture, the environment, the EU and international relations, and energy.

## Data notes

- Three articles have titles that are only question marks (`art-0007`, `art-0050`, `art-0061`); they are like that
  in `preprocessed/arts.json` (probably emoji titles lost when scraping).
- 22 chunks have three words or fewer and 21 repeat another chunk's text; both are kept (see the chunking report).
