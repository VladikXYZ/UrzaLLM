# Reports on real_data

Urza's articles, video descriptions and book paragraphs from `real_data/` (2,911 chunks: 956 articles, 980 video
descriptions cleaned of donation boilerplate, 975 book paragraphs), each embedded from its own Czech text by two
models on the GPU. The reports show the English translation of every chunk; everything computed uses the Czech.

| Model | Folder | Size |
|---|---|---|
| bge-m3 (`BAAI/bge-m3`) | [bge_m3/](bge_m3/) | 1,024 numbers per chunk |
| EmbeddingGemma (`google/embeddinggemma-300m`) | [embeddinggemma_2/](embeddinggemma_2/) | 768 numbers per chunk |

## Reports (in each model folder)

| Report | What it covers |
|---|---|
| `projection_compare.html` | All sources: UMAP maps of raw vs source-centred embeddings, same-source neighbours |
| `book_map.html` | Books only: UMAP and t-SNE maps labelled by chapter, how tightly each chapter clusters |
| `book_pca_ica.html` | Books, linear projections: PCA (variance, cumulative variance, maps) and ICA themes |
| `book_trails.html` | Link graph of the book sections, its backbone and a path explorer |
| `book_nearest.html` | Each book topic points to its nearest topic; the families this forms |
| `all_pca_ica.html` | All chunks, linear projections: PCA axes and how much each follows the source, ICA themes |
| `timeline_map.html` | Ideas over time: the centred map with a year slider |
| `topic_distribution.html` | Articles and videos assigned to book sections: shares, split between the books, fit |
| `political_space.html` | Urza in the general political space (326 items): coverage, depth, blind spots (bge-m3 only, see below) |
| `embedding_report.md` | Checks on the embeddings, similarity, same-source clustering, near-duplicates |

`figures/` holds static PNGs of the maps; the HTML files are self-contained.

**EmbeddingGemma has no political-space report yet:** the 326 political items have to be embedded with the same
model, exactly as the chunks were (same prompt or none):
`python Questions/embed_questions.py --model google/embeddinggemma-300m --name embeddinggemma_2 [--prompt-name …]`.
Then rerun `python Chunks/build_real_reports.py --models embeddinggemma_2 --only political_space`.

## Other pages here

| Page | What it covers |
|---|---|
| [presentation.html](presentation.html) | One-page presentation of the reports above, with a bge-m3 / EmbeddingGemma switch in every section |
| [embedder_comparison.html](embedder_comparison.html) | bge-m3 vs EmbeddingGemma on the same chunks: maps, neighbour agreement, similarity scale |
| [projection_all_sources.html](projection_all_sources.html) | All four sources including the 91,065 video-transcript passages, raw vs centred, with chapter labels |
| [projection_book_space.html](projection_book_space.html) | The 114 book headings as a space with the other sources placed into it |

## Rebuilding

```
python Chunks/real_data.py              # real_data/ -> Chunks/output_real/ (chunk table + both models' vectors)
python Chunks/build_real_reports.py     # all reports, both models -> RealReports/<model>/
python Chunks/presentation_real.py      # presentation.html, from the reports
python Chunks/compare_embedders.py      # embedder_comparison.html
python Chunks/project_all_sources.py    # projection_all_sources.html (UMAPs cached; --recompute to redo)
python Chunks/project_book_space.py     # projection_book_space.html (cached; --recompute to redo)
```
