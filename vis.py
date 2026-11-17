import json
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go

# 1. Configuration
EMBEDDINGS_DIR = Path("embeddings_out")
METADATA_FILE = EMBEDDINGS_DIR / "metadata_index.json"
OUTPUT_HTML = EMBEDDINGS_DIR / "embeddings_explorer.html"

# Map display names to your saved .npy filenames
VECTOR_FILES = {
    "EmbeddingGemma-2": EMBEDDINGS_DIR / "embeddinggemma_2_vectors.npy",
    "Qwen3-0.6B": EMBEDDINGS_DIR / "qwen3_0.6b_vectors.npy",
    "Qwen3-4B": EMBEDDINGS_DIR / "qwen3_4b_vectors.npy",
    "BGE-M3": EMBEDDINGS_DIR / "bge_m3_vectors.npy",
}

# 2. Load Metadata & Format Hover Tooltips
# 2. Load Metadata & Format Hover Tooltips
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

if records and isinstance(records[0], str):
    with open("chunked/chunked.json", "r", encoding="utf-8") as f:
        raw_cs = json.load(f)
    records = [item for item in raw_cs if item.get("text_cs", "").strip()]

def format_hover_html(item) -> str:
    if isinstance(item, str):
        item = {"text_cs": item}

    title = item.get("title_cs", "").strip()
    subtitle = item.get("subtitle_cs", "").strip()
    text = item.get("text_cs", "").strip()
    src = item.get("src", "").strip()

    wrapped_text = "<br>".join(textwrap.wrap(text, width=65))
    if len(wrapped_text) > 600:
        wrapped_text = wrapped_text[:600] + "..."

    parts = []
    if title:
        parts.append(f"<b>{title}</b>")
    if subtitle:
        parts.append(f"<i>{subtitle}</i>")

    parts.append(f"<br>{wrapped_text}")

    if src:
        parts.append(f"<br><span style='color:#636efa;'>Source / URL: {src}</span>")

    return "<br>".join(parts)

df_meta = pd.DataFrame([r if isinstance(r, dict) else {"text_cs": r} for r in records])
df_meta["hover_html"] = [format_hover_html(r) for r in records]
df_meta["type_clean"] = df_meta.get("type", pd.Series([""] * len(df_meta))).fillna("").str.strip().str.lower()
df_meta["src_clean"] = df_meta.get("src", pd.Series([""] * len(df_meta))).fillna("").str.strip().str.lower()
df_meta["title_clean"] = df_meta.get("title_cs", pd.Series([""] * len(df_meta))).fillna("").str.strip().replace("", "Uncategorized")

# 3. Run PCA, ICA, UMAP, and t-SNE for each model
projections = {}  # Key: (model_name, method_name) -> (x, y)

for model_name, vec_path in VECTOR_FILES.items():
    if not vec_path.exists():
        print(f"Skipping {model_name} (file not found: {vec_path})")
        continue

    print(f"Computing 2D projections for {model_name}...")
    X = np.load(vec_path).astype(np.float32)

    # PCA
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    projections[(model_name, "PCA")] = pca_2d

    # ICA (pre-whiten via 50-dim PCA first for numerical stability & speed on high-dim vectors)
    X_pca50 = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    ica_2d = FastICA(n_components=2, random_state=42, max_iter=500).fit_transform(X_pca50)
    projections[(model_name, "ICA")] = ica_2d

    # UMAP (cosine metric works best for L2-normalized text embeddings)
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        compatibility_layout=False,
        n_jobs=-1
    ).fit_transform(X)
    projections[(model_name, "UMAP")] = umap_2d

    # t-SNE (perplexity=30 is standard for ~3,000 chunks)
    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=42
    ).fit_transform(X)
    projections[(model_name, "t-SNE")] = tsne_2d

# 4. Build Interactive Plotly Figure with Dropdown Selector
# 4. Build Interactive Plotly Figure with Dropdown Selector
fig = go.Figure()
combo_keys = list(projections.keys())

# Match both singular and plural 'type' values ("book" or "books", etc.)
is_book_mask = df_meta["type_clean"].isin(["book", "books"]).values
book_titles = sorted(df_meta.loc[is_book_mask, "title_clean"].unique())

trace_specs = []

# 1. Non-book traces (hidden from side legend: showlegend=False)
articles_mask = (~is_book_mask) & df_meta["type_clean"].isin(["article", "articles"]).values
videos_mask = (~is_book_mask) & df_meta["type_clean"].isin(["video", "videos"]).values
anag_other_mask = (~is_book_mask) & (~articles_mask) & (~videos_mask) & (df_meta["src_clean"] == "anag").values
ancap_other_mask = (~is_book_mask) & (~articles_mask) & (~videos_mask) & (df_meta["src_clean"] == "ancap").values

if articles_mask.any():
    trace_specs.append(("Articles", articles_mask, "#FFFFFF", False))
if videos_mask.any():
    trace_specs.append(("Videos", videos_mask, "#FF2B2B", False))
if anag_other_mask.any():
    trace_specs.append(("Anag", anag_other_mask, "#888888", False))
if ancap_other_mask.any():
    trace_specs.append(("Ancap", ancap_other_mask, "#FFD700", False))

# Catch-all for any remaining non-book items
covered_mask = is_book_mask | articles_mask | videos_mask | anag_other_mask | ancap_other_mask
if (~covered_mask).any():
    trace_specs.append(("Other", ~covered_mask, "#AAAAAA", False))

# 2. Book traces (visible in side legend: showlegend=True, single hex color per trace)
for title in book_titles:
    t_mask = is_book_mask & (df_meta["title_clean"] == title).values
    # Check what 'src' this book chapter belongs to ("ancap" -> yellow, "anag" -> gray)
    srcs_in_title = set(df_meta.loc[t_mask, "src_clean"])
    if "ancap" in srcs_in_title:
        color = "#FFD700"  # Yellow
    elif "anag" in srcs_in_title:
        color = "#888888"  # Gray
    else:
        color = "#FFD700"  # Default yellow for books
    trace_specs.append((title, t_mask, color, True))

# Add traces for every (Model, Method) combination
for idx, (model_name, method_name) in enumerate(combo_keys):
    coords = projections[(model_name, method_name)]
    is_visible = (idx == 0)

    for name, mask, color, show_in_legend in trace_specs:
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=show_in_legend,
                visible=is_visible,
                marker=dict(size=6, opacity=0.85, color=color),  # Single hex string works in WebGL!
                customdata=df_meta.loc[mask, "hover_html"].values,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

# 5. Export Standalone HTML Website
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
print(f"\nInteractive visualization saved to: {OUTPUT_HTML.resolve()}")