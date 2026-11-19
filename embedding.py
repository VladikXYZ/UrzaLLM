import json
import gc
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 1. Configuration
INPUT_JSON = Path("real_data/transcripts_urza_clanker/transcripts_urza.json")
OUTPUT_DIR = Path("real_data/embeddings_out_rollo_cs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define the models to benchmark
# Note: Verify the exact HF repo ID if using a local path or specific 0.6B/2B variant
MODELS = {
    "embeddinggemma_2": "google/embeddinggemma-2",
    # "qwen3_0.6b": "Qwen/Qwen3-Embedding-0.6B",
    # "qwen3_4b": "Qwen/Qwen3-Embedding-4B",
    "bge_m3": "BAAI/bge-m3",
    # "qwen3_8b": "Qwen/Qwen3-Embedding-8B",
}

# 2. Load data and extract ONLY 'text_cs'
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

texts_to_embed = [item["context_cs"]+item["text_cs"] for item in raw_data]

print(f"Loaded {len(texts_to_embed)} text chunks for embedding (ignoring titles/subtitles).")

# Save the aligned metadata once so you can map row_index -> original record later
metadata_index_path = OUTPUT_DIR / "metadata_index.json"
with open(metadata_index_path, "w", encoding="utf-8") as f:
    json.dump(texts_to_embed, f, ensure_ascii=False, indent=2)

# 3. Loop through models and generate embeddings
for short_name, model_id in MODELS.items():
    print(f"\n--- Loading {model_id} ---")

    # Adjust batch_size downward for larger 4B/8B models to prevent CUDA OOM
    batch_size = 16 if "8B" in model_id or "4B" in model_id else 32

    try:
        model = SentenceTransformer(
            model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16}  # Saves 50% VRAM on modern GPUs
        )

        # Encode strictly the text_cs list
        # For document/corpus passages, Qwen3 and Gemma expect raw text without query prefixes
        embeddings = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalization for fast cosine similarity via dot product
            convert_to_numpy=True
        )

        # Save compressed float32/float16 matrix: shape (N_chunks, embedding_dim)
        out_file = OUTPUT_DIR / f"{short_name}_vectors.npy"
        np.save(out_file, embeddings)
        print(f"Saved {short_name} embeddings with shape {embeddings.shape} to {out_file}")

    except Exception as e:
        print(f"Error running {model_id}: {e}")

    finally:
        # Crucial: Free VRAM before loading the next multi-billion parameter model
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()