#!/usr/bin/env python3
"""
chunk_stats.py - Comprehensive statistics for UrzaLLM chunks.

Calculates chunk counts, word count metrics (total, mean, median, std dev,
percentiles), character statistics, lexical diversity, and source breakdowns
for books, articles, videos, and transcriptions.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_dataset(
    chunks_path: Optional[str] = None,
    trans_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Loads chunk and transcription JSON datasets with automatic path resolution."""
    resolved_paths: Dict[str, str] = {}
    all_items: List[Dict[str, Any]] = []

    # 1. Resolve chunks file (books, articles, videos)
    chunk_candidates = [
        chunks_path,
        "real_data/chunked.json",
        "chunked/chunked.json",
        "chunked/chunks.json",
    ]
    resolved_chunks = None
    for cand in chunk_candidates:
        if cand and os.path.exists(cand):
            resolved_chunks = cand
            break

    if resolved_chunks:
        resolved_paths["chunks"] = resolved_chunks
        print(f"Loading chunks from: {resolved_chunks}")
        with open(resolved_chunks, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_items.extend(data)
    else:
        print("Warning: Could not find chunks dataset (books, articles, videos).", file=sys.stderr)

    # 2. Resolve transcriptions file
    trans_candidates = [
        trans_path,
        "real_data/trans_chunks.json",
        "transcription/chunks.json",
        "chunked/trans_chunks.json"
    ]
    resolved_trans = None
    for cand in trans_candidates:
        if cand and os.path.exists(cand):
            resolved_trans = cand
            break

    if resolved_trans:
        resolved_paths["transcriptions"] = resolved_trans
        print(f"Loading transcriptions from: {resolved_trans}")
        with open(resolved_trans, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_items.extend(data)
    else:
        print("Warning: Could not find transcriptions dataset.", file=sys.stderr)

    return all_items, resolved_paths


def calculate_category_stats(
    items: List[Dict[str, Any]],
    lang: str = "cs",
    top_n_words: int = 5
) -> Dict[str, Any]:
    """Calculates detailed metrics for a subset of chunks."""
    text_key = f"text_{lang}" if f"text_{lang}" in items[0] else "text_cs" if "text_cs" in items[0] else "text"
    
    word_counts = []
    char_counts = []
    sources = set()
    vocabulary_counter = Counter()

    for item in items:
        raw_text = item.get(text_key, "") or ""
        # Word tokenization (standard whitespace split for consistent word count)
        words = raw_text.split()
        word_counts.append(len(words))
        char_counts.append(len(raw_text))

        src = item.get("src") or item.get("title_cs") or item.get("title_en") or "unknown"
        sources.add(src)

        # Normalized tokenization for vocabulary analysis
        tokens = re.findall(r"\b\w+\b", raw_text.lower())
        vocabulary_counter.update(tokens)

    n_chunks = len(items)
    if n_chunks == 0:
        return {}

    w_arr = np.array(word_counts, dtype=np.float64)
    c_arr = np.array(char_counts, dtype=np.float64)

    total_words = int(np.sum(w_arr))
    mean_words = float(np.mean(w_arr))
    median_words = float(np.median(w_arr))
    std_words = float(np.std(w_arr))
    var_words = float(np.var(w_arr))
    min_words = int(np.min(w_arr))
    max_words = int(np.max(w_arr))
    q1_words = float(np.percentile(w_arr, 25))
    q3_words = float(np.percentile(w_arr, 75))
    iqr_words = q3_words - q1_words
    p90_words = float(np.percentile(w_arr, 90))
    p95_words = float(np.percentile(w_arr, 95))
    p99_words = float(np.percentile(w_arr, 99))

    # Skewness (Pearson's moment coefficient of skewness)
    skewness = float(np.mean(((w_arr - mean_words) / std_words) ** 3)) if std_words > 0 else 0.0

    # Character statistics
    total_chars = int(np.sum(c_arr))
    mean_chars = float(np.mean(c_arr))
    avg_word_length = float(total_chars / total_words) if total_words > 0 else 0.0

    # Vocabulary & Lexical richness
    total_tokens = sum(vocabulary_counter.values())
    vocab_size = len(vocabulary_counter)
    ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0
    hapax_legomena = sum(1 for count in vocabulary_counter.values() if count == 1)
    hapax_ratio = (hapax_legomena / vocab_size) if vocab_size > 0 else 0.0

    # Raw most frequent words
    top_raw_words = vocabulary_counter.most_common(top_n_words)

    return {
        "n_chunks": n_chunks,
        "n_sources": len(sources),
        "total_words": total_words,
        "mean_words": mean_words,
        "median_words": median_words,
        "std_words": std_words,
        "var_words": var_words,
        "min_words": min_words,
        "max_words": max_words,
        "q1_words": q1_words,
        "q3_words": q3_words,
        "iqr_words": iqr_words,
        "p90_words": p90_words,
        "p95_words": p95_words,
        "p99_words": p99_words,
        "skewness": skewness,
        "total_chars": total_chars,
        "mean_chars": mean_chars,
        "avg_word_len": avg_word_length,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "ttr": ttr,
        "hapax_legomena": hapax_legomena,
        "hapax_ratio": hapax_ratio,
        "top_keywords": top_raw_words,
        "word_counts_raw": word_counts  # Keep for plotting
    }


def print_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None):
    """Prints a clean ASCII/Unicode table formatted with borders and alignment."""
    if title:
        print(f"\n{'=' * 88}\n  {title}\n{'=' * 88}")

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    # Header row
    header_str = " | ".join(h.center(col_widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)

    # Data rows
    for row in rows:
        formatted_row = []
        for i, val in enumerate(row):
            sval = str(val)
            # Right-align numeric columns, left-align first text column
            if i == 0:
                formatted_row.append(sval.ljust(col_widths[i]))
            else:
                formatted_row.append(sval.rjust(col_widths[i]))
        print(" | ".join(formatted_row))


def display_results(stats: Dict[str, Dict[str, Any]], lang: str):
    """Outputs structured summary and detailed statistical tables."""
    total_corpus_chunks = stats["Total"]["n_chunks"]
    total_corpus_words = stats["Total"]["total_words"]

    # 1. Primary Overview Table
    overview_headers = ["Category", "Chunks", "Chunks %", "Total Words", "Words %", "Mean Words", "Median", "Std Dev"]
    overview_rows = []
    
    display_order = ["Books", "Articles", "Videos", "Transcriptions", "Total"]
    for cat in display_order:
        if cat not in stats or not stats[cat]:
            continue
        s = stats[cat]
        chunk_pct = f"{s['n_chunks'] / total_corpus_chunks * 100:.2f}%" if total_corpus_chunks > 0 else "0%"
        word_pct = f"{s['total_words'] / total_corpus_words * 100:.2f}%" if total_corpus_words > 0 else "0%"
        overview_rows.append([
            cat,
            f"{s['n_chunks']:,}",
            chunk_pct,
            f"{s['total_words']:,}",
            word_pct,
            f"{s['mean_words']:.2f}",
            f"{s['median_words']:.1f}",
            f"{s['std_words']:.2f}"
        ])

    print_table(overview_headers, overview_rows, f"URZALLM CORPUS STATISTICS (Language: {lang.upper()})")

    # 2. Detailed Distribution & Quantiles Table
    dist_headers = ["Category", "Min", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "IQR", "P90", "P99", "Max", "Skewness"]
    dist_rows = []
    for cat in display_order:
        if cat not in stats or not stats[cat]:
            continue
        s = stats[cat]
        dist_rows.append([
            cat,
            f"{s['min_words']:,}",
            f"{s['q1_words']:.1f}",
            f"{s['median_words']:.1f}",
            f"{s['q3_words']:.1f}",
            f"{s['iqr_words']:.1f}",
            f"{s['p90_words']:.1f}",
            f"{s['p99_words']:.1f}",
            f"{s['max_words']:,}",
            f"{s['skewness']:+.2f}"
        ])
    print_table(dist_headers, dist_rows, "WORD COUNT DISTRIBUTION & QUANTILES")

    # 3. Lexical Diversity, Vocabulary & Characters Table
    lex_headers = ["Category", "Sources", "Unique Vocab", "TTR %", "Hapax %", "Total Chars", "Mean Chars", "Avg Word Len"]
    lex_rows = []
    for cat in display_order:
        if cat not in stats or not stats[cat]:
            continue
        s = stats[cat]
        lex_rows.append([
            cat,
            f"{s['n_sources']:,}",
            f"{s['vocab_size']:,}",
            f"{s['ttr'] * 100:.2f}%",
            f"{s['hapax_ratio'] * 100:.2f}%",
            f"{s['total_chars']:,}",
            f"{s['mean_chars']:.1f}",
            f"{s['avg_word_len']:.2f} chars"
        ])
    print_table(lex_headers, lex_rows, "LEXICAL DIVERSITY & CHARACTER METRICS")

    # 4. Book Breakdown (ancap vs anag)
    if "Book: ancap" in stats and "Book: anag" in stats:
        book_headers = ["Book Title / Source", "Chunks", "Total Words", "Mean Words", "Median", "Std Dev", "Min/Max"]
        book_rows = []
        for b_name in ["Book: ancap", "Book: anag"]:
            s = stats[b_name]
            book_rows.append([
                b_name.replace("Book: ", ""),
                f"{s['n_chunks']:,}",
                f"{s['total_words']:,}",
                f"{s['mean_words']:.2f}",
                f"{s['median_words']:.1f}",
                f"{s['std_words']:.2f}",
                f"{s['min_words']} / {s['max_words']}"
            ])
        print_table(book_headers, book_rows, "BOOK SUB-BREAKDOWN")

    # 5. Top Most Frequent Words (Raw)
    print(f"\n{'=' * 88}\n  TOP MOST FREQUENT WORDS PER CATEGORY (RAW DATA)\n{'=' * 88}")
    for cat in ["Books", "Articles", "Videos", "Transcriptions"]:
        if cat in stats and stats[cat].get("top_keywords"):
            kw_formatted = ", ".join(f"{word} ({cnt:,})" for word, cnt in stats[cat]["top_keywords"])
            print(f"  * {cat.ljust(15)}: {kw_formatted}")
    print(f"{'=' * 88}\n")


def generate_plot(stats: Dict[str, Dict[str, Any]], output_path: str = "real_data/chunk_statistics_plot.png"):
    """Generates a publication-quality dark-themed visual analysis of chunk statistics."""
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation.", file=sys.stderr)
        return

    categories = ["Books", "Articles", "Videos", "Transcriptions"]
    colors = {
        "Books": "#4CAF50",         # Green
        "Articles": "#2196F3",      # Blue
        "Videos": "#FF9800",        # Orange
        "Transcriptions": "#FF2B2B" # Red
    }

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#1e1e2f")

    # 1. Boxplot of word counts (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor("#1e1e2f")
    data_to_plot = [stats[cat]["word_counts_raw"] for cat in categories if cat in stats]
    labels_to_plot = [cat for cat in categories if cat in stats]

    box = ax1.boxplot(
        data_to_plot,
        tick_labels=labels_to_plot,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False
    )
    for patch, cat in zip(box["boxes"], labels_to_plot):
        patch.set_facecolor(colors.get(cat, "#888888"))
        patch.set_alpha(0.8)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    for mean in box["means"]:
        mean.set_color("#FFFF00")
        mean.set_linewidth(1.5)

    ax1.set_title("Word Count Distribution per Chunk (Excluding Outliers)", color="white", fontsize=12, pad=10)
    ax1.set_ylabel("Words per Chunk", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, linestyle="--", alpha=0.2, color="white")

    # 2. Total Words & Chunk Share (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor("#1e1e2f")
    total_words = [stats[c]["total_words"] for c in categories if c in stats]
    bar_colors = [colors[c] for c in categories if c in stats]
    bars = ax2.bar(labels_to_plot, total_words, color=bar_colors, alpha=0.85, edgecolor="black")
    ax2.set_yscale("log")
    ax2.set_title("Total Word Count per Category (Log Scale)", color="white", fontsize=12, pad=10)
    ax2.set_ylabel("Total Words (log)", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, linestyle="--", alpha=0.2, color="white")
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            color="white", fontsize=9
        )

    # 3. Mean vs Median Words Comparison (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor("#1e1e2f")
    x = np.arange(len(labels_to_plot))
    width = 0.35
    means = [stats[c]["mean_words"] for c in labels_to_plot]
    medians = [stats[c]["median_words"] for c in labels_to_plot]

    ax3.bar(x - width/2, means, width, label="Mean", color="#00bcd4", alpha=0.85)
    ax3.bar(x + width/2, medians, width, label="Median", color="#e91e63", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_to_plot)
    ax3.set_title("Mean vs Median Word Count per Chunk", color="white", fontsize=12, pad=10)
    ax3.set_ylabel("Word Count", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#2a2a3e", edgecolor="none", labelcolor="white")
    ax3.grid(True, linestyle="--", alpha=0.2, color="white")

    # 4. Lexical Diversity & Vocabulary Size (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor("#1e1e2f")
    vocabs = [stats[c]["vocab_size"] for c in labels_to_plot]
    ax4.bar(labels_to_plot, vocabs, color=bar_colors, alpha=0.85, edgecolor="black")
    ax4.set_yscale("log")
    ax4.set_title("Unique Vocabulary Size (Log Scale)", color="white", fontsize=12, pad=10)
    ax4.set_ylabel("Distinct Words (log)", color="white")
    ax4.tick_params(colors="white")
    ax4.grid(True, linestyle="--", alpha=0.2, color="white")
    for bar in ax4.patches:
        height = bar.get_height()
        ax4.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            color="white", fontsize=9
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(
        output_path,
        dpi=300,
        facecolor=fig.get_facecolor(),
        edgecolor="none"
    )
    plt.close()
    print(f"Visualization saved to: {output_path}")


def export_json(stats: Dict[str, Dict[str, Any]], output_path: str):
    """Exports computed statistics to a JSON file (stripping raw arrays)."""
    export_data = {}
    for cat, data in stats.items():
        export_data[cat] = {k: v for k, v in data.items() if k != "word_counts_raw"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"Statistics exported to JSON: {output_path}")


def export_csv(stats: Dict[str, Dict[str, Any]], output_path: str):
    """Exports key statistics to CSV format."""
    keys = [
        "Category", "Chunks", "Sources", "Total Words", "Mean Words", "Median Words",
        "Std Dev", "Min Words", "Max Words", "Q1", "Q3", "IQR", "P90", "P99",
        "Skewness", "Total Chars", "Mean Chars", "Avg Word Len", "Vocab Size", "TTR %"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for cat, s in stats.items():
            if not s:
                continue
            writer.writerow([
                cat,
                s["n_chunks"],
                s["n_sources"],
                s["total_words"],
                f"{s['mean_words']:.2f}",
                f"{s['median_words']:.2f}",
                f"{s['std_words']:.2f}",
                s["min_words"],
                s["max_words"],
                f"{s['q1_words']:.2f}",
                f"{s['q3_words']:.2f}",
                f"{s['iqr_words']:.2f}",
                f"{s['p90_words']:.2f}",
                f"{s['p99_words']:.2f}",
                f"{s['skewness']:.2f}",
                s["total_chars"],
                f"{s['mean_chars']:.2f}",
                f"{s['avg_word_len']:.2f}",
                s["vocab_size"],
                f"{s['ttr'] * 100:.2f}%"
            ])
    print(f"Statistics exported to CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive statistics for UrzaLLM chunks (books, articles, videos, transcriptions)."
    )
    parser.add_argument("--chunks", type=str, default=None, help="Path to chunked dataset (books, articles, videos)")
    parser.add_argument("--transcriptions", type=str, default=None, help="Path to transcriptions dataset")
    parser.add_argument("--lang", type=str, choices=["cs", "en"], default="cs", help="Text language to analyze (cs/en)")
    parser.add_argument("--plot", action="store_true", help="Generate and save visual analysis plot")
    parser.add_argument("--plot-path", type=str, default="real_data/chunk_statistics_plot.png", help="Path to save plot")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export statistics as JSON")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export statistics as CSV")
    parser.add_argument("--top-words", type=int, default=5, help="Number of top frequent words to display")

    args = parser.parse_args()

    items, resolved_paths = load_dataset(args.chunks, args.transcriptions)
    if not items:
        print("Error: No data loaded. Please verify file paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully loaded {len(items):,} total records.\n")

    # Partition by type
    categories = {
        "Books": [x for x in items if x.get("type") == "book"],
        "Articles": [x for x in items if x.get("type") == "article"],
        "Videos": [x for x in items if x.get("type") == "video"],
        "Transcriptions": [x for x in items if x.get("type") == "transcription"]
    }

    # Book breakdown
    books_ancap = [x for x in categories["Books"] if x.get("src") == "ancap"]
    books_anag = [x for x in categories["Books"] if x.get("src") == "anag"]

    stats: Dict[str, Dict[str, Any]] = {}
    for cat_name, cat_items in categories.items():
        if cat_items:
            stats[cat_name] = calculate_category_stats(cat_items, lang=args.lang, top_n_words=args.top_words)

    if books_ancap:
        stats["Book: ancap"] = calculate_category_stats(books_ancap, lang=args.lang, top_n_words=args.top_words)
    if books_anag:
        stats["Book: anag"] = calculate_category_stats(books_anag, lang=args.lang, top_n_words=args.top_words)

    # Overall corpus statistics
    stats["Total"] = calculate_category_stats(items, lang=args.lang, top_n_words=args.top_words)

    # Display tables in terminal
    display_results(stats, lang=args.lang)

    # Optional plot
    if args.plot:
        generate_plot(stats, output_path=args.plot_path)

    # Optional exports
    if args.export_json:
        export_json(stats, args.export_json)
    if args.export_csv:
        export_csv(stats, args.export_csv)


if __name__ == "__main__":
    main()
