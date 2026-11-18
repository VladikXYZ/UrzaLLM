"""Build the reports from real_data/ for each embedding model into RealReports/<model>/.

    python Chunks/real_data.py                     # first: chunks + results from real_data/
    python Chunks/build_real_reports.py            # from the repo root; both models (about 10 minutes each on a CPU)
    python Chunks/build_real_reports.py --models bge_m3 --only book_map political_space

Every step runs with URZA_DATASET=real URZA_MODEL=<model> (see dataset.py), so the scripts read
Chunks/output_real/ and write to Chunks/output_real/<model>/projection/; the reports are then copied to
RealReports/<model>/ (HTML, Markdown) and RealReports/<model>/figures/ (PNG). The real_data embeddings are of the
chunk text alone, so only the textonly reports exist. political_space needs the 326 political items embedded with
the same model (Questions/output/ for bge-m3, Questions/output_<model>/ otherwise) and is skipped without them.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from dataset import MODELS  # noqa: E402

REAL = HERE / "output_real"
DEST = ROOT / "RealReports"
STEPS = [("analyze", ["Embedding/analyze.py", "--variant", "textonly"]),
         ("project_compare", ["Projection/project_compare.py", "--variant", "textonly"]),
         ("book_map", ["Projection/book_map.py", "--variant", "textonly"])] + \
        [(n, [f"Projection/{n}.py"]) for n in ("book_trails", "book_nearest", "timeline_map", "book_pca_ica",
                                               "all_pca_ica", "topic_distribution", "political_space")]


VARIANT_FIXES = [  # the scripts were written for three embedding variants; here there is one, so drop the mentions
    (r"(·|,) variant <code>\w+</code>", r"\1 __LABEL__"),
    (r"· <code>textonly</code></span>", "· __LABEL__</span>"),
    (r" \(variant <code>\w+</code>; text in front of each chunk: [^)]*\)", " (each paragraph embedded from its own text alone)"),
    (r" ?\(variant <code>textonly</code>\)", ""),
    (r"\btextonly (?=book map|UMAP|projection|embeddings|maps?\b)", ""),
    (r"\(textonly embeddings\)", "(__LABEL__ embeddings)"),
    (r"(\w+)_textonly(_\w+)?\.(html|png|md)", r"\1\2.\3"),
    (r"\(textonly\s+embeddings, ", "(__LABEL__ embeddings, "),
    (r" \(no chapter path, title or other context line\)", ""),
    (r", any shared context line\)", ")"),
    (r"^\| variant matches \|.*\n", ""),
]


def relabel(text, model):
    """Model name and dimensions for this model, and no mention of embedding variants."""
    label, hf, dim = MODELS[model]
    for a, b in VARIANT_FIXES:
        text = re.sub(a, b, text, flags=re.M)
    text = text.replace("__LABEL__", label)
    if model != "bge_m3":
        text = text.replace("BAAI/bge-m3", hf).replace("bge-m3", label)
        text = re.sub(r"\b1,024\b|\b1024(?=[- ]?dim)", f"{dim:,}", text)
    return text


def clean_name(name):
    return name.replace("_textonly", "")


def copy_outputs(model):
    work, dest = REAL / model / "projection", DEST / model
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(work.glob("*.html")):
        (dest / clean_name(f.name)).write_text(relabel(f.read_text(encoding="utf-8"), model), encoding="utf-8"); n += 1
    for f in sorted(work.glob("*.png")):
        shutil.copy2(f, dest / "figures" / clean_name(f.name)); n += 1
    src = REAL / model / "results_textonly" / "embedding_report.md"
    if src.exists():
        md = relabel(src.read_text(encoding="utf-8"), model)
        md = re.sub(r"^# Embedding report: variant `\w+`", f"# Embedding report: {MODELS[model][0]}", md, flags=re.M)
        md = md.replace(", None s,", ",")
        (dest / "embedding_report.md").write_text(md, encoding="utf-8"); n += 1
    print(f"Copied {n} files to {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--only", nargs="*", help="step names to run (default: all)")
    ap.add_argument("--copy-only", action="store_true")
    args = ap.parse_args()
    for model in args.models:
        env = dict(os.environ, URZA_DATASET="real", URZA_MODEL=model, PYTHONIOENCODING="utf-8")
        work = REAL / model / "projection"
        work.mkdir(parents=True, exist_ok=True)
        qdir = ROOT / "Questions" / ("output" if model == "bge_m3" else f"output_{model}")
        for name, cmd in ([] if args.copy_only else STEPS):
            if args.only and name not in args.only:
                continue
            if name == "political_space" and not (qdir / "embeddings.npy").exists():
                print(f"== {model} {name}: skipped, no {qdir.relative_to(ROOT)}/embeddings.npy"); continue
            t0 = time.time()
            print(f"== {model} {name}", flush=True)
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            (work / f"{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            if r.returncode:
                print(r.stdout[-2000:], r.stderr[-4000:])
                sys.exit(f"Step {model} {name} failed (log: {work / (name + '.log')})")
            print(f"   ok, {time.time() - t0:.0f} s", flush=True)
        copy_outputs(model)


if __name__ == "__main__":
    main()
