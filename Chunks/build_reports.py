"""Build every report from the final chunks and their embeddings into FinalReports/.

    python Chunks/build_reports.py              # from the repo root; runs everything (about 10-20 minutes on a CPU)
    python Chunks/build_reports.py --only book_map political_space   # rerun some steps, then copy

Needs Chunks/output/chunks.parquet (Chunks/chunk.py) and Chunks/output/results_<variant>/ for withsource, nosource
and textonly (Chunks/embed_final.py), plus Questions/output/ (Questions/embed_questions.py).
Every step runs with URZA_DATASET=final, so the scripts read the final data and write to Chunks/output/projection/;
the reports are then copied to FinalReports/ (HTML and Markdown) and FinalReports/figures/ (PNG).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / "output" / "projection"
DEST = ROOT / "FinalReports"
VARIANTS = ["withsource", "nosource", "textonly"]

STEPS = (
    [(f"analyze_{v}", ["Embedding/analyze.py", "--variant", v]) for v in VARIANTS]
    + [("chunking_report", ["Chunks/report.py"])]
    + [(f"project_compare_{v}", ["Projection/project_compare.py", "--variant", v]) for v in VARIANTS]
    # each book map compares itself with the other variant's saved stats, so nosource runs again after textonly
    + [(f"book_map_{v}{'_again' if i == 2 else ''}", ["Projection/book_map.py", "--variant", v])
       for i, v in enumerate(("nosource", "textonly", "nosource"))]
    + [(name, [f"Projection/{name}.py"]) for name in ("book_trails", "book_nearest", "timeline_map", "book_pca_ica",
                                                     "all_pca_ica", "topic_distribution", "political_space")]
)


def copy_outputs():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(WORK.glob("*.html")):
        shutil.copy2(f, DEST / f.name); n += 1
    for f in sorted(WORK.glob("*.png")):
        shutil.copy2(f, DEST / "figures" / f.name); n += 1
    shutil.copy2(HERE / "output" / "chunking_report.html", DEST / "chunking_report.html"); n += 1
    for v in VARIANTS:
        src = HERE / "output" / f"results_{v}" / "embedding_report.md"
        if src.exists():
            shutil.copy2(src, DEST / f"embedding_report_{v}.md"); n += 1
    print(f"Copied {n} files to {DEST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="step names to run (default: all)")
    ap.add_argument("--copy-only", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ, URZA_DATASET="final", PYTHONIOENCODING="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    if not args.copy_only:
        for name, cmd in STEPS:
            if args.only and name not in args.only:
                continue
            t0 = time.time()
            print(f"== {name}", flush=True)
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            (WORK / f"{name}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            if r.returncode:
                print(r.stdout[-2000:], r.stderr[-4000:])
                sys.exit(f"Step {name} failed (log: {WORK / (name + '.log')})")
            print(f"   ok, {time.time() - t0:.0f} s", flush=True)
    copy_outputs()


if __name__ == "__main__":
    main()
