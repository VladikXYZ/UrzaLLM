"""Build the zip to send to whoever runs the embedding.

    ..\\.venv\\Scripts\\python build_package.py

Copies the three needed columns of ../Preprocessing/output/chunks.parquet into
data/chunks.parquet and zips the job (script, run scripts, README, data) into
../dist/urzallm_embedding.zip. Rebuild after rerunning the preprocessing.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Preprocessing" / "output" / "chunks.parquet"
DATA = HERE / "data" / "chunks.parquet"
DIST = HERE.parent / "dist" / "urzallm_embedding.zip"
FILES = ["embed.py", "run.bat", "run.sh", "requirements.txt", "README.md", "data/chunks.parquet"]


def main():
    t = pq.read_table(SRC, columns=["chunk_id", "source", "embed_text"])
    DATA.parent.mkdir(exist_ok=True)
    pq.write_table(t, DATA, compression="zstd")
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    (HERE / "data" / "SOURCE.json").write_text(json.dumps(
        {"from": str(SRC), "rows": t.num_rows, "sha256": digest}, indent=2), encoding="utf-8")

    DIST.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(DIST, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            info = zipfile.ZipInfo.from_file(HERE / f, f"urzallm_embedding/{f}")
            if f == "run.sh":
                info.external_attr = 0o755 << 16       # executable on Mac/Linux
            info.compress_type = zipfile.ZIP_DEFLATED
            data = (HERE / f).read_bytes()
            if f.endswith(".bat"):     # cmd.exe needs CRLF line endings
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            elif f.endswith(".sh"):    # bash needs LF
                data = data.replace(b"\r\n", b"\n")
            z.writestr(info, data)
    print(f"{t.num_rows:,} chunks, data sha256 {digest[:12]}...")
    print(f"Package: {DIST} ({DIST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
