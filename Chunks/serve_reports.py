"""Serve RealReports/ locally, e.g. to preview presentation.html: python Chunks/serve_reports.py [--port 8780]

python -m http.server cuts large pages (the presentation is ~16 MB) short on Windows; this sends each file in one
response with a Content-Length, gzip-compressed when the browser accepts it, over HTTP/1.1.
"""
import argparse
import gzip
import mimetypes
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "RealReports"


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        gz = "gzip" in self.headers.get("Accept-Encoding", "")
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    print(f"Serving {ROOT} at http://127.0.0.1:{args.port}/presentation.html", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
