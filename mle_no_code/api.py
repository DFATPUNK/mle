from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .pipeline import NoCodeMLPipeline

PIPELINE = NoCodeMLPipeline()

HTML = """<!doctype html><html><head><title>No-code ML Pipeline</title><style>body{font-family:sans-serif;margin:2rem}.blocks{display:flex;gap:.5rem;flex-wrap:wrap}.block{border:1px solid #ddd;border-radius:10px;padding:1rem;background:#f8fafc}code{background:#eef;padding:.2rem}</style></head><body><h1>No-code ML Pipeline Builder</h1><p>Visual MVP flow inspired by Zapier.</p><div class=blocks>__BLOCKS__</div><h2>API</h2><ul><li><code>POST /upload?path=data.csv&target=label</code></li><li><code>POST /predict</code> with JSON row</li><li><code>POST /feedback</code> with prediction_index and corrected_value</li><li><code>GET /status</code></li></ul></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            blocks = "".join(f"<div class=block><b>{b['name']}</b><br>{b['status']}</div>" for b in PIPELINE.workflow_blocks())
            self._send(200, HTML.replace("__BLOCKS__", blocks), "text/html")
        elif self.path.startswith("/status"):
            self._json(200, {"blocks": PIPELINE.workflow_blocks(), "logs": [l.__dict__ for l in PIPELINE.logs]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/upload":
                csv_path = query.get("path", [""])[0]
                target = query.get("target", [""])[0]
                PIPELINE.upload_csv(csv_path)
                PIPELINE.split(target=target or None)
                if target:
                    PIPELINE.select_target(target)
                    PIPELINE.train()
                    PIPELINE.tune()
                    artifact = PIPELINE.export_best_model()
                else:
                    artifact = Path("")
                self._json(200, {"dataset": PIPELINE.dataset.to_json() if PIPELINE.dataset else None, "artifact": str(artifact)})
            elif parsed.path == "/predict":
                self._json(200, PIPELINE.predict_one(self._body()))
            elif parsed.path == "/feedback":
                body = self._body()
                self._json(200, PIPELINE.add_feedback(int(body["prediction_index"]), body["corrected_value"], body.get("user", "manual")))
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # API surface returns no-code friendly errors.
            self._json(400, {"error": str(exc)})

    def _body(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, indent=2, default=str), "application/json")

    def _send(self, status: int, payload: str, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    run()
