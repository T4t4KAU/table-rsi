"""Loopback-only round-robin for identical frozen local model replicas."""

import itertools
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

BACKENDS = ["http://127.0.0.1:31002", "http://127.0.0.1:31003"]
cycle = itertools.cycle(BACKENDS)
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, data, content="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/health":
                ok = all(
                    requests.get(b + "/health", timeout=3).status_code == 200
                    for b in BACKENDS
                )
                self.send(200 if ok else 503, b"{}")
                return
            r = requests.get(BACKENDS[0] + self.path, timeout=10)
            self.send(r.status_code, r.content)
        except Exception:
            self.send(503, b'{"error":"local replica unavailable"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 5 * 1024 * 1024:
            self.send(413, b'{"error":"request too large"}')
            return
        data = self.rfile.read(n)
        with lock:
            backend = next(cycle)
        try:
            r = requests.post(
                backend + self.path,
                data=data,
                headers={"Content-Type": "application/json"},
                timeout=175,
            )
            self.send(r.status_code, r.content)
        except Exception:
            self.send(503, b'{"error":"local replica request failed"}')


ThreadingHTTPServer(("127.0.0.1", 31100), Handler).serve_forever()
