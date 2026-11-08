"""Explicit thinking-mode endpoint; preserves total generation and task budgets."""

import itertools
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

port = int(sys.argv[1])
backends = sys.argv[2:]
cycle = itertools.cycle(backends)
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/health":
                ok = all(
                    requests.get(b + "/health", timeout=3).status_code == 200
                    for b in backends
                )
                self.send(200 if ok else 503, b"{}")
                return
            r = requests.get(backends[0] + self.path, timeout=10)
            self.send(r.status_code, r.content)
        except Exception:
            self.send(503, b'{"error":"thinking backend unavailable"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 5 * 1024 * 1024:
            self.send(413, b"{}")
            return
        data = json.loads(self.rfile.read(n))
        data["chat_template_kwargs"] = {"enable_thinking": True}
        data["thinking_token_budget"] = 512
        with lock:
            backend = next(cycle)
        try:
            r = requests.post(backend + self.path, json=data, timeout=175)
            self.send(r.status_code, r.content)
        except Exception:
            self.send(503, b'{"error":"thinking backend request failed"}')


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
