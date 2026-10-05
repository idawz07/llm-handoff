#!/usr/bin/env python3
"""Tiny OpenAI-compatible process used by lifecycle integration tests."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--active-dir", required=True)
    parser.add_argument("--counter-dir", required=True)
    parser.add_argument("--startup-delay", type=float, default=0)
    parser.add_argument("--fail-start", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
ACTIVE_DIR = Path(ARGS.active_dir)
COUNTER_DIR = Path(ARGS.counter_dir)
MARKER = ACTIVE_DIR / f"{ARGS.name}.active"


def increment_counter() -> None:
    COUNTER_DIR.mkdir(parents=True, exist_ok=True)
    path = COUNTER_DIR / f"{ARGS.name}.starts"
    try:
        value = int(path.read_text()) + 1
    except (OSError, ValueError):
        value = 1
    path.write_text(f"{value}\n")


def cleanup() -> None:
    try:
        MARKER.unlink()
    except FileNotFoundError:
        pass


def terminate(_signum, _frame) -> None:
    cleanup()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


increment_counter()
print(f"fake backend {ARGS.name} starting", flush=True)
if ARGS.fail_start:
    print(f"intentional startup failure from {ARGS.name}", flush=True)
    raise SystemExit(17)
time.sleep(ARGS.startup_delay)
ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
MARKER.write_text(str(os.getpid()))
atexit.register(cleanup)
signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *values):
        print(f"{ARGS.name}: {format % values}", flush=True)

    def _json(self, payload, status=200):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Fake-Backend", ARGS.name)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok", "backend": ARGS.name})
        elif self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{
                "id": ARGS.served_model,
                "object": "model",
                "owned_by": ARGS.name,
            }]})
        elif self.path.startswith("/props"):
            self._json({"model_alias": ARGS.served_model, "total_slots": 1})
        elif self.path.startswith("/crash"):
            self.close_connection = True
            cleanup()
            os._exit(23)
        else:
            self._json({
                "path": self.path,
                "backend": ARGS.name,
                "client_port": self.client_address[1],
                "headers": {key.lower(): value for key, value in self.headers.items()},
            })

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        delay = float(body.get("test_delay", 0))
        if delay:
            time.sleep(delay)
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for index, text in enumerate(("one", "two", "three")):
                event = {
                    "id": "fake-stream",
                    "object": "chat.completion.chunk",
                    "model": ARGS.served_model,
                    "choices": [{"index": 0, "delta": {"content": text}}],
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.08)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        self._json({
            "id": "fake-completion",
            "object": "chat.completion",
            "model": ARGS.served_model,
            "backend": ARGS.name,
            "received": body,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        })


server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
print(f"fake backend {ARGS.name} ready on {ARGS.port}", flush=True)
server.serve_forever(poll_interval=0.1)
