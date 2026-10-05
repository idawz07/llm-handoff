#!/usr/bin/env python3
"""Human-friendly control helper for a running Handoff Router.

    routerctl.py [--url URL] [--key KEY] <command> ...

Commands
    status                 full router state
    models                 id, owner, load state, backend
    health                 liveness summary
    load MODEL             warm a model (switches backend if needed)
    unload MODEL           stop the model's backend and free the GPU
    freeze | thaw          block / allow model loads around other GPU work
    reload                 re-read router-backends.yaml (no restart)
    activity [N]           recent lifecycle events
    logs BACKEND [N]       tail a backend's router-owned log
    watch                  live SSE feed of lifecycle events

Environment: ROUTER_URL, ROUTER_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

DEFAULT_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8081")
DEFAULT_KEY = os.environ.get("ROUTER_KEY", "")


def request(url: str, key: str, path: str, *, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method=method)
    req.add_header("content-type", "application/json")
    if key:
        req.add_header("authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            print(f"HTTP {exc.code}: {detail[:400]}", file=sys.stderr)
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="routerctl.py")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("command", choices=[
        "status", "models", "health", "load", "unload", "freeze", "thaw",
        "reload", "activity", "logs", "watch",
    ])
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()
    url, key = opts.url, opts.key

    if opts.command == "watch":
        req = urllib.request.Request(url.rstrip("/") + "/activity/stream")
        if key:
            req.add_header("authorization", f"Bearer {key}")
        with urllib.request.urlopen(req) as stream:
            for line in stream:
                text = line.decode(errors="replace").strip()
                if text.startswith("data:"):
                    print(text[5:].strip())
        return

    if opts.command == "models":
        for model in request(url, key, "/v1/models").get("data", []):
            status = model.get("status", {})
            print(
                f"{model.get('id',''):<46} {model.get('owned_by',''):<10} "
                f"{status.get('value',''):<9} {status.get('backend','')}"
            )
        return

    if opts.command == "activity":
        limit = opts.args[0] if opts.args else "30"
        for event in request(url, key, f"/router/activity?limit={limit}").get("events", []):
            stamp = time.strftime("%H:%M:%S", time.localtime(event.get("at", 0)))
            print(
                f"{stamp}  {event.get('kind',''):<16} "
                f"{event.get('backend') or '-':<12} {event.get('model') or '-':<34} "
                f"{event.get('detail','')}"
            )
        return

    if opts.command == "logs":
        if not opts.args:
            raise SystemExit("usage: routerctl.py logs BACKEND [LINES]")
        lines = opts.args[1] if len(opts.args) > 1 else "200"
        payload = request(url, key, f"/logs/{opts.args[0]}?lines={lines}")
        print("\n".join(payload.get("lines", [])))
        return

    if opts.command in ("load", "unload"):
        if not opts.args:
            raise SystemExit(f"usage: routerctl.py {opts.command} MODEL")
        print(json.dumps(request(url, key, f"/models/{opts.command}", method="POST",
                                 body={"model": opts.args[0]}), indent=2))
        return

    path = {"status": "/router/status", "health": "/health",
            "freeze": "/router/freeze", "thaw": "/router/thaw",
            "reload": "/router/reload"}[opts.command]
    method = "POST" if opts.command in ("freeze", "thaw", "reload") else "GET"
    print(json.dumps(request(url, key, path, method=method), indent=2))


if __name__ == "__main__":
    main()
