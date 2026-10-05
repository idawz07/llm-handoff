"""Bounded activity log for operator visibility (switches, loads, reloads).

The router is a supervisor, so the questions operators ask are "what happened,
what is running now, and why did that request wait?". This module keeps a small
in-memory ring of lifecycle events and broadcasts them to SSE subscribers, which
is enough for a single-process facade without pulling in a metrics stack.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

LOG = logging.getLogger("handoff-router.activity")


@dataclass
class ActivityEvent:
    at: float
    kind: str
    backend: str | None = None
    model: str | None = None
    detail: str = ""
    duration_seconds: float | None = None
    level: str = "info"
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "at": self.at,
            "kind": self.kind,
            "level": self.level,
        }
        if self.backend:
            payload["backend"] = self.backend
        if self.model:
            payload["model"] = self.model
        if self.detail:
            payload["detail"] = self.detail
        if self.duration_seconds is not None:
            payload["duration_seconds"] = round(self.duration_seconds, 3)
        if self.data:
            payload["data"] = self.data
        return payload


class ActivityLog:
    def __init__(self, capacity: int = 200) -> None:
        self._events: deque[ActivityEvent] = deque(maxlen=max(10, capacity))
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._seq = 0

    def record(
        self,
        kind: str,
        *,
        backend: str | None = None,
        model: str | None = None,
        detail: str = "",
        duration_seconds: float | None = None,
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> ActivityEvent:
        self._seq += 1
        event = ActivityEvent(
            at=time.time(),
            kind=kind,
            backend=backend,
            model=model,
            detail=detail,
            duration_seconds=duration_seconds,
            level=level,
            data=dict(data or {}),
        )
        event.data.setdefault("seq", self._seq)
        self._events.append(event)
        payload = event.as_dict()
        for queue in list(self._subscribers):
            # A slow or gone consumer must never slow a backend switch.
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                LOG.debug("dropping activity event for a saturated subscriber")
        return event

    def recent(self, limit: int = 50, *, since: float = 0.0) -> list[dict[str, Any]]:
        events = [event for event in self._events if event.at >= since]
        return [event.as_dict() for event in events[-limit:]]

    def subscribe(self, maxsize: int = 200) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def stream(
        self,
        *,
        disconnected: Any = None,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[bytes]:
        """Yield SSE frames until the client disconnects."""
        queue = self.subscribe()
        loop = asyncio.get_running_loop()
        last_beat = loop.time()
        try:
            while True:
                if disconnected is not None and await disconnected():
                    return
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    payload = None
                if payload is not None:
                    yield b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"
                    last_beat = loop.time()
                elif loop.time() - last_beat >= heartbeat_seconds:
                    yield b": keep-alive\n\n"
                    last_beat = loop.time()
        finally:
            self.unsubscribe(queue)
