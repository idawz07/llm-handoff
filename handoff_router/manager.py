"""Serialized model switching, request draining, idle reclaim, and persisted state."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .activity import ActivityLog
from .backends import BackendAdapter, BackendError, BackendStartupError, create_adapter
from .registry import BackendConfig, ModelConfig, ModelRegistry

LOG = logging.getLogger("handoff-router.manager")


class ModelSwitchError(RuntimeError):
    def __init__(self, message: str, *, backend: str | None = None, logs: str = "") -> None:
        super().__init__(message)
        self.backend = backend
        self.logs = logs


@dataclass
class ModelLease:
    manager: "RouterManager"
    model: ModelConfig
    adapter: BackendAdapter
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.manager.release_request(self.model.backend)


class RouterManager:
    """Owns the running backends and reserves requests under the switch lock."""

    def __init__(self, registry: ModelRegistry, *, config_path: str | Path | None = None) -> None:
        self.registry = registry
        self.settings = registry.settings
        self.config_path = Path(config_path) if config_path else registry.source_path
        self.adapters = {
            name: create_adapter(config, self.settings)
            for name, config in registry.backends.items()
        }
        self.active_model: str | None = None
        self.active_backend: str | None = None
        self.switching_to: str | None = None
        self.active_request_count = 0
        self._backend_requests: dict[str, int] = {}
        self.switch_lock = asyncio.Lock()
        self._request_condition = asyncio.Condition()
        self._shutdown = False
        self.frozen = False  # window freeze (in-memory only, fail-open)
        self.state_path = self.settings.state_dir / "state.json"
        self.activity = ActivityLog(self.settings.activity_history)
        self.last_used: dict[str, float] = {}
        self.last_model_by_backend: dict[str, str] = {}
        # An in-use adapter stays immutable across reload. New launch settings
        # are staged here until its leases drain on the next request for it.
        self._pending_configs: dict[str, BackendConfig] = {}
        self._retired_backends: set[str] = set()
        self._needs_restart: set[str] = set()
        self._supervisor_tasks: list[asyncio.Task] = []
        self._last_failed_mtime: float | None = None
        self.started_at = time.time()

    async def startup(self) -> None:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        state = self._read_state()
        adopted: list[str] = []

        state_backend = state.get("active_backend") if isinstance(state, dict) else None
        process_state = state.get("process") if isinstance(state, dict) else None
        if not isinstance(process_state, dict):
            process_state = {}
        state_pid = process_state.get("pid")
        if state_backend in self.adapters and isinstance(state_pid, int):
            adapter = self.adapters[state_backend]
            if adapter.adopt(state_pid) and await adapter.is_healthy():
                adopted.append(state_backend)
                model_id = state.get("active_model")
                if model_id in self.registry.models and self.registry.models[model_id].backend == state_backend:
                    self.active_model = model_id
                else:
                    self.active_model = self._default_model_for_backend(state_backend)
                self.active_backend = state_backend
                LOG.info("adopted persisted backend %s pid=%s", state_backend, state_pid)
            elif adapter.runtime is not None:
                LOG.warning("terminating unhealthy persisted backend %s pid=%s", state_backend, state_pid)
                await adapter.stop()
            elif adapter.adopt_persisted_group(process_state):
                LOG.warning(
                    "terminating orphaned persisted process group backend=%s pgid=%s",
                    state_backend,
                    process_state.get("pgid"),
                )
                await adapter.stop()

        for backend_name, adapter in self.adapters.items():
            if backend_name in adopted:
                continue
            pid = await adapter.discover_listener_pid()
            if pid is None:
                continue
            if adapter.adopt(pid) and await adapter.is_healthy():
                adopted.append(backend_name)
                LOG.info("discovered existing backend %s pid=%s", backend_name, pid)
            elif adapter.runtime is not None:
                LOG.warning("terminating unhealthy listener for backend %s pid=%s", backend_name, pid)
                await adapter.stop()
            else:
                raise RuntimeError(
                    f"backend {backend_name} is listening with pid {pid}, but its process "
                    "group cannot be adopted safely"
                )

        nonresident = [name for name in adopted if not self.adapters[name].config.resident]
        if len(nonresident) > 1:
            LOG.error("multiple non-resident backends were already running: %s; stopping all", adopted)
            for backend_name in adopted:
                await self.adapters[backend_name].stop()
            adopted.clear()
            self.active_model = None
            self.active_backend = None
            await self.wait_for_gpu_release()
        elif adopted and self.active_backend is None:
            # A resident sidecar plus one large model is a valid deployment.
            self.active_backend = nonresident[0] if nonresident else adopted[0]
            self.active_model = self._default_model_for_backend(self.active_backend)

        for name in adopted:
            model_id = self.active_model if name == self.active_backend else self._default_model_for_backend(name)
            if model_id:
                self.last_model_by_backend[name] = model_id
        self._write_state()
        self._start_supervisors()
        self.activity.record(
            "startup",
            backend=self.active_backend,
            model=self.active_model,
            detail=f"{len(self.adapters)} backends configured",
            data={
                "adopted": sorted(name for name, a in self.adapters.items() if a.is_running()),
                "backends": sorted(self.adapters),
            },
        )

    def _start_supervisors(self) -> None:
        """Idle reclaim + config watching. Both are best-effort background loops."""
        if self._shutdown or self._supervisor_tasks:
            return
        self._supervisor_tasks.append(asyncio.create_task(self._reaper_loop(), name="idle-reaper"))
        if self.config_path is not None:
            self._supervisor_tasks.append(
                asyncio.create_task(self._config_watch_loop(), name="config-watcher")
            )

    async def _supervisor_guard(self, label: str, body: Any) -> None:
        while not self._shutdown:
            try:
                await body()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a supervisor must never take the facade down
                LOG.warning("%s iteration failed: %s", label, exc)
            await asyncio.sleep(1.0)

    async def _reaper_loop(self) -> None:
        await self._supervisor_guard("idle reaper", self._reap_once)

    async def _config_watch_loop(self) -> None:
        await self._supervisor_guard("config watcher", self._watch_config_once)

    async def _reap_once(self) -> None:
        poll = max(1.0, self.settings.idle_poll_seconds)
        if not self._idle_candidates() and not self._retired_backends:
            await asyncio.sleep(poll)
            return
        async with self.switch_lock:
            if self._shutdown or self.switching_to is not None:
                return
            # A config removal must not kill an in-flight stream. Retired
            # backends stay reachable through their existing leases until idle.
            for name in list(self._retired_backends):
                if name == self.active_backend or self._backend_requests.get(name, 0):
                    continue
                if name in self.adapters:
                    await self._stop_backend(name)
                    self.adapters.pop(name, None)
                self._retired_backends.discard(name)
                self.activity.record("backend_retired", backend=name)
            if self.frozen:
                return
            # Recompute under the lock: a request may have completed since the
            # first check, resetting its last-used timestamp.
            for backend_name, idle_seconds in self._idle_candidates():
                adapter = self.adapters.get(backend_name)
                if adapter is None or not adapter.is_running():
                    continue
                LOG.info("unloading idle backend %s (idle %.0fs)", backend_name, idle_seconds)
                if self.active_backend == backend_name:
                    self.active_backend = None
                    self.active_model = None
                    self._write_state()
                self.activity.record(
                    "idle_unload", backend=backend_name,
                    detail=f"idle for {idle_seconds:.0f}s", level="notice",
                )
                unload_started = time.time()
                await self._stop_backend(backend_name)
                self.activity.record(
                    "idle_unloaded", backend=backend_name,
                    duration_seconds=time.time() - unload_started,
                )
            self._write_state()

    def _idle_candidates(self) -> list[tuple[str, float]]:
        """Running backends past a positive idle TTL, least valuable first."""
        now = time.time()
        candidates: list[tuple[int, float, str, float]] = []
        for name, adapter in self.adapters.items():
            if not adapter.is_running() or self._backend_requests.get(name, 0):
                continue
            ttl = adapter.config.idle_ttl_seconds
            if ttl is None:
                ttl = self.settings.idle_ttl_seconds
            if ttl <= 0:  # 0 disables reclaim; a negative value explicitly pins
                continue
            reference = self.last_used.get(name, adapter.runtime.started_at if adapter.runtime else self.started_at)
            idle_seconds = now - reference
            if adapter.runtime and now - adapter.runtime.started_at < self.settings.idle_grace_seconds:
                continue
            if idle_seconds < ttl:
                continue
            candidates.append((adapter.config.evict_cost, -idle_seconds, name, idle_seconds))
        candidates.sort()
        return [(name, idle_seconds) for _cost, _neg, name, idle_seconds in candidates]

    async def _watch_config_once(self) -> None:
        poll = max(0.5, self.settings.config_poll_seconds)
        await asyncio.sleep(poll)
        if not self.settings.watch_config or self.config_path is None or not self.config_path.is_file():
            return
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return
        if mtime <= self.registry.source_mtime or mtime == self._last_failed_mtime:
            return
        await self.reload()

    async def reload(self) -> dict[str, Any]:
        """Validate then atomically stage config edits without stopping a live backend.

        Existing leases retain their original adapter and upstream URL. A changed
        running backend is replaced only after its own requests have drained on
        the next request for it. Removals are retired after their leases finish.
        """
        if self.config_path is None:
            raise ModelSwitchError("router was not started from a config file; cannot reload")
        async with self.switch_lock:
            try:
                fresh = ModelRegistry.from_file(self.config_path)
                if fresh.settings.state_dir != self.settings.state_dir:
                    raise ValueError("changing settings.state_dir requires a service restart")
                if (
                    fresh.settings.upstream_connect_timeout_seconds != self.settings.upstream_connect_timeout_seconds
                    or fresh.settings.upstream_write_timeout_seconds != self.settings.upstream_write_timeout_seconds
                ):
                    raise ValueError("changing upstream HTTP timeouts requires a service restart")
                # Preflight every adapter before changing any state. A bad
                # adapter name must not leave the running registry half-updated.
                prepared = {name: create_adapter(cfg, fresh.settings) for name, cfg in fresh.backends.items()}
                for cfg in fresh.backends.values():
                    _ = cfg.port
            except Exception as exc:
                try:
                    self._last_failed_mtime = self.config_path.stat().st_mtime
                except OSError:
                    pass
                self.activity.record("config_reload_failed", detail=str(exc), level="error")
                raise ModelSwitchError(f"config reload failed: {exc}") from exc

            old_names = set(self.adapters)
            new_names = set(fresh.backends)
            changes: dict[str, list[str]] = {"added": [], "removed": [], "changed": [], "unchanged": []}

            for name in sorted(new_names):
                new_config = fresh.backends[name]
                old = self.adapters.get(name)
                if old is None:
                    self.adapters[name] = prepared[name]
                    changes["added"].append(name)
                elif old.config == new_config:
                    old.settings = fresh.settings
                    changes["unchanged"].append(name)
                elif old.is_running():
                    # Do NOT mutate old.config: a lease may read its upstream URL
                    # after reload releases switch_lock and before it sends.
                    self._pending_configs[name] = new_config
                    self._needs_restart.add(name)
                    old.settings = fresh.settings
                    changes["changed"].append(name)
                else:
                    self.adapters[name] = prepared[name]
                    changes["changed"].append(name)
                if not (old and old.is_running() and old.config != new_config):
                    self._pending_configs.pop(name, None)
                    self._needs_restart.discard(name)
                self._retired_backends.discard(name)

            for name in sorted(old_names - new_names):
                old = self.adapters[name]
                if old.is_running() or self._backend_requests.get(name, 0):
                    # No process is stopped in a reload, even an unselected
                    # resident sidecar with an in-flight stream.
                    self._retired_backends.add(name)
                else:
                    self.adapters.pop(name, None)
                    self._retired_backends.discard(name)
                    if self.active_backend == name:
                        self.active_backend = None
                        self.active_model = None
                    self.last_model_by_backend.pop(name, None)
                self._pending_configs.pop(name, None)
                self._needs_restart.discard(name)
                changes["removed"].append(name)

            self.registry = fresh
            self.settings = fresh.settings
            self._last_failed_mtime = None
            if self.active_model is not None and self.registry.resolve(self.active_model) is None:
                self.active_model = None
            self._write_state()

            summary = {
                "models": len(self.registry.models),
                "backends": len(self.adapters),
                "restart_pending": sorted(self._needs_restart),
                "retired": sorted(self._retired_backends),
            }
            self.activity.record(
                "config_reload",
                detail=f"+{len(changes['added'])} -{len(changes['removed'])} "
                f"~{len(changes['changed'])}",
                data={**changes, **summary},
            )
            LOG.info(
                "config reloaded: added=%s removed=%s changed=%s restart_pending=%s",
                changes["added"], changes["removed"], changes["changed"],
                sorted(self._needs_restart),
            )
            return {"success": True, **changes, **summary}

    async def shutdown(self) -> None:
        self._shutdown = True
        for task in self._supervisor_tasks:
            task.cancel()
        for task in self._supervisor_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._supervisor_tasks.clear()
        async with self.switch_lock:
            try:
                await self._wait_for_drain(self.settings.shutdown_drain_timeout_seconds)
            except ModelSwitchError:
                LOG.warning(
                    "router shutdown drain timed out with %d active requests",
                    self.active_request_count,
                )
            for name, adapter in self.adapters.items():
                if not adapter.is_running():
                    continue
                try:
                    await self._complete_on_cancel(adapter.stop())
                except BackendError as exc:
                    LOG.error("failed to stop backend %s during shutdown: %s", name, exc)
            if not any(adapter.is_running() for adapter in self.adapters.values()):
                self.active_model = None
                self.active_backend = None
            self.switching_to = None
            self._write_state()

    async def acquire(self, model_id: str) -> ModelLease:
        model = self.registry.resolve(model_id)
        if model is None:
            raise ModelSwitchError(f"model {model_id!r} is not configured")
        if self.frozen:
            raise ModelSwitchError("router is frozen (window in progress; new model loads blocked)")
        if self._shutdown:
            raise ModelSwitchError("router is shutting down")

        async with self.switch_lock:
            # Reload may have changed this model's owning backend while the
            # request waited. Never route using a pre-lock registry snapshot.
            model = self.registry.resolve(model_id)
            if model is None:
                raise ModelSwitchError(f"model {model_id!r} is not configured")
            if self._shutdown:
                raise ModelSwitchError("router is shutting down")
            if self.frozen:
                raise ModelSwitchError("router is frozen (window in progress; new model loads blocked)")
            self.switching_to = model_id
            started = time.time()
            try:
                await self._ensure_active_locked(model)
                # Reserve while still holding switch_lock. A competing model switch
                # cannot observe zero active requests between readiness and proxying.
                async with self._request_condition:
                    self.active_request_count += 1
                    self._backend_requests[model.backend] = self._backend_requests.get(model.backend, 0) + 1
                self.active_model = model_id
                self.last_model_by_backend[model.backend] = model_id
                self.last_used[model.backend] = time.time()
                self._write_state()
                self.activity.record(
                    "backend_ready",
                    backend=model.backend,
                    model=model_id,
                    duration_seconds=time.time() - started,
                    detail="cold start" if time.time() - started > 0.5 else "warm",
                )
                return ModelLease(self, model, self.adapters[model.backend])
            except Exception as exc:
                self.activity.record(
                    "switch_failed",
                    backend=model.backend,
                    model=model_id,
                    detail=str(exc),
                    duration_seconds=time.time() - started,
                    level="error",
                )
                raise
            finally:
                self.switching_to = None

    async def acquire_existing(self, model_id: str) -> ModelLease:
        """Lease a running native router for /models/unload; never load a model.

        Unload is permitted during a GPU freeze, but must not start or switch an
        engine just to unload a model that may not be resident in llama.cpp.
        """
        model = self.registry.resolve(model_id)
        if model is None:
            raise ModelSwitchError(f"model {model_id!r} is not configured")
        async with self.switch_lock:
            model = self.registry.resolve(model_id)
            if model is None:
                raise ModelSwitchError(f"model {model_id!r} is not configured")
            adapter = self.adapters.get(model.backend)
            if self._shutdown or adapter is None or not adapter.is_running() or not await adapter.is_healthy():
                raise ModelSwitchError(f"backend {model.backend!r} is not running")
            async with self._request_condition:
                self.active_request_count += 1
                self._backend_requests[model.backend] = self._backend_requests.get(model.backend, 0) + 1
            return ModelLease(self, model, adapter)

    def set_frozen(self, frozen: bool) -> None:
        """Window control: while frozen, acquire() raises ModelSwitchError.

        In-memory only (fail-open on restart). deactivate() is deliberately NOT
        gated, so a window can still unload the base model while frozen. This is
        the knob for non-LLM GPU work (image/video generation, training, Blender)
        that must not have a model reload racing it.
        """
        self.frozen = frozen
        self.activity.record("freeze" if frozen else "thaw", level="notice")
        LOG.info("router freeze state set to %s", frozen)

    async def release_request(self, backend: str) -> None:
        async with self._request_condition:
            if self._backend_requests.get(backend, 0):
                self._backend_requests[backend] -= 1
                self.active_request_count -= 1
                self.last_used[backend] = time.time()  # TTL starts when the stream finishes
            self._request_condition.notify_all()
        self._write_state()

    async def deactivate(self, model_id: str) -> bool:
        model = self.registry.resolve(model_id)
        if model is None:
            raise ModelSwitchError(f"model {model_id!r} is not configured")
        async with self.switch_lock:
            model = self.registry.resolve(model_id)
            if model is None:
                raise ModelSwitchError(f"model {model_id!r} is not configured")
            adapter = self.adapters[model.backend]
            if not adapter.is_running():
                return False
            await self._wait_for_drain(self.settings.drain_timeout_seconds, backend=model.backend)
            await self._stop_backend(model.backend)
            if self.active_backend == model.backend:
                self.active_model = None
                self.active_backend = None
            self.last_model_by_backend.pop(model.backend, None)
            self._write_state()
            self.activity.record("manual_unload", backend=model.backend, model=model.id, level="notice")
            if not any(other.is_running() for other in self.adapters.values()):
                await self.wait_for_gpu_release()
            return True

    async def note_model_unloaded(self, model_id: str) -> None:
        """The native llama.cpp router unloaded one child model, not its server."""
        async with self.switch_lock:
            if self.active_model == model_id:
                self.active_model = None
                self._write_state()
            model = self.registry.resolve(model_id)
            if model and self.last_model_by_backend.get(model.backend) == model_id:
                self.last_model_by_backend.pop(model.backend, None)
            self.activity.record("manual_unload", model=model_id, level="notice")

    async def _ensure_active_locked(self, model: ModelConfig) -> None:
        # Removed backends are allowed to finish their streams; retire them on
        # a later request/reaper pass after their own leases drain, never in reload.
        for name in sorted(self._retired_backends):
            if self._backend_requests.get(name, 0):
                continue  # a compatible target may coexist until this stream ends
            if name in self.adapters:
                await self._stop_backend(name)
                self.adapters.pop(name, None)
            if self.active_backend == name:
                self.active_backend = None
                self.active_model = None
                self._write_state()
            self._retired_backends.discard(name)
            self.activity.record("backend_retired", backend=name)

        target = self.adapters[model.backend]
        if model.backend in self._pending_configs:
            if target.is_running():
                await self._wait_for_drain(self.settings.drain_timeout_seconds, backend=model.backend)
                await self._stop_backend(model.backend)
                if self.active_backend == model.backend:
                    self.active_backend = None
                    self.active_model = None
                    self._write_state()
            # Replacing the object also replaces its adapter *class*, not just
            # its config. Old leases are gone; no caller can read the old URL.
            self.adapters[model.backend] = create_adapter(self._pending_configs.pop(model.backend), self.settings)
            self._needs_restart.discard(model.backend)
            target = self.adapters[model.backend]
        elif target.is_running() and not await target.is_healthy():
            await self._wait_for_drain(self.settings.drain_timeout_seconds, backend=model.backend)
            await self._stop_backend(model.backend)
            if self.active_backend == model.backend:
                self.active_backend = None
                self.active_model = None
                self._write_state()

        if not target.config.resident:
            # A new large backend evicts *all* other non-residents, regardless
            # of which one was most recently selected. Resident sidecars stay.
            victims = sorted(
                (name for name, adapter in self.adapters.items()
                 if name != model.backend and adapter.is_running() and not adapter.config.resident),
                key=lambda name: (self.adapters[name].config.evict_cost, name),
            )
            for name in victims:
                await self._wait_for_drain(self.settings.drain_timeout_seconds, backend=name)
                await self._stop_backend(name)
                if self.active_backend == name:
                    self.active_backend = None
                    self.active_model = None
                    self._write_state()
                self.activity.record("switch", backend=model.backend, model=model.id, detail=f"evicted {name}")
                if name in self._retired_backends:
                    self.adapters.pop(name, None)
                    self._retired_backends.discard(name)

        if target.is_running() and await target.is_healthy():
            # Reusing an already-running resident sidecar must never evict the
            # large backend, even if it was most recently selected.
            try:
                await target.wait_until_model_loaded(model)
            except BackendError as exc:
                if self.active_model == model.id:
                    self.active_model = None
                    self._write_state()
                raise ModelSwitchError(
                    f"backend {model.backend} failed to load model {model.id}: {exc}",
                    backend=model.backend, logs=target.tail_log(),
                ) from exc
            self.active_backend = model.backend
            self.active_model = model.id
            self._write_state()
            return

        if any(name != model.backend and adapter.is_running() for name, adapter in self.adapters.items()):
            self.activity.record("coexist", backend=model.backend, model=model.id,
                                 detail="keeping compatible running backend(s)")

        # GPU gate also covers initial starts and externally started copies.
        try:
            outcome = await self._wait_for_target_or_gpu(target)
        except ModelSwitchError as exc:
            exc.backend = model.backend
            raise
        if outcome == "adopted":
            await self._activate_backend(model)
            return

        LOG.info("starting backend=%s for model=%s", model.backend, model.id)

        async def start_and_probe() -> None:
            try:
                # Last-moment race guard: an external supervisor may have bound
                # the target port while we waited for the GPU. Adopt a healthy
                # copy instead of starting a competing process for the same
                # port and the same GPU.
                if not target.is_running():
                    listener = await target.discover_listener_pid()
                    if listener is not None:
                        if await self._probe_upstream_health(target) and target.adopt(listener):
                            LOG.info(
                                "%s: adopting externally started backend pid=%d instead of starting one",
                                target.config.name,
                                listener,
                            )
                            await self._activate_backend(model)
                            return
                        # A foreign copy holds the port. Starting a second copy
                        # would fight it for the same port and the same GPU.
                        raise ModelSwitchError(
                            f"backend {model.backend} port is held by external pid {listener} "
                            "that is not healthy or cannot be adopted; stop it or retry "
                            "once it is healthy",
                            backend=model.backend,
                        )
                await target.start()
                self.active_backend = model.backend
                self.active_model = model.id
                self._write_state()
                await target.wait_until_ready()
                # The process is up; now make sure the requested model is
                # actually loaded (multi-model backends report healthy while
                # empty). This runs under the switch lock so the requesting
                # client blocks until the model is servable instead of
                # receiving 400 "model is not loaded".
                await target.wait_until_model_loaded(model)
                if not await target.is_healthy():
                    raise BackendStartupError(f"{model.backend} failed its final health check")
            except Exception as exc:
                logs = target.tail_log()
                cleanup_error = ""
                try:
                    await target.stop()
                except BackendError as stop_exc:
                    cleanup_error = f"; cleanup failed: {stop_exc}"
                self.active_model = None
                self.active_backend = None
                self._write_state()
                raise ModelSwitchError(
                    f"failed to start backend {model.backend}: {exc}{cleanup_error}",
                    backend=model.backend,
                    logs=logs,
                ) from exc

        # Complete startup/cleanup before propagating request cancellation so a
        # disconnected client cannot leave a half-started, untracked backend.
        await self._complete_on_cancel(start_and_probe())

    async def _wait_for_drain(self, timeout: float, *, backend: str | None = None) -> None:
        def count() -> int:
            return self._backend_requests.get(backend, 0) if backend else self.active_request_count

        if count() == 0:
            return
        LOG.info("waiting for %d active request(s) to drain from %s", count(), backend or "all backends")

        async def wait() -> None:
            async with self._request_condition:
                await self._request_condition.wait_for(lambda: count() == 0)

        try:
            await asyncio.wait_for(wait(), timeout=timeout)  # also supports Python 3.10
        except asyncio.TimeoutError as exc:
            raise ModelSwitchError(
                f"timed out after {timeout:.1f}s waiting for "
                f"{count()} active request(s) to drain from {backend or 'all backends'}",
                backend=backend,
            ) from exc

    async def _query_free_vram_mib(self) -> int | None:
        """Return current free VRAM in MiB, or None if the query failed."""
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                self.settings.nvidia_smi,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
            if process.returncode != 0:
                return None
            return int(stdout.decode().strip().splitlines()[0].strip())
        except (OSError, ValueError, IndexError, asyncio.TimeoutError):
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return None

    async def wait_for_gpu_release(self) -> None:
        """Poll VRAM only after the stopped process group has exited."""
        threshold = self.settings.min_free_vram_mib
        if threshold <= 0:
            return
        deadline = asyncio.get_running_loop().time() + self.settings.gpu_release_timeout_seconds
        last_free: int | None = None
        while asyncio.get_running_loop().time() < deadline:
            free = await self._query_free_vram_mib()
            if free is None:
                LOG.warning("nvidia-smi VRAM query failed during GPU release wait")
            elif free >= threshold:
                return
            else:
                last_free = free
            await asyncio.sleep(self.settings.gpu_poll_seconds)
        detail = (
            f"last observed free VRAM: {last_free} MiB"
            if last_free is not None
            else "nvidia-smi VRAM query failed"
        )
        raise ModelSwitchError(
            f"GPU memory did not reach configured {threshold} MiB free within "
            f"{self.settings.gpu_release_timeout_seconds:.1f}s ({detail})"
        )

    async def _wait_for_target_or_gpu(self, target: BackendAdapter) -> str:
        """Wait for whichever frees the path first: the target backend or the GPU.

        A foreign supervisor (e.g. the service-manager GPU-handoff workflow)
        can start this same backend while a switch is draining the old one.
        Its load holds the GPU below the free-VRAM threshold, so a plain
        wait_for_gpu_release() times out with a 503 even though the backend
        we want is about to be ready. Polling the target's port lets us adopt
        the external copy as soon as it is healthy instead of racing a second
        large load for the same port.

        Returns "running" (the target already owns a live process), "adopted"
        (an external healthy copy was adopted), or "free" (the VRAM threshold
        is met and the caller should start the backend itself).
        """
        threshold = self.settings.min_free_vram_mib
        if threshold <= 0:
            return "running" if target.is_running() else "free"
        deadline = asyncio.get_running_loop().time() + self.settings.gpu_release_timeout_seconds
        last_free: int | None = None
        last_listener: int | None = None
        waiting_for_adopted_load = False
        while asyncio.get_running_loop().time() < deadline:
            if target.is_running():
                if waiting_for_adopted_load:
                    # An adopted external copy is still loading; keep polling
                    # for health instead of handing back a live-but-unready
                    # runtime.
                    if await target.is_healthy():
                        LOG.info(
                            "%s: externally started backend pid=%d became healthy; "
                            "adopting it instead of starting a duplicate",
                            target.config.name,
                            target.runtime.pid if target.runtime else -1,
                        )
                        return "adopted"
                else:
                    return "running"
            listener = await target.discover_listener_pid()
            if listener is not None:
                last_listener = listener
                stale_runtime = waiting_for_adopted_load and not target.is_running()
                if (not waiting_for_adopted_load or stale_runtime) and target.adopt(listener):
                    waiting_for_adopted_load = True
                    LOG.info(
                        "%s: externally started backend pid=%d is loading; "
                        "waiting for it to become healthy",
                        target.config.name,
                        listener,
                    )
            if threshold > 0:
                free = await self._query_free_vram_mib()
                if free is None:
                    LOG.warning("nvidia-smi VRAM query failed during backend/GPU wait")
                elif free >= threshold:
                    return "free"
                else:
                    last_free = free
            await asyncio.sleep(self.settings.gpu_poll_seconds)
        detail = (
            f"last observed free VRAM: {last_free} MiB"
            if last_free is not None
            else "nvidia-smi VRAM query failed"
        )
        if last_listener is not None:
            raise ModelSwitchError(
                f"backend {target.config.name} never became ready and GPU memory did not "
                f"reach configured {threshold} MiB free within "
                f"{self.settings.gpu_release_timeout_seconds:.1f}s ({detail}). An external "
                f"process (pid {last_listener}) holds the backend port and is likely still "
                "loading; retry once it is healthy or stop it first."
            )
        raise ModelSwitchError(
            f"GPU memory did not reach configured {threshold} MiB free within "
            f"{self.settings.gpu_release_timeout_seconds:.1f}s ({detail})"
        )

    async def _activate_backend(self, model: ModelConfig) -> None:
        """Activate an already running (adopted) target without starting a process."""
        self.active_backend = model.backend
        self.active_model = model.id
        self._write_state()
        try:
            await self.adapters[model.backend].wait_until_model_loaded(model)
        except BackendError as exc:
            if self.active_backend == model.backend and self.active_model == model.id:
                self.active_model = None
                self._write_state()
            raise ModelSwitchError(
                f"backend {model.backend} failed to load model {model.id}: {exc}",
                backend=model.backend, logs=self.adapters[model.backend].tail_log(),
            ) from exc

    async def _probe_upstream_health(self, target: BackendAdapter) -> bool:
        """Probe the upstream health endpoint without requiring an adopted runtime."""
        url = f"{target.upstream_url}{target.config.health_endpoint}"
        try:
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                response = await client.get(url)
            return 200 <= response.status_code < 300
        except (httpx.HTTPError, OSError):
            return False

    async def status(self) -> dict[str, Any]:
        healthy = False
        if self.active_backend:
            healthy = await self.adapters[self.active_backend].is_healthy()
        now = time.time()
        backends: dict[str, Any] = {}
        for name, adapter in self.adapters.items():
            described = adapter.describe_runtime()
            described["resident"] = adapter.config.resident
            described["evict_cost"] = adapter.config.evict_cost
            described["restart_pending"] = name in self._needs_restart
            described["retired"] = name in self._retired_backends
            described["active_requests"] = self._backend_requests.get(name, 0)
            ttl = adapter.config.idle_ttl_seconds
            described["idle_ttl_seconds"] = ttl if ttl is not None else self.settings.idle_ttl_seconds
            reference = self.last_used.get(name)
            if reference is None and adapter.runtime:
                reference = adapter.runtime.started_at
            described["idle_seconds"] = round(now - reference, 1) if reference else None
            backends[name] = described
        return {
            "frozen": self.frozen,
            "active_model": self.active_model,
            "active_backend": self.active_backend,
            "active_request_count": self.active_request_count,
            "switching_to": self.switching_to,
            "healthy": healthy,
            "config": {
                "path": str(self.config_path) if self.config_path else None,
                "watched": bool(self.settings.watch_config and self.config_path),
                "loaded_at": self.registry.source_mtime,
                "models": len(self.registry.models),
            },
            "backends": backends,
        }

    def _default_model_for_backend(self, backend: str) -> str | None:
        models = self.registry.models_for_backend(backend)
        selectors = [model for model in models if model.selector]
        selected = selectors[0] if selectors else (models[0] if models else None)
        return selected.id if selected else None

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    async def _stop_backend(self, backend: str) -> None:
        adapter = self.adapters[backend]
        try:
            await self._complete_on_cancel(adapter.stop())
        except BackendError as exc:
            raise ModelSwitchError(
                f"failed to stop backend {backend}: {exc}",
                backend=backend,
                logs=adapter.tail_log(),
            ) from exc
        self.last_used[backend] = time.time()
        self.last_model_by_backend.pop(backend, None)

    @staticmethod
    async def _complete_on_cancel(awaitable: Any) -> Any:
        """Keep a lifecycle operation serialized even if its HTTP caller leaves."""
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                LOG.exception("backend lifecycle operation failed after caller cancellation")
            raise

    def _write_state(self) -> None:
        try:
            self.settings.state_dir.mkdir(parents=True, exist_ok=True)
            process = None
            if self.active_backend:
                process = self.adapters[self.active_backend].state_record()
            payload = {
                "active_model": self.active_model,
                "active_backend": self.active_backend,
                "active_request_count": self.active_request_count,
                "process": process,
            }
            temporary = self.state_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, self.state_path)
        except OSError as exc:
            # State aids crash recovery, but an unavailable state filesystem must
            # not leak a request lease or undo an otherwise safe backend switch.
            LOG.error("failed to persist router state to %s: %s", self.state_path, exc)
