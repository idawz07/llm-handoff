"""Backend adapters and explicit process-group lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil

from .registry import BackendConfig, ModelConfig, RouterSettings

LOG = logging.getLogger("backend-router.backends")


class BackendError(RuntimeError):
    pass


class BackendStartupError(BackendError):
    pass


class BackendStopError(BackendError):
    pass


@dataclass
class BackendRuntime:
    pid: int
    pgid: int
    process: asyncio.subprocess.Process | None
    adopted: bool
    started_at: float


class BackendAdapter:
    """Base adapter. Engine-specific subclasses isolate request translations."""

    def __init__(self, config: BackendConfig, settings: RouterSettings) -> None:
        self.config = config
        self.settings = settings
        self.runtime: BackendRuntime | None = None
        self.log_dir = settings.state_dir / "logs"
        self.log_path = self.log_dir / f"{config.name}.log"
        self._log_handle: Any = None

    @property
    def upstream_url(self) -> str:
        return self.config.upstream

    def translate_request(self, body: dict[str, Any], model: ModelConfig) -> dict[str, Any]:
        translated = dict(body)
        translated["model"] = model.upstream_model
        return translated

    def translate_query_model(self, model: ModelConfig) -> str:
        return model.upstream_model

    async def start(self) -> None:
        if self.is_running():
            return
        # Reap and close bookkeeping from a process that died while idle before
        # replacing its runtime record.
        if self.runtime is not None:
            if self.runtime.process is not None:
                try:
                    await asyncio.wait_for(self.runtime.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            self.runtime = None
            self._close_log()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        command = list(self.config.command)
        executable = command[0]
        if "/" in executable and not Path(executable).exists():
            raise BackendStartupError(f"backend executable does not exist: {executable}")

        self._log_handle = self.log_path.open("ab", buffering=0)
        marker = (
            f"\n=== router start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
            f"backend={self.config.name} ===\n"
        )
        self._log_handle.write(marker.encode())
        env = os.environ.copy()
        env.update(self.config.environment)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception:
            self._close_log()
            raise
        self.runtime = BackendRuntime(
            pid=process.pid,
            pgid=process.pid,
            process=process,
            adopted=False,
            started_at=time.time(),
        )

    async def stop(self) -> None:
        runtime = self.runtime
        if runtime is None:
            self._close_log()
            return

        timeout = self.config.stop_timeout_seconds or self.settings.stop_timeout_seconds
        if self._group_alive(runtime.pgid):
            try:
                os.killpg(runtime.pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise BackendStopError(
                    f"permission denied stopping {self.config.name} process group {runtime.pgid}"
                ) from exc
            if not await self._wait_group_exit(runtime.pgid, timeout):
                try:
                    os.killpg(runtime.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise BackendStopError(
                        f"permission denied killing {self.config.name} process group {runtime.pgid}"
                    ) from exc
                if not await self._wait_group_exit(runtime.pgid, 10.0):
                    raise BackendStopError(
                        f"{self.config.name} process group {runtime.pgid} survived SIGKILL"
                    )

        if runtime.process is not None:
            try:
                await asyncio.wait_for(runtime.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # The verified process group is gone; an asyncio child-watcher
                # delay must not block a backend switch indefinitely.
                pass
        self.runtime = None
        self._close_log()

    def is_running(self) -> bool:
        runtime = self.runtime
        if runtime is None:
            return False
        # A launcher/leader can die while CUDA or API workers in its process
        # group survive. Track the whole group so shutdown and the next switch
        # reclaim those workers instead of starting a competing backend.
        return self._pid_alive(runtime.pid) or self._group_alive(runtime.pgid)

    async def is_healthy(self) -> bool:
        if not self.is_running():
            return False
        url = f"{self.upstream_url}{self.config.health_endpoint}"
        try:
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                response = await client.get(url)
            return 200 <= response.status_code < 300
        except (httpx.HTTPError, OSError):
            return False

    async def wait_until_model_loaded(self, model: ModelConfig) -> None:
        """Wait until the target model is actually servable on this backend.

        A plain health check only proves the process is up. Backends that
        host multiple models (e.g. llama.cpp --models-preset) report
        /health 200 while no model is loaded yet; the router must warm the
        requested model into the backend's LRU before handing it requests.
        """
        return None

    async def wait_until_ready(self) -> None:
        timeout = self.config.startup_timeout_seconds or self.settings.startup_timeout_seconds
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if not self.is_running():
                code = None
                if self.runtime and self.runtime.process:
                    code = self.runtime.process.returncode
                raise BackendStartupError(
                    f"{self.config.name} exited before becoming healthy"
                    + (f" (exit code {code})" if code is not None else "")
                )
            if await self.is_healthy():
                return
            await asyncio.sleep(self.settings.health_poll_seconds)
        raise BackendStartupError(
            f"{self.config.name} did not become healthy within {timeout:.1f}s"
        )

    async def fetch_models(self) -> list[dict[str, Any]]:
        if not await self.is_healthy():
            return []
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                response = await client.get(f"{self.upstream_url}/v1/models")
                response.raise_for_status()
            payload = response.json()
            return [item for item in payload.get("data", []) if isinstance(item, dict)]
        except (httpx.HTTPError, ValueError, OSError):
            return []

    def adopt(self, pid: int) -> bool:
        if not self.config.adopt_existing or not self.process_matches(pid):
            return False
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            return False
        # Never adopt a listener from an unrelated shell/session and later send
        # a group signal to that shell. All supported launchers are session
        # leaders (router children use start_new_session=True).
        if pgid != pid:
            return False
        self.runtime = BackendRuntime(
            pid=pid,
            pgid=pgid,
            process=None,
            adopted=True,
            started_at=time.time(),
        )
        return True

    def adopt_persisted_group(self, record: dict[str, Any]) -> bool:
        """Recover a router-owned group whose leader died but workers survived."""
        if not self.config.adopt_existing:
            return False
        pid = record.get("pid")
        pgid = record.get("pgid")
        boot_id = self._boot_id()
        if (
            not isinstance(pid, int)
            or not isinstance(pgid, int)
            or pid != pgid
            or pgid <= 1
            or self._pid_alive(pid)
            or not self._group_alive(pgid)
            or not boot_id
            or record.get("boot_id") != boot_id
            or record.get("command") != list(self.config.command)
        ):
            return False

        members: list[int] = []
        current_uid = os.getuid()
        for process in psutil.process_iter(["pid", "uids"]):
            try:
                if os.getpgid(process.pid) != pgid:
                    continue
                uids = process.info.get("uids")
                if uids is None or uids.real != current_uid:
                    return False
                members.append(process.pid)
            except (ProcessLookupError, PermissionError, psutil.Error):
                continue
        if not members:
            return False
        self.runtime = BackendRuntime(
            pid=pid,
            pgid=pgid,
            process=None,
            adopted=True,
            started_at=float(record.get("started_at") or time.time()),
        )
        return True

    async def discover_listener_pid(self) -> int | None:
        if not self.config.adopt_existing:
            return None

        def inspect() -> int | None:
            try:
                connections = psutil.net_connections(kind="tcp")
            except (psutil.AccessDenied, OSError):
                return None
            candidates: list[int] = []
            for connection in connections:
                if (
                    connection.status == psutil.CONN_LISTEN
                    and connection.laddr
                    and connection.laddr.port == self.config.port
                    and connection.pid
                    and self.process_matches(connection.pid)
                ):
                    candidates.append(connection.pid)
            return min(candidates) if candidates else None

        return await asyncio.to_thread(inspect)

    def process_matches(self, pid: int) -> bool:
        if not self._pid_alive(pid):
            return False
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return False
        return all(fragment in command for fragment in self.config.process_match)

    def state_record(self) -> dict[str, Any] | None:
        if not self.runtime or not self.is_running():
            return None
        return {
            "backend": self.config.name,
            "pid": self.runtime.pid,
            "pgid": self.runtime.pgid,
            "adopted": self.runtime.adopted,
            "started_at": self.runtime.started_at,
            "boot_id": self._boot_id(),
            "command": list(self.config.command),
        }

    def tail_log(self, max_lines: int = 40, max_bytes: int = 16_384) -> str:
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                text = handle.read().decode(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-max_lines:])

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "running": self.is_running(),
            "pid": self.runtime.pid if self.runtime else None,
            "pgid": self.runtime.pgid if self.runtime else None,
            "adopted": self.runtime.adopted if self.runtime else False,
            "upstream": self.upstream_url,
            "log": str(self.log_path),
        }

    @staticmethod
    def _boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            return ""

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Permission denied proves that the group still exists; treating it
            # as released could permit a second large backend to start.
            return True

    @classmethod
    async def _wait_group_exit(cls, pgid: int, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if not cls._group_alive(pgid):
                return True
            await asyncio.sleep(0.1)
        return not cls._group_alive(pgid)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None


class LlamaCppAdapter(BackendAdapter):
    async def wait_until_model_loaded(self, model: ModelConfig) -> None:
        """Wait until llama.cpp has the requested model loaded.

        llama.cpp's multi-model router server keeps /health at 200 while a
        model is still loading, and rejects requests for the model (400
        "model is not loaded") until its LRU slot is ready. A GET /props
        with autoload (the default) triggers the load and blocks server-side
        until it completes, so a single long-timeout request both starts and
        awaits the load without re-issuing requests that would cancel the
        in-flight load on client disconnect.
        """
        upstream_model = self.translate_query_model(model)
        timeout = self.config.startup_timeout_seconds or self.settings.startup_timeout_seconds
        url = f"{self.upstream_url}/props"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise BackendStartupError(
                    f"{self.config.name} model {upstream_model!r} was not loaded within {timeout:.0f}s"
                )
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(remaining, connect=5.0), trust_env=False
                ) as client:
                    response = await client.get(url, params={"model": upstream_model})
            except (httpx.HTTPError, OSError) as exc:
                raise BackendStartupError(
                    f"{self.config.name} model {upstream_model!r} readiness probe failed: {exc}"
                ) from exc
            if response.status_code == 200:
                return
            # A fast 4xx/5xx means no load is in flight on our socket (an
            # autoload probe that started a load holds the request until it
            # finishes), so retrying cannot cancel a load. It usually just
            # means the model list is not registered yet right after start.
            detail = ""
            try:
                payload = response.json()
                error = payload.get("error")
                if isinstance(error, dict) and error.get("message"):
                    detail = f" (props returned {response.status_code}: {error['message']})"
            except (ValueError, AttributeError):
                detail = f" (props returned {response.status_code})"
            LOG.debug(
                "%s model %s not ready yet%s; retrying",
                self.config.name, upstream_model, detail,
            )
            await asyncio.sleep(min(self.settings.health_poll_seconds, remaining))


class VllmAdapter(BackendAdapter):
    pass


class NInferAdapter(BackendAdapter):
    pass


class SglangAdapter(BackendAdapter):
    pass


class StrataAdapter(BackendAdapter):
    pass


def create_adapter(config: BackendConfig, settings: RouterSettings) -> BackendAdapter:
    adapters: dict[str, type[BackendAdapter]] = {
        "llamacpp": LlamaCppAdapter,
        "vllm": VllmAdapter,
        "ninfer": NInferAdapter,
        "sglang": SglangAdapter,
        "strata": StrataAdapter,
    }
    try:
        adapter_type = adapters[config.adapter]
    except KeyError as exc:
        raise ValueError(f"unknown backend adapter {config.adapter!r}") from exc
    return adapter_type(config, settings)
