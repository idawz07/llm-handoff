"""Tests for the operator-facing features: hot reload, idle reclaim, resident
backends, activity log, log tails, and a llama-free configuration."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handoff_router.app import create_app  # noqa: E402
from handoff_router.manager import RouterManager  # noqa: E402
from handoff_router.registry import ModelRegistry  # noqa: E402

FAKE = Path(__file__).with_name("fake_backend.py")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class Config:
    path: Path
    ports: dict[str, int]
    active_dir: Path
    counter_dir: Path

    def command(self, name: str, served: str) -> list[str]:
        return [
            sys.executable,
            str(FAKE),
            "--port",
            str(self.ports[name]),
            "--name",
            name,
            "--served-model",
            served,
            "--active-dir",
            str(self.active_dir),
            "--counter-dir",
            str(self.counter_dir),
        ]

    def starts(self, name: str) -> int:
        path = self.counter_dir / f"{name}.starts"
        return int(path.read_text()) if path.exists() else 0

    def active(self) -> set[str]:
        return {path.stem for path in self.active_dir.glob("*.active")} if self.active_dir.exists() else set()

    def write(self, document: dict) -> None:
        self.path.write_text(yaml.safe_dump(document, sort_keys=False))
        # Guarantee the mtime moves, so a fast test still trips the watcher.
        stat = self.path.stat()
        import os
        os.utime(self.path, (stat.st_atime + 1, stat.st_mtime + 1))


def make_config(tmp_path: Path, **settings_override) -> Config:
    ports = {name: free_port() for name in ("llamacpp", "vllm", "ninfer")}
    config = Config(
        path=tmp_path / "router-backends.yaml",
        ports=ports,
        active_dir=tmp_path / "active",
        counter_dir=tmp_path / "counters",
    )
    presets = tmp_path / "presets.ini"
    presets.write_text("[*]\nthreads = 2\n\n[llama-real]\nmodel = /tmp/fake.gguf\n")

    def backend(name: str, served: str, **extra):
        adapter = "llamacpp" if name == "llamacpp" else ("ninfer" if name == "ninfer" else "vllm")
        return {
            "adapter": adapter,
            "command": config.command(name, served),
            "upstream": f"http://127.0.0.1:{ports[name]}",
            "health_endpoint": "/health",
            "startup_timeout_seconds": 5,
            "stop_timeout_seconds": 1,
            "process_match": ["fake_backend.py", f"--name {name}"],
            **extra,
        }

    settings = {
        "state_dir": str(tmp_path / "state"),
        "llama_backend": "llamacpp",
        "llama_presets": str(presets),
        "drain_timeout_seconds": 2,
        "shutdown_drain_timeout_seconds": 1,
        "stop_timeout_seconds": 1,
        "startup_timeout_seconds": 5,
        "health_poll_seconds": 0.03,
        "min_free_vram_mib": 0,
        "gpu_release_timeout_seconds": 2,
        "gpu_poll_seconds": 0.02,
        "idle_poll_seconds": 0.05,
        "config_poll_seconds": 0.05,
        **settings_override,
    }
    config.write(
        {
            "settings": settings,
            "backends": {
                "llamacpp": backend("llamacpp", "llama-real"),
                "vllm": backend("vllm", "vllm-real"),
                "ninfer": backend("ninfer", "ninfer-real"),
            },
            "models": {
                "big-llama": {"backend": "llamacpp", "upstream_model": "llama-real", "selector": True},
                "big-vllm": {"backend": "vllm", "upstream_model": "vllm-real", "selector": True},
                "sidecar": {"backend": "ninfer", "upstream_model": "ninfer-real", "selector": True},
            },
        }
    )
    return config


@dataclass
class Harness:
    config: Config
    manager: RouterManager
    client: httpx.AsyncClient
    server: uvicorn.Server
    server_task: asyncio.Task

    async def chat(self, model: str) -> httpx.Response:
        return await self.client.post(
            "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]}
        )


@pytest_asyncio.fixture
async def harness(tmp_path: Path) -> Harness:
    config = make_config(tmp_path)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry, config_path=config.path)
    await manager.startup()
    app = create_app(registry, manager=manager, manage_lifespan=False)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=15)
    try:
        yield Harness(config, manager, client, server, task)
    finally:
        await client.aclose()
        if app.state.client is not None:
            await app.state.client.aclose()
        server.should_exit = True
        await task
        await manager.shutdown()


@pytest.mark.asyncio
async def test_hot_reload_adds_models_without_touching_the_active_backend(harness: Harness):
    assert (await harness.chat("big-llama")).status_code == 200
    assert harness.manager.active_backend == "llamacpp"
    vllm_starts_before = harness.config.starts("vllm")

    document = yaml.safe_load(harness.config.path.read_text())
    document["models"]["big-vllm"]["metadata"] = {"context_window": 32768}
    document["models"]["brand-new"] = {"backend": "vllm", "upstream_model": "vllm-real", "selector": True}
    harness.config.write(document)

    response = await harness.client.post("/router/reload")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    assert "brand-new" in harness.manager.registry.models

    # The running backend must be untouched: same process, no extra start.
    assert harness.manager.active_backend == "llamacpp"
    assert harness.config.starts("llamacpp") == 1
    assert vllm_starts_before == 0 and harness.config.starts("vllm") == 0
    assert harness.manager.adapters["vllm"].is_running() is False

    listing = (await harness.client.get("/v1/models")).json()
    assert any(item["id"] == "brand-new" for item in listing["data"])


@pytest.mark.asyncio
async def test_changed_launch_line_is_applied_on_the_next_switch(harness: Harness):
    assert (await harness.chat("big-vllm")).status_code == 200
    assert harness.config.starts("vllm") == 1

    document = yaml.safe_load(harness.config.path.read_text())
    document["backends"]["vllm"]["command"] = harness.config.command("vllm", "vllm-real-v2")
    harness.config.write(document)

    response = await harness.client.post("/router/reload")
    assert response.status_code == 200, response.text
    assert "vllm" in response.json()["changed"]
    assert harness.manager.active_backend == "vllm"  # active process survives the reload
    assert harness.config.starts("vllm") == 1

    assert (await harness.chat("big-vllm")).status_code == 200
    assert harness.config.starts("vllm") == 2, "changed backend must restart on the next switch"


@pytest.mark.asyncio
async def test_rejected_config_keeps_the_running_one(harness: Harness):
    assert (await harness.chat("big-llama")).status_code == 200
    harness.config.path.write_text("settings: {state_dir: x}\nbackends: {broken: {upstream: http://x}}\n")
    import os
    stat = harness.config.path.stat()
    os.utime(harness.config.path, (stat.st_atime + 1, stat.st_mtime + 1))

    response = await harness.client.post("/router/reload")
    assert response.status_code == 503
    assert "reload failed" in response.json()["error"]["message"]
    assert harness.manager.active_backend == "llamacpp"
    assert harness.config.active() == {"llamacpp"}


@pytest.mark.asyncio
async def test_config_watcher_picks_up_an_edit_without_an_api_call(harness: Harness):
    assert (await harness.chat("big-llama")).status_code == 200
    document = yaml.safe_load(harness.config.path.read_text())
    document["models"]["watched"] = {"backend": "ninfer", "upstream_model": "ninfer-real", "selector": True}
    harness.config.write(document)

    for _ in range(200):
        if "watched" in harness.manager.registry.models:
            break
        await asyncio.sleep(0.05)
    assert "watched" in harness.manager.registry.models
    assert harness.manager.active_backend == "llamacpp", "watcher must not disturb the active model"


@pytest.mark.asyncio
async def test_idle_ttl_reclaims_the_active_backend(tmp_path: Path):
    config = make_config(tmp_path, idle_ttl_seconds=0.4, idle_grace_seconds=0.0)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry, config_path=config.path)
    await manager.startup()
    try:
        lease = await manager.acquire("big-llama")
        await lease.release()
        assert manager.active_backend == "llamacpp"
        for _ in range(200):
            if manager.active_backend is None and not config.active():
                break
            await asyncio.sleep(0.05)
        assert config.active() == set(), "idle backend should have been unloaded"
        assert manager.active_backend is None
        kinds = [event["kind"] for event in manager.activity.recent()]
        assert "idle_unload" in kinds
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_resident_backend_coexists_with_a_switch(tmp_path: Path):
    config = make_config(tmp_path)
    document = yaml.safe_load(config.path.read_text())
    document["backends"]["ninfer"]["resident"] = True
    config.write(document)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry, config_path=config.path)
    await manager.startup()
    try:
        lease = await manager.acquire("sidecar")
        await lease.release()
        assert manager.active_backend == "ninfer"

        lease = await manager.acquire("big-llama")
        await lease.release()
        assert manager.active_backend == "llamacpp"
        assert config.active() == {"ninfer", "llamacpp"}, "resident backend must survive the switch"
        assert manager.adapters["ninfer"].is_running()

        # And a second request for the resident model costs nothing.
        lease = await manager.acquire("sidecar")
        await lease.release()
        assert config.starts("ninfer") == 1
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_idle_unload_is_cheap_first_and_honors_per_backend_ttl(tmp_path: Path):
    config = make_config(tmp_path, idle_ttl_seconds=0.3, idle_grace_seconds=0.0)
    document = yaml.safe_load(config.path.read_text())
    document["backends"]["ninfer"]["resident"] = True
    document["backends"]["llamacpp"]["evict_cost"] = 50
    document["backends"]["llamacpp"]["idle_ttl_seconds"] = 600  # pinned LLM, short sidecar TTL
    config.write(document)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry, config_path=config.path)
    await manager.startup()
    try:
        for model in ("sidecar", "big-llama"):
            lease = await manager.acquire(model)
            await lease.release()
        assert config.active() == {"ninfer", "llamacpp"}
        manager.last_used["ninfer"] -= 10
        manager.last_used["llamacpp"] -= 10
        await manager._reap_once()
        assert config.active() == {"llamacpp"}, "only the sidecar was past its own TTL"
        assert "idle_unload" in [event["kind"] for event in manager.activity.recent()]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_activity_log_and_log_tails(harness: Harness):
    assert (await harness.chat("big-llama")).status_code == 200
    events = (await harness.client.get("/router/activity")).json()["events"]
    kinds = {event["kind"] for event in events}
    assert "startup" in kinds and "backend_ready" in kinds
    assert all("at" in event and "kind" in event for event in events)

    logs = (await harness.client.get("/logs/llamacpp?lines=20")).json()
    assert logs["running"] is True
    assert any("fake backend llamacpp starting" in line for line in logs["lines"])

    status = (await harness.client.get("/router/status")).json()
    assert status["config"]["path"].endswith("router-backends.yaml")
    assert status["backends"]["llamacpp"]["idle_seconds"] is not None
    assert status["backends"]["vllm"]["running"] is False


@pytest.mark.asyncio
async def test_activity_stream_is_live(harness: Harness):
    async with harness.client.stream(
        "GET", "/activity/stream", timeout=httpx.Timeout(20, read=20)
    ) as response:
        assert response.status_code == 200
        lines = response.aiter_lines()
        await asyncio.sleep(0.3)  # let the server subscribe before generating events
        await harness.chat("big-vllm")
        seen: list[str] = []
        async for line in lines:
            if line.startswith("data:"):
                seen.append(json.loads(line[5:]))
                if any(event["kind"] == "backend_ready" for event in seen):
                    break
        assert any(event.get("model") == "big-vllm" for event in seen)


@pytest.mark.asyncio
async def test_registry_without_a_llama_router(tmp_path: Path):
    ports = {"vllm": free_port()}
    config_path = tmp_path / "router-backends.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "settings": {
                    "state_dir": str(tmp_path / "state"),
                    "startup_timeout_seconds": 5,
                    "stop_timeout_seconds": 1,
                    "health_poll_seconds": 0.03,
                },
                "backends": {
                    "vllm": {
                        "adapter": "vllm",
                        "command": [
                            sys.executable, str(FAKE), "--port", str(ports["vllm"]),
                            "--name", "vllm", "--served-model", "any-vllm",
                            "--active-dir", str(tmp_path / "active"),
                            "--counter-dir", str(tmp_path / "counters"),
                        ],
                        "upstream": f"http://127.0.0.1:{ports['vllm']}",
                        "process_match": ["fake_backend.py", "--name vllm"],
                    }
                },
                "models": {
                    "any-vllm": {
                        "backend": "vllm",
                        "owned_by": "vllm",
                        "metadata": {"context_window": 8192},
                    }
                },
            },
            sort_keys=False,
        )
    )
    registry = ModelRegistry.from_file(config_path)
    assert set(registry.models) == {"any-vllm"}
    manager = RouterManager(registry, config_path=config_path)
    await manager.startup()
    try:
        lease = await manager.acquire("any-vllm")
        await lease.release()
        assert manager.active_backend == "vllm"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_model_load_requires_a_key_unless_opted_out(tmp_path: Path):
    import subprocess

    config = make_config(tmp_path)
    port = free_port()
    proc = subprocess.run(
        [
            sys.executable, "-m", "handoff_router.cli",
            "--config", str(config.path), "--host", "127.0.0.1", "--port", str(port),
            "--api-key", "sekret", "--print-config",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0 and "backends" in proc.stdout

    registry = ModelRegistry.from_file(config.path)
    from handoff_router.app import create_app as build

    manager = RouterManager(registry, config_path=config.path)
    app = build(registry, manager=manager, api_keys={"sekret"}, manage_lifespan=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://r") as client:
        unauthenticated = await client.post("/models/unload", json={"model": "big-llama"})
        assert unauthenticated.status_code == 401
        listed = await client.get("/v1/models")
        assert listed.status_code == 200  # model listing stays public, as llama.cpp does

    app_open = build(
        registry, manager=manager, api_keys={"sekret"}, manage_lifespan=False,
        keyless_model_api=True,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_open), base_url="http://r") as client:
        assert (await client.post("/models/unload", json={"model": "big-llama"})).status_code == 200
