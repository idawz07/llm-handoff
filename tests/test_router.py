from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handoff_router.app import create_app  # noqa: E402
from handoff_router.backends import create_adapter  # noqa: E402
from handoff_router.cli import load_api_keys  # noqa: E402
from handoff_router.manager import ModelSwitchError, RouterManager  # noqa: E402
from handoff_router.registry import ModelRegistry  # noqa: E402

FAKE = Path(__file__).with_name("fake_backend.py")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_registry(
    tmp_path: Path,
    *,
    delays: dict[str, float] | None = None,
    settings_override: dict | None = None,
) -> Path:
    delays = delays or {}
    settings_override = settings_override or {}
    active_dir = tmp_path / "active"
    counter_dir = tmp_path / "counters"
    presets = tmp_path / "presets.ini"
    presets.write_text("[*]\nthreads = 2\n\n[llama-real]\nmodel = /tmp/fake.gguf\n")
    ports = {name: free_port() for name in ("llamacpp", "vllm", "ninfer", "failure")}

    def backend(name: str, served: str, *, fail: bool = False):
        command = [
            sys.executable,
            str(FAKE),
            "--port",
            str(ports[name]),
            "--name",
            name,
            "--served-model",
            served,
            "--active-dir",
            str(active_dir),
            "--counter-dir",
            str(counter_dir),
        ]
        if delays.get(name):
            command += ["--startup-delay", str(delays[name])]
        if fail:
            command.append("--fail-start")
        return {
            "adapter": "llamacpp" if name == "llamacpp" else ("ninfer" if name == "ninfer" else "vllm"),
            "command": command,
            "upstream": f"http://127.0.0.1:{ports[name]}",
            "health_endpoint": "/health",
            "startup_timeout_seconds": 2,
            "stop_timeout_seconds": 1,
            "process_match": ["fake_backend.py", f"--name {name}"],
        }

    settings = {
        "state_dir": str(tmp_path / "state"),
        "llama_backend": "llamacpp",
        "llama_presets": str(presets),
        "drain_timeout_seconds": 2,
        "shutdown_drain_timeout_seconds": 1,
        "stop_timeout_seconds": 1,
        "startup_timeout_seconds": 2,
        "health_poll_seconds": 0.03,
        "min_free_vram_mib": 0,
        "gpu_release_timeout_seconds": 1,
        "gpu_poll_seconds": 0.02,
    }
    settings.update(settings_override)
    config = {
        "settings": settings,
        "backends": {
            "llamacpp": backend("llamacpp", "llama-real"),
            "vllm": backend("vllm", "vllm-real"),
            "ninfer": backend("ninfer", "ninfer-real"),
            "failure": backend("failure", "failure-real", fail=True),
        },
        "models": {
            "qwen3.8-q5": {"backend": "llamacpp", "upstream_model": "llama-real", "selector": True},
            "qwen3.8-vllm": {"backend": "vllm", "upstream_model": "vllm-real", "selector": True},
            "qwen3.8-ninfer": {"backend": "ninfer", "upstream_model": "ninfer-real", "selector": True},
            "broken": {"backend": "failure", "upstream_model": "failure-real", "selector": True},
        },
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@dataclass
class Harness:
    tmp_path: Path
    registry: ModelRegistry
    manager: RouterManager
    client: httpx.AsyncClient
    server: uvicorn.Server
    server_task: asyncio.Task

    def starts(self, name: str) -> int:
        path = self.tmp_path / "counters" / f"{name}.starts"
        return int(path.read_text()) if path.exists() else 0

    def active(self) -> set[str]:
        directory = self.tmp_path / "active"
        return {path.stem for path in directory.glob("*.active")} if directory.exists() else set()


@pytest_asyncio.fixture
async def harness(tmp_path: Path) -> Harness:
    registry = ModelRegistry.from_file(write_registry(tmp_path, delays={"vllm": 0.25}))
    manager = RouterManager(registry)
    await manager.startup()
    app = create_app(registry, manager=manager, manage_lifespan=False)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10, trust_env=False)
    result = Harness(tmp_path, registry, manager, client, server, task)
    try:
        yield result
    finally:
        await client.aclose()
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        await manager.shutdown()


async def chat(client: httpx.AsyncClient, model: str, **extra):
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    return await client.post("/v1/chat/completions", json=payload)


def assert_exclusive(harness: Harness, expected: str) -> None:
    assert harness.active() == {expected}


@pytest.mark.asyncio
async def test_switch_matrix_sticky_models_fields_and_listing(harness: Harness):
    empty_props = await harness.client.get("/props")
    assert empty_props.status_code == 200
    assert empty_props.json()["role"] == "router"

    fields = {
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_completion_tokens": 41,
        "stop": ["END"],
        "reasoning_effort": "medium",
    }
    response = await chat(harness.client, "qwen3.8-q5", **fields)
    assert response.status_code == 200
    assert response.json()["model"] == "qwen3.8-q5"
    received = response.json()["received"]
    assert received["model"] == "llama-real"
    for key, value in fields.items():
        assert received[key] == value
    assert_exclusive(harness, "llamacpp")
    props = await harness.client.get("/props")
    assert props.status_code == 200
    assert props.json()["model_alias"] == "qwen3.8-q5"
    assert props.json()["role"] == "router"

    # Metadata probes from the WebUI must not act like model selections.
    probe = await harness.client.get(
        "/props", params={"model": "qwen3.8-vllm", "autoload": "false"}
    )
    assert probe.status_code == 400
    assert harness.manager.active_model == "qwen3.8-q5"
    assert harness.starts("vllm") == 0

    response = await chat(harness.client, "qwen3.8-vllm")
    assert response.json()["backend"] == "vllm"
    assert_exclusive(harness, "vllm")

    response = await chat(harness.client, "qwen3.8-ninfer")
    assert response.json()["backend"] == "ninfer"
    assert_exclusive(harness, "ninfer")

    response = await chat(harness.client, "qwen3.8-vllm")
    assert response.json()["backend"] == "vllm"
    get_response = await harness.client.get(
        "/echo",
        params={"model": "qwen3.8-vllm", "value": "a b"},
        headers={"X-Preserved": "yes"},
    )
    assert get_response.status_code == 200
    echoed = get_response.json()
    assert "model=vllm-real" in echoed["path"]
    assert "value=a+b" in echoed["path"]
    assert echoed["headers"]["x-preserved"] == "yes"
    starts = harness.starts("vllm")
    response = await chat(harness.client, "qwen3.8-vllm")
    assert response.status_code == 200
    assert harness.starts("vllm") == starts
    assert_exclusive(harness, "vllm")

    listing = (await harness.client.get("/v1/models")).json()["data"]
    ids = {item["id"] for item in listing}
    assert {"llama-real", "qwen3.8-q5", "qwen3.8-vllm", "qwen3.8-ninfer"} <= ids
    statuses = {item["id"]: item["status"]["value"] for item in listing}
    assert statuses["qwen3.8-vllm"] == "loaded"
    assert statuses["qwen3.8-ninfer"] == "unloaded"


@pytest.mark.asyncio
async def test_upstream_connections_are_not_reused(harness: Harness):
    first = await harness.client.get("/echo", params={"model": "qwen3.8-q5"})
    second = await harness.client.get("/echo", params={"model": "qwen3.8-q5"})
    assert first.status_code == second.status_code == 200
    assert first.json()["client_port"] != second.json()["client_port"]


@pytest.mark.asyncio
async def test_two_requests_join_one_backend_swap(harness: Harness):
    assert (await chat(harness.client, "qwen3.8-q5")).status_code == 200
    first, second = await asyncio.gather(
        chat(harness.client, "qwen3.8-vllm"),
        chat(harness.client, "qwen3.8-vllm"),
    )
    assert first.status_code == second.status_code == 200
    assert harness.starts("vllm") == 1
    assert_exclusive(harness, "vllm")


@pytest.mark.asyncio
async def test_switch_waits_for_active_request_to_drain(harness: Harness):
    slow = asyncio.create_task(chat(harness.client, "qwen3.8-q5", test_delay=0.35))
    for _ in range(100):
        if harness.manager.active_request_count == 1:
            break
        await asyncio.sleep(0.01)
    assert harness.manager.active_request_count == 1
    switching = asyncio.create_task(chat(harness.client, "qwen3.8-ninfer"))
    await asyncio.sleep(0.1)
    assert_exclusive(harness, "llamacpp")
    assert not switching.done()
    assert (await slow).status_code == 200
    assert (await switching).status_code == 200
    assert_exclusive(harness, "ninfer")


@pytest.mark.asyncio
async def test_cancelled_proxy_releases_request_lease(tmp_path: Path):
    registry = ModelRegistry.from_file(write_registry(tmp_path))
    manager = RouterManager(registry)
    await manager.startup()
    app = create_app(registry, manager=manager, manage_lifespan=False)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=10) as client:
            request = asyncio.create_task(
                chat(client, "qwen3.8-vllm", test_delay=5)
            )
            for _ in range(200):
                if manager.active_request_count == 1:
                    break
                await asyncio.sleep(0.01)
            assert manager.active_request_count == 1
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            for _ in range(100):
                if manager.active_request_count == 0:
                    break
                await asyncio.sleep(0.01)
            assert manager.active_request_count == 0
    finally:
        if app.state.client is not None:
            await app.state.client.aclose()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_model_status_sse_follows_explicit_load_switch(harness: Harness):
    assert (await chat(harness.client, "qwen3.8-q5")).status_code == 200

    seen: list[str] = []
    async with asyncio.timeout(10):
        async with harness.client.stream("GET", "/models/sse") as response:
            assert response.status_code == 200
            load = asyncio.create_task(
                harness.client.post("/models/load", json={"model": "qwen3.8-vllm"})
            )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line.removeprefix("data: "))
                if event["model"] != "qwen3.8-vllm":
                    continue
                seen.append(event["data"]["status"])
                if event["data"]["status"] == "loaded":
                    break
            loaded = await load

    assert loaded.status_code == 200
    assert "loaded" in seen
    assert harness.manager.active_model == "qwen3.8-vllm"
    assert_exclusive(harness, "vllm")


@pytest.mark.asyncio
async def test_sse_stream_is_forwarded_incrementally(harness: Harness):
    arrivals: list[float] = []
    chunks: list[str] = []
    async with harness.client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qwen3.8-ninfer", "messages": [{"role": "user", "content": "stream"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                arrivals.append(asyncio.get_running_loop().time())
                chunks.append(line)
    assert any("one" in chunk for chunk in chunks)
    assert any("two" in chunk for chunk in chunks)
    json_chunks = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks[:-1]]
    assert all(chunk["model"] == "qwen3.8-ninfer" for chunk in json_chunks)
    assert chunks[-1] == "data: [DONE]"
    assert arrivals[-1] - arrivals[0] >= 0.12
    assert harness.manager.active_request_count == 0


@pytest.mark.asyncio
async def test_backend_startup_failure_includes_log_tail(harness: Harness):
    assert (await chat(harness.client, "qwen3.8-q5")).status_code == 200
    assert_exclusive(harness, "llamacpp")
    response = await chat(harness.client, "broken")
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "backend_switch_error"
    assert error["backend"] == "failure"
    assert "intentional startup failure" in error["logs"]
    assert harness.manager.active_backend is None
    assert harness.active() == set()


@pytest.mark.asyncio
async def test_backend_crash_is_restarted_on_next_request(harness: Harness):
    assert (await chat(harness.client, "qwen3.8-vllm")).status_code == 200
    runtime = harness.manager.adapters["vllm"].runtime
    assert runtime is not None
    os.killpg(runtime.pgid, signal.SIGKILL)
    for _ in range(100):
        if not harness.manager.adapters["vllm"].is_running():
            break
        await asyncio.sleep(0.02)
    response = await chat(harness.client, "qwen3.8-vllm")
    assert response.status_code == 200
    assert harness.starts("vllm") == 2
    assert_exclusive(harness, "vllm")


@pytest.mark.asyncio
async def test_router_restart_adopts_stale_backend_without_duplicate(tmp_path: Path):
    registry_path = write_registry(tmp_path)
    registry1 = ModelRegistry.from_file(registry_path)
    manager1 = RouterManager(registry1)
    await manager1.startup()
    lease = await manager1.acquire("qwen3.8-ninfer")
    await lease.release()
    runtime1 = manager1.adapters["ninfer"].runtime
    assert runtime1 is not None

    registry2 = ModelRegistry.from_file(registry_path)
    manager2 = RouterManager(registry2)
    await manager2.startup()
    assert manager2.active_backend == "ninfer"
    assert manager2.adapters["ninfer"].runtime is not None
    assert manager2.adapters["ninfer"].runtime.pid == runtime1.pid
    counter = tmp_path / "counters" / "ninfer.starts"
    assert int(counter.read_text()) == 1

    lease2 = await manager2.acquire("qwen3.8-ninfer")
    await lease2.release()
    assert int(counter.read_text()) == 1
    await manager2.shutdown()
    await manager1.shutdown()


@pytest.mark.asyncio
async def test_persisted_group_is_recovered_when_launcher_leader_dies(tmp_path: Path):
    registry = ModelRegistry.from_file(write_registry(tmp_path))
    config = replace(
        registry.backends["vllm"],
        command=("/bin/sh", "-c", "sleep 30 &"),
        process_match=("sleep 30",),
    )
    original = create_adapter(config, registry.settings)
    recovered = create_adapter(config, registry.settings)
    try:
        await original.start()
        runtime = original.runtime
        assert runtime is not None and runtime.process is not None
        record = original.state_record()
        assert record is not None
        await runtime.process.wait()
        assert original.is_running()
        assert recovered.adopt_persisted_group(record)
        await recovered.stop()
        assert not recovered.is_running()
    finally:
        await recovered.stop()
        await original.stop()


@pytest.mark.asyncio
async def test_api_key_parity_and_cors_preflight(tmp_path: Path):
    registry = ModelRegistry.from_file(write_registry(tmp_path))
    manager = RouterManager(registry)
    await manager.startup()
    app = create_app(
        registry,
        manager=manager,
        manage_lifespan=False,
        api_keys={"first", "second"},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/v1/models")).status_code == 200
        assert (await chat(client, "qwen3.8-q5")).status_code == 401
        assert (
            await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "first"},
                json={"model": "qwen3.8-q5", "messages": []},
            )
        ).status_code == 200
        assert (
            await client.post(
                "/v1/chat/completions",
                headers={"X-Api-Key": "second"},
                json={"model": "qwen3.8-q5", "messages": []},
            )
        ).status_code == 200
        preflight = await client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "https://example.test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://example.test"
    if app.state.client is not None:
        await app.state.client.aclose()
    await manager.shutdown()


def test_api_key_loader_supports_commas_environment_and_files(tmp_path: Path, monkeypatch):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("# comment\nfrom-file\n\n")
    monkeypatch.setenv("LLAMA_API_KEY", "from-env,env-two")
    assert load_api_keys(["one,two", "three"], [str(key_file)]) == {
        "one",
        "two",
        "three",
        "from-env",
        "env-two",
        "from-file",
    }


@pytest.mark.asyncio
async def test_initial_start_is_blocked_until_vram_threshold(tmp_path: Path):
    available = tmp_path / "available"
    available.write_text("100\n")
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text(f"#!/bin/sh\ncat {available}\n")
    fake_smi.chmod(0o755)
    registry = ModelRegistry.from_file(
        write_registry(
            tmp_path,
            settings_override={
                "nvidia_smi": str(fake_smi),
                "min_free_vram_mib": 500,
                "gpu_release_timeout_seconds": 0.12,
                "gpu_poll_seconds": 0.02,
            },
        )
    )
    manager = RouterManager(registry)
    await manager.startup()
    with pytest.raises(ModelSwitchError, match="GPU memory did not reach"):
        await manager.acquire("qwen3.8-vllm")
    assert not (tmp_path / "counters" / "vllm.starts").exists()

    available.write_text("1000\n")
    lease = await manager.acquire("qwen3.8-vllm")
    await lease.release()
    assert int((tmp_path / "counters" / "vllm.starts").read_text()) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_startup_completes_without_half_started_process(tmp_path: Path):
    registry = ModelRegistry.from_file(write_registry(tmp_path, delays={"vllm": 0.2}))
    manager = RouterManager(registry)
    await manager.startup()
    request = asyncio.create_task(manager.acquire("qwen3.8-vllm"))
    await asyncio.sleep(0.05)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert manager.active_backend == "vllm"
    assert manager.active_request_count == 0
    assert manager.adapters["vllm"].is_running()
    await manager.shutdown()


def test_dispatcher_install_is_idempotent_and_non_router_calls_delegate(tmp_path: Path):
    root = tmp_path / "llama.cpp"
    binary = root / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nprintf 'real:%s\\n' \"$*\"\n")
    binary.chmod(0o755)
    environment = {
        **os.environ,
        "LLAMA_SERVER_ROOT": str(root),
        "HANDOFF_DISPATCHER": str(ROOT / "deploy" / "dispatcher.sh"),
    }
    install = ROOT / "deploy" / "install-dispatcher.sh"
    uninstall = ROOT / "deploy" / "uninstall-dispatcher.sh"
    subprocess.run([str(install)], env=environment, check=True, capture_output=True, text=True)
    subprocess.run([str(install)], env=environment, check=True, capture_output=True, text=True)
    real = binary.with_name("llama-server.real")
    delegated = subprocess.run(
        [str(binary), "--version"],
        env={**environment, "LLAMA_SERVER_REAL": str(real)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert delegated.stdout.strip() == "real:--version"
    subprocess.run([str(uninstall)], env=environment, check=True, capture_output=True, text=True)
    assert "HANDOFF_ROUTER_DISPATCHER" not in binary.read_text()
    assert not real.exists()


@pytest.mark.asyncio
async def test_request_that_waited_for_lock_cannot_restart_after_shutdown(tmp_path: Path):
    registry = ModelRegistry.from_file(write_registry(tmp_path))
    manager = RouterManager(registry)
    await manager.startup()
    await manager.switch_lock.acquire()
    waiting = asyncio.create_task(manager.acquire("qwen3.8-ninfer"))
    await asyncio.sleep(0)
    manager._shutdown = True
    manager.switch_lock.release()
    with pytest.raises(ModelSwitchError, match="shutting down"):
        await waiting
    assert manager.active_backend is None
    assert not any(adapter.is_running() for adapter in manager.adapters.values())
    await manager.shutdown()
