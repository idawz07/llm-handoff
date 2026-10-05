"""Regression tests for the release audit. All engines here are fake CPU servers."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_features import free_port, make_config  # noqa: E402

from handoff_router.app import _model_listing, create_app  # noqa: E402
from handoff_router.manager import RouterManager  # noqa: E402
from handoff_router.registry import ModelRegistry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_legacy_inference_and_props_require_authentication(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry)
    await manager.startup()
    app = create_app(registry, manager=manager, manage_lifespan=False, api_keys={"secret"})
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            for path in ("/v1/chat/completions", "/completion", "/chat"):
                response = await client.post(path, json={"model": "big-llama", "prompt": "hi"})
                assert response.status_code == 401, path
            assert (await client.get("/props?model=big-vllm")).status_code == 401
            # Public GET paths and the optional UI shell must not create a
            # method- or query-based hole in the auth middleware.
            for path in ("/", "/models", "/health", "/v1/models"):
                assert (await client.post(path, json={"model": "big-vllm"})).status_code == 401
            assert (await client.get("/?model=big-vllm")).status_code == 401
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/v1/models")).status_code == 200
            assert not config.active()
            authorized = await client.get("/props?model=big-vllm", headers={"Authorization": "Bearer secret"})
            assert authorized.status_code == 200
            assert config.active() == {"vllm"}
    finally:
        if app.state.client is not None:
            await app.state.client.aclose()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_keyless_model_api_does_not_make_inference_keyless(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry)
    await manager.startup()
    app = create_app(registry, manager=manager, manage_lifespan=False,
                     api_keys={"secret"}, keyless_model_api=True)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            assert (await client.post("/models/load", json={"model": "big-vllm"})).status_code == 200
            assert (await client.post("/chat", json={"model": "big-vllm"})).status_code == 401
    finally:
        if app.state.client is not None:
            await app.state.client.aclose()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_resident_remains_running_in_both_switch_directions_and_can_unload(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["ninfer"]["resident"] = True
    config.write(doc)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        for model in ("sidecar", "big-llama", "sidecar", "big-llama"):
            lease = await manager.acquire(model)
            await lease.release()
            if model == "big-llama":
                assert config.active() == {"ninfer", "llamacpp"}
        assert config.active() == {"ninfer", "llamacpp"}
        assert config.starts("llamacpp") == 1
        listed = await _model_listing(manager.registry, manager)
        assert next(item["status"]["value"] for item in listed if item["id"] == "sidecar") == "loaded"
        assert await manager.deactivate("sidecar") is True
        assert config.active() == {"llamacpp"}
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_zero_ttl_is_disabled_even_if_another_backend_has_ttl(tmp_path: Path):
    config = make_config(tmp_path, idle_ttl_seconds=0, idle_grace_seconds=0, watch_config=False)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["ninfer"]["idle_ttl_seconds"] = 900
    config.write(doc)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        lease = await manager.acquire("big-llama")
        manager.last_used["llamacpp"] = time.time() - 3600
        assert manager._idle_candidates() == []
        await lease.release()
        assert manager.last_used["llamacpp"] > time.time() - 2
        await manager._reap_once()
        assert config.active() == {"llamacpp"}
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_request_waiting_for_switch_lock_uses_reloaded_model_mapping(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        await manager.switch_lock.acquire()
        waiting = asyncio.create_task(manager.acquire("sidecar"))
        await asyncio.sleep(0)  # acquire has resolved the old backend, then blocks
        doc = yaml.safe_load(config.path.read_text())
        doc["models"]["sidecar"]["backend"] = "vllm"
        doc["models"]["sidecar"]["upstream_model"] = "vllm-real"
        config.write(doc)
        manager.registry = ModelRegistry.from_file(config.path)  # simulate reload under the lock
        manager.switch_lock.release()
        lease = await waiting
        assert lease.model.backend == "vllm"
        assert config.active() == {"vllm"}
        await lease.release()
    finally:
        if manager.switch_lock.locked():
            manager.switch_lock.release()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_reload_preserves_leased_url_then_replaces_adapter_class(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        lease = await manager.acquire("big-llama")
        original_url = lease.adapter.upstream_url
        doc = yaml.safe_load(config.path.read_text())
        port = free_port()
        doc["backends"]["llamacpp"]["upstream"] = f"http://127.0.0.1:{port}"
        args = doc["backends"]["llamacpp"]["command"]
        args[args.index("--port") + 1] = str(port)
        doc["backends"]["llamacpp"]["adapter"] = "vllm"
        config.write(doc)
        change = await manager.reload()
        assert change["restart_pending"] == ["llamacpp"]
        assert lease.adapter.upstream_url == original_url
        assert type(lease.adapter).__name__ == "LlamaCppAdapter"
        assert lease.adapter.is_running()
        await lease.release()
        new_lease = await manager.acquire("big-llama")
        assert new_lease.adapter.upstream_url != original_url
        assert type(new_lease.adapter).__name__ == "VllmAdapter"
        assert config.starts("llamacpp") == 2
        await new_lease.release()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_reload_never_kills_an_unselected_resident_with_a_lease(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["ninfer"]["resident"] = True
    config.write(doc)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        sidecar = await manager.acquire("sidecar")
        primary = await manager.acquire("big-llama")
        await primary.release()
        doc = yaml.safe_load(config.path.read_text())
        del doc["backends"]["ninfer"]
        del doc["models"]["sidecar"]
        config.write(doc)
        result = await manager.reload()
        assert "ninfer" in result["retired"]
        assert sidecar.adapter.is_running()
        assert manager.active_request_count == 1
        await sidecar.release()
        await manager._reap_once()
        assert config.active() == {"llamacpp"}
        assert "ninfer" not in manager.adapters
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_removing_a_crashed_selected_backend_clears_stale_pointer(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        lease = await manager.acquire("big-llama")
        await lease.release()
        await manager.adapters["llamacpp"].stop()  # simulate a child crashing outside manager
        doc = yaml.safe_load(config.path.read_text())
        del doc["backends"]["llamacpp"]
        del doc["models"]["big-llama"]
        config.write(doc)
        await manager.reload()
        status = await manager.status()
        assert status["active_backend"] is None
        assert status["active_model"] is None
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_selected_retired_resident_can_finish_while_new_primary_starts(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["ninfer"]["resident"] = True
    config.write(doc)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    await manager.startup()
    try:
        sidecar = await manager.acquire("sidecar")
        doc = yaml.safe_load(config.path.read_text())
        del doc["backends"]["ninfer"]
        del doc["models"]["sidecar"]
        config.write(doc)
        await manager.reload()
        primary = await asyncio.wait_for(manager.acquire("big-llama"), timeout=3)
        assert sidecar.adapter.is_running()
        assert config.active() == {"ninfer", "llamacpp"}
        await primary.release()
        await sidecar.release()
        await manager._reap_once()
        assert config.active() == {"llamacpp"}
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_restart_adopts_one_primary_and_one_resident(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["ninfer"]["resident"] = True
    config.write(doc)
    first = RouterManager(ModelRegistry.from_file(config.path))
    await first.startup()
    second = None
    try:
        for model in ("sidecar", "big-llama"):
            lease = await first.acquire(model)
            await lease.release()
        for task in first._supervisor_tasks:
            task.cancel()
        await asyncio.gather(*first._supervisor_tasks, return_exceptions=True)
        first._shutdown = True  # simulated facade crash: leave children running
        second = RouterManager(ModelRegistry.from_file(config.path))
        await second.startup()
        assert config.active() == {"llamacpp", "ninfer"}
        assert all(second.adapters[name].runtime.adopted for name in ("llamacpp", "ninfer"))
    finally:
        if second is not None:
            await second.shutdown()
        await first.shutdown()


def test_print_config_rejects_bad_yaml_and_unknown_adapter_cleanly(tmp_path: Path):
    config = make_config(tmp_path)
    for document, expected in (
        ("- this is not a mapping\n", "router config must be a YAML mapping"),
        ("backends:\n  x: {command: [python]}\n", "has no upstream URL"),
    ):
        config.path.write_text(document)
        result = subprocess.run(
            [sys.executable, "-m", "handoff_router.cli", "--config", str(config.path), "--print-config"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode != 0 and expected in result.stderr
        assert "Traceback" not in result.stderr
    config = make_config(tmp_path)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["vllm"]["adapter"] = "unsupported"
    config.write(doc)
    result = subprocess.run(
        [sys.executable, "-m", "handoff_router.cli", "--config", str(config.path), "--print-config"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "unknown backend adapter" in result.stderr
    assert "Traceback" not in result.stderr


def test_remote_bind_without_a_key_fails_before_listening(tmp_path: Path):
    config = make_config(tmp_path)
    environment = {**os.environ, "HANDOFF_ROUTER_API_KEY": "", "LLAMA_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-m", "handoff_router.cli", "--config", str(config.path),
         "--host", "0.0.0.0", "--port", str(free_port())],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode != 0
    assert "refusing non-loopback bind without an API key" in result.stderr


def test_adoption_requires_a_process_fingerprint(tmp_path: Path):
    config = make_config(tmp_path)
    doc = yaml.safe_load(config.path.read_text())
    del doc["backends"]["vllm"]["process_match"]
    config.write(doc)
    with pytest.raises(ValueError, match="without process_match"):
        ModelRegistry.from_file(config.path)
    doc["backends"]["vllm"]["adopt_existing"] = False
    config.write(doc)
    manager = RouterManager(ModelRegistry.from_file(config.path))
    assert manager.adapters["vllm"].process_matches(os.getpid()) is False


def test_explicit_relative_command_paths_are_config_relative(tmp_path: Path):
    config = make_config(tmp_path)
    doc = yaml.safe_load(config.path.read_text())
    doc["backends"]["llamacpp"]["command"] += ["--models-preset", "./router-presets.ini"]
    doc["backends"]["llamacpp"]["process_match"] += ["./router-presets.ini"]
    config.write(doc)
    backend = ModelRegistry.from_file(config.path).backends["llamacpp"]
    assert backend.command[-1] == str(tmp_path / "router-presets.ini")
    assert backend.process_match[-1] == str(tmp_path / "router-presets.ini")


def test_upgrading_a_legacy_dispatcher_keeps_real_binary_and_persists_paths(tmp_path: Path):
    root = tmp_path / "llama.cpp"
    binary = root / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\n# LLAMA_MULTI_BACKEND_DISPATCHER\nexec "$LLAMA_SERVER_REAL" "$@"\n')
    binary.chmod(0o755)
    real = binary.with_name("llama-server.real")
    real.write_text('#!/bin/sh\nprintf "real:%s\\n" "$*"\n')
    real.chmod(0o755)
    config = tmp_path / "router-backends.yaml"
    config.write_text("settings: {}\n")
    preset = tmp_path / "router-presets.ini"
    preset.write_text("[*]\n")
    runner = tmp_path / "runner.sh"
    runner.write_text('#!/bin/sh\nprintf "facade:%s\\n" "$*"\n')
    runner.chmod(0o755)
    install_env = {
        **os.environ,
        "HANDOFF_LLAMA_ROOT": str(root),
        "HANDOFF_ROUTER_CONFIG": str(config),
        "HANDOFF_ROUTER_PRESET": str(preset),
        "HANDOFF_ROUTER_RUNNER": str(runner),
    }
    subprocess.run([str(ROOT / "deploy" / "install-dispatcher.sh")], env=install_env, check=True)
    assert real.read_text().startswith("#!/bin/sh\nprintf")  # the original was NOT overwritten
    assert list(binary.parent.glob("llama-server.legacy.*"))
    runtime_env = {"HOME": str(tmp_path / "home"), "PATH": os.environ["PATH"]}
    routed = subprocess.run(
        [str(binary), "--models-preset", str(preset), "--port", "8081"],
        env=runtime_env, check=True, capture_output=True, text=True,
    )
    assert routed.stdout.startswith("facade:")  # install-time values survived
    delegated = subprocess.run(
        [str(binary), "--version"], env=runtime_env, check=True, capture_output=True, text=True,
    )
    assert delegated.stdout.strip() == "real:--version"
    config.unlink()
    missing = subprocess.run(
        [str(binary), "--models-preset", str(preset), "--port", "8081"],
        env=runtime_env, capture_output=True, text=True,
    )
    assert missing.returncode == 78 and "config missing" in missing.stderr
    subprocess.run([str(ROOT / "deploy" / "uninstall-dispatcher.sh")], env=install_env, check=True)
    assert not real.exists()
    assert binary.read_text().startswith("#!/bin/sh\nprintf")


@pytest.mark.asyncio
async def test_frozen_native_model_unload_does_not_restart_router(tmp_path: Path):
    config = make_config(tmp_path, watch_config=False)
    registry = ModelRegistry.from_file(config.path)
    manager = RouterManager(registry)
    await manager.startup()
    app = create_app(registry, manager=manager, manage_lifespan=False, api_keys={"secret"})
    try:
        lease = await manager.acquire("big-llama")
        await lease.release()
        manager.set_frozen(True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client:
            headers = {"Authorization": "Bearer secret"}
            response = await client.post("/models/unload", json={"model": "big-llama"}, headers=headers)
            assert response.status_code == 200
            assert manager.active_model is None
            assert manager.active_backend == "llamacpp"
            assert config.active() == {"llamacpp"}
            assert (await client.get("/props?model=big-vllm", headers=headers)).status_code == 503
    finally:
        if app.state.client is not None:
            await app.state.client.aclose()
        await manager.shutdown()
