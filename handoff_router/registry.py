"""Configuration and model resolution for the Handoff Router facade.

Portability contract: no default in this package may point at a developer's home
directory. Defaults are $HOME-relative, environment-overridable, or resolved
relative to the config file, so the package runs unchanged on another machine,
in a container, or under a non-root service user.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class RouterSettings:
    state_dir: Path
    llama_presets: Path | None = None
    llama_models_snapshot: Path | None = None
    llama_backend: str = "llamacpp"
    drain_timeout_seconds: float = 300.0
    shutdown_drain_timeout_seconds: float = 300.0
    stop_timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 300.0
    health_poll_seconds: float = 0.5
    gpu_release_timeout_seconds: float = 90.0
    gpu_poll_seconds: float = 1.0
    min_free_vram_mib: int = 0
    nvidia_smi: str = "nvidia-smi"
    upstream_connect_timeout_seconds: float = 10.0
    upstream_write_timeout_seconds: float = 300.0
    # Idle reclaim. 0 disables it, preserving "keep the big model resident until
    # something else needs the GPU" - the safe default for multi-minute loads.
    idle_ttl_seconds: float = 0.0
    idle_poll_seconds: float = 5.0
    idle_grace_seconds: float = 30.0
    # Re-read the config file when its mtime changes, without touching the
    # active backend.
    watch_config: bool = True
    config_poll_seconds: float = 2.0
    activity_history: int = 200


@dataclass(frozen=True)
class BackendConfig:
    name: str
    adapter: str
    command: tuple[str, ...]
    upstream: str
    health_endpoint: str = "/health"
    environment: dict[str, str] = field(default_factory=dict)
    process_match: tuple[str, ...] = ()
    startup_timeout_seconds: float | None = None
    stop_timeout_seconds: float | None = None
    adopt_existing: bool = True
    # A resident backend is never evicted by a model switch. Intended for small
    # always-useful sidecars (embedders, rerankers) that coexist with a big LLM.
    resident: bool = False
    # Relative pain of reloading this backend. When more than one backend must
    # stop to make room, the cheapest stop first.
    evict_cost: int = 1
    idle_ttl_seconds: float | None = None

    @property
    def port(self) -> int:
        parsed = urlparse(self.upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"backend {self.name!r} upstream must be an http(s) URL: {self.upstream!r}")
        if parsed.port is not None:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80


@dataclass(frozen=True)
class ModelConfig:
    id: str
    backend: str
    upstream_model: str
    owned_by: str
    input_modalities: tuple[str, ...] = ("text",)
    metadata: dict[str, Any] = field(default_factory=dict)
    selector: bool = False


class ModelRegistry:
    """Loads explicit backend selectors and existing llama.cpp preset IDs."""

    def __init__(
        self,
        settings: RouterSettings,
        backends: dict[str, BackendConfig],
        models: dict[str, ModelConfig],
        llama_model_ids: tuple[str, ...],
        llama_snapshot: dict[str, dict[str, Any]],
        *,
        source_path: Path | None = None,
        source_mtime: float = 0.0,
    ) -> None:
        self.source_path = source_path
        self.source_mtime = source_mtime
        self.settings = settings
        self.backends = backends
        self.models = models
        self.llama_model_ids = llama_model_ids
        self.llama_snapshot = llama_snapshot

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRegistry":
        registry_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(registry_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("router config must be a YAML mapping")
        for section in ("settings", "backends", "models"):
            if raw.get(section) is not None and not isinstance(raw[section], dict):
                raise ValueError(f"{section} must be a YAML mapping")
        base = registry_path.parent
        raw_settings = raw.get("settings") or {}

        def resolved_path(value: str | None, default: str | None = None) -> Path | None:
            selected = value if value is not None else default
            if selected is None:
                return None
            expanded = Path(os.path.expandvars(os.path.expanduser(selected)))
            return (base / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()

        def command_part(value: Any) -> str:
            raw = str(value)
            expanded = os.path.expandvars(os.path.expanduser(raw))
            # Bare command names and model IDs are passed through. Explicit
            # relative paths are relative to this YAML file, not systemd's /.
            if raw.startswith(("./", "../")):
                return str((base / expanded).resolve())
            return expanded

        settings = RouterSettings(
            state_dir=resolved_path(raw_settings.get("state_dir"), "~/.local/state/handoff-router"),  # type: ignore[arg-type]
            llama_presets=resolved_path(raw_settings.get("llama_presets")),
            llama_models_snapshot=resolved_path(raw_settings.get("llama_models_snapshot")),
            llama_backend=str(raw_settings.get("llama_backend", "llamacpp")),
            drain_timeout_seconds=float(raw_settings.get("drain_timeout_seconds", 300)),
            shutdown_drain_timeout_seconds=float(raw_settings.get("shutdown_drain_timeout_seconds", 300)),
            stop_timeout_seconds=float(raw_settings.get("stop_timeout_seconds", 30)),
            startup_timeout_seconds=float(raw_settings.get("startup_timeout_seconds", 300)),
            health_poll_seconds=float(raw_settings.get("health_poll_seconds", 0.5)),
            gpu_release_timeout_seconds=float(raw_settings.get("gpu_release_timeout_seconds", 90)),
            gpu_poll_seconds=float(raw_settings.get("gpu_poll_seconds", 1)),
            min_free_vram_mib=int(raw_settings.get("min_free_vram_mib", 0)),
            nvidia_smi=os.path.expandvars(str(raw_settings.get("nvidia_smi", "nvidia-smi"))),
            upstream_connect_timeout_seconds=float(raw_settings.get("upstream_connect_timeout_seconds", 10)),
            upstream_write_timeout_seconds=float(raw_settings.get("upstream_write_timeout_seconds", 300)),
            idle_ttl_seconds=float(raw_settings.get("idle_ttl_seconds", 0)),
            idle_poll_seconds=float(raw_settings.get("idle_poll_seconds", 5)),
            idle_grace_seconds=float(raw_settings.get("idle_grace_seconds", 30)),
            watch_config=bool(raw_settings.get("watch_config", True)),
            config_poll_seconds=float(raw_settings.get("config_poll_seconds", 2)),
            activity_history=int(raw_settings.get("activity_history", 200)),
        )

        backends: dict[str, BackendConfig] = {}
        for name, item in (raw.get("backends") or {}).items():
            if not isinstance(item, dict):
                raise ValueError(f"backend {name!r} must be a YAML mapping")
            if not isinstance(item.get("command", []), list):
                raise ValueError(f"backend {name!r} command must be a list of argv entries")
            if "upstream" not in item:
                raise ValueError(f"backend {name!r} has no upstream URL")
            if not isinstance(item.get("process_match", []), list):
                raise ValueError(f"backend {name!r} process_match must be a list")
            if not isinstance(item.get("environment") or {}, dict):
                raise ValueError(f"backend {name!r} environment must be a mapping")
            command = tuple(command_part(part) for part in item.get("command", []))
            if not command:
                raise ValueError(f"backend {name!r} has no command")
            environment = {
                str(key): os.path.expandvars(os.path.expanduser(str(value)))
                for key, value in (item.get("environment") or {}).items()
            }
            if bool(item.get("adopt_existing", True)) and not item.get("process_match"):
                raise ValueError(
                    f"backend {name!r} enables adoption without process_match; "
                    "provide a unique command-line fingerprint or set adopt_existing: false"
                )
            backends[name] = BackendConfig(
                name=name,
                adapter=str(item.get("adapter", name)).lower(),
                command=command,
                upstream=str(item["upstream"]).rstrip("/"),
                health_endpoint=str(item.get("health_endpoint", "/health")),
                environment=environment,
                process_match=tuple(command_part(value) for value in item.get("process_match", [])),
                startup_timeout_seconds=(
                    float(item["startup_timeout_seconds"])
                    if "startup_timeout_seconds" in item
                    else None
                ),
                stop_timeout_seconds=(
                    float(item["stop_timeout_seconds"])
                    if "stop_timeout_seconds" in item
                    else None
                ),
                adopt_existing=bool(item.get("adopt_existing", True)),
                resident=bool(item.get("resident", False)),
                evict_cost=int(item.get("evict_cost", 1)),
                idle_ttl_seconds=(
                    float(item["idle_ttl_seconds"]) if "idle_ttl_seconds" in item else None
                ),
            )

        # Discovery inspects listening *ports* (not addresses), and a backend
        # process group must have exactly one owner. Two names on one port can
        # otherwise adopt the same PID and later stop each other's live leases.
        ports: dict[int, str] = {}
        for name, backend in backends.items():
            port = backend.port
            if port in ports:
                raise ValueError(
                    f"backends {ports[port]!r} and {name!r} share port {port}; "
                    "each backend needs a dedicated port"
                )
            ports[port] = name

        llama_backend = str(raw_settings.get("llama_backend", "llamacpp"))
        # llama.cpp's native multi-model router is the reference backend, but it
        # is optional: a deployment that only runs Strata/vLLM/NInfer must load.
        has_llama_router = llama_backend in backends
        if settings.llama_presets is not None and not settings.llama_presets.is_file():
            raise FileNotFoundError(
                f"llama.cpp preset registry not found: {settings.llama_presets}. "
                "Create it (llama.cpp --models-preset format), or delete "
                "settings.llama_presets and the llama_backend entry if you do not "
                "run llama.cpp at all."
            )

        preset_model_ids = (
            cls._read_preset_ids(settings.llama_presets)
            if (has_llama_router and settings.llama_presets is not None)
            else ()
        )
        snapshot = cls._read_snapshot(settings.llama_models_snapshot) if has_llama_router else {}
        # The live llama.cpp router may also advertise cached/Hugging Face models
        # that are not named in the preset file. Retain those cache entries while
        # dropping snapshot entries explicitly marked as presets that have since
        # been renamed or removed from the live preset file.
        snapshot_model_ids = (
            model_id
            for model_id, item in snapshot.items()
            if model_id in preset_model_ids or item.get("source") != "preset"
        )
        llama_model_ids = tuple(dict.fromkeys((*preset_model_ids, *snapshot_model_ids)))
        models: dict[str, ModelConfig] = {
            model_id: ModelConfig(
                id=model_id,
                backend=llama_backend,
                upstream_model=model_id,
                owned_by="llamacpp",
                input_modalities=cls._snapshot_modalities(snapshot.get(model_id)),
                metadata={"source": "preset"},
            )
            for model_id in (llama_model_ids if has_llama_router else ())
        }

        for model_id, item in (raw.get("models") or {}).items():
            if not isinstance(item, dict) or "backend" not in item:
                raise ValueError(f"model {model_id!r} must define a backend")
            backend = str(item["backend"])
            if backend not in backends:
                raise ValueError(f"model {model_id!r} references unknown backend {backend!r}")
            adapter_name = backends[backend].adapter
            models[model_id] = ModelConfig(
                id=model_id,
                backend=backend,
                upstream_model=str(item.get("upstream_model", model_id)),
                owned_by=str(item.get("owned_by", adapter_name)),
                input_modalities=tuple(item.get("input_modalities", ["text"])),
                metadata=dict(item.get("metadata") or {}),
                selector=bool(item.get("selector", True)),
            )

        return cls(
            settings,
            backends,
            models,
            llama_model_ids if has_llama_router else (),
            snapshot,
            source_path=registry_path,
            source_mtime=registry_path.stat().st_mtime,
        )

    def resolve(self, model_id: str) -> ModelConfig | None:
        return self.models.get(model_id)

    def models_for_backend(self, backend: str) -> list[ModelConfig]:
        return [model for model in self.models.values() if model.backend == backend]

    @staticmethod
    def _read_preset_ids(path: Path) -> tuple[str, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"llama.cpp preset registry not found: {path}")
        names: list[str] = []
        section = re.compile(r"^\s*\[([^]]+)]\s*$")
        for line in path.read_text(errors="replace").splitlines():
            match = section.match(line)
            if match and match.group(1) != "*":
                names.append(match.group(1))
        return tuple(dict.fromkeys(names))

    @staticmethod
    def _read_snapshot(path: Path | None) -> dict[str, dict[str, Any]]:
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            str(item["id"]): item
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }

    @staticmethod
    def _snapshot_modalities(item: dict[str, Any] | None) -> tuple[str, ...]:
        if not item:
            return ("text",)
        architecture = item.get("architecture") or {}
        values = architecture.get("input_modalities") or ["text"]
        return tuple(str(value) for value in values)
