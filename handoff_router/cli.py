"""Command-line entry point for Handoff Router.

The facade is usually reached through a launcher shim (see
`deploy/dispatcher.sh`), but it also runs standalone:

    handoff-router --config ./config/router-backends.example.yaml --port 8081
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import uvicorn

from . import __version__
from .app import create_app
from .registry import ModelRegistry

CONFIG_ENV = "HANDOFF_ROUTER_CONFIG"
LEGACY_CONFIG_ENV = "LLAMA_BACKEND_REGISTRY"
DEFAULT_CONFIG_PATHS = (
    os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                 "handoff-router", "router-backends.yaml"),
    "~/.config/handoff-router/router-backends.yaml",
    "config/router-backends.yaml",
    "/etc/handoff-router/router-backends.yaml",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="handoff-router",
        description="One GPU, many engines: a VRAM-gated backend router.",
    )
    result.add_argument(
        "--config",
        "--backend-registry",  # accepted for continuity with earlier setups
        dest="config",
        default=None,
        help="router-backends.yaml (default: $%s or %s)" % (CONFIG_ENV, DEFAULT_CONFIG_PATHS[0]),
    )
    result.add_argument("--host", default="0.0.0.0")
    result.add_argument("--port", type=int, default=8081)
    result.add_argument("--api-key", action="append", default=[])
    api_key_file = os.environ.get("HANDOFF_ROUTER_API_KEY_FILE") or os.environ.get("LLAMA_ARG_API_KEY_FILE")
    result.add_argument("--api-key-file", action="append", default=[api_key_file] if api_key_file else [])
    result.add_argument(
        "--cors-origins",
        default=os.environ.get("HANDOFF_ROUTER_CORS_ORIGINS", os.environ.get("LLAMA_ARG_CORS_ORIGINS", "*")),
    )
    result.add_argument(
        "--cors-methods",
        default=os.environ.get(
            "HANDOFF_ROUTER_CORS_METHODS",
            os.environ.get("LLAMA_ARG_CORS_METHODS", "GET, POST, DELETE, OPTIONS"),
        ),
    )
    result.add_argument(
        "--cors-headers",
        default=os.environ.get(
            "HANDOFF_ROUTER_CORS_HEADERS", os.environ.get("LLAMA_ARG_CORS_HEADERS", "*")
        ),
    )
    result.add_argument("--cors-credentials", dest="cors_credentials", action="store_true")
    result.add_argument("--no-cors-credentials", dest="cors_credentials", action="store_false")
    result.set_defaults(cors_credentials=True)
    result.add_argument("--log-level", "--router-log-level", dest="log_level", default="info")
    result.add_argument(
        "--keyless-model-api",
        dest="keyless_model_api",
        action="store_true",
        default=os.environ.get("HANDOFF_KEYLESS_MODEL_API", "").lower() in ("1", "true", "yes"),
        help="let llama.cpp's web UI call /models/load and /models/unload without an API key",
    )
    result.add_argument(
        "--print-config",
        action="store_true",
        help="resolve and dump the effective model/backend table, then exit",
    )
    result.add_argument("--version", action="version", version=f"handoff-router {__version__}")
    return result


def resolve_config_path(selected: str | None) -> Path:
    if selected:
        return Path(os.path.expandvars(os.path.expanduser(selected))).resolve()
    for candidate in (os.environ.get(CONFIG_ENV), os.environ.get(LEGACY_CONFIG_ENV)):
        if candidate:
            return Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()
    for candidate in DEFAULT_CONFIG_PATHS:
        path = Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()
        if path.is_file():
            return path
    raise SystemExit(
        "no config file found; pass --config /path/to/router-backends.yaml "
        f"(looked in: {', '.join(DEFAULT_CONFIG_PATHS)})"
    )


def load_api_keys(command_line: list[str], files: list[str]) -> set[str]:
    raw_values = list(command_line)
    for env_name in ("HANDOFF_ROUTER_API_KEY", "LLAMA_API_KEY"):
        environment_key = os.environ.get(env_name)
        if environment_key:
            raw_values.append(environment_key)
    keys = {
        key.strip()
        for value in raw_values
        for key in value.split(",")
        if key.strip()
    }
    for name in files:
        for line in Path(name).expanduser().read_text().splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                keys.add(value)
    return keys


def main() -> None:
    args, unknown = parser().parse_known_args()
    if unknown:
        # Launcher shims may forward llama-server style flags; ignore them
        # rather than exiting, so an existing unit keeps working unchanged.
        logging.getLogger("handoff-router.cli").warning("ignoring unsupported arguments: %s", unknown)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = resolve_config_path(args.config)
    try:
        registry = ModelRegistry.from_file(config_path)
    except (OSError, ValueError) as exc:
        # A bad config should read like a config error, not a stack trace.
        raise SystemExit(f"handoff-router: {config_path}: {exc}") from exc

    if args.print_config:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "state_dir": str(registry.settings.state_dir),
                    "idle_ttl_seconds": registry.settings.idle_ttl_seconds,
                    "min_free_vram_mib": registry.settings.min_free_vram_mib,
                    "watch_config": registry.settings.watch_config,
                    "backends": {
                        name: {
                            "adapter": config.adapter,
                            "upstream": config.upstream,
                            "resident": config.resident,
                            "evict_cost": config.evict_cost,
                            "idle_ttl_seconds": config.idle_ttl_seconds,
                            "command": list(config.command),
                        }
                        for name, config in registry.backends.items()
                    },
                    "models": {
                        model.id: {
                            "backend": model.backend,
                            "upstream_model": model.upstream_model,
                            "owned_by": model.owned_by,
                            "modalities": list(model.input_modalities),
                            "metadata": model.metadata,
                        }
                        for model in registry.models.values()
                    },
                },
                indent=2,
            )
        )
        return

    app = create_app(
        registry,
        api_keys=load_api_keys(args.api_key, args.api_key_file),
        cors_origins=args.cors_origins,
        cors_methods=args.cors_methods,
        cors_headers=args.cors_headers,
        cors_credentials=args.cors_credentials,
        keyless_model_api=args.keyless_model_api,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        workers=1,
    )


if __name__ == "__main__":
    main()
