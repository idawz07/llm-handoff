"""OpenAI-compatible HTTP facade with incremental SSE proxying."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import parse_qsl, urlencode

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .manager import ModelLease, ModelSwitchError, RouterManager
from .registry import ModelConfig, ModelRegistry

LOG = logging.getLogger("handoff-router.http")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
PUBLIC_PATHS = {
    "/health", "/v1/health", "/models", "/v1/models",
    # Inference, /props?model= (which can cold-load a backend), and every
    # control endpoint require a key when one is configured. Static UI assets
    # are exempted separately by resolve_web_ui_asset().
    # keyless_model_api=True additionally exempts only /models/load|unload.
}

# Paths that the router proxies to the active backend. Kept in sync with the
# static-file guard below so an API call is never answered with a UI asset.
WEB_UI_API_PREFIXES = (
    "/v1/", "/health", "/slots", "/props", "/completion", "/chat",
    "/models", "/router/", "/metrics", "/version", "/tokenize",
    "/detokenize", "/save_session", "/load_session", "/logs", "/activity",
)

# Upstream (llama.cpp) endpoints that are valid without an explicit model:
# they describe the currently loaded backend. llama.cpp's own server answers
# these model-lessly, so fall back to the router's active model. Chat
# endpoints are included as well: llama.cpp's single-server answers a
# model-less chat from the loaded model, and the web UI occasionally posts
# without a model (fresh conversation before its selection restores), so a
# hard 400 there is the "model name is missing from the request" the user
# sees. Serving the active model matches the upstream behavior.
MODELLESS_PATHS = frozenset({
    "/props", "/slots", "/metrics", "/version", "/tokenize", "/detokenize",
    "/v1/chat/completions", "/completion",
})

# llama.cpp ships its web UI next to the server binary. Point HANDOFF_WEB_UI_DIR
# (or LLAMA_WEB_UI_DIR) at a built copy to get the model picker for free; leave
# it unset and the facade is API-only.
DEFAULT_WEB_UI_DIR = ""


def web_ui_dir() -> Path | None:
    """Directory holding the built llama.cpp web UI, or None if not installed."""
    selected = os.environ.get("HANDOFF_WEB_UI_DIR") or os.environ.get("LLAMA_WEB_UI_DIR") or DEFAULT_WEB_UI_DIR
    if not selected:
        return None
    resolved = Path(selected).expanduser().resolve()
    return resolved if resolved.is_dir() else None


def resolve_web_ui_asset(request: Request) -> Path | None:
    """Resolve a request to a real web UI file, or None for API/SPA requests.

    The returned path is guaranteed to live inside the web UI directory, so this
    is safe to use both as a static-file lookup and as an auth exemption.
    """
    if request.method not in ("GET", "HEAD"):
        return None
    base = web_ui_dir()
    if base is None:
        return None
    request_path = request.url.path
    if any(request_path.startswith(prefix) for prefix in WEB_UI_API_PREFIXES):
        return None
    candidate = (base / request_path.lstrip("/")).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate


def is_web_ui_navigation(request: Request) -> bool:
    """Return True for browser navigations, without swallowing unknown API GETs."""
    if request.method not in ("GET", "HEAD") or web_ui_dir() is None:
        return False
    if any(request.url.path.startswith(prefix) for prefix in WEB_UI_API_PREFIXES):
        return False
    return request.url.path == "/" or "text/html" in request.headers.get("accept", "").lower()


def openai_error(message: str, status_code: int, error_type: str, **extra: Any) -> JSONResponse:
    error: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "code": status_code,
    }
    error.update(extra)
    return JSONResponse({"error": error}, status_code=status_code)


def create_app(
    registry: ModelRegistry,
    *,
    manager: RouterManager | None = None,
    api_keys: set[str] | None = None,
    manage_lifespan: bool = True,
    cors_origins: str = "*",
    cors_methods: str = "GET, POST, DELETE, OPTIONS",
    cors_headers: str = "*",
    cors_credentials: bool = True,
    keyless_model_api: bool = False,
) -> FastAPI:
    router_manager = manager or RouterManager(registry)
    accepted_keys = api_keys or set()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if manage_lifespan:
            await router_manager.startup()
        timeout = httpx.Timeout(
            connect=registry.settings.upstream_connect_timeout_seconds,
            read=None,
            write=registry.settings.upstream_write_timeout_seconds,
            pool=registry.settings.upstream_connect_timeout_seconds,
        )
        # cpp-httplib and HTTPX both default to a five-second keep-alive
        # timeout. Reusing a socket at that boundary races the backend closing
        # it and intermittently raises RemoteProtocolError before a request is
        # dispatched. Local upstreams are cheap to reconnect to, so do not pool
        # idle sockets.
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=0)
        app.state.client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            trust_env=False,
        )
        try:
            yield
        finally:
            await app.state.client.aclose()
            if manage_lifespan:
                await router_manager.shutdown()

    app = FastAPI(title="Handoff Router - multi-engine backend router", lifespan=lifespan)
    origin_values = [value.strip() for value in cors_origins.split(",") if value.strip()]
    origin_regex = None
    if cors_origins == "localhost":
        origin_values = ["localhost"]
        origin_regex = r"https?://(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origin_values,
        allow_origin_regex=origin_regex,
        allow_credentials=cors_credentials,
        allow_methods=[value.strip() for value in cors_methods.split(",") if value.strip()],
        allow_headers=[value.strip() for value in cors_headers.split(",") if value.strip()],
    )
    app.state.registry = registry
    app.state.manager = router_manager
    app.state.client = None
    @app.middleware("http")
    async def preserve_api_key(request: Request, call_next):
        if (
            accepted_keys
            and request.method != "OPTIONS"
            and not (request.method in ("GET", "HEAD") and request.url.path in PUBLIC_PATHS)
            and not (keyless_model_api and request.method == "POST"
                     and request.url.path in {"/models/load", "/models/unload"})
            # The SPA shell at "/" is public, but its hashed JS/CSS/image assets
            # are separate paths; serving them unauthenticated is what lets the
            # chat UI boot at all (the key lives in localStorage, and the UI
            # attaches it to API calls itself).
            and resolve_web_ui_asset(request) is None
            and not is_web_ui_navigation(request)
        ):
            candidate = request.headers.get("authorization", "")
            if candidate.startswith("Bearer "):
                candidate = candidate[len("Bearer "):]
            if not candidate:
                candidate = request.headers.get("x-api-key", "")
            if not any(secrets.compare_digest(candidate, key) for key in accepted_keys):
                return openai_error("Invalid API Key", 401, "authentication_error")
        return await call_next(request)

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        status = await router_manager.status()
        return {
            "status": "ok",
            "active_model": status["active_model"],
            "active_backend": status["active_backend"],
            "backend_healthy": status["healthy"],
        }

    @app.get("/router/status")
    async def router_status() -> dict[str, Any]:
        return await router_manager.status()

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": await _model_listing(router_manager.registry, router_manager)}

    @app.get("/models/sse")
    async def model_events(request: Request) -> StreamingResponse:
        # llama.cpp's WebUI waits for this feed after POST /models/load. The
        # facade owns the cross-backend lifecycle, so publish its status rather
        # than proxying the feed to whichever backend happens to be active.
        return StreamingResponse(
            _model_status_stream(request, router_manager),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            media_type="text/event-stream",
        )

    @app.get("/props")
    async def props(request: Request) -> Response:
        requested_id = request.query_params.get("model")
        requested = router_manager.registry.resolve(requested_id) if requested_id else None
        if requested_id and requested is None:
            return openai_error(f"model {requested_id!r} not found", 404, "invalid_request_error")

        # The WebUI probes model metadata with autoload=false. Such a probe must
        # never switch a 50+ GiB backend; only an explicit /models/load or an
        # inference request may do that.
        autoload = request.query_params.get("autoload", "true").lower() != "false"
        if requested is not None and not autoload:
            if router_manager.switching_to == requested.id:
                return openai_error(
                    f"model {requested.id!r} is loading", 503, "model_loading"
                )
            if not _is_model_active(router_manager.registry, router_manager, requested.id):
                return openai_error(
                    f"model {requested.id!r} is not loaded", 400, "invalid_request_error"
                )

        # A router with no backend loaded still has useful control-plane props.
        # Returning these lets the WebUI boot and present its model picker.
        if requested is None and router_manager.active_model is None:
            return JSONResponse(_empty_router_props())

        response = await _proxy_request(request, router_manager.registry, router_manager)
        if response.status_code in (404, 405):
            # Non-llama.cpp backends (NInfer, vLLM) do not implement /props, so
            # the proxy 404s. Synthesize router-level props from the registry
            # instead of passing the 404 through; the WebUI would otherwise
            # show "Server unavailable" for a perfectly healthy backend.
            target = requested or (
                router_manager.registry.resolve(router_manager.active_model)
                if router_manager.active_model
                else None
            )
            return JSONResponse(_synthesized_router_props(target))
        return response

    @app.options("/{path:path}")
    async def cors_preflight(path: str) -> Response:
        del path
        return Response(status_code=200)

    @app.post("/models/unload")
    async def unload_model(request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return openai_error("request body must be JSON", 400, "invalid_request_error")
        model_id = body.get("model") if isinstance(body, dict) else None
        model = router_manager.registry.resolve(model_id) if isinstance(model_id, str) else None
        if model is None:
            return openai_error(f"model {model_id!r} not found", 404, "invalid_request_error")
        if router_manager.adapters[model.backend].config.adapter == "llamacpp":
            if not router_manager.adapters[model.backend].is_running():
                return JSONResponse({"success": True, "stopped": False})
            response = await _proxy_request(
                request, router_manager.registry, router_manager,
                body_override=body, allow_frozen_unload=True,
            )
            if 200 <= response.status_code < 300:
                await router_manager.note_model_unloaded(model.id)
            return response
        try:
            stopped = await router_manager.deactivate(model_id)
        except ModelSwitchError as exc:
            return _switch_error(exc)
        return JSONResponse({"success": True, "stopped": stopped})

    @app.post("/models/load")
    async def load_model(request: Request) -> Response:
        # llama.cpp's UI sends this when the user picks a model. Natively
        # switch the router to the owning backend and block until the model
        # is loaded, instead of proxying the call to whatever backend happens
        # to be active (a single-model backend 404s it, a multi-model
        # backend would hot-load a second model into VRAM).
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        model_id = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model_id, str) or not model_id:
            model_id = request.query_params.get("model")
        if not isinstance(model_id, str) or not model_id:
            return openai_error("model name is missing from the request", 400, "invalid_request_error")
        if router_manager.registry.resolve(model_id) is None:
            return openai_error(f"model {model_id!r} not found", 404, "invalid_request_error")
        try:
            lease = await router_manager.acquire(model_id)
        except ModelSwitchError as exc:
            return _switch_error(exc)
        await lease.release()
        return JSONResponse({"success": True, "model": model_id})

    @app.post("/router/freeze")
    async def freeze_router() -> JSONResponse:
        # D-062 window control: while frozen, every new model load / proxied
        # inference request fails 503 (ModelSwitchError -> _switch_error).
        # deactivate() (/models/unload) is deliberately NOT frozen.
        router_manager.set_frozen(True)
        return JSONResponse({"frozen": True})

    @app.post("/router/thaw")
    async def thaw_router() -> JSONResponse:
        router_manager.set_frozen(False)
        return JSONResponse({"frozen": False})

    @app.post("/router/reload")
    async def reload_router_config() -> JSONResponse:
        """Apply router-backends.yaml without restarting the process or the model.

        Reload never stops a live process. Changed adapters apply on their next
        request after leases drain; removed backends are retired when safe.
        """
        try:
            result = await router_manager.reload()
            app.state.registry = router_manager.registry
        except ModelSwitchError as exc:
            return _switch_error(exc)
        return JSONResponse(result)

    @app.get("/router/activity")
    async def router_activity(request: Request) -> JSONResponse:
        limit = _bounded_int(request.query_params.get("limit"), default=50, maximum=500)
        since = _bounded_float(request.query_params.get("since"), default=0.0)
        return JSONResponse({"events": router_manager.activity.recent(limit, since=since)})

    @app.get("/activity/stream")
    async def activity_stream(request: Request) -> StreamingResponse:
        return StreamingResponse(
            router_manager.activity.stream(disconnected=request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/logs/{backend_name}")
    async def backend_log(backend_name: str, request: Request) -> JSONResponse:
        adapter = router_manager.adapters.get(backend_name)
        if adapter is None:
            return openai_error(f"backend {backend_name!r} not found", 404, "invalid_request_error")
        lines = _bounded_int(request.query_params.get("lines"), default=200, maximum=5000)
        return JSONResponse(
            {
                "backend": backend_name,
                "log": str(adapter.log_path),
                "running": adapter.is_running(),
                "lines": adapter.tail_log(max_lines=lines, max_bytes=512_000).splitlines(),
            }
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
    )
    async def proxy(request: Request, path: str) -> Response:
        del path
        # Serve web UI static files for GET requests to non-API paths
        if asset := resolve_web_ui_asset(request):
            return FileResponse(asset)
        if is_web_ui_navigation(request):
            # Serve index.html for root and client-side SPA routes. Requiring a
            # browser HTML Accept header keeps unknown GET APIs proxyable.
            return FileResponse(web_ui_dir() / "index.html")  # type: ignore[operator]
        return await _proxy_request(request, router_manager.registry, router_manager)

    return app


async def _model_listing(registry: ModelRegistry, manager: RouterManager) -> list[dict[str, Any]]:
    active_spec = registry.resolve(manager.active_model) if manager.active_model else None
    # A watcher may swap the registry/adapters while fetch_models() awaits I/O.
    # A snapshot prevents dictionary-size errors and stale-name KeyErrors.
    adapters = dict(manager.adapters)
    upstream_by_backend: dict[str, dict[str, dict[str, Any]]] = {}
    healthy: set[str] = set()
    for name, adapter in adapters.items():
        if not adapter.is_running() or (name != manager.active_backend and not adapter.config.resident):
            continue
        if await adapter.is_healthy():
            healthy.add(name)
            upstream_by_backend[name] = {
                str(item["id"]): item
                for item in await adapter.fetch_models() if item.get("id")
            }

    entries: list[dict[str, Any]] = []
    created = int(time.time())
    for model in registry.models.values():
        source = upstream_by_backend.get(model.backend, {}).get(model.upstream_model)
        resident_id = manager.last_model_by_backend.get(model.backend)
        resident_model = registry.resolve(resident_id) if resident_id else None
        active = bool(
            model.backend in healthy
            and (
                (model.backend == manager.active_backend and active_spec is not None
                 and active_spec.backend == model.backend
                 and (manager.active_model == model.id or active_spec.upstream_model == model.upstream_model))
                or (model.backend != manager.active_backend
                    and model.backend in adapters and adapters[model.backend].config.resident
                    and resident_model is not None
                    and (model.id == resident_model.id or model.upstream_model == resident_model.upstream_model)
                    and (source is None or (source.get("status") or {}).get("value", "loaded") == "loaded"))
            )
        )
        if source and not model.selector:
            entry = dict(source)
            entry["id"] = model.id
        else:
            snapshot = registry.llama_snapshot.get(model.id)
            entry = dict(snapshot) if snapshot and not model.selector else {}
            entry.update(
                {
                    "id": model.id,
                    "object": "model",
                    "owned_by": model.owned_by,
                    "created": created,
                    "architecture": {
                        "input_modalities": list(model.input_modalities),
                        "output_modalities": ["text"],
                    },
                }
            )
        status = dict(entry.get("status") or {})
        status.update(
            {
                "value": "loaded" if active else "unloaded",
                "backend": model.backend,
                "upstream_model": model.upstream_model,
            }
        )
        entry["status"] = status
        if model.selector:
            entry["selector"] = True
        entry.update({key: value for key, value in model.metadata.items() if key not in entry})
        entries.append(entry)
    return entries


def _bounded_int(raw: str | None, *, default: int, maximum: int) -> int:
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        return default
    return max(1, min(value, maximum))


def _bounded_float(raw: str | None, *, default: float) -> float:
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return max(0.0, value)


def _is_model_active(registry: ModelRegistry, manager: RouterManager, model_id: str) -> bool:
    """Whether a public model ID resolves to the currently running model."""
    model = registry.resolve(model_id)
    active = registry.resolve(manager.active_model) if manager.active_model else None
    if model is None:
        return False
    adapter = manager.adapters.get(model.backend)
    if adapter is None or not adapter.is_running():
        return False
    if adapter.config.resident and model.backend != manager.active_backend:
        # The last selected model of a resident process remains loaded when a
        # different backend takes the primary pointer; other selectors do not.
        resident_id = manager.last_model_by_backend.get(model.backend)
        resident_model = registry.resolve(resident_id) if resident_id else None
        return bool(resident_model and (resident_model.id == model.id
                    or resident_model.upstream_model == model.upstream_model))
    return bool(
        active and manager.active_backend == model.backend
        and (manager.active_model == model.id or active.upstream_model == model.upstream_model)
    )


def _empty_router_props() -> dict[str, Any]:
    """Minimal llama.cpp router props used while no inference backend is loaded."""
    return {
        "role": "router",
        "max_instances": 1,
        "models_autoload": True,
        "model_alias": "llama-router",
        "model_path": "none",
        "total_slots": 0,
        "default_generation_settings": {"params": {}, "n_ctx": 0},
        "modalities": {"vision": False, "video": False, "audio": False},
        "ui_settings": {},
        "build_info": "handoff-router",
    }


def _synthesized_router_props(model: ModelConfig | None) -> dict[str, Any]:
    """Props synthesized for backends that do not implement /props (NInfer, vLLM).

    Mirrors the dummy router response llama.cpp's own multi-model router returns
    (tools/server/server-models.cpp, get_router_props), enriched with the active
    model's context window and modalities from the registry metadata so the
    WebUI can show a usable context size and input capabilities.
    """
    context_window = 0
    if model is not None:
        raw_window = model.metadata.get("context_window")
        if isinstance(raw_window, int) and raw_window > 0:
            context_window = raw_window
    modalities = {"text": True, "vision": False, "audio": False, "video": False}
    if model is not None:
        for name in model.input_modalities:
            key = "vision" if name == "image" else name
            if key in modalities:
                modalities[key] = True
    return {
        "role": "router",
        "max_instances": 1,
        "models_autoload": True,
        "model_alias": model.id if model is not None else "llama-router",
        "model_path": "none",
        "total_slots": 1,
        "default_generation_settings": {"params": {}, "n_ctx": context_window},
        "modalities": modalities,
        "ui_settings": {},
        "build_info": "handoff-router",
    }


def _model_status_snapshot(
    manager: RouterManager,
) -> dict[str, str]:
    registry = manager.registry
    switching = registry.resolve(manager.switching_to) if manager.switching_to else None
    statuses: dict[str, str] = {}
    for model in registry.models.values():
        if switching is not None and model.id == switching.id:
            statuses[model.id] = "loading"
        elif _is_model_active(registry, manager, model.id):
            statuses[model.id] = "loaded"
        else:
            statuses[model.id] = "unloaded"
    return statuses


async def _model_status_stream(
    request: Request,
    manager: RouterManager,
) -> AsyncIterator[bytes]:
    """Publish facade-owned model state in llama.cpp's /models/sse format."""
    previous: dict[str, str] = {}
    last_heartbeat = asyncio.get_running_loop().time()
    while not await request.is_disconnected():
        current = _model_status_snapshot(manager)
        records: list[bytes] = []
        for model_id, status in current.items():
            if previous.get(model_id) == status:
                continue
            event = {
                "model": model_id,
                "event": "model_status",
                "data": {"status": status},
            }
            records.append(
                b"data: "
                + json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode()
                + b"\n\n"
            )
        if records:
            yield b"".join(records)
            previous = current
            last_heartbeat = asyncio.get_running_loop().time()
        elif asyncio.get_running_loop().time() - last_heartbeat >= 10:
            yield b": keep-alive\n\n"
            last_heartbeat = asyncio.get_running_loop().time()
        await asyncio.sleep(0.25)


async def _proxy_request(
    request: Request,
    registry: ModelRegistry,
    manager: RouterManager,
    *,
    body_override: dict[str, Any] | None = None,
    allow_frozen_unload: bool = False,
) -> Response:
    raw_body = await request.body()
    body: dict[str, Any] | None = body_override
    if body is None and raw_body:
        try:
            parsed = json.loads(raw_body)
            if isinstance(parsed, dict):
                body = parsed
        except json.JSONDecodeError:
            body = None

    model_id = body.get("model") if body else request.query_params.get("model")
    if (not isinstance(model_id, str) or not model_id) and request.url.path in MODELLESS_PATHS:
        # llama.cpp's UI calls GET /props (and friends) before a model is chosen.
        # Answer them from the already-active backend instead of 400:ing.
        model_id = manager.active_model
    if not isinstance(model_id, str) or not model_id:
        return openai_error("model name is missing from the request", 400, "invalid_request_error")
    model = registry.resolve(model_id)
    if model is None:
        return openai_error(f"model {model_id!r} not found", 400, "invalid_request_error")

    try:
        lease = (await manager.acquire_existing(model_id) if allow_frozen_unload
                 else await manager.acquire(model_id))
    except ModelSwitchError as exc:
        return _switch_error(exc)

    # Reload may have changed the model mapping while acquire() waited for the
    # switch lock. Use the leased model, not the registry snapshot from above.
    model = lease.model
    adapter = lease.adapter
    translated_body = adapter.translate_request(body, model) if body is not None else None
    if translated_body is not None:
        upstream_body = json.dumps(translated_body, separators=(",", ":"), ensure_ascii=False).encode()
    else:
        upstream_body = raw_body

    query_pairs = parse_qsl(request.url.query, keep_blank_values=True)
    translated_query = [
        (key, adapter.translate_query_model(model) if key == "model" else value)
        for key, value in query_pairs
    ]
    query = urlencode(translated_query, doseq=True)
    url = f"{adapter.upstream_url}{request.url.path}"
    if query:
        url += f"?{query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "accept-encoding"
    }
    try:
        upstream_request = manager_http_client(request).build_request(
            request.method,
            url,
            headers=headers,
            content=upstream_body,
        )
        upstream_response = await manager_http_client(request).send(upstream_request, stream=True)
    except asyncio.CancelledError:
        # A client can disconnect after acquire() reserved its lease but before
        # an upstream response exists. Never leave that lease blocking switches.
        await asyncio.shield(lease.release())
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        await lease.release()
        LOG.warning("upstream connection failed backend=%s: %s", model.backend, exc)
        return openai_error(
            f"upstream backend {model.backend} connection failed: {exc}",
            502,
            "backend_connection_error",
            backend=model.backend,
            logs=adapter.tail_log(),
        )

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-encoding"
    }
    content_type = upstream_response.headers.get("content-type", "")
    stream_requested = bool(body and body.get("stream") is True) or content_type.startswith("text/event-stream")

    if stream_requested:
        return StreamingResponse(
            _stream_upstream(
                upstream_response,
                lease,
                public_model=model.id,
                upstream_model=model.upstream_model,
            ),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=None,
        )

    try:
        content = await upstream_response.aread()
        if "json" in content_type.lower():
            content = _rewrite_json_model_identity(
                content,
                model.id,
                model.upstream_model,
                force_router_role=request.url.path == "/props",
            )
    except (httpx.HTTPError, OSError) as exc:
        return openai_error(
            f"upstream backend {model.backend} disconnected: {exc}",
            502,
            "backend_connection_error",
            backend=model.backend,
            logs=adapter.tail_log(),
        )
    finally:
        # This also runs for task cancellation while reading a non-SSE body.
        await asyncio.shield(_close_and_release(upstream_response, lease))
    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


def _rewrite_json_model_identity(
    content: bytes,
    public_model: str,
    upstream_model: str,
    *,
    force_router_role: bool = False,
) -> bytes:
    """Replace backend-native model IDs at the top level of a JSON response."""
    if public_model == upstream_model and not force_router_role:
        return content
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return content
    if not isinstance(payload, dict):
        return content

    rewritten = False
    for field in ("model", "model_alias"):
        if payload.get(field) == upstream_model:
            payload[field] = public_model
            rewritten = True
    if force_router_role and payload.get("role") != "router":
        payload["role"] = "router"
        payload.setdefault("max_instances", 1)
        payload.setdefault("models_autoload", True)
        rewritten = True
    if not rewritten:
        return content
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


def _rewrite_sse_line(line: bytes, public_model: str, upstream_model: str) -> bytes:
    """Rewrite a JSON SSE data line while preserving non-JSON events verbatim."""
    content = line.rstrip(b"\r\n")
    ending = line[len(content):]
    if not content.startswith(b"data:"):
        return line
    value = content[len(b"data:"):]
    whitespace = value[:len(value) - len(value.lstrip())]
    payload = value[len(whitespace):]
    if not payload or payload == b"[DONE]":
        return line
    rewritten = _rewrite_json_model_identity(payload, public_model, upstream_model)
    if rewritten == payload:
        return line
    return b"data:" + whitespace + rewritten + ending


async def _stream_upstream(
    response: httpx.Response,
    lease: ModelLease,
    *,
    public_model: str,
    upstream_model: str,
) -> AsyncIterator[bytes]:
    # SSE JSON can be split across arbitrary HTTP chunks. Buffer only through
    # the next newline so each event still reaches the client incrementally.
    pending = b""
    try:
        async for chunk in response.aiter_raw():
            if not chunk:
                continue
            pending += chunk
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                line, pending = pending[:newline + 1], pending[newline + 1:]
                yield _rewrite_sse_line(line, public_model, upstream_model)
        if pending:
            yield _rewrite_sse_line(pending, public_model, upstream_model)
    except (httpx.HTTPError, OSError) as exc:
        LOG.warning("stream disconnected backend=%s: %s", lease.model.backend, exc)
    finally:
        await asyncio.shield(_close_and_release(response, lease))


async def _close_and_release(response: httpx.Response, lease: ModelLease) -> None:
    try:
        await response.aclose()
    finally:
        await lease.release()


def manager_http_client(request: Request) -> httpx.AsyncClient:
    client = request.app.state.client
    if client is None:
        # Used only by direct ASGI tests that opt out of app lifespan.
        timeout = httpx.Timeout(connect=10, read=None, write=300, pool=10)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=0)
        client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)
        request.app.state.client = client
    return client


def _switch_error(exc: ModelSwitchError) -> JSONResponse:
    return openai_error(
        str(exc),
        503,
        "backend_switch_error",
        backend=exc.backend,
        logs=exc.logs,
    )
