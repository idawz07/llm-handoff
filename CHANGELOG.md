# Changelog

## 0.2.0 — standalone release

Packaged for general use; extracted from a production single-RTX-5090 setup
(llama.cpp router + Strata + NInfer) where it has been the GPU arbiter since
2026-09.

**Added**
- **Config hot reload** — `watch_config` (on by default) plus `POST /router/reload`.
  Applying an edit never restarts the active backend; changed launch lines apply
  on the next switch; a broken file is rejected and the running config stays.
- **Idle reclaim** — `settings.idle_ttl_seconds` with per-backend overrides and
  negative = pinned; `idle_grace_seconds` protects a just-warmed backend.
- **Coexistence** — `backends.<name>.resident: true` survives a model switch
  (embedders, rerankers), and `evict_cost` orders what idle reclaim stops first.
- **Activity log** — bounded lifecycle ring, `GET /router/activity`,
  `GET /activity/stream` (SSE), with durations for cold starts and unloads.
- **Log access** — `GET /logs/<backend>?lines=N`.
- **Optional llama.cpp** — a config with no `llama_presets` / no llama backend now
  loads, so pure vLLM/Strata/NInfer deployments work.
- **Portability** — no hardcoded developer paths; `~`/`$VAR`/relative path
  resolution everywhere; `--config` with `$HANDOFF_ROUTER_CONFIG` and XDG default;
  `--print-config`; `handoff-router` console script; `pyproject.toml`; CI.
- **Auth default** — `/models/load` and `/models/unload` now require an API key
  (`--keyless-model-api` restores llama.cpp's keyless web-UI behaviour).
- `scripts/routerctl.py` operator helper, systemd units (system + user), memory
  guard drop-in, container deploy files.

**Changed**
- package `backend_router` → `handoff_router`; `--backend-registry` still accepted.
- idle unload clears the active-model label *before* teardown, so the facade never
  advertises a backend that is going away.
- web UI directory is opt-in via `HANDOFF_WEB_UI_DIR` instead of a hardcoded path.

## 0.1.x — llama.cpp in-tree (`tools/backend-router`)

Facade over llama.cpp's `--models-preset` router with serialized switching,
request leases and draining, process-group teardown, a configurable free-VRAM
gate, process adoption, persisted crash-recovery state, per-backend logs,
`/router/freeze|thaw` windows, and llama.cpp web UI compatibility including
synthesized `/props` for engines that do not implement it.
