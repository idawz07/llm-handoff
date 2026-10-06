# Changelog

## 0.2.2 — release audit follow-up

- Reject duplicate backend ports and refuse a hot-reload rename that would
  let two adapters adopt the same running process and kill an in-flight stream.
- Leave matching external listeners that are still loading alone at startup;
  fail and retry instead of terminating a process just because health is 503.
- Stop standalone llama.cpp servers (including the sample embedder) for
  `/models/unload`; forward native unload only to `--models-preset` routers.
- Install a readable shim and a stable dispatcher helper beside the real
  binary, independent of checkout location and installer account.
- Increase the default shutdown drain to 300 seconds; align example service
  stop grace with its 1800-second drain, and document the bounded cutoff.
- Fix pipx-only configuration instructions and add regressions for the above.

## 0.2.1 — safety and deployment audit

- Require configured keys for legacy inference routes and `/props`, including
  method/query variants of otherwise-public paths. Default to a loopback bind;
  require a key to expose a remote listener unless explicitly overridden.
- Preserve legacy shim installations and their real binary; persist install-time
  config/preset paths in the new shim, and fail closed if its config disappears.
- Track leases per backend; protect in-flight resident streams during removal,
  preserve coexistence in both switch directions and across router restarts.
- Stage running adapter changes immutably until their own leases drain; reload
  can change adapter type without redirecting an old request to a new port.
- Correct idle TTL `0` semantics; start the idle clock on request completion.
- Require a nonempty adoption fingerprint and the same UID. Resolve explicit
  `./`/`../` argv paths relative to the YAML file.
- Restore Python 3.10 compatibility; repair mobile comparison-table overflow,
  first-run systemd instructions and container path guidance.
- Add fake-backend and installer regressions for these cases.

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
- **Portability** — no hardcoded developer paths; `~`/`$VAR` and settings-path
  resolution; `--config` with `$HANDOFF_ROUTER_CONFIG` and XDG default;
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
