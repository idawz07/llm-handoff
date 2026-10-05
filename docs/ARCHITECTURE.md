# Architecture

Four modules, one process, ~2.1k lines. The design goal is not throughput — the
engines handle that. The goal is that **the GPU is never owned by two things at
once, and no switch ever destroys work in progress.**

```
handoff_router/
  cli.py        argument + env resolution, --print-config, uvicorn bootstrap
  registry.py   router-backends.yaml -> RouterSettings / BackendConfig / ModelConfig
  app.py        FastAPI facade: auth, proxying, SSE rewriting, control plane
  manager.py    the supervisor: switch lock, leases, drain, idle reaper, reload
  backends.py   per-engine adapters + process-group lifecycle + adoption
  activity.py   bounded lifecycle event ring + SSE fan-out
```

---

## Invariants

These are the properties the code is organized around. A change that breaks one
of them is a bug, even if every test passes.

1. **One non-resident backend at a time.** `_ensure_active_locked()` refuses to
   start a backend while another non-resident one is running.
2. **A switch never kills a live generation.** Drain happens before teardown,
   bounded by `drain_timeout_seconds`; a timeout fails the switch, it does not
   cut the stream.
3. **The lease is reserved while holding the switch lock.** `acquire()` increments
   the active-request count before releasing `switch_lock`, so a competing switch
   can never observe "zero requests" in the window between readiness and proxying.
   (This was a real race in an earlier revision.)
4. **Client disconnection cannot leak state.** Leases are released through
   `asyncio.shield`, and start/cleanup runs inside `_complete_on_cancel`, so a
   browser tab closing mid-load does not leave a half-started 50 GiB process.
5. **Process groups, not PIDs.** Children start with `start_new_session=True`;
   stop is `killpg(SIGTERM)` → `killpg(SIGKILL)`. A launcher that dies while its
   CUDA/API workers survive is still reclaimed.
6. **Adoption over duplication.** A healthy process matching `process_match` on
   the target port is adopted instead of starting a second copy for the same port
   and the same GPU. Adoption requires the pid to be its own process-group leader,
   so a group signal can never hit an unrelated shell.
7. **Free the GPU before starting the next engine.** Process-group exit is the
   primary signal; `min_free_vram_mib` from `nvidia-smi` is the backstop for a
   consumer the router cannot identify.
8. **Model readiness ≠ process readiness.** `wait_until_ready()` proves the
   process is up; `wait_until_model_loaded()` (adapter-specific) proves the model
   is servable, so a client is never handed a 400 "model is not loaded".
9. **Crash recovery is metadata, not a daemon.** `state.json` records
   `{active_backend, active_model, pid, pgid, boot_id, command}` atomically. On
   restart, a matching healthy group is adopted; a stale group from the same boot
   is terminated; anything from another boot is ignored.
10. **Reload never disturbs the active model.** Changed launch lines apply on the
    next switch; a removed-but-active backend survives until then.

---

## Two paths that are easy to get wrong

**External supervisor vs. the router.** A second supervisor (a job runner, a
service manager) may start the very backend we are about to start, while our
switch is still draining. `_wait_for_target_or_gpu()` therefore races two
conditions: "the target is becoming healthy" (adopt it) and "the GPU is free"
(start it ourselves). Whichever wins, we do not start a duplicate.

**llama.cpp presets inside a facade.** llama.cpp's multi-model router answers
`/health` with 200 while *no* model is loaded, and starts a load only when
something asks for a model. The adapter issues a single long-timeout `GET /props`
with autoload — which both triggers and awaits the load server-side — instead of
re-issuing probes, because a client disconnect during a probe cancels the load.

---

## What is deliberately not here

- **No request queue of our own beyond asyncio's lock.** Callers serialize on the
  switch lock; per-model concurrency belongs to the engine.
- **No scheduler that packs several big models onto one card.** Coexistence is
  declared by the operator (`resident`), not inferred.
- **No response buffering.** SSE is forwarded line-by-line with only enough
  buffering to find a newline, so token latency is the engine's, not ours.
- **No model identity leakage.** The `model` field is rewritten on the way in
  (`public id → upstream id`) and back out (`upstream id → public id`) in both
  JSON bodies and SSE lines, so a client can pin one public id while the backend
  underneath changes.

---

## Testing philosophy

`tests/fake_backend.py` is a 150-line OpenAI-ish server that starts, writes an
"active" marker, counts starts, and exits on SIGTERM. That is enough to test the
things that actually break in production: the full switch matrix, coalescing
concurrent startups, graceful draining, cancellation mid-start, incremental SSE,
post-start crashes, stale-process adoption, recovered process groups, the
one-non-resident-backend invariant, the VRAM gate (via a fake `nvidia-smi`), hot
reload, idle reclaim, and resident coexistence. No GPU, no model files, ~14s.
