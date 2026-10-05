# Operations

## Endpoints

Inference and llama.cpp-compatibility endpoints are proxied to whichever backend
owns the requested model: `/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/models/sse`, `/props`, `/slots`, `/metrics`, `/tokenize`,
`/completion`, `/chat`, …

Control plane (authenticated when an API key is configured):

| Endpoint | What it does |
|---|---|
| `GET /health` | active model/backend + upstream health |
| `GET /router/status` | full state: frozen, active, switching_to, per-backend pid/pgid/running/resident/evict_cost/idle_seconds/restart_pending, config path + watched flag |
| `POST /models/load` `{"model": id}` | warm a model now (switches engine if needed) and block until servable |
| `POST /models/unload` `{"model": id}` | drain, stop that backend's process group, wait for the GPU gate |
| `POST /router/freeze` / `POST /router/thaw` | block/allow model loads while other GPU work runs |
| `POST /router/reload` | re-read `router-backends.yaml` (no process restart, no model restart) |
| `GET /router/activity?limit=N&since=ts` | recent lifecycle events |
| `GET /activity/stream` | live SSE feed of the same events |
| `GET /logs/<backend>?lines=N` | tail the router-owned log for a backend |

Helper: `python3 scripts/routerctl.py status|models|load|unload|freeze|thaw|reload|activity|logs|watch`.

## The switch sequence

```
request for model M
  └ switch_lock
      ├ resolve M -> backend B
      ├ if B active & healthy ...................... reuse (no restart)
      ├ else: drain in-flight requests (<= drain_timeout_seconds)
      ├ stop previous backend process group (SIGTERM -> SIGKILL)
      ├ resident backends are left running on purpose
      ├ GPU gate: wait for min_free_vram_mib free, OR adopt an external copy
      ├ start B (start_new_session -> own process group)
      ├ wait: /health 200 -> model actually loaded (adapter-specific)
      └ reserve the request lease, then proxy
```

Every step is audited in `/router/activity`:
`startup`, `switch`, `backend_ready` (with `duration_seconds`), `switch_failed`,
`coexist`, `idle_unload`/`idle_unloaded`, `manual_unload`, `freeze`/`thaw`,
`config_reload`, `config_reload_failed`.

## Freeze windows for non-LLM GPU work

Image/video generation, training, DCC tools, and benchmarks must not have a model
reload racing them:

```bash
python3 scripts/routerctl.py freeze      # loads and switches now return 503
# ...run ComfyUI / Blender / a benchmark...
python3 scripts/routerctl.py thaw
```

Note on `/models/unload` parity: when the requested model belongs to the
llama.cpp router backend, the call is forwarded to that child router so
llama.cpp unloads its own LRU slot; for every other backend the facade stops the
whole process group. `GET /v1/models` and `/models/sse` always reflect the
facade's view, not the active engine's.

`/models/unload` is deliberately **not** blocked by freeze: a window can still
free the model it is standing on. Freeze is in-memory (fail-open on restart), so
pair it with your job runner's own bookkeeping.

Leases from clients: the router only counts requests it is currently proxying. If
you have an interactive agent that holds the model "logically" between turns
(e.g. a coding CLI), have it touch `/health` or write its own lease file and let
your idle TTL respect it. This is deliberately left as an integration point.

## Hot reload semantics

The watcher (or `POST /router/reload`) applies a config edit like this:

| Change | Effect |
|---|---|
| new model / metadata edit | live immediately |
| new backend | usable on the next request that names it |
| edit to a **stopped** backend | live immediately |
| edit to a **running** backend | the new launch line is applied on the **next switch** to that backend (`restart_pending: true` in status); an idle changed backend is stopped right away |
| remove a backend (idle) | stopped, then forgotten |
| remove a backend (active) | kept until the next switch, and reported |
| broken YAML / bad reference | rejected: the running config stays, `config_reload_failed` is recorded, `POST /router/reload` answers 503 |

The active model is **never** killed by a reload. That is the whole point: adding
a checkpoint should not cost you a three-minute cold load.

## Logs

Router-owned capture: `settings.state_dir/logs/<backend>.log` — a `=== router
start ... ===` marker, then the child's stdout and stderr merged. Read it with
`GET /logs/<backend>` (or `tail`). Engines that write their own log files are
unaffected; the facade only captures what they print.

## Capacity and queueing behaviour to expect

- Cross-engine switches are serialized: one at a time, other callers wait rather
  than getting a partial state.
- Requests for the *same* backend coalesce into one in-progress start.
- A cold load blocks the requesting client until the model is servable — set
  client read timeouts longer than `startup_timeout_seconds`.
- Nothing unloads on its own unless you set an idle TTL.

## Monitoring ideas

- `GET /router/status` is a single JSON document designed for scraping: alert on
  `healthy: false` while `active_backend` is non-null, on
  `restart_pending` sticking, or on `switching_to` older than your SLO.
- `GET /router/activity?since=<ts>` gives a cheap changelog for a dashboard.
- `/metrics` is proxied to the active backend (llama.cpp exposes Prometheus
  metrics with `--metrics`), so per-token performance stays visible.
