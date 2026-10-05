# Configuration reference

One file, three sections. Settings paths are resolved relative to this file;
backend `command`/`process_match` entries beginning `./` or `../` are also
config-relative. `~` and `$VARS` expand everywhere. Bare executable names use
`PATH`, and other arguments pass through literally. `handoff-router --config
<file> --print-config` shows the result without starting anything.

```yaml
settings:  { ... }   # global envelope
backends:  { ... }   # one entry per engine/process the router may own
models:    { ... }   # public selector names -> backend
```

---

## `settings`

| Key | Default | Meaning |
|---|---|---|
| `state_dir` | `~/.local/state/handoff-router` | holds `state.json` (crash recovery) and `logs/<backend>.log` |
| `llama_backend` | `llamacpp` | name of the backend whose `--models-preset` IDs are auto-imported |
| `llama_presets` | *(none)* | path to the llama.cpp preset `.ini`. Every `[Section]` becomes a public model ID |
| `llama_models_snapshot` | *(none)* | cached `/v1/models` JSON from llama.cpp; used only for input modalities of preset IDs |
| `drain_timeout_seconds` | `300` | how long a switch waits for in-flight requests/streams to finish before failing (never kills them) |
| `shutdown_drain_timeout_seconds` | `30` | same, for service shutdown |
| `stop_timeout_seconds` | `30` | SIGTERM grace before SIGKILL for a process group |
| `startup_timeout_seconds` | `300` | global ceiling for "process up → healthy → model servable" |
| `health_poll_seconds` | `0.5` | health polling interval |
| `min_free_vram_mib` | `0` | gate: no backend starts until `nvidia-smi` reports this much free. `0` disables |
| `gpu_release_timeout_seconds` | `90` | how long to wait for that gate |
| `gpu_poll_seconds` | `1` | gate polling interval |
| `nvidia_smi` | `nvidia-smi` | path to the query tool |
| `upstream_connect_timeout_seconds` | `10` | proxy connect timeout |
| `upstream_write_timeout_seconds` | `300` | proxy request-body write timeout |
| `idle_ttl_seconds` | `0` | idle reclaim. `0` = off. A negative value opts out of idle reclaim (not manual unload or model switches) |
| `idle_poll_seconds` | `5` | reaper interval (floored at 1s) |
| `idle_grace_seconds` | `30` | never unload a backend younger than this |
| `watch_config` | `true` | reload `router-backends.yaml` when its mtime changes |
| `config_poll_seconds` | `2` | watcher interval (floored at 0.5s) |
| `activity_history` | `200` | size of the in-memory lifecycle event ring |

---

## `backends.<name>`

| Key | Default | Meaning |
|---|---|---|
| `command` | *required* | argv that launches the engine. Must bind to loopback with a private `--port` |
| `upstream` | *required* | base URL the facade proxies to, e.g. `http://127.0.0.1:18087` |
| `adapter` | backend name | one of `llamacpp`, `vllm`, `sglang`, `strata`, `ninfer` (model-loading semantics per engine) |
| `health_endpoint` | `/health` | readiness probe used after start and before reuse |
| `environment` | `{}` | extra env vars for the child process |
| `process_match` | *required if adoption enabled* | nonempty literal substrings that must **all** appear in `/proc/<pid>/cmdline` before adoption. Make them unique |
| `startup_timeout_seconds` | settings | per-engine override (cold MoE loads need minutes) |
| `stop_timeout_seconds` | settings | per-engine override |
| `adopt_existing` | `true` | allow adopting an already-running copy instead of starting a duplicate; set `false` if no safe `process_match` is available |
| `resident` | `false` | never evicted by a model switch — for small co-resident sidecars |
| `evict_cost` | `1` | relative pain of reloading. Idle reclaim stops cheap backends first |
| `idle_ttl_seconds` | settings | per-backend idle TTL; zero disables idle reclaim for that backend, negative opts out explicitly |

**Adoption is the safety feature here.** An empty fingerprint is rejected when
adoption is enabled; the process must also have the router's UID and lead its
own process group. With correct `process_match` entries the router can (a) reuse
a healthy engine left by a previous router process, (b) adopt
an engine that an external supervisor started while a switch was draining, and
(c) refuse to start a duplicate that would fight for the same port and GPU.

---

## `models.<public-id>`

```yaml
models:
  flash-next-256k:
    backend: strata                 # which backend owns it (required)
    upstream_model: flash-next-iq3  # name sent upstream (defaults to the public id)
    owned_by: strata                # reported in /v1/models
    selector: true                  # show as an addressable selector entry
    input_modalities: [text, image] # drives the web UI's vision badge
    metadata:
      context_window: 262144        # surfaced via /props and /v1/models
      max_tokens: 65536
      description: free-form note for clients and UIs
```

llama.cpp preset IDs are **not** declared here — they come from
`settings.llama_presets` automatically. Use a `models:` entry when you want a
public name that differs from the upstream ID, or for any non-llama.cpp engine.

---

## Coexistence recipe

Default behaviour is strictly one non-resident backend at a time. To keep an
embedder/reranker up while swapping big LLMs:

```yaml
backends:
  embedder:
    resident: true
    evict_cost: 1
    idle_ttl_seconds: 900
  big-moe:
    evict_cost: 30          # expensive to reload; reclaimed last
    idle_ttl_seconds: 1200
```

Then subtract the sidecar's VRAM from `min_free_vram_mib`, otherwise the gate
never opens while the sidecar is resident.

## Idle reclaim recipe

```yaml
settings:
  idle_ttl_seconds: 0        # big models stay resident (cold load = minutes)
backends:
  embedder:   { idle_ttl_seconds: 300 }     # reclaim the small one fast
  big-moe:    { idle_ttl_seconds: 1800 }    # reclaim after 30 min of quiet
  pinned:     { idle_ttl_seconds: -1 }      # never unload
```
