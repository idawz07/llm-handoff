# Handoff Router

**One GPU. Many engines. Never a fight for VRAM.**

An OpenAI-compatible facade that supervises llama.cpp, Strata, vLLM, SGLang,
NInfer — *any* local model server — as backends of a single public endpoint, and
hands the GPU from one large engine to the next without two non-resident
backends overlapping (small `resident` sidecars may coexist).

```
your clients ───────────────► :8081  Handoff Router
                                    ├─ llama.cpp router     :18084   (--models-preset, its own LRU intact)
                                    ├─ Strata  (125B MoE)   :18087
                                    ├─ NInfer  (NVFP4 27B)  :18089
                                    └─ embedder (resident)  :18090
```

You pick a model in the request. The facade decides which engine that needs,
drains whatever is running, stops its **process group**, waits until
`nvidia-smi` reports enough free VRAM, starts the right engine, waits until the
model is genuinely servable, and only then forwards your request. Streaming is
forwarded chunk by chunk; only the `model` field is rewritten.

> **Why this exists.** [llama-swap](https://github.com/mostlygeek/llama-swap),
> [Ollama](https://github.com/ollama/ollama),
> [LocalAI](https://github.com/mudler/LocalAI), and
> [GPUStack](https://github.com/gpustack/gpustack) already solve model swapping
> very well, and you should probably use one of them. This project exists for one
> narrower constraint: **several engines that each want the whole of one GPU**,
> where a swap that overlaps a teardown OOMs the box. Here the unit of management
> is an *engine*, not a model — and llama.cpp's own `--models-preset` router is
> kept as a backend, so your GGUF presets never get re-declared. See
> [docs/COMPARISON.md](docs/COMPARISON.md) for the full, honest picture.

---

## Features

| | |
|---|---|
| **Engine-level swapping** | any HTTP server with a health endpoint becomes a backend; adapters for `llamacpp`, `vllm`, `sglang`, `strata`, `ninfer` |
| **VRAM-gated handoff** | `min_free_vram_mib` gate + process-group (`killpg`) teardown, so a 50 GiB reload never races a 27B unload |
| **Adoption, not duplication** | a healthy leftover process matching `process_match` is adopted instead of starting a second copy on the same port and GPU |
| **Drain before destroy** | in-flight generations finish first (`drain_timeout_seconds`); a slow client fails the switch, it never cuts a stream |
| **llama.cpp native** | `--models-preset` presets become addressable model IDs automatically; `/props`, `/slots`, `/models/sse` and the llama.cpp web UI keep working — even against non-llama.cpp engines (props are synthesized) |
| **Zero-diff adoption** | a launcher shim lets your existing systemd unit, port, and API key stay exactly as they are |
| **Config hot reload** | reload does not stop a live backend; changed launch lines apply on that backend's next request, after its leases drain |
| **Idle reclaim** | per-backend `idle_ttl_seconds`, cheap-first by `evict_cost`, `resident: true` for sidecars you want to keep |
| **Coexistence** | `resident` backends (embedders, rerankers) survive a big-model switch instead of being evicted |
| **Operator control plane** | `POST /router/freeze|thaw` windows for other GPU work, `GET /router/activity`, `GET /activity/stream`, `GET /logs/<backend>`, `POST /router/reload` |
| **Crash recovery** | persisted pid/pgid/boot-id state; healthy groups are adopted, stale ones reaped, foreign boots ignored |
| **Tested without a GPU** | integration tests against fake backends: switch matrix, coalescing, draining, cancellation, adoption, VRAM gate, reload, idle reclaim |

---

## Quickstart

```bash
pipx install git+https://github.com/your-org/handoff-router.git

mkdir -p ~/.config/handoff-router
BASE=https://raw.githubusercontent.com/your-org/handoff-router/main/config
curl -o ~/.config/handoff-router/router-backends.yaml \
  $BASE/router-backends.example.yaml
curl -o ~/.config/handoff-router/router-presets.ini \
  $BASE/router-presets.example.ini   # only if you run llama.cpp's --models-preset router
# then edit: engine commands, ports, process_match fingerprints

# resolve settings paths and explicit ./command paths without touching the GPU
handoff-router --config ~/.config/handoff-router/router-backends.yaml --print-config

handoff-router --config ~/.config/handoff-router/router-backends.yaml --port 8081 --api-key "$KEY"
```

Then the interesting part — switching engines by *naming a model*:

```bash
curl -s localhost:8081/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model":"flash-next-256k","messages":[{"role":"user","content":"hi"}]}'

python3 scripts/routerctl.py status      # who owns the GPU right now
python3 scripts/routerctl.py activity    # what happened, with durations
```

Already running llama.cpp's router under systemd? The shim preserves its unit,
port, and API key. **Back up your binary and review the [migration steps](docs/DEPLOY.md#5-adopting-it-onto-an-existing-llamacpp-router-install-zero-client-change) before restarting.**

```bash
HANDOFF_ROUTER_CONFIG=$HOME/.config/handoff-router/router-backends.yaml \
HANDOFF_ROUTER_PRESET=$HOME/Models/router-presets.ini \
  deploy/install-dispatcher.sh
# then, during a planned maintenance window: systemctl restart llama-router
```

Full guide: **[docs/DEPLOY.md](docs/DEPLOY.md)**.

---

## Control plane

```http
GET  /router/status          state + per-backend pid/pgid/idle/cost/resident
GET  /v1/models              every id, with status.value and status.backend
POST /models/load            warm now, block until servable
POST /models/unload          drain, stop, free the GPU
POST /router/freeze          block model loads while ComfyUI/training runs
POST /router/thaw
POST /router/reload          apply config edits with no restart, no cold load
GET  /router/activity        recent lifecycle events
GET  /activity/stream        live SSE feed
GET  /logs/<backend>         tail the router-owned log for that backend
```

---

## Comparison at a glance

| | Handoff Router | llama-swap | Ollama | GPUStack |
|---|---|---|---|---|
| unit of management | **engine** | model | model | model instance |
| arbitrary engine commands | ✅ | ✅ | built-in runtimes | ✅ |
| free-VRAM gate before start | ✅ | cost model instead | ✅ VRAM-fit | ✅ |
| adopt a process it didn't spawn | ✅ | ❌ | ❌ | ❌ |
| idle TTL unload | ✅ opt-in | ✅ | ✅ | ✅ |
| coexistence rules | `resident` + `evict_cost` | `groups`/`matrix` DSL | automatic | scheduler |
| keep your existing llama-router unit | ✅ shim | own port | ❌ | ❌ |
| Anthropic / image / audio endpoints | ❌ | ✅ | ❌ | ✅ |
| shape | Python, one process | Go binary | Go daemon | cluster manager |

**Use llama-swap** (or Ollama/LocalAI/GPUStack) for breadth. Use this when one
card must be handed between runtimes that each want all of it.
[docs/COMPARISON.md](docs/COMPARISON.md) goes further, including everything we
think they do better and would happily steal.

---

## Documentation

| | |
|---|---|
| [docs/DEPLOY.md](docs/DEPLOY.md) | install, systemd (system + user), containers, shim onto an existing llama-router, GPU sizing, verification, rollback, troubleshooting |
| [docs/CONFIG.md](docs/CONFIG.md) | every `settings` / `backends` / `models` key |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | endpoints, switch sequence, freeze windows, reload semantics, monitoring |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the ten invariants, and the two races that shaped this code |
| [docs/COMPARISON.md](docs/COMPARISON.md) | llama-swap · Ollama · LocalAI · GPUStack, honestly |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) | dev setup, testing philosophy, disclosure |

---

## Publishing

Choose the GitHub owner/repository before publishing; the URLs deliberately
remain provisional until then:

```bash
python3 scripts/prepare_release.py OWNER/handoff-router
python3 scripts/prepare_release.py --check
```

## Development

```bash
pip install -e ".[dev]"
pytest -q            # fake backends, no GPU
```

## Safety notes

- Backends bind to `127.0.0.1`; only the facade port is reachable.
- The facade binds `127.0.0.1` by default; non-loopback binds require an API
  key. `/router/reload` can execute commands **as the router's service user**
  if an attacker can also write its config. Use a trusted network or TLS proxy.
- Set the cgroup memory caps drop-in (`deploy/systemd/memory-guard.conf.example`)
  so a runaway load degrades instead of freezing the machine.

## Credits and lineage

Built on llama.cpp's own multi-model router (`--models-preset`, `--models-max`).
Ideas borrowed openly from llama-swap's `groups`/`matrix` router, `ttl`, and
streamed logs, and from Ollama's VRAM-fit scheduler. Nothing here is a claim that
those projects should be replaced; this is one focused integration of those ideas.

## License

MIT — see [LICENSE](LICENSE).
