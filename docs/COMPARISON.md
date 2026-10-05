# How this compares (and when you should not use it)

Honest comparison against tools we like and use. This project is small, and the
tools below are excellent, widely used, and mostly *better defaults* than this one
for the common case. We would rather send you to the right tool than flatter ours.

Verified against upstream documentation in October 2026: `llama-swap` README and
knowledge base, `ollama` FAQ, `GPUStack` README, `LocalAI` docs.

---

## The short version

| If your situation is… | Use |
|---|---|
| "I have several llama.cpp/vLLM/ComfyUI servers and want one OpenAI endpoint that starts the right one" | **llama-swap.** It is the general-purpose answer, single Go binary, richer feature set |
| "I want a polished local model runner with auto load/unload and a desktop app" | **Ollama** or **LM Studio** |
| "I want one endpoint for many modalities (LLM, TTS, images, embeddings) with a backend gallery" | **LocalAI** |
| "I have several GPUs/worker nodes and want placement, metrics, and autoscaling" | **GPUStack** (or vLLM Production Stack / Kubernetes) |
| "I have **one GPU** and **several engines** that each need most of its VRAM, and handoffs must never OOM the box" | this project |

The distinction is real: llama-swap's unit of management is a **model** (one
command per model), while ours is an **engine** (one supervised process group per
runtime, which may itself host many models).

---

## Feature by feature

| Capability | Handoff Router | llama-swap | Ollama | LocalAI | GPUStack |
|---|---|---|---|---|---|
| Language / shape | Python, ~2.1k lines, one process | Go, single binary | Go, daemon | Go + C++ backends | Python, server + workers |
| Arbitrary engine commands | ✅ | ✅ | ❌ (built-in runtimes) | gallery backends | pluggable engine configs |
| Multi-engine on one GPU | ✅ core purpose | ✅ (`groups`/`matrix` router) | ✅ memory-aware | ✅ (watchdog) | ✅ scheduler |
| Model-per-entry config | ❌ by design: engine-per-entry, llama.cpp presets auto-imported | ✅ | ✅ | ✅ | ✅ |
| Free-VRAM gate before start | ✅ `min_free_vram_mib` via `nvidia-smi` | ❌ (cost model instead: `evict_costs`) | ✅ VRAM-fit scheduling | partial | ✅ |
| Process-group teardown (`killpg`) | ✅ | per-process (`cmd-stop`) | internal | internal | container/job |
| Adopt a process it did not start | ✅ (`process_match`, listener discovery, persisted pgid) | ❌ | ❌ | ❌ | ❌ |
| Drain in-flight streams before switch | ✅ `drain_timeout_seconds` | ✅ queues/`unloadTimeout` | ✅ queues | partial | n/a |
| Idle TTL unload | ✅ `idle_ttl_seconds` (off by default) | ✅ `ttl` | ✅ `keep_alive` (5 min default) | ✅ watchdog idle | ✅ scale-to-zero |
| Coexistence rules | ✅ `resident` + `evict_cost` (simple) | ✅ `groups` / `matrix` boolean DSL (far more expressive) | automatic | partial | ✅ |
| Config hot reload | ✅ never disturbs the active model | ✅ `--watch-config` | n/a | ✅ | ✅ |
| Freeze window for other GPU work | ✅ `/router/freeze` | `hooks`/manual | ❌ | ❌ | ✅ drain/undeploy |
| Operate the *existing* llama.cpp router untouched | ✅ binary-path shim | ❌ (own port/service) | ❌ | ❌ | ❌ |
| llama.cpp web UI works against non-llama.cpp engines | ✅ synthesized `/props` | ❌ | n/a | own UI | own UI |
| Anthropic Messages API | ❌ (contributions welcome) | ✅ | via Ollama API | partial | ✅ |
| Non-LLM endpoints (SDAPI, audio.cpp, ComfyUI) | ❌ | ✅ | ❌ | ✅ | ❌ |
| Aliases, per-model API keys, macros/hooks, profiles | partial (selector IDs, metadata) | ✅ rich | ✅ | ✅ | ✅ |
| Kubernetes deployment | docs-level | ✅ `kubeswap` | ✅ | ✅ | ✅ |
| Test suite without a GPU | ✅ integration tests, fake backends | ✅ | ✅ | ✅ | ✅ |

---

## Where llama-swap is better

We use the same mental model and we are glad it exists; these are genuine
advantages worth stealing (and the ones we would upstream):

- **Hot config reload with a richer change model** — plus a web UI, `/running`,
  `/upstream/:model_id`, and streamed upstream logs.
- **`matrix` routing DSL with `evict_costs`** — a boolean solver over which
  models may coexist. Our `resident` + `evict_cost` pair is a deliberately small
  subset of it.
- **Protocol breadth** — OpenAI *and* Anthropic *and* llama.cpp *and* SDAPI/audio
  endpoints, so llama-swap fronts image and audio servers too.
- **Deployment ergonomics** — one static binary, Docker images, Kubernetes
  recipes, tailscale integration, docs served over MCP.
- **Distribution** — thousands of installs and years of edge-case hardening.

## Where Ollama is better

Memory-aware scheduling and idle TTL are built in and require no configuration,
plus a model registry, resumable pulls, and a huge client ecosystem. Its tradeoff:
the engines are the ones Ollama ships, so a custom runtime (an expert-streaming
MoE engine, an NVFP4 stack, a llama.cpp fork with different kernels) does not
slot in.

## Where this project is different

1. **It inherits llama.cpp's own router.** `--models-preset` + `--models-max 1`
   already gives you a working LRU over GGUF presets. We keep it and front it, so
   you never re-declare your GGUF collection in a second config language.
2. **Handoff safety over one GPU.** Process-group teardown, an explicit
   free-VRAM gate, adoption instead of duplication, and a single switch lock that
   reserves the request lease before releasing the lock. The failure mode this
   prevents is specific and painful: a 50 GiB reload and a 27B unload overlapping
   on one card.
3. **Zero-diff adoption** via the launcher shim, for people whose router is
   already running under a service manager they do not want to rewrite.
4. **An operator control plane in the response headers of your own tooling**:
   freeze windows, `/router/activity`, per-backend log tails, and a reload path
   that never costs you a cold load.

## Interop, not replacement

These compose. llama-swap can front the facade (or vice versa: the facade can own
the engines that llama-swap would otherwise start). llama.cpp users get real value
from both: keep llama-swap for the many-models-many-engines breadth, and use this
where a single card must be handed between runtimes that each want all of it.

## Known gaps in ours

- No Anthropic API surface, no image/audio endpoints.
- Coexistence rules are coarse (`resident`, `evict_cost`) compared to a solver.
- No web UI of our own (the llama.cpp UI works, including against foreign
  engines, via synthesized props).
- Single process: no multi-GPU placement, no distributed workers.
- Python dependency chain instead of one static binary.

If any of those matter more to you than single-card handoff safety, pick the tool
that already solves it.
