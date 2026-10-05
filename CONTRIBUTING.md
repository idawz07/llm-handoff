# Contributing

Small codebase, high-stakes behaviour: a bug here shows up as a lost 50 GiB model
load or a frozen workstation. Tests and invariants matter more than volume.

## Setup

```bash
git clone https://github.com/your-org/handoff-router && cd handoff-router
pip install -e ".[dev]"
pytest -q            # fake backends, no GPU
handoff-router --config config/router-backends.example.yaml --print-config
```

Before the **first public release**, the maintainer runs
`python3 scripts/prepare_release.py OWNER/handoff-router` and then `--check` to
replace all provisional URLs. Contributors need not rewrite these links in PRs.

## Where to work

| Area | File | Notes |
|---|---|---|
| switching, draining, leases, idle reclaim, reload | `handoff_router/manager.py` | read `docs/ARCHITECTURE.md` first: the ten invariants |
| process lifecycle, adoption, per-engine readiness | `handoff_router/backends.py` | new engine = new adapter subclass + one entry in `create_adapter` |
| config surface | `handoff_router/registry.py` | every new key needs a default, a `docs/CONFIG.md` row, and a test |
| HTTP surface, SSE rewriting, auth | `handoff_router/app.py` | keep llama.cpp endpoint parity in `WEB_UI_API_PREFIXES` |

## Good first contributions

1. **Anthropic Messages API** (`/v1/messages`) — the facade is protocol-agnostic
   except for the `model` rewrite; llama-swap already proved the demand.
2. **Prometheus metrics** for `switch_total`, `switch_duration_seconds`,
   `idle_unload_total`, `backend_up_total` — the activity ring already has the data.
3. **A richer coexistence model.** `resident`/`evict_cost` is a deliberate subset
   of llama-swap's `matrix` router. A cost-based planner (model bytes vs. free
   VRAM from `nvidia-smi`) would fit here naturally.
4. **Non-Linux process handling.** Adoption uses `/proc` and `killpg`; a
   Windows/macOS path would widen the audience.
5. **`sleep()` instead of kill** where the engine supports it (vLLM sleep mode),
   which would turn a minutes-long cold start into a sub-second wake.

## Landing page

`index.html` at the repository root is the GitHub Pages site (self-contained: no
build step, no external assets, `.nojekyll` is present). Publish it with
**Settings → Pages → Deploy from a branch → `main` → `/ (root)`**. Update the
the GitHub URLs via `scripts/prepare_release.py` before publishing. Keep the
comparison section factual: it is the part people judge us on.

## Test expectations

Every behaviour change needs a test in `tests/`. Lifecycle changes need
`tests/test_router.py` (fake backends, real subprocesses, real TCP/SSE);
operator-facing features go in `tests/test_features.py`; audit regressions live
in `tests/test_audit_regressions.py`. Tests must not require a
GPU, a model file, or network access.

## Style

Python 3.10+, `from __future__ import annotations`, type hints, no new runtime
dependency without a strong reason. Comment *why*, especially around races and
signal handling — several comments in this repo record an incident.
