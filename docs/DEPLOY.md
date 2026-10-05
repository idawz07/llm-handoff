# Deploy manual

Everything needed to run Handoff Router for real: on a workstation, in a user
session, as a system service, or in a container. No GPU is required to run the
test suite, and no GPU is required to *read* this file — but every command below
was written for a Linux host with an NVIDIA GPU and one or more local model
servers.

---

## 0. What you are deploying

```
clients (OpenAI-compatible)  ->  :8081  Handoff Router (facade)
                                     ├─ llama.cpp router     :18084   (--models-preset)
                                     ├─ Strata / vLLM / ...  :18087
                                     └─ embedder (resident)  :18090
```

The facade owns one public address. It starts, stops, adopts, and drains backend
process **groups**, and it rewrites only the `model` field of a request. SSE
streams pass through chunk-by-chunk.

---

## 1. Requirements

| Requirement | Notes |
|---|---|
| Linux (x86_64 / arm64) | uses `/proc`, process groups, `killpg` |
| Python ≥ 3.10 | 3.11–3.13 are what CI runs |
| fastapi, httpx, psutil, PyYAML, uvicorn | pulled in automatically |
| One or more inference servers | any HTTP server with a health endpoint; adapters ship for `llamacpp`, `vllm`, `sglang`, `strata`, `ninfer` |
| `nvidia-smi` (optional) | only for the free-VRAM gate; set `min_free_vram_mib: 0` without it |
| Single-process | the supervisor is in-memory; do **not** run with `--workers > 1` (uvicorn default is 1) |

---

## 2. Install

```bash
# from a published release (if the package is on your index)
pipx install handoff-router

# or straight from the repository - what works today
pipx install git+https://github.com/your-org/handoff-router.git

# or, in a venv
python -m venv ~/.venvs/handoff-router && source ~/.venvs/handoff-router/bin/activate
pip install git+https://github.com/your-org/handoff-router.git

# from a checkout (development)
git clone https://github.com/your-org/handoff-router && cd handoff-router
pip install -e ".[dev]"
```

Verify the binary:

```bash
handoff-router --version
```

---

## 3. Configure

```bash
mkdir -p ~/.config/handoff-router
cp config/router-backends.example.yaml ~/.config/handoff-router/router-backends.yaml
cp config/router-presets.example.ini   ~/.config/handoff-router/router-presets.ini
$EDITOR ~/.config/handoff-router/router-backends.yaml
```

Three things to get right before the first start:

1. **`backends.<name>.command`** — the exact argv that launches each engine, with
   loopback-only `--host 127.0.0.1` and a dedicated `--port`. Backends must not be
   reachable by clients directly; only the facade address should be.
2. **`backends.<name>.process_match`** — 2–4 literal substrings that uniquely
   identify that process's command line (`cmdline` is matched with *all* fragments
   present). This is what lets the router adopt a leftover process instead of
   starting a duplicate that fights it for the port. Make them specific.
3. **`settings.min_free_vram_mib`** — see §6.

Validate without touching the GPU:

```bash
handoff-router --config ~/.config/handoff-router/router-backends.yaml --print-config
```

This resolves every `~`/`$VAR`/relative path and prints the effective backend and
model table. Errors here (missing preset file, unknown backend reference, empty
command) are the ones that would otherwise surface at first request.

---

## 4. Run it

### 4.1 Foreground (first run)

```bash
handoff-router --config ~/.config/handoff-router/router-backends.yaml \
  --host 0.0.0.0 --port 8081 --api-key "$MY_KEY"
```

### 4.2 System service

```bash
sudo useradd -r -m -G video handoff                # or reuse your account
sudo -u handoff mkdir -p ~/.config/handoff-router
sudo -u handoff cp config/router-backends.example.yaml ~/.config/handoff-router/router-backends.yaml
sudo cp deploy/systemd/handoff-router.service /etc/systemd/system/handoff-router.service
sudo sed -i 's/HANDOFF_USER/handoff/g' /etc/systemd/system/handoff-router.service
sudo systemctl daemon-reload
sudo systemctl enable --now handoff-router
journalctl -u handoff-router -f
```

Keep the API key out of the unit file:

```bash
sudo install -d -m 750 -o root -g handoff /etc/handoff-router
printf 'HANDOFF_ROUTER_API_KEY=%s\n' "$(openssl rand -hex 32)" | sudo tee /etc/handoff-router/env
sudo chmod 640 /etc/handoff-router/env
```

### 4.3 User service (desktop box, no root)

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/handoff-router.service.user ~/.config/systemd/user/handoff-router.service
systemctl --user daemon-reload
systemctl --user enable --now handoff-router
loginctl enable-linger "$USER"       # keep it running after logout
```

### 4.4 Container

```bash
docker build -t handoff-router .
docker run --rm --gpus all -p 8081:8081 \
  -e HANDOFF_ROUTER_API_KEY="$MY_KEY" \
  -v ~/.config/handoff-router:/config:ro \
  -v ~/.cache/llama.cpp:/engines:ro \
  -v ~/models:/models:ro \
  handoff-router --config /config/router-backends.yaml
```

Container caveats: the router sends `SIGTERM`/`SIGKILL` to **process groups** of
backends it starts, so the container must let it (avoid `--pid=container` sharing
with an unrelated init, and do not wrap the router in a shell that swallows
signals). Keep model and engine paths bind-mounted identically inside and outside
the container if you want `process_match` fingerprints to stay readable.

---

## 5. Adopting it onto an existing llama.cpp router install (zero client change)

If you already run llama.cpp's multi-model router under a service, you do not
have to change the unit, the port, or the API key. `install-dispatcher.sh` moves
the real binary aside and leaves a narrow shim in its place:

```bash
HANDOFF_LLAMA_ROOT=~/llama.cpp \
HANDOFF_ROUTER_PRESET=$HOME/Models/router-presets.ini \
HANDOFF_ROUTER_CONFIG=$HOME/.config/handoff-router/router-backends.yaml \
  deploy/install-dispatcher.sh

systemctl restart llama-router          # your existing unit
```

Behaviour of the shim:

- the invocation matching `--models-preset <HANDOFF_ROUTER_PRESET>` **and**
  `--port <HANDOFF_ROUTER_PORT>` runs the facade;
- every other `llama-server` invocation (including `--version`) delegates to
  `build/bin/llama-server.real` byte-for-byte;
- `HANDOFF_ROUTER_BYPASS=1` forces the untouched binary — handy for debugging;
- `deploy/uninstall-dispatcher.sh` restores the original binary.

Rerun the installer after any llama.cpp rebuild: a rebuild replaces
`build/bin/llama-server` and removes the shim.

**Migrating from the llama.cpp in-tree `tools/backend-router`:** the package was
renamed `backend_router` → `handoff_router` and now reads `--config` (the old
`--backend-registry` flag is still accepted). Point your unit at
`handoff-router --config <your router-backends.yaml>`, set
`settings.llama_presets` explicitly, and keep your existing `process_match`
entries — they are the only part that must stay byte-identical for adoption to
work.

---

## 6. Sizing the GPU safety envelope

Two independent guards, both important:

**a) `settings.min_free_vram_mib` (VRAM gate).** After the old backend's process
group exits, no new backend starts until `nvidia-smi` reports at least this much
free. Set it to roughly the working set of the *largest model you switch to*:

| Largest target model | suggested `min_free_vram_mib` |
|---|---|
| 7–14B Q4/Q5 + 8–32K ctx | 12000 |
| 27–35B Q6 + 64–128K ctx, Q8 KV | 26000–32000 |
| 70–125B MoE (partly CPU-resident) + 128–256K ctx | 42000–50000 |
| no `nvidia-smi` / CPU-only | `0` (disabled) |

If you run **resident** backends (§ docs/CONFIG.md), subtract their footprint
from the threshold, or they will keep the gate permanently shut.

**b) systemd memory caps.** The VRAM gate protects the GPU; cgroup caps protect
the *host* from a model whose weights spill into RAM/swap:

```bash
sudo mkdir -p /etc/systemd/system/handoff-router.service.d
sudo cp deploy/systemd/memory-guard.conf.example \
     /etc/systemd/system/handoff-router.service.d/memory.conf
sudo systemctl daemon-reload && sudo systemctl restart handoff-router
```

Pick `MemoryMax` so that `total RAM − MemoryMax ≥ ~3 GiB`, and keep
`MemorySwapMax` small (e.g. 8G) so a runaway load degrades instead of freezing the
desktop. `OOMPolicy=kill` then makes a kernel kill tear the whole unit down so
`Restart=on-failure` brings back a clean router.

---

## 7. Verify the deployment

```bash
KEY=... ; U=http://127.0.0.1:8081
curl -s -H "Authorization: Bearer $KEY" $U/health | jq
# {"status":"ok","active_model":null,"active_backend":null,"backend_healthy":false}

curl -s -H "Authorization: Bearer $KEY" $U/v1/models | jq -r '.data[]|.id+"  "+.status.value'

# cold start the first model (this is the slow one)
time curl -s -X POST -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"<one-of-your-ids>","messages":[{"role":"user","content":"hi"}]}' \
  $U/v1/chat/completions | jq .choices[0].message.content

# switch engines: the previous backend must be gone before the new one starts
curl -s -H "Authorization: Bearer $KEY" $U/router/status | jq '.active_backend, .backends'
python3 scripts/routerctl.py activity 20
```

Pass criteria:

- `/v1/models` lists llama.cpp preset IDs **and** your `models:` selectors, each
  with `status.value` and `status.backend`;
- during a cross-engine switch the response's `model` is the *public* id, never
  the upstream id;
- `routerctl.py activity` shows `switch` → `backend_ready` with a plausible
  `duration_seconds`;
- `journalctl -u handoff-router` never shows two backends running at once unless
  they are `resident: true`;
- `POST /router/freeze` then any model load returns HTTP 503 `backend_switch_error`;
  `POST /router/thaw` restores it.

---

## 8. Upgrades, reloads, rollback

| Change | Action |
|---|---|
| edit `router-backends.yaml` | nothing — the watcher reloads within `config_poll_seconds`; or `python3 scripts/routerctl.py reload` |
| changed a **running** backend's `command` | applied automatically on the **next switch** to that backend (never kills a live model) |
| removed a backend that is running but idle | stopped immediately; if it was active, it is kept until the next switch and reported as `restart_pending` |
| llama.cpp rebuild (if you use the shim) | rerun `deploy/install-dispatcher.sh`, restart your unit |
| `pip install --upgrade handoff-router` | restart the service (drains in-flight generations first) |
| rollback | `pip install handoff-router==<old>` + restart; the `state.json` format is backward compatible |

`settings.state_dir/state.json` is only crash-recovery metadata (active backend,
pid, pgid, boot_id, command). Deleting it is safe: on next start the router
discovers listeners and adopts or cleans up by itself.

---

## 9. Security posture

- **Bind the facade, not the engines.** All backend `command`s must use
  `--host 127.0.0.1`. Only the facade port may be reachable.
- **API keys** (`--api-key`, `--api-key-file`, `HANDOFF_ROUTER_API_KEY`) protect
  everything except `/health`, `/v1/health`, `/models`, `/v1/models`, and the
  optional llama.cpp web UI shell. That matches llama.cpp's own behaviour so
  existing clients keep working — put a TLS-terminating reverse proxy in front if
  the box is not on a trusted network.
- **The control plane is authenticated**: `/router/status|freeze|thaw|reload`,
  `/router/activity`, `/activity/stream`, `/logs/<backend>`, `/models/load`,
  `/models/unload`. `POST /router/reload` re-reads a config file that can launch
  arbitrary commands — treat the API key as root-equivalent on that host.
- Run as an unprivileged user in the `video` group, models owned by that user,
  config readable only by it.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `503 backend_switch_error: GPU memory did not reach N MiB free` | the old backend's memory is still being released, or a foreign process holds the GPU | check `nvidia-smi`, lower `min_free_vram_mib`, or stop the foreign consumer; the error text includes the last observed free VRAM |
| `backend X port is held by external pid N that is not healthy` | a hand-started engine occupies the port but fails its health check | stop it, or fix `process_match` if it belongs to a *different* service |
| `refusing to start X; other backends still running: Y` | a non-resident backend survived a stop | check `journalctl` for `BackendStopError`; usually a launcher that re-parents workers — set a larger `stop_timeout_seconds` |
| model list shows a llama preset you deleted | cached metadata from `llama_models_snapshot` | delete the snapshot file, or keep it — it only feeds modalities for non-selector IDs |
| a switch hangs, then `timed out waiting for active request(s) to drain` | a client is streaming longer than `drain_timeout_seconds` | raise the timeout, or fix the client that never closes the stream |
| cold loads take minutes every time | normal for multi-tens-of-GB weights; also check page cache warmth | raise `startup_timeout_seconds`, keep the model resident (`idle_ttl_seconds: 0`), or enable OS-level caching |
| logs are empty | backend writes to its own log file, not stdout | look at `routerctl.py logs <backend>` for router-owned output, and the engine's own log path |

Router-owned logs: `settings.state_dir/logs/<backend>.log` (a marker line is
written at each start, then the child's stdout+stderr).
