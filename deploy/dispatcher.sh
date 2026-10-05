#!/usr/bin/env bash
# HANDOFF_ROUTER_DISPATCHER
# A launcher shim for people who already run llama.cpp's router under a service
# manager. It preserves the existing binary path, unit file, and API key: only
# the exact router invocation enters Handoff Router, every ordinary
# llama-server invocation delegates byte-for-byte to the real binary.
#
# Installed by deploy/install-dispatcher.sh, removed by deploy/uninstall-dispatcher.sh.
set -euo pipefail

REAL="${HANDOFF_SERVER_REAL:-${LLAMA_SERVER_REAL:-$HOME/llama.cpp/build/bin/llama-server.real}}"
CONFIG="${HANDOFF_ROUTER_CONFIG:-$HOME/.config/handoff-router/router-backends.yaml}"
PRESET="${HANDOFF_ROUTER_PRESET:-$HOME/.config/handoff-router/router-presets.ini}"
PORT="${HANDOFF_ROUTER_PORT:-8081}"
HOST="${HANDOFF_ROUTER_HOST:-0.0.0.0}"
RUNNER="${HANDOFF_ROUTER_RUNNER:-}"

[[ -x "$REAL" ]] || {
  echo "handoff-router dispatcher: real server binary not found: $REAL" >&2
  echo "   (set HANDOFF_SERVER_REAL, or run deploy/install-dispatcher.sh)" >&2
  exit 127
}

# Escape hatch: run the untouched llama-server even for the router invocation.
if [[ "${HANDOFF_ROUTER_BYPASS:-0}" == "1" || "${LLAMA_BACKEND_ROUTER_BYPASS:-0}" == "1" ]]; then
  exec "$REAL" "$@"
fi

preset=""
port=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --models-preset)
      ((i + 1 < ${#args[@]})) && preset="${args[$((i + 1))]}"
      ;;
    --port)
      ((i + 1 < ${#args[@]})) && port="${args[$((i + 1))]}"
      ;;
  esac
done

if [[ "$preset" == "$PRESET" && "${port:-$PORT}" == "$PORT" ]]; then
  [[ -f "$CONFIG" ]] || { echo "handoff-router dispatcher: config missing: $CONFIG" >&2; exit 78; }
  if [[ -n "$RUNNER" ]]; then
    exec "$RUNNER" --config "$CONFIG" --host "$HOST" --port "$PORT" "$@"
  fi
  exec "${HANDOFF_ROUTER_BIN:-handoff-router}" --config "$CONFIG" --host "$HOST" --port "$PORT" "$@"
fi

exec "$REAL" "$@"
