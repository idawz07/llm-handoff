#!/usr/bin/env bash
# Install a safe shim over llama-server without changing an existing systemd unit.
# Never move an already-installed dispatcher into llama-server.real: that would
# make ordinary llama-server invocations recurse into the old dispatcher.
set -euo pipefail

ROOT="${HANDOFF_LLAMA_ROOT:-${LLAMA_SERVER_ROOT:-$HOME/llama.cpp}}"
TARGET="$ROOT/build/bin/llama-server"
REAL="$ROOT/build/bin/llama-server.real"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${HANDOFF_DISPATCHER:-$HERE/dispatcher.sh}"

[[ -x "$TARGET" ]] || { echo "missing executable: $TARGET" >&2; exit 1; }
[[ -x "$WRAPPER" ]] || { echo "missing executable dispatcher: $WRAPPER" >&2; exit 1; }
# The installer receives these values once. The generated shim must persist
# them because systemd does NOT inherit an install command's environment.
: "${HANDOFF_ROUTER_CONFIG:?set HANDOFF_ROUTER_CONFIG to the absolute YAML path}"
: "${HANDOFF_ROUTER_PRESET:?set HANDOFF_ROUTER_PRESET to the unit --models-preset path}"
CONFIG="$(realpath -m -- "$HANDOFF_ROUTER_CONFIG")"
PRESET="$(realpath -m -- "$HANDOFF_ROUTER_PRESET")"
[[ -f "$CONFIG" ]] || { echo "router config not found: $CONFIG" >&2; exit 1; }
[[ -f "$PRESET" ]] || { echo "llama.cpp preset file not found: $PRESET" >&2; exit 1; }
PORT="${HANDOFF_ROUTER_PORT:-8081}"
HOST="${HANDOFF_ROUTER_HOST:-0.0.0.0}"
BIN="${HANDOFF_ROUTER_BIN:-$(command -v handoff-router || printf 'handoff-router')}"
RUNNER="${HANDOFF_ROUTER_RUNNER:-}"
if [[ -n "$RUNNER" ]]; then
  [[ -x "$RUNNER" ]] || { echo "router runner not executable: $RUNNER" >&2; exit 1; }
elif ! command -v "$BIN" >/dev/null 2>&1; then
  echo "handoff-router CLI not found: $BIN (set HANDOFF_ROUTER_BIN or HANDOFF_ROUTER_RUNNER)" >&2
  exit 1
fi
WEB_UI="${HANDOFF_WEB_UI_DIR:-}"
if [[ -n "$WEB_UI" ]]; then
  [[ -d "$WEB_UI" ]] || { echo "web UI directory not found: $WEB_UI" >&2; exit 1; }
  WEB_UI="$(realpath -m -- "$WEB_UI")"
fi

if grep -aq 'HANDOFF_ROUTER_DISPATCHER' "$TARGET" 2>/dev/null; then
  [[ -x "$REAL" ]] || { echo "dispatcher installed but real binary missing: $REAL" >&2; exit 1; }
  if grep -aEq 'LLAMA_MULTI_BACKEND_DISPATCHER|HANDOFF_ROUTER_DISPATCHER' "$REAL"; then
    echo "real binary is another dispatcher: $REAL; restore it from a .previous.* backup first" >&2
    exit 1
  fi
  echo "Updating existing Handoff Router dispatcher"
elif grep -aq 'LLAMA_MULTI_BACKEND_DISPATCHER' "$TARGET" 2>/dev/null; then
  [[ -x "$REAL" ]] || { echo "legacy dispatcher found, but real binary missing: $REAL" >&2; exit 1; }
  if grep -aEq 'LLAMA_MULTI_BACKEND_DISPATCHER|HANDOFF_ROUTER_DISPATCHER' "$REAL"; then
    echo "refusing to use another dispatcher as the real llama-server: $REAL" >&2
    exit 1
  fi
  backup="$TARGET.legacy.$(date +%Y%m%d-%H%M%S).$$"
  cp -p -- "$TARGET" "$backup"
  echo "Preserved legacy shim at $backup; keeping the original $REAL untouched"
else
  # A fresh llama.cpp build has replaced the shim. Keep its new binary at .real
  # and retain the previous binary as a rollback copy if one exists.
  if [[ -e "$REAL" ]]; then
    backup="$REAL.previous.$(date +%Y%m%d-%H%M%S).$$"
    mv -- "$REAL" "$backup"
    echo "Preserved previous binary at $backup"
  fi
  cp -p -- "$TARGET" "$REAL.staging.$$"
  mv -- "$REAL.staging.$$" "$REAL"
fi

shim="$(mktemp "$TARGET.handoff.XXXXXXXX")"
trap 'rm -f -- "$shim"' EXIT
{
  printf '#!/usr/bin/env bash\n# HANDOFF_ROUTER_DISPATCHER\n'
  printf 'export HANDOFF_SERVER_REAL=%q\n' "$REAL"
  printf 'HANDOFF_DEFAULT_CONFIG=%q\n' "$CONFIG"
  printf 'HANDOFF_DEFAULT_PRESET=%q\n' "$PRESET"
  printf 'HANDOFF_DEFAULT_PORT=%q\n' "$PORT"
  printf 'HANDOFF_DEFAULT_HOST=%q\n' "$HOST"
  printf 'HANDOFF_DEFAULT_BIN=%q\n' "$BIN"
  printf 'HANDOFF_DEFAULT_RUNNER=%q\n' "$RUNNER"
  printf 'HANDOFF_DEFAULT_WEB_UI=%q\n' "$WEB_UI"
  printf 'export HANDOFF_WEB_UI_DIR="${HANDOFF_WEB_UI_DIR:-$HANDOFF_DEFAULT_WEB_UI}"\n'
  printf 'export HANDOFF_ROUTER_CONFIG="${HANDOFF_ROUTER_CONFIG:-$HANDOFF_DEFAULT_CONFIG}"\n'
  printf 'export HANDOFF_ROUTER_PRESET="${HANDOFF_ROUTER_PRESET:-$HANDOFF_DEFAULT_PRESET}"\n'
  printf 'export HANDOFF_ROUTER_PORT="${HANDOFF_ROUTER_PORT:-$HANDOFF_DEFAULT_PORT}"\n'
  printf 'export HANDOFF_ROUTER_HOST="${HANDOFF_ROUTER_HOST:-$HANDOFF_DEFAULT_HOST}"\n'
  printf 'export HANDOFF_ROUTER_BIN="${HANDOFF_ROUTER_BIN:-$HANDOFF_DEFAULT_BIN}"\n'
  printf 'export HANDOFF_ROUTER_RUNNER="${HANDOFF_ROUTER_RUNNER:-$HANDOFF_DEFAULT_RUNNER}"\n'
  printf 'exec %q "$@"\n' "$WRAPPER"
} > "$shim"
chmod +x "$shim"
mv -f -- "$shim" "$TARGET"
trap - EXIT
printf 'Installed dispatcher at %s (real binary: %s)\n' "$TARGET" "$REAL"
printf 'Config: %s\nMatched preset: %s\nNow restart the service during a planned window.\n' "$CONFIG" "$PRESET"
