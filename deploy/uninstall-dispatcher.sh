#!/usr/bin/env bash
# Remove the dispatcher shim and restore the real llama.cpp binary.
set -euo pipefail

ROOT="${HANDOFF_LLAMA_ROOT:-${LLAMA_SERVER_ROOT:-$HOME/llama.cpp}}"
TARGET="$ROOT/build/bin/llama-server"
REAL="$ROOT/build/bin/llama-server.real"

if ! grep -aq 'HANDOFF_ROUTER_DISPATCHER' "$TARGET" 2>/dev/null; then
  echo "no Handoff Router dispatcher is installed at $TARGET" >&2
  exit 1
fi
[[ -x "$REAL" ]] || { echo "real llama.cpp binary is missing: $REAL" >&2; exit 1; }
if grep -aEq 'LLAMA_MULTI_BACKEND_DISPATCHER|HANDOFF_ROUTER_DISPATCHER' "$REAL"; then
  echo "refusing to restore a dispatcher as the real server: $REAL (check .real.previous.* backups)" >&2
  exit 1
fi
mv -f -- "$REAL" "$TARGET"
WRAPPER="$TARGET.handoff-dispatcher"
if [[ -f "$WRAPPER" ]] && grep -aq 'HANDOFF_ROUTER_DISPATCHER' "$WRAPPER"; then
  rm -f -- "$WRAPPER"
fi
printf 'Restored %s\n' "$TARGET"
