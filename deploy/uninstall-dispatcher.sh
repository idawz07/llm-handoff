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
rm -f "$TARGET"
mv "$REAL" "$TARGET"
printf 'Restored %s\n' "$TARGET"
