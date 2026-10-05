#!/usr/bin/env bash
# Install the dispatcher shim over an existing llama.cpp router install without
# touching the service unit or its API key. Idempotent: rerun it after every
# llama.cpp rebuild, which replaces build/bin/llama-server with a fresh binary.
set -euo pipefail

ROOT="${HANDOFF_LLAMA_ROOT:-${LLAMA_SERVER_ROOT:-$HOME/llama.cpp}}"
TARGET="$ROOT/build/bin/llama-server"
REAL="$ROOT/build/bin/llama-server.real"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="${HANDOFF_DISPATCHER:-$HERE/dispatcher.sh}"

[[ -x "$TARGET" ]] || { echo "missing llama-server: $TARGET" >&2; exit 1; }
[[ -f "$WRAPPER" ]] || { echo "missing dispatcher: $WRAPPER" >&2; exit 1; }
chmod +x "$WRAPPER"

if grep -aq 'HANDOFF_ROUTER_DISPATCHER' "$TARGET" 2>/dev/null; then
  [[ -x "$REAL" ]] || { echo "dispatcher installed but real binary is missing: $REAL" >&2; exit 1; }
  printf 'Dispatcher already installed at %s\n' "$TARGET"
else
  if [[ -e "$REAL" ]]; then
    mv -f "$REAL" "$REAL.previous.$(date +%Y%m%d-%H%M%S)"
  fi
  mv "$TARGET" "$REAL"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# HANDOFF_ROUTER_DISPATCHER\n'
    printf 'export HANDOFF_SERVER_REAL=%q\n' "$REAL"
    printf 'exec %q "$@"\n' "$WRAPPER"
  } > "$TARGET"
  chmod +x "$TARGET"
  printf 'Installed dispatcher at %s (real binary moved to %s)\n' "$TARGET" "$REAL"
fi

cat <<EOF

Next:
  1. Point your config at this install:
       HANDOFF_ROUTER_CONFIG=\$HOME/.config/handoff-router/router-backends.yaml
     and set HANDOFF_ROUTER_PRESET to the --models-preset path your unit already uses.
  2. Restart your existing service, e.g.: systemctl restart llama-router
  3. Verify:  curl -H "Authorization: Bearer \$KEY" localhost:8081/router/status
EOF
