# Security policy

**Report privately:** open a private security advisory on the repository, or email
the maintainers listed there. Please do not open a public issue first.

## Threat model

Handoff Router is a local/administrative tool, and it is *root-equivalent on its
own host* by design: it launches arbitrary commands from
`router-backends.yaml`, reads arbitrary files given a path, and can SIGTERM or
SIGKILL process groups.

Assume:

- the config file and the process environment are trusted input, already;
- the API key is equivalent to local code execution, because
  `POST /router/reload` + a writable config = command execution;
- anything able to reach the facade port can change which model owns the GPU.

## Defaults

- `/health`, `/v1/health`, `/models`, `/v1/models` and the optional web UI shell
  are unauthenticated (llama.cpp compatibility). Everything else, including
  `/models/load`, `/models/unload`, `/router/*`, `/logs/*`, and `/activity/*`,
  requires the API key when one is configured.
- Backends are expected to bind `127.0.0.1`; only the facade is exposed.
- `--api-key-file` / `HANDOFF_ROUTER_API_KEY` keep secrets out of the unit file
  and shell history. Use `--api-key-file` with a `0640` root-owned file.

## Hardening checklist

1. Bind the facade to a tailnet/VLAN interface, or terminate TLS in a reverse
   proxy that adds auth at the edge.
2. Run as a dedicated user in the `video` group; models and config readable only
   by it; `router-backends.yaml` mode `0600` if any operator has shell access.
3. Keep the cgroup memory drop-in installed so a compromised or buggy engine
   cannot exhaust the host.
4. Do not expose `/router/reload` to untrusted networks.
