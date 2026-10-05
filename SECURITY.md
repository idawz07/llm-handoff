# Security policy

**Report privately:** open a private security advisory on the repository, or email
the maintainers listed there. Please do not open a public issue first.

## Threat model

Handoff Router is a local/administrative tool with the privileges of its
**service account**: it launches commands from `router-backends.yaml`, reads
files available to that account, and can signal its adopted process groups.
Never run it as root unnecessarily.

Assume:

- the config file and the process environment are trusted input, already;
- the API key grants model load/unload and control-plane access; if an attacker
  can also modify the trusted config, `/router/reload` can execute commands as
  the router's service user;
- without an API key, anyone who can reach the listener can switch models.

## Defaults

- `/health`, `/v1/health`, `/models`, `/v1/models` and optional static web UI
  assets are public. Inference, `/props` (which can start a model), load/unload,
  logs and control endpoints all require a key when configured.
- The facade defaults to `127.0.0.1`; a non-loopback bind requires an API key
  unless explicitly overridden with `--allow-unauthenticated-remote`.
- Adoption requires a nonempty `process_match` fingerprint and the same UID.
  Backends must bind loopback; only the authenticated facade is exposed.
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
