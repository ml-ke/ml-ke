---
name: mistral-vibe
description: "Configure and drive the Mistral Vibe CLI (vibe)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, Mistral, Vibe, Configuration, API-Key]
    related_skills: [opencode, codex, claude-code, hermes-agent]
---

# Mistral Vibe CLI (mistral-vibe)

Mistral's agentic coding CLI (`vibe`), installed on this machine as a uv tool at
`~/.local/share/uv/tools/mistral-vibe/` with three binaries: `vibe`, `vibe-acp`,
`vibe-app-server`. Use it as a coding delegate the same way as `opencode`/`codex`,
and use this skill when the task is about its **provider, model, or API-key
configuration**.

## File layout (get this right before editing anything)

| Path | What it is |
|------|------------|
| `~/.vibe/config.toml` | Provider/model/`active_model`, theme, `[tools.bash].allowlist`, `applied_migrations` |
| `~/.vibe/.env` | Global secrets file (GLOBAL_ENV_FILE). Loaded at startup by every entrypoint, so keys here work in cron/non-login shells too |
| `~/.vibe/trusted_folders.toml` | Per-directory trust list |
| `~/.vibe/whoami_cache.json` | Cached account state: `api_base`, `plan_name`, `vibe_base` |
| `~/.vibe/logs/vibe.log` | Runtime log — the proof of which provider/model actually served a call |
| `~/.vibe/logs/session/<id>/meta.json` + `messages.jsonl` | Per-session resolved config and transcript |

Installed package source (read it when config semantics are unclear):
`~/.local/share/uv/tools/mistral-vibe/lib/python3.12/site-packages/vibe/core/config/`
(`vibe_schema.py` holds `DEFAULT_PROVIDERS`, `DEFAULT_MODELS`, merge rules;
`_defaults.py` holds path/URL/env-var constants).

## Mistral-native defaults

- Provider `mistral` → `https://api.mistral.ai/v1`, `api_key_env_var = "MISTRAL_API_KEY"`, `backend = "mistral"`
- Default model: name `mistral-vibe-cli-latest`, alias `mistral-medium-3.5` (priced $1.5/$7.5 per M tokens)
- Cheaper alternatives selectable as `active_model`: `mistral-vibe-cli-fast`, `ministral-8b-latest`, `ministral-3b-latest`
- Auth is either the API key or an interactive browser sign-in; `whoami_cache.json` shows which

## Key resolution order

1. Non-empty **process/shell env var** wins
2. Otherwise the value is read from `~/.vibe/.env`
3. Otherwise the CLI reports the key as missing

Never write the secret into `config.toml` — it is visible in dotfile listings and
gets copied into bug reports. Put it in `~/.vibe/.env`, `chmod 600`.

## Re-pointing the CLI at a provider (the common ask)

Adding a `[[providers]]` block to `config.toml` (e.g. an OpenRouter entry with
`api_style = "openai"`, `backend = "generic"`) plus its `[[models]]` entry and a
matching pinned `active_model` **redirects every call away from the Mistral API**.
That is how a Vibe install silently ends up running a third-party free model.
The fix is not just flipping `active_model`:

1. `cp ~/.vibe/config.toml ~/.vibe/config.toml.bak.<provider>.$(date +%Y%m%d_%H%M%S)`
2. Delete the foreign `[[providers]]` block **and** its `[[models]]` block.
   Providers merge by `name` (union) and models deep-merge by `alias`, so a
   stale provider entry survives a partial edit and keeps winning.
3. Write the Mistral provider explicitly and pin `active_model` to the Mistral
   alias — pinning also defeats GrowthBook `routed_default_model` re-routing,
   which only applies while `active_model` is unpinned.
4. Put `MISTRAL_API_KEY` in `~/.vibe/.env` (`chmod 600`).
5. Verify (below) — do not declare success from the config alone.

Known-good file: `templates/vibe-config-mistral.toml`.

## Step 1 — Find a key before asking for one

The user's projects usually already contain a working Mistral key. Check project
`.env` files before asking:

```bash
# Key-name inventory across known projects (fast, bounded)
for f in ~/ProG/*/.env; do grep -oE '^[A-Z0-9_]+=' "$f" | grep -iE 'mistral|api_key'; done
# Confirm a value exists without printing it
awk -F= '/^MISTRAL_API_KEY=/{v=$2; gsub(/["\r]/,"",v); print length(v), substr(v,1,4)}' ~/ProG/<proj>/.env
```

Mistral API keys are 32-char alphanumeric. **Bound your search to explicit
directories** — a recursive grep over `~/ProG ~/Dev` times out because of
node_modules/.venv trees.

Also check `~/.hermes/.env`, but note that a commented-out entry there is inert:
a key name appearing in a file is not a key value.

## Step 2 — Verify the key live before wiring it in

```bash
cd ~/ProG/<proj> && set -a && . ./.env && set +a
curl -s -o /tmp/mistral_models.json -w "http=%{http_code}\n" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" https://api.mistral.ai/v1/models
```

HTTP 200 with ~50 model IDs = key is live. 401 = bad key; 402 = no credits (see
pitfalls). `/v1/models` proves **authentication only, not quota**.

## Step 3 — Write the secret without echoing it

```bash
cd ~/ProG/<proj> \
  && KEY=$(awk -F= '/^MISTRAL_API_KEY=/{v=$2} END{gsub(/["\r]/,"",v); print v}' .env) \
  && printf 'MISTRAL_API_KEY=%s\n' "$KEY" > ~/.vibe/.env \
  && chmod 600 ~/.vibe/.env \
  && awk -F= '/^MISTRAL_API_KEY=/{print "stored_len="length($2)}' ~/.vibe/.env
```

Report the length/prefix, never the value.

## Step 4 — Verify the switch with real traffic

```bash
cd /tmp && timeout 180 vibe -p "Reply with exactly: VIBE-OK" --max-turns 1 --output text
```

Programmatic mode needs no pty. Then confirm **where** the call went:

```bash
tail -5 ~/.vibe/logs/vibe.log
# expect: INFO Model call completed model=mistral-medium-3.5 ... prompt_tokens=...
grep -oE '"provider": "[a-z]+"|api\.mistral\.ai' ~/.vibe/logs/session/<newest>/meta.json | sort | uniq -c
```

A `Model call completed` line carrying token counts is the proof; the config file
and a `--help` run are not.

## Pitfalls

- **402 Payment Required on premium models.** A free/api-plan account can
authenticate fine and still be refused on `mistral-medium-3.5`. Probe with a real
completion, not `/v1/models`, and keep a fallback ready: swap `active_model` to
`mistral-vibe-cli-fast` or `ministral-8b-latest` (one line, no provider change).
- **Partial provider edits don't take.** Providers union-merge by name and models
deep-merge by alias, so removing only one of the two blocks leaves the old route
alive and the CLI keeps using the old provider.
- **`active_model` unpinned means "route decides"**, which may not be Mistral. Pin
the alias whenever the requirement is "must use <provider>".
- **`vibe.log` is append-only across versions**, so it holds historical provider
traffic; judge by the newest timestamp, not by grep counts.
- **Keep terminal payloads small.** Long single-line shell commands with nested
quotes (awk + grep + curl in one string) trip Hermes' inline-command parser and
come back as a hardline block. Split into short commands, or write a `.sh` and
run it.
- TUI sessions need a pty; `vibe -p` does not.

## Rules

1. A config change is not done until a real completion succeeds and `vibe.log`
   shows the intended model — verify, don't assume.
2. Always back up `config.toml` before rewriting it.
3. Secrets live in `~/.vibe/.env` (600), never in `config.toml`, never printed.
4. Reuse an existing working key from the user's projects instead of asking for a
   new one; ask only when no live key exists.
5. After a provider/model switch, record the working model and any quota limits in
   the lesson bank (`~/Dev/ATLAS-LEARNINGS/LESSONS.md` §04).
