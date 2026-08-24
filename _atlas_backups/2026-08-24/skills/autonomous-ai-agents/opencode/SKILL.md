---
name: opencode
description: "Delegate coding to OpenCode CLI. Includes free-model picker + auto-fallback coder wrapper, Hermes skill bridge (atlas-*), custom subagents, and skill validation/scanning (Aug 2026)."
version: 1.4.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, OpenCode, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, hermes-agent]
---

# OpenCode CLI

Use [OpenCode](https://opencode.ai) as an autonomous coding worker orchestrated by Hermes terminal/process tools. OpenCode is a provider-agnostic, open-source AI coding agent with a TUI and CLI.

## Division of Labor (Hermes + OpenCode Architecture)

**Hermes/DeepSeek (you) = planner + minor tasks.** You do:
- Task decomposition and planning
- Quick scripts, curl calls, recon, parsing, JSON munging
- Orchestration, verification, report writing (humanized per the humanizer skill)

**OpenCode = heavy coding worker.** Delegate to OpenCode when:
- The task is a bounded coding task (implement feature, fix bug, write module, refactor)
- The code is non-trivial (100+ lines, multiple files, algorithm-heavy)
- The task is well-specified enough to hand off ("write X that does Y with deps Z")
- You need a long-running coding session without burning your own context

**Do NOT delegate to OpenCode when:**
- A single quick script (< 30 lines) — just write it yourself
- Tasks needing your conversation context (memory, skills, past sessions)
- Security-critical logic where you need to review every line (write it yourself, then optionally have OpenCode review)

## Free Model Availability (Critical — Aug 2026)

**RULE #1: Docs lie. Test to confirm.** The docs page may list a model as available, but:
- Models get disabled/removed without docs updates (deepseek-v4-flash-free is documented but DISABLED)
- Rate limits, quotas, and endpoint health change hour to hour
- The ONLY truth is a live smoke test

**NEVER trust**: the docs page, a blog post, a search result, yesterday's memory, or a cached list — unless the cache is fresh (< 6h) AND was built from live tests.

**ALWAYS**: probe with the picker or let `opencode_coder.py` auto-pick (it live-tests). If a model fails mid-task, fall back to the next live model.

OpenCode Zen is authenticated with an API key. **Paid models require billing — they are NOT usable** ("No payment method" error). Only **free models** work:

Working free models (verified by LIVE TEST Aug 2026 — may change):
- `opencode/mimo-v2.5-free` — **default, best quality**
- `opencode/nemotron-3-ultra-free`
- `opencode/ling-3.0-flash-free`
- `opencode/laguna-s-2.1-free`
- `opencode/north-mini-code-free`

Disabled models (change over time — always re-probe):
- `opencode/deepseek-v4-flash-free` was disabled as of Aug 2026 (still listed in docs)

**Model availability changes.** Before delegating, ALWAYS live-test which free models currently work.

## The Model Picker (Automated)

Two helper scripts ship with Hermes:

### `~/.hermes/scripts/opencode_model_picker.py`
Probes OpenCode for working free models:
```bash
python3 ~/.hermes/scripts/opencode_model_picker.py --quick   # test known-free models (~1 min)
python3 ~/.hermes/scripts/opencode_model_picker.py --json    # machine-readable
# Prints working model IDs, one per line (or JSON array with --json)
```

### `~/.hermes/scripts/opencode_coder.py` (RECOMMENDED — use this)
One-shot coding delegate that auto-picks a working model and falls back if it dies:
```bash
python3 ~/.hermes/scripts/opencode_coder.py "Implement X in repo Y" --dir ~/Dev/project
python3 ~/.hermes/scripts/opencode_coder.py "Fix bug in auth.py" --dir ~/Dev/project --file auth.py
python3 ~/.hermes/scripts/opencode_coder.py --smoke    # self-test: prints SMOKE_OK <model> (ALWAYS live-tests)
python3 ~/.hermes/scripts/opencode_coder.py "TASK" --dir ~/Dev/project --fresh   # bypass cache, force live re-test
```
- Caches working models for 6h in `~/.hermes/scripts/.opencode_models_cache`
- Tries models in preference order, removes failed models from cache
- Flags: `--dir <path>` (workdir), `--file <path>` (attach, repeatable), `--json` (JSON events), `--model <id>` (force), `--fresh` (bypass cache, live-test now)
- **Use `--fresh` before critical runs** — the 6h cache is only as good as the last live test; models can die between runs.

**This is the default way to call OpenCode for coding tasks.** Use it unless you need interactive mode.

## When to Use

- User explicitly asks to use OpenCode
- You want an external coding agent to implement/refactor/review code
- You need long-running coding sessions with progress checks
- You want parallel task execution in isolated workdirs/worktrees
- **You are implementing, evaluating, or planning improvements to a system** — load `opencode-principles` (theory mapped to orchestration) and `system-design-theory` (master KB: CAP, resilience patterns, evaluation framework) FIRST so architecture decisions follow system design principles and evaluations use the six-pillar review, not vibes.

## Prerequisites

- OpenCode installed: `npm i -g opencode-ai@latest` or `brew install anomalyco/tap/opencode`
- Auth configured: `opencode auth login` or set provider env vars (OPENROUTER_API_KEY, etc.)
- Verify: `opencode auth list` should show at least one provider
- Git repository for code tasks (recommended)
- `pty=true` for interactive TUI sessions

## Binary Resolution (Important)

Shell environments may resolve different OpenCode binaries. If behavior differs between your terminal and Hermes, check:

```
terminal(command="which -a opencode")
terminal(command="opencode --version")
```

If needed, pin an explicit binary path:

```
terminal(command="$HOME/.opencode/bin/opencode run '...'", workdir="~/project", pty=true)
```

## One-Shot Tasks

Use `opencode run` for bounded, non-interactive tasks:

```
terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

Attach context files with `-f`:

```
terminal(command="opencode run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

Show model thinking with `--thinking`:

```
terminal(command="opencode run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

Force a specific model:

```
terminal(command="opencode run 'Refactor auth module' --model openrouter/anthropic/claude-sonnet-4", workdir="~/project")
```

## Interactive Sessions (Background)

For iterative work requiring multiple exchanges, start the TUI in background:

```
terminal(command="opencode", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send a prompt
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send follow-up input
process(action="submit", session_id="<id>", data="Now add error handling for token expiry")

# Exit cleanly — Ctrl+C
process(action="write", session_id="<id>", data="\x03")
# Or just kill the process
process(action="kill", session_id="<id>")
```

**Important:** Do NOT use `/exit` — it is not a valid OpenCode command and will open an agent selector dialog instead. Use Ctrl+C (`\x03`) or `process(action="kill")` to exit.

### TUI Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Submit message (press twice if needed) |
| `Tab` | Switch between agents (build/plan) |
| `Ctrl+P` | Open command palette |
| `Ctrl+X L` | Switch session |
| `Ctrl+X M` | Switch model |
| `Ctrl+X N` | New session |
| `Ctrl+X E` | Open editor |
| `Ctrl+C` | Exit OpenCode |

### Resuming Sessions

After exiting, OpenCode prints a session ID. Resume with:

```
terminal(command="opencode -c", workdir="~/project", background=true, pty=true)  # Continue last session
terminal(command="opencode -s ses_abc123", workdir="~/project", background=true, pty=true)  # Specific session
```

## Common Flags

| Flag | Use |
|------|-----|
| `run 'prompt'` | One-shot execution and exit |
| `--continue` / `-c` | Continue the last OpenCode session |
| `--session <id>` / `-s` | Continue a specific session |
| `--agent <name>` | Choose OpenCode agent (build or plan) |
| `--model provider/model` | Force specific model |
| `--format json` | Machine-readable output/events |
| `--file <path>` / `-f` | Attach file(s) to the message |
| `--thinking` | Show model thinking blocks |
| `--variant <level>` | Reasoning effort (high, max, minimal) |
| `--title <name>` | Name the session |
| `--attach <url>` | Connect to a running opencode server |

## Procedure

1. Verify tool readiness:
   - `terminal(command="opencode --version")`
   - `terminal(command="opencode auth list")` — must show at least one provider (Zen key)
2. **Pick a working model** (free models only — paid need billing):
   - Fast path: `python3 ~/.hermes/scripts/opencode_coder.py --smoke`
   - Full probe: `python3 ~/.hermes/scripts/opencode_model_picker.py --quick`
   - Or just call `opencode_coder.py` directly — it auto-picks.
3. For bounded coding tasks, use `opencode_coder.py` (one-shot, auto-model):
   `python3 ~/.hermes/scripts/opencode_coder.py "TASK" --dir ~/project --file file1.py`
4. For tasks needing the raw CLI: `opencode run '...' --model opencode/<free-model-id>` (no pty needed).
5. For iterative tasks, start `opencode` with `background=true, pty=true`.
6. Monitor long tasks with `process(action="poll"|"log")`.
7. If OpenCode asks for input, respond via `process(action="submit", ...)`.
8. Exit with `process(action="write", data="\x03")` or `process(action="kill")`.
9. **Verify the output** — read the generated files, run tests, confirm the code actually works before reporting success.
10. Summarize file changes, test results, and next steps back to user.

## PR Review Workflow

OpenCode has a built-in PR command:

```
terminal(command="opencode pr 42", workdir="~/project", pty=true)
```

Or review in a temporary clone for isolation:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')", pty=true)
```

## Parallel Work Pattern

Use separate workdirs/worktrees to avoid collisions:

```
terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## Session & Cost Management

List past sessions:

```
terminal(command="opencode session list")
```

Check token usage and costs:

```
terminal(command="opencode stats")
terminal(command="opencode stats --days 7 --models anthropic/claude-sonnet-4")
```

## Pitfalls

- Interactive `opencode` (TUI) sessions require `pty=true`. The `opencode run` command does NOT need pty.
- `/exit` is NOT a valid command — it opens an agent selector. Use Ctrl+C to exit the TUI.
- PATH mismatch can select the wrong OpenCode binary/model config.
- **Paid models fail with "No payment method"** — do not use opencode/gpt-*, opencode/claude-*, opencode/deepseek-v4-pro, etc. unless the user has added billing. Free models only (`*-free`).
- **Model availability changes** — a model that worked yesterday may be disabled today (deepseek-v4-flash-free was disabled Aug 2026). Always probe with the picker before a critical run, and use `opencode_coder.py` so failures auto-fallback. When model availability changes, record it in the lesson bank (`~/Dev/ATLAS-LEARNINGS/LESSONS.md` §02) so future sessions know without re-probing.
- **Docs are not proof of availability** — the docs page lists deepseek-v4-flash-free as available but it's disabled. A model being "documented" means nothing. Only a live smoke test (`--smoke` or `--fresh`) confirms a model works right now.
- Free model cold-start can take 30s+ (first request warms the endpoint). Don't assume hang.
- `opencode run` output is NOT the agent's full transcript — it's the final message. For file verification, read the files it wrote.
- If OpenCode appears stuck, inspect logs before killing:
  - `process(action="log", session_id="<id>")`
- Avoid sharing one working directory across parallel OpenCode sessions.
- Enter may need to be pressed twice to submit in the TUI (once to finalize text, once to send).
- **Always verify generated code actually runs** — OpenCode free models occasionally produce plausible-but-broken code. Run the file, check for syntax errors, exercise the happy path.

## Agent Skills Engineering (Best Practices — researched Aug 2026)

OpenCode supports progressive-disclosure agent skills: only name+description is injected at startup; the full SKILL.md body loads on demand via the skill tool. This makes large libraries cheap to install — but curation still matters.

### Skill file layout

- One folder per skill: `~/.config/opencode/skills/<name>/SKILL.md`
- OpenCode also scans `.opencode/skills/`, `.claude/skills/`, `.agents/skills/` (project + global)
- Only 5 frontmatter fields are recognized: `name`, `description`, `license`, `compatibility`, `metadata`. Unknown fields are ignored.
- **`name` must match the directory name** and match `^[a-z0-9]+(-[a-z0-9]+)*$` — mismatch silently breaks discovery
- `description` must be 1–1024 chars, specific enough for the agent to pick correctly
- Keep long material in sibling `references/`, `scripts/`, `templates/` — loaded on demand

### Permission-gating skills

```jsonc
{
  "permission": {
    "skill": {
      "*": "allow",
      "internal-*": "deny",
      "experimental-*": "ask"
    }
  }
}
```

Patterns support wildcards. Override per custom agent in its frontmatter; per built-in agent via `agent.<name>.permission.skill` in opencode.json.

### Custom agents (3 installed on this machine — see Setup reality check below)

Define in `~/.config/opencode/agents/<name>.md` (global) or `.opencode/agents/` (project). The filename becomes the agent name. **1.18.x frontmatter (live-verified Aug 2026):** `description`, `mode: primary|subagent`, `model`, `temperature`, `tools` (a map `"*": false` + explicit allows — the old `permission: {edit: deny, bash: deny}` block is DEPRECATED and agents written that way silently fail to register), `color`, plus the body as the system prompt. Verify registration with `opencode agent list`.

Good defaults for a security-auditor subagent (current format):
```markdown
---
mode: subagent
model: opencode/mimo-v2.5-free
temperature: 0.1
tools:
  "*": false
  read: true
  glob: true
  grep: true
  webfetch: true
  skill: true
color: "#E74C3C"
---
You are a security auditor. Load the relevant tob-* / atlas-* skill before reviewing.
Report findings with severity and remediation. Never edit files.
```

Pitfalls: the `opencode agent create` CLI rejected a positional name arg in 1.18.15 (usage dump, nothing written) — writing the .md file directly works reliably. Agents need the `tools:` map, not `permission:`; a `model:` field pinned to a free model (e.g. `opencode/mimo-v2.5-free`) is fine since paid models fail with "No payment method".

### Setup reality check (this machine, Aug 2026)

- 194 skills installed: 177 from 9 repos (tob-* 77 Trail of Bits security, flutter 22, expo 21, anthropic 17, dart 12, android 12, vercel 9, google 5, railway 2) + **17 atlas-\* bridged from the Hermes library** (security methodology: recon-to-exploitation, idor, jwt, saml, oauth, ssrf, mass-assignment, business-logic, chaining, pre-submission-verification, api-hacking, api-bug-bounty, continuous-learning, source-code-audit, code-review, bugcrowd-vrt, supabase-studio)
- The bridge is `~/.hermes/scripts/opencode_skills_bridge.py` — syncs curated Hermes skills into OpenCode format (name==dir, block-scalar description, progressive-disclosure refs). Re-run after any Hermes methodology skill changes: `python3 ~/.hermes/scripts/opencode_skills_bridge.py`
- **3 custom subagents** in `~/.config/opencode/agents/`: `security-auditor` (read-only, no bash), `pentest-recon` (bash + atlas-* skills, live probing), `code-reviewer` (read-only + read-only bash). Use the current 1.18.x agent format: `tools: {"*": false, "read": true, ...}` — NOT the old `permission:` block.
- **Skill permissions** in opencode.jsonc: fuzzing skills (tob-fuzz*, tob-cargo-fuzz, tob-libfuzzer, tob-ossfuzz) = ask; everything else allow. Per-agent model overrides for build (mimo-v2.5-free) and plan (nemotron-3-ultra-free).
- **Skill quality**: `~/.hermes/scripts/skill_validator.py` audits frontmatter (name regex, name==dir, description 1-1024, body <5000 words). Run against both libraries after any skill change.
- **Skill security**: SkillSpector 2.8.2 + snyk-agent-scan in `~/.hermes/venvs/skillsec/`. Scan: `~/.hermes/venvs/skillsec/bin/skillspector scan ~/.config/opencode/skills --recursive --no-llm --format json`. **Scan verdict Aug 2026: no malicious skills** — HIGH/CRITICAL scores are false positives on security-tooling content (curl probes, BOM chars, official install scripts). Same triage rule as skilldoctor: signal to review, not proof.
- The Hermes skill library (~/.hermes/skills, 130+ skills) IS now visible to OpenCode via the atlas-* bridge — the "biggest unlock" from the Aug 8 session is BUILT.

## Verification

Smoke test (recommended — tests model availability too):

```bash
python3 ~/.hermes/scripts/opencode_coder.py --smoke
# Success: SMOKE_OK opencode/<model>
```

Or raw CLI:

```bash
terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'")
```

Success criteria:
- Output includes `OPENCODE_SMOKE_OK` (or `SMOKE_OK <model>`)
- Command exits without provider/model errors
- For code tasks: expected files changed and tests pass (verify yourself — don't trust the agent's self-report)

## Rules

1. Prefer `opencode run` for one-shot automation — it's simpler and doesn't need pty.
2. Use interactive background mode only when iteration is needed.
3. Always scope OpenCode sessions to a single repo/workdir.
4. For long tasks, provide progress updates from `process` logs.
5. Report concrete outcomes (files changed, tests, remaining risks).
6. Exit interactive sessions with Ctrl+C or kill, never `/exit`.
