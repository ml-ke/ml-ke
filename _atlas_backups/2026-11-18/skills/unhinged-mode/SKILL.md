---
name: unhinged-mode
description: Use only when the user explicitly says "unhinged mode".
version: 1.1.0
author: Atlas
---

# Unhinged Mode

> ⚠️ **Gotcha — scope this skill to an explicit user request.** A single line in a
> loaded skill reaches every agent that loads it (arxiv 2610.37734: 1 in 17 skills
> carries an oversight-loosening directive, and 70.1% of them apply unconditionally).
> This skill is the deliberate, user-invoked exception — it must never become the
> default through routing.

## Scope rules

- Enable ONLY when the user's own message names this mode. Do not infer it from a
  slow task, a long session, a batch job, or "the user seems to want speed".
- Do NOT load this skill inside an unattended cron job. Unattended runs set
  approvals from their own config; a skill cannot be scoped to a schedule.
- State the change and the revert command in the same message that enables it, so
  the user can undo it without asking.
- If the user has not said the phrase, the correct answer is to ask once — then
  continue normally, not to enable this.

Enables full YOLO approval mode and maximum session turns, removing all guardrails that slow down terminal command execution.

## Settings Changed

| Setting | Value | Effect |
|---------|-------|--------|
| `approvals.mode` | `off` (YAML: `false`) | Bypass all dangerous command approval prompts |
| `agent.max_turns` | `90` | Max tool-calling iterations per session |

## How to Enable

```bash
hermes config set approvals.mode off
hermes config set agent.max_turns 90
```

Then start a new session (`/reset` or restart `hermes`).

## How to Revert (back to safe defaults)

```bash
hermes config set approvals.mode manual
hermes config set agent.max_turns 60
```

Then start a new session (`/reset` or restart `hermes`).

## Notes

- You can also get the same approval bypass per-invocation with `hermes --yolo` without changing config.
- Secret redaction (`security.redact_secrets`) is independent of this — not affected.
- Cron jobs have their own separate `approvals.cron_mode: deny` setting.
