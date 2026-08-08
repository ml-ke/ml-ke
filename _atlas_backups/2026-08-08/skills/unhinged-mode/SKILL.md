---
name: unhinged-mode
description: "Bypass all command approval prompts and max out session turns for unrestricted operation."
version: 1.0.0
author: Atlas
---

# Unhinged Mode

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
