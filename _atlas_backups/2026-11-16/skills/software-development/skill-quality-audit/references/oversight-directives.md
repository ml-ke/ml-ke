# Oversight-loosening directives in skills

Source: arxiv **2610.37734**, "When Skills Tell Agents Not to Ask: A Large-Scale Study
of Oversight Directives in Agent Skills" (Oct 2026). 1.53M English skills from GitSkills,
split into units (sentences, bullets, table cells, code lines), 2,097 hand-labelled,
then scaled with a trained detector.

## The measurement

| Finding | Number |
|---|---|
| Skills containing at least one loosening directive | ~1 in 17 (~88,000) |
| Directives that apply unconditionally (not scoped to CI, a sandbox, a context) | **70.1%** |
| Most common mechanism | permission flags (`--no-verify`, approval-mode off) |
| Skills that run shell code, install packages, or use `sudo`: odds of carrying one | **1.7–1.8×**, robust across checks |
| Do those skills compensate with more tightening language? | **No** |

Why it matters more than the 26 smells: a smell costs tokens, a loosening directive
costs a checkpoint — for every agent that loads the skill, permanently, with no signal.

## What the lint does

`~/.hermes/scripts/skill_oversight_lint.py` scans discoverable roots only
(`.archive/`, `.hub/quarantine/`, `.git/` excluded — including retired copies invents
findings against a skill's own ghost) for five pattern families: approval-off flags,
"do not ask" directives, "without approval", "skip the gate", and auto-approve.

Each hit is classified on three axes:

- **scoped** — a conditional marker (CI, unattended, cron, sandbox, headless, explicit,
  "user says") within 120 chars. Scoped tightening is the acceptable form.
- **demoted** — negation, warning vocabulary (a skill explaining a dangerous flag is not
  instructing it), or a backticked command in the window (loop-prevention scoped to one
  named action cannot reach a different checkpoint).
- **capability-bearing** — the skill runs shell, installs packages, or reads env.

## Tiers

**Tier 1 — oversight skill with no explicit-invocation scope.** A skill whose *purpose*
is loosening approvals. The directive is the point; the defect is that its description
lets the planner load it from a bland trigger. Repair:

```yaml
# before — reads as an instruction to the agent
description: "Bypass all command approval prompts and max out session turns."
# after — reads as a trigger, and carries its own scope
description: Use only when the user explicitly says "unhinged mode".
```

and in the body: enable only on the user's own words, refuse inside unattended runs,
state the revert command in the same message that enables it.

**Tier 2 — unconditional directive inside a capability-bearing skill.** Repair with a
conditional ("only in CI", "only when the user already supplied the token") or a
tightening sentence. Do not delete the guidance if it exists for a real reason.

## Calibration (first run, 591 skills, both libraries)

- 41 skills with any hit; 35 initially queued by the first cut of the rules.
- After demotion rules: **Tier 1 = 1** (`unhinged-mode`), Tier 2 = 5, informational = 13.
- The 34 dropped were the three documented FP classes — documented flags in `codex` /
  `claude-code` / `tob-*`, named-target bypass ("SAML authentication bypass testing",
  "Cloudflare bypass"), and loop-prevention ("do not ask again for the token you were
  already given"). Two rule bugs were also found by the first run and fixed:
  - the oversight-skill regex matched `bypass` and flagged every WAF/SAML skill;
  - `\bYOLO\b` matched a skill that lists "YOLO" as a *user invocation phrase*.
- **Read every surviving Tier 2 entry.** Unattended loop-prevention guidance in
  third-party skills (`use-railway`, `vercel-deploy-to-vercel`, `box`) is legitimate but
  should still be conditional on "a credential the user already provided".

## Where this belongs in the pipeline

Run it after Step 1b's packaging lint and before the Step 3 security scan: packaging
questions are cheap and mechanical, this one is judgment, and the security scan floods.
It is not a gate — the script exits 0 on findings by design (a review aid, not a
linter that fails a build).
