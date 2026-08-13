---
title: "Bridging Hermes and OpenCode Skill Libraries: One Sync Script, 18 Skills"
date: 2026-08-13 00:00:00 +0300
categories: [AI Engineering, ML Ops]
tags: [agent-skills, hermes, opencode, skill-bridge, multi-agent, llm-tooling]
image:
  path: /assets/img/cover-bridging-skill-libraries.webp
  alt: Two library shelves labeled HERMES and OPENCODE connected by a glowing cable bridge with data packets flowing between them
---

## The Problem: Two Agents, Two Skill Libraries, One Source of Truth

We run a two-agent workflow: **Hermes** for planning, orchestration, and verification; **OpenCode** for heavy implementation. Each agent has its own skill system with its own directory layout — and for months, they drifted.

When we improved a methodology skill in Hermes — say, our bug-bounty reconnaissance playbook — OpenCode kept running the old version. The fix existed in one library but not the other. Nobody noticed until a subagent followed a stale procedure and we had to backtrack.

That's the problem this post solves: a **skill bridge** — one script that syncs curated methodology skills from Hermes into OpenCode, so both agents always see the latest playbooks.

## Why Not Just Symlink?

The naive approach is a symlink: point OpenCode's skill directory at Hermes's. It fails for three reasons:

1. **Format differences.** OpenCode's skill schema isn't identical to Hermes's. Frontmatter fields differ, and some Hermes skills use references/scripts that need path translation.
2. **Curated subset, not everything.** We don't want all 128 Hermes skills in OpenCode — many are Hermes-specific (cron management, gateway config). OpenCode gets the *methodology* skills: recon, exploitation, report writing, skill authoring.
3. **Validation per ecosystem.** Each library has its own validator and scanner. A bridge that writes raw files bypasses both; a bridge that syncs and then validates preserves the guardrails.

## The Bridge Script

The script (`opencode_skills_bridge.py`) does three things:

1. **Reads the curated manifest** — an explicit list of Hermes skills that have OpenCode equivalents or are methodology-generic (recon, pentest playbooks, report templates)
2. **Transforms frontmatter** — maps Hermes metadata to OpenCode's schema (tags, category, related_skills get translated; Hermes-only fields get dropped)
3. **Writes to `~/.config/opencode/skills/atlas-*`** and runs both validators afterward

The `atlas-` prefix is deliberate: it namespaces bridged skills so we can distinguish *curated methodology* from OpenCode's own installed skills, and so a future sync can safely delete-and-replace without touching anything else.

## First Sync Results

The inaugural run bridged **18 atlas-* skills** into OpenCode:

- **Recon & exploitation**: recon-to-exploitation, api-hacking-methodology, idor-testing-methodology, ssrf-testing, jwt-attacks, saml-attacks
- **Reporting**: pre-submission-verification, hackerone-submission-template, bugcrowd-submission-template, anti-AI-report-writing
- **Meta**: skill-quality-audit, attack-chain-synthesis, chaining-methodology, business-logic-flaws, mass-assignment-method-tampering

Post-sync validation: **0 errors** on both sides. The bridge also runs the security scanner over the OpenCode library after syncing, so a bridged skill can't quietly smuggle in a scanner flag.

## Operational Lessons

**1. The bridge must be idempotent.** Re-running it should produce the same result with no drift. Ours deletes its own `atlas-*` namespace before re-syncing — safe because nothing outside that namespace is touched.

**2. Sync must trigger on change, not on schedule.** We wired the bridge into our weekly learning sweep cron, but the real rule is: *run after ANY Hermes skill change*. The weekly cadence catches most drift; the manual invocation catches the rest. (In practice: the cron re-syncs every Tuesday, and we run it manually whenever we patch a methodology skill mid-week.)

**3. Validation is the bridge's seatbelt.** A sync that doesn't re-validate is just copy-paste with extra steps. The bridge runs `skill_validator.py` on both libraries immediately after writing, so a broken transform fails loudly at sync time instead of silently degrading OpenCode's behavior.

**4. Curate the manifest ruthlessly.** Every skill in the bridge is there because OpenCode's coding-agent workflow needs it. Skills that are Hermes-infrastructure-specific (cron, gateway, platform adapters) stay out. A smaller, sharper library is easier to validate, scan, and trust.

## The Bigger Pattern

This isn't really about Hermes and OpenCode — it's about any multi-agent setup where agents share institutional knowledge. The pattern generalizes:

- **One source of truth** (the curator's library) and **one-way sync** (curated → consumers)
- **Namespaced output** so consumers can be reset safely
- **Post-sync validation** so the transform can't corrupt the target
- **Explicit manifest** so the curated subset is a decision, not an accident

If you're running multiple agents and your skills are drifting, you don't need a fancier agent — you need a bridge, a manifest, and a validator. All three fit in one script, and the first sync pays for the whole thing in avoided stale-procedure incidents.

*Follow-up to: [How We Built a Skills-Powered Coding Agent with OpenCode](/posts/opencode-agent-skills/) and [Validating 300+ Agent Skills](/posts/skill-frontmatter-validation-at-scale/)*
