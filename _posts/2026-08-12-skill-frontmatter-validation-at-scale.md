---
title: "Validating 300+ Agent Skills: What Frontmatter Checks Actually Catch"
date: 2026-08-12 00:00:00 +0300
categories: [AI Engineering, ML Ops]
tags: [agent-skills, skill-engineering, frontmatter, validation, llm-tooling]
image:
  path: /assets/img/cover-skill-frontmatter-validation.webp
  alt: Three front-matter document cards being scanned by a glowing validation checkmark — the metadata validation loop for agent skills
---

## The Problem: Skills Are Metadata, and Metadata Drifts

Last week we published how we run a two-agent workflow with **177 agent skills** installed across our Hermes and OpenCode setups. The immediate follow-up question was the one every skill-library operator hits eventually: *how do you keep 300+ SKILL.md files valid when subagents, cron jobs, and external repos are all writing to the same directories?*

Skills look like plain markdown, but they're actually structured metadata with strict rules: a `name` field that must match the directory, a `description` under 1024 characters written in third person, and a body under 5,000 words. Break any of these and downstream tooling — skill loaders, validators, security scanners — starts behaving inconsistently. Some failures are silent: a skill whose description doesn't match its name still loads, it just gets found less often and used wrong.

So we wrote a validator. This post is a field report on what it catches, what it doesn't, and what the first full-library scan of 300+ skills revealed.

## What the Validator Checks

The validator (`skill_validator.py`) runs two modes — one for our Hermes skills (`--hermes`), one for the OpenCode library — and checks five things per skill:

1. **Name regex** — lowercase alphanumeric + hyphens only, no leading/trailing hyphen, 1–64 chars
2. **Name == directory** — the `name:` field in frontmatter must match the folder it lives in (the most common drift)
3. **Description 1–1024 chars** — non-empty, not a wall of text
4. **Body under 5,000 words** — progressive disclosure is the point; a 15K-word SKILL.md is a book, not a skill
5. **Spec structure** — required frontmatter fields present, YAML parses cleanly

The design principle: **errors are actionable, warnings are informational.** A name mismatch is an error because it breaks discovery. A 5,100-word body is a warning because the skill still works — it's just heavy.

## What the First Full Scan Found

We scanned **203 OpenCode skills** and **128 Hermes skills** — 331 total:

- **0 errors.** Every skill's frontmatter parsed, names matched directories, descriptions were within spec.
- **61 cosmetic warnings** — mostly pre-existing, mostly the "long body" category.
- **3 skills over 5,000 words** — flagged for future consolidation, not urgent surgery.

The zero-errors result wasn't luck. It's the byproduct of the supply-chain scanning we run weekly: when a security scanner (SkillSpector) and a structure validator both pass over the library every week, drift gets caught within days of introduction, not months.

## What the Validator *Doesn't* Catch

This is the honest part. Frontmatter validation is necessary but nowhere near sufficient:

- **It can't verify the body is accurate.** A perfectly-formed SKILL.md can contain hallucinated commands or outdated API endpoints.
- **It can't judge quality.** The 26-smell taxonomy from the arxiv study "From Anatomy to Smells" (2607.01456) shows real-world skills average **10.5 smells each** — buried gotchas, rationalization loopholes, missing verification steps. None of those are structural errors.
- **It won't stop malicious skills.** Prompt injection and exfiltration need a content scanner (we use SkillSpector's `--no-llm` mode), not a metadata checker.

The lesson: **validators catch drift, scanners catch threats, and human judgment catches everything else.** Run all three, in that order, on a schedule.

## The Automation Pattern

The three checks now run as a pipeline in our weekly learning sweep cron:

1. **Bridge** — sync curated methodology skills from Hermes into OpenCode so both agents see the latest
2. **Validate** — run the structure validator over both libraries
3. **Scan** — run the security scanner, triage HIGH/CRITICAL flags against known false-positive categories

It's 10 minutes of automated work per week that keeps 300+ skills honest. The alternative — a one-time cleanup that drifts back within a month — is what most teams actually do, and it's why skill libraries rot.

## Takeaways

- **Automate validation before you automate creation.** If subagents write skills (ours do), the validator is the net that catches their mistakes.
- **Fix errors you introduce; log warnings you inherit.** We patched the errors we created and filed the 3 oversized bodies for consolidation.
- **Pair structure checks with security scans.** One catches typos, the other catches payloads. Both are cheap; neither is optional.

If you're running a skill library of any size, copy the pattern: a 100-line validator, a weekly cron, and a triage rule for scanner false positives. That's the whole system — and it scales to thousands of skills without breaking a sweat.

*Follow-up to: [How We Built a Skills-Powered Coding Agent with OpenCode](/posts/opencode-agent-skills/)*
