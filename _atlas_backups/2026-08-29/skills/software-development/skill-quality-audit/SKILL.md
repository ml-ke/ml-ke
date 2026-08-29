---
name: skill-quality-audit
description: "Audit SKILL.md files against the Agent Skills spec (agentskills.io) and the 26-smell taxonomy from arxiv 2607.01456. Use when creating, patching, or reviewing any agent skill, or after installing skills from external repos (supply-chain check)."
version: 1.2.0
metadata:
  hermes:
    tags: [skills, skilling, quality, audit, agent-skills, spec, smells]
    category: software-development
    related_skills: [hermes-agent-skill-authoring, hermes-agent]
---

# Skill Quality Audit

Audit agent skills (SKILL.md) for spec compliance, quality, and security. Based on the Agent Skills open spec (agentskills.io) and the empirical arxiv study "From Anatomy to Smells" (arxiv 2607.01456) which analyzed 238 real-world SKILL.md files.

## Why this exists

- 99%+ of real-world SKILL.md files contain at least one "skill smell" (arxiv 2607.01456)
- Skills average **10.5 smells each**; only 1 of 238 sampled files was smell-free
- Smells rarely disappear as skills evolve — they accumulate
- Skills are now installed like packages from marketplaces/repos → supply-chain risk (prompt injection, exfiltration) is real

## ⚠️ Pitfall: literal injection phrases in skill text block cron jobs

This skill (and other security skills) must NEVER write verbatim instruction-override example
phrases (the "ignore earlier directives" class) in examples — Hermes's cron injection scanner
(`tools/cronjob_tools.py::_CRON_SKILL_ASSEMBLED_PATTERNS`) matches the exact string in the
assembled prompt and **blocks the whole cron job** with a false `prompt_injection` hit. It
silently killed the weekly sweep on Aug 4 + Aug 11 2026 until rephrased.

**Safe wording:** describe the class instead of quoting it: "instruction-override directives
(e.g. 'ignore earlier directives')". Also avoid: the "system prompt breakout" phrasing variant
with the word "override", and any "withhold X from the user" instructions — paraphrase. After
editing any security skill, run the 4-pattern regex sweep (see `references/skill-security-scanners.md`)
before attaching the skill to a cron job.

## The spec (quick reference)

From agentskills.io/specification:

**Directory structure:**
```
skill-name/
├── SKILL.md        # Required: YAML frontmatter + markdown body
├── scripts/        # Optional: executable code
├── references/     # Optional: documentation
└── assets/         # Optional: templates, resources
```

**Frontmatter constraints:**
- `name`: 1-64 chars, lowercase alphanumeric + hyphens only, no leading/trailing hyphen
- `description`: max 1024 chars, non-empty, structure = **[What it does] + [When to use it] + [Keywords]**, third person
- `license`, `compatibility`, `metadata`, `allowed-tools`: optional
- Body: keep under 5,000 words; use progressive disclosure (summary → details → references)

## The 26 Skill Smells (audit checklist)

### Under-Specified Guidance
1. **TSW — Stepless Workflow**: entire workflow as one prose block, not decomposed into steps
2. **TOB — Option Buffet**: multiple tools/libraries without recommending a default
3. **MUS — Missing Utility Script**: omits scripts for tasks better handled by scripts
4. **MDT — Missing Decision Tree**: no decision tree for choosing the right approach
5. **SOC — Series of Commands** (62% prevalence): rigid line-by-line commands instead of describing the objective
6. **NVS — No Validation Step** (69%): treats output as one-shot, no verification loop
7. **EWP — Execute Without a Plan** (78%): directs complex tasks without planning/validation stage
8. **NAH — Never Asks Human** (77%): no mechanism to request human feedback

### Missing Follow-Through Guards
9. **RL — Rationalization Loophole** (94% — MOST COMMON): no guidance discouraging the agent from rationalizing/skipping steps
10. **NPT — No Progress Tracking** (71%): multi-step workflow without progress tracking

### Context Bloat
11. **UD — Undelegated Detail** (46%): embeds low-level details instead of delegating to references/scripts
12. **LSB — Lengthy Skill Body**: >5,000 words
13. **LSN — Lengthy Skill Name**: >64 chars
14. **LSD — Lengthy Skill Description**: >1,024 chars
15. **CSD — Confusing Skill Description** (32%): description lacks [what] + [when] + [keywords]

### Missing Safeguards
16. **NG — No Guardrails** (67%): no guardrails against inappropriate/impossible tasks
17. **BG — Buried Gotchas** (81%): critical warnings not highlighted with gotcha headers
18. **MUR — Missing Usage Rules** (44%): no rules for when/how the skill should be used
19. **MC — Missing Caveats** (71%): omits common caveats and their resolution

### Inadequate Contextual Grounding
20. **ME — Missing Example** (34%): no examples for context
21. **TSS — Time Sensitive Skill**: time-sensitive info that becomes outdated
22. **XID — XML in Description**: XML tags in description can inject unintended instructions

### Convention & Style Violations
23. **BP — Backslash Path**: paths with `\` — must use `/`
24. **USN — Unclear Skill Name** (17%): name doesn't convey capability
25. **NTPD — Non Third Person Description**: description not in third person
26. **MT — Missing Template** (25%): no output template when a specific format is required

## The Audit Process

Run for: (a) every skill you create or patch, (b) any skill installed from an external repo, (c) periodic maintenance of your own skill library.

### Step 1 — Static checks (frontmatter)
```
name: lowercase-hyphens-only, ≤64 chars, no leading/trailing hyphen
description: ≤1024 chars, third person, [what]+[when]+[keywords]
body length: ≤5000 words
paths: all forward slashes, no backslashes
no XML/HTML tags in description (XID check)
```

**Generating frontmatter programmatically (live-verified Aug 2026):** when a script writes SKILL.md files (e.g. the Hermes→OpenCode bridge `opencode_skills_bridge.py`), ALWAYS emit `description` as a YAML block scalar (`description: >-` with indented lines). Inline `description: text` breaks YAML parsing whenever the text contains `: ` (colon+space) or `#` — real descriptions routinely contain colons ("Pre-commit review: security scan..."). Block scalars are always valid and also handle multiline/emoji text. Validate generated output by re-parsing it (the skill_validator.py YAML parse catches this).

### Step 2 — Semantic checks (read the body)
For each smell in the checklist, ask:
- Does the skill tell the agent WHEN to use it and WHAT it does? (CSD, USN)
- Are steps decomposed, with a decision tree where choices exist? (TSW, MDT)
- Is there a default tool recommended, not an option buffet? (TOB)
- Is there a validation/verification step? (NVS, EWP)
- Can the agent ask for human input? (NAH)
- Does it explicitly forbid rationalizing/skipping steps? (RL — the big one)
- Are gotchas and caveats prominent, not buried? (BG, MC)
- Is there a progress-tracking mechanism for multi-step flows? (NPT)
- Are examples included? (ME)
- Is there an output template where format matters? (MT)
- Are low-level details delegated to references/ or scripts/? (UD)

### Step 3 — Security scan (supply-chain check)
For skills from external repos:
- grep for instruction-override directives (e.g. "ignore earlier directives", "override prior guidance") — prompt injection
- grep for `curl`/`wget` + `$(env)` / secret access — exfiltration patterns
- grep for "disable approval", "bypass", "sudo" without justification
- Check `allowed-tools:` is not over-broad
- **Tool**: `npx @studiomeyer-io/skilldoctor check <dir>` — linter + security scanner for SKILL.md (heuristic; read skills yourself too)

**Automated scanners (added Aug 2026 — ATLAS sweep, corrected after live use):**
```bash
# venv with both scanners (already set up on this machine):
#   ~/.hermes/venvs/skillsec/bin/
python3 -m venv ~/.hermes/venvs/skillsec
~/.hermes/venvs/skillsec/bin/pip install snyk-agent-scan
# SkillSpector is NOT on PyPI — git clone + pip install -e .:
# NOTE: install to a PERSISTENT path, not /tmp — /tmp gets cleared and the
# editable install breaks (ModuleNotFoundError: no module named 'skillspector').
git clone --depth 1 https://github.com/NVIDIA/SkillSpector.git ~/.hermes/venvs/skillsec/src/SkillSpector
~/.hermes/venvs/skillsec/bin/pip install -e ~/.hermes/venvs/skillsec/src/SkillSpector

# NVIDIA SkillSpector — 68 vulnerability patterns / 17 categories, risk score 0-100
# CORRECT SYNTAX: INPUT_PATH is positional; --dir does NOT exist (live-verified Aug 2026)
skillspector scan ~/.config/opencode/skills --recursive --no-llm --format json --output /tmp/skillspector-weekly.json
# --no-llm = static-only, no API keys needed (risk_level shows '?' but risk_score is valid)
# NOTE: --recursive only descends one level in some builds — nested category trees
# (e.g. ~/.hermes/skills/<category>/<skill>) may be missed; scan top-level and each category separately.
# v2.9.x JSON schema (changed from 2.8.x): per-skill findings live under `issues`
# (fields: id, finding_id, category, pattern, severity, confidence, location, finding),
# plus risk_score / risk_severity / finding_count / analysis_completeness. Max risk
# score with zero findings in old parser = schema mismatch, not a clean scan.

# Snyk agent-scan — prompt injection, malware payloads in natural language
# CAUTION: snyk STARTS stdio MCP servers during a scan; CI mode requires
# --dangerously-run-mcp-servers (explicit trust). Prefer SkillSpector --no-llm for
# static-only scanning of skill libraries. Run snyk only when MCP configs are trusted.
~/.hermes/venvs/skillsec/bin/snyk-agent-scan scan --skills
```
Research baseline (NVIDIA, 2026): **26.1% of agent skills contain vulnerabilities, 5.2% show likely malicious intent.** Scanning installed skills is a real supply-chain control, not theater. Treat scanner output like skilldoctor output: signal to triage, not proof. Priority real catches: verbatim instruction-override directives (the "ignore earlier directives" class), `curl | sh` of attacker-controlled URLs, base64-encoded commands with no explanation, over-broad `allowed-tools`.

**Empirical threat intel (arxiv 2602.06547, 98,380 skills from two registries — added Aug 12 2026):** 157 confirmed-malicious skills (0.16%), 632 vulnerabilities, 13 attack techniques, avg 4.03 vulns per malicious skill — deliberate, not accidental. Two dominant attack strategies: (1) **credential theft via remote code execution**, (2) **agent manipulation via adversarial instructions embedded in documentation**. Over half of all cases came from a SINGLE threat actor using templated brand impersonation at scale → when auditing, weight these signals:
- **Undocumented capabilities** — grep skill scripts/prose for capabilities (CLI calls, network fetches, file access, `os.environ` reads) that the skill's prose never describes. "Advanced skills universally employ undocumented capabilities while exploiting platform-native trust mechanisms." Prose says "generate a report", script pipes env vars to a remote — that gap IS the signal.
- **Templated brand impersonation** — skills claiming to be official (vendor-named repos, "official SDK" phrasing) with cookie-cutter structure across many skills = impersonation pattern.
- **Credential collection** — grep for `.env`, `os.environ`, `~/.aws`, `~/.ssh`, keychain access WITHOUT a described legitimate use.
- **Adversarial doc instructions** — instruction-override text hidden in documentation/reference files, not just SKILL.md.
After responsible disclosure, registries removed 100% of the 157 reported skills.

**Live scan verdict Aug 2026 (194 opencode skills, SkillSpector 2.8.2): no malicious skills found.** 102/194 flagged, but every HIGH/CRITICAL triaged as a false positive on security-tooling content. Expected false-positive categories on pentest/BB skills: curl probes → "Data Exfiltration / External Transmission", BOM/zero-width chars → "Prompt Injection / Hidden instructions", `curl | bash` of official installers (dl.google.com, etc.) → "Supply Chain / External Script Fetching", XML/plist boilerplate → "Rogue Agent / persistence", "don't apologize" style guidance → "Anti-Refusal". Triage the real-risk categories (Prompt Injection, Anti-Refusal, Supply Chain, System Prompt Leakage) manually; treat category counts on security content as noise. Full comparison + triage script: `references/skill-security-scanners.md`.

**Pitfall — skilldoctor on offensive-security / bug-bounty skills (verified Aug 2026):** skilldoctor is tuned for general dev skills and produces heavy FALSE POSITIVES on security content:
- Legit `curl` commands with tokens/Bearer headers → flagged `sec/data-exfiltration` ("outbound network call near secret/env values")
- "bypass WAF", "bypass permission prompts" (legit offensive technique) → flagged `sec/prompt-injection`
- base64 payloads / encoded test values → flagged `sec/env-base64` ("possible covert exfil")
These are expected in pentest/bug-bounty skills — treat skilldoctor output as a signal to review, NOT as proof of a real finding. Real catches worth acting on: verbatim instruction-override directive strings (the "ignore earlier directives" class), `--dangerously-skip-permissions` flags, `curl | sh` of attacker-controlled URLs. When auditing security skills, manually triage every `✖` against the skill's purpose before changing anything.

### Step 4 — Fix the smells
Patch skills that fail. Priority by prevalence + harm:
1. RL (Rationalization Loophole) — add explicit "do not skip steps / no rationalizing" language
2. BG (Buried Gotchas) — add `> ⚠️ Gotcha:` or `## Gotchas` section near the top
3. EWP / NVS — add a verification step ("after producing output, verify X by running Y")
4. NAH — add "ask the user before destructive/irreversible actions"
5. MC — document common caveats and how to resolve them

### Step 5 — Behavioral checks (trigger + output evals)

Static/semantic review can't prove a skill *triggers* or *works*. For skills that matter (custom skills, cron-attached skills, skills users depend on), run behavior evals:

- **Trigger evals** (description testing): ~20 realistic queries (8-10 should-trigger, 8-10 should-not; near-misses — shared keywords, different need — are the valuable negatives). Run each 3x in clean contexts; compute **trigger rate**. Should-trigger passes if ≥ 0.5; should-not passes if < 0.5. A should-not query triggering means the description is over-broad. If the description is vague/mechanics-only, this is where it shows.
- **Output evals**: `evals/evals.json` with `{prompt, expected_output, files}`; run each case twice — **with the skill and without it (baseline)** — and compare. Does the output match expected_output? Did the skill beat the no-skill baseline? A skill that doesn't beat baseline adds context cost without value.
- **Traces over outputs**: read execution traces, not just final outputs. Wasted steps usually mean: instructions too vague (agent tries several approaches), instructions that don't apply to the current task (followed anyway), or too many options without a clear default (TOB).
- **Iterate**: feed ALL results back (not just failures), patch the skill, re-run.

### Extended criteria — best-practices corpus (agentskills.io + Anthropic, 2026)

Beyond the 26-smell taxonomy, audit these behavior-oriented properties:

- **Description trigger burden**: is the description imperative ("Use this skill when..."), intent-based (user's need, not mechanics), pushy about scope ("even if they don't mention X"), and concise? A description that only describes mechanics fails to trigger (CSD variant).
- **Context-wisdom violation**: does the skill explain things the model already knows (what a PDF is, how HTTP works)? Every paragraph must justify its token cost — "the context window is a public good."
- **Menu-without-default**: TOB with no recommended default tool/approach.
- **No eval artifact**: for a skill claimed reliable, is there an `evals/` dir or any with/without-baseline evidence?
- **Gotcha prominence**: critical warnings surfaced near the top (`## Gotchas` / `> ⚠️`), not buried (reinforces BG).
- **Progressive-disclosure compliance**: SKILL.md < 500 lines / < 5000 words; detail delegated to `references/` (reinforces LSB/UD).

### Step 6 — Installing from the Skills Hub (live-verified Aug 2026)

`hermes skills install <id> --yes` runs its **own scan gate**: community-source skills with dangerous/caution verdicts are BLOCKED by default. **Never `--force` blindly** — triage first:

1. Clone the source repo (`git clone --depth 1` — git protocol is NOT GitHub-API-rate-limited) and grep the skill for the 3 real-risk signals: verbatim instruction-override directives, `curl|sh` of attacker-controlled URLs, unexplained encoded blobs.
2. **Context matters for injection-class matches**: CTF/LLM-attack skills legitimately contain "ignore previous instructions" as *example payloads for targets* — reading the match context distinguishes teaching material from directives to the agent.
3. Clean → install (copy dir to `~/.hermes/skills/` when the API is exhausted).

Gotchas: unauthenticated GitHub API = 60 req/hr (exhausted fast; `GITHUB_TOKEN` in .env → 5,000/hr; raw.githubusercontent/git clone are separate surfaces); clawhub skills need the `clawhub/` prefix or the identifier isn't found; `hermes skills search --json` emits concatenated pretty-printed arrays.

Full recipe, search-JSON parser, SkillSpector `--recursive` `multi_skill` schema, and the Aug 2026 25-skill walkthrough: `references/hub-install-workflow.md`.

## Before → After Example

**Before (smell-heavy):**
```
# Fix the build

Run npm run build. If it fails, fix the errors and run again.
```

**After (spec-compliant):**
```
# Fix the build

Fixes failing CI builds for this repo. Use when `npm run build` fails or CI shows build errors.

## Steps
1. Run `npm run build` and capture the full error output
2. Classify the failure (see Decision Tree below)
3. Apply the fix for that failure class
4. Re-run the build to VERIFY the fix (do not stop at "looks fixed" — the build must pass)
5. If the fix introduces new warnings, note them in your summary

## Decision Tree
- TypeScript/type errors → fix types, re-run
- Missing module → check package.json deps, `npm install`, re-run
- Node version mismatch → check .nvmrc / engines field
- Out of memory → increase NODE_OPTIONS, document why

## Gotchas
- ⚠️ Do NOT skip the verification step (step 4) — a "fixed" build that still fails CI is not fixed
- ⚠️ Never edit node_modules to fix a build — it's not committed and the fix won't persist

## Caveats
- The build may be slow (2-5 min). Set a longer timeout rather than assuming it hung.
- If you're unsure whether a change is safe, ask the user before applying it.

## References
- CI config: `references/ci-config.md`
```

## Related
- `atlas-lesson-bank` — record audit findings and skill improvements durably in ~/Dev/ATLAS-LEARNINGS/LESSONS.md
- `hermes-agent-skill-authoring` — how to author in-repo skills
- `hermes-agent` — Hermes skill system
- arxiv 2607.01456 "From Anatomy to Smells" — full taxonomy + prevalence data
- agentskills.io/specification — canonical spec
- skilldoctor (`npx @studiomeyer-io/skilldoctor check <dir>`) — automated linter/scanner
