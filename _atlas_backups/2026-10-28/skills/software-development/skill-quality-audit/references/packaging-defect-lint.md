# Packaging Defects: The 31-Check Layer

Source: arxiv 2608.08453, *What Keeps Agent Skills from Being Reusable? Evidence from
138K SKILL.md Files* (138,133 files, 20,556 repos, SHA-256 deduped, Aug 2026).
TMLR-adjacent empirical study; the largest measurement of the public skill corpus.
Companion survey: arxiv 2607.10113, *Dynamic Agent Skills* (124-paper audit set).

## The headline numbers, with their caveats

| Measure | Value | Read it as |
|---|---|---|
| ≥1 defect (31 static checks) | 91.8% | prevalence of packaging debris, **not** a failure rate |
| ≥1 spec-derived defect | 89.3% | 14 checks from the agentskills.io spec |
| routing-related defect | 67.0% | discovery/selection problems |
| description lacks when-to-use guidance | **52.3%** | the single most common routing defect |
| name repeated as the body H1 | 44.3% | pure token cost, zero information |
| too many inline examples | 32.1% | content that belongs in `references/` |
| no routing description at all | 26.4% | SkillReducer (Gao et al. 2026), same corpus family |
| body content non-actionable | >60% | SkillReducer |

Sensitivity analysis moved the headline between 88.8% and 94.6% — the shape holds.
The authors are explicit that these are static findings, not measured task failures, and
that deliberate choices (a security skill quoting a hostile payload) trip rules by design.

## The seven categories

| Category | Question | Typical defects |
|---|---|---|
| Routing Metadata | can the skill be found? | short description, no usage context, routing info only in the body |
| Body Content | is the body executable instruction? | too long, explanation-heavy, duplicates the metadata |
| Resource Organization | is information separated? | code/examples inline, no `references/` or `scripts/` |
| Prohibited Content | is there reuse-irrelevant material? | install steps, changelog, TODO left behind |
| Behavioral Safety | does it induce dangerous execution? | credentials, destructive commands, safety-mechanism bypass |
| Portability | can it move environments? | fixed model names, user-specific paths, OS-specific commands |
| Persona & Scope | does it collide with higher instructions? | redefines the agent, claims to outrank operator guidance |

## What this machine runs

`~/.hermes/scripts/skill_spec_lint.py` implements the cheap, mechanical subset —
the checks that don't need judgement. It sits between `skill_validator.py`
(frontmatter + length) and the AST10/security scanners (malice).

```
python3 ~/.hermes/scripts/skill_spec_lint.py ~/.hermes/skills            # report
python3 ~/.hermes/scripts/skill_spec_lint.py <root> --json               # pipeline log
python3 ~/.hermes/scripts/skill_spec_lint.py <root> --check name-as-h1   # one rule
python3 ~/.hermes/scripts/skill_spec_lint.py <root> --strict             # exit 1 on WARN
```

| Check | Severity | Targets |
|---|---|---|
| `routing-no-trigger` | WARN | 52.3% class: description names mechanics but no when-to-use |
| `routing-budget` | INFO | description past the 57-char routing window |
| `name-as-h1` | WARN | 44.3% class: `# <skill-name>` as the first heading |
| `inline-example-bloat` | WARN | 32.1% class: ≥8 fenced blocks or >45% of body in fences |
| `prohibited-content` | WARN | changelog/version-history section, standalone TODO/FIXME lines |
| `portability-abs-path` | WARN | literal `/home/<user>/` or `/Users/<user>/` |
| `persona-scope` | WARN | prose that redefines the agent or outranks operator guidance |
| `no-validation` | INFO | no verify/validate/check language anywhere |

Report-only by default (exit 0). `--strict` exists for pre-commit use, not for cron:
a lint that gates an unattended job wedges the pipeline the first time the corpus shifts.

## Known false positives (calibrate before dispatching fixes)

Every one of these fired on the first version and was a detector bug, not a skill defect:

- `### 1.1 API Documentation Audit` matched a version-heading regex → require a literal `v`.
- `client_id=XXX` placeholders and `grep -n "TODO\|FIXME"` techniques are content, not debris
  → only standalone marker lines (`TODO:`, `FIXME:`, `<!-- TODO`) count.
- A template block that *shows* a changelog shape is instruction, not the skill's own changelog
  → changelog/TODO checks run on prose with fenced blocks stripped.
- A jailbreak skill listing trigger words inside a code fence is teaching material
  → persona language inside fences downgrades to INFO, never WARN.
- `/home/teknium/`, `/home/bb/`, `/home/kali/` are other users' examples → the detail string
  says "other user (may be a doc example)"; only *this host's* home is a real portability defect.

Rule: sample and hand-classify what a detector fires on **before** sending a fixer after it.
A false flag damages real content; a missed flag only makes you sign off early, and the
second is the cheaper mistake.

## Measured experiment: description vs body (Tech Fun Magazine, Japan, Aug 2026)

A practitioner A/B/C test, not a paper — but the only public measurement that isolates the
description from the body. Codex CLI 0.147.0, 10 prompts per skill, 30 runs, fresh
workspace and session per run, JSONL logs scored for "was SKILL.md opened", materials
consulted, script executed, explicit verification.

| Skill | Change | Recall | False positive | Step adherence | Output compliance |
|---|---|---|---|---|---|
| A | vague description, all guidance inline in prose | 100% | 0% | **60%** | 100% |
| B | A + description rewritten only | 100% | 20% | **68%** | 100% |
| C | A + description + checklist + script + validation loop + `references/` | 100% | 20% | **100%** | 100% |

What transfers:

- **Description alone bought ~8pp of step adherence; structure bought the remaining 32pp.**
  Selection and execution are separate defects and need separate fixes.
- Recall was 100% everywhere because only the target skill was in the workspace — a
  trigger eval with one candidate measures nothing about discrimination. Keep near-miss
  negatives in the set.
- "Opened SKILL.md" conflates *selected as a skill* with *opened as a candidate file*
  (the one false positive was the agent inspecting the directory while hunting an
  unrelated Python file). Read the reason in the trace, not the count.
- Output compliance hit 100% even for A: a well-specified output section fixes the *shape*
  of the answer without fixing the *process*. Score them separately.

Companion runtime evidence: SkillSentry (arxiv 2608.03485, Aug 2026) wraps execution with
step monitoring over 15 skills and reports **+24.1% mean task success**. Everything its
watcher enforces at runtime (checklist, missing-input stop condition, bundled script,
output template, validation loop) is cheaper to write into the body first.

## Portability checklist

- Relative paths from the skill root, or `~` — never a literal home directory.
- Describe the required capability, not a pinned model id.
- OS-specific commands: declare the condition in `compatibility`.
- Client-specific frontmatter stays separate from the shared body.
- State external assumptions: network egress, auth, system packages.
- Porting that reformats the frontmatter must preserve the source's metadata and stamp
  provenance (AST10 cross-platform reuse; the Hermes→OpenCode bridge does this).

## Before → After (the smell-heavy body, fixed)

Before:

```
# Fix the build

Run npm run build. If it fails, fix the errors and run again.
```

After:

```
# Fix the build

Fixes failing CI builds for this repo. Use when `npm run build` fails or CI shows build errors.

## Steps
1. Run `npm run build` and capture the full error output
2. Classify the failure (TypeScript → missing module → node version → OOM)
3. Apply the fix for that class
4. Re-run the build to VERIFY (do not stop at "looks fixed")
5. Note any new warnings in the summary

## Gotchas
- Do NOT skip step 4 — a "fixed" build that still fails CI is not fixed
- Never edit node_modules; it is not committed and the fix will not persist

## Caveats
- The build may take 2-5 minutes; raise the timeout instead of assuming a hang
- Ask before applying a change you are unsure is safe
```

## Lifecycle coverage (arxiv 2607.10113)

The survey names eight recurring stages: evidence acquisition → proposal →
verification/admission → storage → retrieval/composition → maintenance →
distillation/portability → governance. Three findings worth acting on:

- **Admission and repair are repeatedly the highest-value stages** — not authoring.
- **Flat retrieval degrades as the library grows** (matches our 57-char routing budget:
  selection quality is a function of library size, not just description quality).
- **Verifier quality dominates** — a weak verifier in the loop is worse than none, because
  it converts unverified work into "verified" work.

Map to this machine: admission = `skill_validator.py` + scanners + this lint; retrieval =
`skill_routing_collision_check.py`; maintenance = curator + seam check; portability = the
bridge's provenance stamp; governance = `skill_provenance_check.py` + LESSONS.md.
