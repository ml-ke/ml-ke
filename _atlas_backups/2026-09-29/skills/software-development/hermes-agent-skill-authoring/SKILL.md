---
name: hermes-agent-skill-authoring
description: "Author in-repo SKILL.md: frontmatter, validator, structure."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [skills, authoring, hermes-agent, conventions, skill-md]
    related_skills: [writing-plans, requesting-code-review]
---

# Authoring Hermes-Agent Skills (in-repo)

## Overview

There are two places a SKILL.md can live:

1. **User-local:** `~/.hermes/skills/<maybe-category>/<name>/SKILL.md` — personal, not shared. Created via `skill_manage(action='create')`.
2. **In-repo (this skill is about this case):** `/home/bb/hermes-agent/skills/<category>/<name>/SKILL.md` — committed, shipped with the package. Use `write_file` + `git add`. `skill_manage(action='create')` does NOT target this tree.

## When to Use

- User asks you to add a skill "in this branch / repo / commit"
- You're committing a reusable workflow that should ship with hermes-agent
- You're editing an existing skill under `/home/bb/hermes-agent/skills/` (use `patch` for small edits, `write_file` for rewrites; `skill_manage` still works for patch on in-repo skills, but not for `create`)

## Required Frontmatter

Source of truth: `tools/skill_manager_tool.py::_validate_frontmatter`. Hard requirements:

- Starts with `---` as the first bytes (no leading blank line).
- Closes with `\n---\n` before the body.
- Parses as a YAML mapping.
- `name` field present.
- `description` field present, ≤ **1024 chars** (`MAX_DESCRIPTION_LENGTH`).
- Non-empty body after the closing `---`.

Peer-matched shape used by every skill under `skills/software-development/`:

```yaml
---
name: my-skill-name               # lowercase, hyphens, ≤64 chars (MAX_NAME_LENGTH)
description: Use when <trigger>. <one-line behavior>.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [short, descriptive, tags]
    related_skills: [other-skill, another-skill]
---
```

`version` / `author` / `license` / `metadata` are NOT enforced by the validator, but every peer has them — omit and your skill sticks out.

## Size Limits

- Description: ≤ 1024 chars (enforced).
- Full SKILL.md: ≤ 100,000 chars (enforced as `MAX_SKILL_CONTENT_CHARS`, ~36k tokens).
- Peer skills in `software-development/` sit at **8-14k chars**. Aim for that range. If you're pushing past 20k, split into `references/*.md` and reference them from SKILL.md.

## Peer-Matched Structure

Every in-repo skill follows roughly:

```
# <Title>

## Overview
One or two paragraphs: what and why.

## When to Use
- Bulleted triggers
- "Don't use for:" counter-triggers

## <Topic sections specific to the skill>
- Quick-reference tables are common
- Code blocks with exact commands
- Hermes-specific recipes (tests via scripts/run_tests.sh, ui-tui paths, etc.)

## Common Pitfalls
Numbered list of mistakes and their fixes.

## Verification Checklist
- [ ] Checkbox list of post-action verifications

## One-Shot Recipes (optional)
Named scenarios → concrete command sequences.
```

Not every section is mandatory, but `Overview` + `When to Use` + actionable body + pitfalls are the minimum for the skill to feel like a peer.

## Directory Placement

```
skills/<category>/<skill-name>/SKILL.md
```

Categories currently in repo (confirm with `ls skills/`): `autonomous-ai-agents`, `creative`, `data-science`, `devops`, `dogfood`, `email`, `gaming`, `github`, `leisure`, `mcp`, `media`, `mlops/*`, `note-taking`, `productivity`, `red-teaming`, `research`, `smart-home`, `social-media`, `software-development`.

Pick the closest existing category. Don't invent new top-level categories casually.

## Workflow

1. **Survey peers** in the target category:
   ```
   ls skills/<category>/
   ```
   Read 2-3 peer SKILL.md files to match tone and structure.
2. **Check validator constraints** in `tools/skill_manager_tool.py` if unsure.
3. **Draft** with `write_file` to `skills/<category>/<name>/SKILL.md`.
4. **Validate locally**:
   ```python
   import yaml, re, pathlib
   content = pathlib.Path("skills/<category>/<name>/SKILL.md").read_text()
   assert content.startswith("---")
   m = re.search(r'\n---\s*\n', content[3:])
   fm = yaml.safe_load(content[3:m.start()+3])
   assert "name" in fm and "description" in fm
   assert len(fm["description"]) <= 1024
   assert len(content) <= 100_000
   ```
5. **Git add + commit** on the active branch.
6. **Note:** the CURRENT session's skill loader is cached — `skill_view` / `skills_list` will not see the new skill until a new session. This is expected, not a bug.

## Cross-Referencing Other Skills

`metadata.hermes.related_skills` unions both trees (`skills/` in-repo and `~/.hermes/skills/`) at load time. You CAN reference a user-local skill from an in-repo skill, but it won't resolve for other users who clone the repo fresh. Prefer referencing only in-repo skills from in-repo skills. If a frequently-referenced skill lives only in `~/.hermes/skills/`, consider promoting it to the repo.

## Editing Existing In-Repo Skills

- **Small fix (typo, added pitfall, tightened trigger):** `skill_manage(action='patch', name=..., old_string=..., new_string=...)` works fine on in-repo skills.
- **Major rewrite:** `write_file` the whole SKILL.md. `skill_manage(action='edit')` also works but requires supplying the full new content.
- **Adding supporting files:** `write_file` to `skills/<category>/<name>/references/<file>.md`, `templates/<file>`, or `scripts/<file>`. `skill_manage(action='write_file')` also works and enforces the references/templates/scripts/assets subdir allowlist.
- **Always commit** the edit — in-repo skills are source, not runtime state.

## Writing Skills That Trigger and Work (agentskills.io + Anthropic best practices)

The structure above makes a skill *valid*. This section makes it *useful*. Distilled from agentskills.io skill-creation docs and Anthropic's agent-skills engineering guidance.

### Description is the whole trigger burden
At startup only `name` + `description` are loaded (median ~80 tokens per skill across Anthropic's official skills). The body loads only when the description triggers. If the description doesn't trigger, the skill doesn't exist. Rules:
- **Imperative phrasing**: "Use this skill when..." not "This skill does..."
- **User intent, not implementation**: the agent matches against what the user asked for, so describe the *need*, not your internals
- **Err pushy**: explicitly list contexts, including "even if they don't mention X"
- **Concise**: a few sentences to a short paragraph; hard cap 1024 chars
- **Test triggering**: build ~20 eval queries (8-10 should-trigger, 8-10 should-not). The valuable negatives are **near-misses** — queries sharing keywords but needing something different ("update formulas in my Excel budget" vs a CSV-analysis skill). Run each 3x in clean contexts; trigger rate ≥ 0.5 = pass. Agents only consult skills for tasks needing knowledge beyond base capability — a skill that adds nothing over base behavior has no reason to exist.

### Context is a public good
The context window is shared with the system prompt, history, and other skills' metadata. **Default assumption: the model is already smart.** Only add what it would get wrong without the skill. Challenge every paragraph: *"Would the agent get this wrong without this instruction?"* — if no, cut it. "Does this paragraph justify its token cost?" No explanation of what a PDF is, how HTTP works, or what a database migration does.

### Calibrate control
- **Defaults, not menus**: when several tools/approaches exist, recommend ONE default. Option buffets waste agent time (the agent tries several before finding the one that works).
- **Match specificity to fragility**: exact commands for fragile steps; goals for robust ones.
- **Procedures over declarations**: "run X, check Y, if Z do W" beats "handle errors appropriately".
- **Refine with real execution**: run the skill against real tasks, feed ALL results back (false positives, misses, what to cut) — not just failures. Read execution traces, not only outputs: wasted steps usually mean vague instructions, inapplicable instructions, or too many options without a default.

### Structure that survives activation
- `## Gotchas` or `> ⚠️` blocks for critical warnings — never bury them mid-body
- Output-format templates when the format matters (MT smell)
- Checklists for multi-step workflows + progress tracking (NPT smell)
- **Validation loops (plan-validate-execute)**: after producing output, verify by running it — no one-shot trust (NVS/EWP smells)
- **Progressive disclosure**: SKILL.md < 500 lines / < 5000 words; move detail to `references/` loaded on demand

### Eval-driven iteration (for skills that must be reliable)
- `evals/evals.json` in the skill dir: test cases `{prompt, expected_output, files}`
- Run each case twice — **with the skill and without it (baseline)** — to prove the skill adds value
- Results in `evals/iteration-N/<case>/{with_skill,without_skill}/{outputs, timing.json, grading.json}` + `benchmark.json`; each run in a clean context (subagents or fresh sessions)
- Grading: assertions over outputs, aggregate, analyze patterns, review with a human, iterate

### Scripts
- One-off commands: reference `uvx`/`pipx`/`npx`/`bunx`/`deno run`/`go run` directly (auto-resolve deps, pin versions) instead of bundling
- Bundled `scripts/`: self-contained or document deps; helpful error messages; handle edge cases

## Common Pitfalls

1. **Using `skill_manage(action='create')` for an in-repo skill.** It writes to `~/.hermes/skills/`, not the repo tree. Use `write_file` for in-repo creation.

2. **Leading whitespace before `---`.** The validator checks `content.startswith("---")`; any leading blank line or BOM fails validation.

3. **Description too generic.** Peer descriptions start with "Use when ..." and describe the *trigger class*, not the one task. "Use when debugging X" > "Debug X".

4. **Forgetting the author/license/metadata block.** Not validator-enforced, but every peer has it; omitting makes the skill look half-finished.

5. **Writing a skill that duplicates a peer.** Before creating, `ls skills/<category>/` and open 2-3 peers. Prefer extending an existing skill to creating a narrow sibling.

6. **Expecting the current session to see the new skill.** It won't. The skill loader is initialized at session start. Verify in a fresh session or via `skill_view` using the exact path.

7. **Linking to skills that don't exist in-repo.** `related_skills: [some-user-local-skill]` works for you but breaks for other clones. Prefer only in-repo links.

## Verification Checklist

- [ ] File is at `skills/<category>/<name>/SKILL.md` (not in `~/.hermes/skills/`)
- [ ] Frontmatter starts at byte 0 with `---`, closes with `\n---\n`
- [ ] `name`, `description`, `version`, `author`, `license`, `metadata.hermes.{tags, related_skills}` all present
- [ ] Name ≤ 64 chars, lowercase + hyphens
- [ ] Description ≤ 1024 chars and starts with "Use when ..."
- [ ] Total file ≤ 100,000 chars (aim for 8-15k)
- [ ] Structure: `# Title` → `## Overview` → `## When to Use` → body → `## Common Pitfalls` → `## Verification Checklist`
- [ ] `related_skills` references resolve in-repo (or are explicitly OK to be user-local)
- [ ] `git add skills/<category>/<name>/ && git commit` completed on the intended branch
