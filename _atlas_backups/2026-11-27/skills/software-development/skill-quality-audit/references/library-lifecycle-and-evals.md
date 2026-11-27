# Library lifecycle, restoration and behavioral evals

Detail for `skill-quality-audit` Steps 3, 5, 7 and 8. Read this when a skill has been archived,
when you are deciding whether to restore one, or when you actually want to *measure* whether a
skill helps instead of scanning it.

---

## 1. Restore is not a fix — pin is

`hermes curator` moves managed skills `active → stale (14d) → archived (30d)` from
`last_activity`. Measured on this box (2026-11):

| step | date | evidence |
|---|---|---|
| skill archived | ≤ 2026-08-05 | `.archive/api-bug-bounty-methodology/` |
| user restores it | 2026-11-17 | `hermes curator ledger` → `restore` entry |
| automatic pass re-archives it | 2026-11-22 | `ledger` → `archive` entry, 5 days later |
|lifetime uses at the time | — | `hermes curator usage` → 268 activity, last_activity 82d |

**Why:** `restore` copies the directory back but does **not** reset `last_activity`, so the skill is
immediately eligible again on exactly the grounds it was archived for. A restore-then-rearchive loop
is a 5-day fix that looks permanent.

**Durable fix:**

```bash
hermes curator restore <name>
hermes curator pin <name>        # "will bypass auto-transitions"
hermes curator status            # confirm: pinned (N): <name>
```

Docs (`user-guide/features/curator`) confirm the mechanism: *"Pinned skills and skills referenced by
any cron job (including paused/disabled jobs) are skipped entirely — treated like pin for
auto-transitions."* Two corollaries:

- **A skill a cron job depends on cannot be archived out from under the job** — cron references are
  an implicit pin. Do not "fix" a broken cron by re-pointing it away from an archived skill; the
  reference is why the skill survived.
- **Never-used skills get a grace floor** (`use_count == 0` is not archived until `stale_after_days`
  old). A skill with real historical use and an old `last_activity` is *worse off* than a brand-new
  one: it archives on the next pass.

**Before restoring, check who points at it.** Restore only if a live skill, MEMORY.md, SOUL.md or a
cron job names it. If nothing does, the prune was correct — leave it archived.

```bash
grep -rl "<archived-skill-name>" ~/.hermes/skills ~/.config/opencode/skills \
  --include=SKILL.md | grep -v '\.archive'
python3 ~/.hermes/scripts/skill_reference_check.py      # ARCHIVED-XSKILL verdict
```

`--dry-run` exists for the other direction: `hermes curator run --dry-run` produces the review
report without mutating the library, so a library-wide change can be previewed before a pass.

---

## 2. Cross-skill pointers (`ARCHIVED-XSKILL`)

`skill_reference_check.py` originally matched only pointers that *start* with
`references/|scripts/|templates/`. The shape that actually breaks after a prune is different:

```
See `api-bug-bounty-methodology/references/scope-verification-case-study.md` for the worked example.
```

The owner is the first path segment; the checker now resolves it against the live roots and
`.archive/`. It reports `ARCHIVED-XSKILL` only on a **three-way match** — the named skill is absent
from live roots, present in `.archive/`, and the exact file still exists there — which keeps it
near-zero false positive. Measured: 5 skills pointed into one archived skill (2 named reference
files, 3 body mentions); the intra-skill pass saw none of them.

What it does **not** flag, by design:

- A pointer into a *live* sibling (`ARCHIVED-XSKILL` needs an archived owner).
- A prose-qualified pointer already reported as `CROSS-SKILL`.
- A path of a tool or repo that is not a skill at all (`PROJECT-REL`).

---

## 3. SkillEvaluator — an independent static ruleset (live-verified 2026-11)

`NVIDIA/SkillEvaluator` (4.5K★, official NVIDIA org, paper arxiv 2608.20614 as **ACES**) ships
`skillevaluator` on PyPI: 500+ static rules across structure, security (YARA + its own patterns),
harness compatibility, PII and self-modification, plus Tier 2 semantic dedup and Tier 3 live
with/without evaluation.

One venv, pinned interpreter (never `/usr/bin/python3` — a distro bump orphans it):

```bash
uv venv --python 3.12 ~/.hermes/venvs/skilleval
uv pip install --python ~/.hermes/venvs/skilleval/bin/python "skillevaluator[tier3]"
```

Tier 1 runs keyless and is the only tier usable without provider credentials:

```bash
~/.hermes/venvs/skilleval/bin/skillevaluator tier1 ~/.hermes/skills/<category>/<skill>
# reports written to ./reports/skillevaluator-tier1.{json,html} — always run from a scratch cwd
```

Live-verified behaviour:

- `tier1 <skill-dir>` → exit 1 when any check fails; **the exit code is not the finding set** — parse
  `.finding_groups[]` (`severity`, `category`, `check_name`, `message`) from the JSON.
- On `skill-quality-audit` itself: 58 unique findings, 3 critical / 12 high — **all three criticals
  were YARA matches on attack signatures quoted inside the audit skill's own detection examples**
  (CobaltStrike, `coinhive.min.js`, PHP `preg_replace('/.*/e')`), plus `nsenter`, `.env` reads and
  keychain access. Same FP class as SkillSpector: security-teaching content scores as malware.
- `tier2`/`tier3` require a provider: `SKILL_EVAL_EMBEDDING_PROVIDER` / `SKILL_EVAL_LLM_PROVIDER`
  = `nv_build` or `openai` + the matching key. With neither configured, `health-check` reports
  `Public LLM provider: fail` and tier2 exits with `No provider is configured`. Do not treat that as
  a scan result.
- `tier2` takes **one skill directory** and an optional `--catalog FILE` for inter-skill comparison;
  passing a category directory errors (`PATH must be one skill directory containing SKILL.md`).

Use it as a **second** independent static scanner on skills you depend on, not as a library-wide gate
— arxiv 2609.17274 measured three scanners disagreeing on 23,702 of 61,990 skills, so agreement is
information and any single "0 findings" is unmeasured.

---

## 4. Why static gates cannot tell you whether a skill helps (ACES)

arxiv 2608.20614 (NVIDIA): across **145 real skills** from internal enterprise repos and public
catalogs, scan-only gates and an LLM-judge quality score agree at **Spearman ρ = 0.14**. They measure
complementary facets — structure/security vs whether the package helps a live agent — so neither
substitutes for the other.

On **947 scored paired cases** (58 of 64 production skills, four harnesses), running each task *with*
and *without* the skill under the same model, sandbox and grading policy:

- mean composite **Skill Lift 0.2134** (95% paired-case CI [0.1967, 0.2301]); outcome-only lift 0.1799
- composite lift **positive in 72.8%** of cases

Read that last number as the practical warning: roughly **one skill in four gives zero or negative
lift** under a real harness. That is the same conclusion as arxiv 2608.23067 (injected skills cut
Pass@2 by 1.3-4.2%) reached from a different direction, and the reason Step 5 asks for a
with/without baseline instead of a review.

The reporting unit is **Skill Lift** — target skill's added value for a fixed task, harness,
workspace and scorer. Do not report a lift number without naming those four.

---

## 5. Ranking vs set selection (Skill-Composer)

arxiv 2611.30715 (Skill-Composer): a marketplace interface returns *single* skills ordered by
relevance or popularity, which cannot answer the developer's real question — *which set covers the
task?* Highest-ranked skills are often functionally redundant while genuinely required capabilities
stay uncovered. The paper reframes acquisition as **set selection**: clarify the requirement into
atomic points, retrieve candidates, decompose their capabilities, discard overlaps progressively,
and **report the gaps**.

Applied here, at a 1-3 skill per task scale: when a task spans several capabilities, name the
non-redundant bundle and state what it does not cover. That is one sentence in a plan, and it
prevents the failure our routing checker cannot see — two loaded skills that cover the same 80% and
neither of which covers the missing 20%.
