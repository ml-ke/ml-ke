# Collection-level seams — auditing a skill *library*, not a skill file

Source: arxiv 2609.13321 **SkillSeam: Six Principles for Auditing Agent Skill Collections**
(empirical, Sep 2026), reinforced by 2609.09233 (Subagents vs Agent Skills), 2609.17274
(OpenClaw registry governance) and 2609.14079 (SkillSecurer).

The thesis: *a folder of competent skills is not yet a reliable system.* Skills rarely fail
alone — they fail at the **seams**. The 26-smell taxonomy audits one file; this audits the
relationships between files. Run it after any library-scale change (bulk install, rename,
mirror/bridge sync, catalog growth), not on every edit.

## When to run this

- A bulk install, mirror, or bridge re-sync happened (names can now collide).
- The library grew past a few dozen skills, or a whole category moved.
- Routing feels wrong: the wrong skill loads, two skills fight, or the same skill loads twice.
- Quarterly, as the counterpart to per-file audits.

Do **not** run this per-file on trivial edits — that is the excessive-procedure trap
(arxiv 2608.11888).

## The six principles — mechanism, observable, perturbation test

Each row is a *testable* property with a measured failure channel, not authoring advice.

| # | Principle | Failure mechanism | Observable | Measured effect of breaking it |
|---|---|---|---|---|
| 1 | **Persistence gradient** | Flattening the summary → detail hierarchy puts everything in the always-loaded layer | loaded-skill token count after a task | **+60% loaded-skill tokens** |
| 2 | **System coherence** | Dangling anchors — pointers to renamed/moved/absent files or skills | unresolved `references/`/skill links | **+64% total tokens, −3.1pp accuracy** |
| 3 | **Regime gating** | Synonymous aliases / duplicate directories load the same capability twice | noncanonical routes; duplicate `name:` values | **noncanonical routes 0/32 → 15/32**, half of matched paraphrase pairs flipped |
| 4 | **Orthogonal coverage** | Overlapping lanes: two skills claim the same capability | ownership conflicts reported | **0/16 → 14/16** |
| 5 | **Flow** | Bland triggers: descriptions that don't say *when* | routing conflicts; loaded tokens | **3/32 → 30/32 conflicts, 3.7× tokens** |
| 6 | **Granularity discipline** | Granularity mis-mix (some skills too coarse, some too fine, in one collection) | per-skill task accuracy | **−12.5pp accuracy — the single largest drop** |

Principle 3 is the one most likely to bite us: every mirrored/bridged library (`atlas-*`
copies, multi-agent installs, `gh skill install` into a second root) is an alias farm.
Principle 6 is the one with the largest measured cost and the least visible symptom.

## Collection audit procedure

1. **Inventory** — enumerate every discoverable skill root and every `name:` value.
   Duplicate `name:` across roots = regime-gating failure (principle 3). Check with a
   name-set intersection, not by eyeballing directory listings.
2. **Anchor check** — for every relative pointer in a SKILL.md (`references/x.md`,
   `scripts/y.py`, "see skill Z"), confirm the target exists. Missing target = principle 2.
3. **Layer check** — confirm SKILL.md carries only the summary + decision rules and that
   detail lives in lazily-loaded files (principle 1).
4. **Lane check** — list capabilities and note any claimed by two skills; assign exactly one
   owner (principle 4).
5. **Trigger check** — for the collection's skills that must route reliably, run the trigger
   evals in Step 5 of the parent skill: near-miss negative queries are where bland triggers
   show up (principle 5).
6. **Granularity check** — sample tasks and ask whether each was solved by a skill that is
   the right size; a mis-mix is invisible per-file and expensive (principle 6).
7. **Record** — log the numbers (loaded tokens, unresolved anchors, duplicate names,
   ownership conflicts) so the next run can diff them. A seam audit without numbers is opinion.

## Invocation is part of the seam (arxiv 2609.09233)

How a skill is *invoked* changes whether it helps. Loading a heavy skill's instructions into
the main context degrades as the horizon grows (reasoning quality falls as context
accumulates); invoking the same skill package in a **fresh subagent context** wins when the
skill exposes a clear input/output contract, at the cost of coordination tokens.

- Long-horizon / multi-step work → execute the skill as a subagent task and pass the contract
  as a schema (see `delegate_task`'s per-task `output_schema`).
- Short, single-turn work → load it in the main context; the coordination overhead is not repaid.
- A skill with no clear input/output contract cannot be invoked as a subagent — that alone is
  a reason to fix its interface.

## Verdicts at registry scale — never trust one scanner

The OpenClaw registry study (arxiv 2609.17274) measured 61,990 skills across three ClawHub
snapshots:

- the three security scanners **disagreed on 23,702** of the 61,990 skills they all covered;
- after human adjudication, weighted scanner sensitivity ranged **21.67% – 61.06%**;
- **85.06%** of readable skills carried privilege evidence (shell / network / credential /
  file / process capability) while **77.86%** had zero stars and zero comments;
- attention is concentrated: the top 10% of skills took **46.93%** of downloads, and no simple
  metadata feature (size, downloads) predicted continued listing once cohort and age were
  controlled.

SkillSecurer (arxiv 2609.14079) adds the detection-side number: with an LLM in the loop it was
the only scanner of the compared set to reach a 100% injection-detection rate, and it found
latent injection risk in **more than 17%** of popular skills sampled from a large public registry.

**Operating rules that follow:**

- A single scanner's verdict is not evidence. Treat "0 findings" as *unmeasured*, not *clean*.
- Use ≥2 independent scanners on first-party skills; adjudicate disagreements by hand.
- Static-only scanning is the low-sensitivity mode. For skills we author, spend an LLM-assisted
  pass on the ones that changed — bounded to the changed set, not the whole catalog.
- Scan the *harness* too, not only the skills: agent configs, MCP definitions and instruction
  files are part of the same supply chain (arxiv 2609.07360).

## One-screen checklist

```
[ ] every discoverable skill root enumerated; name sets intersected (no duplicate names)
[ ] every relative pointer resolves (references/, scripts/, "see skill X")
[ ] SKILL.md = summary + decision rules only; detail lazily loaded
[ ] exactly one owner skill per capability
[ ] trigger-shaped descriptions for skills that must route reliably
[ ] granularity reviewed against real tasks (largest measured accuracy risk)
[ ] changed-set scanned twice, independently; disagreements adjudicated by hand
[ ] numbers logged so the next run can diff
```
