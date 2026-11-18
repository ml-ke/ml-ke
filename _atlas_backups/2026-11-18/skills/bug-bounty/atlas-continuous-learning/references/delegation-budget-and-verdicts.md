# Delegation Budget Gate & Schema-Validated Verdicts

Sources: `cloudflare/security-audit-skill` (4.5K★ official Cloudflare org — the skill that
seeded Cloudflare's fleet-wide vulnerability harness, blog.cloudflare.com/build-your-own-vulnerability-harness)
and the Hermes docs pages `features/delegation` + `guides/delegation-patterns` (Sep 15 2026).
Both are high-reputability; the budget-gate rule is Cloudflare's, the `output_schema`
mechanism is Hermes-native and was read from the docs, not inferred.

## 1. The budget gate

The failure it prevents: dispatching a hunting wave that exhausts the budget before any
verification can run, so every candidate ends up unverified and the whole run is worthless.

**Reserve-then-assign order:**

1. Estimate the unit count from recon (one unit ≈ one hunter assignment).
2. Reserve, in this order, before launching anything:
   - 1 post-wave coverage critic
   - 1 *distinct* final-clean critic
   - ~1 verifier per expected candidate (when unsure, 30% of the balance after critic reserve)
3. Spend the remainder on hunters, highest-priority units first.
4. After each wave, re-reserve the next wave's critic. If it no longer fits, mark the
   remaining units `deferred` with reason `budget_cannot_reserve_critics_and_validation`
   and let the retained final critic record the gap.

**Terminal states** — a run ends in exactly one of these, never "mid-phase stop":

- all artifacts written and the validators pass, or
- `run_status: "incomplete"` with an exact reason, disclosed in the report.

Incomplete reasons worth reusing verbatim: `budget_cannot_fund_reconnaissance_and_reserves`,
`critic_budget_exhausted`, `validation_budget_exhausted`.

**Never exceed a user-set budget silently.** If a strict budget cannot cover every
candidate, stop hunting, validate in fingerprint order while the budget lasts, and keep each
unvalidated fingerprint linked to a `candidate` unit with its unresolved reason.

## 2. Schema-validated child output

`delegate_task` takes an optional `output_schema` (a JSON Schema) per task. The child sees
it as an output contract ("return ONLY the JSON value — no prose, no code fence"). The
parent validates the answer and, on failure, sends exactly ONE bounded correction turn
carrying the validation errors — the schema is not re-pasted. The result then gains:

- `schema_valid: true|false`
- `schema_errors: [...]` on failure
- `schema_note` saying the text is unvalidated

A contract miss does **not** discard the child's work: status stays `completed` with the raw
text in `summary`, so extract what you need rather than re-running a long task. Prose or a
code fence around otherwise-valid JSON is tolerated.

Keep schemas forgiving — require only fields you will actually read. Over-strict schemas are
themselves an "excessive procedure" problem (arxiv 2608.11888).

### Always pass everything the child needs

A subagent starts with **zero** conversation history. The `goal` + `context` fields are its
entire briefing: file paths, exact curl patterns, scope rules, output format, honesty rules,
and the lesson-bank path. If the parent has a resolved workspace, project context files
(`.hermes.md > AGENTS.md > CLAUDE.md > .cursorrules`) are embedded automatically — SOUL.md
is not.

### Child lifetimes

Background terminal processes belong to the agent that started them. Closing a child
terminates its processes, including work from earlier turns, without touching the parent's
or siblings'. Tell each child to wait for its own builds/tests before returning its final
summary; to keep a watcher or server alive, start it in the parent.

## 3. Verdict contract (feeds the findings ledger)

A verifier child must be a *fresh* agent that did not hunt the candidate, and its job is to
**refute**, not to agree. Ask for exactly one JSON object:

```json
{"decision": "confirmed|needs_validation|rejected", "record": { ... }}
```

Verifier rules worth restating inside the prompt:

1. Re-read every cited current source location; never trust the hunter's line numbers.
2. Independently reproduce the minimum observed result where safely possible; stop there.
3. A corrected record replaces the hunter's wording; preserve the fingerprint for the same
   root cause across every state.
4. `needs_validation` is **never** a parking place for a speculative idea — it requires one
   exact, decisive, genuinely external fact plus a non-destructive plan.
5. Demote a proposed `confirmed` to `needs_validation` when a specific deployment or runtime
   fact remains unknown.
6. Discard a malformed or prose-wrapped result without repairing it; re-run with a fresh
   verifier if budget permits, otherwise it stays an unvalidated candidate.

Validate the aggregated ledger with:

```bash
python3 ~/.hermes/scripts/findings_ledger.py --check ~/Dev/REPORTS/<Target>/findings.json
```

## 4. Minimal worked call

```python
delegate_task(tasks=[{
  "goal": "Refute or confirm candidate <fingerprint>. Return only the JSON object.",
  "context": """Target artifacts: ~/Dev/REPORTS/<Target>/raw_*.txt
Scope: <in-scope hosts only>. Do NOT touch production. No writes outside /tmp.
Re-read the cited source at <path>:<line>. Reproduce the minimum observed result.
Decisions: confirmed (needs full trace + pasted output + reproduced=true + crossed
boundary) | needs_validation (ONE exact unresolved fact, NO severity) | rejected
(refuting reason). Lesson bank: ~/Dev/ATLAS-LEARNINGS/LESSONS.md §01.""",
  "output_schema": {
    "type": "object",
    "properties": {
      "decision": {"type": "string"},
      "record": {"type": "object"}
    },
    "required": ["decision", "record"]
  }
}])
```

## 5. Why this beats prose reports

Prose reports cannot enforce that an unresolved claim carries no severity — which is exactly
how Fireblocks MPC 004/005 and the Rapyd spec-only findings got written up as submittable
(see `LESSONS.md` §01). A schema-validated verdict plus the `findings_ledger.py` validator
makes that mistake a hard error instead of a judgement call.
