---
name: hodaripay-adr
description: >-
  ADR workflow for the YucanPay repo (BongweKE/hodaripay). Use when making
  or reviewing an architecturally significant decision: new dependency,
  new service, contract change, money-flow change, new auth/security
  mechanism, new infrastructure or CI/CD pattern. Triggers on "ADR",
  "architecture decision", "decision record", "why did we choose",
  "should we change", "decision log". Encodes the repo's MADR convention
  (format, location, gate, when to write one).
metadata:
  last_modified: Sat, 26 Aug 2026 00:00:00 GMT
---

# YucanPay ADR Workflow

The project's Architecture Decision Records live in `docs/decisions/` and
follow the [MADR](https://adr.github.io/madr/) 4.0 format. They are
**immutable once accepted** — if a decision is reversed, a new ADR
supersedes the old one; never silently rewrite an existing ADR.

## When to write an ADR

Write an ADR when the change is **architecturally significant**:
- New external dependency or service integration.
- New data model or contract change (API, wire format, database schema).
- New security or authentication mechanism.
- New payment rail or money-flow change.
- New CI/CD or deployment pattern.
- New infrastructure or hosting decision.

**Do NOT write an ADR for**: bug fixes, copy changes, minor refactors,
adding tests, dependency version bumps.

## The format — MADR 4.0

Every ADR follows the template at `docs/decisions/adr-template.md`. The
core sections:

1. **Frontmatter**: `status` (proposed/accepted/rejected/deprecated/
   superseded), `date`, `decision-makers`, `consulted`, `informed`.
2. **Context and Problem Statement**: 2-3 sentences describing the context
   and the question being answered.
3. **Decision Drivers**: forces/concerns that shaped the decision.
4. **Considered Options**: 2-4 options that were evaluated.
5. **Decision Outcome**: which option was chosen and why.
6. **Consequences**: good and bad outcomes.
7. **Confirmation**: how to verify the decision is implemented correctly.
8. **Pros and Cons of the Options**: tradeoff analysis per option.
9. **More Information**: links, references, notes on re-visit conditions.

## Where they live

```
docs/decisions/
├── README.md                          # Decision log index
├── adr-template.md                    # MADR 4.0 template (copy this)
├── 0001-record-architecture-decisions.md
├── 0002-use-choicebank-baas.md
├── ...
└── NNNN-title-with-dashes.md          # Next ADR (use next sequential NNNN)
```

## How to create a new ADR

1. Determine the next sequential number from the README index.
2. Copy `docs/decisions/adr-template.md` to
   `docs/decisions/NNNN-title-with-dashes.md`.
3. Fill in the template — all sections are required for architecturally
   significant decisions.
4. Update `docs/decisions/README.md` — add a row to the decision log table.
5. Link the ADR in your PR description: "Decision: ADR-NNNN".
6. Run `bash scripts/check-adrs.sh` to validate before committing.

## Naming conventions

- Filename: `NNNN-title-with-dashes.md` (lowercase, dashes, no `.md`
  duplicate).
- Title in the H1: short, descriptive, action-oriented (e.g. "Use
  ChoiceBank BaaS for all banking and payment rails").
- Numbers are sequential across the whole project (not per-category).

## Status lifecycle

```
proposed → accepted → (optionally: superseded by ADR-NNNN)
proposed → rejected
accepted → deprecated
```

- **proposed**: under discussion, not yet decided.
- **accepted**: decision is in effect.
- **rejected**: considered and declined.
- **deprecated**: no longer relevant (superseded or obsolete).
- **superseded by ADR-NNNN**: replaced by a newer decision.

## CI gate

`scripts/check-adrs.sh` validates:
1. Each ADR has a sequential NNNN number.
2. Each ADR contains the required MADR headings.
3. Each ADR is listed in `README.md`.
4. No duplicate NNNN numbers.

The script runs in `pr.yml` — ADR validation is part of the PR gate.

## Linking ADRs to PRs

Every ADR should be linked to the PR that proposes/accepts it. The ADR
itself is committed in the same PR as the code change it documents. The
PR description should reference the ADR number.

## Key facts

- ADRs are versioned in git alongside the code they document.
- The README.md index is the single source of truth for the decision log.
- `docs/decisions/adr-template.md` is the template — copy it, don't edit
  the template itself.
- The `scripts/check-adrs.sh` gate must pass before merging.
