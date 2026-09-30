---
name: hodaripay-fintech-brs
description: >-
  Business Requirement Specification (BRS) checklist for fintech features
  in the YucanPay repo. Use when writing or reviewing a feature spec for
  any payment, onboarding, compliance, or money-movement feature. Covers
  the compliance dimensions a fintech PRD must address that a normal
  software PRD doesn't: KYC/AML, reconciliation, idempotency, limits,
  audit trail, retention, and regulatory alignment. Triggers on "BRS",
  "business requirement", "PRD", "feature spec", "requirement document",
  "compliance checklist", "what does the spec need to cover".
metadata:
  last_modified: Sat, 26 Aug 2026 00:00:00 GMT
---

# YucanPay Fintech BRS Checklist

Every feature that touches money, identity, or compliance needs a BRS
(Business Requirement Specification) that goes beyond the usual software
PRD. This checklist covers the fintech-specific dimensions that a normal
PRD misses.

## When to use this checklist

- Writing a new feature spec for payments, payouts, onboarding, or
  compliance.
- Reviewing a PRD for a money-movement feature.
- Scoping a feature that involves KYC/AML, transaction limits, or audit
  requirements.

## The BRS checklist

### 0. State machine & user journey FIRST (repo planning rule)

Before filling in the rest, draw the process (repo AGENTS.md → "Planning
discipline — diagrams before implementation"):

- [ ] A **mermaid state machine** (states, transitions, terminal states,
  error/retry paths) is written into `docs/brs/NNNN-*.md` while still a plan.
- [ ] A **user journey** covers the happy path AND edge cases — multi-wallet
  display, sandbox-unsupported writes, already-assigned resources, offline/
  retry, already-held account kinds.
- [ ] Diagrams are **validated against the vendor docs + current code**
  (endpoints, error codes, columns, existing UI) before implementation —
  note what was verified live.
- [ ] Reference the canonical example `docs/brs/0002-shortcode-collection.md`
  (state machine + journey + multi-wallet display rules) and
  `docs/brs/0001-personal-wallet-kyc-otp.md`.

### 1. Regulatory & compliance

- [ ] **KYC/AML/CFT**: what identity verification is required? (ID docs,
  KRA PIN, business registration, UBO declaration).
- [ ] **Sanctions screening**: is the feature subject to sanctions checks?
  (ChoiceBank handles this for onboarding — confirm scope.)
- [ ] **Data protection**: Kenya Data Protection Act 2019 — lawful basis
  for processing, consent capture, retention schedule, DSAR workflow.
- [ ] **CBK regulations**: National Payment System Act / Regulations —
  does the feature require notification or approval from the Central Bank
  of Kenya?
- [ ] **Licensing scope**: does the feature fall outside the existing
  payment service provider scope?

### 2. Money movement & ledger

- [ ] **Double-entry / ledger**: does the feature move money? If so, does
  every credit have a corresponding debit? Are synthetic transaction rows
  written for all balance changes?
- [ ] **Reconciliation**: how is the balance verified against ChoiceBank's
  authoritative response? When does reconciliation run? What happens on
  mismatch?
- [ ] **Idempotency**: is every outbound API call idempotent? What is the
  idempotency key strategy? What happens on retry?
- [ ] **Decimal precision**: money is `numeric(18,2)` KES. Are all
  calculations exact (integer cents or Decimal)? No `double` arithmetic.
- [ ] **Partial failure**: if a batch has 50 rows and row 12 fails, what
  happens to rows 1-11 and 13-50? Is the batch resumable?

### 3. Limits & controls

- [ ] **Transaction limits**: per-transaction min/max? Daily/weekly caps?
  Per-rail limits? (Personal wallet: KES 20,000/day outbound, KES
  300,000 holding.)
- [ ] **Velocity controls**: rate limiting per user/IP/account? Brute-force
  protection? (Login: 5 attempts → 5 min lockout. PIN: same.)
- [ ] **Maker-checker**: does the feature require a second approver?
  (Reversals/refunds: initiator ≠ approver.)
- [ ] **Kill switch**: can the feature be disabled without a deploy?
  (`app_settings` table — not yet wired.)

### 4. User journeys & roles

- [ ] **Role scoping**: which roles can trigger this feature? Which roles
  can approve it? Which roles can view audit data?
- [ ] **Error UX**: what does the user see on failure? (Never raw server
  messages — map error codes to user-friendly text client-side.)
- [ ] **OTP channel**: which channel for this operation? (SMS on mobile,
  WhatsApp/email on web. ChoiceBank OTP for transfers, Twilio for auth.)
- [ ] **PIN requirement**: does this operation require transaction PIN
  authorization?

### 5. Audit & observability

- [ ] **Audit trail**: every money-movement or lifecycle action is logged
  with `who/what/when/why`. Which table? Which fields?
- [ ] **Webhook/callback**: does this operation produce a callback? Is it
  idempotent? Is the signature verified?
- [ ] **Logging**: what is logged server-side? Is PII redacted? (Phone
  numbers, names, amounts must be masked.)
- [ ] **Alerting**: does this operation need monitoring? What are the
  failure thresholds?

### 6. Data retention & security

- [ ] **Retention schedule**: how long is raw response data kept? When is
  it purged?
- [ ] **PII minimization**: is the minimum necessary PII returned to the
  client? (No `raw_response` to merchants — M5.)
- [ ] **Encryption at rest / in transit**: is the data encrypted? (Neon
  handles at-rest; TLS for in-transit.)
- [ ] **Token storage**: where are auth tokens stored? (Mobile: Keychain.
  Web: localStorage — M6 risk.)

### 7. Edge cases & failure modes

- [ ] **Timeout behavior**: what happens when ChoiceBank times out?
  (`TimeoutException` → `ChoiceBankException` → 502.)
- [ ] **Callback delay**: if the callback is delayed, how does the user
  know the status? (Polling, retry, SLA alerting.)
- [ ] **Duplicate prevention**: what prevents duplicate operations?
  (IdempotencyService, unique constraints.)
- [ ] **Rollback**: can the operation be reversed? What is the reversal
  flow? (No direct reversal endpoint — internal workflow with
  two-person approval.)

### 8. Testing & verification

- [ ] **Sandbox verification**: has the flow been verified end-to-end in
  the ChoiceBank sandbox?
- [ ] **Regression tests**: are there tests for the happy path and the
  failure modes?
- [ ] **Edge case tests**: what about zero amounts, max amounts, duplicate
  submissions, concurrent access?

## How to write a BRS

1. Copy this checklist into a new file: `docs/brs/NNNN-feature-name.md`.
2. Fill in every section — mark items as N/A only with explicit
   justification.
3. Link the BRS to the ADR that captures the architectural decision.
4. The BRS is reviewed before the PR is opened.

## Key facts

- BRS is not a code artifact — it's a requirements artifact. It lives in
  `docs/brs/` (to be created when the first BRS is written).
- The ADR captures the *decision*; the BRS captures the *requirements*.
- For features already shipped, a BRS can be written retroactively to
  document what was implemented (useful for audit/compliance).
