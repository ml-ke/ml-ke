# Personal current-account onboarding — verified against official docs (2026-09-05)

Fetched from the Choice Bank GitBook markdown twins (`.md` suffix) on
2026-09-05. This is the authoritative flow for current accounts (C001) — it
differs from wallet (C002, OTP-gated) and SME (B001, OTP + 72 h).

## Flow (docs: account/current-account.md + account/current-account/onboard.md)

1. `POST /onboarding/submitOnboardingRequest` — text KYC only.
   Required params (from the docs table): `userId`, `firstName`, `lastName`,
   `birthday` (yyyy-mm-dd), `address`, `gender` (0 F / 1 M), `countryCode`,
   `mobile`, `idType` (101 Kenya national ID · 102 Alien ID · 103 Passport),
   `idNumber`, `kraPin`, `employmentStatus` (A employee · B self-employed ·
   C unemployed · D employer · E student · F others), plus income/address
   fields (truncated in the fetched page — `monthlyIncome` and
   `businessIndustry`/`specifyIndustry` appear in `getUserKyc`). Optional:
   `middleName`, `email`. Response: `{onboardingRequestId}`.
2. `POST /onboarding/uploadMedia` — `{onboardingRequestId, mediaType,
   mediaBase64, contentType? (pdf/image)}`. **Within 30 minutes of step 1 or
   the request expires.** Required media by idType:
   - KYCF00001 front of national ID (idType 101) · KYCF00002 back (101)
   - KYCF00003 passport (103) · KYCF00004/00005 alien ID front/back (102)
   - KYCF00006 **selfie — "Must be verified via Smile ID through your
     front-end"** (mandatory for every idType).
   - Since **2026-01-19** the old KYCF00007 selfie-video is no longer accepted.
     Image specs (docs wallet-account/image-specifications.md): ≥1500 px on the
     long side for ID cards, ≥2200 px passports, 600 kB–10 MB, upright
     orientation, whole document in frame.
3. Wait for review: status → `2-PROCESSING`.
4. Result via callback `0001` (Personal Onboarding Result Notification) or
   `POST /onboarding/getOnboardingStatus` `{userId | onboardingRequestId |
   mobile}` → `{onboardingRequestId, onboardingStatus, rejectionReasonId/msg,
   accountId, accountType}`.

## Status & diagnostics

- `POST /onboarding/getUserKyc {onboardingRequestId}` returns `status`
  (1 submitted · 2 processing · 3 passed · 4 rejected · 5 account closed ·
  9 manual review) and **`profileCheck`** (0 not checked · 1 submitted ·
  2 validated · 3 declined · 4 processing) + `profileCheckResultCode/Text`.
  `profileCheck: 0` with media already uploaded = the Smile ID job never ran —
  the request is parked at 9 and will not open.
- `POST /onboarding/personal/getKycMediaList {onboardingRequestId}` →
  `{mediaList:[{mediaType, fileTypeId, mediaUrl}]}`.
- `POST /onboarding/getOnboardingRequestId {userId? | mobile?}`.
- `POST /onboarding/cancelOnboardingRequest {onboardingRequestId}` only works
  for status `1` — a status-9 request needs Choice ops to close.

## Smile ID handoff (open question, verified 2026-09-03 live investigation)

Choice requires Smile ID verification in the partner front-end; exactly how the
evidence reaches Choice (uploadMedia of the Smile-captured selfie vs the Smile
job alone advancing `profileCheck`) was NOT settled by the docs — test live
before re-enabling. The HodariPay repo implements the backend for this in
`SmileService` (ADR-0028, migration 0028) and keeps current-account onboarding
gated behind `CURRENT_ACCOUNT_ONBOARDING_ENABLED` + Smile env config.

## Fetch recipe (for future doc checks)

```bash
curl -sL https://choice-bank.gitbook.io/choice-bank/llms.txt          # full page map
curl -sL https://choice-bank.gitbook.io/choice-bank/account/current-account/onboard.md
curl -sL https://choice-bank.gitbook.io/choice-bank/appendix/type-status-ids.md
curl -sL https://choice-bank.gitbook.io/choice-bank/notifications/callback-notifications.md
```
Every GitBook page has a `.md` twin and the footer links llms.txt. Don't trust
web_search snippets for these pages — they are barely indexed.
