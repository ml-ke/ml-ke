---
name: smileid-kyc
description: >-
  Smile ID (docs.usesmileid.com) identity-verification integration reference.
  Use when implementing or debugging Smile ID in ANY project — especially the
  ChoiceBank current-account mandate (selfie KYCF00006 must be Smile-ID-verified
  via the partner front-end; BRS-0003/ADR-0023). Covers the v3 API
  (POST /v3/token → JWT; POST /v3/biometric_kyc async; webhook with
  Response-Signature HMAC), the v12 hosted web flow (window.SmileIdentity),
  sandbox vs production, result codes (1012/1022/1023), callback URLs, and the
  verified doc-URL map. Load before writing token/session/callback code.
---

# Smile ID — v3 API + v12 Web integration (verified 2026-09-05)

Official docs are GitBook-style with markdown twins: append `.md` to any page
URL and the site index is at https://docs.usesmileid.com/llms.txt.

## The model (verify against live docs before shipping)

- **v3 token**: `POST /v3/token` with headers `smileid-partner-id` (numeric,
  no leading zeros) + `smileid-api-key` → `{token}` = short-lived JWT (15 min,
  environment-specific). Pass it as `SmileID-Token` on all v3 calls.
  Optional multipart body: `user_id`, `product`, `partner_params`,
  `payload` (JSON string binding country/id_type/id_number/given_names/
  last_name/email/phone_number/callback_url/consent — PII is validated then
  replaced with opaque `pii_`-prefixed refs; token-bound fields cannot be
  altered by the client and override body fields on downstream endpoints).
- **Async verification products** (e.g. biometric_kyc): submit multipart to
  `/v3/biometric_kyc` (selfie_image + 6-8 liveness_images + consent + country/
  id_type/id_number + user_details + callback_url) → `202 {job_id, user_id,
  created_at}` → result POSTed to callback_url when done.
- **Webhook**: server must respond 2xx within 35 s; Smile retries up to 3
  times; at-least-once, unordered. Auth = headers `Response-Signature` +
  `Response-Timestamp`; signature = base64(HMAC-SHA256("$timestamp${partnerId}sid_request",
  apiKey)). Verify BEFORE trusting. Callback URLs are configured per
  environment in the portal (Developer > Security Settings > Callback URLs);
  a per-job `callback_url` overrides the dashboard default and must be on the
  allowed domains list. Allowlist IPs: prod 13.51.0.119/34.240.137.52/
  51.20.27.3/52.213.46.74; sandbox 13.48.228.158/16.170.104.93/54.246.37.255/
  99.81.237.141.
- **v12 hosted web flow**: load `<script src="https://cdn.usesmileid.com/inline/v12/js/script.min.js">`
  (registers window.SmileIdentity), mint a v3 token on the backend (never ship
  the API key to the browser), then call
  `window.SmileIdentity({ token, product: 'biometric_kyc', callback_url,
  environment: 'sandbox'|'production', partner_details: {partner_id, name,
  logo_url, policy_url, theme_color}, … })`. `partner_details` all 5 fields are
  required. `onResult` is the recommended callback (onSuccess/onClose/onError
  deprecated). Result also arrives at callback_url.
- **Result codes**: 1012 = approved/valid; 1022 = no match/rejected; 1023 =
  not found/rejected. job_status endpoint exists for confirming results
  server-side (POST /v1/job_status, signed yourself).
- **Environments**: sandbox host https://testapi.smileidentity.com, prod
  https://api.smileidentity.com. Keys are environment-specific. Test: set
  `sandbox_result` (0/1/2) to force outcomes. KE supported: NATIONAL_ID,
  PASSPORT, ALIEN_CARD (consent screen required for KE authority checks).

## ChoiceBank handoff (BRS-0003 / ADR-0023, verified 2026-09-03)

ChoiceBank current-account onboarding = `POST /onboarding/submitOnboardingRequest`
(text KYC) → `POST /onboarding/uploadMedia` (KYCF00001..06) within 30 min → review.
Since **2026-01-19** the selfie KYCF00006 must be Smile-ID-verified through the
partner front-end (`skipApiSubmission: true` on the old SDKs); raw selfies leave
`getUserKyc.profileCheck = 0` and the application parks at Choice status 9
(manual review) — not API-cancellable, effectively dead. `profileCheck`:
0 not checked, 1 submitted, 2 validated, 3 declined, 4 processing.

**Verified relay mechanics (2026-09-05)**: Smile webhooks do NOT carry image
bytes — they carry signed `image_links` (an object: `selfie_image`,
`id_card_image`, `id_card_back_image`) that **expire ~15 minutes**. The
YucanPay webhook handler fetches them and relays to Choice `uploadMedia`
(KYCF00006 selfie always; KYCF00001/02 for idType 101, KYCF00004/05 for 102,
KYCF00003 for 103) so Choice's own Smile-side profileCheck can advance.
**Still empirically open**: whether Choice needs uploadMedia at all once a Smile
job exists, or only the Smile-captured provenance — verify live (see ADR-0028).

**Verification Links (portal "Smile Links") are manual-only**: there is NO REST
API to create them (`POST /v3/verification_links` does not exist in the docs) —
only portal Single-User/Multi-User link creation. They are a test/ops vehicle,
NOT an automatable product flow. The automated flows are (a) v12 hosted web
widget (token → `window.SmileIdentity`, product biometric_kyc) and (b)
server-side v3 job creation from images you already captured (products like
`enhanced_kyc` / document verification take doc+selfie images over multipart;
`biometric_kyc` additionally needs 6-8 liveness frames — capture UI required).

## Backend env vars (YucanPay pattern)

`SMILE_PARTNER_ID`, `SMILE_API_KEY`, `SMILE_ENV` (test|live), `SMILE_CALLBACK_URL`.
Enabled only when partner id + key are both set. Callback route is public and
signature-verified. See backend/api/lib/services/smile_service.dart (Hermes
mirror) — mintSessionToken / verifyWebhook / handleWebhook / recordSessionStarted.

## Key doc URLs

- Index: https://docs.usesmileid.com/llms.txt
- Products: /products/onboarding-with-biometrics/{biometric-kyc,document-verification,enhanced-document-verification}.md
- Web SDK: /developer-resources/sdks/web/web-sdk/{setup,getting-result-of-a-verification}.md
- Auth: /api-reference/authentication.md, /api-reference/set-up/access.md
- Webhooks: /developer-resources/essentials/verification-webhooks/{receive-webhooks/configure-your-webhook-server,webhook-types/*}.md
- ID coverage KE: /id-coverage/verify-with-id-number/kenya.md
- Migrations v11→v12: /guides/migration-guides/*-web-integration-migration-guide-v11-to-v12.md

## Gotchas

- Docs mix legacy (body signature `timestamp+partner_id+"sid_request"`,
  PascalCase ResultCode/ResultText) and current (v3 headers/JWT, snake_case)
  shapes — be tolerant parsing webhooks; gate on 1012/1022/1023 first.
- Never mint tokens from the browser; API key must stay server-side.
- Mint a fresh token per verification session (15-min expiry).
- Webhook bodies are unordered/at-least-once — persist idempotently by job_id
  and always ack 2xx even on internal errors (avoid retry storms).
