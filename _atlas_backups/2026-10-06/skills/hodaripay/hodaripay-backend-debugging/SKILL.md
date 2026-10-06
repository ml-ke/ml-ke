---
name: hodaripay-backend-debugging
description: >-
  Debugging playbook for the HodariPay Dart Frog backend + ChoiceBank BaaS
  stack. Use when fixing or investigating backend/API issues in this repo:
  "internal server error", TLS/CA certificate failures, camelCase vs snake_case
  wire-contract mismatches, SQL placeholder ($N) errors, ChoiceBank signed
  requests, STK push / depositFromMpesa, callback (webhook) processing, missing
  columns, or "it works in my head but the UI shows empty/error". Encodes the
  hard-won lessons from the onboarding + payments + dashboard debugging sprint
  so the next agent doesn't rediscover them the slow way.
metadata:
  last_modified: Fri, 14 Aug 2026 00:00:00 GMT
---

# HodariPay Backend Debugging Playbook

The stack: Flutter (web) → Dart Frog API (`backend/api`) → Neon Postgres →
ChoiceBank BaaS sandbox. This playbook is ordered by likelihood: the highest
ROI checks first, then the deep dives.

## 1. Read the API's own errors before trusting the UI

The Flutter app swallows most failures:

- `DashboardPage._safe()` / page-level `catch (_)` degrade to empty lists and
  show "No wallet yet", "No transactions", or a setup banner — even when the
  API returned data that simply failed to parse.
- The ChoiceBank callback route (`routes/choicebaas/callback.dart`) replies
  `200 ok` and **swallows every processing error**. `webhook_events.processed`
  stays `false` when a handler threw.

**Always confirm against the live API and DB before concluding "the data isn't
there":**

```bash
# Login (staging password auth is on for the test merchant):
TOKEN=$(curl -s -X POST https://api-staging-testing.up.railway.app/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"phone":"797715551","password":"5207418Password"}' \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['accessToken'])")
curl -s https://api-staging-testing.up.railway.app/merchant/wallet -H "Authorization: Bearer $TOKEN"
curl -s https://api-staging-testing.up.railway.app/merchant/payment-requests -H "Authorization: Bearer $TOKEN"
curl -s "https://api-staging-testing.up.railway.app/admin/webhooks?limit=20" -H "Authorization: Bearer $TOKEN"
```

Query the DB directly with the `postgres` package (never `psql`; it's not
installed). See "Database access" below. Check `webhook_events.processed` —
`false` means a handler threw after ack.

## 2. The wire contract is camelCase — snake_case in responses breaks parsing

`jsonResponse`/`dataResponse` in `backend/api/lib/http.dart` recursively
converts **snake_case DB row keys → camelCase** (`merchant_profile_id` →
`merchantProfileId`). Every shared model in `packages/shared/lib/src/models`
parses camelCase with `as String` casts.

- If you bypass `jsonResponse` (raw `Response.json`, or a route that returns a
  row map directly), the app's `fromJson` throws and the UI shows empty/error.
- **Never cast `json['x'] as String` without a fallback** in a model for a
  field the API might omit.
- `_toCamelCase` is idempotent on already-camelCase keys (ChoiceBank payloads
  are untouched), so the conversion is safe to apply globally.

## 3. SQL `$N` placeholders are POSITIONAL — reuse means more params

`Database.query` converts **every** `$N` occurrence to a distinct `@pN`
(`$1` twice → `@p1` AND `@p2`). So:

```sql
-- WRONG: needs @p1 AND @p2 but only one value passed → "missing parameter p2"
WHERE account_id = $1 OR choice_user_id = $1

-- RIGHT: pass the value once per occurrence
WHERE account_id = $1 OR choice_user_id = $2   -- params: [acctId, acctId]
```

A `Database.query` guard (`convertPlaceholders` + count check in
`database.dart`) throws a `StateError` when the param count is too low — so a
placeholder mismatch is now loud, not a silent 500. Regression tests live in
`backend/api/test/database_test.dart`.

**This exact bug silently dropped every real ChoiceBank callback** and broke
`_matchTransaction` (payment verification) and the search filters (merchant
transactions, support tickets, admin transactions/merchants) for the whole
session until found. If you edit SQL with `$N`, re-check the param list length.

## 4. TLS/CA certificates — the runtime image needs a trust store

`backend/api/Dockerfile` stage 2 is `debian:bookworm-slim`, which ships **no
`ca-certificates`**. Dart's `HttpClient` on Linux validates against the system
trust store, so outbound HTTPS to ChoiceBank/Twilio failed with
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.

- The fix is `apt-get install ca-certificates` in the runtime stage — already
  applied; keep it if you change the Dockerfile.
- **Local Dart (Flutter-bundled) embeds root certs; the `dart:stable` AOT build
  used by Railway does not.** So "works on my machine" (HTTP 200) does NOT
  prove TLS works in the deployed image. Reproduce by compiling a tiny probe
  with `dart:stable` and running it inside the actual container:
  `docker run --rm -v /tmp/probe:/probe:ro <image> /probe https://...`.

## 5. ChoiceBank signed requests — params must match exactly

- Signing is in `backend/api/lib/signer.dart`; **do not change the algorithm**.
- Wrong param names → `12001/12003/12004` style codes, not clear messages.
  Probe the sandbox read-only with empty/invalid params to learn the real
  field names (the API tells you what it wants):
  `POST /trans/depositFromMpesa` with nothing → "Please input payee account";
  with `accountId` only → "Please input mpesa mobile No." → reveals `accountId`,
  `amount`, `mobile`.
- **STK push** = `depositFromMpesa {accountId: <wallet account_id>, amount,
  mobile: <9-digit, no leading 0/254>}`. Returns `txId` (`UTRANS...`). A `0000`
  code means the prompt was **initiated** — it may still fail at M-Pesa
  (`TXERR0001 Insufficient Funds`). Check the tx outcome with
  `/query/getTransResult {txId}`.
- **Payments recap**: sandbox Paybill `4101847` backs all merchant accounts;
  a settled deposit arrives as callback `0002` and `_handleTransaction`
  records it + `_markPaymentRequestPaid` links it to the `payment_requests`
  row via `stk_tx_id`.
- **Payouts (B2C) are OTP-gated** — a "wallet→M-Pesa not working" bug was just
  this: the tx was created at status 1 but never confirmed. For
  `applyForTransfer` (M-Pesa/PesaLink) you MUST
  `/common/sendOtp {businessId: txId}` then
  `/common/confirmOperation {businessId: txId, otpCode}` (advances to `2`,
  then settles). Works for personal (`TTID0001`) AND SME (`TTID0002`) accounts.
  Minimum KES 10. `quickTransfer` needs whitelisting (`11000 No permission`).
  Sandbox recipients can fail `TXERR0002 Invalid Account` (not a code bug) —
  use a Choice "payee test account".

## 6. Database access from this machine

`psql` is not installed and the Neon CLI hits IPv6 issues. Use a throwaway
Dart script with the `postgres` package (see `backend/api/pubspec.lock` for the
version):

```dart
// copy the Endpoint/Pool plumbing from backend/api/lib/database.dart
final pool = Pool.withEndpoints([endpoint], settings: PoolSettings(
  maxConnectionCount: 2, connectTimeout: const Duration(seconds: 20),
  sslMode: SslMode.require));
```

Get `DATABASE_URL` via `railway variables --json` (the MCP returns names only).
Remember: `citext`/unregistered types arrive as `UndecodedBytes` — utf8-decode
them (the app's `Database._rows` already does this).

## 7. Migrations are schema, not data

Test-data seeding for a specific user belongs in a **direct DB update on the
testing branch**, NOT a migration file — migrations run on production during
promotion too. Schema changes (new columns/tables) go in
`infra/neon/migrations/NNNN_*.sql` (idempotent, `IF NOT EXISTS`).

`payment_requests` originally had **no `updated_at` column** (0001 omission) —
the callback paid-marking UPDATE threw until migration 0005 added it. When a
new UPDATE/SELECT references a column, confirm it exists first.

## 8. Async hygiene

- `Future<String>` values must be `await`ed before being put into a params
  map / JSON body — otherwise `JsonUnsupportedObjectError: Converting object to
  an encodable object failed: Instance of 'Future<String>'` and a generic 500.
- `http`/`http`-client calls need `.timeout(...)`; transport errors
  (`SocketException`, `HandshakeException`, `ClientException`, `TimeoutException`)
  should be mapped to domain exceptions (`ChoiceBankException`, `TwilioException`)
  so `mapError` returns a clean 502 `E_CHOICEBANK`, not an opaque 500.

## 9. Verify in the container, not just on your machine

After backend changes: `dart analyze && dart test` (all 3 packages), then
`dart pub global run dart_frog_cli:dart_frog build`. To prove TLS + signing
end-to-end, compile a small probe with `dart:stable` and run it inside the
built image against the real sandbox URL with the real sender/key (from
Railway vars). Clean up any images/containers you create
(`docker rmi` / `docker system prune -f`).

## 10. Persist ephemeral flags the UI depends on

Anything the create-response returns that the UI needs after a subsequent
poll must be **stored in the DB** (or re-derived on read). Real case: `POST
/merchant/payouts` returned `otpRequired` + `operationRef` only in the create
response. A 6s status poll (`GET /merchant/transactions/:id`) then dropped
both, so the "Enter code"/"Resend code" buttons vanished while the transfer
was still awaiting OTP — the card froze on "checking status" forever and a
DB `-1` timeout never surfaced. Fix: persist `otp_required` on `transactions`
(migration 0010) and re-derive `operation_ref` (= the Choice tx_id) in
`getTransaction`/`refreshTransactionStatus` when `otp_required`.

Rule of thumb: if a UI state machine depends on a value that only exists in
one response, persist it. Also, a page-level state machine should **always
poll** a non-terminal resource (not just when "no OTP needed") so a timeout /
failure in the DB reflects in the UI.

## 11. Rail-aware status mapping for batched ChoiceBank rows

Bulk payments reuse per-row ChoiceBank rails, and the same Choice txStatus
means the same thing per rail: **every rail (paybill/till/mpesa/pesalink/
choice) is OTP-gated** (verified on sandbox 2026-08) — a row at `txStatus 1`
is `awaiting_otp`, status `2` is `processing`, `8/4/-1` terminal. The earlier
assumption that Paybill/Till settle without a code was WRONG and left those
sends hanging at status 1 until timeout. Keep the batch's per-row status
mapping rail-aware (`_refreshBulkItemStatus` uses the shared `_otpGatedRails`
set in `MerchantService`).

## 12. Local smoke test against the real stack (safe pattern)

To prove a new endpoint end-to-end without deploying: temporarily point the
local `.env` at the staging Neon pooler URL + the real Choice creds, run
`dart_frog build` then `dart build/bin/server.dart` on a spare port, and curl
the new routes with a real login token. Watch for the two foot-guns:
- The local `.env` may hold a **stub `DATABASE_URL`** (`DATABASE_URL=` empty)
  that silently breaks DB-backed endpoints — append the real pooler URL, and
  restore the file afterwards (back it up first).
- `dotenv load()` reads `['.env']`; add flags like `PASSWORD_AUTH_ENABLED=true`
  to enable a login path. Kill the server with
  `ps aux | grep "[b]uild/bin/server"` (never `pgrep -f server.dart`, which
  matches the grep itself). Restore `.env` from the backup when done.

## 13. Physical-device testing: deploy the backend FIRST

Phones run the **deployed staging API**, not your laptop. A route that works in
local `dart_frog` returns **404 on the phone** until it's deployed. Real case:
`/auth/pin/verify` (new) was 404 on staging → the app showed "PIN 0000 failing"
/ a hang. `railway up --ci` from `backend/api` (linked to the testing env)
fixes it. So: gates green → deploy backend to staging → reinstall the APK →
test. Also, the shared `HodariPayApi._send` must have a **timeout + 401-refresh**
retry, or a stale session makes the UI hang with no error (see
`docs/flutter-app-gotchas.md` §7-9).

## 14. Stateful wizards: resume from server status, never from cached client flags

The SME business-onboarding "OTP received but no code-entry screen" bug
(2026-09-05) was this class:

- The wizard's resume (`onboarding_page.dart _resume`) gated on a cached
  `AppUser.merchantProfileId`. The profile row is created **server-side on
  initiate**, but `AppUser` only refreshes at login — so after initiate +
  refresh/re-login the flag is stale null, resume early-returns, and step 0
  re-renders with a live OTP SMS nobody can enter. The `E_ONBOARDING_ACTIVE`
  (409) catch then called the same broken resume.
- The SME status payload had no persistent `otpRequired` — the **personal
  wallet flow already had one** on `/personal/onboarding` (submit → poll →
  render OTP card). The wallet flow works because OTP-required is a persistent
  polled status; the SME flow broke because it was an ephemeral local
  `_step = 1` transition that a refresh could discard.

Fixes that are now regression-locked (see `merchant_onboarding_otp_resume_test.dart`,
`onboarding_status_otp_required_test.dart`, `onboarding_otp_resume_test.dart`):
- Backend: `onboardingStatus` (and initiate/send/confirm responses) expose
  `otpRequired = activeRequest && !otpConfirmed && !blocked`; resend persists
  `status='otp_sent'`; `_resumeStep` maps `'initiated' || 'otp_sent'` →
  otpConfirmed ? 2 : 1.
- App: resume requires only `isMerchant` (never a cached profile id) and drives
  `_step` from server status; `E_ONBOARDING_ACTIVE` fetches status and forces
  `_step = 1` when `otpRequired`; resume errors are `debugPrint`-logged, not
  swallowed.

Rule of thumb: a wizard that spans refresh/re-login must re-derive its step
from a server status endpoint (which must carry an explicit "waiting on user
action" flag per product), and every "already active" error must route the user
to the exact action the server is waiting on.

## Checklist when a UI page shows empty/error for a working backend

1. Does the endpoint return camelCase (not raw snake_case)? — check via curl.
2. Does the shared model field match the API key exactly?
3. Does `webhook_events.processed` = true for the expected callbacks?
4. Does the SQL reference `$N` more times than params provided?
5. Is the deployed image the latest commit (Railway SKIPs unchanged builds)?
6. Does the callback/route swallow an exception you haven't seen? Add a
   `stderr.writeln` or check Railway logs before assuming success.
7. **Is the page reading an inherited widget in `initState`?** A `_load()` that
   calls `AppStateScope.of(context)` (or `Theme.of`) before the first frame
   **throws in DEBUG builds** and — if the call sits inside a try/catch — is
   swallowed → the page renders its empty state while the backend works. This
   is SYSTEMIC (16 pages hit it): "Pay from" dropdown not clickable (empty),
   wallet/admin lists empty. Release builds don't throw, so it "works on
   web/release but is empty on the debug phone". Fix: defer
   `WidgetsBinding.instance.addPostFrameCallback((_) => _load())`; if the load
   writes a `late final Future` field, make it nullable and `setState` in the
   callback.
8. **Is the read path running many sequential DB queries?** On the cross-region
   Neon pooler each query costs ~1s (TLS + pooler). N sequential queries ≈ N
   seconds. Batch into ONE aggregate query with joins/subqueries — `getWallets`
   went 6 queries/~4.5s → 1 query/~2.7s. For **slow-changing content** (blog
   posts, FAQs — admin-managed, read repeatedly), add a small in-memory TTL
   cache (`backend/api/lib/content_cache.dart`, 30s) with **write-invalidation**
   on create/update/delete, so repeated reads skip the DB entirely. In-memory is
   fine on a single Railway instance; move to Redis for multi-instance.
9. Client has no timeout / no 401-refresh → a stale session hangs with no
   error (see `docs/flutter-app-gotchas.md` §7).
10. **CI web deploy fails with "Missing entry-point"** → wrangler v3 can't
    deploy assets-only Workers (no `main`). Pin `wranglerVersion: '4'` in
    `cloudflare/wrangler-action@v3` (staging + production workflows).
7. Is a polled resource dropping ephemeral flags (`otpRequired`/`operationRef`)
   that the UI's state machine relies on? Persist or re-derive them.
