---
name: hodaripay-testing
description: >-
  Repo-specific testing playbook for YucanPay (BongweKE/hodaripay): testing
  conventions for every layer — the pure-Dart shared package, the Dart Frog
  backend, the Flutter app, and real-device/Android verification. Load before
  writing or reviewing ANY test in this repo, or when a local/CI test gate
  fails. Encodes the class-of-bug lessons (wire-contract drift, positional SQL
  placeholders, swallowed callback errors, go_router redirect loops, PIN gate
  conflicts, client-without-timeout hangs) so gates fail on the right things.
---

# HodariPay / YucanPay — Testing Playbook

Deep, repo-specific testing conventions for every layer we ship: the pure-Dart
**shared package**, the **Dart Frog backend**, the **Flutter app**, and
**real-device/Android** verification. Goal: catch the *class* of bug, not just
the instance — our CI gates + local gates must fail on the same mistakes we've
already made (wire-contract drift, positional SQL placeholders, swallowed
callback errors, go_router redirect loops, PIN gate conflicts, hangs from a
client with no timeout).

Load this before writing or reviewing any test in this repo.

---

## 1. The gates (run all three before committing)

```bash
cd packages/shared && dart analyze && dart test
cd backend/api    && dart analyze && dart test
cd apps/hodaripay && flutter analyze && flutter test
cd backend/api && ~/.pub-cache/bin/dart_frog build   # catches rogue routes
```

`pr.yml` runs exactly these. A green local run ⇒ a green PR.
`dart format .` on shared+backend first (run `dart pub get` before, or the
formatter uses a stale language version).

## 2. Shared package tests (`packages/shared/test/`)

Everything here is pure Dart — fast, no Flutter.

**Models**: every `fromJson`/`toJson` must round-trip. Test BOTH directions and
the *exact* backend wire shape (camelCase). Add a fixture that mirrors a real
`curl` response so wire drift breaks the test, not the UI.

**Enums**: code↔label mapping (`TxStatus.fromCode`, `BusinessType.fromCode`,
`PaymentChannel.fromWire`).

**ApiClient** (`api_client_test.dart`): inject a fake `http.BaseClient` to test
the transport contract without a server:
- **401 + expired token** → `onAccessTokenExpired` fires once and the request
  retries (`retried: true`). This is what prevents "stale session = silent
  hang" on the PIN screen.
- **ClientException** → `ApiException` with `E_NETWORK`.
- **Timeout** (never-responding client) → `ApiException` with `E_TIMEOUT`.
- **`data` envelope unwrap**: routes differ — some return bare objects
  (`/auth/me`), others wrap in `{"data": {...}}` (`dataResponse`, e.g.
  `/auth/pin/verify`). Assert the client method returns the unwrapped shape so
  pages reading `result['mustChange']` don't silently get null.

## 3. Backend tests (`backend/api/test/`)

Unit-test the pure logic; the DB + ChoiceBank are exercised by the live smoke
test (§6). Existing coverage: `signer` (must match the Node reference exactly —
NEVER change the algorithm), `http` camelCase conversion, `database`
placeholder guard, `kenya_banks` normalization/fallback, `sme_documents` KYCF
mapping, `config`.

**Always cover**: any new static/pure helper, the `_ph` placeholder count
(see database_test), error→response mapping in `http.dart`, and the OTP/pin
lockout arithmetic if it's pure.

## 4. Flutter widget tests (`apps/hodaripay/test/`)

Environment realities:
- **Network is blocked** (flutter_test returns HTTP 400) → `_load()` calls
  throw and pages render empty/error states. Test THOSE states, not happy paths.
- **`pumpAndSettle` HANGS** on repeating animations (dashboard shimmer). Use
  fixed `pump(Duration)` steps — usually several 200ms frames.
- Mock `FlutterSecureStorage.setMockInitialValues({})` and `FilePicker.platform`
  (it's settable). Never let a plugin hit the platform.
- **`initState` must not read inherited widgets** (`AppStateScope.of`,
  `Theme.of`) — it throws in debug. Defer with
  `WidgetsBinding.instance.addPostFrameCallback((_) => _load())`.
- go_router shell gives pages **loose/unbounded** constraints — page-level
  Material buttons crash. Wrap tests like the support-page layout test does.

**Deep tests that caught real bugs (keep this pattern):**
- `onboarding_navigation_test` — `unlocking the PIN navigates off the lock
  screen to the shell`: authenticate WITHOUT unlocking → assert the router pins
  us to `/pin` (this caught the infinite redirect loop between the PIN gate and
  the merchant-onboarding redirect → go_router "Page Not Found"). Then
  `state.unlockPin()` → pump ≥6 fixed frames → assert we're on the merchant
  shell. **Navigate-by-state tests must pump enough frames** — the
  redirect-on-refresh navigation isn't instant.
- `pin_lock_test` — renders the lock screen and surfaces a blocked-network
  error gracefully.
- `receive_page_collection_test` — the Paybill collection card renders with the
  fallback shortcode when channels can't load.

## 5. Business/UI/UX specs the tests must encode

These are product invariants — a regression here is a money/security bug:

- **PIN**: default `0000` for everyone; merchants forced to change
  (`mustChange`), **staff are NOT** (so the team is never blocked). 5 wrong
  attempts → 5-min lockout. Payments (`POST /merchant/payouts`) require `pin`.
  Forgot PIN → support (no self-serve OTP while Twilio's absent); admin resets
  via `/admin/users/:id/pin-reset` (capability `reset_user_pin`, audited).
- **Till/Paybill are OTP-gated** (all rails). A row at txStatus 1 is
  `awaiting_otp`; there is no "settles without a code" rail.
- **Personal-wallet limit** KES 20,000/day outbound, 300,000 holding; enforced
  in `createPayout`, surfaced as `limits`/`dailyUsed` on `/merchant/wallet`.
- **Receive** = Paybill shortcode + per-wallet account number + amount-scoped
  M-Pesa QR (`generateQRCode`); QR needs a whole KES amount ≤ 250,000.
- **Balance eye** masks balances app-wide (persisted per device).
- **Platform journeys**: Android bottom-nav + camera KYC; web rail + file-picker;
  desktop = web-at-desktop-width. No platform package folders yet.

When you add a feature, add a test that locks in its business rule above.

## 6. Live smoke test (backend end-to-end, safe)

Prove a new endpoint against the real stack without deploying:
1. Back up `backend/api/.env`; point `DATABASE_URL` at the staging Neon pooler
   (`hodaripay` DB on the testing branch) + keep the real Choice creds.
2. `dart_frog build` then `dart build/bin/server.dart` (background).
3. Mint a JWT with `tool/mint_token.dart` (remove after) using the **staging**
   secrets, or log in with a known staging account
   (merchant `797715551` / `5207418Password`, admin `admin@hodaripay.co.ke`).
4. `curl` the routes; assert the camelCase wire shape.
5. Restore `.env`.

## 7. Real-device / Android verification

Phones hit the **deployed staging API**, not your laptop — a new route 404s on
the phone until you `railway up --ci` from `backend/api` (linked to the testing
env). Flow: gates green → deploy backend → rebuild APK
(`--dart-define=API_URL=https://api-staging-testing.up.railway.app`) → install.

```bash
ADB=/home/pro-g/Android/Sdk/platform-tools/adb
$ADB devices -l                       # Wi-Fi (adb pair/connect) + USB
$ADB -s <serial> install -r app-debug.apk
$ADB -s <serial> shell pm clear co.ke.yucanpay.app   # fresh session (automatic)
$ADB -s <serial> shell am start -n co.ke.yucanpay.app/.MainActivity
$ADB -s <serial> shell uiautomator dump /sdcard/ui.xml && cat /sdcard/ui.xml | grep -oE 'text="[^"]*"'
$ADB -s <serial> logcat -d | grep -iE "E/flutter|Unhandled|ApiException|Exception"
```

Gotchas:
- **Stale `ui.xml`**: always delete it before dumping, or you read yesterday's
  screen (this misled a whole PIN debugging session — Bitwarden's vault-lock
  looked like our PIN screen).
- **Filter logcat** aggressively (`SensorService|wlan|mtk|BufferQueue|SurfaceFlinger|…`)
  or the SoC noise buries the signal.
- `mali_gralloc`/`Usage not permitted` GPU errors on Tecno/MediaTek are
  Impeller-rendering noise, not app bugs — the app still renders.
- `pm clear` wipes the session so you can sign in as a different account
  without hunting for a "sign out" (the lock screen also has "Not you? Sign out").
- Read backend request logs via Railway (`railway_get-logs` or
  `railway logs`) — they show the exact `/auth/*` sequence (e.g. `pin/verify
  400 → 200`) so you can tell backend-correct from app-stuck.

## 8. When to be suspicious (past bugs → red flags)

- Page shows empty/error but `curl` works → wire-contract drift (camelCase) or
  the shared model's `fromJson` field name mismatch.
- A screen hangs with no error → the HTTP client has no timeout / no 401-refresh
  (`HodariPayApi._send` must have both).
- Router "Page Not Found" after a state change → conflicting redirects
  (a locked session must be confined to `/pin`; nothing else may redirect it).
- A ChoiceBank transfer parks at status 1 forever → it's OTP-gated; confirm the
  code flow (don't assume paybill/till settle without OTP).
- A callback shows `processed=false` → the handler threw; the route swallows it.
- An endpoint is slow (10s+) → it's synchronously calling the upstream
  (ChoiceBank) in the request path instead of serving the local read model.

## 9. Performance / system-design — the read-model rule

**Symptom we fixed**: `GET /merchant/transactions` took 22–28s because the
handler synchronously re-pulled ChoiceBank (`getTransList`, paginated) for
every wallet before returning. The Flutter client's 20s timeout then killed it
→ "Could not load your account" on mobile (web had no timeout, so it worked
slowly — the worst kind of bug: environment-dependent).

**Why (system-design lens)**:
- The request path blocked on the *authoritative-but-slow* source instead of
  the *fast local read model*. ChoiceBank is cross-continent (~150ms-2s/call,
  multiplied by pagination × wallets); Postgres is ~1-5ms.
- This is **CQRS/read-model + eventual consistency**: writes go to ChoiceBank;
  reads come from the DB, which the **20s poller + callbacks** hydrate.
  The read path should NEVER trigger a synchronous upstream pull.
- **Latency budget**: a client timeout is a contract — the API must return
  inside it or the work must be async (background job + poll). Timeouts are not
  a tuning knob to relax; the sync path is the bug.

**The fix pattern (use for ANY read endpoint)**:
- Serve stored DB rows immediately; fire the reconcile in the background
  (`unawaited(_reconcileWallet(...))`, throttled by `last_synced_at`).
- The poller keeps balances/transactions fresh within ~20s — acceptable
  staleness for a dashboard, documented as a decision.
- Test: time the endpoint (assert <~1s warm), and assert the sync work is
  backgrounded (the response returns before ChoiceBank round-trips finish).

**Multi-wallet / multi-account business rule**: wallets and transactions are
USER-scoped (personal wallet/current accounts have no `merchant_profile_id`).
Admin views must query `merchant_profile_id = X OR user_id = Y`, and a
merchant's effective status is "opened" when ANY active wallet exists (not just
a business profile). The admin list surfaces `wallet_count` per merchant.

## 10. Debug vs release behavior — the class of "works on web, empty on phone"

Debug and release builds behave differently: **reading an inherited widget
(`AppStateScope.of`, `Theme.of`) before the first frame throws in DEBUG but is
permitted in release.** A `_load()` called from `initState` that reads the
scope throws in debug → caught → the page renders empty. Phones ran debug
builds → "Pay from" dropdown not clickable, wallet/admin lists empty — while
web/release worked.

**Fix + regression-test pattern**:
- Fix: defer every first load with
  `WidgetsBinding.instance.addPostFrameCallback((_) => _load())`. If the load
  populates a `late final Future` field, change it to a nullable field and
  `setState` in the callback.
- Test: **pump each page** under `AppStateScope` + `MaterialApp` with the
  network blocked, then assert the page renders its chrome (title, dropdown
  label, filter chips) — not just that it doesn't crash. A page that only
  "doesn't crash" would have passed while showing empty. Lock the controls
  that were broken (e.g. `payouts_page_render_test` asserts the `Pay from`
  selector exists; `admin_tickets_page_test` asserts the filter chips).
- Grep for the class: any `initState` that leads to an `AppStateScope.of`
  within ~25 lines is suspicious — the load must be deferred.

## 11. Caching + dashboard-link tests

- **Content cache** (`ContentCache`): assert GETs are cached (no DB re-hit on a
  second read within TTL) and that **writes invalidate** (create/update/delete
  must clear the `posts:`/`faqs:` prefixes so edits appear immediately).
- **Dashboard metrics are links**: a widget test for `AdminDashboardPage`
  (blocked network → error state) asserting the metric cards render; a deeper
  test taps a card and asserts navigation to the target route. Never leave a
  metric card inert.
- **Release/versioning**: when a version bumps, update `docs/changes.md` with a
  dated entry + the GitHub release link, and re-run the full gates.



