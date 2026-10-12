---
name: hodaripay-choicebank
description: >-
  Reference for working with the ChoiceBank BaaS sandbox as integrated in the
  HodariPay repo. Use when building or debugging anything that calls ChoiceBank:
  signed-envelope requests, onboarding (SME + personal wallet/current), STK
  push deposits, transfers/payouts (M-PESA / PesaLink / Choice-to-Choice),
  bulk payments (staged per-row batch), inter-wallet transfers, OTP flow,
  bank-code lookups, callbacks/webhooks, and account seeding. Covers the
  sandbox quirks, the endpoint catalog, the correct param names, and the
  verification workflow that proved each flow end-to-end. For a
  framework-agnostic Choice Bank API reference, load `choicebank-baas`.
metadata:
  last_modified: Sat, 15 Aug 2026 00:00:00 GMT
---

# ChoiceBank BaaS — HodariPay integration reference

Full architecture write-up: `docs/choicebank.md`. This skill is the condensed,
verified-in-this-session playbook. **Every detail below was proven against the
live sandbox**, not copied from docs. See the standalone `choicebank-baas`
skill for the framework-agnostic API catalog + the official GitBook URL.

## Environments & credentials

| | Sandbox (staging) | Live |
|---|---|---|
| Base URL | `https://baas-pilot.choicebankapi.com` | `https://api.choicebankapi.com` |
| Sender ID | `yucan` | live sender |
| Sender key | 64-char private key (Railway `CHOICE_SENDER_KEY`) | live key |
| Callback | `https://api-staging-testing.up.railway.app/choicebaas/callback` | prod |

Creds live on Railway (never in git). Read them with
`railway variables --json` → `CHOICE_SENDER_ID/KEY/BASE_URL`.

## Signed envelope (do not change the algorithm)

Every call is `POST` with:
`{requestId, sender, locale:'en_KE', timestamp(ms), salt, params, signature}`.
Signature = flatten body to `key=value` (dots for objects, `[i]` for arrays),
append `senderKey=<key>`, sort ASCII, join `&`, SHA-256.
`backend/api/lib/signer.dart` ports this 1:1 from the validated Node version.
Success code is `00000`. Non-zero codes: `12001` = missing/invalid param,
`12003` = invalid sender, `12004` = invalid signature, `12010` = invalid mobile,
`13000` = account not found, `13224` = missing [userId|mobile], `15001` = time
range error, `15002` = pageSize > 50, `11200-11203` = SMS OTP throttled/
frequent/locked/wrong.

## Probing to learn the API (read-only, safe)

The sandbox tells you the real param names if you probe with empty/partial
params — each returns `12001 "please input X"` revealing the next required
field. Use a throwaway signed POST (see
`/tmp` probe pattern: build envelope, sign, POST, print raw JSON).

Example — learning `depositFromMpesa`:
- `{}` → "Please input payee account"
- `{accountId}` → "Please input mpesa mobile No."
- `{accountId, amount, mobile}` → `00000` + `data.txId` (STK prompt initiated)

## STK push (payments) — the verified flow

- **Trigger**: `POST /trans/depositFromMpesa`
  `{accountId: <wallet.account_id>, amount: <whole KES>, mobile: <9-digit, no
  0/254>}`. The 9-digit format is required — `0797715551` and `+254...` both
  return `12010 Invalid mobile number`; `797715551` succeeds.
- Returns `data.txId` (`UTRANS...`). Code `00000` = initiated, NOT settled.
- **Outcome**: `POST /query/getTransResult {txId}`. `txStatus 8`=success,
  `4`=failed (e.g. `TXERR0001 Insufficient Funds`), `2`=processing,
  `-1`=timeout.
- **Sandbox Paybill**: `4101847` backs all merchant accounts (prod: `444174`).
  A customer paying that Paybill with the merchant's account number as the
  account creds the merchant.
- **Backend wiring** (`MerchantService.createPaymentRequest`): when
  `channel == 'stk_push'` and a `customerPhone` is present, call
  `depositFromMpesa`, store the returned `txId` in `payment_requests.stk_tx_id`
  (migration 0005). The 0002 callback then links + marks it paid. The route
  accepts `walletId` so the push targets a chosen wallet.
- **Verification tool**: simulate the settlement callback by POSTing a signed
  `0002` body to the callback URL (`{notificationType:'0002', requestId,
  sender, timestamp, salt, params:{txId, txStatus:'8', amount, currency,
  accountId}}`). Expect HTTP 200 and then check the DB:
  `payment_requests.status='paid'`, `webhook_events.processed=true`.

## Transfers / payouts (B2C) — verified on sandbox

Wallet/current/SME → M-Pesa (B2C) is done via **General Transfer** — works for
ALL account types:

- **Initiate**: `POST /trans/v2/applyForTransfer`
  `{payerAccountId, payeeBankCode:"M-PESA", payeeAccountId: <9-digit phone>,
  currency:"KES", amount}`. `payeeAccountName` optional for M-Pesa.
- **OTP is mandatory**: tx created at `txStatus 1`; only
  `/common/sendOtp {businessId: txId, otpType:"sms"}` +
  `/common/confirmOperation {businessId: txId, otpCode}` advances to `2`.
  Without OTP the tx hangs and eventually times out (`-1`).
- **txType by payer account**: personal (C001/C002) → `TTID0001` Withdraw to
  M-PESA; SME (B001) → `TTID0002` Transfer Out. Both settle — same flow.
- **Minimum amount KES 10** (`14015`); keep tests ≥ 10.
- **`/trans/quickTransfer`** (`txType:0` B2C / `1` B2B) returns `11000 No
  permission` unless whitelisted — use `applyForTransfer`.
- **Sandbox recipient quirks**: some numbers fail `TXERR0002 Invalid Account`
  at settlement even when they work as STK-push payers. Choice provides "payee
  test accounts"; use a known-good one. `14042/14043` = whitelist errors.
- **Repo wiring**: `POST /merchant/payouts` initiates + auto-sends OTP for
  **every** rail — `mpesa`/`pesalink`/`choice` (applyForTransfer) and
  `paybill`/`till` (applyForMpesaBusinessTransfer) are all OTP-gated (verified
  on sandbox; Paybill/Till do NOT settle without the code). Returns
  `otpRequired` + `operationRef`; `POST /merchant/transfers/otp`
  confirms/resends. The 20s poller advances via `getTransResult` — no callback
  needed. `otp_required` is PERSISTED on `transactions` (migration 0010) so a
  polled row keeps offering enter/resend.

## Choice→Choice (inter-wallet) — verified on sandbox

- **Same `applyForTransfer` rail** as M-Pesa/PesaLink, but with
  `payeeBankCode: '46'` (Choice Microfinance Bank — from
  `/staticData/getBankCodes`). **Literal `'CHOICE'` is rejected**
  (`10000 invalid payee BankCode`).
- `payeeAccountId` = the destination ChoiceBank account id (any of the user's
  own wallets, or any Choice customer's account).
- **OTP-gated exactly like M-Pesa**: tx starts at `txStatus 1`, needs
  `sendOtp` + `confirmOperation`, settles via `getTransResult`/callback.
  `paymentChannel` reports `INTERNAL_TRANSFER`; `oppoBankCode` `CIC0231`.
- **Repo wiring**: `POST /merchant/payouts` with `kind: 'choice'`,
  `choiceAccount` (dest account id) + optional `beneficiaryName`. The Wallet
  tab's "Transfer between wallets" uses this rail.
- **IMT endpoint note**: `/trans/v2/applyForImtTransfer` and
  `/trans/applyForInnerTransfer` return **HTTP 500 in the sandbox** — the
  doc'd path is `/trans/applyForImtTransfer` (whitelisted, no OTP). The `46`
  bank-code path on `applyForTransfer` is the verified way to move
  Choice→Choice.

## Bulk payments — staged batch, per-row OTP (verified on sandbox)

- **The dedicated bulk endpoint the repo client referenced
  (`/trans/v2/applyForBulkTransfer`) returns HTTP 500 for every param shape**
  in the sandbox. The current doc'd bulk endpoints are
  `/trans/v2/generalBulkTransfer` (SME) and
  `/trans/batchTransitionalFundsDisbursement` (merchant, whitelisted + no OTP).
  Both still need account-manager whitelisting. Until whitelisted, bulk is
  built on the per-row verified rails as a **staged, resumable batch**.
- **Schema**: `bulk_payments` (batch) + `bulk_payment_items` (rows) — migration
  0009. A batch belongs to one user + wallet; rows carry a `rail`
  (`mpesa|till|paybill|pesalink|choice`).
- **Processing** (`MerchantService.createBulkPayment` → `_advanceBulk`):
  - **Every rail is OTP-gated** (verified 2026-08: Paybill/Till B2B included) —
    `paybill`/`till` → `applyForMpesaBusinessTransfer`; `mpesa`/`pesalink`/
    `choice` → `applyForTransfer`. Each initiated row parks at txStatus 1 and is
    auto-`sendOtp`'d, set to `awaiting_otp`, batch paused.
  - `confirmBulkOtp` confirms the ONE awaiting row, then `_advanceBulk` moves
    to the next OTP row — exactly one code outstanding at a time. Resumable
    across sessions.
- **Rows never double-submit**: `_advanceBulk` refreshes awaiting/processing
  rows from ChoiceBank before acting; all rows are mirrored to
  `transactions` (`type='bulk_<rail>'`, `channel='bulk'`) so history/balance
  stay correct. Status mapping is rail-aware: any rail at txStatus 1 is
  `awaiting_otp` (OTP-gated); status 2 = `processing`; 8/4/-1 terminal.
- **`generateQRCode`** (`POST /trans/generateQRCode {accountId, amount}`) →
  base64 PNG M-Pesa PayBill QR (whole KES, ≤250,000; amount is mandatory —
  `12001` without it). Exposed via `GET /merchant/receive/channels?amount=`.
- **Repo wiring**: `POST /merchant/bulk-payments` (create),
  `GET /merchant/bulk-payments` + `/…/:id` (list/detail),
  `POST /merchant/bulk-payments/:id/otp` (`{otp}` / `{resend:true}`).
  Flutter: `BulkPaymentsPage` (Pay tab → Bulk payments) uploads CSV/XLSX,
  previews, pays in batches. The bulk 0008 callback is recognised but a no-op.

## Callback / webhook processing

- Endpoint: `POST /choicebaas/callback`; always replies `200 ok` to stop
  retries (docs: reply string `ok`; max 5 attempts), stores each event
  idempotently (`webhook_events.request_id` UNIQUE).
- `webhook_events.processed=false` **always means a handler threw** — the route
  swallows exceptions. Debug by reproducing the handler logic directly.
- Types: `0001`/`0005` personal onboarding, `0002` transaction result, `0003`/
  `0022` balance, `0006` SME opening, `0004` merchant bulk, `0008` SME bulk,
  `0019` closure, `0021` status change, `0027` IMT, `0023` inter-bank.
- **Gotcha (fixed this session)**: `_resolveMerchant` used `$1 OR choice_user_id
  = $1` with one value — the positional placeholder conversion made that a
  "missing parameter p2" error that silently dropped EVERY real callback.

## Onboarding endpoints

- SME: `applyForSmeOnboarding` (OTP-gated) → confirm OTP (`businessId =
  onboardingRequestId`) → `submit{Store,Company,Partnership,Organisation}
  OnboardingRequest` → members/shareholders → `uploadMedia` (KYCF codes) →
  `submitOrPullBackRequest`. Query: `getBusinessOnboardingStatus`.
- Personal wallet: `/onboarding/v3/submitEasyOnboardingRequest` (single call,
  photos inline, `backSidePhoto` only for non-passport ID types).
- Personal current: `/onboarding/submitOnboardingRequest` + `uploadMedia`;
  KRA/employment/income/address mandatory.
- Recovery: `getUserKyc`, `getOnboardingRequestId` (by `userId` OR `mobile` —
  passing neither → `13224`).

## Short codes (collection aliases) — verified live 2026-09-03 (BRS-0002, ADR-0021)

- A `shortCode` is a **much shorter account-number alias** ChoiceBank binds to a
  specific `accountId` — per wallet, NOT per user, NOT a Safaricom Paybill.
  Collection stays: customer pays the env Paybill (`444174` prod / `4101847`
  sandbox) and types the short code **or** the full `accountId` into the account
  field.
- `POST /account/applyForShortCode {accountId}` → `{accountId, shortCode}`.
  Synchronous, **no OTP**. Re-applying for an account that already has one →
  business code `13302` "short code exist for this user" — treat as
  **query-and-capture**, not an error.
- `POST /account/queryForShortCode {accountId}` → `{accountId, shortCode}`
  (`shortCode: null` when unassigned). `POST /account/queryAccountByShortCode
  {shortCode}` → reverse lookup. `/query/getAccountDetails` also returns
  `shortCode`.
- **Live fact**: freshly-opened wallets return `shortCode: null` — an alias must
  be requested. Sandbox may not support the `applyForShortCode` **write**; the
  repo surfaces a retryable `E_SHORTCODE_UNAVAILABLE` and collection still works
  via Paybill + the full account id.
- **Repo wiring**: `POST /merchant/receive/shortcode {walletId}`
  (`MerchantService.requestShortcode`: apply → fall back to query on
  `13302`/empty → persist `wallets.short_code`). `receiveChannels`/`getWallets`
  carry the per-wallet `shortCode`; the read path lazy-captures
  (`queryForShortCode`) codes Choice assigned out-of-band.

## Query APIs that work in sandbox

- `/query/getTransList {accountId, startTime, endTime(epoch ms), pageNo,
  pageSize}` → `{result:[], totalRows}`. Docs list `txType`/`txStatus` as
  required but the sandbox tolerates omission. pageSize > 50 → `15002`.
- `/query/getTransResult {txId}` → full tx detail.
- `/query/getMerchantAccountDetails {accountId}` → **`13000 Account does not
  exist` for SME accounts seeded via onboarding in this sandbox** — do not
  rely on it; use the onboarding-status query instead.
- `/query/getAccountBalance {accountId}` → **HTTP 500 for SME accounts in this
  sandbox**. The app degrades to the stored `wallets.balance`.
- `/staticData/getBankCodes` → `data.bankCodeList` (265 banks; `46` = Choice
  Microfinance Bank, `M-PESA` = Mpesa) — the only reliable bank lookup.

## Seeding an existing ChoiceBank account to a user

When a ChoiceBank customer already exists (dashboard) but isn't linked in our
DB:

1. Verify read-only with `getBusinessOnboardingStatus {onboardingRequestId}`
   → confirms `accountId`, `status:7`, `accountType`.
2. Insert `merchant_profiles` with `status='account_opened'`,
   `choice_user_id = <the REAL ChoiceBank userId>`, `account_id`, `mobile`
   (9-digit), `otp_confirmed=true`, `kyc_complete=true`, `choice_status=7`.
   `choice_user_id` MUST be ChoiceBank's id, not a generated `HP...`, so
   callbacks resolve via `_resolveMerchant`.
3. Insert a `wallets` row (`account_type='sme'`, `status='active'`).
4. **Do this as a direct DB update on the testing branch, NOT a migration**
   (migrations run on production too).
5. `status='account_opened'` drives `onboardingComplete` in `/auth/me` → the
   app sends the user straight to the dashboard.

## Verification workflow (proven end-to-end)

1. `curl` the staging API with a real login token; confirm camelCase JSON.
2. Query the DB with a `postgres`-package script for the ground truth.
3. Trigger a real STK push via `POST /merchant/payment-requests
   {channel:'stk_push', customerPhone, amount}` → user gets an M-Pesa prompt.
4. Check `getTransResult` for the tx outcome; simulate the 0002 callback for
   success; confirm `payment_requests.status='paid'`.
5. For transfers: `POST /merchant/payouts` → OTP dialog → confirm →
   poll `getTransaction` until `8`/`4`/`-1`. For bulk: upload a sheet → the
   `awaiting_otp` row prompts → confirm → next row.
