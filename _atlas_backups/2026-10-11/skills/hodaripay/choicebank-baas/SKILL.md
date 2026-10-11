---
name: choicebank-baas
description: >-
  Workbench reference for Choice Bank BaaS (Kenya banking-as-a-service).
  Use when building or debugging ANY integration that calls Choice Bank:
  signed-envelope requests, STK push deposits (depositFromMpesa), general
  transfers (applyForTransfer: M-PESA / PesaLink / Choice-to-Choice), Paybill /
  Till (applyForMpesaBusinessTransfer), SME & merchant bulk transfer
  (generalBulkTransfer / batchTransitionalFundsDisbursement), IMT,
  RTGS/EFT/SWIFT, OTP (sendOtp / confirmOperation / resendOtp), bank-code
  lookups, webhook callbacks, and account onboarding. Includes the official
  GitBook doc URL (https://choice-bank.gitbook.io/choice-bank/) plus the
  empirically-verified caveats the docs understate — endpoint names that 500 in
  sandbox, literal-vs-code bank values, OTP behaviour, and sandbox-only quirks.
metadata:
  source: >-
    Official docs https://choice-bank.gitbook.io/choice-bank/ (llms.txt +
    *.md pages) cross-checked against live sandbox probes from the HodariPay
    repo (https://baas-pilot.choicebankapi.com, sender `yucan`).
  last_modified: Sat, 15 Aug 2026 00:00:00 GMT
---

# Choice Bank BaaS — integration reference (verified)

Everything below was **proven against the live sandbox** in the HodariPay
repo, then cross-checked against the official GitBook docs
(`https://choice-bank.gitbook.io/choice-bank/`, index `…/llms.txt`, pages are
`.md`-suffixed). Where the docs and the sandbox disagree, the sandbox wins and
the difference is called out under "Caveats the docs understate".

## Environments & credentials

| | Sandbox (staging) | Live |
|---|---|---|
| Base URL | `https://baas-pilot.choicebankapi.com` | `https://baas.choicedigitalbank.com` |
| Sender ID | provided by account manager (e.g. `yucan`) | live sender |
| Sender key | 64-char private key (signing only) | live key |
| Paybill backing merchant accounts | sandbox `4101847` | prod `444174` |

- **Live base URL**: `https://baas.choicedigitalbank.com`. The older
  `api.choicebankapi.com` / `baas.choicebankapi.com` hosts are NXDOMAIN /
  dead (verified 2026-09-02, ADR-0017 in the HodariPay repo). Do not copy an
  old `.env` or SDK sample that points at `*.choicebankapi.com`.

- Creds are issued by the account manager, never ship in code. The gateway
  rejects **every** unsigned request, so there is no way to read the API
  without a valid sender + key.
- The gateway only accepts `POST` + a signed JSON envelope. `GET` requests
  return code `12000 "Type of GET request is not supported"` — there is no
  public OpenAPI/Swagger; the GitBook docs + probing are the source of truth.

## Signed envelope (do NOT change the algorithm)

Every request body:

```json
{
  "requestId": "APPREQ…",      // unique per call
  "sender": "<senderId>",
  "locale": "en_KE",
  "timestamp": 1650533105687,  // epoch ms
  "salt": "<random>",
  "params": { ... },
  "signature": "<sha256>"
}
```

Signature = flatten the body (minus `signature`) to `key=value` pairs
(dots for objects, `[i]` for arrays), append `senderKey=<privateKey>`, sort
ASCII, join with `&`, SHA-256 hex. Response envelopes are signed the same way
(sender `apigw.baas.choice`) — always verify.

- Success code is `00000` ("Completed successfully").
- Common codes: `12000` GET not supported · `12001` invalid params · `12002`
  invalid request id · `12003` invalid sender id · `12004` invalid signature ·
  `12010` invalid mobile number · `12020/12021` invalid txType/txStatus ·
  `13000` account does not exist · `13224` missing [userId|mobile] ·
  `14009` insufficient balance · `14015` M-PESA withdraw min KES 10 ·
  `14042/14043` whitelist errors · `15001` query time range · `15002` page
  oversize (pageSize > 50) · `11200-11203` SMS OTP throttled/locked/wrong.
  Full list in the docs' Error Codes page.

## Probing to learn a new endpoint (safe, read-only)

Send a signed POST with `{}` / partial params. The API replies
`12001 "please input X"`, revealing the real param name one field at a time.
Example — `depositFromMpesa`: `{}` → "please input payee account";
`{accountId}` → "please input mpesa mobile No."; `{accountId, amount, mobile}`
→ `00000` + `data.txId`.

## Endpoint catalog (params verified or doc-confirmed)

### Deposits
- `POST /trans/depositFromMpesa` — STK push prompt.
  `{accountId, mobile: <9-digit no 0/254>, amount: <KES whole number>}`.
  Returns `data.txId` (`UTRANS…`); `00000` = prompt initiated, NOT settled.
  Docs: for current/wallet accounts the STK can only prompt the account's own
  Safaricom number; SME accounts can prompt any M-Pesa number.
- Paybill deposit: customer pays Paybill shortcode (sandbox `4101847`,
  prod `444174`) with the **Choice account number** as the account field.

### Transfers
- `POST /trans/v2/applyForTransfer` — **General Transfer**: M-Pesa / Airtel /
  PesaLink / **Choice-to-Choice (internal)**. The universal rail.
  `{payerAccountId, payeeBankCode, payeeAccountId, currency:'KES', amount,
  payeeAccountName?, paymentPurpose?, remark?, payeeMobileForNotification?}`.
  - M-Pesa: `payeeBankCode:'M-PESA'`, `payeeAccountId` = 9-digit phone,
    `payeeAccountName` optional.
  - PesaLink: real bank code (e.g. `68` Equity), `payeeAccountName` MANDATORY,
    `paymentPurpose` = PesaLink code (e.g. `OTHR`, `SALA`, `RENT`).
  - **Choice→Choice: `payeeBankCode:'46'` (Choice Microfinance Bank)** — the
    literal `'CHOICE'` is REJECTED (`10000 invalid payee BankCode`). Target is
    another Choice account id.
  - Multi-currency per docs (KES/USD/GBP/EUR/TZS/UGX/RWF) but PesaLink is KES.
- `POST /trans/v2/applyForMpesaBusinessTransfer` — Paybill / Till (B2B).
  `{payerAccountId, payeeShortCode, payType: 0|1 (Paybill|Till),
  payeeReferenNumber? (Paybill account), amount, description?}`.
  **OTP-gated like every transfer** (verified on sandbox 2026-08): initiate →
  `sendOtp` (`businessId=txId`) → `confirmOperation` → settles. Without the
  code it parks at txStatus 1 and times out (−1) — it does NOT settle on its
  own. Some paybills reject incoming B2B with `10000 paybill owner restricted`.
  Also exposes `POST /trans/generateQRCode {accountId, amount}` → base64 PNG
  M-Pesa PayBill QR (whole KES, ≤250,000; amount is mandatory — `12001`
  without it).
- `POST /trans/v2/generalBulkTransfer` — **SME bulk transfer** (docs; the
  repo's client had a stale `/trans/v2/applyForBulkTransfer` which 500s).
  `{payerAccountId, txDetails: [{benificiaryName, benificiaryType 0|1,
  payType?, benificiaryAccount, benificiarySubAccount?, benificiaryBankCode,
  amount, paymentPurpose?, remark?, payeeMobileForNotification?}]}` →
  `data.bulkPaymentOrderId`. Query: `POST /trans/queryBatchTransactionResult
  {orderId, pageNo, pageSize}`.
- `POST /trans/batchTransitionalFundsDisbursement` — **Merchant bulk transfer**
  (whitelisted, **no OTP**). Supports M-Pesa B2C/B2B, Airtel B2C,
  Choice-to-Choice, PesaLink. `{payerAccountId, orgBatchId?,
  beneficiaryArray:[{orgTxId?, payeeAccountName, payeeAccountId,
  payeeBankCode, payeeType?, payType?, payeeSubAccount?, currency, amount,
  paymentPurpose?, remark?, payeeMobileForNotification?}]}` → `data.orderId`.
  Query: `POST /trans/getInternalBatchTransactionResult {orderId?|orgBatchId?}`.
  Callback `0004` carries `resultArray`.
- `POST /trans/applyForImtTransfer` — **IMT (international to M-Pesa)**,
  whitelisted, no OTP. NOTE the path has **no `v2`**; `/trans/v2/…` 500s.
  Many mandatory sender KYC fields (orgTxId, ImtSenderName, …
  imtFundsOriginCountryCode, payeeAccountId, payeeAccountName, purpose,
  imtForeignCurrency, imtConversionRate, imtOriginalAmount, amount,
  currency:'KES', remark≥2 chars). Query: `/trans/getImtTransferResult`.
- `POST /trans/quickTransfer` — exists but returns `11000 No permission`
  unless whitelisted; use `applyForTransfer`.
- `applyForImtTransfer`/`applyForInnerTransfer` variants seen in other SDKs
  return HTTP 500 in sandbox — use the doc'd paths above.

### OTP (all transfers to M-Pesa/PesaLink/Choice are OTP-gated)
- `POST /common/sendOtp` `{businessId: <txId>, otpType: 'sms'|'email'|'whatsapp'}`
  — whatsapp OTP only for applyForTransfer / applyForMpesaBusinessTransfer /
  generalBulkTransfer.
- `POST /common/confirmOperation` `{businessId: <txId>, otpCode}` — advances
  tx from status 1 → 2 (processing).
- `POST /common/resendOtp` `{businessId, otpType:'sms'|'email'}`.
- **Without OTP the tx hangs at status 1 then times out (-1).** OTP codes are
  real SMS to the account holder's phone; there is no fixed sandbox code.
  `11203` = wrong code; `11200` too many codes today; `11201` too frequent;
  `11202` locked after too many failures.
- `POST /common/groupApplications` `{applicationIdList:[]}` — group single
  txs to confirm with ONE OTP (`data.applicationId`).
- `POST /common/sendAccAdminOtp` — SME account-administrator OTP.

### Queries
- `POST /query/getTransResult {txId}` → full tx detail.
  `txStatus`: `1` pending, `2` processing, `4` failed, `8` success,
  `-1` **timeout**.
- `POST /query/getTransList {userId?, accountId, txType:[], txStatus:[], ·
  startTime, endTime (epoch ms), pageNo, pageSize, orderByDesc}`
  → `{totalRows, result:[…]}`. Docs mark `txType`/`txStatus` required; the
  sandbox tolerates omission. `pageSize` > 50 → `15002`.
- `POST /query/getAccountBalance {accountId}` — **HTTP 500 for SME accounts in
  sandbox**; degrade to stored balance.
- `POST /query/getMerchantAccountDetails {accountId}` — **`13000` for SME
  accounts in sandbox**; use onboarding-status query instead.

### Lookups
- `POST /staticData/getBankCodes` → `data.bankCodeList` (`M-PESA`, `46` =
  Choice Microfinance Bank, `68` Equity, `01` KCB, …). Reliable + cached.
- `POST /staticData/getMonthlyIncomeEnumeration`,
  `POST /staticData/getEmploymentEnumeration` — current-account onboarding.
- `POST /staticData/getOperationalAccounts {type: 0|1|2 (FX|RTGS|EFT)}`,
  `POST /staticData/getFcyCooperativeBanks`.

### Callbacks / webhooks
- `POST /choicebaas/callback` — reply body `ok` / HTTP 200 to stop retries
  (≤5 attempts). Signature present; **always verify** (HodariPay's security
  backlog flags it as not enforced — enforce it).
- Types: `0001` personal onboarding · `0002` tx result · `0003`/`0022` balance
  · `0004` merchant bulk · `0005` wallet upgrade · `0006` SME opening · `0007`
  utility · `0008` bulk (SME) · `0012` multi-account open · `0013`/`0026` FX ·
  `0014` bulk airtime · `0015` file job · `0016` interest · `0019` closure ·
  `0020` OTP (partner-delivered) · `0021` status change · `0023` inter-bank ·
  `0024` profile check · `0025` fixed deposit · `0027` IMT · `0028` CNY.

### Onboarding
- SME: `applyForSmeOnboarding` (OTP-gated) → confirm OTP (`businessId =
  onboardingRequestId`) → `submit{Store,Company,Partnership,Organisation}
  OnboardingRequest` → members/shareholders → `uploadMedia` (KYCF codes) →
  `submitOrPullBackRequest`. Status query: `getBusinessOnboardingStatus`.
  Status: 1 initiated · 2 processing · 3 passed · 4 rejected · 7 opened ·
  9 manual review (docs also show 7 as "expired" — ambiguous).
- Personal wallet: `/onboarding/v3/submitEasyOnboardingRequest` (single call,
  photos inline, `backSidePhoto` only for non-passport ID types).
- Personal current: `/onboarding/submitOnboardingRequest` (text KYC: KRA,
  employment A–F, income, address, idType 101 national / 102 alien / 103
  passport) → **`uploadMedia` within 30 minutes** (KYCF00001..06, `mediaType`
  per file, `mediaBase64`, `contentType` pdf/image) → Choice reviews → callback
  `0001` / `getOnboardingStatus` / `getUserKyc`. **NO OTP step** — the
  30-minute window is for documents, not a phone code (the wallet OTP model
  must NOT be generalised to current).
- **Current-account Smile ID mandate (since 2026-01-19)**: the selfie
  (KYCF00006) must be **verified via Smile ID through the partner's front-end**
  (`skipApiSubmission: true`); a raw base64 selfie leaves
  `getUserKyc.profileCheck` at `0` and the application parks at Choice status
  `9` (manual review) — **not API-cancellable** (`/onboarding/cancelOnboardingRequest`
  only cancels status `1`), effectively dead. `profileCheck`: 0 not checked,
  1 submitted, 2 validated, 3 declined, 4 processing; callback `0024` reports
  it. Wallet→current upgrade: `/onboarding/walletAccountUpgrade`
  (`userId, kraPin, selfiePhoto` mandatory; front/back/passport optional).
- Recovery: `getUserKyc`, `getOnboardingRequestId` (by `userId` OR `mobile`).
  Full verified detail + doc URLs: see `references/current-account-onboarding.md`.

## Caveats the docs understate (verified the hard way)

1. **Endpoint-name drift in third-party SDKs.** The repo's `client.js` port
   used `/trans/v2/applyForBulkTransfer` and `/trans/v2/applyForImtTransfer`
   — **both HTTP 500 in the sandbox for every param shape**. The doc'd names
   are `/trans/v2/generalBulkTransfer` and `/trans/applyForImtTransfer`.
   Always re-check the current GitBook before trusting a copied SDK.
2. **Choice-to-Choice = bank code `46`**, not the literal `'CHOICE'` (which
   fails `10000 invalid payee BankCode`). `46` is only discoverable from
   `/staticData/getBankCodes`; the docs' General Transfer page just says
   "internal transfers".
3. **STK-push mobile must be 9 digits, no leading 0/254** — `0797…` and
   `+254…` both return `12010`. Whole KES amounts only for deposits.
4. **`getAccountBalance`/`getMerchantAccountDetails` misbehave for SME
   accounts** in the sandbox — always have a stored-balance fallback.
5. **OTP is not simulatable** — there is no sandbox test code; codes are real
   SMS. Some M-Pesa recipient numbers fail `TXERR0002 Invalid Account` at
   settlement even when they work as STK-push payers (Choice provides payee
   test accounts; use a known-good one).
6. **`getTransList` docs say `txType`+`txStatus` are required; the sandbox
   accepts their omission.** Keep them optional but be ready for stricter
   behaviour live.
7. **`-1` timeout is a real, distinct terminal state** (initiated but not
   confirmed in time) — surface it separately from `failed` (4) in UI.
8. **Whitelisting gates several features**: bulk (both flavours), IMT,
   OTP-exempt rails, quickTransfer, and some B2C rails return
   `14042/14043`/`11000` until the account manager whitelists the account.
9. **Callback replies**: body `ok` (exact string) or 2xx stops retries; 5
   attempts max. Log `processed=false`-style flags — a swallowed handler error
   looks like success.
10. **Airtel Money remark constraints**: alphanumeric only, max 64 chars
    (`DP00900001013` otherwise).
11. **Reading the official GitBook**: search engines index only ~8 of its
    pages — web_search can NOT verify onboarding/account endpoints. Fetch
    `https://choice-bank.gitbook.io/choice-bank/llms.txt` for the full page map,
    then curl any page as markdown by appending `.md` to its URL
    (`…/account/current-account.md`). GitBook docs follow
    apply → `sendOtp`/`confirmOperation` (businessId) → poll/callback
    universally, but the account pages carry the real specifics.

## Integration checklist

- Probe before coding (`{}` → next required param).
- Verify response signatures.
- Persist `otpRequired`/`operationRef` (the Choice txId) so a status-polled
  row can still offer enter/resend OTP.
- Poll `getTransResult` for non-terminal sends (callbacks can be missed).
- Map `txStatus` -1/1/2/4/8 and TXERR codes to user-friendly messages.
- Time out / retry ChoiceBank HTTP calls (TLS trust store needed in AOT/runtime
  images — `debian:bookworm-slim` has no CA certs by default).
