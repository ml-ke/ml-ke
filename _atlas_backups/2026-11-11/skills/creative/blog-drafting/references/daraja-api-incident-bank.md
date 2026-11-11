# Daraja / Mobile-Money API Security — Verified Anchor Bank

Class: mobile-money API security (Fri Cybersecurity rotation). Covers Daraja
(M-PESA developer API) integration failures, aggregator/partner access abuse,
credential leaks around payment APIs, and the consumer-side STK-push fraud
that payment integrations enable. First used: Sep 4 2026 post
`mpesa-daraja-api-pitfalls` (published `_posts/2026-09-04-mpesa-daraja-api-pitfalls.md`).
Directly reusable for the Sep 6 "third-party API / BaaS integration security"
playbook slot and any future payment-API/aggregator post.

## Anchors (all verified with the sources listed)

### 1. Safaricom v. East African Data Handlers — KES 20.3M aggregation loss
- Events from **2016**; suit reported **Feb 2020**.
- EADH was contracted to manage transactions across the **Lipa na M-PESA and
  Buy Goods platforms** and recruit merchants (an aggregator).
- Safaricom alleged EADH failed to remit money collected from merchants;
  EADH countersued alleging **Safaricom employees held back-end access to the
  aggregation system** and made irregular transactions without its knowledge.
- Lesson: aggregation = third party's access is your exposure; a dispute over
  who held which back-end right IS the security failure.
- Sources: Tech-ish Kenya (Dickson Otieno, Feb 6 2020),
  https://tech-ish.com/2020/02/06/safaricom-east-african-data-handlers-kes-20-million-m-pesa-loss/
  (reports the Business Daily suit — BD itself is paywalled; cite Tech-ish +
  "reporting the Business Daily suit").

### 2. Constitutional Petition E095 of 2026 — High Court ruling vs Safaricom
- **May 18, 2026**: High Court ordered Safaricom to pay **KES 9.9M** to **11
  subscribers (KES 900,000 each)** whose **betting histories and transaction
  records** were pulled from internal systems and handed to outside parties.
- Presiding: **Justice Bahati Mwamuye**. Safaricom's "rogue employee" defense
  was **REJECTED** — company held liable for the access it granted.
- Petitioners' claim (pre-ruling reporting): rogue employees **systematically
  sold subscriber data to betting firms 2018–2019**, allegedly touching
  **11.5M subscribers**; WhatsApp-chat evidence quoted; petitioners led by
  Austin Taabu, represented by Mola Kimosop Advocates.
- Lesson: "you own every credential and back-end right you issue"; least
  privilege/monitoring/revocation is a legal defense, not optional hygiene.
- Sources: Techweez (May 18 2026),
  https://techweez.com/2026/05/18/safaricom-data-breach-high-court-ruling/ ;
  Nairobi Wire (Apr 22 2026, pre-judgment),
  https://nairobiwire.com/2026/04/safaricom-data-breach-case-judgment-11-million-subscribers.html

### 3. Quest Web — Kenyan agency M-PESA/Daraja integration guidance (quote source)
- "Leaked M-Pesa credentials **can be exploited to initiate unauthorized
  transactions from your business account**. If you accidentally commit
  credentials, **rotate them immediately** in the Daraja portal — don't just
  delete the commit, as Git history preserves it."
- Other documented footguns in the same guide: sandbox vs production use
  **different security credentials/certificates** ("deploy to production with
  the sandbox certificate and wonder why every B2C request fails"); go-live
  requires HTTPS callback URLs (HTTP = rejection); CSRF tokens for web-form
  payment initiation; mask phone numbers in logs (2547****5678); encrypt
  receipt numbers/amounts at rest.
- Source: https://www.questdesigners.com/blog/mpesa-integration-to-website
  (14 years / 85+ projects claim in byline — characterize as "Kenyan agency
  Quest Web").

### 4. The Star — M-PESA fraud scheme explainer (Mar 27 2025, Felix Kipkemoi)
- **Fake reversal instructions**: scammers posing as Safaricom care direct a
  victim to an M-PESA agent, then talk the agent into transacting on the
  customer's line.
- **Fake/old M-PESA messages**: doctored SMS that looks genuine but shows a
  LOCKED balance; used to pressure victims into "refunding" money never sent.
- Verification guidance: genuine M-PESA messages come from the M-PESA service
  line, never a customer's number; forward suspicious messages to **456**.
- Source: https://www.the-star.co.ke/news/2025-03-27-explainer-what-you-need-to-know-about-m-pesa-fraud-schemes

### 5. r/nairobitechies threads — SNIPPET-LEVEL ONLY (bot-walled, see SKILL.md)
- "How I Traced an MPESA API Hack to a Single Line of Code" (2025, id
  1m9juez): Kenyan team's Daraja integration drained; suspected consumer
  key/secret leak; **rotating both did not stop losses** until root cause
  traced to one line in their own application (community discussion suggests
  exposed creds or a backdoor in rushed/"vibe-coded" code).
- "Potential Mpesa Fraud Using STK" (2026, id 1q75yqk): user received an
  unsolicited PIN prompt to pay BETGR8_CS for a transaction never initiated —
  unsolicited-STK-push pattern.
- Citation rule: title + URL + only what the search snippet shows; phrase
  "a widely shared thread reported..." and note "cited at snippet level
  (bot-walled)" in the post and calendar.

### 6. Daraja mechanics (for demo/code sections)
- STK Push password = `base64(shortcode + passkey + timestamp)` — timestamp
  makes each request time-specific so captured requests can't be replayed.
  Sandbox shortcode 174379 + passkey
  bfb279f9aa9bdbcf158e97dd9a4673c3e8a1b6e1f are public test values.
  Source: Koda School, https://kodaschool.com/blog/how-to-integrate-mpesa-daraja-api-with-node-js
- Daraja rejects HTTP callback URLs / errors on go-live callback verification;
  use HTTPS via Let's Encrypt. Source: KenZobe,
  https://www.kenzobe.com/blog/mpesa-daraja-api-errors
- OAuth flow: consumer key+secret → short-lived access token → STK Push /
  C2B / B2C calls; callbacks POST ResultCode/CheckoutRequestID/MpesaReceiptNumber.

## Reusable demo recipe (verified, deterministic, no network)
Python guard with two parts, both runnable offline and output-quoted verbatim:
1. Replay control: `fresh(ts)` rejects an STK password replayed from an old
   timestamp (1-min window). Note: old/new passwords share the same first ~24
   base64 chars — the timestamp changes only the tail — so prefix matching
   alone is NOT a replay control.
2. Callback join: verify `CheckoutRequestID` exists in the merchant's own
   request store before fulfilling; a forged `ResultCode: 0` callback with an
   unknown ID → REJECT, genuine → FULFIL.
Full code in `_posts/2026-09-04-mpesa-daraja-api-pitfalls.md` (wrap fence in
`{% raw %}` if dict literals with `}}` are present). Cover metaphor used:
"payment rail with numbered open joints" (server ↔ Daraja gateway, red attack
arrows at joints 1/2/3 = LEAKED SECRET / FAKE CALLBACK / SPOOFED PUSH) —
distinct from the CI-pipeline key-leak cover and the vault cover.

## Cross-links that pair well
secrets-in-ci-credential-leaks (GH013/push protection), insider-threat-
privileged-access-fintech (NCBA), vendor-risk-fintech-contractors,
fraud-ml-mobile-money, ml-secrets-management, mlops-regtech-model-governance
(data governance/privacy), fraud-model-drift-monitoring (reconciliation).

## Mobile-identity / SIM-swap anchors (verified 2026-09-11 — the Fri Cybersecurity lane above the rail)

Body-level verified; use for authentication/account-takeover slots (distinct from the
integration-joint and aggregator layers already banked above).

- **DTB v Safaricom** — High Court at Machakos, Justice Asenath Ongeri, **June 18 2026**,
  dismissing both appeals and upholding the Mavoko Chief Magistrate ruling (Hon. R.W. Gitau,
  Mar 2024): **KES 4,418,601** taken after a **Feb 6 2022** SIM swap executed via an M-PESA
  agent; split **60:40 → Safaricom KES 2,630,000 / DTB KES 1,788,601**. Ruling quote: "A bank
  cannot hide behind a customer's PIN when it is presented with a series of transactions that
  are so glaringly out of the ordinary that a reasonable banker would have been put on inquiry."
  Timeline: line reinstated Feb 7; losses from ~05:23 Feb 8 across three days, each transfer
  under DTB's **KES 2m daily limit**. Sources: TechCabal (Jul 13 2026), Techweez, Tech-ish.
- **Safaricom volume** (CCSO Nick Mulila, The Star, Nov 6 2024): "In a month we get about 40
  fraudulent swaps out of about 750K swaps"; ~28,000 swaps/day; M-PESA wallet access
  temporarily blocked after swaps made through partners.
- **INTERPOL African Cyberthreat Assessment 2026** (via Dawan): SIM-swap fraud **+327% during
  2025**, 123,000+ fraudulent SIM cards, ~US$3.8m taken from mobile wallets; Kenya cited as a
  regional hotspot.
- **SEC @SECGov on X** (Jan 9 2024): fake Bitcoin-ETF approval after a SIM swap of the SEC cell
  number — no SEC systems touched (BleepingComputer).
- **Scattered Spider**: CISA/FBI advisory **AA23-320A** (Nov 2023, updated Jul 2025) — SIM swaps +
  help-desk social engineering to reset passwords / transfer MFA tokens, then self-enrol MFA;
  MGM Resorts ~$100m (Sep 2023, NBC); Noah Michael Urban sentenced 10 years Aug 20 2025,
  ~$13m restitution (Krebs on Security).
- **0ktapus** (2022, Group-IB): 169 Okta-impersonating domains, real-time code relay to Telegram —
  9,931 credentials + **5,441 records containing MFA codes** across 136 orgs (Help Net Security).
- **WindRelay + SpyNote** (Aug 2026, Group-IB via Malwarebytes): NFC-relay malware, 13-minute
  bank-impersonation call, loan arranged in the victim's name — on-device OTP capture.
- **Standards**: NIST SP 800-63B restricts PSTN out-of-band verification; CISA's MFA page pushes
  FIDO/WebAuthn + number matching; FCC SIM-swap/port-out rules in force Jul 8 2024 (Federal Register).

**Reusable demo recipe:** a 30-line step-up gate modelling the Kariuki pattern — number re-bind
age, new beneficiary, and "crowds the daily limit" signals scored by COUNT (≥2 → step-up), showing
PIN-only approval releases the full amount while the signal gate holds it. Verbatim output quoted
in the `sim-swap-otp-interception-mobile-banking` post.
