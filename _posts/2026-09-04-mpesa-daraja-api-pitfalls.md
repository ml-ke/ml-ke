---
title: "Mobile Money API Security: Daraja Integration Pitfalls That Lose Real Money"
date: 2026-09-04 00:00:00 +0300
categories: [Fintech, Cybersecurity]
tags: [api-security, daraja, m-pesa, mobile-money, secrets-management, fintech-security, payment-security]
image:
  path: /assets/img/cover-mpesa-daraja-api-pitfalls.webp
  alt: A merchant server and the Safaricom Daraja gateway joined by a dashed payment rail, with red attack arrows at the credential, callback and push-prompt joints of the integration
---

## A payment rail with four open joints

Every M-PESA business integration is a small payments company. Your server exchanges a consumer key and secret for a short-lived OAuth token, fires an STK Push or C2B/B2C request at Safaricom's Daraja API, then — the part teams forget — **becomes a public endpoint that Safaricom calls back with the verdict**. Money moves on that rail, and the rail has joints: the credentials in your codebase, the callback URL you expose, the payment prompt a customer sees, and the humans who hold privileged access.

> **The key concept**
> Daraja authenticates the merchant and encrypts the channel. It cannot protect *your* side of the integration: a leaked consumer secret, an unverified callback, a confusing payment prompt, or an over-privileged insider turns a secure API into a pipe for your losses.

## Joint 1: Your consumer key is a spending key

Daraja apps authenticate with a **consumer key + consumer secret**, exchanged for a token that can initiate STK Push charges and, on B2C-enabled apps, move money *out*. Treating that secret as ordinary configuration is how integrations get drained.

Kenyan agency Quest Web, which has built M-PESA integrations for over a decade, states the exposure plainly: *"Leaked M-Pesa credentials can be exploited to initiate unauthorized transactions from your business account. If you accidentally commit credentials, rotate them immediately in the Daraja portal — don't just delete the commit, as Git history preserves it"* ([Quest Web](https://www.questdesigners.com/blog/mpesa-integration-to-website)). Leaking is the default: GitHub found **more than 39 million secrets exposed in 2024 alone** — the story in [Secrets in CI](/posts/secrets-in-ci-credential-leaks/), where this blog's own repo hit GitHub's GH013 push-protection block after a live token landed in a backup commit (fixed only by rebase, redaction, force push).

The scarier version is when rotation is not enough. In a widely shared [r/nairobitechies post from 2025](https://www.reddit.com/r/nairobitechies/comments/1m9juez/how_i_traced_an_mpesa_api_hack_to_a_single_line/), a Kenyan team reported attackers stealing money through their Daraja integration: *"no matter how they kept rotating their key and secret... the problem kept recurring,"* until they traced the loss to "a single line of code" in their own application. When rotation does not stop the bleeding, the leak is inside your code: a debug endpoint that echoes secrets, a backdoor from a rushed build, or an API route that lets anyone trigger payments.

## Joint 2: The people you trust become the API

Integrations rarely stop at your own server. Merchants route through aggregators, agencies recruit agents, contractors hold production keys. Every trusted hand on the rail is a new joint.

**The aggregator case.** In February 2020, Safaricom sued aggregation partner **East African Data Handlers** over **KES 20.3 million** in losses from 2016, when the firm was contracted to manage transactions across the Lipa na M-PESA and Buy Goods platforms ([Tech-ish](https://tech-ish.com/2020/02/06/safaricom-east-african-data-handlers-kes-20-million-m-pesa-loss/), reporting the Business Daily suit). Safaricom alleged the partner failed to remit money collected from merchants; East African Data Handlers countersued with the mirror-image allegation — that Safaricom employees held back-end access to the aggregation system and made irregular transactions without its knowledge. When money flows through a third party's access, a dispute over who held which back-end right *is* the security failure.

**The rogue-employee ruling.** On May 18, 2026, the High Court ordered Safaricom to pay **KES 9.9 million** to 11 subscribers (KES 900,000 each) whose betting histories and transaction records were pulled from its internal systems and handed to outside parties — Constitutional Petition E095 of 2026 ([Techweez](https://techweez.com/2026/05/18/safaricom-data-breach-high-court-ruling/)). The petitioners alleged rogue employees systematically sold subscriber data to betting firms between 2018 and 2019, a scheme they said touched **11.5 million subscribers** ([Nairobi Wire](https://nairobiwire.com/2026/04/safaricom-data-breach-case-judgment-11-million-subscribers.html)). Safaricom's defense was the classic one — *a rogue employee did it, so the company is not responsible* — and Justice Bahati Mwamuye rejected it. For any fintech running a payment API: **you own every credential and back-end right you issue**. Least privilege, monitoring and revocation are your legal defense, not optional hygiene — the same lesson as [The Three-Minute Insider](/posts/insider-threat-privileged-access-fintech/) (NCBA Rwanda) and [When the Contractor Has the Keys](/posts/vendor-risk-fintech-contractors/).

## Joint 3: The prompt is the product

STK Push works because a customer approves a payment with their PIN. The whole security model leans on the customer correctly reading **who** they are paying — and attackers lean back.

Unsolicited pushes are a documented nuisance shading into fraud: a 2026 [r/nairobitechies thread](https://www.reddit.com/r/nairobitechies/comments/1q75yqk/potential_mpesa_fraud_using_stk/) shows a user receiving a PIN prompt to pay **BETGR8_CS** for a transaction they never initiated. A distracted user approving an out-of-context prompt is the attack completing itself.

Around the prompt, fraudsters build fake confirmation layers. The Star's March 2025 explainer catalogs the patterns: **fake reversal instructions** (scammers posing as Safaricom care direct a victim to an agent, then talk the agent into transacting on the customer's line) and **fake or old M-PESA messages** — a doctored SMS that looks genuine but shows a LOCKED balance, used to pressure victims into "refunding" money that never arrived ([The Star](https://www.the-star.co.ke/news/2025-03-27-explainer-what-you-need-to-know-about-m-pesa-fraud-schemes)). A genuine M-PESA message comes from the M-PESA service line, never a customer's number; Safaricom's guidance is to forward suspicious messages to **456**.

For a business, Joint 3 is a UX responsibility: prompts that name your till/paybill, amount and account reference; confirmation inside *your* app before the push fires; and never asking customers to "confirm" outside the official M-PESA prompt.

## Joint 4: Callbacks, certificates, and clocks

The remaining joints are engineering hygiene, each with a documented footgun:

- **Callbacks must be HTTPS and verified.** Safaricom rejects go-live callback URLs that error or use plain HTTP ([KenZobe](https://www.kenzobe.com/blog/mpesa-daraja-api-errors)). But HTTPS only protects the channel — it does not prove the callback matches an STK request *you* initiated. The guard below joins callbacks to your request store before fulfilling orders.
- **Sandbox and production are different worlds.** The STK Push password is `base64(shortcode + passkey + timestamp)`; the timestamp exists so a captured request cannot be replayed — *"TimeStamp ensures that each transaction request is time-specific and unique... prevents resending a valid request to perform unauthorized transactions"* ([Koda School](https://kodaschool.com/blog/how-to-integrate-mpesa-daraja-api-with-node-js)). Quest warns of developers who "deploy to production with the sandbox certificate and wonder why every B2C request fails" — mixing environments breaks disbursements or leaves test credentials guarding real money.
- **Secrets belong in a vault.** Env vars, secret managers, push protection — the full pipeline in [ML Secrets Management](/posts/ml-secrets-management/) applies to the Daraja consumer secret.

## A 20-line guard: verify before you fulfil

Two checks stop forged callbacks and replayed requests, small enough to paste into any integration:

{% raw %}
```python
import base64, time

# --- Part 1: STK password = base64(shortcode + passkey + timestamp) ---
SHORTCODE = "174379"          # public sandbox shortcode
PASSKEY   = "bfb279f9aa9bdbcf158e97dd9a4673c3e8a1b6e1f"  # public sandbox passkey

def stk_password(ts: str) -> str:
    return base64.b64encode(f"{SHORTCODE}{PASSKEY}{ts}".encode()).decode()

def fresh(ts: str, window_min: int = 1) -> bool:
    t = time.strptime(ts, "%Y%m%d%H%M%S")
    return abs(time.mktime(t) - time.mktime(time.strptime("20260904120000", "%Y%m%d%H%M%S"))) <= window_min * 60

new_ts  = "20260904120000"    # issued now
old_ts  = "20260902120000"    # captured two days ago, replayed today

print("password(new):", stk_password(new_ts)[:24], "...")
print("password(old):", stk_password(old_ts)[:24], "...")
print("replayed request inside 1-min window?", fresh(old_ts))

# --- Part 2: verify a callback against requests WE initiated ---
sent = {"ws_CO_040920261200001234": {"order": "ORD-8821", "amount": 1250}}

def on_callback(payload):
    if payload.get("ResultCode") != 0:
        return "FAIL", f"transaction not completed (ResultCode={payload.get('ResultCode')})"
    rid = payload.get("CheckoutRequestID")
    if rid not in sent:
        return "REJECT", f"no matching STK request for {rid} -- do NOT fulfil order"
    return "FULFIL", f"match order {sent[rid]['order']} for KES {sent[rid]['amount']}"

forged = {"CheckoutRequestID": "ws_CO_999999999999999", "ResultCode": 0, "Amount": 1250}
genuine = {"CheckoutRequestID": "ws_CO_040920261200001234", "ResultCode": 0, "Amount": 1250}
print(on_callback(forged))
print(on_callback(genuine))
```
{% endraw %}

```
password(new): MTc0Mzc5YmZiMjc5ZjlhYTli ...
password(old): MTc0Mzc5YmZiMjc5ZjlhYTli ...
replayed request inside 1-min window? False
('REJECT', 'no matching STK request for ws_CO_999999999999999 -- do NOT fulfil order')
('FULFIL', 'match order ORD-8821 for KES 1250')
```

Two details. First, old and new passwords share the same first 24 characters — the timestamp changes only the **tail** of the Base64 string — so naive "have I seen this password?" checks fail; the timestamp-window check is the real replay control. Second, the forged callback carries `ResultCode: 0` — a "success" — yet the guard rejects it because no `CheckoutRequestID` matches the merchant's request store. **Never fulfil an order on a callback alone.**

## Closing the joints: a checklist

| Joint | Failure mode | Control |
|-------|-------------|---------|
| 1 — Credentials | Consumer secret in git, client code, or a leaked backup | Vault/env vars only; rotate on exposure; push protection; never ship the secret in a mobile app |
| 2 — Trusted access | Aggregator/employee back-end abuse; "rogue employee" defense | Least privilege; session recording; daily reconciliation; monitor back-end access |
| 3 — Payment prompt | Unsolicited/spoofed pushes; fake confirmations | Descriptive prompts; verify via official channels (456); never transact outside the M-PESA prompt |
| 4 — Hygiene | Forged callbacks; replays; sandbox/prod mix-ups | HTTPS callbacks; join callbacks to your request store; timestamp-window checks; separate environments |

## The bottom line

Daraja gives Kenyan businesses a world-class payment rail — then hands the security of the last mile back to the integrator. The losses above did not come from broken cryptography at Safaricom; they came from leaked spending keys, over-trusted insiders, unverified callbacks and payment prompts nobody was watching. Close the four joints and the rail holds.

## References

- Quest Web — [M-Pesa Daraja API Integration Guide](https://www.questdesigners.com/blog/mpesa-integration-to-website)
- Tech-ish Kenya — [Safaricom sues East African Data Handlers over KES 20M M-PESA loss](https://tech-ish.com/2020/02/06/safaricom-east-african-data-handlers-kes-20-million-m-pesa-loss/)
- Techweez — [Safaricom ordered to pay KES 9.9M over customer data breach](https://techweez.com/2026/05/18/safaricom-data-breach-high-court-ruling/)
- Nairobi Wire — [High Court to rule on Safaricom data breach case](https://nairobiwire.com/2026/04/safaricom-data-breach-case-judgment-11-million-subscribers.html)
- The Star — [Explainer: M-PESA fraud schemes](https://www.the-star.co.ke/news/2025-03-27-explainer-what-you-need-to-know-about-m-pesa-fraud-schemes)
- r/nairobitechies — [MPESA API hack traced to one line](https://www.reddit.com/r/nairobitechies/comments/1m9juez/how_i_traced_an_mpesa_api_hack_to_a_single_line/) · [Potential Mpesa fraud using STK](https://www.reddit.com/r/nairobitechies/comments/1q75yqk/potential_mpesa_fraud_using_stk/)
- Koda School — [Integrating M-PESA Daraja with Node.js](https://kodaschool.com/blog/how-to-integrate-mpesa-daraja-api-with-node-js)
- KenZobe — [M-Pesa Daraja API common errors](https://www.kenzobe.com/blog/mpesa-daraja-api-errors)

## Related posts

- [Secrets in CI: How Credential Leaks Actually Happen](/posts/secrets-in-ci-credential-leaks/)
- [The Three-Minute Insider: Privileged Access in Fintech](/posts/insider-threat-privileged-access-fintech/)
- [When the Contractor Has the Keys: Vendor Risk in Financial Systems](/posts/vendor-risk-fintech-contractors/)
- [Fraud ML in Mobile Money: the 70-Account Loophole](/posts/fraud-ml-mobile-money/)
- [ML Secrets Management](/posts/ml-secrets-management/)
- [MLOps for RegTech: Model Governance](/posts/mlops-regtech-model-governance/)
