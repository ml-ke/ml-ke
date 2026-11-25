# Scope Verification Case Study: Rapyd Sample Code Mistake

## What Went Wrong

During a Rapyd bug bounty research session, a full 2 days of work was invested analyzing `github.com/Rapyd-Samples/rapyd-ts-client` — sample code that merchants deploy to integrate with Rapyd's API. **The bugs were real, but they weren't on any in-scope target.**

The mistake: assuming that because Rapyd publishes the code and links to it from their docs, bugs in it would affect `api.rapyd.net`.

## The Architecture That Should Have Been Checked First

```
rapyd-ts-client/
├── rapyd-client/          ← Auto-generated OpenAPI client (points TO Rapyd's API)
│   └── Base: sandboxapi.rapyd.net
│
└── src/                   ← Sample merchant application (runs on MERCHANT's server)
    ├── webhook.controller.ts   ← Merchant's server
    ├── validateRapydWebhook.ts ← Merchant's server
    ├── signRapydRequest.ts     ← Merchant's server
    ├── checkout.controller.ts  ← Merchant's server
    └── local.strategy.ts       ← Merchant's server
```

Everything in `src/` runs on the MERCHANT's server. Only `rapyd-client/` talks to Rapyd's actual API.

## The 6-Question Framework (applied)

1. **Where does this code run?** → `src/` runs on merchant's server. NOT api.rapyd.net.
2. **Is the endpoint on the target?** → There is no `POST /webhook/rapyd` on api.rapyd.net. Rapyd SENDS webhooks TO merchants.
3. **Verify against live?** → PoC ran against a local Node.js server. That proved our code works, not Rapyd's vulnerability.
4. **Listed in scope?** → GitHub repos are NOT listed as Rapyd targets. Only domains.
5. **Vendor docs?** → "Use this sample code" means it runs on the MERCHANT's infra.
6. **Spec vs live?** → OpenAPI spec analysis was treated as confirmed vulnerability. Live api.rapyd.net might behave entirely differently.

## The Cost

- **3 chained reports** (012 Webhook Takeover, 013 Cart IDOR, 014 Session→HMAC) were based on sample code — none submittable
- **1 report** (016 No Security Scheme) was a documentation issue — not exploitable
- Only **2 research notes** (013 idempotency/mass assignment, 015 PII) might eventually be valid — and both need production API access to verify

## Lesson Embedded

Before ANY report writes up findings from a GitHub repo, answer the 6 questions. If the answer to #1 is "merchant's server" and #4 says no source code target — **do NOT write the report.** Spend the time on something that targets a live, in-scope endpoint.
