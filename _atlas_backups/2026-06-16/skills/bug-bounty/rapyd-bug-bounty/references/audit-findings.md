# Rapyd Audit — Honest Status of All Findings

## OpenAPI Spec (rapyd-openapi.yaml, 33,511 lines)
These describe api.rapyd.net — ALL UNVERIFIED against live API:

| Finding | Source | Verified? | Needs |
|---------|--------|-----------|-------|
| Idempotency missing on 192+ POST | No `idempotency-key` required in spec | ❌ No | Iceland prod account |
| Mass assignment (inline beneficiary/customer) | `beneficiary` + `customer` as string|object | ❌ No | Iceland prod account |
| PII in response schemas | Full PII fields in response bodies | ❌ No | Iceland prod account |

## TypeScript Sample Code (github.com/Rapyd-Samples/rapyd-ts-client)
NOT an in-scope target. Code runs on merchant's server. NOT submittable.

- `src/utils/validateRapydWebhook.ts:10` — Hardcoded ngrok URL (merchant-side)
- `src/controller/webhook.controller.ts:21` — Array OOB (merchant-side)
- `src/controller/webhook.controller.ts:14` — Silent void return (merchant-side)
- `src/strategies/local.strategy.ts:33-38` — Password hash in session (merchant-side)
- `src/controller/checkout.controller.ts:47` — Cart IDOR (merchant-side)
- `src/utils/signRapydRequest.ts:16` — Self-referential HMAC (merchant-side)

## HMAC Signing (verified against sandboxapi.rapyd.net)
```python
def sign(secret_key, access_key, method, path, body=""):
    salt = base64.b64encode(os.urandom(12)).decode()
    ts = str(int(time.time()))
    to_sign = method.lower() + path + salt + ts + access_key + secret_key + body
    h = hmac.new(secret_key.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
    signature = base64.b64encode(bytes.fromhex(h)).decode()
    return salt, ts, signature
```
