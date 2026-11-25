# Worked Example: Blockchain.com Partner API Key Verification

## The Finding That Wasn't

During Blockchain.com recon (Bugcrowd program), partner API keys were discovered in two places:
- `https://blockchain.info/Resources/wallet-options.json` — public endpoint, no auth
- `github.com/blockchain/wallet-options/prod/wallet-options.json` — public repo

## Exposed Keys Found

| Key | Source | Value (truncated) |
|-----|--------|-------------------|
| SFOX API Key | `partners.sfox.apiKey` | `f31614a7-5074-49f2-8c2a-bfb8e55de2bd` |
| Plaid | `partners.sfox.plaid` | `0b041cd9e9fbf1e7d93a0d5a39f5b9` |
| Plaid Env | `partners.sfox.plaidEnv` | `production` |
| SiftScience | `partners.sfox.siftScience` | `a19cc360a1` |
| ShapeShift API Key | `shapeshift.apiKey` | `b7a7c320c19ea3a8e276...` |

## Verification Results

### SFOX Key (`f31614a7-...`)
- **Auth mechanism** (from `bitcoin-sfox-client/src/api.js`):
  ```javascript
  headers['X-SFOX-PARTNER-ID'] = this._apiKey;  // Custom header
  // URL: https://api.sfox.com/v2/partner/blockchain/{endpoint}
  ```
- **Live test**:
  ```bash
  curl -sL "https://api.sfox.com/v2/partner/blockchain/quotes" \
    -H "X-SFOX-PARTNER-ID: f31614a7-5074-49f2-8c2a-bfb8e55de2bd"
  # Response: "Cannot GET /v2/partner/blockchain/quotes"
  ```
- **Verdict**: SFOX was acquired by Zero Hash. Partner endpoints at `api.sfox.com` no longer exist.

### Plaid Key (`0b041cd9e9fbf1e7d93a0d5a39f5b9`)
- **Auth mechanism**: Plaid requires BOTH `client_id` AND `secret` for API authentication
- **Verdict**: Cannot authenticate with `client_id` alone. The `secret` is not exposed.

### ShapeShift Key (`b7a7c320...`)
- **Platform status**: ShapeShift converted from centralized exchange to DEX (decentralized exchange) in 2021
- **Verdict**: Platform defunct as a centralized service. Key worthless.

### SiftScience Key (`a19cc360a1`)
- **Purpose**: Client-side fraud detection. Embedded in JS for fingerprinting.
- **Verdict**: Intentionally public by design. Not a secret.

## Key Takeaway

Of 5 partner keys exposed, ZERO were usable. The finding was "information disclosure" without demonstrated impact → P5 (Informational). The report was not submitted.

**Root cause of the false positive**: Writing the report BEFORE verifying the keys against the partner APIs. The pattern was:
1. Find keys in config ✅
2. Write report describing exposure ✅
3. User challenges impact ✅
4. Verify keys → all dead ends ✅
5. Delete report (unwritten, caught in time) ✅

**Correct order** is: find → verify → classify → only THEN write report.
