# Blockchain.com Bug Bounty Program Intelligence

**Platform**: Bugcrowd (started Apr 2025, made public Jun 2025)
**Payouts**: P1 $7K-$10K, P2 $3K-$5K, P3 $700-$1,250, P4 $100-$250, P5 $0
**Required header**: `X-Bug-Bounty: <username>` on all HTTP traffic
**Signup email**: h0d4r1@bugcrowdninja.com (password stored in memory)
**Bounties rewarded**: 9 (as of Jun 2026) — low competition

## Scope

### In-Scope Targets
| Target | Type | Tech |
|--------|------|------|
| `www.blockchain.com` | Web (NextJS) | Cloudflare-protected, JS challenge |
| `api.blockchain.info` | API | Nabu Gateway + ETH RPC proxy |
| `ws.blockchain.info` | WebSocket | `wss://ws.blockchain.info/inv` |
| iOS/Android Wallets | Mobile | iOS (Swift), Android (Kotlin) |

### Verified OOS
- `exchange.blockchain.com` — behind Cloudflare
- `wallet-helper.blockchain.com` — nginx, all paths return 404
- `support.blockchain.com` — Zendesk, third-party
- `blog.blockchain.com` — Medium, third-party
- `docs.blockchain.com` — GitBook, third-party
- `api.blockchain.com` (exchange API v1) — behind Cloudflare

## GitHub Source Repos (github.com/blockchain)

| Repo | Content | Value for BB |
|------|---------|-------------|
| `wallet-options` | **PROD config**: partner API keys (SFOX, Plaid, SiftScience, ShapeShift), hot wallet ETH addresses, all internal domain names, dev/staging configs | **HIGH** — API key exposure |
| `network-config` | Internal blockchain node RPC URLs across 8 chains (ETH, Polygon, BNB, Avalanche, Arbitrum, Optimism, Base, Stellar) | **HIGH** — maps full RPC backend |
| `service-my-wallet-v3` | Express.js wallet backend (bitcoinjs-lib 2.2.0, bip39 2.3.1) | **MEDIUM** — 11 vulns found in code audit |
| `blockchain-wallet-v4-frontend` | TypeScript React wallet + Nabu Gateway API client | **MEDIUM** — maps all API calls |
| `blockchain-wallet-v4-frontend` | Main frontend app (the wallet UI) | **LOW** — UI code |
| `My-Wallet-V3` | Core wallet library (blockchain-wallet-client v3.43.8) | **LOW** — crypto ops |
| `docs-exchange-api` | Exchange API docs (Slate-based) | **LOW** |
| `coin-definitions` | Asset definitions, logos | **LOW** |

## Live Accessible Endpoints (No CF Challenge)

### Static Config
```
GET https://blockchain.info/Resources/wallet-options.json
  → Full prod config: partner keys, domains, hot wallet addresses
  → Also accessible at: www.blockchain.com/Resources/wallet-options.json
```

### Public Blockchain Data
```
GET https://blockchain.info/uuid-generator?format=json&n=3
  → {"uuids": ["...", "...", "..."]} — generates UUIDs without auth

GET https://blockchain.info/multiaddr?format=json
  → Public blockchain data (latest block, fees, addresses)

GET https://blockchain.info/mempool/fees
  → Fee estimates {limits, regular, priority}
```

### Nabu Gateway (api.blockchain.info/nabu-gateway/)
```
Headers required by auth endpoints:
  - user-agent (any browser UA)
  - x-app-version (e.g. 6.27.2)
  - x-client-type (WEB, EXPLORER)
  - x-device-id (any UUID)

Unauthenticated:
  GET  /nabu-gateway/countries?scope=SIGNUP       → 200 (country list)
  GET  /nabu-gateway/countries/US/states          → 200 (state list)

Exists but needs params (returns 400 not 404):
  POST /nabu-gateway/auth                          → 400 MISSING_PARAM
  POST /nabu-gateway/users                         → 400 MISSING_PARAM
  POST /nabu-gateway/signup                        → requires countryCode + recaptcha

Protected (returned UNKNOWN_USER or INVALID_CREDENTIALS):
  GET  /nabu-gateway/kyc/tiers                     → UNKNOWN_USER
  GET  /nabu-gateway/kyc/extra-questions            → UNKNOWN_USER
  POST /nabu-gateway/onboarding/quiz               → UNKNOWN_USER
  GET  /nabu-gateway/address-capture/find           → UNKNOWN_USER
  GET  /nabu-gateway/trades                         → INVALID_CREDENTIALS
  GET  /nabu-gateway/price                          → empty (no response)
  GET  /nabu-gateway/currencies                     → empty

Custom headers revealed in error messages:
  authorization, x-wallet-guid, blockchain-ipcountry, x-wallet-email,
  x-signature, x-hmac-signature, x-auth-client, x-session-id,
  blockchain-auth, blockchain-origin, blockchain-csrf,
  x-sofi-jwt-aes-ciphertext, x-sofi-aes-iv, x-sofi-aes-tag,
  x-sofi-aes-key-ciphertext, x-taxbit-signature
```

### ETH RPC Proxy (api.blockchain.info/eth/nodes/rpc)
```
Geth v1.16.7-stable / linux-amd64 / go1.25.1
Net version: 1 (Ethereum Mainnet)
Synced: true

Available methods: web3_clientVersion, eth_blockNumber, net_version,
  net_peerCount, eth_gasPrice, eth_syncing, eth_getLogs, eth_call

BLOCKED methods: admin_nodeInfo, debug_traceTransaction, eth_coinbase,
  personal_*, miner_*

Behavior: Standard JSON-RPC with admin methods disabled.
```

## Source Code Audit Findings (11 Vulnerabilities)

### Filed under service-my-wallet-v3/src/

| # | Finding | CVSS | File |
|---|---------|------|------|
| 1 | GUID-based IDOR — wallet ownership by GUID only, no session binding | 8.6 | server.js, api.js |
| 2 | Password via GET query string — `req.query` has priority over `req.body` | 7.5 | server.js:224 |
| 3 | No rate limiting — unlimited login attempts per GUID | 7.5 | server.js (entire) |
| 4 | CSRF/Replay — password sent with every request, no CSRF tokens | 8.1 | server.js |
| 5 | Weak PBKDF2 — 5000 iterations (OWASP recommends 600K) | 5.9 | wallet-cache.js:163 |
| 6 | Race condition — concurrent wallet logins overwrite cache promises | 6.8 | wallet-cache.js:108-119 |
| 7 | Wallet GUID in URL path — logged everywhere | 5.3 | server.js:26 |

### Filed under wallet-options/prod/

| # | Finding | CVSS | Details |
|---|---------|------|---------|
| 8 | **SFOX prod API key** | 9.1 | `f31614a7-5074-49f2-8c2a-bfb8e55de2bd` — in git + live |
| 9 | **Plaid prod key** | 9.1 | `0b041cd9e9fbf1e7d93a0d5a39f5b9` (plaidEnv: production) — in git + live |
| 10 | **SiftScience key** | 9.1 | `a19cc360a1` — in git + live |
| 11 | **ShapeShift API key** | 9.1 | `b7a7c320c19ea3a8e276...` — in git + live |
| 12 | Hot wallet ETH addresses exposed | 5.3 | swap/exchange/simplebuy/lending/rewards |
| 13 | Frontend API key hardcoded | 6.5 | `1770d5d9-bcea-4d28-ad21-6cbd5be018a8` in store/index.ts |

### Nabu Gateway Issues
- CORS: `Access-Control-Allow-Origin: *` on auth endpoints
- Auth headers revealed in error messages (leaks internal auth mechanism names)

## Cloudflare Block Pattern

Blockchain.com uses Cloudflare's JS challenge (not just firewall) on:
- `www.blockchain.com` — full site
- `login.blockchain.com` — auth portal
- `blockchain.info` — legacy wallet (some paths pass through)
- `exchange.blockchain.com` — exchange

**Passes through CF:**
- `api.blockchain.info` — most paths pass through
- `blockchain.info/Resources/wallet-options.json` — static file
- `blockchain.info/uuid-generator?format=json` — lightweight endpoint
- `blockchain.info/multiaddr?format=json` — public data

**CF-blocked (403/JS challenge):**
- `blockchain.info/wallet/*`
- `blockchain.info/wallet-options.json`
- `blockchain.info/pin-store`
- All `*.dev.blockchain.info` and `*.staging.blockchain.info`

## Dev/Staging Environment (Directly from wallet-options)

### Dev
```
root:       https://explorer.dev.blockchain.info
comRoot:    https://dev.blockchain.com
walletApp:  https://login-dev.blockchain.com
ws:         wss://ws.dev.blockchain.info/inv
api:        https://api.dev.blockchain.info
helper:     https://wallet-helper.dev.blockchain.info
```

### Staging
```
root:       https://explorer.staging.blockchain.info
comRoot:    https://staging.blockchain.com
walletApp:  https://wallet.staging.blockchain.com
ws:         wss://ws.staging.blockchain.info/inv
api:        https://api.staging.blockchain.info
helper:     https://wallet-helper.staging.blockchain.info
```

### Testnet
```
root:       https://testnet.blockchain.info
ws:         wss://ws.testnet.blockchain.info/inv
api:        https://api-testnet.blockchain.info
helper:     https://wallet-testnet3-helper.blockchain.info
stellar:    https://horizon-testnet.stellar.org
```

All dev/staging domains return 403 (CF block). Testnet DNS does not resolve.

## Blockchain Node RPC Endpoints (from network-config)

| Chain | RPC URL |
|-------|---------|
| Ethereum | `https://api.blockchain.info/eth/nodes/rpc` |
| Polygon | `https://api.blockchain.info/matic-bor/nodes/rpc` |
| BNB Chain | `https://api.blockchain.info/bnb/nodes/rpc` |
| Avalanche | `https://api.blockchain.info/avax/nodes/rpc/ext/bc/C/rpc` |
| Arbitrum | `https://api.blockchain.info/enodes/rpc/arb` |
| Optimism | `https://api.blockchain.info/enodes/rpc/op` |
| Base | `https://api.blockchain.info/base/nodes/rpc/` |
| Stellar | `https://api.blockchain.info/stellar` |
| Solana | `https://ssc-dao.genesysgo.net/` (third-party, not Blockchain.com infra) |

## Hot Wallet ETH Addresses (Production)

| Service | Address |
|---------|---------|
| Swap | `0xC88F7666330b4b511358b7742dC2a3234710e7B1` |
| Exchange | `0x9AA65464b4cFbe3Dc2BDB3dF412AeE2B3De86687` |
| Simple Buy | `0x23f4569002a5A07f0Ecf688142eEB6bcD883eeF8` |
| Lending | `0x67f889e6C1CE3E817705E00D528eB7F8be492B9E` |
| Rewards | `0xA00E2A7652248AbEb209398227DAE413E9479e52` |

## Attack Vectors Not Fully Tested

Due to Cloudflare blocking the browser, these vectors remain untested:
1. **Wallet auth bypass** — MFA-skip, session-guid enumeration (requires passing CF)
2. **Authenticated Nabu Gateway** — KYC escalation, IDOR in wallet ops (needs session)
3. **Exchange API CSRF** — Order manipulation (needs exchange account access)
4. **WebSocket** — Message injection, session hijacking (requires connection token)
5. **SSRF via wallet-helper** — Could not probe because all paths return 404

## Credentials (for retesting)

- Email: `h0d4r1@bugcrowdninja.com`
- Password: stored in memory (ask user)
- Bugcrowd username: `h0d4r1`
