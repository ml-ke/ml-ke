---
title: API Bug Bounty Methodology
name: api-bug-bounty-methodology
description: Systematic approach to finding vulnerabilities in web APIs for bug bounty programs. Covers recon, IDOR, mass assignment, SSRF, auth bypass, and business logic testing. Uses Notion API as the primary practice target.
---

# API Bug Bounty Methodology

Systematic approach to finding vulnerabilities in web APIs for bug bounty programs. This skill covers the full workflow: recon, endpoint mapping, vulnerability testing, and reporting.

## Core Principles (from Bug Bounty Bootcamp Ch22 + Hacking APIs Ch0)

1. **Authorization first** — Never test without explicit permission (HackerOne program scope)
2. **Map before you dig** — Full API surface enumeration before deep testing
3. **Two-account testing** — Create two accounts to test for IDORs
4. **Document everything** — Every request, response, and finding
5. **Escalate impact** — Don't stop at the first finding; chain vulnerabilities

## Step 0: Program Intelligence (Pre-Recon)

Before any endpoint testing, fully evaluate the program's scope, rules, and tech stack. This phase answers: *"Is this program a good fit for my tools, and what do I need the user to set up?"*

### 0A — HackerOne Program Page Scrape

Browse `https://hackerone.com/{program}` with the browser tool. Navigate to:

1. **Scope** tab → extract every asset:
   - Asset name, type (Domain / API / iOS / Android), coverage (In scope / Out of scope), max severity, bounty eligibility
   - Note resolved reports per asset — 0 reports = untapped opportunity
   - Download the Burp Suite config file via the download link
2. **Program guidelines** tab → extract:
   - **Rules**: report format requirements, disclosure policy, test plan (including UAT URLs and credential notes), production signup instructions
   - **Non-qualifying bugs**: DoS, SSL/TLS, email best practices, rate limiting, open redirect, self-XSS, CORS, exposed API keys, UAT-only info disclosure
   - **Disqualifiers**: brute forcing on production, data destruction, social engineering
   - **Test plan**: UAT environment URL, credential availability policy
3. **Hacktivity** tab → understand what types of bugs are being found and how active the program is
4. Note **key constraints**:
   - `X-HackerOne-Research: [username]` header requirement for production testing
   - @wearehackerone.com email requirement for signup
   - Credential availability for new researchers

### 0B — Asset Reconnaissance

For each production asset (www, api, subdomains):

```bash
# DNS resolution — identify CDN/proxy layer
dig +short A $domain
dig +short CNAME $domain

# HTTP headers — fingerprint the stack
curl -sI https://$domain/ --max-time 10 2>&1 | head -20
# Key signals: server:, x-powered-by:, set-cookie:, content-security-policy:,
# x-wf-* (Webflow), cf-ray (Cloudflare), x-amz-* (AWS)

# Subdomain discovery — check common patterns
for sub in blog help api docs status admin; do
  dig +short A $sub.syfe.com 2>/dev/null && echo "$sub.syfe.com -> found"
done

# MX / TXT records — identify GSuite, SendGrid, Zendesk, etc.
dig +short MX syfe.com
dig +short TXT syfe.com

# S3 bucket recon — check for public access and delete markers
for bucket_uri in assets.syfe.com files.syfe.com static.syfe.com cdn.syfe.com uploads.syfe.com; do
  response=$(curl -sI "https://$bucket_uri/" --max-time 8 2>/dev/null | head -10)
  echo "$bucket_uri → $(echo "$response" | grep -iE 'HTTP/|x-amz|server' | head -2)"
done
# Key S3 signals: x-amz-error-code, x-amz-delete-marker (true = bucket was used, now drained),
# x-amz-bucket-region, server: AmazonS3, x-amz-request-id
```

### 0C — Tech Stack Fingerprinting

From headers, DNS, and CSP policy (leaked from Content-Security-Policy header):

| Signal | Technology |
|--------|-----------|
| `cf-ray` header | Cloudflare |
| `x-wf-region`, `surrogate-key: webflow.*` | Webflow |
| `x-amz-*`, `s3.amazonaws.com` in CSP | AWS S3 |
| `_cfuvid` cookie, `webflow.io` domain | Webflow hosting |
| `csp:` frame-ancestors, connect-src | Leaks all 3rd-party integrations |
| `server: cloudflare` | Cloudflare proxy |

Extract the **Content-Security-Policy** header and parse `connect-src`, `script-src`, and `frame-src` to identify every third-party integration (analytics, chat, SSO, CDN, monitoring). This is often the quickest way to map the full tech stack.

### 0D — Capability Assessment

Map the identified tech stack to attack surfaces and assess tool fit:

| Tech Signal | Attack Surface | Agent Capability |
|------------|---------------|-----------------|
| API endpoints (api.*) | IDOR/BOLA, SSRF, injection, auth bypass | **Strong** — curl, browser, execute_code |
| OAuth (Google/Apple Sign-in) | OAuth misconfig, CSRF, token theft | **Strong** — intercept + replay |
| Webflow / CMS | CMS-specific vulns, exposed endpoints | **Moderate** — browser probing |
| JWT auth | Weak signing, alg confusion, missing validation | **Strong** — terminal decode + test |
| GraphQL (probe) | Introspection, batching, depth attacks | **Strong** — curl + queries |
| File upload / URL import | SSRF via DNS bypass | **Strong** — our speciality |
| Business logic (fintech) | Multi-step workflow abuse | **Strong** — browser + scriptable |
| iOS/Android app | APK/IPA decompile, runtime hooking | **Weak** — needs user to run tools |

### 0E — Setup Requirements Summary

After analysis, tell the user:
1. What they need to register (production account via @wearehackerone.com)
2. What credentials they need (UAT — if available to new researchers)
3. Your HackerOne username for the X-HackerOne-Research header
4. Optional: whether mobile app APK/IPA is worth targeting
5. Recommended first attack vector based on tech stack strengths

### 0F — Next.js App Frontend Recon & JS Bundle Analysis

When a program uses a SPA, the frontend JavaScript is the richest source of API endpoints, auth mechanisms, and internal architecture. Look for **Next.js** specifically — most production apps use it.

**Signals — Next.js in HTML source:**
```html
<script src="https://www.target.com/_next/static/chunks/pages/login-*.js"></script>
<script src="https://www.target.com/_next/static/chunks/framework-*.js"></script>
<script src="https://www.target.com/_next/static/chunks/main-*.js"></script>
```

**Step-by-step JS bundle extraction:**

1. **Enumerate all script tags** from the page source (login, home, dashboard pages)
2. **Download the `_app.js` chunk** — it's usually the largest and contains the most API calls:
   ```bash
   curl -s "https://www.target.com/_next/static/chunks/pages/_app-*.js" > /tmp/app.js
   ```
3. **Extract absolute API URLs** (direct references to api.* domains):
   ```bash
   grep -oP '["'"'"']https?://[^"'"'"']*(target\.com|amazonaws)[^"'"'"']*["'"'"']' /tmp/app.js | sort -u
   ```
4. **Extract relative API paths** (the SPA's internal route paths):
   ```bash
   grep -oP '["'"'"']/[a-zA-Z][a-zA-Z0-9/_\-]{3,80}["'"'"']' /tmp/app.js | \
     grep -viE '(\.(js|css|png|jpg|svg|woff|ico|json|map)|/_next|//|google|facebook|twitter)' | \
     sort -u
   ```
5. **Extract auth/JWT storage patterns** — search for `token`, `access_token`, `localStorage`, `bearer`:
   ```bash
   grep -oP '.{0,50}(token|jwt|bearer|authorization|access_token|refresh_token).{0,50}' /tmp/app.js
   ```

**What to look for in the extracted paths:**
- `/auth/*` routes → auth flow (login, signup, MFA, passkeys, TOTP, password reset)
- `/account/*` or `/user/*` → user/account operations (IDOR targets)
- `/portfolio/*`, `/trade/*`, `/invest/*` → financial operations (business logic vulns)
- `/promo/*` → promo/referral (abuse potential)
- `/advisor/*`, `/admin/*` → privileged endpoints
- `/myinfo/*` → government data API integrations (Singapore MyInfo)
- `/api/*` → Next.js API proxy routes

**Download login page chunk specifically** — it reveals the auth flow:
```bash
curl -s "https://www.target.com/_next/static/chunks/pages/login-*.js" > /tmp/login.js
grep -oP '["'"'"']/[a-zA-Z][a-zA-Z0-9/_\-]{3,50}["'"'"']' /tmp/login.js | sort -u
```

**Config leak from Webflow pages:**
Webflow-hosted pages often embed configuration variables in `<script>` tags visible in the raw HTML. These are present even on error/404 pages:
```javascript
var DOMAIN = 'https://api.target.com';
var BASEAPI_URL = DOMAIN;
var GTM_ID = 'GTM-XXXXXXX';
var DATADOG_APPLICATION_ID = '...';
var DATADOG_CLIENT_TOKEN='***';
var ENVIRONMENT_NAME = 'staging'; // May differ from actual env
```
Extract these from the raw HTML source of any page (login, 404, etc.) — grep for `var ` declarations or configuration objects.

**SPA framework-specific patterns:**
- **Next.js**: `_next/static/chunks/*.js`, `_next/data/{buildId}/page.json` → SSG data endpoints
- **React SPA**: Large JS bundles with webpack chunk hashes
- The `_next/data/*.json` endpoints can reveal server-rendered page props:
  ```bash
  curl -s "https://www.target.com/_next/data/{buildId}/login.json"
  ```

**`__NEXT_DATA__` SSR data extraction** — Next.js embeds server-rendered state in a `<script id="__NEXT_DATA__">` tag. This often contains the entire initial Redux/app state including:
- Auth token state (null for unauthenticated, but pattern confirmed)
- Bank lists, portfolio metadata, locale config
- User info, account settings, and client-side routing state
- Social login configuration (Google/Apple SSO)

```bash
# Extract from any server-rendered page
curl -s "https://www.target.com/login" | \
  grep -oP '"props":\{"pageProps":\{[^}]+' | head -1

# Or extract the full JSON (may be very large)
curl -s "https://www.target.com/login" | \
  grep -oP '<script id="__NEXT_DATA__"[^>]*>.*?</script>' | \
  sed 's/<[^>]*>//g' > /tmp/next-data.json
```

**Auth storage patterns — token format identification:**
When extracting auth code from JS bundles, identify the token type:
- **JWT**: Three base64 segments separated by dots — decode to inspect claims, algorithm, expiry
- **UUID**: 8-4-4-4-12 hex format (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) — opaque session identifier
- **Opaque random string**: Variable length alphanumeric — no introspection possible

```bash
# JWT test
echo "eyJxxx.eyJxxx.xxx" | cut -d. -f2 | base64 -d 2>/dev/null
# UUID test
echo "bf92d3db-1896-46cc-9ff4-f7b82b84202e" | grep -qP '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' && echo "UUID format"
```

**Token storage patterns found in the wild:**
```javascript
// From Redux reducer in JS bundle:
localStorage.getItem('token')          // → UUID string
localStorage.getItem('validTill')      // → ISO date string

// Axios/fetch config:
Authorization: "Bearer ".concat(token) // Bearer prefix
Accept: "application/vnd.syfe.v3+json" // Custom content negotiation
```

Check for **APP_CLIENT_ID** in localStorage — some apps require a client ID alongside the auth token:

### 0G — API Proxy Pattern Recognition

Many Next.js apps proxy API calls through the frontend domain (www) to the backend API (api.*), creating a two-layer architecture.

**How to detect:**
```bash
# Direct API domain returns 403 (Cloudflare WAF)
curl -s -o /dev/null -w "%{http_code}" https://api.target.com/auth/login
# → 403

# But the www proxy route returns 200 (Next.js handler HTML)
curl -s -o /dev/null -w "%{http_code}" https://www.target.com/api/auth/login
# → 200
```

**What this means for exploitation:**
- The proxy may forward headers differently — try adding `X-HackerOne-Research`, `Origin`, `Referer`, `X-Forwarded-For`
- Direct API access is locked (Cloudflare WAF), but the www proxy may have different or weaker rules
- The proxy may strip or add authentication headers — test both routes

### 0I — Program Selection & Comparison (Cross-Program Assessment)

When the user says "find me a better program" or asks you to evaluate multiple targets, use this rubric to produce an apples-to-apples comparison that justifies a recommendation.

**The Decision Matrix:**
```
Program A | Program B | Program C
─────────────────────────────────
Bounties:    $X–$Y     | $A–$B     | $P–$Q
Scope:       9 assets  | 20 assets  | 5 assets
Code access: Closed    | Open       | Open
Free acct:   ❌ KYC    | ✅ Yes     | ✅ Yes
Response:    2d triage | 6h triage  | 11h triage
Resolved:    21 (2.4%) | 1,000+     | 500+
Barrier:     HIGH      | LOW        | MEDIUM
```

**Barrier-to-Entry Scale:**
- **LOW** — Free account signup, no verification, no payment needed (GitLab.com, WordPress.com)
- **MEDIUM** — Free account but needs KYC or email verification (most SaaS)
- **HIGH** — Needs funded account, business subscription, or mobile app extraction (fintech, ecommerce)
- **BLOCKED** — Private program, needs invitation, or target requires physical access

**When to recommend moving on:**\n- User has expressed doubt about a program's opportunity (trust your user's intuition)\n- KYC/funding barrier + no unique attack angle (Syfe needed funded portfolio to test IDOR)\n- Program has suspended bounties (Nextcloud)\n- Cloudflare WAF blocks all CLI access AND browser proxy is unreliable\n- Another program scores higher on every dimension\n- Business verification required for production access — pivot instead of asking user to submit sensitive info\n\n**The Rapyd→Zendesk pivot (worked example):**\n- Rapyd: 4/4 scope, $185 avg, BLOCKED by business verification → abandoned after 2 days work\n- Zendesk: 2/4 scope, P1 up to $20K, self-service signup, new program (Dec 2025)\n- Decision matrix that justified the switch:\n  - Newer program = less competition (Zendesk Dec 2025 > Rapyd Nov 2022 > Just Eat Apr 2019 > Okta Nov 2016)\n  - AI Agent scope = unique for Anthropic research (only Zendesk had this on the list)\n  - Self-service signup with @bugcrowdninja.com email (no KYC, no business info)\n  - P1 payout $20K > $500 ⇒ better ROI per finding

**Bounty-targets-data resource for program discovery:**
- `github.com/arkadiyt/bounty-targets-data` — Hourly-updated dumps of HackerOne/Bugcrowd scopes
- `github.com/jakejarvis/bounty-domains` — Domain lists from bounty programs
- Use these to discover new programs or verify scope without crawling HackerOne's UI

### 0J — Cloudflare-Protected Target Reconnaissance

When the main web/app surface (`www.*`, `login.*`) is behind Cloudflare JS challenge, you cannot use the browser or regular curl to interact with the target. However, internal API gateways and static resources may be accessible through other paths. This is especially common with crypto/fintech companies where the consumer-facing site is CF-protected but internal APIs are exposed.

#### 0J.1 API Gateway Discovery

Crypto companies often separate their consumer site (Cloudflare) from their API gateway (Cloudflare pass-through). The API gateway may have different, weaker rules:

```bash
# Probe subdomains systematically — look for 400/405 responses, NOT 404
for sub in api login wallet wallet-api exchange stx explorer; do
  code=$(curl -sI -m 5 "https://$sub.blockchain.com/" 2>/dev/null | head -1)
  ct=$(curl -sI -m 5 "https://$sub.blockchain.com/" 2>/dev/null | grep -i content-type)
  echo "$sub.blockchain.com → $code [$ct]"
done
```

**Response interpretation:**
| Code | Content-Type | Meaning |
|------|-------------|---------|
| 403 (html) | `text/html` — usually Cloudflare block page | WAF or region block |
| 302 (html) | HTML redirect | Might redirect to auth or marketing |
| 400 | `application/json` | **Endpoint exists!** Needs proper params/body |
| 405 | empty body | Endpoint exists, wrong HTTP method |
| 404 | `text/html` | Route doesn't exist on that host |

**Key insight: 400 ≠ 404.** A 400 response with an error body means the endpoint IS there — it just doesn't like your request format. This is the strongest signal that you've found a real, accessible backend.

#### 0J.2 Static Resource Discovery

Config files, JS bundles, and static assets are often served from paths outside the CF-protected zone:

```bash
# Common config endpoints (Blockchain.com pattern)
curl -sL "https://blockchain.info/Resources/wallet-options.json"
# Returns full production config with partner API keys, hot wallet addresses, domains
# This may not trigger Cloudflare JS challenge!

# Other static paths to try
# /Resources/
# /static/
# /config/
# /assets/config.json
# /version.json
# /build-info.json
# /runtime-config.js
```

**Wallet-options pattern (crypto-specific):** Crypto companies (Blockchain.com, Coinbase, exchanges) serve a client-side configuration JSON that includes:
- Partner API keys (SFOX, ShapeShift, Plaid, MoonPay, etc.)
- Hot wallet addresses for swap/exchange/lending/rewards
- Domain configuration for all internal services (api, wallet, ws, helper)
- Dev/staging domain names
- Partner income tracking keys (SiftScience, Plaid env, etc.)

These configs are technically needed by the frontend JS, so they're intentionally public — but **finding them in a public GitHub repo** (github.com/$org) with full git history is a different risk: git history preserves keys forever, even if the live endpoint is rotated.

#### 0J.3 Source Code Intel From Public Repos

Clone the program's GitHub repos to find ALL API endpoints, auth mechanisms, and internal architecture — then cross-reference against live targets:

```bash
# Clone relevant repos
for repo in wallet-options service-my-wallet-v3 blockchain-wallet-v4-frontend network-config; do
  git clone --depth 1 "https://github.com/$org/$repo.git"
done

# Key intel to extract:
# 1. Config files → domain names, API keys, partner endpoints
grep -r "api\." wallet-options/ --include='*.json' --include='*.js' | head -20

# 2. Route definitions → all API endpoints
grep -rn "router\.\|app\.\(get\|post\|put\|delete\|all\|use\)" service-my-wallet-v3/src/ --include='*.js'

# 3. Auth mechanisms → headers, tokens, signatures
grep -rn "sessionToken\|Authorization\|Bearer\|api_code\|apiKey" blockchain-wallet-v4*/src/ --include='*.ts' | head -30

# 4. Internal network architecture → RPC endpoints
cat network-config/config.json | jq '.networks[].nodeUrls'

# 5. Dev/staging domain names → alternate attack surface
grep -r "dev\." wallet-options/dev/ --include='*.json' | head -20
```

**Cross-reference strategy:** The config repo tells you:
- What APIs exist (`domains: { api:, wallet:, ws: }`)
- What third-party services are integrated (partner API keys)
- What internal blockchain nodes are proxied (nodeUrls)
- What dev/staging/testnet environments exist
- What hot wallet addresses are in use

Probe each discovered API domain independently — they may have different Cloudflare rules.

#### 0J.4 The Nabu Gateway / API Gateway Pattern

Modern crypto companies often build a second-generation API gateway (often named differently: Nabu Gateway, Apollo Federation, BFF layer) that runs alongside or replaces the legacy wallet API:

**Signals you've found an API gateway:**
```
curl -sI "https://api.target.com/nabu-gateway/countries"
# → x-blockchain-cp-b: nabu-gateway  ← Custom backend identifier!
# → HTTP 200 with JSON body            ← Unauthenticated endpoint!
```

**How to find gateway paths (from frontend source):**
```typescript
// Search for API base URL patterns
grep -rn "'/nabu-gateway" --include='*.ts' blockchain-wallet-v4-frontend/
// → '/nabu-gateway/dex', '/nabu-gateway/countries', '/nabu-gateway/auth'
// → '/nabu-gateway/kyc/tiers', '/nabu-gateway/kyc/extra-questions'
// → '/nabu-gateway/onboarding/quiz', '/nabu-gateway/payments/beneficiaries'
```

**Unauthenticated gateway endpoints to test:**
```bash
# Countries/locale data — almost always public
GET /nabu-gateway/countries?scope=SIGNUP
GET /nabu-gateway/countries/US/states

# Price/rate data — often public
GET /nabu-gateway/price
GET /nabu-gateway/price/eth-usd
GET /nabu-gateway/currencies

# Auth endpoints — return 400 (not 401) without credentials = exists!
POST /nabu-gateway/auth
POST /nabu-gateway/users
POST /nabu-gateway/signup
POST /nabu-gateway/registrations
```

**Key auth endpoint behavior:** API gateway auth endpoints commonly:
- Return **400 (MISSING_PARAM)** without credentials, NOT 401
- List expected headers in the error message (`"Expected header(s): user-agent, x-app-version, x-client-type, x-device-id"`)
- Accept `Access-Control-Allow-Origin: *` even on protected endpoints
- Use custom auth headers (`x-wallet-guid`, `x-session-id`, `x-signature`, `x-hmac-signature`, `blockchain-auth`)

The error messages themselves are useful — they enumerate the entire header whitelist, revealing exactly what auth mechanisms are supported.

#### 0J.5 Internal Blockchain Node RPC Proxies (Crypto-Specific)

Crypto companies often proxy blockchain node RPCs through their API domain. These proxies may accept JSON-RPC calls without authentication:

```bash
# From network-config, find all RPC endpoints
# Common paths:
/eth/nodes/rpc
/matic-bor/nodes/rpc
/bnb/nodes/rpc
/avax/nodes/rpc/ext/bc/C/rpc
/enodes/rpc/arb
/enodes/rpc/op
/base/nodes/rpc/
/stellar

# Test standard methods
curl -sL "https://api.target.com/eth/nodes/rpc" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}'
# Returns: Geth/v1.16.7-stable/linux-amd64/go1.25.1

# Test admin methods (usually blocked but worth checking)
curl -sL "..." -d '{"jsonrpc":"2.0","method":"admin_nodeInfo","params":[],"id":1}'
# → method does not exist/is not available (blocked — good security)

# Test debug methods
curl -sL "..." -d '{"jsonrpc":"2.0","method":"debug_traceTransaction","params":[...],"id":1}'
# → method does not exist/is not available (blocked)

# Test coinbase (leaks node's ETH address)
curl -sL "..." -d '{"jsonrpc":"2.0","method":"eth_coinbase","params":[],"id":1}'
# → method does not exist/is not available (blocked)
```

**What RPC response headers reveal:**
```
x-blockchain-cp-b: coin-node-eth-rpc    ← Internal backend name
x-blockchain-cp-f: xj66 0.004 - <hash>  ← Internal load balancer
x-blockchain-server: BlockchainFE/1.0    ← Framework version
x-blockchain-language: en               ← Locale
x-original-host: api.blockchain.info     ← Original request target
```

These headers are useful for fingerprinting and for SSRF testing (you can see what internal services are named).

#### 0J.6 Summary: CF-Protected Target Workflow

1. **Start with GitHub repos** — extract all API domains, partner keys, routes, auth mechanisms
2. **Probe each subdomain independently** — some may not have CF protection
3. **Test static resource paths** — find config files, JS bundles, assets
4. **Look for API gateways** — check paths like `/nabu-gateway/`, `/api/v3/`, `/gateway/`
5. **Use 400 (not 401) as an endpoint-exists signal** — error messages reveal expected parameters
6. **Test blockchain/WebSocket endpoints** — these often bypass the CF layer entirely
7. **Document the full attack surface** — even if CF blocks the main app, you've mapped the ecosystem

**Reference**: `references/blockchain-program-intelligence.md` — complete worked example of this workflow applied to Blockchain.com (Bugcrowd), including source repo map, API endpoint catalog, partner key exposure, and Nabu Gateway discovery.
**Reference**: `references/chime-program-intelligence.md` — Chime Bugcrowd program scope, targets, access status (all Cloudflare-gated), QA endpoint discovery, and Next.js stack fingerprinting.
**Reference**: `references/fireblocks-program-intelligence.md` — Fireblocks Web (Bugcrowd) scope, targets, auth mechanism (JWT-per-request signing), SDK structure (66 API modules), and sandbox signup flow. Work in progress — sandbox credentials not yet obtained.

### 0K — Region/Multi-Environment Exploitation

**Why region-specific subdomains are worth probing:**
- Different Cloudflare WAF configuration per region
- Region-specific deployments may be less hardened
- Stale endpoints may exist in one region but not another

**Discovery:**
Look for region-specific API URLs extracted from the JS bundle's absolute API references. Common patterns:
```
api-au.target.com | api-hk.target.com | api-eu.target.com | api-us.target.com
```

**Verification:**
```bash
for region in "" "-au" "-hk"; do
  for path in auth/login auth/signup promo/validate/TEST; do
    code=$(curl -s -o /dev/null -w "%{http_code}" \
      "https://api${region}.target.com/$path" --max-time 8)
    ct=$(curl -s -o /dev/null -w "%{content_type}" \
      "https://api${region}.target.com/$path" --max-time 8)
    echo "api${region}.target.com/$path → $code ($ct)"
  done
done
```

**Response interpretation:**
- **403 (text/html)** = Cloudflare challenge/block (locked down)
- **405 (application/json)** = Endpoint exists, wrong method (real backend responding)
- **404 (application/json)** = Endpoint not found at that path
- **400/401 (application/json)** = Endpoint exists, needs valid credentials
- **200 (application/json)** = Unauthenticated endpoint! Investigate immediately

**Reference**: `references/syfe-program-intelligence.md` — full worked example of this workflow applied to Syfe (syfe_bbp), including the decision to recommend GitLab instead (Section 0I criteria applied).\n**Reference**: `references/gitlab-program-intelligence.md` — GitLab BBP scope, bounties, codebase layout, and SSRF attack vector mapping.\n**Reference**: `references/zendesk-program-intelligence.md` — Zendesk Bug Bounty scope, payouts, target endpoints, AI Agent attack vectors (P1 up to $20K), and initial recon data.

## Step 1: Reconnaissance

### API Documentation Harvesting
```bash
# Check for documentation indexes
curl -s https://developers.target.com/llms.txt

# Common docs locations
curl -s https://target.com/api/docs
curl -s https://developers.target.com
curl -s https://target.com/.well-known/security.txt
```

### Endpoint Discovery from Docs
- Crawl API reference pages for all HTTP methods and paths
- Note authentication requirements per endpoint
- Map relationships between endpoints (e.g., pages → blocks → comments)

### Live Endpoint Discovery via Non-200 Responses (Answer Bot Pattern)

HTTP 400 (Bad Request) is a strong signal the endpoint EXISTS — it just needs the right parameters. HTTP 404 means the path doesn't exist. Use this distinction:
- **HTTP 400 with empty/generic body** — Endpoint exists but rejecting your param format. Try different body structures
- **HTTP 404** — Path doesn't exist. Move on
- **HTTP 403** — Endpoint exists but not authorized. Try with proper auth
- **HTTP 405** — Endpoint exists, wrong method

**Response header fingerprinting**: Custom headers identify the backend service:
```bash
curl -sv https://target.com/api/v2/endpoint -X POST -H "Content-Type: application/json" -d '{}' 2>&1 | grep -iP 'x-zendesk|x-envoy|via|service'
# → zendesk-service: answer-bot-service (confirms a separate microservice handles this)
# → x-envoy-upstream-service-time: 21ms
```

**JWT extraction from non-error responses**: Decode and inspect tokens:
```python
parts = jwt.split(".")
padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
decoded = json.loads(base64.b64decode(padded))
# Check: account_id, user_id, ticket_id, article_ids, exp
```

**Worked example — Answer Bot API**: All param formats returned 400 with empty body. Headers showed `zendesk-service: answer-bot-service`. After researching docs, the correct format was `{"enquiry":"text","reference":"id","locale":"en-us"}`. Response returned JWTs containing `account_id`, `brand_id`, `deflection_id`.

### Asset Enumeration (from Hacking APIs Ch6)
```bash
# Subdomain enumeration
curl -s https://api.target.com/v1/health
curl -s https://api.target.com/v1/users/me
```

## Step 2: Authentication Testing (Hacking APIs Ch8)

### Custom JWT Per-Request Signing (Fireblocks Pattern)

Some APIs (e.g., Fireblocks) use a **per-request signed JWT** instead of a static bearer token. Every HTTP request carries a freshly-signed JWT in the `Authorization: Bearer` header. This means:
- No refresh tokens, no expiry beyond 55 seconds
- The JWT covers both the request URI AND body hash (integrity protection)
- Python's `PyJWT.encode()` and Node's `jsonwebtoken.sign()` produce DIFFERENT byte-level RS256 signatures for the same payload — Node.js works, Python may not

**JWT payload structure (Fireblocks example):**
```json
{
  "uri": "/v1/vault/accounts",
  "nonce": "uuid-v4",
  "iat": 1780349000,
  "exp": 1780349055,
  "sub": "your-api-key",
  "bodyHash": "sha256-of-body"
}
```

### JWT Nonce Tracking — The Real Replay Protection

When testing JWT-based APIs, the `nonce` field is the actual replay protection — not the idempotency key, not the expiry time alone. Always test nonce tracking before claiming replay vulnerabilities:

```javascript
// Test: send same JWT twice with different idempotency keys
const capturedJwt = /* observe from first request */;

// Request 1 (works)
POST /endpoint
Authorization: Bearer <capturedJwt>
Idempotency-Key: AAA → 200 OK

// Request 2 (same JWT, different idempotency key)
POST /endpoint
Authorization: Bearer <capturedJwt>   // SAME JWT!
Idempotency-Key: BBB → 401 "This nonce was already used"
```

If the server returns an error like `"This nonce was already used in a previous request"` (code -13 in Fireblocks), the nonce IS properly tracked and replay is blocked. In this case, the idempotency key not being in the JWT is NOT a vulnerability — it's a design inconsistency with no security impact.

**Severity trap:** Do NOT claim "Authentication Bypass (P1)" for idempotency-key-not-signed when the nonce prevents replay. That's overclaiming. The user will (and should) challenge it.

**Headers:**
```
X-API-Key: {api_key}
Authorization: Bearer {signed-jwt}
Content-Type: application/json
Content-Length: {explicit_byte_count}  ← REQUIRED for POST, Node.js does not auto-set this
```

**Critical pitfalls:**

1. **bodyHash for POST must be SHA256 of JSON.stringify(body)** not SHA256 of the raw object. If you pass a JS object to `crypto.createHash().update()`, old Node.js silently hashes `[object Object]`, new Node.js v22+ throws TypeError. Always use `JSON.stringify(body)` first.

2. **Content-Length header is mandatory for POST** — Node.js `https.request()` does NOT set Content-Length when the body is written via `req.write(str)`. Without it, the server returns `401 code -7` (Error getting User certificate / rate limited). Always set:
   ```javascript
   'Content-Length': Buffer.byteLength(JSON.stringify(body))
   ```

3. **Python JWT vs Node.js JWT differences** — The same payload + private key produces different RS256 signatures between Python's `jwt.encode()` (PyJWT) and Node's `jsonwebtoken.sign()`. The Fireblocks server accepts Node.js signatures but rejects Python's. When in doubt, use Node.js.

4. **Rate limiting mimics auth errors** — Rapid firing of requests (10+ in 30s) causes the same `401 code -7` error as actual auth failures. If GET also starts failing after a burst of POSTs, wait 10-30s for the rate limit to clear. Test GET after resting — if GET recovers, it was rate limiting, not a permanent auth failure.

5. **URI must be pathname + search only** — Use `new URL(fullUrl).pathname + new URL(fullUrl).search` to extract the URI. Signing the full URL (with host) causes `code -4` (wrong URL).

**Pattern recognition:** When you see `401` with a JSON error body and numeric error code (not an HTML page), the API is alive and processing your requests — it just doesn't like your parameters. This is better than 404 (no such endpoint) or an HTML page (Cloudflare).
```bash
# Test with missing token
curl -s -H "Content-Type: application/json" https://api.target.com/v1/users/me

# Test with invalid token
curl -s -H "Authorization: Bearer invalid" https://api.target.com/v1/users/me

# Test with expired/revoked token
curl -s -H "Authorization: Bearer ntn_expired..." https://api.target.com/v1/users/me
```

### Token Introspection
- What does `/v1/users/me` return? User ID? Email? Roles?
- Can I enumerate users via sequential IDs?
- Can I access endpoints with a token from a different workspace?

### When CLI Tools Fail (Bash Quoting Issues)

Complex API calls with Bearer tokens, nested JSON, and special characters break in bash due to quoting issues. Common failure modes:
- Token interpolation via `$(cat file)` produces syntax errors or truncation
- Nested JSON quotes conflict with bash quoting
- Multi-line token strings break command substitution

**Fix**: Write standalone Python scripts to disk and execute them.

```python
#!/usr/bin/env python3
import json, urllib.request
token = open('/home/pro-g/.token-file').read().strip()
base = "https://api.target.com/v1"

def api(path, method="GET", data=None):
    url = f"{base}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    if data:
        req.data = json.dumps(data).encode()
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:500]}
```

Then: `python3 /tmp/test.py`

Advantages over bash+curl:
- No quoting issues with JSON payloads
- Native error handling (non-2xx doesn't crash)
- Easy iteration, comparison, conditional branching
- Token never appears in `ps aux` or command logs
- Use `write_file` to create the script, then `terminal` to execute

## Step 3: IDOR Testing (Bug Bounty Bootcamp Ch10, Real-World BH Ch16)

### Core Definition (from PortSwigger + Intigriti)
IDOR occurs when an attacker can access or modify a reference to an object **that belongs to a different entity** (user/workspace/tenant) without authorization. Sequential IDs alone do NOT constitute IDOR — the boundary crossing is the vulnerability.

### Two-Account Methodology (from 3 Real-World BH Ch16 case studies)
1. Create Account A in scope A → capture resource IDs
2. Create Account B in scope B → authenticate as Account B
3. Try Account A's resource IDs while authenticated as Account B
4. If Account B can access Account A's resources → IDOR confirmed
5. If you only tested within one scope → NOT ready for submission

**Case studies from the book:**
- **Binary.com ($300)**: Used pin from account A in account B's session → withdrawal access
- **Moneybird ($100)**: Account B created app with account A's administration_id → permission bypass
- **Twitter Mopub ($5,040)**: Leaked org_id via public crash URLs → API key + build_secret theft → account takeover

### Before submitting any IDOR finding, run `pre-submission-verification` skill Gate C1 (Access Control).

### Common IDOR Patterns
```bash
# Page IDOR
curl -s -H "Authorization: Bearer $TOKEN" "https://api.notion.com/v1/pages/$PAGE_ID"

# Block children IDOR
curl -s -H "Authorization: Bearer $TOKEN" "https://api.notion.com/v1/blocks/$BLOCK_ID/children"

# User enumeration
curl -s -H "Authorization: Bearer $TOKEN" "https://api.notion.com/v1/users/$USER_ID"
```

### UUID Manipulation
- UUIDv4 is random — can't enumerate sequentially
- But can you find UUIDs through other means? (search, shared pages, comments)
- Are there any endpoints that leak UUIDs?

## Step 4: Mass Assignment (Hacking APIs Ch11)

### Testing
```bash
# Try adding extra fields to POST/PATCH requests
curl -X PATCH -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Notion-Version: 2026-03-11" \
  -d '{"properties":{"title":[{"type":"title","title":[{"type":"text","text":{"content":"test"}}]}]},"permissions":{"type":"workspace"}}' \
  "https://api.notion.com/v1/pages/$PAGE_ID"
```

### Fields to test
- `permissions`, `access`, `visibility` — can you escalate privileges?
- `role`, `type` — can you change user roles?
- `parent` — can you move pages to other workspaces?
- Extra properties — can you inject unexpected data?

## Step 5: SSRF Testing

### File Upload / Import Endpoints
```bash
# Test with external URL imports
curl -X POST -H "Authorization: Bearer *** \
  -H "Content-Type: application/json" \
  -H "Notion-Version: 2026-03-11" \
  -d '{"url":"http://169.254.169.254.nip.io/latest/meta-data/"}' \
  "https://api.notion.com/v1/files"
```

### DNS Resolver Bypass Domains

Domains that resolve to private IPs via public DNS — these bypass string-based URL validators that don't resolve DNS before checking:

| Bypass Domain | Resolves To | Use Case |
|--------------|-------------|----------|
| `localtest.me` | 127.0.0.1 (and ::1) | Localhost access |
| `lvh.me` | 127.0.0.1 | Localhost access |
| `*.nip.io` | Arbitrary IP in subdomain | `169.254.169.254.nip.io` → AWS/GCP metadata |
| `*.sslip.io` | Arbitrary IP in subdomain | Same concept, independently maintained |

**What these bypass**: Any URL validation that checks the hostname string against a blocklist of IPs or hostnames (e.g. `is-localhost(hostname)` or `blockedIps.includes(parsedHostname)`) without resolving DNS first. The domain string is not in the blocklist, so validation passes. Then `fetch()` resolves the domain to the private IP and connects to internal services.

### SSRF Confirmation — Local Server PoC (Preferred)

When you find an endpoint that fetches external URLs, prove actual localhost access with a self-contained PoC. This is **always stronger** than third-party webhook evidence.

1. **Start a local HTTP server** on an unpredictable high port:
   ```bash
   python3 -m http.server 52589 --bind 127.0.0.1
   ```

2. **Capture DNS resolution** so you have it for evidence:
   ```bash
   dig +short localtest.me A
   host 169.254.169.254.nip.io
   ```

3. **Send test URL** through the vulnerable endpoint:
   - `http://localtest.me:52589/poc-proof`
   - `http://169.254.169.254.nip.io/latest/meta-data/`

4. **Capture all evidence** — DNS output, validation result, and the local server HTTP log showing the incoming request (method, path, User-Agent, headers, timestamp).

**Why this beats webhook-only evidence**:
- Proves the request reaches **this machine**, not just "somewhere on the internet"
- Eliminates the "CDN cache hit" / "could be their infrastructure" objection from triage
- Fully reproducible without external dependencies
- You control the server, so you capture full request details

### SSRF Confirmation — Webhook Verification (Fallback)

When local PoC is impossible (serverless target, cloud-only exploitation):

1. **Create a webhook endpoint**: POST to `https://webhook.site/token`
2. **Inject webhook URL** into the vulnerable feature
3. **Monitor**: GET `https://webhook.site/token/{uuid}/requests`
4. **Analyze**: source IP reveals internal infrastructure; User-Agent identifies the service making the request (NotionEmbedder, Iframely, notion-api, etc.)

### SSRF Confirmation Checklist

| Step | What It Proves | Evidence Needed |
|------|---------------|-----------------|
| DNS resolution | Bypass domain → private IP | Terminal output of `dig`/`host` |
| Validation bypass | URL passes the validator | Test output showing PASS |
| Live connection | Request reaches internal/local target | Local server log or webhook request log |
| Impact scope | What's reachable | Response content or timing analysis |

### GitLab-Specific SSRF Vectors

GitLab has a history of SSRF bugs — the codebase has patches that can be studied for incomplete fixes.

**High-value endpoints to probe (after obtaining a PAT):**

```bash
# Webhooks — classic SSRF vector
POST /api/v4/projects/:id/hooks      # url field
POST /api/v4/groups/:id/hooks        # url field

# Import — fetches external repositories
POST /api/v4/import                   # url field
POST /api/v4/import/github            # personal_access_token + repo_url

# Services/integrations (Slack, Jira, etc.) — may fetch URLs
PUT /api/v4/projects/:id/services/slack  # webhook field

# Avatar/image URLs — user-supplied fetch target
PUT /api/v4/users/:id/avatar            # avatar_url
PUT /api/v4/projects/:id                # avatar_url

# CI/CD pipeline triggers — may fetch external resources
POST /api/v4/projects/:id/trigger/pipeline  # token + ref + variables

# Snippet imports — older vector
POST /api/v4/snippets                  # file content with URLs
```

**Workhorse (`sendurl/`) SSRF hypothesis:**
The `workhorse/internal/sendurl/` package handles URL-based response sending. If it fetches a user-supplied URL without proper validation, it's an SSRF vector. Search for:
```bash
cd ~/Dev/gitlab/gitlab/workhorse
grep -rn "url\." --include="*.go" internal/sendurl/ | head -20
grep -rn "http\.\|Host\|Dial\|Transport\|RoundTrip" --include="*.go" internal/sendurl/
```

**Known SSRF patterns to check in git log:**
```bash
cd ~/Dev/gitlab/gitlab
git log --all --oneline --grep="SSRF" --grep="url.*valid" --all-match -20
git log --all --oneline --grep="CVE" --grep="ssrf" --all-match -10
git log --all --oneline --grep="internal.*network" --grep="protect" --all-match -10
```

### Redirect Chaining

Test if redirects bypass validation. Some validators check the initial URL but not each redirect hop:
- Set up a redirect from a public URL → `http://127.0.0.1/`
- Or from a public URL → `http://169.254.169.254/latest/meta-data/`
- If the target follows redirects without re-validating, the SSRF is trivially exploitable

> **Reference**: `references/ssrf-dns-bypass-techniques.md` — comprehensive catalog of all DNS bypass domains, IP obfuscation formats (decimal/hex/octal/IPv6-mapped), URL parser differential exploitation, redirect-based bypass, cloud metadata endpoints per provider (AWS/GCP/Azure), and a Python test script.
> **Reference**: `references/gitlab-program-intelligence.md` — includes GitLab SSRF attack vectors including Workhorse sendurl/path mapping and known SSRF patch analysis commands.

## Step 6: Rate Limit & Race Condition Testing

### Rate Limit Discovery
```bash
# Send burst requests to find rate limits
for i in {1..100}; do
  curl -s -H "Authorization: Bearer $TOKEN" "https://api.notion.com/v1/users" > /dev/null &
done
wait
```

### Race Conditions (Bug Bounty Bootcamp Ch12)
- Concurrent page creation with same title/ID
- Concurrent delete + update on same resource
- TOCTOU on permission changes
- Double-spending on file upload quotas

## Step 7: Business Logic Testing

### Workflow Abuse
- Invite user → remove → re-invite → permission persistence?
- Archive page → access via API → still visible?
- Share page → remove share → still accessible via cached tokens?
- Cross-workspace data leakage via shared pages

## Technique: Parallel Subagent Audits via delegate_task

For large codebases or multi-angle recon, spawn **3 parallel subagents** using `delegate_task` with `tasks` array. Each subagent gets isolated context + its own terminal.

### When to use parallel audits
- **Code audit across multiple repos**: Clone 3 repos simultaneously, one per subagent
- **Multi-angle recon**: API scan + source code audit + browser recon in parallel
- **Multiple vulnerability classes**: SSRF potential + auth issues + injection points simultaneously
- **Implementation comparison**: Same spec implemented in multiple languages (e.g., JWT SSO in PHP, Node, Python, Java, C#, Ruby, Go) — assign one subagent per 2-3 implementations

### Pattern
```python
# From execute_code or a Python script
from hermes_tools import delegate_task

results = delegate_task(
    tasks=[
        {"goal": "Audit repo A...", "context": "...", "toolsets": ["terminal", "file"]},
        {"goal": "Audit repo B...", "context": "...", "toolsets": ["terminal", "file"]},
        {"goal": "Audit repo C...", "context": "...", "toolsets": ["terminal", "file"]},
    ]
)
```

### Critical constraints
- **Max 3 concurrent** tasks per user (configurable via `delegation.max_concurrent_children`)
- **Nested delegation is OFF** (max_spawn_depth=1) — subagents are leaves, cannot spawn further
- **No user interaction** — subagents cannot call `clarify` or `memory`
- **Results are self-reported** — subagents say what they found; verify critical claims yourself
- **Sync within parent turn** — if the parent is interrupted, children are cancelled
- **Pass all context** — subagents have NO memory of your conversation

### Verification after parallel audit
```python
# Re-read key files the subagent claims to have found
# Stat paths to confirm they exist
# Run the subagent's PoC to confirm it works
```

## Step 8: PoC Archive Creation for HackerOne Submission

⚠️ **Many reports are rejected with "Needs more info - missing archive file with PoC."** Triage wants a downloadable .zip/.tar.gz they can extract and run without extra setup. A single code snippet pasted into the report is often insufficient.

Build a PoC archive when:
- The report involves running code (SSRF, RCE, injection, path traversal)
- The vulnerability is non-obvious (needs a demo to prove it's real)
- Triage specifically asks for an archive (common pattern on HackerOne)
- You're chaining multiple proof steps

### PoC Archive Structure

```
vuln-name-poc/
├── poc-bypass.js          # Individual proof: validation bypass (one vulnerability aspect)
├── poc-dns.js             # Individual proof: DNS resolution to private IP (context evidence)
├── poc-live-ssrf.js       # Individual proof: live connection proves SSRF (strongest evidence)
├── poc-all.js             # Runner: executes all PoCs sequentially, prints summary
├── README.md              # Instructions: exact steps to reproduce
├── report.md              # Full HackerOne report text (for reference)
└── src/                   # [Optional] Vulnerable source files for context
    ├── vulnerable-file.ts
    └── caller.ts
```

Archive it with:
```bash
cd ~/Dev && zip -r vuln-name-poc.zip vuln-name-poc/
# Verify size and contents
unzip -l vuln-name-poc.zip
```

### Archive Construction Rules

1. **Use published packages, not repo clones.** Triage can `npm install @scope/package` in seconds; they won't clone a 100MB repo. If the vulnerable function is in a published npm package, install the latest:
   ```bash
   # In README — exact copy-paste commands for triage
   npm install @ai-sdk/provider-utils
   node poc-bypass.js
   ```

2. **Multi-file with clear roles.** Each file proves one thing:
   - `poc-bypass.js` — passes URLs through the vulnerable function, reports pass/fail per URL
   - `poc-dns.js` — shows DNS resolution of bypass domains to private IPs
   - `poc-live-ssrf.js` — starts a local HTTP server, connects via DNS bypass, proves bidirectional communication
   - `poc-all.js` — orchestrator that runs each script in sequence

3. **README must be self-contained.** Include:
   - Prerequisites (Node.js, Python, etc.)
   - Exact install commands: `npm install @scope/package`
   - Exact run commands: `node poc-all.js`
   - Expected output (annotated with ✓/✗ so triage can compare)
   - What each script demonstrates

4. **Live SSRF evidence > webhook-only evidence.** When possible, include a PoC that starts a local server and connects to it via DNS bypass. This is unrebuttable — triage can't claim "CDN cache hit" or "third-party service." Always pair with DNS resolution capture so triage sees the full chain.

5. **Contrast evidence strengthens credibility.** If the same project has a correct implementation elsewhere (e.g., Next.js image optimizer resolves DNS → AI SDK does not), include both the vulnerable code and the correct code in `src/`. This proves the fix is known, cheap, and just wasn't applied.

6. **Zip on the target's file system.** Triage downloads and extracts the archive. Use `zip -r` on your local machine so there are no encoding issues. Verify with `unzip -l` before claiming success.

7. **Include source files for context.** Copy the actual vulnerable function source (`src/vulnerable-file.ts`) so triage can see exactly what the code does without navigating the repo. Pair it with the caller (`src/caller.ts`) so the data flow is clear.

### PoC Archive Template — README.md

```markdown
# Vulnerability Name — PoC Archive

## Prerequisites
- Node.js >= 18
- npm >= 9

## Setup
```bash
npm install @scope/package
```

## Reproduction Steps

### 1. Validation Bypass
```bash
node poc-bypass.js
```
Expected output:
```
BYPASS: http://localtest.me/              (8/8 bypass)
BYPASS: http://169.254.169.254.nip.io/
BYPASS: http://127.0.0.1.sslip.io/
...
BLOCKED: http://127.0.0.1/               (3/3 blocked as expected)
BLOCKED: http://localhost/
```

### 2. DNS Resolution (context)
```bash
node poc-dns.js
```
Expected output:
```
localtest.me → 127.0.0.1
169.254.169.254.nip.io → 169.254.169.254
```

### 3. Live SSRF (strongest evidence)
```bash
node poc-live-ssrf.js
```
Expected output:
```
DNS: localtest.me → ["127.0.0.1"]
<<< SSRF REQUEST RECEIVED >>>
✓ SSRF CONFIRMED: localhost reached via DNS bypass!
```

### All-at-once
```bash
node poc-all.js
```

## Architecture
```
poc-bypass.js     — Tests URL validation bypass (8/8 confirmed)
poc-dns.js        — Shows DNS resolution to private IPs (context evidence)
poc-live-ssrf.js  — Starts local HTTP server, proves bidirectional SSRF (strongest)
poc-all.js        — Runs all 3 PoCs sequentially
src/              — Vulnerable source files for triage reference
```

## Impact
[Brief impact description — cloud metadata exfiltration, internal network scanning]
```

## Step 9: Bugcrowd-Specific Reporting & Vulnerability Chaining

### Bugcrowd VRT Taxonomy

Bugcrowd uses a Vulnerability Rating Taxonomy (VRT) v1.18 (updated 9 Mar 2026) with P1 (Critical) through P5 (Informational) severity levels. Before writing a Bugcrowd report, search the VRT at https://bugcrowd.com/vulnerability-rating-taxonomy to find the correct category for your finding. The search box filters by vulnerability name and variant.

Key VRT mappings for common findings:
- **Authentication Bypass** (Broken Auth category) → **P1**
- **IDOR Modify/View Sensitive Info (Iterable IDs)** (BAC category) → **P1-P3**
- **SSRF Internal High Impact** (Server Sec Misconfig) → **P2**
- **Race Condition** (Server Sec Misconfig) → **Varies**
- **Hardcoded Credential Non-Privileged** (OS/Firmware) → **P2**

> **Reference**: `references/bugcrowd-vrt-taxonomy.md` — full taxonomy overview including all P-levels for SSRF, IDOR, auth bypass, race conditions, hardcoded creds, and sensitive data exposure.
> **Reference**: `references/scope-verification-case-study.md` — worked example of the scope verification framework applied to Rapyd sample code, showing the cost of not checking where code actually runs.

### Bugcrowd Explicitly Endorses Chaining

From the VRT Usage Guide:

> *"many hackers have used such vulnerabilities within 'exploit chains' consisting of two or three vulns resulting in creative, valid, and high-impact submissions"*

This means you should chain lower-severity findings when they compose into a higher-impact exploit path. Do NOT submit each individual low-severity bug separately — package them as one report with one PoC showing the full chain.

### Bugcrowd Submission Format (Exact Fields)

From the Bugcrowd report form:

| Field | Description |
|-------|-------------|
| **Summary** | 2-3 sentence overview. "Help us get an idea of what this vulnerability is about." |
| **Submission title** | Descriptive title summarizing the vulnerability or chain impact |
| **Target** | Dropdown from program scope targets. Targets not in scope may not be eligible. |
| **Technical severity** | VRT v1.18 baseline. Baseline is a guide — impact context may raise or lower. |
| **VRT Category** | Dropdown from VRT categories (e.g., "Broken Auth → Authentication Bypass") |
| **URL / Location** | Optional. Exact URLs or file paths. For source code: repo URL + file paths. |
| **Description** | Max 25,000 chars. Vulnerability, impact, and PoC/replication steps. |
| **Attachments** | Max 20 files, <400MB each. .jpg/.gif/.png <50MB embeddable as markdown. |

Each report DESCRIPTION section follows this field sequence:
```
## Summary
## Submission title
## Target
## Technical severity
## VRT Category
## URL / Location of vulnerability
## Description
### Vulnerability Chain Overview
### Proof of Concept
### Impact
### Remediation
## Attachments
```

### ⚠️ CRITICAL: Scope Verification — Verify Where Code Actually Runs Before Claiming Targets

**This is the most common and most costly mistake in bug bounty.** Always ask: where does this code actually run, and is that host listed in the program's targets?

#### The Framework: Six Questions Before Any Report

1. **Where does this code run?**
   - `api.vendor.com` → vendor infra ✅
   - `github.com/vendor/sample-code` → deployed by customers on their servers, NOT the vendor's infra

2. **Is the endpoint actually on the target?**
   ```bash
   curl -s -o /dev/null -w "%{http_code}" https://api.target.com/v1/some-endpoint
   # 200/400/401 = endpoint exists
   # 404 = endpoint doesn't exist on their infra
   ```

3. **Is the vulnerability verifiable against the live target?**
   - Running a local server to prove it → you're testing your own code, not the target
   - If you can't send a request to the live API → you haven't confirmed anything

4. **Is the target listed in the program scope?**
   - No GitHub repos listed → sample code findings target merchant infra
   - Some programs list "Source code" as a target type (rare — check the dropdown)

5. **What does the vendor's own docs say?**
   - "Use this GitHub code as your integration" → it runs ON THE MERCHANT'S SERVER

6. **Is the OpenAPI spec describing what exists or what's possible?**
   - Specs list ALL possible fields. Live APIs may mask/strip them.
   - Idempotency may be enforced server-side even if undocumented.
   - **Spec analysis ≠ confirmed vulnerability. Verify against the live API.**

#### Symptom Checklist — Are You Mapping Findings Wrong?\n\n| Symptom | Likely Cause |\n|---------|-------------|\n| PoC requires running a local server | You're testing sample code, not the live API |\n| Endpoint returns 404 when curling vendor's API | The endpoint doesn't exist on their infra |\n| "Verified" output is from your own local server | You confirmed your code works, not their vulnerability |\n| Finding requires merchant dashboard auth | It's a merchant-side issue, not a vendor API bug |\n| Finding is in `src/` of a sample repo | That's merchant-deployable code, not vendor infra |\n| OpenAPI analysis without live testing | Spec → assumption. Not evidence. |\n\n### ⚠️ CRITICAL: Impact Demonstration — "as an attacker I could..."

**This is the #1 reason Bugcrowd findings get rated P5 (Informational/no reward).** An endpoint returning data without auth is NOT a vulnerability unless an attacker can DO something with that data. Before submitting, apply the impact test: *"As an attacker I could..."*

**The Impact Test — Four Questions:**
1. What specific action can an attacker perform? If the data alone doesn't enable a second unauthorized action, severity drops to P5.
2. Does the leaked data enable a SECOND unauthorized action, or is the chain blocked at step 1?
3. Can you demonstrate the full attack chain end-to-end, or does it stop at info disclosure?
4. Is the exposed data sensitive (API keys, PII) or trivial (form names, standard field IDs)?

**Common P5 pitfalls:** unauth endpoint returning non-sensitive/public data; leaked data needing a 2nd vuln that doesn't exist; "info disclosure" without demonstrated harm; security best practice violations without exploitability.

**Escalate a P5 (from Bugcrowd feedback):** *"if you are able to exploit this finding further (or chain it with other findings) to meet the definition of another item within the VRT, please do submit a new report."* If no escalation path exists, the finding is genuinely P5.

### API Key Impact Verification — Finding Keys ≠ Usable Keys

One of the most common "false positives" in API bug bounty: finding partner API keys (SFOX, Plaid, ShapeShift, etc.) in a config file, environment variable dump, or public GitHub repo. **These keys must be verified against the partner's LIVE API before you can claim impact.** Partner keys found in a config endpoint are trivially discovered by anyone — the question is whether they're usable.

#### The Workflow — Test Each Key Independently

For every exposed key, run this 5-step verification:

1. **Identify the partner service** — What company/service owns the API the key is for? (SFOX = crypto trading, Plaid = bank linking, SiftScience = fraud detection)
2. **Find their current API docs** — Search for `docs.<service>.com`, `<service>/api`, `<service>/developers`
3. **Determine the auth mechanism** — Most APIs use one of:
   - `Authorization: Bearer <token>` — Standard bearer token
   - `X-API-Key: <key>` — Custom header
   - `X-<PARTNER>-PARTNER-ID: <key>` — Partner-specific header (SFOX pattern)
   - `client_id + secret` (both required, e.g. Plaid)
   - IP allowlist only (no key alone will work)
4. **Send a test request to the partner's API** with the key, using the documented auth mechanism:
   ```bash
   # Example: SFOX partner API with custom header
   curl -sL "https://api.partner.com/v2/partner/blockchain/quotes" \
     -H "X-PARTNER-PARTNER-ID: <key>"

   # Example: Bearer token
   curl -sL "https://api.partner.com/v1/account" \
     -H "Authorization: Bearer <key>"
   ```
5. **Classify the result:**

   | Response | Meaning | Impact Verdict |
   |----------|---------|---------------|
   | `200` with valid data | Key works! Document what it can do | **HIGH** — usable key |
   | `401` / `403` | Key rejected or IP-restricted | **NONE** — can't use |
   | `404` / route not found | API URL pattern wrong, or service deprecated | **NONE** — fix URL or check service status |
   | "Cannot GET /..." (Express) | API exists but that endpoint doesn't | **NONE** — try other paths |
   | Connection refused / timeout | API down or IP-blocked | **NONE** — can't reach |
   | `400 MISSING_PARAM` | Key format accepted, needs more params | **LOW** — partial, may escalate |

#### Common Dead-Ends Found in Practice

| Key Type | Why It Fails | Example |
|----------|-------------|---------|
| **Acquired company's key** | The company was bought, API migrated/deprecated | SFOX → Zero Hash (2023). `api.sfox.com/v2/partner/blockchain/*` returns 404. |
| **Incomplete credential pair** | API needs TWO values (client_id + secret). You found one. | Plaid: `client_id` alone can't authenticate. The `secret` is server-side only. |
| **Client-side-only key** | Designed to be public, rate-limited by origin/referrer, not a secret | SiftScience site keys are embedded in every page's JS. Public by design. |
| **Defunct platform** | Service no longer exists or radically changed business model | ShapeShift converted from centralized exchange to DEX; API key worthless. |

#### Report Writing Rule

**Do NOT write a report claiming "exposed API secret" unless verification step 4 returned a 200 with usable data.** A report that says "these keys exist" without demonstrating they do anything is P5 (Informational). The config file already serves these keys to every browser — GitHub just mirrors what's already public. Without proof the keys grant unauthorized access, there's no vulnerability.

#### User Intuition Signal

When the user says **"Are you sure this affects the target?"** — stop, verify, don't defend. The user's intuition about scope is often ahead of yours because they have better mental models of the program. Do NOT argue or rationalize. Re-run the 6 questions from scratch. If the answer to #1 or #4 is anything less than definitive, the finding is a hypothesis, not a confirmed vulnerability. Delete the hypothesis and find something that passes the 6-question test.

#### The Terminal Verification Imperative

When the user asks **"As an attacker I would...? Finish this sentence. Is it possible to verify this on the terminal?"** — they are demanding terminal-level proof of impact, not theoretical analysis. This is the single most important quality gate. Follow this pattern immediately:

1. **Stop writing reports.** Do not create REPORT.md until you can answer the question from live terminal output.
2. **Run the actual command** against the live target. If it requires an API key, token, or cookie — ask the user, don't assume it's blocked.
3. **Capture the output.** If the command fails (404, 403, wrong endpoint, defunct service), that IS the answer. Report it honestly — don't spin failure as "well if it worked."
4. **Present: "Here's the terminal output. As an attacker, I can [X] because of [Y]."** If you can't finish that sentence with evidence, the finding is not ready.

**Bad pattern (what got corrected this session):**
```
I wrote a report about partner API keys being exposed. The user asked:
"So for the vuln you've written up, as an attacker I would...?"
→ I hadn't actually tested the keys against the partner APIs.
→ Turns out SFOX was acquired, Plaid needs client_id+secret, ShapeShift is defunct.
→ Impact: nothing. No finding.
```

**Good pattern:**
```
Found API keys in config → Test each against its live API → Classify result →
Only report if terminal shows the key actually works → Present terminal output as evidence.
```

#### Evidence Hierarchy (Revised for Scope)

When deciding what to include in a report, evaluate by what it proves ABOUT THE TARGET:

1. 🟢 **Live API request + response** — Proves the endpoint exists and behaves as claimed on vendor infra.
2. 🟡 **OpenAPI spec analysis** — Shows what the API SHOULD do. Always note "needs live verification."
3. 🔴 **Sample code analysis** — Shows what vendor-published code does. Does NOT prove anything about vendor's live infra.
4. 🔴 **PoC running against local simulation** — Proves the concept in theory. Does NOT prove it affects any target.

### Systematic Unauthenticated Endpoint Scan (Zendesk Pattern)

When testing a new target for unauth access, use this systematic pattern rather than spot-checking:

1. **Create endpoint list** — Compile ALL known API v2 paths from the vendor's API docs
2. **Send each WITHOUT auth** — Just `Accept: application/json` header. Record HTTP status codes.
3. **Compare against similar endpoints** — If `/ticket_forms` returns 200 but `/ticket_fields` returns 401, that's the finding. The comparison proves the auth gap is intentional — the vendor knows to lock these down.
4. **Individual access test** — The list endpoint may be open but individual resource access should be 401. Test `/resource/:id` to confirm.
5. **Escalation test** — Try creating/modifying without auth (POST/PUT/DELETE). Usually these are properly locked down even if GET lists are open.
6. **Cross-reference with docs** — Check if the endpoint is documented as public. If docs say "requires auth" but doesn't, that's additional evidence.

**Key insight from Zendesk testing**: `/api/v2/ticket_forms` (list) returned 200 without auth while `/api/v2/ticket_forms/:id` (individual) returned 401. The list endpoint was the only one missing auth. Always test both list and individual endpoints — they may have different auth configurations.

### Chain Report Structure

**User style preference**: This user values directness and accuracy over quantity. When they ask "Are you sure?" — do not defend or rationalize. Stop, verify from scratch, and present the honest answer including negative results. Delete false work, don't keep it around. Verbose explanations that defend an incorrect position waste their time. If a finding doesn't survive fact-checking, delete it and move on.

When submitting a chained finding to Bugcrowd, the DESCRIPTION body should contain:

1. **Chain Breakdown** — Table per finding with file:line and VRT baseline
2. **Attack Flow** — ASCII/text or numbered steps showing chain composition
3. **PoC** — Single script demonstrating full chain (not per-bug PoCs). Must be VERIFIED against a running server or simulation, with console output included as evidence.
4. **Impact** — What a real attacker achieves end-to-end
5. **VRT Classification Table** — Per-finding baseline priority within the chain

**VERIFY PoCs before writing the report.** Run the PoC against a local server or simulation. Include the verified console output directly in the report. A table of "expected output" is not evidence — actual terminal output is.

**Do NOT chain when**: findings are independent with different root causes and fixes (submit separately), or one finding is already P1 standalone (adding weaker findings dilutes signal).

**DO chain when**: individual findings are P3-P5 but together create a P1-P2 exploit path.

### Report Folder Consolidation Pattern

When findings chain together, restructure from many tiny folders to a few consolidated ones:

```
BEFORE (11 individual findings):
~/Dev/REPORTS/Program/001/  (individual bug)
~/Dev/REPORTS/Program/002/  (individual bug)
...
~/Dev/REPORTS/Program/011/  (individual bug)

AFTER (3 chains + standalones):
~/Dev/REPORTS/Program/012/  Chain A — Webhook     (covers 004+001+002+003)
~/Dev/REPORTS/Program/013/  Chain B — Cart        (covers 006+007+008)
~/Dev/REPORTS/Program/014/  Chain C — Session     (covers 005+010)
~/Dev/REPORTS/Program/015/  Standalone — PII      (doesn't chain)
~/Dev/REPORTS/Program/016/  Standalone — No Sec   (doesn't chain)
```

Each folder: REPORT.md (Bugcrowd format) + poc/ subfolder. Delete old folders after consolidation.

## Step 10: Reporting

Use this structure (from Bug Bounty Bootcamp Ch2):
1. **Descriptive title** — e.g., "IDOR in PATCH /v1/pages/{id} allows modifying pages not shared with the integration"
2. **Summary** — Clear one-paragraph description
3. **Severity** — CVSS score and rationale
4. **Steps to Reproduce** — Exact curl commands, numbered, with terminal output
5. **Proof of Concept** — Screenshot or terminal output showing exploitation
6. **Impact** — What an attacker can achieve, with attack scenarios
7. **Mitigation** — How to fix (optional but signals professionalism)

### Evidence Quality — Ranked from Strongest to Weakest

When choosing what to include in a report, prefer evidence that is irrefutable and self-contained:

1. **🟢 Live local PoC** — Start a local HTTP server, prove the request arrives at your machine. Unrebuttable. The target's infrastructure cannot claim it's a "cache hit" or "CDN edge."
2. **🟢 DNS resolution capture** — `dig`/`host` output showing bypass domain → private IP. Strengthens impact argument by proving attacker-controlled routing.
3. **🟢 Contrast evidence** — Show that a similar component elsewhere in the same codebase DOES have correct validation (e.g., Next.js image optimizer resolves DNS → AI SDK does not). This proves the fix is known, inexpensive, and just wasn't applied. Makes triage harder to dismiss as "not a security issue."
4. **🟡 Webhook pingbacks** — Use only when local PoC is impossible (serverless, cloud-only targets). Weaker because triage can argue "third-party service" or "not confirmed on our infrastructure."
5. **🔴 Code analysis alone** — Weakest. Always confirm with execution/simulation if at all possible. A code-level analysis without PoC is often closed as "informative" or "needs more evidence."

**Rule of thumb**: If your evidence could be explained away by a CDN cache, an intermediary proxy, or a third-party service — find stronger evidence.

### Contrast Evidence — Why It Wins

When you find a vulnerability in one component but the same project has a correct implementation elsewhere for the same class of bug, include both in the report:

```
File A (vulnerable): packages/provider-utils/src/validate-download-url.ts
  - Checks hostname string only. No DNS resolution.
  - Allows: localtest.me (→ 127.0.0.1), 127.0.0.1.nip.io (→ 127.0.0.1)

File B (correct): packages/next/src/server/dev/image-optimizer.ts (line 863)
  - Same input: external URL fetch
  - But: resolves DNS → gets resolved IP → checks against blocklist
  - Blocks: localtest.me, 127.0.0.1.nip.io, any domain resolving to private IP
```

This makes two arguments triage cannot easily counter:
- The developer **knew this was a risk** (they fixed it in File B)
- The fix is **low-cost** (just port the DNS resolution check from File B)
- The vulnerability is **real** (you're not "finding things that aren't there"; you're finding things that were **missed**)

## Notion API Quick Reference

### Base Info
- **Base URL**: `https://api.notion.com`
- **Auth**: `Authorization: Bearer ntn_...`
- **Version header**: `Notion-Version: 2026-03-11` (latest)
- **Content-Type**: `application/json`
- **Rate limit**: ~3 req/s average (2700 per 15 min)

### Key Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/v1/users` | List workspace users |
| GET | `/v1/users/{id}` | Get user info |
| GET | `/v1/pages/{id}` | Get page |
| PATCH | `/v1/pages/{id}` | Update page |
| POST | `/v1/databases` | Create database |
| POST | `/v1/databases/{id}/query` | Query database |
| GET | `/v1/blocks/{id}/children` | List block children |
| PATCH | `/v1/blocks/{id}/children` | Append block children |
| POST | `/v1/comments` | Create comment |
| POST | `/v1/search` | Search content |
| POST | `/v1/files` | Initiate file upload |
| POST | `/v1/oauth/token` | Exchange OAuth code |
| GET | `/v1/oauth/token/{id}/introspect` | Introspect token |

### Testing Authentication Flow
```bash
# Set your token
export NOTION_TOKEN='ntn_your_token_here'

# Test basic connectivity
curl -s -H "Authorization: Bearer $NOTION_TOKEN" \
  -H "Notion-Version: 2026-03-11" \
  "https://api.notion.com/v1/users/me" | jq .

# Search
curl -s -X POST -H "Authorization: Bearer $NOTION_TOKEN" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{}' "https://api.notion.com/v1/search" | jq .
```
