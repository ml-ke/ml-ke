# Nutaku HAL/HATEOAS Reconnaissance — Worked Example

## Target
- **Program**: Nutaku (Aylo / Intigriti)
- **Domain**: www.nutaku.com
- **Stack**: Laravel (PHP) + nginx frontend, Spring Boot HATEOAS backend

## Discovery Chain

### Step 1: JavaScript Fingerprinting
Downloaded `atlasbundle.min.js` and `nutaku-main.js` from the homepage. Found references to `_xd` API paths in network requests — the browser was making GET requests to `/_xd/api/d/jsfp/<hash>` and `/_xd/api/health` returned `{"status":"UP"}`.

### Step 2: HAL API Discovery
Probing `/_xd/api/` with different paths revealed a Spring HATEOAS REST API.

```bash
# Found these non-404 endpoints:
/_xd/api/login       → 405 (POST works, returns 400/401)
/_xd/api/logout      → 405
/_xd/api/metrics     → 401 (requires admin auth)
/_xd/api/d/user      → 200 (empty body — needs app context)
/_xd/api/d/me        → 200 (empty body)
/_xd/api/d/session   → 200 (empty body)
/_xd/api/d/auth      → 200 (empty body)
/_xd/api/d/wallet    → 200 (empty body)
/_xd/api/d/gold      → 200 (empty body)
/_xd/api/d/games     → 200 (empty body)
/_xd/api/d/library   → 200 (empty body)
/_xd/api/d/favorites → 200 (empty body)
/_xd/api/d/config    → 200 (empty body)
/_xd/api/health      → 200 ({"status":"UP"})
```

All `/_xd/api/` endpoints return HAL JSON error format for 4xx responses.

### Step 3: Auth Endpoint Fingerprinting

**Correct format, wrong credentials → 401 (empty body)**:
```bash
printf '{"username":"real@email.com","password":"realpassword"}' > /tmp/login.json
curl -sv -X POST "https://www.nutaku.com/_xd/api/login" \
  -H "Content-Type: application/json" -d @/tmp/login.json
# → HTTP/2 401, content-length: 0
```

**Wrong format → 400 (reveals DTO structure)**:
```bash
curl -s -X POST "https://www.nutaku.com/_xd/api/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com"}'
# → {
#     "message": "Bad Request",
#     "_links": {"self": {"href": "/login", "templated": false}},
#     "_embedded": {"errors": [
#       {"message": "usernamePasswordCredentials.password: must not be blank"},
#       {"message": "usernamePasswordCredentials.username: must not be blank"}
#     ]}
#   }
```

**Key insight**: Even invalid usernames return 401 (same as real usernames with wrong password). No username enumeration. The `_xd` API uses a SEPARATE credential store from the Laravel frontend — our Nutaku credentials don't work there.

### Step 4: Actuator Probing on Gateway API

```bash
gateway-api.nutaku.net:

/actuator/env        → 403 (EXISTS — explicitly blocked)
/actuator/health     → 404 (doesn't exist)
/actuator/info       → 404
/actuator/metrics    → 404
/env                 → 404 (no Spring Boot 1.x prefix)
```

Only `/actuator/env` exists. The 403 (vs 404) proves it's deliberately secured — Spring Security `denyAll()` on that specific path. No bypass found with X-Forwarded-For or method manipulation.

### Step 5: Technology Fingerprinting

| Component | Technology | Fingerprint |
|-----------|-----------|-------------|
| Frontend | Laravel + nginx | Meta CSRF token, `/execute-login/` route, Blade templates |
| API Gateway | Spring Boot / Jetty | Jetty HTML 404 (<title>Error 404...</title> with URI table) |
| Internal API | Spring HATEOAS | HAL JSON (`_links`, `_embedded`) |
| CDN | StackPath | x-cdn-diag header format |
| Tracking | Nats/Atlas | atlasbundle.min.js, nats_cookie |

### Step 6: Cookies & Auth

| Cookie | Type | Notes |
|--------|------|-------|
| Nutaku_TOKEN | 64-char hex (SHA-256) | Main auth token, HttpOnly, SameSite=Lax |
| Nutaku_userLoggedIn | 0/1 | Login state flag |
| NUTAKUID | Random string | Session ID, 2hr expiry, HttpOnly |
| LBSERVERID | Server ID | Load balancer routing |

### What Didn't Work

- **All credentials return 401** on `_xd/api/login` — separate credential store
- **Actuator env** blocked at app level (403) — no bypass via headers
- **JS bundle search** — no embedded credentials or API keys
- **Mobile API** (mobileapi.nutaku.com) — always returns 500
- **Subdomain takeover** on `*.nutakupublishing.com` — no DNS resolution

### Lessons Learned

1. HAL `_links.self.href` reveals internal path aliases — `/login` tells you the auth endpoint routes to a Spring Security login resource
2. 401 vs 400 on auth endpoints is the KEY distinguisher — 400 with field validation errors is information disclosure
3. 403 on actuator ≠ 404 on actuator — always distinguish; 403 means the endpoint exists behind an explicit deny
4. JavaScript fingerprint API (`_xd/api/d/jsfp/<hash>`) is a common tracking endpoint that reveals the API prefix pattern
5. When regex grepping JS files for URLs fails (minified code), use performance API entries to find live network calls made by the page
