# Nutaku Recon Session — June 2, 2026

## Summary
Full recon on Nutaku (Aylo, Intigriti public program). Zero reportable findings but discovered critical infrastructure details and learned APK analysis workflow.

## What We Found

### Tech Stack
- Frontend: Laravel (PHP) + nginx on www.nutaku.com and www.nutaku.net
- API Gateway: Spring Boot (Java/Jetty) at gateway-api.nutaku.net
- Internal REST: Spring HATEOAS (HAL JSON) at /_xd/api/
- Mobile API: nginx + PHP at mobileapi.nutaku.com
- OpenSocial: osapi.nutaku.com (v2.0)
- CDN: StackPath

### Hidden Endpoints Discovered

**From web recon (/_xd/api/):**
- /_xd/api/login — POST {"username":"...","password":"..."} — always 401 (different credential store)
- /_xd/api/metrics — 401 without admin
- /_xd/api/logout — 405
- /_xd/api/d/* — 200 empty (delivery endpoints, need game-specific auth)
- /_xd/api/health — {"status":"UP"}

**From APK analysis (gateway-api.nutaku.net/v1/):**
- /v1/authapi/user/login — POST — 422, expects clientId, clientSecret, provider, username, password
- /v1/authapi/user/social-login — POST — 422, also needs device
- /v1/auth/user/login — POST — 422 (same as above, different path)

**From APK analysis (other domains):**
- metaapi.nutaku.net — 403
- vendor-gateway-api.nutaku.net — 404
- sbox-osapi.nutaku.com/social_android/rest/ — sandbox OpenSocial
- sbox-mobileapi.nutaku.com — sandbox mobile API
- stage-api.gateway.nutaku.net/v1/ — staging gateway
- stage-api.vendor-gateway.nutaku.net/ — staging vendor gateway
- stage-newtaku.nutaku.net — staging env
- stage-metaapi.nutaku.net — staging metadata API

### Auth System Architecture
- Laravel web auth: CSRF token + reCAPTCHA v3, stores Nutaku_TOKEN (SHA-256) cookie
- _xd API: Spring HATEOAS HAL login endpoint, different credential store (always 401)
- Gateway API: OAuth2 client credentials flow with clientId, clientSecret, provider fields
- Values embedded in APK but obfuscated in Kotlin/Java code

### Account
- h0d4r1254@intigriti.me registered, ID 182574650, email unverified
- Works on www.nutaku.com (Laravel) but NOT on _xd API or gateway API

### Blockers
- Primary blocker: cannot find clientId/clientSecret values in APK (obfuscated)
- Secondary: timing attack on _xd/login shows 1.0s vs 1.28s pattern but needs single-packet sync for precision
- Program OOS rules eliminate: account compromise, info disclosure without impact, rate limiting

### Key Learnings
- APK reverse engineering revealed more endpoints than weeks of web recon
- 422 error responses leak the exact data model (class name, field names)
- Always iterate request formats based on error messages
- Staging/sandbox environments (sbox-*, stage-*) exist and may have weaker security
