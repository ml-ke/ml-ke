# Nutaku Gateway API Recon & Findings

## Catalog Endpoint Information Disclosure

**Finding**: 224 game API credentials leaked through public `GET /v1/catalog/titles` endpoint.

The endpoint returns `socialApi.consumerKey` and `socialApi.consumerSecret` for every game. These are OpenSocial OAuth 1.0 credentials used by game servers to authenticate against the platform API. The endpoint requires NO authentication.

### Confirmed Leaked Credentials (sample)

| Game | consumerKey | consumerSecret |
|------|------------|----------------|
| smutstone | (numeric) | `=LkMegXXQ1TZ?zNlrhlV9CiBdbGJDz0x` |
| harem-heroes | (numeric) | `UA?Lo9av7KJiNP5DYBiM3q[Tm?[iSqgu` |
| kamihime-r | (numeric) | `oVdbGgskQK17vaaDL49l6A4j_OYn0mI@` |
| kink-inc | (numeric) | `iZuaevBS=eZt$UDIwotTyqVw5gSRl]4u` |
| eden-survivors | (numeric) | `TocV36rkARMx0=sd4kYuT8PSRoXU1kfB` |

Total: 224 credentials across 100 games (multiple projects per game).

### PoC (no auth needed)
```
GET /v1/catalog/titles?per-page=100
Host: gateway-api.nutaku.net
```

## User Enumeration

**Finding**: Sequential numeric user IDs at `/v1/users/{id}` return 200 (exists) vs 404 (doesn't exist). Requires Bearer token.

Confirmed active users across the range:
- ID 1000: AEunJi
- ID 1,000,000: lollo98
- ID 18,257,4649: Yusif (adjacent to ours)
- ID 18,257,4651: Valoh

## Hardcoded Gateway Credentials

APK decompilation revealed static OAuth2 client credentials:
- clientId: `oauth-front`
- clientSecret: `oauth-frontpass`
- provider: `nutaku`
- grantType: `authorization_code`

These enable unlimited API access, account creation, and brute force without CAPTCHA.

## Gateway API Auth Flow

```
POST /v1/authapi/user/login
Content-Type: application/json

{"clientId":"oauth-front","clientSecret":"oauth-frontpass",
 "provider":"nutaku","grantType":"authorization_code",
 "username":"...","password":"..."}

Response 200:
{
  "access_token": "eyJ...JWT",
  "token_type": "Bearer",
  "expires_in": 8999,
  "refresh_token": "...",
  "user": { "id": 182574650, "nickname": "h0d4r1254", ... }
}
```

JWT properties:
- Algorithm: RS256, key ID `nutakujwtkey`
- Issuer: `http://private-authapi.nutaku.net/v1` (internal)
- Audience: `oauth-front`
- Expiry: 2.5 hours

## Authentication Bypass Analysis

The gateway login endpoint at `authapi/user/login` accepts username/password WITHOUT reCAPTCHA or CSRF protection. The main Laravel frontend at `/execute-login/` requires both. This is a security control bypass.

However, the credentials require the OAuth2 clientId/clientSecret which are embedded in the APK — meaning any user who decompiles the app gets them.

## What Did NOT Work

| Vector | Result |
|--------|--------|
| JWT alg:none | 401 (properly rejected) |
| Mass assignment on login | Grade stays 1, no role injection |
| IDOR on favorite games | All users return empty |
| Purchase price manipulation | 402 — server-enforced pricing |
| Gold manipulation | 405 — read-only endpoint |
| Social login (Google) | 503 — upstream unavailable |
| _xd API delivery endpoints | Empty — game-specific auth |
