# Nutaku APK Analysis & Gateway Authentication

## Target
Nutaku (Aylo) — Intigriti bug bounty program  
Android APK `nutakuclientlatest.apk` (11.6 MB, v1.44.0)  
Package: `com.project.nutaku`

## APK Decompilation Workflow

### 1. Extract APK
```bash
unzip nutakuclientlatest.apk -d extracted/
```
Check for DEX files (classes.dex, classes2.dex, classes3.dex), native libs, assets, resources.

### 2. Search for Credentials (Fast Pass)
```bash
strings classes*.dex | grep -iE 'client.?[_-]?id|client.?[_-]?secret|api.?[_-]?key|provider|grant.?type'
strings resources.arsc | grep -iE 'AIza|google_api|google_app|api_key|client|secret|token'
```
**Limitation**: Obfuscated class/field names won't appear as plaintext in strings output.

### 3. Decompile with jadx
```bash
jadx --show-bad-code --no-debug-info -d decompiled/ nutakuclientlatest.apk
```
For large APKs (3+ DEX, 8K+ classes), use `--threads-count 4` and let it run.
Results land in `decompiled/sources/` with the original package structure.

### 4. Find Obfuscated Config Classes
The credentials are often in a `BuildConfig`-like class (not the actual BuildConfig, but a project config class with obfuscated name).  
Search pattern: `find decompiled/sources/ -name "C[0-9]*.java" | head -20`  
Look for classes with:
- `final String` constants containing recognizable values
- URLs, API keys, client IDs
- `final String fXXXXX = "some_value"` patterns

The Nutaku config was in `ya/C5490n.java`:
```java
f87930m = "nutaku";                          // Default provider
f87931n = "sp";                               // Device type
f87932o = "authorization_code";               // Grant type
f87936s = "oauth-front";                      // Client ID
f87937t = "oauth-frontpass";                  // Client Secret
```

### 5. Find the Retrofit API Interface
Search for `@f("`, `@o("`, `@p("`, `@n(` annotations. These are Retrofit API endpoint definitions.  
Nutaku's interface was in `Yb/a.java` — all 28 endpoints in one file.

## Nutaku Auth Architecture

### Two-Layer Auth

**Layer 1: Web Frontend (Laravel)**
- Login at `www.nutaku.com/execute-login/`
- Requires CSRF token + reCAPTCHA v3
- Sets HttpOnly `Nutaku_TOKEN` cookie (SHA-256, 64 hex chars)
- Used by `www.nutaku.com/_xd/api/` endpoints

**Layer 2: Gateway API (Spring Boot)**
- Base: `https://gateway-api.nutaku.net/v1`
- Login: `POST /v1/authapi/user/login`
- Requires: `clientId`, `clientSecret`, `provider`, `grantType`, `username`, `password`
- Returns: `access_token` (JWT, RS256), `refresh_token`, `expires_in`, user profile
- Default creds: `oauth-front` / `oauth-frontpass` / `nutaku`
- All subsequent calls use `Authorization: Bearer <token>`

### Discovered Credentials
| Parameter | Value | Source |
|-----------|-------|--------|
| `clientId` | `oauth-front` | `C5490n.f87936s` |
| `clientSecret` | `oauth-frontpass` | `C5490n.f87937t` |
| `provider` | `nutaku` | `C5490n.f87930m` |
| `grantType` | `authorization_code` | `C5490n.f87932o` |
| `device` | `sp` | `C5490n.f87931n` |

### Token Properties
- Format: JWT, RS256 signed, key ID `nutakujwtkey`
- Issuer: `http://private-authapi.nutaku.net/v1`
- Subject: member ID (`z20260602193239018043291260`)
- Audience: `oauth-front`
- Expiry: 8999 seconds (~2.5 hours)
- Refresh: `POST /v1/authapi/user/login/refresh` (returns new token pair)

### Working Authenticated Endpoints
| Endpoint | Returns |
|----------|---------|
| `GET /v1/users/me` | User profile (id, nickname, grade, email, provider, site) |
| `GET /v1/users/me/funds/gold` | Wallet balance `{free, paid, total}` |
| `GET /v1/users/me/campaign-resources` | Campaign list |
| `GET /v1/users/me/sexual-preferences` | Int list |
| `GET /v1/users/{userId}/favorite-games` | Favorite games (IDOR-tested: empty) |
| `GET /v1/calendars` | Calendar list |
| `POST /v1/users/me/favorite-games` | Add favorite |
| `PATCH /v1/users/me/favorite-games/{titleId}` | Reorder favorites |

### Public Endpoints (no auth)
| Endpoint | Returns |
|----------|---------|
| `GET /v1/catalog/titles?...` | Game catalog (paginated, filterable) |
| `GET /v1/catalog/titles/{titleId}` | Single game details |
| `GET /v1/meta/campaign-resources?...` | Campaign/banner resources |
| `GET /v1/games-meta/events` | Event list |
| `GET /v1/news-and-updates/articles` | News articles |

## Auth Endpoint Error Codes (Gateway)
| HTTP | Body | Meaning |
|------|------|---------|
| 422 | `{"errors":["clientSecret must not be null","clientId must not be null","provider must not be null"]}` | Missing required OAuth2 fields |
| 400 | `{"error_code":"invalid_client","description":"No client found for clientId: ..."}` | Client ID unknown to upstream auth service |
| 200 | `{"access_token":"...","token_type":"Bearer",...}` | Successful authentication |

## New Domains Discovered via Decompilation
| Domain | Purpose | Status |
|--------|---------|--------|
| `userapi.nutaku.net` | User API | 403 |
| `metaapi.nutaku.net` | Metadata API | 403 |
| `vendor-gateway-api.nutaku.net` | Vendor gateway | 404 root |
| `gateway-api.nutaku.net` | Main gateway | ✅ Working |
| `cdn-updater.nutaku.net` | APK/CDN updates | ✅ Working |
| `private-authapi.nutaku.net` | Internal auth (JWT issuer) | Not DNS-reachable |

## Skills Integration
- This reference is part of `api-hacking-methodology` — the APK decompilation phase feeds directly into API auth bypass testing
- See `atlas-continuous-learning` v2.0.0 for the broader learning framework
- See `intigriti-vpn` for VPN tunnel setup to reach PWN environment
