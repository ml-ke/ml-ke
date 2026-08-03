---
name: skoda-vw-connected-car-api
description: "API reconnaissance & vulnerability testing methodology for Skoda/VW Group connected car apps (MyŠkoda, WeConnect, etc.) — auth flow, endpoints, MQTT, attack vectors."
version: 2.0.0
author: ATLAS
---

# Skoda/VW Connected Car API Methodology

## Overview

The MyŠkoda mobile app and other VW Group connected car apps share a common backend infrastructure. This skill covers the API architecture, auth flow, and vulnerability testing methodology.

The **open-source Python library** at `github.com/skodaconnect/myskoda` (v2.13+) is the single best reference — it reverse-engineers the entire API. Clone it before starting any Skoda hunting:
```bash
git clone --depth 1 https://github.com/skodaconnect/myskoda.git ~/Dev/skoda/myskoda
```
Key files: `const.py` (endpoints, config), `auth/authorization.py` (full auth flow), `rest_api.py` (all API calls), `mqtt.py` (MQTT client), `models/` (data models).

## Architecture

```
App -> identity.vwgroup.io (OIDC Auth with PKCE) -> Access Token
    -> mysmob.api.connect.skoda-auto.cz (Primary REST API - base path: /api)
    -> prod.emea.mobile.charging.cariad.digital (Charging service - Ktor engine)
    -> mqtt.messagehub.de:8883 (MQTT - real-time control & events)
    -> Firebase Cloud Messaging (Push notifications)
```

Two API generations:
1. **VW-Group API** (msg.volkswagen.de) - Older cars, `fs-car/bs/*` paths
2. **Skoda Native API** (mysmob.api.connect.skoda-auto.cz) - Newer cars (Enyaq etc.)

## Auth Flow

- IDP: `https://identity.vwgroup.io`
- Client ID: `7f045eee-7003-4379-9968-9355ed2adb06@apps_vw-dilab_com`
- Redirect URI: `myskoda://redirect/login/`
- Flow: OIDC Authorization Code with PKCE

**Pitfall — Browser detection**: The VW IDK server checks for browser capabilities (JavaScript, User-Agent). Curl without proper headers returns a `browserFeaturesMissingError` page with `template:"browserFeaturesMissingError"` instead of the login form. Use a realistic User-Agent and accept cookies between steps. The Python library (`myskoda` on PyPI v2.13.0) uses aiohttp with proper session handling — reference its `auth/authorization.py` for the exact flow. (S256)

### Full Scope List (from Python library)
```
address badge birthdate cars driversLicense dealers email mileage mbb
nationalIdentifier openid phone profession profile vin
```

The `vin` and `cars` scopes are Skoda-specific — they may grant access to vehicle data not available with standard OIDC scopes.

### Auth Flow Steps

**Step 1 — Generate PKCE challenge:**
```python
import hashlib, base64
verifier = "".join(random.choices(string.ascii_uppercase + string.digits, k=64))
verifier_hash = hashlib.sha256(verifier.encode("utf-8")).digest()
challenge = base64.b64encode(verifier_hash).decode().replace("+", "-").replace("/", "_").rstrip("=")
```

**Step 2 — Initial authorize:**
```
GET /oidc/v1/authorize?client_id=7f045eee-7003-4379-9968-9355ed2adb06@apps_vw-dilab_com
    &nonce={16-char random}
    &redirect_uri=myskoda://redirect/login/
    &response_type=code
    &scope=address+badge+birthdate+cars+driversLicense+dealers+email+mileage+mbb+nationalIdentifier+openid+phone+profession+profile+vin
    &code_challenge={S256_challenge}
    &code_challenge_method=s256
    &prompt=login
```
→ Returns `302` redirect to `signin-service/v1/signin/{client_id}?relayState={...}`

**Step 3 — Follow redirect to signin page:**
```
GET signin-service/v1/signin/{client_id}?relayState={...}
```
→ Returns HTML (7809 bytes) containing `window._IDK = { ... }` JSON with `csrf_token`, `hmac`, and `relayState`. Extract with:
```python
import re, json
idk_match = re.search(r'window\._IDK\s=\s({.*?})\s*,\s*disabledFeatures', html)
idk_data = json.loads(idk_match.group(1))
csrf = idk_data['csrf_token']
hmac = idk_data['templateModel']['hmac']
relay_state = idk_data['templateModel']['relayState']
```

**Step 4 — POST email (form-encoded, NOT JSON):**
```
POST /signin-service/v1/{client_id}/login/identifier
Content-Type: application/x-www-form-urlencoded

relayState={relay_state}&email={email}&hmac={hmac}&_csrf={csrf}
```
→ Returns next auth page HTML with password form tokens.

**Step 5 — POST password (form-encoded):**
```
POST /signin-service/v1/{client_id}/login/authenticate
Content-Type: application/x-www-form-urlencoded

relayState={relay_state}&password={password}&hmac={hmac}&_csrf={csrf}
```
→ Follow redirects to extract auth code from callback URI.

**Step 6 — Exchange code for tokens:**
```
POST /oidc/v1/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code={auth_code}&redirect_uri=myskoda://redirect/login/&code_verifier={verifier}&client_id={client_id}
```
→ Returns JSON with `accessToken`, `refreshToken`, `idToken`.

**Step 7 — Refresh token (uses Skoda API, not VW IDP):**
```
POST /api/v1/authentication/refresh-token?tokenType=CONNECT
Content-Type: application/json

{"token": "{refresh_token}"}
```
⚠️ This endpoint returns **403 Access Denied** from non-VW IP ranges — IP-restricted.

### Pitfalls
- The signin-service POSTs **must** be `application/x-www-form-urlencoded`, not JSON (returns 405 otherwise)
- CSRF tokens and relayState expire — each auth attempt needs fresh values
- The VW IDK server returns `browserFeaturesMissingError` via curl (detects non-browser User-Agent/JS)
- The refresh endpoint is IP-restricted (403) — only works from VW/Skoda internal networks
- Old VW API at `msg.volkswagen.de` is 404/dead — only Skoda Native API is active

## API Endpoints (Skoda Native)

Base URL: `https://mysmob.api.connect.skoda-auto.cz/api`

### Read Endpoints
| Path | Description |
|------|-------------|
| `GET /v1/users` | User profile |
| `GET /v2/garage` | Vehicle list (with capabilities) |
| `GET /v2/garage/vehicles/{vin}` | Vehicle info |
| `GET /v1/charging/{vin}` | Charging status |
| `GET /v1/charging/{vin}/profiles` | Charging profiles |
| `GET /v1/charging/{vin}/history` | Charging history |
| `GET /v2/vehicle-status/{vin}` | Vehicle status |
| `GET /v2/vehicle-status/{vin}/driving-range` | Driving range |
| `GET /v2/air-conditioning/{vin}` | AC status |
| `GET /v2/air-conditioning/{vin}/auxiliary-heating` | Aux heating |
| `GET /v1/maps/positions?vin={vin}` | Position history |
| `GET /v3/maps/positions/vehicles/{vin}/parking` | Current location |
| `GET /v1/trip-statistics/{vin}` | Trip data |
| `GET /v1/single-trip-statistics/{vin}` | Single trip data |
| `GET /v1/vehicle-health-report/warning-lights/{vin}` | Health (warning lights) |
| `GET /v1/maintenance/{vin}` | Maintenance info |
| `GET /v1/maintenance/{vin}/report` | Maintenance report |
| `GET /v1/spin/verify` | SPIN verification check |
| `GET /v2/garage/vehicles/{vin}/departure-timers` | Departure timers |
| `GET /v1/driving-score/{vin}` | Driving score |
| `GET /v1/vehicle-info?vin={vin}` | Vehicle detailed info |
| `GET /v1/vehicle-renders` | Vehicle images |
| `GET /v1/vehicle-equipment/{vin}` | Vehicle equipment |
| `GET /v1/vehicle-connection-status/{vin}` | Connection status |
| `GET /v1/software-update/{vin}` | Software update status |
| `GET /v1/widget` | Widget data |
| `GET /v1/loyalty-program` | Loyalty program details |
| `GET /v1/loyalty-program/member` | Loyalty member info |
| `GET /v1/loyalty-program/badges` | Loyalty badges |
| `GET /v1/loyalty-program/badge/{id}` | Single badge |
| `GET /v1/loyalty-program/challenges` | Loyalty challenges |
| `GET /v1/loyalty-program/games` | Loyalty games |
| `GET /v1/loyalty-program/rewards` | Loyalty rewards |
| `GET /v1/loyalty-program/transactions` | Loyalty transactions |
| `GET /v1/loyalty-program/salesforce-contacts` | Salesforce contacts |

All endpoints return **401** with empty body and `www-authenticate: Bearer` header when unauthenticated. No error details or stack traces leaked.

### Write Endpoints (via REST)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/spin/verify` | Verify S-PIN (needs valid token + body) |
| POST | `/v1/fcm/register` | Register FCM push token |

### Write Operations (via MQTT)
All car control commands go through MQTT topics on `mqtt.messagehub.de:8883`. Full list in `references/mqtt-topics.md`.

### MQTT Auth Failure Detection
MQTT uses FCM-derived TOTP for credential validation. Auth failure codes:
- v3.1.1: 4 (Bad username/password), 5 (Not authorised)
- v5: 0x86 (Bad User Name or Password), 0x87 (Not authorized), 0x8C (Bad authentication method)

## Firebase Configuration

From `const.py`:
| Parameter | Value |
|-----------|-------|
| Project ID | `678067506455` |
| App ID | `1:678067506455:android:4afca86c91d6d4c235bb52` |
| Sender ID | `678067506455` |
| Android Cert | `E567A2E2E6C5E889CDB37EF07EBEC1576C196325` |
| Package | `cz.skodaauto.myskoda` |
| FCM Client | `fcm-25.0.1` |
| Android OS | `35` |
| GMS Version | `260000000` |

API key is obfuscated in the open-source library — extract from the actual APK via jadx decompilation.

## App Versioning
- App version: `8.12.0`
- App version code: `260430001`
- Package: `cz.skodaauto.myskoda`

## Attack Vectors

### 1. Auth Bypass on VIN-scoped Endpoints
Test if you can access any `{vin}` endpoint WITHOUT a valid token, or with another user's token.
```bash
curl -s "https://mysmob.api.connect.skoda-auto.cz/api/v2/vehicle-status/{vin}" \
  -H "Authorization: Bearer x"           # arbitrary token test
curl -s "https://mysmob.api.connect.skoda-auto.cz/api/v2/garage" \
  # no auth header at all
```

### 2. VIN Enumeration
Check if `/v2/garage` or `/v2/garage/vehicles/{vin}` leaks VINs of other users.
Sequential VIN patterns (WVWZZZ...) may be enumerable.

### 3. SPIN Verification Weakness
- Check rate limiting on `POST /v1/spin/verify`
- Check if SPIN can be bypassed (empty body, edge cases)
- Check if SPIN is transmitted in cleartext
- Check if known default SPINs work (0000, 1234, etc.)

### 4. MQTT Token Abuse
- MQTT uses the same access token for auth
- If token is compromised, attacker can send car control commands
- Check if MQTT token has different scope/expiry than REST token
- Check if token refresh extends MQTT access

### 5. Position Tracking
- `/v1/maps/positions?vin={vin}` - historical positions
- `/v3/maps/positions/vehicles/{vin}/parking` - current location
- No apparent consent per-query — if auth bypass found, full tracking possible

### 6. FCM Token Hijacking
- FCM token registered at `POST /v1/fcm/register`
- If you can register your own FCM token for another user's vehicle...
- Firebase project ID: `678067506455`
- Android cert: `E567A2E2E6C5E889CDB37EF07EBEC1576C196325`

### 7. Charging API
- Check if setting charge limits below safe thresholds is possible
- Check if remote charging start/stop can be abused
- Charging service at `prod.emea.mobile.charging.cariad.digital` (Ktor engine, returns 400 on bare request)

### 8. JWT Algorithm Confusion
- JWKS at `https://identity.vwgroup.io/v1/jwks` returns 3 RSA keys (kid hash, RS256)
- Test if the API accepts algorithm downgrade: RS256 → HS256 using the public key as HMAC secret
- Test if kid injection works (SQLi or path traversal via kid header)

## Tools Setup

```bash
# For intercepting app traffic
mitmproxy -p 8080 --mode transparent
adb reverse tcp:8080 tcp:8080  # with physical device

# For static APK analysis
pip3 install --break-system-packages jadx-cli && jadx -d decompiled/ app.apk

# For cloning the reference Python library
git clone --depth 1 https://github.com/skodaconnect/myskoda.git ~/Dev/skoda/myskoda

# For testing API
curl -s -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "https://mysmob.api.connect.skoda-auto.cz/api/v2/garage"
```

## Old Library Endpoints (skodaconnect - archived)

For older cars that still use the VW-Group API:
- Base: `https://msg.volkswagen.de` (⚠️ returns 404 — likely deprecated)
- Endpoints use `fs-car/bs/{service}/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/`
- Services: batterycharge, climatisation, departuretimer, cf (position), tripstatistics, rs (heater), vsr (refresh)
- See `references/old-api-endpoints.md` for legacy paths

## Sources

- https://github.com/skodaconnect/myskoda (modern — **primary reference**, v2.13+)
- https://github.com/skodaconnect/skodaconnect (archived)
- https://github.com/tillsteinbach/CarConnectivity-connector-skoda
- https://github.com/robinostlund/volkswagencarnet
- https://pypi.org/project/myskoda/ (Python library)
- App version: 8.12.0, package: cz.skodaauto.myskoda
