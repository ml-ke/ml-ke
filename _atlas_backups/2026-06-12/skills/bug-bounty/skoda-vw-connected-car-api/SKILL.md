---
name: skoda-vw-connected-car-api
description: "API reconnaissance & vulnerability testing methodology for Skoda/VW Group connected car apps (MyŠkoda, WeConnect, etc.) — auth flow, endpoints, MQTT, attack vectors."
version: 1.0.0
author: ATLAS
---

# Skoda/VW Connected Car API Methodology

## Overview

The MyŠkoda mobile app and other VW Group connected car apps share a common backend infrastructure. This skill covers the API architecture, auth flow, and vulnerability testing methodology.

## Architecture

```
App -> identity.vwgroup.io (OIDC Auth) -> Access Token
    -> mysmob.api.connect.skoda-auto.cz (Primary REST API - base path: /api)
    -> prod.emea.mobile.charging.cariad.digital (Charging service)
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

### Steps:
1. `GET /.well-known/openid-configuration` -> get authorization_endpoint
2. `GET /oidc/v1/authorize?client_id=...&redirect_uri=...&response_type=code id_token&scope=...&nonce=...&state=...`
3. `POST /signin-service/v1/{client_id}/login/identifier` (email)
4. `POST /signin-service/v1/{client_id}/login/authenticate` (password + CSRF + HMAC)
5. Follow redirects to extract auth code from callback URI
6. Exchange code for tokens (IDK session, refresh token, access token)

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
| `GET /v2/vehicle-status/{vin}/driving-range` | Range |
| `GET /v2/air-conditioning/{vin}` | AC status |
| `GET /v2/air-conditioning/{vin}/auxiliary-heating` | Aux heating |
| `GET /v1/maps/positions?vin={vin}` | Position history |
| `GET /v3/maps/positions/vehicles/{vin}/parking` | Current location |
| `GET /v1/trip-statistics/{vin}` | Trip data |
| `GET /v1/vehicle-health-report/warning-lights/{vin}` | Health |
| `GET /v1/spin/verify` | SPIN verification |
| `GET /v2/garage/vehicles/{vin}/departure-timers` | Departure timers |
| `GET /v1/driving-score/{vin}` | Driving score |

### Write Endpoints (via REST)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/spin/verify` | Verify S-PIN |
| POST | `/v1/fcm/register` | Register FCM push token |

### Write Operations (via MQTT)
All car control commands go through MQTT topics on `mqtt.messagehub.de:8883`:
- `air-conditioning/start-stop-air-conditioning`
- `air-conditioning/set-target-temperature`
- `charging/start-stop-charging`
- `charging/update-charge-limit`
- `charging/update-charge-mode`
- `vehicle-access/honk-and-flash`
- `vehicle-access/lock-vehicle`
- `vehicle-wakeup/wakeup`

Full MQTT operation list in `references/mqtt-topics.md`.

### MQTT Real-Time Events
Broker: `mqtt.messagehub.de:8883` (TLS)
Topics: air-conditioning, charging, departure, vehicle-status/*, vehicle-connection-status-update, vehicle-ignition-status

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
- Charging service at `prod.emea.mobile.charging.cariad.digital`

## Tools Setup

```bash
# For intercepting app traffic
mitmproxy -p 8080 --mode transparent
adb reverse tcp:8080 tcp:8080  # with physical device

# For static APK analysis
pip3 install --break-system-packages jadx-cli && jadx -d decompiled/ app.apk

# For testing API
curl -s -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "https://mysmob.api.connect.skoda-auto.cz/api/v2/garage"
```

## Old Library Endpoints (skodaconnect - archived)

For older cars that still use the VW-Group API:
- Base: `https://msg.volkswagen.de`
- Endpoints use `fs-car/bs/{service}/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/`
- Services: batterycharge, climatisation, departuretimer, cf (position), tripstatistics, rs (heater), vsr (refresh)
- See `references/old-api-endpoints.md` for legacy paths

## Sources

- https://github.com/skodaconnect/myskoda (modern)
- https://github.com/skodaconnect/skodaconnect (archived)
- https://github.com/tillsteinbach/CarConnectivity-connector-skoda
- https://github.com/robinostlund/volkswagencarnet
- App version: 8.12.0, package: cz.skodaauto.myskoda
