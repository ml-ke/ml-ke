# AdultForce Site API Discovery — Jun 2026

## Pivot Chain

Nutaku (locked down) → JS bundle URL extraction → AdultForce (sibling Aylo platform) → unauthenticated Spring Data REST API → 155 brand properties exposed

## Start State

- Nutaku main target: all gateways connection refused, _xd API login protected, OSAPI endpoints 404, web login CSRF+recaptcha locked
- Two user accounts existed but couldn't be used due to CSRF protection

## JS Bundle Discovery

Extracted from Nutaku's `atlasbundle.min.js`:

```bash
curl -sk 'https://www.nutaku.net/js/atlasbundle.min.js?t=48ef7c03' \
  | grep -oP 'https?://[^"'"'"'\s,;)]+' | sort -u
```

Found: `https://www.adultforce.com/api/call_postback/pixel/`

## AdultForce Recon

- AdultForce is the publisher platform for TrafficJunky (Aylo)
- In scope via `*.adultforce.com` (Tier 2 wildcard on TrafficJunky Intigriti program)
- Runs Spring Boot / Java backend
- Also integrates with Salesforce (CSP headers reveal `trafficjunky--adultforce.sandbox.my.salesforce.com`)

## Spring Data REST Endpoints Discovered

| Endpoint | Response | Auth |
|----------|----------|------|
| `GET /api/config` | 200 — 11 config entries (operational messages) | ❌ None |
| `GET /api/config/{id}` | 401 Access Denied | ✅ Required |
| `GET /api/site` | 200 — 177 site entries (first 20 returned) | ❌ None |
| `GET /api/site/{id}` | 200 — Full site detail | ❌ None |
| `POST /api/site` | 403 Access Denied | ✅ Required |
| `PUT /api/site/{id}` | 403 Access Denied | ✅ Required |
| `DELETE /api/site/{id}` | 403 Access Denied | ✅ Required |
| `GET /api/campaign` | 401 Access Denied | ✅ Required |
| `GET /api/offer` | 401 Access Denied | ✅ Required |
| `GET /api/payout` | 401 Access Denied | ✅ Required |

## Site API Data (155 sites accessible by ID enumeration)

Key data per site:
- `id` — Sequential integer (1-199)
- `siteName` — Brand name (Nutaku.net, Brazzers, Mofos, etc.)
- `siteCode` — Internal 2-6 char code (NTKN, bz, MFN)
- `probillerSiteId` — Payment gateway account ID (e.g. 863 for Nutaku.net, 2753 for Brazzers)
- `probillerLegacySiteIds` — Historical billing IDs (e.g. Brazzers has 23 legacy IDs)
- `siteTypeId` — Classification: 1=premium (144 sites), 4=affiliate (18), 8=platform (18)
- `orientationId` — Content type: 1=straight, 2=gay
- `networkId` — Business unit grouping (1=Nutaku, 2=Twistys, 13=gay, 14=Brazzers, 15=Mofos, 16=Babes, 17=Digital Playground)
- `nsfw` — Content rating flag (0 or 1)
- `uaAccount` — Google Analytics ID (e.g. UA-11234847-15)
- `defaultSiteEntryPoint.url` — Landing page URL
- `defaultSiteEntryPoint.desktopThumbnailURL` — S3 bucket path (`//mg-atlas.s3.amazonaws.com/upload-prod/...`)
- `defaultSiteEntryPoint.validUrl` — Site health flag (false for TeenPinkVideos)

## Auth Asymmetry Pattern

Three distinct auth tiers on the same controller:
1. GET list — No auth (200 with data)
2. GET detail — No auth (200 with full data)
3. POST/PUT/DELETE — Auth required (403 Access Denied)

The write operations prove auth middleware exists. The GET operations were missed during implementation. This is a textbook Spring Boot "forgot @PreAuthorize on read" pattern.

## Impact

- 155 Aylo brand properties enumerated with payment billing IDs
- Internal business unit structure revealed (7 network groups)
- AWS S3 bucket disclosed: `mg-atlas.s3.amazonaws.com` (verified public access)
- Google Analytics account IDs exposed
- Site health flags identify broken entry points (potential domain takeover)
- Testing/staging entries exposed (IDs 64, 141, 148)
