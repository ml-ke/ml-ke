# TrafficJunky OpenAPI Spec Analysis (June 2026)

## Target
api.trafficjunky.com — Aylo/TrafficJunky bug bounty, Intigriti

## What Was Found
Full OpenAPI 3.0 specs publicly accessible at:
- `/docs/api-docs.json` — V1 spec (119KB, 25 endpoints)
- `/docs/v2/api-v2-docs.json` — V2 spec (735KB, 31 endpoints)

The specs are served by Scalar API Reference UI v1.55.3.

## V1 Endpoints (25)
| Category | Endpoints | Notes |
|----------|-----------|-------|
| Campaigns | GET/PUT `/api/campaigns/{id}`, GET list, POST/DELETE timetargets | campaignId in path — IDOR vector |
| Ads | GET `/api/ads/{id}/one`, PATCH `/api/ads/{id}`, PUT weights/pauses | adId in path |
| Bids | GET `/api/bids/{id}/one`/stats, PUT `/api/bids/{id}/set` | bid amount manipulation |
| Member | GET/PUT `/api/member`, GET `/api/member/fund/modifications` | Financial data |
| Spots | GET `/api/spots/{campaignId}`, GET `/api/spots/{id}/revenue` | Revenue data |

## V2 Endpoints (31)
| Category | Endpoints | Notes |
|----------|-----------|-------|
| Campaigns | GET/POST `/campaigns`, GET/PATCH `/campaigns/{id}`, POST clone | Full CRUD |
| Ad Configs | GET/POST campaigns/{id}/ad-configurations, PATCH by ID | Under campaign scope |
| Creatives | GET/POST `/creatives`, GET/PATCH `/creatives/{id}` | Ad creative management |
| Collections | 23 read-only endpoints | Countries, cities, browsers, languages, etc. — no auth needed? |

## Auth
- Both V1 and V2 use BearerAuth (http type)
- No auth flow documented in the OpenAPI specs
- All endpoints return 401 "Unauthorized" without a Bearer token
- Token generation is via the TrafficJunky website dashboard (not an API)

## Blockers
- No TrafficJunky account available — email verification not arriving
- Nutaku Bearer tokens DO NOT work on TJ API (separate auth system)
- Social login (Google, LinkedIn, PayPal) available on login page — potential bypass for email verification

## Key IDOR Vectors (if/when account access is obtained)
- `/api/campaigns/{campaignId}` — campaign enumeration
- `/api/bids/{bidId}/set` — bid manipulation (financial impact)
- `/api/member/fund/modifications` — fund history of other members
- V2 `/campaigns/{id}/ad-configurations` — cross-campaign ad access
