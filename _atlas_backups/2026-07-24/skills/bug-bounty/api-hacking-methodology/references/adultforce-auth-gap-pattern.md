# AdultForce Auth Gap Pattern — List vs Detail Endpoint

## Target
www.adultforce.com (Aylo/TrafficJunky bug bounty, Intigriti)

## Finding
`GET /api/config` is **public** — returns a paginated list of all configuration entities (200 OK, no auth required).
`GET /api/config/{id}` is **protected** — returns 401 "Access Denied".

## The Pattern
List endpoint had no auth check — returns summaries of all 11 config items.
Detail endpoint requires authentication.
Contrast with `/api/health` (also public) and `/api/docs` (403 blocked).

## Data Exposed (11 config items, names only)
- Maintenance Mode (x2)
- Post-back Service - Downtime
- Transactions Delay
- Stats Delay
- Traffic Delay (x2)
- Global Message - Admin/Client/Public (x3)
- Disable Video Library (active=1)

## Tech Stack
- Express.js backend + Angular SPA frontend
- Error format: `{"status":403,"errorMessage":"Access Denied","uri":"/api/config/1"}`
- Rate limit headers: `x-ratelimit-time-in-seconds: 60`, `x-ratelimit-remaining: 59`
- `/api/auth`, `/api/login`, `/api/register` all return 403 (exist, need session)

## Cross-Reference
See api-hacking-methodology section 2.5 "Endpoint Auth Comparison Testing"
