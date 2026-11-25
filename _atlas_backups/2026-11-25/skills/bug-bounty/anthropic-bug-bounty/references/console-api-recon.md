# Console API Reconnaissance (platform.claude.com)

Date: 2026-05-31
Auth method: Magic link (h0d4r1@wearehackerone.com)
Account type: Individual (no credits purchased)

## Auth Flow

1. POST email → magic link sent
2. Magic link URL format: `https://platform.claude.com/magic-link#<token>:<base64-email>`
3. React SPA reads hash fragment from URL, exchanges for session cookie
4. Session cookie set on `.claude.com` domain
5. Paywall on `/create/credits` blocks most UI features until credits purchased

Note: Browser automation may need JS SPA to process hash fragment. Direct navigation to magic link URL works if SPA loads properly.

## Framework

- Next.js SPA (React)
- CSS framework: `cds-*` classes (internal design system)
- Analytics: `antalytics` (internal)
- Routes are client-side rendered; server returns minimal shell HTML

## Route Map

| Route | Status | Purpose |
|-------|--------|---------|
| `/` | 200 | Home / Dashboard (paywalled) |
| `/dashboard` | 200 | Dashboard (paywalled) |
| `/login` | 200 | Login page |
| `/org-discovery` | 200 | Join/create org (authenticated) |
| `/create/credits` | 200 | Credit purchase (paywall) |
| `/settings/organization` | 200 | Org settings (redirect from /settings, /account) |
| `/usage` | 200 | Usage monitoring (paywalled) |
| `/billing` | 404 | Not found |
| `/workspace` | 404 | Not found |
| `/projects` | 404 | Not found |
| `/api-keys` | 404 | Not found |
| `/settings/apikeys` | 404 | Not found |

## API Endpoint Map

All endpoints use `/api/` prefix (not `/api/v1/`). Auth via session cookies (credentials: 'include').

### Working Endpoints
```
GET /api/organizations
  → 200 [{"id": <int>, "uuid": "<uuid>", "name": "<name>", "settings": {...}}]

GET /api/organizations/{uuid}
  → 200 {"id": <int>, "uuid": "<uuid>", "name": "<name>", ...}
```

### Permission-Gated Endpoints (403 for individual accounts)
```
GET /api/organizations/{uuid}/members       → 403 "Missing permissions"
GET /api/organizations/{uuid}/usage         → 403 "Missing permissions"
GET /api/organizations/{uuid}/invites       → 403 "Missing permissions"
```

### Not Found (endpoint doesn't exist at this path)
```
GET /api/organizations/{uuid}/api_keys      → 404
GET /api/organizations/{uuid}/workspaces    → 404
GET /api/organizations/{uuid}/billing       → 404
GET /api/v1/*                               → 404 (wrong prefix)
GET /api/organizations/{id}/members         → 400 (numeric ID, needs UUID)
```

### IDOR Testing

Tested: `00000000-0000-0000-0000-000000000000` → 404 with `error_visibility: user_facing`
Different org UUIDs → 404 (no data leak)
No sequential ID enumeration possible (UUID-based)

### XSS Testing

Tested: org name containing `'0"/><img scr=x><a href=http://evil.com>click</a>`
Result: Rendered as escaped text (`&gt;`/`&lt;`), not executable HTML
Someone had already tested this vector — payload was already in the DB.

## Recommendations for Further Testing

1. **Organization/Enterprise plan**: Create an org account (or join existing one) to access:
   - API key creation/management endpoints
   - Workspace CRUD (Admin API: create, get, list, update, archive)
   - Member role management and invitation flow (IDOR on invites)
   - Billing operations
2. **Workbench**: Test the in-browser API Workbench for CSRF or XSS
3. **CLI vs Web**: Check if CLI-based API key creation bypasses web UI restrictions
4. **OAuth flow**: Test Workload Identity Federation for token leakage
5. **Support portal**: Check support.anthropic.com for Zendesk-specific vulns
