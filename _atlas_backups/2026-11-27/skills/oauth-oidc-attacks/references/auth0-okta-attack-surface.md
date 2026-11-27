# Auth0 / Okta Bug Bounty Attack Surface

## Source
Derived from hands-on testing against Auth0 by Okta's Bugcrowd researcher environment (June 2026).
Tenant: `bugcrowd-1471.cic-bug-bounty.auth0app.com`
Payout: Up to $50,000 (P1), $10,000-$50,000 on Tier 1

## Tenant Structure
- Standard Auth0 tenant per researcher
- Dashboard: `manage.cic-bug-bounty.auth0app.com`
- Auth0 domain: `{tenant-name}.cic-bug-bounty.auth0app.com`
- Management API audience: `https://{tenant}.cic-bug-bounty.auth0app.com/api/v2/`
- Each tenant has 3 users by default (can invite each other)

## Key Endpoints Discovered

### OIDC (Standard)
- `/.well-known/openid-configuration` — Full OIDC config
- `/oauth/token` — Token endpoint (client_credentials, password, auth_code, device_code)
- `/oauth/authorize` — Authorization endpoint
- `/.well-known/jwks.json` — JWKS keys (RS256)
- `/userinfo` — UserInfo endpoint

### Management API v2
- `/api/v2/tenants/settings` — Tenant configuration
- `/api/v2/clients` — Applications list
- `/api/v2/connections` — Identity providers (Username-Password, Google, etc.)
- `/api/v2/users` — User CRUD
- `/api/v2/email-templates` — Email template CRUD (Liquid syntax)
- `/api/v2/emails/provider` — SMTP provider config
- `/api/v2/actions/actions` — Actions CRUD (Node.js code execution)
- `/api/v2/actions/actions/{id}/deploy` — Deploy an action
- `/api/v2/actions/actions/{id}/test` — Test an action with custom payload
- `/api/v2/actions/triggers` — List available triggers
- `/api/v2/actions/triggers/{triggerId}/bindings` — Get/update trigger bindings
- `/api/v2/logs` — Tenant logs
- `/api/v2/grants` — OAuth grants
- `/api/v2/roles` — RBAC roles
- `/api/v2/branding` — Branding settings

### Available Grant Types (from OIDC config)
- client_credentials, authorization_code, refresh_token, password, implicit
- device_code, token-exchange, password-realm, mfa-oob, mfa-otp, mfa-recovery-code

### Available Triggers (for Actions binding)
- post-login (v1/v2/v3), credentials-exchange (v1/v2)
- pre-user-registration (v1/v2), post-user-registration (v1/v2)
- post-change-password (v1/v2), send-phone-message (v1/v2)
- custom-phone-provider, custom-email-provider
- password-reset-post-challenge, custom-token-exchange
- event-stream, password-hash-migration
- **mcp-tool** (v1) — Auth0 has an MCP tool trigger!

## Liquid SSTI in Email Templates

### Supported Liquid Tags
| Category | Tags |
|---|---|
| Iteration | for, cycle, tablerow |
| Control flow | if, unless, elsif, else, case |
| Variable | assign, increment, decrement |
| File | include, layout ({% include %} is blocked by Auth0 validation) |
| Language | raw, comment |
| **Custom** | **debug** — Auth0 custom tag, dumps ALL template variables |

### Supported Liquid Filters
| Category | Filters |
|---|---|
| Math | plus, minus, modulo, times, floor, ceil, round, divided_by, abs |
| String | append, prepend, capitalize, upcase, downcase, strip, split, replace |
| HTML/URI | escape, escape_once, url_encode, url_decode, strip_html |
| Array | slice, map, sort, sort_natural, first, last, join, reverse, size |
| Date | date |
| Misc | default |

### Variables Available in Templates
- user.email, user.email_verified, user.picture (untrusted)
- user.nickname, user.given_name, user.family_name, user.name (untrusted)
- user.app_metadata, user.user_metadata (untrusted if user-provided)
- application.name, application.client_metadata
- connection.name
- tenant, friendly_name, support_email, support_url
- organization.id, organization.display_name, organization.metadata
- custom_domain.domain, custom_domain.domain_metadata

### SSTI Testing Flow
1. Create SMTP provider (needed before email templates)
2. Create email template with Liquid syntax
3. Create a user with malicious user_metadata containing Liquid payloads
4. Update template to reference {{ user.user_metadata.field }}
5. Trigger verification email
6. Check if Liquid is processed (SSTI confirmed) or escaped (properly sanitized)

**The {% debug %} tag is critical** — it dumps ALL template variables. Include it in test templates.

## Auth0 Actions (Node.js Code Execution)

### Action Structure
exports.onExecutePostLogin = async (event, api) => {
  // event.user.email, event.user.user_metadata, etc.
  // event.request, event.connection, event.tenant, event.client
  // event.secrets, event.organization
  // api.access.deny('reason')
  // api.user.setAppMetadata('key', 'value')
};

### Testing with the Test Endpoint
The POST /api/v2/actions/actions/{id}/test endpoint accepts a realistic event payload:
Returns action_duration_ms and full output including console.log.

### Trigger Binding
Dashboard method (Actions > Library > select action > Add to Flow > Login > Apply) is more reliable than the API.

## Rate Limits & Rules
- Max 5 requests/second (from program brief)
- No automated scanning tools
- No disclosure allowed (private program)
- No DoS/DDoS
- Chaining is welcomed
- If you get server access: STOP, report, don't pivot

## Cross-Tenant Testing
- Each tenant has independent Management API credentials
- Tokens are scoped to the source tenant
- Cross-tenant test: can Tenant A's dashboard user access Tenant B's settings?
- Cross-tenant template test: can Tenant A read/modify Tenant B's email templates?

## Out of Scope for Auth0 Program
- GitHub Actions vulnerabilities
- Customize Login Page XSS
- Race conditions bypassing limits
- Incomplete PoCs, theoretical vulnerabilities
- Clickjacking, CSRF on anonymous forms
- Rate limiting, username enumeration
- Missing security headers, SSL/TLS issues
- SDK issues relying on incorrect implementation
