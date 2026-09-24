# Auth0 by Okta — Attack Vectors & Recon Notes

## Bugcrowd Program Focus (April 2026 multiplier)
- **Brand Customization (Emails)** — Liquid template injection in email templates
- **Management API endpoints** — API privilege escalation via `/api/v2/`
- **Authentication bypass in enterprise connections** — SAML/OIDC SSO bypass

## Infrastructure

### Standard Auth0 Tenant OIDC
```
https://{tenant}.{region}.auth0.com/.well-known/openid-configuration
```

Discovered tenants:
| Tenant | Region | Status |
|--------|--------|--------|
| `auth0.us.auth0.com` | US | 200 — full OIDC config available |
| `auth0.eu.auth0.com` | EU | 200 |
| `auth0.au.auth0.com` | AU | 200 |
| `bugcrowd.auth0.com` | US (Bugcrowd) | 200 — Bugcrowd itself uses Auth0 |

### Key Endpoints
| Endpoint | Notes |
|----------|-------|
| `/.well-known/openid-configuration` | Full OIDC metadata |
| `/.well-known/jwks.json` | RSA256 signing keys |
| `/oauth/token` | Token exchange — requires valid client credentials |
| `/oauth/revoke` | Token revocation |
| `/authorize` | Authorization — accepts `connection` param for specific IdP |
| `/co/authenticate` | **Cross-Origin Authentication** — validates web origins |
| `/oidc/register` | Dynamic client registration — **disabled** (returns 400) |
| `/dbconnections/signup` | Database connection signup — returns "connection not found" if connection doesn't exist |
| `/api/v2/...` | Management API — needs valid API token |
| `/samlp/metadata` | SAML metadata — 404 on this tenant |
| `/userinfo` | OIDC userinfo endpoint |
| `/mfa/challenge` | MFA challenge endpoint |

## Attack Vectors

### 1. Cross-Origin Authentication (`/co/authenticate`)
The `/co/authenticate` endpoint enables cross-origin login flows. It validates the `Origin` header against registered web origins:
```bash
curl -X POST "https://{tenant}.auth0.com/co/authenticate" \
  -H "Content-Type: application/json" \
  -H "Origin: https://example.com" \
  -d '{"client_id":"{valid_client_id}","credential_type":"http://auth0.com/oauth/grant-type/password-realm","realm":"Username-Password-Authentication","username":"user","password":"pass"}'
```
Error response reveals validation behavior: `"WEB ORIGINS"` check.

Known vulnerability: Sentor Security disclosed an Auth0 authentication bypass via `/co/authenticate` that let attackers impersonate any user.

### 2. Enterprise Connection Bypass (SAML/OIDC)
The program specifically highlights **authentication bypass in enterprise connections**. Key resources:
- **PortSwigger** "The Fragile Lock: Novel Bypasses For SAML Authentication" (Dec 2025) — full SAML bypass via attribute pollution, namespace confusion, parser inconsistencies
- **CVE-2026-47201** — authentik SAML XML Signature Wrapping authentication bypass
- **CVE-2026-44748** — SAP SAML XML Signature Wrapping (CVSS 9.9)
- **SSO Bypass techniques**: parser differentials, XML signature wrapping, JWT alg confusion

Test enterprise connections (SAML, OIDC, Google Workspace, Azure AD) for:
- SAML response manipulation (signature stripping, XML wrapping)
- JWT alg confusion (RS256→HS256 with public key)
- OIDC parser differentials between Auth0 and the IdP

### 3. Brand Customization — Email Templates (Liquid SSTI)
Auth0 email templates use **Liquid syntax**. Test for server-side template injection:
```liquid
{{user.email}}
{{7*7}}
{% assign x = "hello" %}{{x}}
```
The Management API endpoints for email templates:
```
POST /api/v2/email-templates          — Create template
GET  /api/v2/email-templates/{name}   — Get template
PATCH /api/v2/email-templates/{name}  — Update template
```

### 4. Management API
Accessible at `/api/v2/*` on the tenant domain. Requires a Management API token (different from user tokens). Key endpoints:
```
GET  /api/v2/clients             — List applications (401 without token)
GET  /api/v2/tenants/settings    — Tenant settings (401 without token)
POST /api/v2/email-templates     — Create email templates
PATCH /api/v2/sessions/{id}      — Modify session metadata
```

To get a Management API token, you need:
1. An Auth0 tenant where you're an admin, OR
2. Valid client credentials with `audience=https://{tenant}.auth0.com/api/v2/`

### 5. OIDC Configuration Weaknesses
- Check if `token_endpoint_auth_method` allows `none` (unsigned client auth)
- Check if `grant_types` includes `implicit` (deprecated, less secure)
- Check if `response_types` includes `token id_token` (implicit flow — token in URL fragment)
- Try the `connection` parameter in `/authorize` to force specific IdP bypass

### 6. Deprecated Features (Nov 2026 EOL)
Auth0 **Rules and Hooks** are being deprecated November 18, 2026. After EOL they stop executing. Legacy code in these features is often less audited — test for:
- Insecure data handling in Rules
- Auth bypass via deprecated Hook endpoints

## Recon Checklist
- [ ] Get Bugcrowd program scope (Auth0 by Okta engagement)
- [ ] Determine which Auth0 tenants are in scope
- [ ] Check for test credentials or sandbox tenant access
- [ ] Probe `/co/authenticate` with valid web origins from scope
- [ ] Search for Management API token leaks in public repos
- [ ] Test email template Liquid injection (Brand Customization)
- [ ] Test enterprise connection bypass (SAML/OIDC)
- [ ] Test OIDC configuration for weak grant types
