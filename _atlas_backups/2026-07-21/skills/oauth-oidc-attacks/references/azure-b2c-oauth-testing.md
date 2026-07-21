# Azure B2C OAuth Testing Reference

## Overview
Azure AD B2C is a common identity provider in enterprise bug bounty programs. Unlike standard OAuth, B2C uses **user flows** (predefined policies) instead of a single OIDC well-known endpoint.

## OIDC Discovery

Azure B2C requires **user-flow-specific** discovery paths:

```bash
# Standard OIDC (won't work for B2C)
curl -s {base}/.well-known/openid-configuration
# → "The resource you are looking for has been removed"

# B2C user-flow-specific (use the configured flow name)
curl -s "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{user-flow}/v2.0/.well-known/openid-configuration"
```

User flows are visible in the Angular/SPA config as `userFlowSignInSMS`, `userFlowSignInPhonecall`, etc.

## Key B2C Endpoint Pattern

```
Authorization: https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{user-flow}/oauth2/v2.0/authorize
Token:          https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{user-flow}/oauth2/v2.0/token
JWKS:           https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{user-flow}/discovery/v2.0/keys
Logout:         https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{user-flow}/oauth2/v2.0/logout
```

## What to Check in the OIDC Config

### Dangerous Signs (likely vulnerable)
- `response_types_supported` contains `token` or `id_token token` (implicit grant) — enables token-in-URL-fragment leakage
- `token_endpoint_auth_methods_supported` is empty or only `none`
- `scopes_supported` contains more than just `openid` (like `admin`, `read:all`)

### Good Signs (well-secured)
- `response_types_supported` only `code` (authorization code only)
- `claims_supported` shows custom extension claims like `extension_PersonCode`, `extension_Tenant` (these go into tokens)
- PKCE enforced via `code_challenge` requirement

## Testing Checklist for Azure B2C

### 1. PKCE Enforcement
```bash
# Try authorization code flow WITHOUT code_challenge
curl -sv "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{flow}/oauth2/v2.0/authorize?client_id=XXX&response_type=code&redirect_uri=YYY&scope=openid&state=test"

# Expected if PKCE is enforced:
# "AADB2C99059: The supplied request must present a code_challenge"
# Expected if PKCE is optional (VULNERABLE):
# Returns a 302 with authorization code
```

### 2. Redirect URI Validation
```bash
curl -sv "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{flow}/oauth2/v2.0/authorize?client_id=XXX&response_type=code&redirect_uri=https://evil.com&scope=openid"

# Expected if enforced: "redirect_uri_mismatch" error
# Expected if vulnerable: 302 redirect to evil.com with code
```

### 3. Implicit Grant (token response type)
```bash
curl -sv "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{flow}/oauth2/v2.0/authorize?client_id=XXX&response_type=token&redirect_uri=YYY&scope=openid"

# Azure B2C typically requires a resource scope for implicit:
# "AADB2C90055: The scope 'openid' must specify a resource"
```

### 4. Scope Escalation
```bash
curl -s "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{flow}/oauth2/v2.0/authorize?client_id=XXX&response_type=code&redirect_uri=YYY&scope=openid+admin+profile+email"
```

### 5. User Flow Enumeration
Check if OTHER user flows exist by trying common names:
```bash
for flow in B2C_1_signin B2C_1_signup B2C_1_signin_sms B2C_1_signin_phonecall B2C_1_profile B2C_1_passwordreset B2C_1_signup_signin; do
  code=$(curl -sk -o /dev/null -w "%{http_code}" "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/$flow/v2.0/.well-known/openid-configuration")
  echo "$flow → $code"
done
```

### 6. Client Credentials Grant
Check if `client_credentials` grant is supported:
```bash
curl -s "https://{tenant}.b2clogin.com/{tenant}.onmicrosoft.com/{flow}/oauth2/v2.0/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=XXX" \
  -d "client_secret=XXX" \
  -d "scope=https://{tenant}.onmicrosoft.com/{api}/.default"
```

## Claims to Watch For

Custom claims in B2C tokens often leak internal identifiers:

| Claim | Meaning |
|---|---|
| `extension_PersonCode` | Internal person identifier |
| `extension_Tenant` | Tenant/multi-tenant indicator |
| `extension_WebUser` | Web user reference |
| `extension_PersonCodeBdslux` | Luxembourg-specific person code |
| `extension_WebUserBdslux` | Luxembourg web user |

If these appear in the OIDC config's `claims_supported`, they're embedded in tokens. An IDOR on these identifiers could be high impact.

## Real-World Example: Delen Private Bank

Delen uses Azure B2C with:
- Tenant: `delenb2c.onmicrosoft.com`
- B2C Login: `b2clogin.delen.bank`
- User flows: `B2C_1_signin_sms`, `B2C_1_signin_phonecall`
- Client IDs per region (CH: 4001cae1..., LU: c5e1354b..., NL: 230490c5...)
- PKCE: ENFORCED ✓
- Redirect validation: ENFORCED ✓
- Implicit grant: enabled but requires resource scope (low risk)
- Custom claims: PersonCode, Tenant, WebUser, PersonCodeBdslux, WebUserBdslux
