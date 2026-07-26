---
name: oauth-oidc-attacks
description: "OAuth 2.0 and OIDC vulnerability testing methodology — state, PKCE, redirect_uri, scope, token handling."
version: 1.1.0
---

# OAuth 2.0 & OIDC Attack Testing

## OIDC Discovery
Always start with:
```bash
curl -s {base}/.well-known/openid-configuration
```
Key things to check:
- `scopes_supported` — unusual scopes = custom functionality
- `grant_types_supported` — `password`, `client_credentials`, `token-exchange` are dangerous
- `response_types_supported` — `token` (implicit) is dangerous
- `token_endpoint_auth_methods_supported`
- `claims_supported`

## Attack Vectors

### 1. Missing state parameter (CSRF)
- OAuth flow without state → attacker can link their account to victim's profile
- Test: Remove state from authorize request, see if server accepts it
- Test: Use a static state value, check if it's validated

### 2. PKCE Downgrade
- Remove `code_challenge` and `code_challenge_method` from authorize request
- If server accepts → PKCE is optional → authorization code interception possible
- Test: Exchange code without `code_verifier`

### 3. Redirect URI Validation
Test these redirect_uri variations:
```
redirect_uri=https://evil.com
redirect_uri=https://target.com.evil.com
redirect_uri=https://target.com/callback/../evil
redirect_uri=https://target.com/callback?url=https://evil.com
redirect_uri=https://target.com/callback#https://evil.com
redirect_uri=https://127.0.0.1
```

### 4. Scope Escalation
- Request broader scopes than needed
- Try: `scope=openid+admin+read:all+write:all`
- Compare scopes in token vs scopes requested

### 5. Token Exchange (dangerous grant)
- If `urn:ietf:params:oauth:grant-type:token-exchange` is supported
- Try exchanging a low-scope token for a high-scope token

### 6. Authorization Code Injection
- Intercept a code, try to use it from a different client/session
- Check if code is single-use

### 7. Implicit Grant Vulnerabilities
- Token in URL fragment → leaks via Referer header
- Check if implicit grant is enabled (`response_types_supported` contains `token`)

### 8. Custom URI Scheme Interception
- Check if redirect_uri uses custom scheme (e.g., `myapp://callback`)
- Malicious apps can register same scheme

## Testing Flow
```bash
# 1. Check OIDC config
curl -s {base}/.well-known/openid-configuration

# 2. Test missing state
curl -s "{base}/authorize?client_id=XXX&response_type=code&redirect_uri=YYY"

# 3. Test PKCE downgrade  
curl -s "{base}/authorize?client_id=XXX&response_type=code&redirect_uri=YYY&scope=openid"

# 4. Test redirect_uri
curl -s "{base}/authorize?client_id=XXX&response_type=code&redirect_uri=https://evil.com"

# 5. Test scope escalation
curl -s "{base}/authorize?client_id=XXX&response_type=code&redirect_uri=YYY&scope=openid+admin"
```

## Identity Platform Testing (Auth0 / Okta)

Auth0 bug bounty programs (like Auth0 by Okta on Bugcrowd, $50k max) require a different approach than generic OAuth testing. The researcher gets a dedicated tenant and Management API access.

### Phase 1: Tenant Discovery
1. Check OIDC config at `https://{tenant}.{domain}/.well-known/openid-configuration`
2. Note: `grant_types_supported`, `scopes_supported`, `token_endpoint`
3. The tenant domain format is usually `{tenant-name}.{env-domain}` (e.g. `bugcrowd-1471.cic-bug-bounty.auth0app.com`)

### Phase 2: Management API Token (CRITICAL — do this FIRST)
```python
import requests
r = requests.post(f"https://{tenant}/oauth/token", json={
    "client_id": "{MGMT_API_CLIENT_ID}",
    "client_secret": "{SECRET}",
    "audience": f"https://{tenant}/api/v2/",
    "grant_type": "client_credentials"
})
token = r.json()['access_token']
```
**Key**: The Management API token gives FULL access to the tenant. The client credentials come from the Auth0 Dashboard → Applications → APIs → Auth0 Management API → Machine to Machine Applications tab.

### Phase 3: Attack Surface Enumeration
```python
# List all applications, connections, users, email templates, actions
GET /api/v2/clients
GET /api/v2/connections
GET /api/v2/users  
GET /api/v2/email-templates
GET /api/v2/actions/actions
GET /api/v2/actions/triggers
```

### Phase 4: Liquid SSTI in Email Templates (HIGHEST PRIORITY — $10k-$50k)
Auth0 uses Liquid (Ruby-based) for email template customization. Attack vectors:

1. **Basic output test**: Create template with `{{ 7 | times: 7 }}` → should render as `49`
2. **Debug tag**: Auth0 has a custom `{% debug %}` tag that dumps ALL template variables
3. **File inclusion**: `{% include 'snippet' %}` — often blocked by Auth0 validation
4. **User metadata injection**: Set user's `user_metadata` to contain Liquid syntax, then reference it in template as `{{ user.user_metadata.field }}`
5. **Escape bypass**: Auth0 recommends `{{ user.nickname | escape }}` for user-provided data — test if escape can be bypassed with filter reordering

**API call to create email template:**
```python
# First configure an email provider (SMTP)
POST /api/v2/emails/provider
{"name": "smtp", "enabled": True, "credentials": {"smtp_host": "...", ...}}

# Then create the template
POST /api/v2/email-templates
{"template": "verify_email", "body": "<html>{% debug %}</html>", 
 "subject": "Test {{ 7 | plus: 3 }}", "syntax": "liquid", "enabled": True}
```

### Phase 5: Auth0 Actions (Node.js Code Execution)
Auth0 Actions run Node.js on Auth0's servers during auth flows. This is a significant attack surface.

1. **Create action**: `POST /api/v2/actions/actions` with Node.js code
2. **Deploy action**: `POST /api/v2/actions/actions/{id}/deploy` 
3. **Test action**: `POST /api/v2/actions/actions/{id}/test` with a realistic event payload
4. **Bind to trigger**: `PATCH /api/v2/actions/triggers/{triggerId}/bindings`
   - Trigger IDs: `post-login`, `credentials-exchange`, `pre-user-registration`, etc.
   - Binding payload format: `{"bindings": [{"ref": {"type": "action_id", "value": "{id}"}, "priority": 1}]}`

**Note**: Auth0's test endpoint (`POST .../test`) accepts a realistic event payload and returns the full execution output including console.log output. This is invaluable for developing and debugging actions before binding them to a real trigger.

### Phase 6: Cross-Tenant Escalation
When you have multiple tenants, test:
1. Can Tenant A's API token access Tenant B's resources?
2. Can Tenant A read/modify Tenant B's email templates?
3. Can Tenant A invite itself to Tenant B?
4. Is there a tenant-switching vulnerability in the dashboard?

Each tenant has its OWN Management API client credentials. Tokens are scoped to individual tenants.

### Phase 7: Enterprise Connection Bypass
Set up enterprise connections (SAML/OIDC) and test:
1. SAML XML Signature Wrapping (see `saml-attack-techniques` skill)
2. NameID/email spoofing in federated auth
3. OIDC claim manipulation

## Reference Files
- `references/azure-b2c-oauth-testing.md` — Azure AD B2C-specific OAuth testing (user flows, PKCE enforcement, implicit grant, claim analysis)
- `references/auth0-okta-attack-surface.md` — Auth0/Okta specific attack surface: Management API, Actions, email templates, cross-tenant testing
