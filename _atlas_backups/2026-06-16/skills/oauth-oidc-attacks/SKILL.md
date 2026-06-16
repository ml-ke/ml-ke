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

## Reference Files
- `references/azure-b2c-oauth-testing.md` — Azure AD B2C-specific OAuth testing (user flows, PKCE enforcement, implicit grant, claim analysis)
