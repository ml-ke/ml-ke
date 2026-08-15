# Visma AI Assistant Reconnaissance — Worked Example (Jun 2026)

## Target: Visma Bug Bounty (Intigriti)
- **Program**: Public, €100–€7,500
- **In-scope**: AI Assistant (`aiassistant.stage.vismaonline.com`), eAccounting, AutoInvoice, Visma Scanner, Dinero, Visma Connect, Developer Portal
- **New feature**: AI Assistant added to scope Jan 2026 — fresh, less-audited surface

## Recon Pipeline

### Step 1: OIDC Scope Recon — Find Hidden API Surfaces
The Visma Connect IdentityServer at `connect.identity.stagaws.visma.com` exposes `.well-known/openid-configuration` publicly. The scopes list reveals AI-related resources:

```
vsn-assistant-api:chat          ← Chat API
vsn-assistant-api:internal      ← Internal API
vsn-assistant-api:mobile-chat   ← Mobile chat
vsn-assistant-api:inspect       ← Inspect endpoint
vsn-assistant-mcp:mcp           ← MCP (Model Context Protocol)
vsn-assistant-mcp-local:mcp     ← Local MCP
visma.net.erp.ai.mcp:execute    ← AI MCP execute
agent_builder.registry:read     ← Agent builder
agent_builder.registry:install
VFRAiGw:mcp:tools:read          ← AI Gateway MCP
VFRAiGw:mcp:tools:execute
eplus-ai-agent:vectors:admin    ← AI agent admin
SMARTSKILLMCP_DEV:*             ← Smart Skills MCP dev
```

**Dangerous grant types**: `token-exchange`, `client_credentials`, `delegation`, `urn:ietf:params:oauth:grant-type:device_code`

**Key technique**: OIDC scope strings with naming conventions like `vsn-assistant-api:*` and `*-mcp:mcp` directly reveal the resource names. Search for these on GitHub/docs to find the client IDs and API docs.

### Step 2: SPA JS Bundle Extraction — Find the Real Backend
The AI Assistant frontend is a React SPA on Azure Static Web Apps. All `/api/*` paths return 404 from the static host. The real backend is embedded in the JS bundle.

**Extraction command**:
```bash
curl -sk "https://aiassistant.stage.vismaonline.com/static/js/main.aa6127e9.js" \
  | grep -oP 'https?://[a-zA-Z0-9._-]+\.azurewebsites\.net[^\s,;)"'']*'
```

**Result**: `https://vsit-aiassistant-stg.azurewebsites.net/`

### Step 3: Backend Probing Results
| Test | Result | Signal |
|------|--------|--------|
| `GET /` | 200 — "AI Assistant" | Backend alive |
| `POST /api/chat` | 405 | Route exists (method not allowed) |
| `GET /api/agents` | 404 | Different routing |
| `GET /api/health` | 404 | Not exposed |
| Auth: `Bearer test` | "Invalid token format." | Validates JWT format |
| `OPTIONS /api/agents` | 204, CORS: `aiassistant.stage.vismaonline.com` | Strict CORS |
| `x-rate-limit-limit` | 1m (200 req/min) | Rate limited |
| `/messageRelayHub` (SignalR) | 401 | Real-time hub exists but needs auth |
| CSP | `frame-ancestors 'none'` | No clickjacking |
| HSTS | `max-age=31536000` | Good hardening |

### Step 4: Visma Connect Auth Flow
- Production IDP: `identity.vismaonline.com`
- Staging IDP: `connect.identity.stagaws.visma.com`
- Visma Online IDP: `identity.stage.vismaonline.com`
- Grant types: authorization_code, client_credentials, refresh_token, **token-exchange**, device_code, CIBA, delegation
- Token auth: client_secret_basic, client_secret_post
- Scopes for eAccounting: `ea:api`, `ea:sales`, `ea:accounting`, `offline_access`, etc.

### Step 5: Developer Documentation Surface
| Endpoint | Content |
|----------|---------|
| `developer.vismaonline.com/llms.txt` | Full API doc index in Markdown |
| `developer.vismaonline.com/docs/authentication.md` | OAuth2 auth guide |
| `developer.vismaonline.com/docs/environments.md` | Sandbox URLs |
| `developer.vismaonline.com/docs/spiris-mcp-server.md` | MCP server docs (beta) |
| `developer.vismaonline.com/docs/sandbox-faq.md` | Sandbox FAQ |
| `selfservice.developer.vismaonline.com/` | **Sandbox registration portal** |

### Step 6: Other Visma API Endpoints
| Endpoint | HTTP | Notes |
|----------|------|-------|
| `eaccounting.vismaonline.com` | 302 → /Login | eAccounting app |
| `eaccountingapi.vismaonline.com` | 404 | API (needs auth route) |
| `vlsapi.vismaonline.com` | 200 | Payroll API |
| `api.stage.vismaonline.com` | 200 | "VON Api" (empty shell) |
| `advisorapi.stage.vismaonline.com` | 302 | Advisor API docs |
| `mcp.spiris.se/mcp` | 401 | MCP server (unauthorized) |
| `snowplow.visma.com` | 200 | Analytics |

### Step 7: Registration Method
To get OAuth tokens, you need a **developer sandbox account**:
1. Go to `https://selfservice.developer.vismaonline.com/` (React SPA with Turnstile CAPTCHA)
2. Fill registration form → receives client_id + client_secret + sandbox company credentials via email
3. Use these to get tokens from `identity.vismaonline.com/connect/token`
4. Training code for Visma bug bounty student accounts: `wr0d4` (changed from `g004t`)

## Attack Vectors to Pursue (with credentials)

1. **AI Assistant Prompt Injection**: Get a valid token with `vsn-assistant-api:chat` scope, then send prompts designed to leak system instructions, training data, or backend configuration.

2. **Token Exchange Escalation**: The `token-exchange` grant type may allow escalating from a `vsn-assistant-api:chat` token to `vsn-assistant-api:internal` or `vsn-assistant-api:inspect` scopes.

3. **MCP Server Unauthorized Access**: The Spiris MCP server at `mcp.spiris.se/mcp` returns 401 but the `client_credentials` grant with the right client may bypass user-level auth entirely.

4. **Cross-Tenant IDOR in eAccounting**: Sandbox allows creating multiple companies. Test if Company A can access Company B's invoices, customers, or vouchers via the eAccounting API.

5. **CORS Misconfiguration**: The AI Assistant backend allows `aiassistant.stage.vismaonline.com` only. Test if other Visma subdomains (`eaccounting.vismaonline.com`, `admin.stage.vismaonline.com`) can be used to bypass this CORS policy.

6. **SignalR Hub Auth Bypass**: The `/messageRelayHub` endpoint at `vsit-aiassistant-stg.azurewebsites.net` returns 401 without auth. Test if specific SignalR methods (chat, escalate) bypass the auth check differently.

## Subdomain Discovery (Visma Online Ecosystem)
```bash
# Live vismaonline.com subdomains:
aiassistant    → 200  (AI Assistant - in scope!)
api            → 200  (VON Api)
admin          → 302
eaccounting    → 302  (eAccounting app)
vlsapi         → 200  (Payroll API)
advisor        → 302
advisorapi     → 302
identity       → 302  (OIDC production)
developer      → 302  (Developer portal)
```

## References
- Visma Bug Bounty Program: `https://app.intigriti.com/programs/visma/visma/detail`
- Visma OIDC Config: `https://connect.identity.stagaws.visma.com/.well-known/openid-configuration`
- Developer Docs: `https://developer.vismaonline.com/llms.txt`
- Sandbox Registration: `https://selfservice.developer.vismaonline.com/`
