# OIDC Scope Enumeration for Hidden API Discovery

## Technique

When an OIDC provider exposes `/.well-known/openid-configuration`, the `scopes_supported` array often lists custom scopes that directly name undocumented backend services. Each custom scope is an OAuth resource indicator — an explicit declaration that a service exists.

## Workflow

```bash
# 1. Fetch OIDC config and extract custom scopes
curl -s "https://identity.target.com/.well-known/openid-configuration" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(*d.get('scopes_supported',[]), sep='\n')" | \
  grep -vE '^(openid|profile|email|phone|address|offline_access)$' | sort

# 2. Also check grant_types_supported for token-exchange
curl -s "https://identity.target.com/.well-known/openid-configuration" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('grant_types_supported',[]),indent=2))"
```

## Scope Naming Patterns

| Pattern | What It Reveals | Example |
|---------|----------------|---------|
| `service-name:action` | REST service + permission | `vsn-assistant-api:chat` |
| `service-name:mcp` | MCP (Model Context Protocol) server | `vsn-assistant-mcp:mcp` |
| `product.mcp:execute` | AI tool-calling endpoint | `visma.net.erp.ai.mcp:execute` |
| `service:internal` | Internal-only admin API | `vsn-assistant-api:internal` |
| `service:inspect` | Debug/inspection endpoint | `vsn-assistant-api:inspect` |
| `agent.*registry:*` | Agent registry | `agent_builder.registry:read` |
| `ai-agent:vectors:admin` | Vector DB admin API | `eplus-ai-agent:vectors:admin` |
| `*ApiWellKnownEndpoint:update` | Internal config registry | `9441601398ApiWellKnownEndpoint:update` |
| `grant_type=token-exchange` | Privilege escalation vector | In grant_types_supported |

## Attack Surface

1. **AI Assistant APIs** (`*-assistant-api:*`) — Chat bots, probe for `/api/chat`, SignalR hubs, session history
2. **MCP Servers** (`*:mcp`) — MCP endpoints: try `POST {"jsonrpc":"2.0","method":"tools/list","id":1}`
3. **Inspect/Internal** (`*:inspect`, `*:internal`) — Debug endpoints, may leak config
4. **Agent registries** (`agent*.registry:*`) — May allow installing/modifying AI agents
5. **Token exchange** (`token-exchange` grant) — Scope escalation via token exchange

## Cross-Reference with JS Bundles

When you find a suspicious scope name, search the target's web app JS bundles for that service name. The bundle often contains the actual backend URL.

Example: Finding `vsit-aiassistant-stg.azurewebsites.net` in a React SPA bundle confirmed the AI Assistant backend after discovering `vsn-assistant-api:chat` in the OIDC config.

## Worked Example: Visma AI Assistant

1. Probed `connect.identity.stagaws.visma.com/.well-known/openid-configuration`
2. Found 100+ custom scopes including `vsn-assistant-api:chat`, `vsn-assistant-mcp:mcp`, `vsn-assistant-api:inspect`, `vsn-assistant-api:internal`, `agent_builder.registry:read`
3. Probed `aiassistant.stage.vismaonline.com` — React SPA on Azure Static Web Apps (HTTP 200)
4. Extracted JS bundle → found backend URL `vsit-aiassistant-stg.azurewebsites.net`
5. Backend revealed: Azure Functions, SignalR hub, JWT auth, rate limiting
