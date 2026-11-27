# MCP (Model Context Protocol) Server Reconnaissance

MCP is an emerging open standard (Anthropic) that lets AI assistants connect to external tools and data sources through a JSON-RPC 2.0 interface. MCP servers are becoming common in enterprise SaaS platforms — they expose APIs for creating invoices, managing customers, registering payments — essentially full CRUD over an AI-friendly protocol.

## Why MCP Matters for Bug Bounty

- **New attack surface**: Most researchers aren't probing MCP endpoints yet
- **Different auth model**: MCP might have weaker auth than the same company's REST API
- **Scope bypass**: A read-only REST token might grant write access through MCP
- **Tenant isolation gaps**: MCP tools might not properly scope to the authenticated user's company

## Detection

### 1. OIDC Scope Mining

Look for scopes containing `mcp`, `agent`, `tool`, or `assistant` in the OIDC config:

```bash
curl -sk "https://identity.target.com/.well-known/openid-configuration" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(s) for s in d.get('scopes_supported',[]) if any(x in s.lower() for x in ['mcp','agent','tool','assistant'])]"
```

Real examples from Visma Connect (stagaws):
- `vsn-assistant-mcp:mcp` — AI assistant MCP
- `vsn-assistant-mcp-local:mcp` — local dev MCP
- `visma.net.erp.ai.mcp:execute` — ERP AI MCP
- `VFRAiGw:mcp:tools:read` — VFR AI Gateway read tools
- `VFRAiGw:mcp:tools:execute` — VFR AI Gateway execute tools
- `SMARTSKILLMCP_DEV:*` — Smart Skills dev MCPs
- `visma.index.mcp:execute` — Index MCP

### 2. JS Bundle Search

Search for `mcp`, `agent_builder`, or `tool_registry` in SPA JS bundles:

```bash
curl -sk "https://target.com/static/js/main.*.js" | grep -oP '["'\'']https?://[^"'\'']*mcp[^"'\'']*["'\'']' | sort -u
```

### 3. Documentation Search

Many companies document MCP endpoints in their developer portals:

```bash
# Check llms.txt (AI agent documentation index)
curl -sk "https://developer.target.com/llms.txt" | grep -i mcp

# Check for MCP-specific docs
curl -sk "https://developer.target.com/docs/spiris-mcp-server.md"
```

### 4. Standard Path Probing

```bash
for path in "/mcp" "/api/mcp" "/v1/mcp" "/mcpserver" "/api/tools" "/agent"; do
  code=$(curl -sk -o /dev/null -w "%{http_code}" "https://target.com${path}")
  echo "${path} -> ${code}"
done
```

## Probing the MCP Endpoint

MCP uses JSON-RPC 2.0. Always try list methods first to discover available tools:

```bash
# Without auth (to check exposure)
curl -sk "https://mcp.target.com/mcp" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Initialize handshake
curl -sk "https://mcp.target.com/mcp" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"0.1.0","capabilities":{}},"id":1}'

# List resources (data endpoints)
curl -sk "https://mcp.target.com/mcp" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"resources/list","id":1}'
```

## Auth Testing on MCP

MCP servers typically use OAuth2 tokens. Test if the auth is properly scoped:

```bash
# Test with a read-only token
curl -sk "https://mcp.target.com/mcp" \
  -H "Authorization: Bearer $READ_ONLY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# If read-only token can see WRITE tools -> potential scope bypass
# Try calling a write tool with the read-only token
curl -sk "https://mcp.target.com/mcp" \
  -H "Authorization: Bearer $READ_ONLY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_invoice","arguments":{"customerId":"123"}},"id":1}'
```

## Attack Vectors

1. **Scope escalation**: Read-only REST token grants write access through MCP
2. **Tenant IDOR**: MCP tools don't verify the authenticated user's company
3. **Prompt injection via MCP tool descriptions**: If the AI reads tool descriptions from user-controlled content, the description itself can be a prompt injection vector
4. **Unvalidated tool arguments**: Tools that accept user-supplied IDs without ownership verification
5. **Rate limit bypass**: MCP endpoints might have different (weaker) rate limits than the REST API
6. **Tool definition leakage**: tools/list might return tool schemas that reveal internal data structures

## Real-World Example — Visma Spiris MCP (Jun 2026)

Discovered during Visma bug bounty recon:

- **URI**: `mcp.spiris.se/mcp`
- **Auth**: Returns 401 "unauthorized" without valid token
- **Tools (from public docs)**: Customers (CRUD), Orders (CRUD), Invoices (create/send/search/pay), Suppliers, Vouchers, Accounts, Attachments, Projects, Cost centers
- **MCP tool counts**: ~20+ distinct tool definitions
- **Auth model**: OAuth2 via Visma Connect (same as REST API)
- **OIDC scope**: The scope `spiris-mcp:mcp` in `scopes_supported` at `connect.identity.stagaws.visma.com`

### Probing results from this session:

```bash
# 401 without auth (expected — secure)
curl -sk "https://mcp.spiris.se/mcp" -H "Content-Type: application/json" -d '{"...tool list..."}'
# Response: 401 "unauthorized"

# No unauthenticated exposure found
# Would need valid OAuth2 token with spiris-mcp:mcp scope to proceed
```

### Next steps if token were available:

1. Call `tools/list` to get the full tool catalog
2. Call `tools/call` for each write tool with different company IDs (IDOR test)
3. Test if a token scoped to `ea:accounting_readonly` can call write tools (scope escalation)
4. Test cross-tenant data access by manipulating customer/invoice IDs
