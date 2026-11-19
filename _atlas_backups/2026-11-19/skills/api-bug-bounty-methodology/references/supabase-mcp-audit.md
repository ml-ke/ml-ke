# Supabase MCP Server — Code-Level Audit

Date: 2026-06-01
Source: github.com/supabase-community/supabase-mcp (cloned to ~/Dev/supabase-mcp/)
Branch: main

## Package Structure

```
packages/
├── mcp-server-supabase/     # Main Supabase MCP server
│   └── src/
│       ├── index.ts         # Entry point
│       ├── server.ts        # Server creation + tool registration
│       ├── platform/        # Platform abstraction layer
│       │   ├── api-platform.ts  # Management API operations (815 LOC)
│       │   └── types.ts
│       ├── management-api/  # HTTP client (openapi-fetch)
│       ├── content-api/     # GraphQL client for docs API
│       ├── tools/           # Tool implementations
│       │   ├── account-tools.ts
│       │   ├── branching-tools.ts
│       │   ├── database-operation-tools.ts
│       │   ├── debugging-tools.ts
│       │   ├── development-tools.ts
│       │   ├── docs-tools.ts
│       │   ├── edge-function-tools.ts
│       │   └── storage-tools.ts
│       └── advisories/      # Security advisory generation
├── mcp-server-postgrest/    # PostgREST MCP server
└── mcp-utils/               # Shared MCP utilities
```

## Tool-by-Tool Security Analysis

### 1. search_docs (docs-tools.ts)

**Input**: `graphql_query` (z.string()) — raw GraphQL query string
**Behavior**: Sends query to hardcoded `https://supabase.com/docs/api/graphql` via GET
**Security**: 
- ✅ URL is fixed at construction time (server.ts line 96)
- ✅ GraphQL query is URL-encoded via URLSearchParams
- ❌ No query complexity limiting — could be used for heavy GraphQL queries (DoS potential, but OOS)
- ✅ Response validated against graphqlResponseSchema
- **Verdict**: No SSRF, safe.

### 2. execute_sql (database-operation-tools.ts)

**Input**: `query` (z.string()) + `project_id`
**Behavior**: Sends raw SQL to `/v1/projects/{ref}/database/query`
**Security**:
- ⚠️ SQL injection by design (user's own database)
- ✅ Results wrapped in `<untrusted-data-{uuid}>` boundaries (line 376-381)
- ✅ LLM warned not to follow instructions in untrusted data
- ✅ `readOnly` option enforced server-side
- **Verdict**: Intended functionality. Prompt injection protection is instruction-based, not technical.

### 3. apply_migration (database-operation-tools.ts)

**Input**: `query` (z.string()) + `name`
**Behavior**: Sends SQL to `/v1/projects/{ref}/database/migrations`
**Security**:
- ⚠️ SQL injection by design
- ✅ Intentionally omits result from response (line 234-236) — prevents prompt injection
- ✅ `readOnly` mode blocks execution
- **Verdict**: Good prompt injection defense, intentional SQLi.

### 4. deploy_edge_function (edge-function-tools.ts)

**Input**: `name`, `entrypoint_path`, `files[]` (name + content)
**Behavior**: Uploads Deno runtime code to user's project
**Security**:
- ⚠️ Arbitrary Deno code execution (by design — it's a function deployment tool)
- ✅ `readOnly` mode blocks execution
- **Verdict**: Intended functionality.

### 5. API Platform (api-platform.ts)

**Key observations**:
- `createManagementApiClient()` takes `baseUrl` and `accessToken` (lines 72-74)
- `apiUrl` option (line 59) allows overriding Management API URL at construction
- All operations use typed `openapi-fetch` client with path parameters (no injection possible)
- Path parameter injection is prevented by openapi-fetch's URL template resolution
- **Verdict**: Secure client, no SSRF at call time.

## Prompt Injection Protections

| Location | Mechanism | Effectiveness |
|----------|-----------|---------------|
| execute_sql result | `<untrusted-data-{uuid}>` wrapper + LLM instruction | Medium (LLM obeying instruction depends on model) |
| apply_migration | Result intentionally hidden | High (no data returned = no injection) |
| search_docs result | Returned as `z.unknown()` | None (raw data passed to LLM) |

## SSRF Analysis

**No SSRF found at runtime.** All external URLs are:
1. Fixed at construction time (`apiUrl`, `contentApiUrl`)
2. Hardcoded in the `openapi-fetch` client path templates
3. Passed through URLSearchParams with standard encoding

The only SSRF-like vector is the `apiUrl` option in `SupabaseApiPlatformOptions` (api-platform.ts:50-60), which allows the HOST application to override the Management API URL. This is by design and documented.

## Recommendation for Further Testing

1. Check if the `contentApiUrl` can be controlled via environment variables or config files
2. Test the STDIO transport for argument injection (similar to MCP SDK STDIO injection pattern)
3. Check GraphQL query depth on `search_docs` for abuse potential
4. Review `mcp-utils` for shared vulnerabilities across all MCP servers
