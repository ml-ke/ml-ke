# Anthropic Deep Audit Results — June 1, 2026

## Assets Audited

| Asset | Source | Verdict |
|-------|--------|---------|
| Claude Code v2.1.88 (leaked source) | ~/Dev/claude-code-source/source/ (16K+ lines, 80+ files read) | **Hardened** — permission system solid |
| Python SDK | ~/Dev/claude-agent-sdk-python/ (30+ source files, full audit) | **Well-secured** — no injection, safe credential handling |
| TypeScript SDK | ~/Dev/claude-agent-sdk-typescript/ (just SessionStore adapters) | **No attack surface** — just PG/S3/Redis adapters |
| platform.claude.com | Live probe (unauthenticated) | **Auth-gated** — proper 403/404 on all paths |
| api.anthropic.com/* | Live probe | **Auth-gated** |
| MCP proxy | Live probe (mcp-proxy.anthropic.com) | **Auth-gated** (401) |
| platform.staging.ant.dev | Live probe | **Same protections as prod** |

## What Was Checked (and Found Clean)

### Permission System (Claude Code)
- 80+ source files read across the permission pipeline
- All tools (built-in, MCP, plugin MCP) go through same `hasPermissionsToUseTool()` path
- Plugin-installed MCP servers do NOT bypass permissions (Issue #448 is NOT a bypass — plugins use same `passthrough` → `ask` flow)
- Tool descriptions use actual execution input — no misrepresentation found
- All tool execution emits structured NDJSON output — no hidden execution found
- SDK transport layer uses list-based `execve` — no shell injection

### SDK Security (Python)
- API keys: passed via environment variables, NEVER on CLI command line — industry standard
- Credential files: `0o600` permissions, OAuth tokens redacted before temp write
- Path traversal: thorough validation (`_is_safe_subpath()`) — rejects `..`, absolute paths, NUL bytes
- Message parsing: pattern-matching on verified dict keys with proper error handling
- Error handling: 3 low-severity concerns (all would close Informative)

### Web Apps (platform.claude.com)
- Cloudflare + auth gating
- Proper error distinction (403 permission vs 404 not found vs 405 method)
- No unauthenticated API endpoints discovered

## Remaining Attack Surface

The bugs that likely exist require **authenticated multi-account testing**:

| Test | What You'd Need |
|------|-----------------|
| Conversation sharing IDOR | Two platform accounts, share conversations |
| Workspace privilege escalation | Org admin + regular member accounts |
| API key permission boundaries | Keys with different scopes |
| MCP connector sandboxing | Desktop app with custom MCP server |
| OAuth token scope abuse | OAuth client with scope manipulation |

## Conclusion

Anthropic's developer-facing code (CLI, SDKs, APIs) is professionally engineered with deliberate security investment. After 27 advisories on claude-code alone, the easy paths are gone. The permission system is genuinely well-hardened. Without authenticated testing with multiple account tiers, this program is unlikely to yield findings.

**Recommendation**: Pivot to programs with source code access and lower competition (self-hostable TypeScript/Node.js OSS like Supabase, Discourse, GitLab).
