---
title: AI Platform Security Audit Methodology
name: ai-platform-security-audit
description: Systematic approach to auditing AI platform integrations, MCP (Model Context Protocol) servers, AI CLI tools, and SDK security. Covers SSRF in fetch tools, path traversal in filesystem tools, prompt injection vectors in tool descriptions, and API auth bypass.
---

# AI Platform Security Audit Methodology

Class-level methodology for auditing AI platform integrations, including:
- **MCP (Model Context Protocol) servers** — official reference implementations and third-party
- **AI CLI tools** (Claude Code, Codex, etc.) — web fetch, file system, bash execution
- **AI SDKs** — API client libraries, auth handling, URL validation
- **AI platform web apps** (Claude.ai, API endpoints) — SSRF, IDOR, GraphQL

## Attack Surface Overview

AI platforms expose multiple attack surfaces that traditional web apps don't:

| Surface | Examples | Our Experience |
|---------|----------|----------------|
| MCP Servers | Fetch, Filesystem, Git, Memory | SSRF, Path traversal |
| CLI Tools | WebFetchTool, BashTool, FileReadTool | HTTP request handling, command injection |
| SDKs | API clients, auth libraries | SSRF in URL fetching, auth bypass |
| Web Apps | Claude.ai, API.anthropic.com | SSRF, IDOR, GraphQL |
| Plugins | Claude Code plugins, MCP extensions | File access, network access |

## Phase 1: MCP Server Security Audit

MCP (Model Context Protocol) servers are standalone processes that provide tools, resources, and prompts to AI models. They communicate via stdin/stdout (stdio) or HTTP (SSE/Streamable HTTP).

### 1.1 SSRF in Fetch-Type Tools

The most common vulnerability in MCP servers is SSRF in tools that fetch URLs. Check:

```python
# Python MCP servers often use httpx.AsyncClient
async with AsyncClient(proxy=proxy_url) as client:
    response = await client.get(url, follow_redirects=True, ...)
```

**Checklist:**
- [ ] Is there any IP filtering? (private range, loopback, metadata)
- [ ] Is `follow_redirects` enabled? Are redirect targets re-validated?
- [ ] Is there a hostname blocklist (localhost, metadata endpoints)?
- [ ] Is there a protocol restriction (only http/https)?
- [ ] Is there a URL length limit?
- [ ] Is a proxy URL configurable? (could route through attacker proxy)

**Vulnerable pattern found in `mcp-server-fetch` (Python):**
```python
# NO IP filtering, NO redirect validation, NO hostname restrictions
async def fetch_url(url, user_agent, force_raw=False, proxy_url=None):
    async with AsyncClient(proxy=proxy_url) as client:
        response = await client.get(url, follow_redirects=True, ...)
```

**Comparison: Claude Code's WebFetchTool (TypeScript) has defense-in-depth:**
- `validateURL`: blocks hostnames with <2 parts (blocks `localhost` but NOT `127.0.0.1`)
- `isPermittedRedirect`: only allows same-host redirects (with/without www.)
- `maxRedirects: 0` in axios, custom redirect handler
- `checkDomainBlocklist`: server-side check against Anthropic's blocklist
- HTTP→HTTPS upgrade

### 1.2 Path Traversal in Filesystem Tools

Check if filesystem operations validate paths against allowed directories:

```
path-validation.ts — isPathWithinAllowedDirectories(absolutePath, allowedDirectories)
```

**Checklist:**
- [ ] Uses `path.resolve(path.normalize(input))` before comparison
- [ ] Rejects null bytes (`\x00`)
- [ ] Uses `startsWith` check with `path.sep` appended
- [ ] Handles symlinks (uses `fs.realpath` on allowed dirs)
- [ ] Home directory expansion (`~`) handled correctly
- [ ] Windows drive letter normalization

**Vulnerability patterns:**
- Symlink inside allowed directory → escape to parent
- Case sensitivity mismatch (Windows)
- WSL/Unix path format confusion
- UNC path injection (`\\server\share\..\..\`)
- Time-of-check-to-time-of-use (TOCTOU) between validation and file operation

### 1.3 Tool Permission/Scope Analysis

MCP servers define tools with input schemas. Check:
- Are file paths constrained to allowed directories?
- Are URLs restricted to allowed domains?
- Is there a destructive hint annotation (`destructiveHint: true`)?
- Are there preapproved hosts that bypass permission prompts?

## Phase 2: AI CLI Tool Security Audit

AI CLI tools (Claude Code, Codex, OpenCode) expose tools that run on the user's machine.

### 2.1 Web Fetch Tool Analysis

```typescript
// Input schema
z.strictObject({
  url: z.string().url().describe('The URL to fetch content from'),
  prompt: z.string().describe('The prompt to run on the fetched content'),
})
```

**Checklist:**
- [ ] URL validation: does it block hostnames with <2 parts? (blocks `localhost` but not `127.0.0.1`)
- [ ] Redirect handling: automatic follow or user-prompted?
- [ ] Cross-host redirect: blocked or permitted?
- [ ] HTTP→HTTPS upgrade?
- [ ] Server-side domain blocklist?
- [ ] URL length limit?
- [ ] Max content size?
- [ ] Timeout?
- [ ] Credential stripping (username/password in URL)?
- [ ] Preapproved hosts list (bypasses permission prompt)?

### 2.2 File System Tool Analysis

- Path validation: does it constrain to allowed/working directories?
- Symlink handling: `fs.realpath` used?
- Binary file handling: base64 encoding, persisted to disk?
- Concurrent access: TOCTOU between validation and read?

### 2.3 Bash/Shell Tool Analysis

- Command restrictions (allowlist of commands?)
- Argument sanitization?
- Environment variable exposure?
- Working directory constraints?

## Phase 3: Source Code Analysis Approach

When auditing AI platform source code (TypeScript/Node.js or Python):

### 3.1 Key Files to Find

```bash
# TypeScript/Node.js
find . -name "*.ts" -not -path "*/node_modules/*" | xargs grep -l "axios\|fetch\|httpx" 2>/dev/null
find . -name "*.ts" -not -path "*/node_modules/*" | xargs grep -l "URL\|url\|uri" 2>/dev/null
find . -name "*.ts" -not -path "*/node_modules/*" | xargs grep -l "follow_redirect\|redirect" 2>/dev/null

# Python  
find . -name "*.py" -not -path "*/venv/*" | xargs grep -l "httpx\|requests\|aiohttp" 2>/dev/null
find . -name "*.py" -not -path "*/venv/*" | xargs grep -l "url\|URI" 2>/dev/null
```

### 3.2 Tool Registration Pattern

```typescript
// TypeScript MCP server — look for tool registration
server.registerTool("fetch", {
  inputSchema: { url: z.string() },
  ...
}, handlerFunction)

// Python MCP server — look for tool handlers
@server.list_tools()
async def list_tools():
    return [Tool(name="fetch", inputSchema=Fetch.model_json_schema(), ...)]
```

### 3.3 URL Validation Pattern

```typescript
// Look for validateURL, isSafeUrl, isPermittedRedirect, sanitizeUrl
// Check what the function filters out:
// - Hostname parts < 2 → blocks single-word names
// - Private IP ranges → should block 10.x, 172.16-31.x, 192.168.x, 127.x
// - Metadata IPs → should block 169.254.x.x
// - Credentials → should block username:password@
```

### 3.4 SDK Source Code Audit Approach

When auditing AI SDKs (API client libraries like Python SDKs for AI CLI tools or MCP):

**Architecture mapping**: SDKs typically have a transport layer (subprocess/HTTP), options/configuration layer, and public API surface. Map all three before deep-diving.

```python
# Python SDK typical structure:
src/<package>/
├── __init__.py          # Public API exports
├── client.py            # High-level client
├── query.py             # One-shot API
├── types.py             # Options dataclasses
└── _internal/
    ├── transport/
    │   └── subprocess_cli.py  # Core: builds CLI command, spawns subprocess
    └── query.py              # Control protocol, message routing
```

**Security checklist for SDK transport layers:**

1. **Shell injection**: Does the SDK pass args as a `list[str]` (safe) or shell string (dangerous)?
   - Safe: `subprocess.Popen([binary, "--flag", value], ...)` — each arg is a separate element
   - Dangerous: `subprocess.Popen(f"{binary} --flag {value}", shell=True, ...)` — shell interpolation

2. **Flag injection via option values**: Can a controlled option value (e.g., `system_prompt`) contain `--` that breaks out? Only possible with shell strings — list-based args prevent this entirely.

3. **Environment injection**: Does the SDK merge caller-provided env vars with inherited env?
   ```python
   process_env = {**os.environ, **options.env}
   ```
   This can override `PATH`, `LD_PRELOAD`, `PYTHONPATH` — check if any dangerous keys can be set.

4. **Mode flags**: Does the SDK set a non-interactive flag on the subprocess?
   ```python
   process_env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"  # Non-interactive mode
   ```
   In non-interactive mode, trust dialogs and permission prompts are typically **bypassed**. This is architectural — the SDK assumes the consumer has already decided trust.

5. **`--settings` / `--mcp-config` flags**: Does the SDK pass arbitrary JSON configs with hooks or MCP server definitions? These can execute shell commands. In non-interactive mode, the trust dialog is skipped:
   ```typescript
   // From CLI's hooks.ts (Claude Code source):
   function shouldSkipHookDueToTrust(): boolean {
       const isInteractive = !getIsNonInteractiveSession()
       if (!isInteractive) { return false }  // SDK mode — hooks always execute
   }
   ```

6. **`extra_args` / pass-through options**: Does the SDK allow arbitrary CLI flags? If so, the consumer can bypass security controls:
   ```python
   options.extra_args = {"dangerously-skip-permissions": None}
   # → passes --dangerously-skip-permissions to the CLI
   ```

7. **Tool restriction API**: Does the SDK's tool restriction actually work?
   - `allowed_tools` is typically auto-approve (not allowlist) — critical distinction
   - `disallowed_tools` works as deny list
   - Verify by checking what the corresponding CLI flag does

**Common SDK vulnerability patterns** (seen in real audits of claude-agent-sdk-python):
- `allowed_tools` whitelist bypass (Issue #361): All built-in tools provided regardless of whitelist
- Plugin MCP permission bypass (Issue #448): Unclear if plugins bypass `canUseTool` callbacks (open 6+ months)
- Non-interactive trust bypass (architectural): SDK mode implicitly trusts all settings content
- Arbitrary MCP server command execution: MCP configs accept `command` fields spawning subprocesses
- Environment override: Caller-provided env vars override `PATH`, enabling subprocess hijacking

**Real-world example**: See `anthropic-bug-bounty` skill → `references/python-sdk-audit.md` for the full audit of claude-agent-sdk-python at github.com/anthropics/claude-agent-sdk-python

## Phase 4: Program-Specific Testing Patterns

### 4.1 Anthropic (HackerOne Program — Launched May 2026)

**Scope**: Claude.ai, Anthropic API, Claude Code, SDKs, MCP integrations, official clients
**Source**: github.com/anthropics (87+ repos), github.com/modelcontextprotocol/servers

**Key source locations:**
- `~/Dev/mcp-servers/` — MCP reference servers (cloned)
- `~/Dev/claude-code-source/` — Claude Code leaked source (cloned from Exhen/claude-code-2.1.88)
- `~/Dev/anthropic-claude-code/` — Claude Code official repo (plugins/meta)

**MCP servers (7 total):**
| Server | Language | Key Risk |
|--------|----------|----------|
| `src/fetch` | Python | SSRF — no IP filtering, no redirect validation |
| `src/filesystem` | TypeScript | Path traversal — symlink bypass possible |
| `src/git` | Python | Command injection in git operations |
| `src/everything` | TypeScript | Reference server, all MCP features |
| `src/memory` | TypeScript | Knowledge graph, file persistence |
| `src/sequentialthinking` | TypeScript | Pure logic, no I/O |
| `src/time` | Python | Timezone queries, no I/O |

### 4.2 MCP Server Specific Checks

**fetch server** (Python, `server.py`):
```python
# CRITICAL: No SSRF protection whatsoever
async def fetch_url(url, user_agent, force_raw=False, proxy_url=None):
    async with AsyncClient(proxy=proxy_url) as client:
        response = await client.get(url, follow_redirects=True, ...)
        # No check on URL scheme, hostname, or resolved IP
```

- URL directly from user input → `httpx.AsyncClient.get(url, follow_redirects=True)`
- No IP filtering, no private range check, no loopback check
- `follow_redirects=True` follows redirects blindly
- Attack: Prompt Claude to fetch `http://127.0.0.1:9200/` or metadata endpoints

**filesystem server** (TypeScript, `index.ts`):
```typescript
// Path validation present but has gaps
export function isPathWithinAllowedDirectories(absolutePath, allowedDirectories) {
  normalizedPath = path.resolve(path.normalize(absolutePath));
  return allowedDirectories.some(dir => 
    normalizedPath === normalizedDir || normalizedPath.startsWith(normalizedDir + path.sep)
  );
}
```

- Uses string comparison (`startsWith`), not realpath — symlink bypass possible
- No TOCTOU protection between validation and file read
- Validates allowed dirs at startup with `fs.realpath`, but not individual file paths at runtime

## Phase 5: HackerOne Program Assessment

When evaluating an AI platform bug bounty program:

### 5.1 Key Indicators

| Factor | Good Signal | Bad Signal |
|--------|-------------|------------|
| Program age | New (<6 months) = less competition | Old (>2 years) = well-audited |
| Resolved reports | <50 total = wide open | >200 total = picked over |
| Scope breadth | Multiple asset types (web + CLI + SDK + MCP) | Single domain |
| Source available | GitHub repos with source code | Closed-source, black-box only |
| Tech stack match | TypeScript/Node.js, Python | Java, C++, C# |

### 5.2 Prioritization Order

1. **MCP servers and tooling** — Newer, less researched, our SSRF skills apply directly
2. **CLI tools** — Source code often available (leaked or official), Node.js/TypeScript
3. **Web apps** — Traditional web app testing (SSRF, IDOR, GraphQL)
4. **SDKs** — HTTP client libraries, auth handling
5. **Mobile/desktop clients** — Harder to test, Electron apps

## Phase 6: Report Structure

Follow the pattern established for traditional bug bounty reports:

```markdown
# Finding Title
**Product**: <component name>
**File(s)**: <source file paths with line numbers>
**Severity**: <CVSS score and vector>

## Summary
## Root Cause
## Attack Chain (step-by-step)
## Impact
## CVSS Rationale
## Recommended Fix
## PoC
## Source References
```

Include specific source code references with line numbers. For MCP servers, reference the exact function and line where the vulnerability exists.

## Reference Files

- `references/mcp-server-audit-results.md` — MCP Fetch server SSRF analysis and Claude Code WebFetchTool comparison
