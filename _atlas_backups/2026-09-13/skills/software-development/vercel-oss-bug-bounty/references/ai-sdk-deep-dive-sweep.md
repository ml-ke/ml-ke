# AI SDK Comprehensive Security Sweep — Session Log

Date: 2026-05-29
Target: `vercel/ai` (all 56 packages)
Analyzer: Hermes Agent

## Results Summary

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 HIGH | 1 | SSRF bypass in `validateDownloadUrl` (DNS resolver domains) |
| ⚠️ LOW | 1 | Path traversal in Next.js devtools (`__nextjs_launch-editor`) |
| ✅ Safe | 54 | All other packages cleared |

## Package-by-Package Results

### Core Packages
- **`ai`** — Tool parsing uses `secureJsonParse` (prototype pollution safe). `download()` calls `validateDownloadUrl()` (SSRF vulnerable — see finding below). Tool execution uses validated schemas.
- **`provider-utils`** — `validateDownloadUrl.ts`: blocks literal IPs but **never resolves DNS**. `secure-json-parse.ts`: well-implemented (scans for `__proto__` and `constructor.prototype`). `post-to-api.ts`: standard fetch, headers from config only.

### MCP (Model Context Protocol)
- **`mcp`** — `create-child-process.ts`: uses `spawn()` with `shell: false` ✓. SSE transport: validates endpoint origin matches connection origin (line 162-166) ✓. HTTP transport: `redirect: 'error'` by default ✓.

### Provider Implementations
- **`openai`** — `postJsonToApi()` with headers from config. Tool schemas are provider-defined (no SDK-side execution). Responses API handled via standard patterns.
- **`anthropic`** — Prompt converter uses `secureJsonParse` for tool call arguments. `extractErrorValue()` uses `JSON.parse` in try-catch (safe).
- **`anthropic-aws`** — SigV4 signing via `aws4fetch`. API key header injection is standard.
- **`google` / `google-vertex`** — Auth via `google-auth-library`. File upload: URL from Google API response header, not user input. Properly structured.
- **`azure`** — Standard OpenAI-compatible patterns.
- **`amazon-bedrock`** — AWS SigV4 signed requests via standard library.
- **`gateway`** — BaseURL defaults to Vercel's gateway. Dev-configured only.
- **`openai-compatible`** — Generic provider; URL path hardcoded. Headers from config.

### Framework Integrations
- **`react`, `vue`, `svelte`, `angular`, `rsc`** — UI wrappers. No server-side execution paths of concern.

### Tooling
- **`codemod`** — `execSync()` with string concatenation for jscodeshift. CLI tool only (dev runs it on own machine). Not exploitable by remote attacker.
- **`devtools`** — Developer debugging tools.
- **`otel`** — OpenTelemetry tracing. No network requests.

### Media/Speech Providers
- **`elevenlabs`, `deepgram`, `replicate`, `assemblyai`, `revai`, `lmnt`** — Standard API wrappers. Use shared `postJsonToApi` infrastructure.

### Experimental
- **`workflow`** — Agent loop with default max 20 steps. Uses shared `generateText`/`streamText`. No new code paths.

## Key Security Patterns Observed

1. **`secureJsonParse` everywhere** — Prototype pollution via JSON is well-mitigated.
2. **No `shell: true`** — Zero occurrences across 56 packages.
3. **No hardcoded credentials** — All API keys from config/env.
4. **Input validation** — Zod schemas validate all tool inputs.
5. **DNS resolution missing** — Only `validateDownloadUrl` has this gap. Ironically, **Next.js image optimizer** (`fetchExternalImage` in `image-optimizer.ts:872`) does DNS resolution correctly — same company, different team.
