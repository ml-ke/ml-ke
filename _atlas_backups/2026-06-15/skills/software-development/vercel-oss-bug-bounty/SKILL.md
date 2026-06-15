---
title: Vercel OSS Bug Bounty
name: vercel-oss-bug-bounty
description: Systematic approach to finding vulnerabilities in Vercel open source projects (Next.js, AI SDK, Turborepo, SWR) for the HackerOne bug bounty program. Combines static analysis, API auditing, and code review.
---

# Vercel OSS Bug Bounty Methodology

Systematic approach to finding security vulnerabilities in Vercel open source projects for the HackerOne bug bounty program. Covers static analysis with grep/semgrep, API security auditing, and targeted code review of high-risk areas.

## Target Programs

**Vercel Open Source** — HackerOne program. Scope includes:

- **Next.js** — React framework (22K+ source files, 343MB)
- **AI SDK** (vercel/ai) — AI/LLM SDK (4K+ source files, 114MB) — **primary target, newer + less audited**
- **Turborepo** — Monorepo build system (Rust + JS, 1.6K source files)
- **SWR** — Data fetching (261 source files)
- Nuxt, Svelte, other Vercel-adjacent OSS

Average bounty: **$700–$752** (with managed retesting + collaboration)

## Setup

```bash
# Clone repos
mkdir -p ~/Dev/vercel
cd ~/Dev/vercel
git clone --depth=1 https://github.com/vercel/next.js.git
git clone --depth=1 https://github.com/vercel/ai.git
git clone --depth=1 https://github.com/vercel/turborepo.git
git clone --depth=1 https://github.com/vercel/swr.git
```

## Token & Credential Security

**NEVER** put API tokens, passwords, or secrets directly in command-line arguments or script files that will be echoed to the terminal. They can be exposed by:
- Shell history (`.bash_history`)
- Process listings (`ps aux`)
- Tool output logging (command lines captured in context)
- `write_file` tools interpolating token strings
### Safe Patterns

```bash
# BEST: Read from file with restricted permissions
chmod 600 ~/.notion-token
TOKEN=$(cat ~/.notion-token | tr -d '\n')
curl -s "https://api.notion.com/v1/users/me" -H "Authorization: Bearer $TOKEN" -H "Notion-Version: 2026-03-11"

# BETTER for complex API testing: Write Python scripts
# Python avoids bash quoting issues with JSON and Bearer tokens.
# Save to /tmp/ and run: python3 /tmp/test.py
```python
import json, urllib.request
token = open('/home/pro-g/.notion-token').read().strip()

def api(path, method="GET", data=None):
    req = urllib.request.Request(f"https://api.notion.com{path}", method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Notion-Version", "2026-03-11")
    req.add_header("Content-Type", "application/json")
    if data:
        req.data = json.dumps(data).encode()
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:500]}
```

The `$(cat token)` pipe pattern can fail silently. Python scripts handle JSON natively, allow conditional branching, and avoid shell escaping issues entirely.

**Known issue with execute_code + bash**: The `execute_code` tool's `terminal()` helper runs commands through a shell that has quoting conflicts with JSON payloads and `$(cat ...)` token patterns. Always prefer standalone Python scripts (written to /tmp/ via write_file) for any API testing involving JSON or Bearer tokens.

### What Not to Do

```bash
# NEVER: Inline token in command
curl -H "Authorization: Bearer ntn_..."  # TOKEN LEAKED

# NEVER: In script file that gets echoed
echo "Token: ntn_..."  # TOKEN IN OUTPUT

# NEVER: In variable assignment visible in process list
TOKEN="ntn_..." curl ...  # VISIBLE IN `ps aux`
```

## Phase 1: Static Analysis (grep-based)

Use grep on non-compiled source files only (exclude `node_modules/`, `compiled/`, `.test.`, `.spec.`).

### 1. Command Injection / RCE Sinks

```bash
# child_process usage
grep -rn "child_process\.exec\|child_process\.execSync\|child_process\.spawn\|child_process\.fork" packages/*/src/ --include='*.ts' --include='*.tsx' | grep -v compiled | grep -v node_modules

# eval() calls
grep -rn "eval(" packages/*/src/ --include='*.ts' --include='*.tsx' | grep -v compiled | grep -v node_modules | grep -v '.test.'

# Dangerous Function() constructor
grep -rn "new Function(" packages/*/src/ --include='*.ts' --include='*.tsx' | grep -v compiled | grep -v node_modules
```

### 2. Prototype Pollution

```bash
# __proto__ references in source (not in protection code)
grep -rn "__proto__" packages/ --include='*.ts' --include='*.tsx' --include='*.js' | grep -v node_modules | grep -v compiled

# Object.assign with user-controlled objects
grep -rn "Object\.assign(" packages/ --include='*.ts' | grep -v node_modules | grep -v '.test.' | grep -v '.spec.'

# merge/clone functions
grep -rn "merge\|clone\|extend" packages/ --include='*.ts' | grep -v node_modules | grep -v '.test.' | grep -v '.spec.'
```

### 3. Path Traversal

```bash
# path.join with user input
grep -rn "path\.join\|path\.resolve\|path\.normalize" packages/ --include='*.ts' | grep -v node_modules | grep -v compiled

# fs operations with variable paths
grep -rn "fs\.readFile\|fs\.readFileSync\|fs\.createReadStream\|fs\.writeFile\|fs\.unlink" packages/ --include='*.ts' | grep -v node_modules | grep -v compiled
```

### 4. Regex Injection (ReDoS)

```bash
# dynamic Regex construction
grep -rn "new RegExp(" packages/ --include='*.ts' | grep -v node_modules | grep -v '.test.' | grep -v '.spec.'
```

### 5. SSRF Vectors

```bash
# fetch() calls with dynamic URLs
grep -rn "fetch(" packages/ --include='*.ts' | grep -v node_modules | grep -v '.test.'
```

### 6. Server-Side Request Forgery (SSRF) Bypass

Check URL validation functions for bypass techniques:
- DNS rebinding
- URL parser differentials
- IPv6 mapped IPv4 (`::ffff:192.168.1.1`)
- Octal IP notation (0300.0250.0.1)
- DNS over HTTPS endpoints
- Redirect chaining bypass

### 7. Template/Prompt Injection (NEW)

The AI SDK processes user prompts and system prompts. Check for:
- **System prompt override** — Can user message content break out of its context and override system instructions?
- **Tool call injection** — Are tool call arguments validated before execution?
- **Schema injection** — Can JSON schemas for tools be manipulated to trigger ReDoS or injection?
- **Template variable escaping** — Are template variables properly escaped in prompt construction?

```bash
# Prompt/system prompt construction
grep -rn "system\|prompt\|template" packages/*/src/ --include='*.ts' | grep -v node_modules | grep -v '.test.' | grep -v '.spec.'

# Tool call parsing
grep -rn "toolCall\|tool_call\|parseTool\|executeTool" packages/*/src/ --include='*.ts' | grep -v node_modules
```

### 8. Response Manipulation / Stream Poisoning

The AI SDK handles streaming responses from LLMs. Check for:
- **Stream injection** — Can an LLM's response break the stream protocol?
- **Response splitting** — Can response content be crafted to inject new messages?

```bash
# Stream handling
grep -rn "stream\|pipe\|chunk\|delta" packages/*/src/ --include='*.ts' | grep -v node_modules | grep -v '.test.'
```

### 9. Information Disclosure via Error Messages

Check for:
- **StackTrace exposure** — Do errors leak internal paths, API keys, or configuration?
- **Verbose error messages** — Do validation errors reveal too much about internal state?

## Phase 2: Critical Area Deep Dives

### Angle 1: AI SDK Provider API Request Construction → No Issue
- Headers come from provider config only (dev-configured `apiKey`, custom `headers`)
- User prompt content flows through `JSON.stringify()` body — no injection possible
- `postJsonToApi` → `postToApi` → `fetch()` — standard, safe patterns

### Angle 2: AI SDK MCP Implementation → No Issue
- `createChildProcess()` uses `spawn()` with `shell: false` — prevents shell injection
- Command and args come from developer-configured `StdioConfig`, not user input
- SSE transport validates endpoint origin matches connection origin (line 162-166)
### Angle 6: Next.js Image Optimizer — DNS Rebinding TOCTOU (NEW, UNFIXED)

**File**: `packages/next/src/server/lib/image-optimizer.ts` — `fetchExternalImage()`

**Root cause**: The function resolves DNS and checks IPs in one step, then `fetch()` re-resolves DNS in a separate step. An attacker can switch DNS records between the check and the fetch, bypassing the IP validation:

```typescript
// Step 1: Resolve DNS and check IPs (TOCTOU window opens here)
const { hostname } = new URL(href)
const ip = await dnsLookup(hostname)       // resolves once
if (isPrivateIp(ip)) return null            // IP check passes

// Step 2: fetch() re-resolves DNS independently (TOCTOU window closes here)
const response = await fetch(href, ...)      // re-resolves — attacker's second answer wins
```

**Attack**: Register domain `evil.example.com`. Point it to `1.2.3.4` (public IP). Check passes. Then within the TOCTOU window, switch DNS to `127.0.0.1`. The `fetch()` resolves to `127.0.0.1` and connects to localhost.

**Contrast with AI SDK**: The AI SDK's `validateDownloadUrl()` validates at URL level (never resolves DNS), so it's NOT vulnerable to TOCTOU. The image optimizer DOES resolve DNS but creates a gap.

**Impact**: SSRF via DNS rebypass against Next.js Image Optimization API. An attacker who can trigger a Next.js image optimization request can reach internal services on the server.

**Status**: This is a known class of vulnerability (DNS rebinding) but the specific TOCTOU window in `fetchExternalImage` is UNFIXED in current master. The fix would be to use a single DNS resolution and pass the resolved IP + Host header to `fetch()` instead of the hostname.

- **File**: `packages/next/src/next-devtools/server/launch-editor.ts` (core logic)
- **Endpoint**: `packages/next/src/server/dev/middleware-webpack.ts:634` / `middleware-turbopack.ts:398`
- **Endpoint**: `GET /__nextjs_launch-editor?file=PATH&line1=1&column1=1` — no CSRF protection (GET-based, no Origin/Referer check)
- **Path Traversal** (Linux & macOS): `openFileInEditor()` at line 452-455 — absolute paths (`/etc/passwd`) bypass `nextRootDirectory` restriction. Linux has **no filename validation at all** (no allowlist, unlike Windows which has a regex allowlist). File checked only for existence, not location.
- **Windows RCE** — ⚠️ ONLY via `EDITOR` / `VISUAL` env vars:
  - `spawn('cmd.exe', ['/C', editor, ...args])` at line 401. `cmd.exe /C` evaluates shell metacharacters.
  - `REACT_EDITOR` goes through `shellQuote.parse()` — `&` becomes `{op:"&"}` object which stringifies to `[object Object]`, NOT the operator. **No RCE via REACT_EDITOR.**
  - `EDITOR`/`VISUAL` are used directly as raw strings → CAN carry shell metacharacters → RCE via cmd.exe /C
  - Constraint: payload must have NO spaces (Node.js wraps space-containing args in quotes, making `&` literal). `code&whoami` works; `code & whoami` does not.
  - Requires `.env.local` or environment control (supply chain attack via malicious repo).
- **Linux/macOS binary execution**: `spawn(editor, args)` without shell. `$EDITOR`/`$VISUAL` set to any executable path fires it. Not shell-injection RCE, but attacker-controlled binary execution.
- **Editor resolution order**: `REACT_EDITOR` (shellQuote.parse) → running process detection (`ps x`) → `VISUAL` (raw string) → `EDITOR` (raw string)
- **File existence oracle (confirmed)**: 204 (exists) vs 404 (not found) vs 500 (error reading) response differential via `middleware-response.ts`. Triage-relevant for impact framing.
- **CVE-2025-11953 precedent (Metro4Shell, CVSS 9.8)**: React Native Metro server had the EXACT same vulnerability (`/open-stack-frame` endpoint). Actively exploited in the wild — 3,500+ exposed servers, Rust malware deployed on developer machines. The fix sanitized the file path to restrict it to the project directory. Our Next.js bug is the same class.
- **5 escalation paths identified** (REACT_EDITOR removed — shellQuote.parse prevents RCE):
  1. CVE-2025-11953 pattern: Network-reachable when `next dev --host 0.0.0.0` is used
  2. CSRF auto-trigger: `<img src="/__nextjs_launch-editor?file=...">` — no Origin/Referer validation on GET endpoint
  3. 2 env var RCE entry points: VISUAL and EDITOR (raw strings, not shellQuote.parse'd)
  4. Supply chain via .env.local: Malicious npm packages with poisoned .env.local
  5. Linux editor-to-exec: EDITOR pointing to /tmp/malware.sh executes on trigger
- See `references/launch-editor-deep-dive.md` for full analysis and fact-check notes.

### Angle 4: AI SDK SSRF Bypass → ⚠️ CONFIRMED (High Severity)
- **File**: `packages/provider-utils/src/validate-download-url.ts`
- Blocks literal IPs (127.0.0.1, 10.x.x.x, 192.168.x.x) and known localhost names
- **Never resolves DNS** — domains resolving to private IPs bypass entirely
- **Bypass techniques**:
  - `localtest.me` → 127.0.0.1
  - `lvh.me` → 127.0.0.1
  - `*.nip.io` → arbitrary embedded IP (e.g., `169.254.169.254.nip.io` → AWS metadata)
  - `*.sslip.io` → same concept
- **PoC**: 8/8 bypass URLs pass `validateDownloadUrl()` with zero blocked
- **Attack chain**: User file URL → `download()` → `validateDownloadUrl()` (bypasses) → `fetch()` (connects to private IP via DNS)
- **Redirects also bypass**: Same `validateDownloadUrl()` is called after redirects

### Angle 5: Turborepo Command Execution → No Issue
- Uses `tokio::process::Command::new()` — direct exec, no shell
- Task scripts resolved via package manager binary (`npm run <task>`)
- Pass-through args appended safely via `arg_separator()`

### SSRF in AI SDK `validateDownloadUrl` — CONFIRMED (High Severity, NOW DUPLICATE)

**Status**: Submitted but closed as **duplicate** of report #3766258 (submitted May 28, 2026). Same root cause, same bypass domains. Analyst called it a quality report with good PoC, but someone beat us by 2 days.

**Key lesson**: SSRF is a crowded finding in Vercel's AI SDK. The codebase has been audited by multiple researchers. Don't focus exclusively on SSRF — the next finding needs to be a **different vulnerability class** (IDOR, auth bypass, injection, path traversal, RCE).

**File**: `packages/provider-utils/src/validate-download-url.ts`
**Call chain**: `download.ts` → `validateDownloadUrl()` → `fetch()`

**Root cause**: `validateDownloadUrl()` checks the **hostname string** but never resolves DNS. Domains that resolve to private IPs via DNS resolver services bypass all checks.

**Bypass domains (all 8 confirmed, now public via duplicate report #3766258)**:
- `localtest.me`, `lvh.me` → 127.0.0.1
- `*.nip.io`, `*.sslip.io` → arbitrary embedded private IPs

### IDN Homograph SSRF Bypass

The hostname checks in `validateDownloadUrl` use exact ASCII string matching (`hostname === 'localhost'`, `hostname.endsWith('.local')`), but `new URL()` converts Unicode hostnames to Punycode. This means Unicode homoglyph characters bypass the hostname-based checks:

```
input:        http://lоcalhоst:8080/   (Cyrillic 'о' U+043E)
new URL():    hostname = "xn--lcalhst-9ige"  (Punycode-encoded)
check:        "xn--lcalhst-9ige" === "localhost"? → NO → BYPASSED
```

All 3 hostname checks are bypassable:
- `hostname === 'localhost'` — replace any 'o' with Cyrillic 'о' (U+043E)
- `hostname.endsWith('.local')` — replace 'o' in 'local' with Cyrillic 'о'
- `hostname.endsWith('.localhost')` — same technique

**Exploitation**: Register a domain like `xn--lcalhst-9ige.com` (looks like "localhost" in browsers) pointing to 127.0.0.1. When the AI SDK processes a URL on this domain, DNS resolves it to the private IP and the Punycode hostname bypasses all validation.

See `references/idn-homograph-bypass.md` for full analysis and Punycode encoding details.

### 🔍 New Finding: Incomplete Private IP Range Validation (CGNAT Bypass)

The `isPrivateIPv4()` function in `validateDownloadUrl` only blocks 6 IP ranges but misses 7+ additional reserved/private ranges. This is a **different class of bypass** from the DNS bypass — even if DNS IS resolved and checked, these IPs pass validation:

```
100.64.0.0/10  → NOT BLOCKED  CGNAT (Carrier-Grade NAT)
198.18.0.0/15  → NOT BLOCKED  Benchmarking (RFC 2544)
192.0.2.0/24   → NOT BLOCKED  TEST-NET-1
198.51.100.0/24→ NOT BLOCKED  TEST-NET-2
203.0.113.0/24 → NOT BLOCKED  TEST-NET-3
240.0.0.0/4    → NOT BLOCKED  Reserved
```

**Impact**: CGNAT (100.64.0.0/10) is used by Google Cloud, Azure, and AWS EKS for internal VPC networking. GitLab's `UrlBlocker` correctly blocks this range — the AI SDK should too.

**PoC**: `validateDownloadUrl('http://100.64.0.1:8080/config')` → passes validation. See `scripts/test-cgnat-bypass.mjs` and `references/ssrf-cgnat-bypass.md`.

### Multi-hop Redirect SSRF → NOT Exploitable

Analyzed in-depth. The `download.ts` function calls `validateDownloadUrl(response.url)` AFTER all redirects are followed (line 43-45). Since `fetch()` uses `redirect: 'follow'` (default), intermediate hops are followed internally and only the FINAL URL is checked. A chain like [public → private → public] would still have the final public URL checked. Not exploitable.

**PoC**: See `/tmp/poc-final.mjs` or `~/Dev/vercel/report-ssrf.md`

### Deep-Dive Results — All AI SDK Packages Swept

| Package | What I Checked | Result |
|---------|---------------|--------|
| `ai` | Tool parsing, download, streaming | **SSRF in download** |
| `provider-utils` | JSON parsing, URL validation, request construction | **SSRF in validateDownloadUrl** |
| `mcp` | Stdio spawn, SSE/HTTP transports | ✅ Safe (shell=false, origin validation) |
| `openai` | Responses API, tool schemas, error handling | ✅ Safe |
| `openai-compatible` | Request construction, headers | ✅ Safe |
| `anthropic` | Prompt conversion, tool error parsing | ✅ Uses try-catch, safe defaults |
| `anthropic-aws` | SigV4 signing, API key auth | ✅ Safe |
| `google` / `google-vertex` | Auth, file uploads, message conversion | ✅ Safe |
| `gateway` | Provider URL, auth headers | ✅ Dev-configured only |
| `amazon-bedrock` | Fetch wrapper | ✅ Safe |
| `codemod` | `execSync` with string concatenation | ✅ CLI tool, developer-only |
| `workflow` | Agent loop, providers | ✅ Default max 20 steps |
| `react` / `vue` / `svelte` / `angular` | UI framework integrations | (not reviewed in depth for XSS) |
| `replicate` / `elevenlabs` / `deepgram` etc. | Provider-specific implementations | (uses shared infrastructure) |

### Notable Security Patterns Found

1. **SSRF protection exists in Next.js image optimizer** (`fetchExternalImage` at line 872 of `image-optimizer.ts`) — properly resolves DNS and checks IPs. The AI SDK should follow this pattern.

2. **`secureJsonParse` used throughout** — protects against prototype pollution via JSON.

3. **No `shell: true` anywhere** — all child processes use direct exec.

4. **URL validation best practices** — MCP transports validate origin, redirect mode defaults to 'error'.

5. **Risky `JSON.parse()` usages** (all in try-catch):
   - `convert-to-anthropic-prompt.ts` — tool error values, try-catch safe
   - `zod3-to-json-schema/options.ts` — developer schema descriptions only
   - Various `JSON.parse(part.input)` — LLM response parsing, try-catch safe

6. **Missing in all providers**: No DNS-resolution step before fetching external URLs. Only the Next.js image optimizer does this correctly.

### AI SDK — Provider API Request Construction

The AI SDK sends user prompts to LLM APIs (OpenAI, Anthropic, Google, etc.). Check:

1. **Header injection** — Can user-controlled input (model name, provider URL) inject newlines into HTTP headers?
2. **SSRF via provider URL** — If baseURL is user-configurable, can it point to internal services?
3. **Tool call argument injection** — Can tool call arguments escape their JSON context?
4. **Schema injection** — Can JSON schema validation patterns trigger ReDoS?

### AI SDK — MCP (Model Context Protocol) Implementation

The MCP package handles tool invocation and could be a vector for:
1. **Command injection** in MCP STDIO transport (spawns subprocesses)
2. **Path traversal** in tool file operations
3. **SSRF** in MCP SSE transport

### Next.js — Devtools Server

The `launch-editor.ts` file uses `child_process.spawn` with editor names from env vars:
1. Check if `REACT_EDITOR` / `VISUAL` / `EDITOR` env vars are properly sanitized
2. Check if `shellQuote.parse()` properly handles shell escaping
3. Determine if this is reachable from untrusted contexts

### Next.js — Telemetry

Check if telemetry collection handles user-controlled data safely.

## Phase 3: SSRF Confirmation & Webhook Verification

When you find an endpoint that fetches external URLs, confirm server-side SSRF with a webhook:

1. **Create a webhook endpoint** to capture incoming requests:
```bash
curl -s -X POST "https://webhook.site/token" -H "Accept: application/json"
# Use the returned UUID: https://webhook.site/{uuid}
```

2. **Inject the webhook URL** into the target via the vulnerable feature (embed block, file upload URL, etc.)

3. **Monitor for incoming requests** from the target's infrastructure:
```python
import urllib.request, json
url = "https://webhook.site/token/{uuid}/requests?sorting=newest"
req = urllib.request.Request(url)
req.add_header('Accept', 'application/json')
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
for r in data.get('data', []):
    print(f"{r.get('method')} {r.get('url')} IP:{r.get('ip')} UA:{r.get('headers',{}).get('user-agent','')}")
```

4. **Analyze the requests**: IP address reveals internal infrastructure; User-Agent identifies the service making the request (NotionEmbedder, Iframely, etc.); HEAD vs GET indicates probe vs content fetch.

5. **Test with private IPs**: Use DNS bypass domains (`localtest.me`, `*.nip.io`, `*.sslip.io`) to target cloud metadata endpoints (`169.254.169.254`) and internal services. See `references/ssrf-dns-bypass-techniques.md` for the full list of bypass domains.

### Confirmation & Exploitation Checklist

For each finding:

1. **Verify it's reachable** — trace the code path from public API to the vulnerable function
2. **Check if input is sanitized** — look for validation upstream
3. **Build minimal PoC** — confirm exploitation
4. **Escalate impact** — what's the worst-case scenario?

For each finding:

1. **Verify it's reachable** — trace the code path from public API to the vulnerable function
2. **Check if input is sanitized** — look for validation upstream
3. **Build minimal PoC** — confirm exploitation
4. **Escalate impact** — what's the worst-case scenario?

## Vulnerability Types by Target

| Target | Best Vuln Types | Reason |
|--------|----------------|--------|
| **AI SDK** | SSRF bypass, injection, ReDoS, auth bypass | Processes user prompts, constructs HTTP requests to LLM APIs |
| **AI SDK MCP** | Command injection, path traversal | Spawns subprocesses for tools |
| **Next.js** | RCE (devtools), telemetry info leak | Wide deployment, complex build pipeline |
| **Turborepo** | Command injection, path traversal | Build tooling with spawned processes |

## Pitfalls & Lessons Learned

### SSRF is Crowded — Avoid Over-Investing
The `validateDownloadUrl()` SSRF (8 bypass domains) was duplicate #3766258 — someone else found it 2 days earlier. SSRF is the most-reported bug in Vercel's AI SDK. Before spending hours on SSRF analysis:
1. Check recent disclosed reports on HackerOne
2. Look for a DIFFERENT vulnerability class first (IDOR, auth bypass, path traversal, RCE)
3. Only invest in SSRF if you have a bypass technique that hasn't been publicly documented

### Impact Escalation — Turn Small Bugs Into Big Payouts
A single vulnerability class can be expanded into multiple independent findings by exploring all attack surfaces:

**Chain-based escalation**: A dev-only path traversal becomes:
- Standalone report: Path traversal (Medium)
- With env var control: Windows RCE via cmd.exe /C (High)  
- With --host flag: Network-reachable RCE (Critical — CVE-2025-11953 pattern)
- Supply chain: .env.local in malicious npm package (High)
- CSRF auto-trigger: No user interaction needed (Medium)

**Precedent research**: When you find a bug, search for CVEs in similar products/tools:
- Next.js launch-editor → CVE-2025-11953 (Metro4Shell, CVSS 9.8) — identical `/open-stack-frame` endpoint in React Native Metro server. Actively exploited with Rust malware. Cite this in your report to show the class IS exploitable.
- GitLab UrlBlocker blocks 100.64.0.0/10 → validates CGNAT blocking is an industry standard

**Separate bugs, separate reports**: Independent root causes in the same function/feature should be submitted as separate HackerOne reports. Examples from `validateDownloadUrl()`:
1. DNS-resolution bypass (hostnames not resolved to IPs) — duplicate
2. CGNAT IP range gap (isPrivateIPv4 misses 100.64.0.0/10) — NEW, submitted
3. IDN homograph bypass (Punycode hostnames bypass ASCII string matching) — separate bug

Each report needs its own PoC, impact analysis, and fix recommendation. Triage evaluates each independently.

### Report Organization — Structure for Triage Clarity
```
~/Dev/REPORTS/<Program>/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup, NOT in zip. Read this first.
└── poc/
    └── submission.zip  ← PoC code + src/ + package.json. Triager: unzip && npm install && node poc.mjs
```

Rules:
- REPORT.md goes at the folder root, never inside the zip
- The zip contains everything needed to reproduce (npm install + node poc)
- Use published npm packages (not repo clones) — triager should not need to build from source
- Include annotated source code (src/) showing the vulnerable function with comments marking the gap

### Bug Bounty Bootcamp — Prioritize These Chapters for Non-SSRF Bugs

When SSRF is crowded, pivot to these vulnerability classes from the Bug Bounty Bootcamp book:
- **Chapter 10 (IDOR)**: Parameter tampering, UUID enumeration, access control testing
- **Chapter 14 (Insecure Deserialization)**: JSON parsing, prototype pollution
- **Chapter 16 (Template Injection)**: SSTI in rendering engines
- **Chapter 17 (Application Logic Errors)**: Business logic flaws, privilege escalation
- **Chapter 19 (Same-Origin Policy)**: SOP bypass, CORS misconfigurations
- **Chapter 21 (Information Disclosure)**: Error message verbosity, path disclosure
- **Chapter 24 (API Hacking)**: API-specific testing methodology

## Support Files

- `scripts/poc-ssrf.mjs` — Test script for AI SDK SSRF bypass (8/8 URLs confirmed)
- `scripts/test-cgnat-bypass.mjs` — Test script for CGNAT/incomplete IP range validation bypass
- `references/ssrf-report-template.md` — Full HackerOne report template for the AI SDK SSRF
- `references/ssrf-cgnat-bypass.md` — CGNAT bypass analysis, PoC, fix recommendations
- `references/launch-editor-deep-dive.md` — Next.js launch-editor path traversal + Windows RCE analysis
- `references/ai-sdk-deep-dive-sweep.md` — Comprehensive security sweep of all 56 AI SDK packages
- `references/ssrf-dns-bypass-techniques.md` — DNS bypass domains, IP formats, webhook confirmation methodology
- `references/notion-api-testing.md` — Full Notion API testing methodology, endpoint reference, PoC scripts

## Reporting

See Bug Bounty Bootcamp (Ch2) for report structure:
1. Descriptive title
2. Clear summary
3. Severity assessment
4. Steps to Reproduce (STR)
5. Proof of Concept (PoC)
6. Impact and attack scenarios
7. Recommended mitigations

**PoC Archive**: When triage asks for an archive file (".zip/.tar.gz with working PoC"), see `api-bug-bounty-methodology` skill → Step 8: PoC Archive Creation for HackerOne Submission. The archive should use published npm packages (not repo clones), have individual proof files plus a runner, and include `src/` with vulnerable source code for context.

The SSRF report template is at `references/ssrf-report-template.md` — modify for your specific finding.

## Bookshelf Resources

Located at `~/Documents/BOOKS/Hackinh/`:
- **Bug Bounty Bootcamp** (Vickie Li) — Web vuln methodology
- **Hacking APIs** (Corey Ball) — API security testing
- **Web Application Hacker's Handbook** (Stuttard & Pinto) — Classic reference
- **RTFM Red Team Field Manual** (Ben Clark) — Quick reference

---

## Case Study: Notion API Security Testing

**Two SSRF vectors confirmed** in Notion's API (High severity). Full documentation in `references/notion-api-testing.md`.

### SSRF #1: Embed/Bookmark Blocks
- **Service**: NotionEmbedder (HEAD) + Iframely (GET)
- **URL types**: HTTP + HTTPS (no private IP validation)
- **Block types**: embed, bookmark, image, video, audio, file
- **DNS bypass**: `localtest.me`, `*.nip.io`, `*.sslip.io` all work

### SSRF #2: File Upload External URL  
- **Service**: notion-api (HEAD) + notion (GET) — separate infrastructure
- **URL types**: HTTPS only
- **Endpoint**: `POST /v1/file_uploads` with `mode: external_url`

### Not Vulnerable
- IDOR via UUID enumeration
- Mass assignment
- `javascript:`/`data:` URL injection
- Search scope isolation

See `references/notion-api-testing.md` for full methodology, PoC scripts, and webhook verification results.

