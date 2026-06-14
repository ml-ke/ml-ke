---
title: Source Code Security Audit
name: source-code-security-audit
description: Systematic white-box vulnerability research on open source codebases for bug bounties. Covers target selection, security book research, grep-based static analysis, PoC construction, and report triage. Also covers AI platform/CLI/SDK security audit and MPC cryptographic library audit (absorbed specializations).
---

# Source Code Security Audit

Run a systematic white-box security audit on an open source codebase to find reportable vulnerabilities for bug bounties. The methodology mirrors what a senior application security engineer does during a code review but automated and traceable.

## Trigger

Use this skill when the user asks you to:
- Audit a codebase for security vulnerabilities
- Find bugs in an open source project
- Research vulnerability patterns in a repo
- Prepare for a bug bounty submission
- Perform a white-box security review

## Prerequisites

- Target repo(s) cloned locally
- `grep` / `ripgrep` installed
- `semgrep` (optional, for advanced static analysis)
- `nodejs` / `npm` (for JavaScript/TypeScript projects)
- `pdftotext` (from poppler-utils, for extracting book content)

## Token & Credential Security (Mandatory)

**NEVER** put API tokens, passwords, or secrets directly in command-line arguments or script files that will be echoed to the terminal. They can be exposed by shell history, process listings, tool output logging, and string interpolation in write tools.

### Safe Patterns

```bash
# READ FROM FILE with restricted permissions
chmod 600 ~/.target-token
TOKEN=*** ~/.target-token | tr -d '\n')
curl -s "https://api.target.com/v1/endpoint" -H "Authorization: Bearer *** 

# ENVIRONMENT VARIABLE (short-lived shell)
export TOKEN=***
curl -s -H "Authorization: Bearer *** 

# PYTHON SCRIPTS — construct auth header by concatenation
with open('/home/pro-g/.target-token') as f:
    t = f.read().strip()
ah = 'Authorization: Bearer' + ' ' + t
# Then use 'ah' in curl arguments — no token string in source code
```

### What Not to Do

```bash
# NEVER: Inline token in command — leaks to shell history, ps aux, tool logs
curl -H "Authorization: Bearer ntn_actual_token_here"

# NEVER: Token in script files that get echoed or written with write_file
# The Hermes write_file tool interpolates text matching certain patterns

# NEVER: Token in variable assignment visible in process list
TOKEN="ntn_actual_token_here" curl ...  # VISIBLE IN `ps aux`
```

Research materials (recommended):
- **Bug Bounty Bootcamp** (Vickie Li) — web vulnerability methodology
- **Hacking APIs** (Corey Ball) — API security testing
- **The Web Application Hacker's Handbook** (Stuttard & Pinto) — classic reference
- Online: PayloadsAllTheThings, HackTricks, PortSwigger Research

## HackerOne Submission Strategy (Reputation Management)

On HackerOne, new researchers have a limited number of initial report submissions that determine their signal rating. Each submission counts — even duplicates. Manage this carefully.

### Key Rules
- **Only 5 initial submissions** (approximately) determine whether you can keep submitting or get rate-limited
- Each report must be independently verifiable and non-duplicate
- **Duplicates do NOT improve your signal** — even if your report is well-written and the bug is real
- A Medium finding with accurate, verified analysis is worth more than a High with inflated claims

### Avoiding Duplicates

| Vulnerability Class | Duplicate Risk | Strategy |
|---|---|---|
| **SSRF via DNS bypass** (nip.io, localtest.me, sslip.io) | **VERY HIGH** — everyone checks this | Avoid SSRF unless you have a truly novel bypass technique |
| **SSRF via IP range gaps** (missing CGNAT, missing reserved ranges) | **HIGH** — well-documented | Only pursue if the target's code clearly differs from known patterns |
| **Path traversal** | **MEDIUM** — depends on codebase | Examine the actual validation code before claiming |
| **IDOR / Auth bypass** | **LOW-MEDIUM** — depends on feature | Newer features are less likely to be duplicate |
| **XSS with CSP bypass** | **LOW** — CSP bypasses are target-specific | Good for newer/less-audited programs |
| **Business logic flaws** | **LOW** — application-specific | Hard to duplicate |

### Program Selection Criteria

Pick programs where your specific skillset gives you a unique advantage:

| Factor | Good Pick | Bad Pick |
|--------|-----------|----------|
| **Tech stack match** | TypeScript/Node.js or Ruby on Rails (our proven stacks) | Go, Rust, Java, PHP (we're slower here) |
| **Source code access** | Open source on GitHub (can audit directly) | Proprietary only (black-box testing) |
| **Program maturity** | <500 resolved reports, newer program | 2000+ reports (picked clean) |
| **New feature angle** | Recently added features (AI, workflows, integrations) | Legacy code that's been audited for years |
| **Bounty range** | $500+ average for mid-tier impact | VDP only or under $50 minimum |
| **Attack surface** | Large codebase, many plugins/integrations | Tiny library or single-purpose tool |

### Before Committing to a Program

1. Check the scope page — what assets are in scope? Source code? Specific domains?
2. Read the out-of-scope section — avoid wasting time on excluded classes
3. Check the bounty table — what's the payout for Medium vs High?
4. Check the hacktivity page — what types of reports are being submitted and resolved?
5. Check response times — 10+ hours first response is fine; 2+ days is a red flag
6. Clone the source code if open source — verify you can navigate it before committing

### What Not to Hunt (Based on Experience)

- **SSRF in GitLab's UrlBlocker** — saturated, well-documented, multiple researchers have covered it
- **SSRF in any DNS-resolving validation** — the basic bypass is too well-known
- **Discourse** — bounties are suspended indefinitely
- **Node.js core** — funding for bug bounties has been lost
- **Internet Bug Bounty (IBB)** — paused, not accepting new submissions
- **General GraphQL introspection** — by design on most platforms, not a finding
- **CSRF on public/unauth forms** — explicitly out of scope on most programs
- **Missing security headers** — explicitly out of scope

## Step-by-step Methodology

### 1. Target Selection

When picking a bug bounty program from HackerOne/Bugcrowd:

| Factor | Good | Avoid |
|--------|------|-------|
| **Source access** | Open source (can audit code directly) | Proprietary (need accounts/cookies) |
| **Attack surface** | Large codebase with diverse features | Tiny utility library |
| **Program maturity** | Newer or medium-traffic (less picked over) | 2000+ resolved reports (picked clean) |
| **New feature angle** | Recently added features often have less review | Legacy code that's battle-tested |
| **Bounty range** | $500+ avg for mid-tier impact | Under $50 minimum (low ROI) |
| **Program features** | Retesting + Collaboration flags are good signs | VDP only (no payout) |

Check the program page for: scope (domains, repos, asset types), bounty ranges, rules of engagement, out-of-scope list.

### 2. Project Reconnaissance

```bash
# Get a feel for the codebase
du -sh *
find . -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' -o -name '*.go' | wc -l

# Check for compiled/ vendored code (usually out of scope for audits)
find . -type d -name 'compiled' -o -name 'dist' -o -name 'build' -o -name 'node_modules'

# Read documentation and AGENTS.md / CONTRIBUTING.md for architecture understanding
cat AGENTS.md 2>/dev/null | head -50
cat README.md 2>/dev/null | head -50
```

### 3. Research Vulnerability Patterns

For each vulnerability class you plan to search for, consult security references:

**From Bug Bounty Bootcamp (Ch22: Conducting Code Reviews):**
- Fast approach: `grep` for dangerous patterns first
- Dangerous function calls: `eval`, `exec`, `spawn`, `child_process`
- Leaked secrets: API keys, tokens, passwords in code
- New patches: `git log --oneline -20` — recent changes are prime targets
- Debug endpoints: routes/paths starting with `debug`, `test`, `admin`, `internal`
- Developer comments with security implications: `TODO`, `FIXME`, `HACK`, `XXX`

**For each vulnerability class, use the following grep patterns:**

```bash
# Command Injection / RCE sinks
grep -rn "child_process\.exec\|child_process\.spawn\|child_process\.execFile\|child_process\.fork\|child_process\.execSync\|child_process\.spawnSync" \
  --include='*.ts' --include='*.tsx' --include='*.js' packages/ src/ lib/ \
  2>/dev/null | grep -v compiled | grep -v node_modules | grep -v '.test.'

# eval() / Function() calls
grep -rn "eval\|new Function(" --include='*.ts' --include='*.js' packages/ src/ \
  2>/dev/null | grep -v compiled | grep -v node_modules | grep -v '.test.'

# Prototype pollution
grep -rn "__proto__\|\.constructor\.prototype\|Object\.assign(" \
  --include='*.ts' --include='*.tsx' --include='*.js' packages/ src/ \
  2>/dev/null | grep -v node_modules | grep -v '.test.' | grep -v compiled

# Path traversal
grep -rn "readFile\|readFileSync\|createReadStream\|sendFile\|path\.join\|path\.resolve" \
  --include='*.ts' --include='*.tsx' --include='*.js' packages/ src/ \
  2>/dev/null | grep -v node_modules | grep -v '.test.' | grep -v compiled \
  | grep -v "\.json\|\.md\|\.txt"

# Dynamic RegExp (ReDoS risk)
grep -rn "new RegExp\|RegExp(" --include='*.ts' --include='*.js' packages/ src/ \
  2>/dev/null | grep -v node_modules | grep -v '.test.' | grep -v compiled \
  | grep -i "user\|input\|param\|query\|config\|env"

# HTTP requests / fetch (SSRF potential)
grep -rn "fetch(\|new Request\|\.request(" --include='*.ts' --include='*.js' packages/ src/ \
  2>/dev/null | grep -v node_modules | grep -v '.test.' | grep -v compiled

# Server-side template injection
grep -rn "\.render\|\.compile\|template(" --include='*.ts' --include='*.js' packages/ src/ \
  2>/dev/null | grep -v node_modules | grep -v '.test.' | grep -v compiled
```

### 4. SSRF via URL Validation Without DNS Resolution

A critical and common vulnerability pattern: URL validation functions that block private IPs by checking the **hostname string** but never **resolve DNS**. The domain `169.254.169.254.nip.io` passes all string-level checks (hostname is not a literal IP, not `localhost`, not a `.local` domain) but resolves to the cloud metadata IP.

**Testing tool — DNS resolver bypass domains:**

| Domain | Resolves to | SSRF Target |
|--------|-------------|-------------|
| `localtest.me` | 127.0.0.1 | Localhost |
| `lvh.me` | 127.0.0.1 | Localhost |
| `*.nip.io` (e.g. `169.254.169.254.nip.io`) | Embedded IP | Any private IP |
| `*.sslip.io` | Embedded IP | Any private IP |

**Full test vector set:**

```
# Direct private IPs (should be blocked)
http://127.0.0.1/  http://10.0.0.1/  http://192.168.1.1/
http://169.254.169.254/  http://localhost/

# DNS resolver bypass (should also be blocked — often isn't)
http://localtest.me/  http://lvh.me/
http://127.0.0.1.nip.io/  http://169.254.169.254.nip.io/
http://10.0.0.1.nip.io/  http://192.168.1.1.nip.io/
http://127.0.0.1.sslip.io/  http://169.254.169.254.sslip.io/

# URL parser differentials (Node.js normalizes these — only work if code
# parses URLs differently from Node's URL class)
http://0x7f000001/  http://2130706433/  http://127.1/  http://0/
```

**Confirmation steps:**
1. Run URL through validation function — assert it passes
2. Verify it resolves: `nslookup <domain>` or `node -e "dns.resolve4('...')"`
3. Confirm code calls `fetch()` after validation passes

**Fix pattern (Node.js):**
```typescript
import * as dns from 'node:dns/promises';
const addresses = await dns.resolve4(hostname);
if (addresses.some(ip => isPrivateIP(ip))) throw Error('SSRF blocked');
```

### 4a. DNS Rebinding TOCTOU in fetchExternalImage (UNFIXED Pattern)

A subtle SSRF bypass exists in some image-loading code paths where DNS resolution and HTTP fetch happen in separate steps:

```typescript
// VULNERABLE pattern — TOCTOU via DNS rebinding
async function fetchExternalImage(url: string) {
  const dnsResult = await dnsLookup(url.hostname)  // Step 1: Resolve DNS
  if (isPrivateIP(dnsResult)) throw Error('Blocked')  // Step 2: Check IP
  const response = await fetch(url)  // Step 3: FETCH (re-resolves DNS!)
}
```

The gap: `dns.lookup()` resolves the hostname and checks against `isPrivateIp()`, but `fetch()` makes a fresh DNS resolution. An attacker can:
1. Point `attacker.com` to a public IP (8.8.8.8) — passes the DNS check
2. After the check passes, switch DNS to a private IP (127.0.0.1) — `fetch()` resolves the new IP

**Detection:**
```bash
# Search for the separate-check-then-fetch pattern
grep -rn "dns.lookup\|dns.resolve" --include='*.ts' --include='*.tsx' --include='*.js' . \
  2>/dev/null | grep -v node_modules | grep -v '.test.'
# Then check if fetch() or http.request() uses a SEPARATE resolution from the check
```

**Fix**: Use `fetch()` with a URL that has the pre-resolved IP, not the hostname. Or pass the `agent` option with the pre-resolved IP.

### 5a. SSRF Confirmation via Webhook

When you find an endpoint that accepts external URLs, confirm **server-side fetching** with a webhook. This distinguishes SSRF from client-only rendering:

1. **Create a webhook endpoint** at webhook.site:
```bash
WEBHOOK=$(curl -s -X POST "https://webhook.site/token" -H "Accept: application/json")
UUID=$(echo "$WEBHOOK" | python3 -c "import sys,json; print(json.load(sys.stdin)['uuid'])")
echo "Webhook URL: https://webhook.site/$UUID"
```

2. **Inject the webhook URL** into the vulnerable feature (embed block, file upload, bookmark, etc.)

3. **Wait and check for incoming requests** from the target's infrastructure:
```python
import urllib.request, json
time.sleep(20)
url = f"https://webhook.site/token/{UUID}/requests?sorting=newest"
req = urllib.request.Request(url)
req.add_header('Accept', 'application/json')
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
for r in data.get('data', []):
    hdrs = r.get('headers', {})
    ua = ''
    if isinstance(hdrs, dict):
        for k, v in hdrs.items():
            if k.lower() == 'user-agent':
                ua = v[0] if isinstance(v, list) else str(v)
    print(f"{r.get('method')} {r.get('url')} IP:{r.get('ip')} UA:{ua}")
```

4. **Analyze the results**:
   - **IP address**: reveals internal infrastructure network ranges
   - **User-Agent**: identifies the internal service (`NotionEmbedder`, `notion-api`, `Iframely`, etc.)
   - **HEAD vs GET**: HEAD indicates a probe/validation; GET indicates content fetch
   - **Multiple services**: Some features trigger multiple fetchers (e.g., NotionEmbedder probes, Iframely fetches)

**What confirms SSRF:**
- Requests originate from the target's IP ranges (e.g., `131.149.232.x` for Notion — look up via `whois`)
- User-Agent string references internal service names
- Multiple request types (HEAD + GET) for a single URL injection
- File upload status changes from `pending` to `uploaded` (proving content was fetched)

**What proves blind vs. reflected SSRF:**
- Check if the block content changes after fetching (blind if not stored)
- Check if the fetched content is displayed on the page (reflected if visible to viewers)
- If blind, exploitation requires: redirect chains, DNS rebinding, or side-channel attacks

### 5. Skill Growth from Security Books

Extract targeted knowledge from reference books during a session:

```bash
# Extract specific chapter with pdftotext
pdftotext "Bug Bounty Bootcamp.pdf" - | sed -n '/^22$/,/^23/p' | head -300
```

| Book | Best For |
|------|----------|
| **Bug Bounty Bootcamp** (Vickie Li) | Web vuln methodology, code review (Ch22), RCE (Ch18), SSTI (Ch16) |
| **Hacking APIs** (Corey Ball) | API testing: auth bypass (Ch8), fuzzing (Ch9), authorization (Ch10), injection (Ch12), GraphQL (Ch14) |
| **Web Application Hacker's Handbook** | Classic deep reference on all web app vuln classes |
| **RTFM Red Team Field Manual** | Quick command syntax reference |

### 6. Build Proof of Concept

For each confirmed vulnerability:

```typescript
// Minimal reproduction script
const payload = '...';
const result = await vulnerableFunction(payload);
console.log('Expected: safe, Got:', result);
```

Capture:
- The vulnerable code path (file + line)
- The triggering input
- The result (error, data leak, command execution)
- A plausible attack scenario

### 7. Pre-Submission Fact-Check

**CRITICAL: Look across ALL active findings before submitting.** This is the most common workflow error. When the user asks you to research escalation techniques or review findings, you MUST audit across ALL programs simultaneously — not deep-dive one target and miss the others.

Before submitting a report, verify every claim against actual source code and live testing:

1. **Check migration files** — `t.string` creates `varchar(255)` in PostgreSQL, `t.text` is unbounded. Don't claim full payload exposure for truncated columns.
2. **Trace the full auth chain** — Read the guardian/policy file (not just the controller). Does it verify OWNERSHIP or just VISIBILITY?
3. **Test on the live instance** — Verify the route exists, auth works as expected, and error codes match your claims.
4. **Verify escalation claims** — Can you really enumerate IDs? Is the data really accessible cross-user? Or do rate limits, auth gates, or column truncation prevent it?
5. **Test redirect behavior** — Does the endpoint follow redirects? Are redirect targets re-validated?
6. **Check for duplicate risk** — Search publicly disclosed reports and security advisories for similar findings before submitting.
7. **Check for unmerged fix branches** — The vendor may already know about the issue but haven't released:
   ```bash
   git branch -a | grep -i "fix\|auth\|security\|middleware\|patch"
   git log --all --oneline --grep="apiWrapper\|IS_PLATFORM\|auth.*bypass\|security.*fix" -10
   ```
   An unmerged fix on a branch does NOT invalidate the finding — it strengthens the report by proving the vendor is aware. But it does mean you should accelerate disclosure.
8. **Be honest about severity** — A Medium with accurate analysis builds signal. A High with inflated claims that get disputed hurts your reputation.
9. **Verify configuration constants are actually USED in source code** — A `#define` or `constexpr` in a header is NOT evidence of an implemented feature. Run `grep -rn CONSTANT_NAME src/` to confirm it's actually referenced in a runtime code path. In this session, `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION` was defined in `mpc_globals.h` but NEVER checked in any source file — the claimed behavior (unencrypted MTA at v2) did not exist. Always trace the constant to a code path before making claims about its effect.
10. **Verify cross-primitive comparisons are valid** — Before comparing key sizes or security levels, confirm you're comparing the SAME cryptographic primitive. STD Paillier (for encryption) and Paillier Commitment (for ZK proofs) have different structures and different key size requirements. A claim like "2048-bit vs 3072-bit security gap" is misleading if the two are different primitives. Check the actual function call in source code to identify the primitive type.
11. **Check what the existing test suite already tests** — Run the target's test suite and look for `attack_helpers.h`, `SECTION("attack: ..." )` patterns, or any file with "attack" in its name. If the test suite already probes a specific attack vector (e.g., "encrypted_partial_sig all zeros"), the developers are likely already aware of it. Run the existing tests before writing novel PoCs — they may confirm the oracle works (saving build time) or reveal the path is already defended.
12. **"As an attacker I would..." framing** — When describing impact, start with "As an attacker I would..." and follow with a concrete, verifiable action. This is the preferred style for impact claims. Example: "As an attacker I would craft Paillier ciphertexts that pass proof verification but fail signature verification, extracting the decrypted value from the server's response." Avoid abstract claims like "an attacker could potentially recover the key."

### 8. Report Structure

All reports follow this layout:

```
~/Dev/REPORTS/<Program>/<SubmissionNumber>/<finding-name>/
├── REPORT.md           ← Standalone writeup (NOT in zip). Root of finding folder.
└── poc/
    ├── poc-<name>.py   ← Working exploit/PoC script
    └── submission.zip  ← PoC archive for HackerOne upload
```

Rules:
- REPORT.md sits at the finding folder root, never inside the zip
- The zip contains everything needed to reproduce (npm install + run, or python3 poc.py)
- Use published npm packages (not repo clones) — triager should not need to build from source
- Source repos stay at `~/Dev/<vendor>/`
- PoC/test code written to test the program stays at `~/Dev/` root (not in REPORT/)

Report content structure:
1. **Title**: Descriptive (e.g., "Unscoped ID Lookup in AI Audit Log Exposes Raw Prompts")
2. **Summary**: 2-3 sentence overview
3. **Severity**: CVSS score + rationale
4. **Root Cause**: Code path with line numbers
5. **Steps to Reproduce**: Numbered, starting from registration/login
6. **PoC**: Code or request/response pairs
7. **Impact**: Worst-case scenario (data exposure, RCE, etc.)
8. **Escalation**: How an attacker could chain or expand the finding
9. **Fix Recommendation**: Specific code change

## Compiled/C++ Library Security Audit

When auditing compiled C/C++ codebases (especially cryptographic libraries), the methodology differs fundamentally from web/API audits.

### Build System Reconnaissance

```bash
# Identify the build system
cat CMakeLists.txt 2>/dev/null | head -30
cat Makefile 2>/dev/null | head -10

# Check compilation flags — especially visibility (PoC blocker)
grep -n "VISIBILITY\|visibility\|-fvisibility" CMakeLists.txt Makefile* 2>/dev/null
```

### Symbol Visibility — CRITICAL PoC Pitfall

Many C++ libraries compile with `-fvisibility=hidden`, making internal functions needed for PoCs (verification, encrypt/decrypt, ZKP helpers) unlinkable from the shared library.

**Detection:** `nm -C lib*.so | grep -i "verify\|proof\|encrypt" | head -10` — lowercase 't' = hidden.

**Workaround — static library mirror:**
```cmake
add_library(yourlib_static STATIC ${SOURCES})
set_target_properties(yourlib_static PROPERTIES
    CXX_VISIBILITY_PRESET "default" C_VISIBILITY_PRESET "default"
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(yourlib_static PUBLIC OpenSSL::Crypto)
# Apply same compile_options + include_directories as shared target
```
Then link PoCs against `yourlib_static` instead of `yourlib`.

### Trace the Nonce/RNG Path

For signature protocols, trace nonce (k) generation — VM snapshot = key recovery:

`BN_rand_range(tmp, EC_GROUP_get0_order(ctx->curve))` = NO RFC 6979 = VM clone → k reuse → key recovery.

### Identify Oracle Channels

Verification functions with multiple early-return checks create CCA oracles:

1. **Input validity** (`is_coprime_fast`) — rejects invalid-format ciphertexts
2. **Proof structure** (size checks) — may leak expected parameter sizes
3. **Algebraic equations** (EC + Paillier commitments) — rejects inconsistent proofs
4. **Output verification** (sig verify after decryption) — confirms decryption happened

Each gate produces a distinguishable error. If a crafted ciphertext passes all equation checks but fails the final sig verify, the server **decrypted your value**. That's a decryption oracle.

### BN_CTX Frame Nesting — PoC Pitfall

When writing PoCs that use OpenSSL BIGNUM operations inside a function that already has a `BN_CTX_start()` frame, creating a NEW `BN_CTX` (via `BN_CTX_new()`) for the inner operations avoids frame corruption. Never use the same BN_CTX for nested `BN_CTX_start/end` pairs across function call boundaries unless you perfectly balance every start with an end.

**Bad pattern — frame corruption risk:**
```cpp
BN_CTX_start(ctx);           // outer frame
auto result = some_function(ctx);  // may internally BN_CTX_start(ctx) again
BN_CTX_end(ctx);             // may pop the wrong frame if exception skipped inner end
```

**Good pattern — isolated CTX per scope:**
```cpp
BN_CTX_start(ctx);           // outer frame
{
    BN_CTX* inner = BN_CTX_new();
    BN_CTX_start(inner);
    // ... inner operations ...
    BN_CTX_end(inner);
    BN_CTX_free(inner);
}
BN_CTX_end(ctx);
```

Also: C++ exceptions thrown between `BN_CTX_start` and `BN_CTX_end` will skip the end call unless RAII wrappers are used. Always wrap frames in `std::unique_ptr` with `BN_CTX_end` deleter, or use the `bn_ctx_frame` RAII struct.

### Leverage Existing Test Infrastructure

The target's test suite is the most powerful PoC tool:

1. **Run existing attack tests** first — tells you what developers already considered
2. **Check for `attack_helpers.h`** — the team may have explicit attack helper functions you can reuse directly
3. **Append to existing test files** rather than fighting the build system — Catch2 SECTION isolation gives clean state per probe
4. **Each probe = one signing session** — fresh tx_id per probe avoids state consumption issues

### CCA Attack Pattern for Paillier-based Protocols

1. Encrypt known plaintext m with known randomness r → you can generate a VALID proof
2. Submit → if all proof gates pass, server decrypts your value
3. Modify ciphertext homomorphically: `c' = c * Enc(Δ)` → observe gate changes
4. ~256-512 queries recover the Paillier private key λ
5. With λ, decrypt any ciphertext → full key recovery
6. P1 threshold: <1000 aborts — ~256-512 probes fits comfortably

### Information Leakage Through Proof Structure

Serialized proofs encode parameter sizes as plain integers. The first 4 bytes of a BAM well-formed proof encode the expected Paillier modulus byte size — leaking the key size without any crypto operation. Check serialized proof/commitment headers for similar leaks in other protocols.

## Reference Files

- **`references/bug-bounty-programs.md`** — notes on specific HackerOne/Bugcrowd programs analyzed, their scope, bounty ranges, and angles.
- **`references/notion-api-reference.md`** — Notion REST API reference: endpoints, auth model, rate limits, and attack vectors discovered during live API testing.
- **`references/user-preferences.md`** — user's workflow style, reporting preferences, and autonomy expectations for bug bounty sessions.
- **`references/paillier-oracle-poc.md`** — Detailed BAM Paillier CCA oracle PoC reproduction (Fireblocks MPC program), covering the hidden symbol workaround, gate-by-gate analysis, and probe methodology.
- **`references/ssrf-bypass-techniques.md`** — Comprehensive SSRF bypass domain catalog, IP obfuscation formats, URL parser differentials, and DNS rebinding testing.
- **`references/hackerone-programs-analysis.md`** — Evaluated programs (GitLab, Nextcloud, Zendesk, Elastic, etc.) with scope, bounty ranges, and attack vectors.
- **`references/ai-mcp-server-audit-results.md`** — MCP Fetch server SSRF analysis and Claude Code WebFetchTool comparison.
- **`references/mpc-lib-source-analysis.md`** — Concrete source audit findings from Fireblocks mpc-lib: nonce generation code paths, DRNG patterns, protocol architecture.
- **`references/fs-hash-truncation-poc.md`** — 3-phase Fiat-Shamir hash truncation PoC blueprint: size verification → seed collision → challenge equivalence, with exact SHA256/DRNG replication code.
- **`references/mpc-real-world-precedents.md`** — BitForge, TSSHOCK, CVE-2023-33241/33242 case studies for MPC vulnerability precedents.
- **`references/poc-assessment-june2026.md`** — PoC status table, build/run instructions, and submission outcome analysis for Fireblocks mpc-lib.
- **`references/range-proof-extended-seed-hardcode.md`** — Adjacency-hunting finding: `use_extended_seed=0` hardcoded in 5 range proof ZKP locations even at version >= 11.

**Cross-reference**: For PoC archive construction (zipping multi-file evidence for HackerOne triage, using published packages for standalone reproducibility), see `api-bug-bounty-methodology` skill.

## Cross-SDK Comparison — Powerful Audit Technique

When a vendor provides SDKs in multiple languages, compare how they implement the **same auth/signing logic** across SDKs. Discrepancies reveal bugs that no single-SDK audit catches.

### How to Do It

1. **Clone ALL language SDKs** — Look at the vendor's GitHub org for repos matching `*-sdk-*` or `*-api-*` in Python, Java, Go, Ruby, Rust, PHP, C#, TypeScript, etc.
2. **Find the auth/signing component** — Search for `sign`, `hash`, `bodyHash`, `hmac`, `jwt`, `token`, `bearer` in each repo.
3. **Compare the EXACT same operation** — For Fireblocks: how does each SDK compute `bodyHash` for POST requests?
   ```python
   # Python: json.dumps(body) — correct
   # Java:  JSON.stringify then getBytes — correct  
   # Go:    json.Marshal then sha256 — correct
   # TS:    raw bodyJson object — WRONG!
   ```
4. **Cross-reference with older SDK versions** — The old JS SDK had the correct code; the new TS SDK was auto-generated and introduced the bug. Check git history.
5. **Verify against test vectors** — If any SDK has test files with expected hash values (e.g., `bodyHash: "9724c1e2..."`), test all other SDKs against it.

### Why It Works

- Auto-generated SDKs (OpenAPI Generator) can have template bugs that affect only one language
- Different teams maintain different SDKs — one team may fix a bug others don't know about
- Language-specific edge cases (JavaScript object → string coercion, Python vs JSON serialization defaults)
- The bug is invisible when auditing one SDK in isolation — it only appears when you ask "why does Python produce a different hash than TypeScript for the same input?"

### When to Use This

- Vendors with 3+ language SDKs (Fireblocks has 9+, Notion has 6+, many API-first companies)
- Auth schemes involving request signing, hashing, or JWT construction
- Complex serialization chains (object → string → hash → sign)
- Any SDK auto-generated from an OpenAPI spec

---

## 7. AI Platform / CLI / SDK Security Audit

Class-level methodology for auditing AI platform integrations including MCP (Model Context Protocol) servers, AI CLI tools, and SDKs. Absorbed from the former `ai-platform-security-audit` skill.

### 7.1 Attack Surface Overview

| Surface | Examples | Key Risks |
|---------|----------|-----------|
| MCP Servers | Fetch, Filesystem, Git, Memory | SSRF, Path traversal, Command injection |
| CLI Tools | Claude Code WebFetchTool, BashTool | HTTP request handling, DNS rebinding |
| SDKs | API client libraries, subprocess wrappers | Shell injection, trust bypass, flag injection |
| Web Apps | AI platform dashboards | SSRF, IDOR, GraphQL |
| Plugins | MCP extensions, Claude Code plugins | File access, network access bypass |

### 7.2 MCP Server Security Audit

MCP servers communicate via stdin/stdout (stdio) or HTTP (SSE/Streamable HTTP).

**SSRF in Fetch-Type Tools** (most common vulnerability):

```python
# Vulnerable pattern — no IP filtering, no redirect validation
async with AsyncClient(proxy=proxy_url) as client:
    response = await client.get(url, follow_redirects=True, ...)
```

**Checklist:**
- [ ] IP filtering: private range, loopback, metadata endpoints?
- [ ] `follow_redirects` enabled? Redirect targets re-validated?
- [ ] Hostname blocklist (localhost, metadata endpoints)?
- [ ] Protocol restriction (only http/https)?
- [ ] URL length limit?
- [ ] Proxy URL configurable? (could route through attacker proxy)
- [ ] Does it use DNS resolution before connecting or only string checks?

**Comparison**: Anthropic's `mcp-server-fetch` (Python) has NO IP filtering. Claude Code's WebFetchTool (TypeScript) has defense-in-depth: hostname validation, same-host redirects, domain blocklist, HTTP→HTTPS upgrade.

**Path Traversal in Filesystem Tools:**

```typescript
// Check if paths are validated with realpath, not string comparison
normalizedPath = path.resolve(path.normalize(absolutePath));
return allowedDirectories.some(dir =>
    normalizedPath.startsWith(normalizedDir + path.sep)
);
```

**Checklist:**
- [ ] Uses `path.resolve(path.normalize(input))` before comparison?
- [ ] Rejects null bytes (`\\x00`)?
- [ ] Uses `startsWith` check with `path.sep` appended?
- [ ] Handles symlinks (`fs.realpath` on allowed dirs)?
- [ ] TOCTOU between validation and file operation?

### 7.3 AI CLI Tool Security Audit

**Web Fetch Tool Analysis:**
- URL validation: blocks hostnames with <2 parts? (blocks `localhost` but NOT `127.0.0.1`)
- Redirect handling: automatic follow or user-prompted?
- Cross-host redirect: blocked or permitted?
- Preapproved hosts list (bypasses permission prompt)?
- Credential stripping (username:password in URL)?

**Bash/Shell Tool Analysis:**
- Command restrictions (allowlist or blocklist)?
- Argument sanitization?
- Environment variable exposure?
- Working directory constraints?

### 7.4 SDK Transport Layer Audit

When auditing an SDK that spawns a CLI subprocess (e.g., `claude-agent-sdk-python`):

1. **Shell injection**: List-based args (`subprocess.Popen([binary, \"--flag\", value])`) are safe. Shell strings are dangerous.
2. **Flag injection via options**: Can controlled values contain `--` to inject flags? Only possible with shell strings.
3. **Environment injection**: Does the SDK merge caller env vars? Can `PATH`, `LD_PRELOAD`, `PYTHONPATH` be overridden?
4. **Non-interactive trust bypass**: SDK mode typically bypasses permission dialogs. `--settings` / `--mcp-config` flags passing arbitrary JSON with `command` fields can execute arbitrary processes without user consent.
5. **Tool restriction API**: `allowed_tools` is typically auto-approve (not allowlist); `disallowed_tools` works as deny list. Know the distinction.

### 7.5 LLM Prompt Injection Testing

AI platforms expose prompt injection as a unique attack surface:

- **Direct**: Input containing conflicting instructions (\"Ignore previous instructions...\")
- **Indirect**: Content ingested by LLM (web pages, documents) with hidden instructions
- **Tool prompt injection**: Inject instructions into tool descriptions the model reads
- **Data exfiltration via model output**: Encoded channels (base64, emoji) in outputs

**Testing workflow:** Send conflicting instructions, upload documents with hidden commands, inject into MCP tool descriptions, verify model output for leaked system prompts.

**Reference**: `references/ai-mcp-server-audit-results.md` — MCP Fetch server SSRF analysis, Claude Code WebFetchTool comparison, filesystem path traversal assessment.

---

## 8. MPC (Multi-Party Computation) Library Audit

Audit methodology for MPC/threshold-signature cryptographic libraries (GG18/GG20, Lindell17, Fireblocks mpc-lib, BitGo TSS). Absorbed from the former `mpc-crypto-audit` skill. Expands the C/C++ library audit methodology (section 6) with MPC-specific patterns.

### 8.1 The Core Pattern: Missing/Wrong ZK Proofs

The most dangerous MPC vulnerabilities come from missing or weakened zero-knowledge proofs. The **two-proof minimum rule**: any MPC protocol using Paillier encryption needs BOTH:
1. **Range proof** — proves encrypted values are bounded (not huge)
2. **Paillier well-formedness proof** — proves modulus n is a valid biprime

**ZKP Checklist for every MPC implementation:**

| Check | What It Proves | If Missing |
|-------|---------------|------------|
| Schnorr proof | Knowledge of secret key for public key | Anyone can claim any public key |
| Paillier Blum ZKP | n is a Blum integer (p×q, p≡3 mod 8, q≡7 mod 8) | Multi-prime modulus passes (CVE-2023-33241) |
| Range proof (MTA values) | Encrypted values bounded | CCA oracle via crafted ciphertexts |
| Range proof (Paillier factors) | p,q are large enough | Small factors pass, attacker factors n |
| Ring Pedersen ZKP | Prover knows λ s.t. s = t^λ mod n | Commitment scheme weakened |
| Cross-party uniqueness | Modulus not reused across parties | Key reuse attack |

### 8.2 Nonce Generation — RFC 6979 Gap

ECDSA/EdDSA nonces via `BN_rand_range()` instead of RFC 6979 deterministic nonces create key-recovery risk if RNG is cloned (VM snapshot, process fork).

```bash
grep -rn "BN_rand_range\\|RAND_bytes" src/  # RNG call sites
grep -rn "rand(" src/ | grep -i "nonce\\|scalar"  # Nonce generation sites
```

### 8.3 Version-Gated Crypto Strength

Some MPC libraries (Fireblocks mpc-lib) make crypto strength version-dependent. Uniquely dangerous because:
- `if (version > metadata.version)` only blocks upgrade, not downgrade
- ZKP seed computation changes with version — simple seed (v<11) omits key binding
- Ciphertext validation: `strict_ciphertext_length` only enforced at high versions

```bash
grep -rn "version >=\\|use_extended_seed\\|strict_ciphertext" src/
```

### 8.4 Oracle Pattern: Distinguishable Verification Gates

Verification functions with multiple early-return checks create CCA oracles. Each gate produces a distinguishable error → attacker learns which gate caught their crafted input.

**Fix**: All verification failures should produce the SAME error message regardless of which check triggered.

### 8.5 Fiat-Shamir Hash Truncation

A common copy-paste bug: using one proof value's byte count (`BN_num_bytes(proof.X)`) to hash a different proof value (`proof.Y`), truncating the hash window. Proven via 3-phase PoC: size verification → seed collision → challenge equivalence.

See `references/fs-hash-truncation-poc.md` for the complete PoC blueprint with Catch2 gotchas.

### 8.6 Adjacency Hunting

After finding one vulnerability, search the same developer's parallel code paths for the same class of issue:

1. Identify the ROOT CAUSE pattern (e.g., \"version gates crypto parameters\")
2. Search ALL code paths implementing the same pattern (every `verify_*` function receiving `version`)
3. Check both GENERATE side AND VERIFY side for hardcoded weaker modes

**Case study**: MTA Fiat-Shamir seed weakness → adjacency check → range proof ZKPs hardcode `use_extended_seed=0` in 5 places even at version >= 11.

### 8.7 Honest Assessment: When to Stop Escalating

Before submitting ANY crypto finding, apply the **PoC-Claim Congruence Test**:

```
Bad title: \"Key Recovery via Distinguishable Verification Gates\"
PoC output: \"ec_left does not equal ec_right commitment\"
→ Title claims key recovery. PoC shows gate failure. MISMATCH.

Good title: \"BAM Well-Formed Proof Verification Has Distinguishable Error Channels\"
PoC output: \"ec_left does not equal ec_right commitment\"
→ Title matches PoC output. MATCH.
```

**Rule**: The PoC's last printed line before \"ALL PASS\" or \"DONE\" defines what you can claim.

### 8.8 PoC Build Infrastructure

- Use existing test framework — already links against all crypto libs
- Static library > shared library (shared libs hide internal symbols)
- Separate BN_CTX per library call — never share between harness and library
- No chained `&&` in Catch2 REQUIRE
- Use `bn_ctx_frame` RAII or wrapped `std::unique_ptr` for frame safety

**Templates:** See `templates/cosigner_test_CMakeLists.txt` for CMakeLists.txt snippet to add PoC files to the cosigner_test target.

**References:**
- `references/mpc-lib-source-analysis.md` — nonce generation, DRNG patterns, BAM status
- `references/fs-hash-truncation-poc.md` — 3-phase Fiat-Shamir truncation PoC blueprint
- `references/mpc-real-world-precedents.md` — BitForge, TSSHOCK, CVE-2023-33241/33242
- `references/poc-assessment-june2026.md` — PoC status table and build/run instructions
- `references/range-proof-extended-seed-hardcode.md` — adjacency finding: hardcoded `use_extended_seed=0`
- `references/paillier-oracle-poc.md` — working Paillier CCA oracle PoC

## Common Pitfalls

### 1. Compiled/vendored code is noise
Always exclude `node_modules/`, `compiled/`, `dist/`, `build/`, `vendor/` directories from grep searches. They contain third-party code that is typically out of scope. Focus on `packages/`, `src/`, `lib/`.

### 2. Test/spec files are red herrings
Unit tests contain deliberate "bad" inputs to test error handling. Grep results in `*.test.ts` or `*.spec.ts` are almost always false positives.

### 3. Not all `eval()` is insecure
Code that evals its own hardcoded source (e.g., bundlers, transpilers) is different from code that evals user input. Check if the argument to `eval` is a constant string or a variable.

### 4. Mature programs have high bar
Programs with 1000+ resolved reports (like Twilio at 2987) have been heavily audited. Focus on newer programs or newly added features in mature ones.

### 5. Read the out-of-scope rules
Many programs explicitly exclude: DoS, rate limiting, clickjacking on non-sensitive pages, self-XSS, missing SPF/DMARC records. Don't waste time on these.

### 6. Cross-reference nmap OS detection
If profiling a network alongside code audit, nmap `-O` results can be wildly wrong (e.g., Apple TV identified as Xbox 360). Use service-level fingerprints (mDNS, companion-link, AirPlay banners) for ground truth.
