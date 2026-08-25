---
name: atlas-continuous-learning
description: "Use when ATLAS needs to learn about a target, vulnerability class, or hacking methodology from curated resources. Consult PortSwigger, H1 Hacktivity, Bugcrowd blog, DEF CON, and top hackers (Tomnomnom, jhaddix, Nahamsec, KUGG) to extract knowledge systematically."
version: 2.4.0
author: ATLAS
license: MIT
metadata:
  hermes:
    tags: [bug-bounty, learning, research, recon, methodology]
    related_skills: [api-hacking-methodology, source-code-security-audit, pre-submission-verification]
---

# ATLAS Continuous Learning System

## Overview

This skill is a meta-learning engine. When asked to learn about a new target, vulnerability class, or hacking technique, use it to systematically extract knowledge from curated resources — then synthesize what you find into actionable methodology. The skill encodes knowledge from PortSwigger Research, HackerOne Hacktivity, Bugcrowd Blog, DEF CON talks/books, and four major hackerschools (Tomnomnom, jhaddix, Nahamsec, KUGG).

## When to Use

- When the user explicitly says "learn / research / teach yourself about [X]"
- User says "use your skills to find better ways / approaches"
- Before starting a new bug bounty program — load this skill to orient
- **After finding potential vulnerabilities — IMMEDIATELY load `pre-submission-verification` skill and run PART 3 (Triage) + PART 4 (Victim Perspective) BEFORE evaluating or reporting.** Do not skip this step. The user will call you out if you present findings without running the gates first.

## Do Not Use For (added Aug 2026 — ECC curation pattern)

- **A single, already-known technique** — use the class-specific skill directly (idor-testing-methodology, ssrf-testing, jwt-attacks, etc.). This skill is the meta-layer for *learning*, not for executing a known attack.
- **Routine bug bounty execution** — that's `recon-to-exploitation` + class skills. Loading this skill mid-hunt for generic "what next" adds context bloat; use the pivot table in the Iterative Deep-Dive section instead.
- **Report writing** — use the submission templates + pre-submission-verification.
- If you find yourself loading this skill for a task that has a dedicated skill, load the dedicated one instead.

## Resource Hub

### Primary Research Portals

**Learning from other AI agents worldwide** (weekly sweep — see `references/worldwide-agent-skills-sweep.md`): how to research agent-skills repos across the globe (GitHub API by stars, arxiv extraction, regional language searches for Japan/China/Russia/Brazil/etc.), rank source reputability, and apply+record improvements. Used by cron `3ded7d48e350` every Tue 15:00 EAT; run on demand for the same class of research. **Sweep repo catalog (Aug 12 2026 update):** ECC (239K★, When-to-Activate pattern + agent-architecture-audit meta-skills), Anthropic-Cybersecurity-Skills (27.5K★, 817 MITRE-mapped skills), NVIDIA/SkillSpector (14.4K★, skill vuln scanner — see skill-quality-audit references/skill-security-scanners.md), snyk/agent-scan, 0xNyk/awesome-hermes-agent (Hermes-specific: hermes-dojo, oh-my-hermes, hermes-skill-factory), planning-with-files, microsoft/skills, mattpocock/skills, uphiago/recon-skills, Threekiii/Awesome-Redteam (CN), CyberStrikeus/CyberStrike, mksglu/context-mode (Think-in-Code, script-first analysis — adopted Aug 12), microsoft/SkillOpt (validation-gate skill training), SnailSploit/Claude-Red, kursku/skills (BR), superpowers-zh (CN), SkillsMP index. Details: references/worldwide-agent-skills-sweep.md

- **PortSwigger Research Articles**: https://portswigger.net/research/articles -- full article archive (2024-2026). Key recent articles by topic:
  - **SAML/OAuth auth bypass**: "The Fragile Lock: Novel Bypasses For SAML Authentication" (Dec 2025), "SAML roulette: the hacker always wins" (Mar 2025)
  - **Cookie/WAF bypass**: "Cookie Chaos: bypassing __Host/__Secure cookie prefixes" (Sep 2025), "Bypassing WAFs with the phantom $Version cookie" (Dec 2024), "Stealing HttpOnly cookies with the cookie sandwich technique" (Jan 2025)
  - **Unicode/input validation**: "Bypassing character blocklists with unicode overflows" (Jan 2025), "Concealing payloads in URL credentials" (Oct 2024), "URL validation bypass cheat sheet" (Sep 2024)
  - **HTTP smuggling**: "Introducing HTTP Anomaly Rank" (Nov 2025), "How to distinguish HTTP pipelining from request smuggling" (Aug 2025)
  - **Cache poisoning**: "Gotta cache 'em all: bending the rules of web cache exploitation" (Aug 2024)
  - **Timing attacks**: "Listen to the whispers: web timing attacks that actually work" (Aug 2024)
  - **Race conditions**: "Race condition: the atomic molecule" (Aug 2024), Single-packet attack for HTTP/2
  - **WebSocket**: "WebSocket Turbo Intruder: Unearthing the WebSocket Goldmine" (Sep 2025)
  - **CSS data exfiltration**: "Inline Style Exfiltration: leaking data with chained CSS conditionals" (Aug 2025)
  - **Email parser exploits**: "Splitting the email atom: exploiting parsers to bypass access controls" (Aug 2024)
  - **Server-side prototype pollution**: PortSwigger whitepaper — black-box detection without DoS (Gareth Heyes, 2024)
  - **Supply chain attacks**: VS Code extension marketplace risks — 500+ leaked secrets in VSCode/Open VSX extensions (Wiz Research 2025), TigerJack campaign (11+ malicious VS Code extensions, 2025)
  - **DNS rebinding for SSRF bypass**: TOCTOU-based bypass of UrlBlocker-style protections, lock.cmpxchg.io / 1r.mk services
  - Check the annual Top 10: https://portswigger.net/research/top-10-web-hacking-techniques
  - **Extraction pattern**: For each article extract → attack vector, preconditions, detection method, impact chain, bypasses for existing mitigations
- **H1 Hacktivity (API -- no auth)**: `https://api.hackerone.com/v1/hackers/hacktivity?queryString=disclosed:true&page[size]=25` -- returns JSON with severity, CWE, bounty, report structure. Use browser console or curl from the hackerone.com origin.
- **H1 Hacktivity (UI)**: https://hackerone.com/hacktivity/overview?queryString=disclosed%3Atrue -- JS-heavy but browseable
- **H1 Hacktivity disclosed report patterns** (from research of 10,000+ reports):
  - **Top paying classes**: SSRF ($6K-$17.5K), RCE/Code Injection ($12K-$20K), IDOR ($375-$2.5K), Stored XSS ($0-$18.9K via chaining), Account Takeover ($0-$15.3K via chaining)
  - **2025 trends**: Authorization flaws (IDOR, PrivEsc) RISING, AI vulns up 540%, SSRF stable high, XSS/SQLi DECLINING
  - **Chain impact**: Reports that chained bugs got 5-10x higher payouts than same-class standalone reports (IDOR alone=$0-375, IDOR+XSS chain=$15.3K ATO)
  - **Acceptance rate**: ~30-40% of submitted reports accepted (varies by program). Top barrier: scope (30-40%), dupe (most common), low impact
  - **Triage speed**: Clear title, numbered steps, request/response pairs, impact section = faster triage
- **Bugcrowd Blog**: https://www.bugcrowd.com/blog/ -- methodology, "Inside the Mind of a Hacker" interviews, case studies. Browse by topic.
- **Bugcrowd VRT**: Load the `bugcrowd-vrt` skill for classification

### Conference Archives

- **DEF CON Book List**: https://defcon.org/html/links/book-list.html -- 89 books across 5 categories
- **DEF CON Speaker Lists**: https://defcon.org/html/defcon-33/dc-33-speakers.html (and defcon-32/, defcon-31/ etc.) -- talk titles + researchers
- **DEF CON Media Archive**: https://media.defcon.org/ -- slides, audio, video
- **DEF CON Schedule**: https://defcon.org/html/defcon-33/dc-33-schedule.html

### Research by Vulnerability Class

- **Healthcare / PACS Systems**: SAML-based SSO (Shibboleth, SimpleSAMLphp), XERO Viewer, WADO API. Check IdP metadata endpoints for certs and entity IDs. Check Discovery Feeds for federation models. Known SimpleSAMLphp CVEs: CVE-2023-37282 (XXE), CVE-2020-5301 (timing). PortSwigger "The Fragile Lock" paper covers SAML XML wrapping.
- **Auth0/OIDC Bypass in Enterprise Connections**: Enterprise connections (SAML, OIDC, Azure AD, Google Workspace) in Auth0 have known bypass vectors. PortSwigger "The Fragile Lock" (Dec 2025) covers SAML parser differentials. Sentor Security disclosed an Auth0 authentication bypass. Check /co/authenticate endpoint, /dbconnections/signup, Management API /api/v2/email-templates (Liquid SSTI potential).

- **OIDC / OAuth / SSO Attacks**: Check `.well-known/openid-configuration` first for weak grants (password, implicit). See `recon-to-exploitation` skill Phase 4.6 for full methodology. Hunt for: redirect_uri manipulation, CSRF on OAuth, state validation, CORS misconfiguration on OAuth-protected APIs.
- **SSRF**: Cloud metadata endpoints, DNS rebinding, CGNAT bypass (100.64.0.0/10), IPv6-mapped IPv4 bypass, two-phase validation testing (creation vs execution), TOCTOU DNS rebinding for UrlBlocker bypass
- **IDOR**: Numeric/GUID enumeration, PUT/DELETE on same endpoint, GraphQL batch queries
- **Race Conditions**: Single-packet attack, Turbo Intruder, TOCTOU in payments/coupons

### APK Reverse Engineering for Bug Bounty

When a target has a mobile app, the APK is a primary recon source. See `references/apk-analysis-for-bug-bounty.md` for full methodology.

Key workflow: extract APK -> strings DEX for URLs/endpoints -> identify auth model classes -> probe discovered endpoints -> iterate request format based on error responses. The 422 error response often reveals the exact data model expected by the API.

**Pitfall**: clientId/clientSecret values are often obfuscated in Kotlin/Java code. Plain strings won't find them -- need jadx or apktool for full decompilation.

### APK Decompilation for Hidden API Discovery

When a target has a mobile app, decompile it — the app often contains:
- **OAuth2 client credentials** (`clientId`, `clientSecret`) — obfuscated in config classes
- **Complete API endpoint maps** via Retrofit interfaces (all endpoints in one file)
- **Staging/sandbox environment URLs** with weaker security
- **Hardcoded tokens, API keys, or Firebase config** in resources

Workflow: `jadx --show-bad-code --no-debug-info -d output/ app.apk` then search for `@f("` (Retrofit GET), `@o("` (POST) annotations to find all API endpoints.

**Worked example**: See `api-hacking-methodology` reference `nutaku-apk-analysis-and-gateway-auth.md` for the full Nutaku case — APK → credentials → gateway auth.

### GitHub Release CVE Hunting

When a target runs open-source software, compare the running version against the latest GitHub release for security patches:

1. Find version via: frontend JS, response headers, API endpoints, HTML meta tags
2. Find the GitHub repo via API search
3. List releases and check for security-related keywords
4. Compare commit diffs between versions
5. Search GitHub Security Advisories for CVEs

### Tomnomnom (Unix Pipe Philosophy)

Full philosophy, tool catalog (hacks/), six-phase pipeline, and key one-liners are in the Four Schools section below and `references/hacker-schools-detail.md`.

### Iterative Deep-Dive Pattern — 7+ Pass Methodology (Added Jun 2026)

When a target surface is large and initial probes return dead ends or rejections, use this multi-pass approach. Each pass is one research-probe-analyze cycle.

**The core loop** (repeat 7+ times or until a viable finding emerges):

```
Iteration N:
  1. LEARN — Search PortSwigger, H1 Hacktivity, DEF CON for techniques relevant
     to the current state of the target
  2. PROBE — Execute the technique against live infrastructure
  3. ANALYZE — What worked? What didn't? What changed?
  4. PIVOT — Based on results, choose the next technique or target angle
  5. REPEAT — Go to iteration N+1
```

**Practical examples of pivot decisions after each iteration:**

| Iteration Result | Next Iteration Pivot |
|-----------------|---------------------|
| Gateway API down (Connection Refused) | Switch from gateway creds to two-account app-level testing |
| All user endpoints 404 | Switch from REST probing to JS bundle analysis for hidden endpoints |
| Login CSRF-protected | Switch from authentication bypass to config disclosure |
| Main API locked down | Switch to staging/sandbox environments |
| Staging down | Switch to sibling domains (same company, different program) |
| JS reveals sibling domain | Switch to that domain's API |
| Single endpoint accessible | Deep-probe that one endpoint for all methods, params, and data |
| Registration blocked (captcha, org number, invite) | Search for official program documentation — getting-started PDFs, GitHub repos, support KBs — which often contain test account setup instructions |
| APK download blocked (Cloudflare on all mirrors) | Try adb pull from real device, Google Play credentials with apkeep, or switch to web API recon via swagger docs |
| **Program suspended** | Check if there's a related VDP or different program for the same company (e.g., Nexuzhealth main was suspended but Web PACS was active) |
| **Auth wall with no public creds** | Check credentials section of the program, FAQ for test accounts, or ask user to register |

**The stopping condition**: When you have either (a) a viable finding ready to submit through the pre-submission-verification gates, or (b) exhausted all reasonable attack surfaces with no result + clear reason why (target locked down, infrastructure dead, program scope too narrow).

Worked examples (Nutaku 7-iteration pivot, multi-target parallel recon, Visma doc-search): `references/iterative-deep-dive-examples.md`.
Where to find official program docs (registration blocked): search site:<domain> getting started / test account; program GitHub SECURITY.md; Azure blob `<company>bugbountyprod.z16.web.core.windows.net/*.pdf`; support KBs; developer portals; FireBounty. Worked examples: `references/iterative-deep-dive-examples.md`.
**Key rule**: Between each iteration, load the continuous learning skill (this one) or search PortSwigger/H1/PortSwigger for new technique ideas. Never repeat the same failed probe — the target has already told you something. Listen to what the target says about itself.

### JS Bundle API Extraction

Modern SPAs bundle all frontend logic into a few JS files. These bundles often contain:
- API endpoint paths (extract via `grep -oP '"/api/[^"]*"' bundle.js`)
- Hardcoded OAuth/Azure AD config (clientId, scopes, authority)
- Environment variables with API base URLs
- `dangerouslySetInnerHTML` usage (potential XSS)
- Hardcoded credentials, client secrets, API keys
- Sibling-domain URLs — full URLs pointing to other domains owned by the same company (use grep for `https?://` patterns, not just paths)

### Sibling-Domain Pivot via JS Bundle URL Extraction

When a target is locked down (all gateways dead, CSRF-protected, auth walls everywhere), extract FULL URLs from its JS bundles — not just API paths. These often reveal **sibling domains** owned by the same company that may be less protected:

```bash
# Extract ALL full URLs from all JS bundles
for url in $(grep -oP 'src="[^"]*\.js[^"]*"' target_page.html | cut -d'"' -f2); do
  curl -sk "$base$url" | grep -oP 'https?://[^"'"'"'\s,;)]+' | sort -u
done
```

**Real example — Nutaku to AdultForce pivot (Jun 2026):**
- Nutaku main site: all gateways down, login CSRF-protected, user endpoints 404
- Searched Nutaku's JS bundles for full URL references
- Found `https://www.adultforce.com/api/call_postback/pixel/` embedded in atlasbundle.min.js
- AdultForce is a sibling platform (same parent: Aylo, different Intigriti program: TrafficJunky)
- Probed AdultForce → found `/api/config` (11 config entries) and `/api/site` (155 brand properties with billing IDs)

**Key insight**: Companies often run multiple platforms under different programs on the same bug bounty platform. JS bundles on one platform commonly reference URLs for sibling platforms. Those siblings may have weaker security postures.

## Chaining Concept — The Meta-Skill That Makes Reports Succeed

The single biggest differentiator between accepted and rejected reports is **how impact is demonstrated through chaining**, not the bug class itself.

### Chain Patterns That Work (from H1 Hacktivity data):

| Standalone | Payout Impact | Chained With | Combined Impact |
|-----------|------|-------------|-----------------|
| IDOR (enumerate IDs) | $0-$375 | XSS → account takeover | $15,300 |
| Path traversal (read file) | $0-$500 | RCE via accessible command | $12,000-$20,000 |
| Blind SSRF (DNS only) | $0-$1,000 | Full-response SSRF via internal service | $6,000-$17,576 |
| Stored XSS (widget) | $0-$500 | Cache poisoning → widespread delivery | $18,900 |
| Token leak (by itself) | $0-$1,000 | Token used for auth → full ATO | $15,300 |

### How to Chain — Systematic Approach

After ANY finding, ask these 8 escalation questions:

1. **Enumeration**: Are IDs sequential? Can I iterate? Is there pagination?
2. **Cross-user access**: Can User A see User B's data with the same finding?
3. **Bulk extraction**: Can I script this? Is there a rate limit to bypass?
4. **Sensitive data**: API keys in config dumps? Passwords in errors? Connection strings?
5. **Chain with other findings**: Does this unlock another attack? (e.g., audit log IDOR reveals API keys → keys unlock RCE)
6. **Config-dependent severity**: Worse when a specific setting is enabled?
7. **Data lifetime**: Is exposed data still valid? (Expired keys = lower impact)
8. **Auth vs unauth**: Requires login or remote? (Higher impact = remote/no auth)

### The Victim Chain Test

For every chain:
1. V1: Who is the specific victim?
2. V2: What must they DO (or NOT do) for the attack to work?
3. V3: Can the attacker FORCE the prerequisites?
4. V4: Can you write a 3-sentence attack story?

If V4 fails, the chain isn't real.

## Four Schools of Hacking Thought (compact — full detail in `references/hacker-schools-detail.md`)

1. **Tomnomnom — Unix pipe philosophy**: every tool reads stdin / writes stdout; small tools chained with `|`; text is the API; dedupe always (`anew`); identify friction then automate. Six-phase pipeline: passive subdomain discovery → live host probing → content discovery → vuln pattern discovery → parameter fuzzing → response analysis. Tool catalog + one-liners: see reference.

2. **Nahamsec — recon-driven hunting**: more recon = easier exploitation; follow the data to forgotten assets. Workflow: CT logs → passive amass → dorking → Shodan/Censys → GitHub recon → Wayback → DNS bruteforce → takeover checks → live probing → dirsearch/ffuf → JS analysis → S3 enum → port scan → nuclei → report. Automation: `lazyrecon.sh -d domain.com`.

3. **jhaddix — systematic methodology (TBHM)**: heat-map the app; find seed domains (Crunchbase, acquisitions, ASN) then systematically expand the attack surface: ASN enum → subdomain scrape → bruteforce → alteration scanning → ports → screenshots → content discovery → nuclei → takeover → automate with Interlace.

4. **Modern 2025 ProjectDiscovery pipeline**: `subfinder -d target.com -all | dnsx -a -resp | httpx -status-code -tech-detect -title | katana -jc -kf all -aff | nuclei -t ~/nuclei-templates/` + persistent recon (cron) with `anew` + notify.

5. **Triage perspective — see the report through their eyes**: the five gates (scope, reproducibility, validity, uniqueness, impact) and the Victim Perspective Framework (V1–V6: who is the victim, what must they do, what prerequisites, can the attacker force them, actual harm, 3-sentence attack story). Self-test before every submission: *"Why would triage reject this?"* — if you can find a reason, address it or don't submit. Key references: Intigriti KB, HackerOne "View from the Other Side", Bugcrowd substates docs (full detail in the reference file).

## Evaluating Whether Known Attack Techniques Apply

After researching known attacks (CVEs, academic papers, disclosed reports), systematically verify each precondition against the target codebase. This prevents the common mistake of claiming a known attack applies when the target's defenses block it.

### The Code-Defense Matrix

For each known attack technique, build a matrix:

| Attack Component | Required Precondition | Our Target's Status | Verdict |
|-----------------|---------------------|-------------------|---------|
| Binary FS challenges | Challenge bits are independent (1 bit each) | SHA256 full-scalar challenge (256 bits) | 🚫 Not applicable — full-scalar FS prevents α-shuffle |
| Reduced dlnproof iterations | < 80 iterations | 128+ iterations (standard) | 🚫 Not applicable at full iterations |
| Missing Blum ZKP | No Paillier well-formedness proof | `paillier_generate_paillier_blum_zkp` called during setup | 🚫 Defense exists |
| Missing range proof | No bounded-value ZKP for MtA | Range proofs at every MtA step | 🚫 Defense exists |
| Weak FS key binding | Key modulus values omitted from hash | Algebraic verification still binds to specific key values | ⚠️ Seed strength is reduced but proof forgery not enabled |
| Unencrypted MTA | MTA sent in plaintext at low version | `MPC_DONT_ENCRYPT` constant = dead code, encryption always on | 🚫 Not possible regardless of version |

### Search Patterns for Precondition Verification

After identifying a known attack, search the codebase for these defenses:

```bash
# Search for ZKP defenses
grep -rn "blum_zkp\|paillier_blum\|ring_pedersen_parameters_zkp" src/

# Search for version-gated crypto hardening  
grep -rn "MPC_EXTENDED_MTA\|use_extended_seed\|strict_ciphertext" src/

# Search for abort handling (scope guards)
grep -rn "scope\|finally\|cleanup\|on_failure" src/

# Search for key validation
grep -rn "paillier_public_key_size\|ring_pedersen_public_size\|key.*check\|check.*key" src/
```

If the defense exists and is correctly applied, the known attack does NOT apply. This is a useful negative result — it means you need to find a NOVEL variant, not reuse the known technique.

### Real Example: This Session (Fireblocks mpc-lib, June 2026)

Researched attacks that DID NOT apply:
- **TSSHOCK α-shuffle**: Requires binary FS challenge bits (1 per dlnproof iteration). Our target uses full SHA256 scalar challenges. α-shuffle is infeasible.
- **CVE-2023-33241 (GG18/20 Paillier)**: Requires missing Blum ZKP. Our target has `paillier_generate_paillier_blum_zkp` at setup. Not applicable.
- **CVE-2023-33242 (Lindell17 abort)**: Requires deviating from paper by mishandling abort state. Our target uses C++ scope guards for state cleanup. Not applicable.

Researched attacks that PARTIALLY applied:
- **Fiat-Shamir key binding (MTA)**: Confirmed as valid (duplicate finding). The MTA verifier seed omits key modulus values at version < 11. But the non-extended seed is still a SHA256 full-scalar challenge — not reducible to binary guesses.

Key insight: Knowing WHY attacks DON'T apply is as valuable as knowing which ones do. It tells you exactly which attack surface is still open vs. already covered, and prevents wasted effort on blocked attack vectors.

1. **Scope analysis**: Read program scope, identify in-scope/out-of-scope assets. Big scope = heavy recon; small scope = skip most recon.
2. **Seed discovery**: Use Crunchbase, LinkedIn, WHOIS, ASN lookups to find all related domains/subdomains
3. **Passive recon**: Certificate transparency (crt.sh, certspotter), search engines, Shodan/Censys, GitHub dorking, Wayback Machine
4. **Active recon**: DNS bruteforce, subdomain takeover checks, port scanning, live host probing, GitHub release comparison for self-hosted software
5. **Content discovery**: Directory bruteforcing, JS analysis, parameter discovery, S3 bucket enumeration
6. **OIDC/OAuth recon**: Check `.well-known/openid-configuration` for weak grants, PAR requirements, scope surface
7. **Vulnerability scanning**: Nuclei templates, manual testing with Burp Suite, GitHub release CVE hunting
8. **Synthesize findings**: Document attack surface, prioritize high-value targets (SSRF, IDOR, auth bypass, logic flaws)

> **Concrete example**: See `recon-to-exploitation` skill's `references/intigriti-program-recon.md` for a full walkthrough of OIDC metadata probing, JS bundle API extraction, and GitHub version-based CVE discovery on Intigriti's own bug bounty program.

**Visma AI Assistant recon (Jun 2026)**: See `references/visma-ai-assistant-recon.md` in this skill for a full worked example. Covers OIDC scope analysis revealing hidden AI/MCP scopes (`vsn-assistant-api:chat`, `vsn-assistant-mcp:mcp`), JS bundle extraction to discover the Azure Functions backend (`vsit-aiassistant-stg.azurewebsites.net`), backend security posture probing (JWT validation, rate limits, CORS, SignalR), developer documentation surface (`developer.vismaonline.com/llms.txt`), and sandbox registration workflow.

## How to Learn About a Vulnerability Class

1. Search PortSwigger Research articles for the class (e.g., "SAML", "race condition", "cache deception")
   ...
   - Browse by category: https://portswigger.net/research/articles (sorted by date)
   - Filter by category URL: https://portswigger.net/research?category=server-side (sub in client-side, crypto, etc.)
   - Read top-10: https://portswigger.net/research/top-10-web-hacking-techniques
   - For each article, extract: attack vector, preconditions, detection method, impact chain, bypasses for existing mitigations
2. Query H1 Hacktivity API with CWE filters to see real disclosed reports:
   - API: `https://api.hackerone.com/v1/hackers/hacktivity?queryString=disclosed:true&page[size]=25`
   - Add CWE filter: `&filter[reported_to]=program_handle` (narrow to specific programs)
   - Parse response for: severity, bounty_amount, weakness (CWE), report title, summary, vulnerability_information
   - Look for patterns: which attack vectors got highest payouts, which CWEs had most accepted reports
   - Note report structure for writing your own
3. Search Bugcrowd blog for methodology articles on that class:
   - Browse: https://www.bugcrowd.com/blog/
   - Search: https://www.bugcrowd.com/blog/?s=TOPIC
   - Focus on: Inside the Mind of a Hacker series, methodology deep-dives, top researcher interviews
4. Search DEF CON talk archives for presentations on the topic
5. Look for tooling: Tomnomnom's gf patterns, nuclei templates, Burp extensions
6. Identify: What are the preconditions? How is it detected? What's the impact chain? What bypasses exist?
7. **Synthesize findings into a concrete workflow**: Don't just collect links — write numbered steps with exact commands the agent can execute against a target.

## Report Writing Best Practice (compact — full detail in `references/report-writing.md`)

Structure: title `[Vuln Type] in [Component]` (with impact) → summary → severity/CVSS → numbered copypasta-ready steps → PoC (screenshots, request/response pairs) → impact → remediation → references (CWE/CVE). Quality self-assessment: clean and easy to read? Contains everything the program owner needs? Reproducible using ONLY the steps in the report? Faster reproduction = faster triage = higher acceptance.

## Common Pitfalls

1. **Starting exploitation without adequate recon** — the more recon, the easier exploitation. Nahamsec's #1 rule.
2. **Using only automated scanners** — Brett Buerhaus specifically avoids Burp/SQLmap for manual testing. Know what scanners miss.
3. **Submitting low-hanging fruit on crowded programs** — first 30 min is a race for P2-P3 issues. Hunt in flows scanners can't reach.
4. **Writing reports with missing context** — never assume the rewarder understands impact. Demonstrate with solid PoC.
5. **Memory full** — ATLAS memory is capped at 2200 chars. Save detailed knowledge as skills, not memory entries.
6. **Not verifying claims against live infrastructure** — the user demands data-backed reasoning. "Convince me."
7. **Forgetting VRT classification** — load bugcrowd-vrt skill before submitting. VRT = CLASSIFICATION, not validation.
11. **Relying on summaries instead of original sources** — When referencing books, PDFs, or papers, extract the table of contents and relevant sections directly from the source file. Chapter summaries are often inaccurate or renumbered across editions. A book's table of contents in the actual PDF is authoritative — not a summary you wrote in a previous session or inferred from memory. **Example corrected this session**: Real-World Bug Hunting Ch16 is "Insecure Direct Object References" (not "Carriage Return Line Feed Injection" which is Ch6). Verify by extracting the actual PDF text and searching for the chapter heading.

12. **Deep-diving an asset before checking its bounty status** — Program scope tables often mix paid, no-bounty, and out-of-scope assets. Investing time on a No-bounty asset (like Rancher on Ninja Kiwi) wastes multiple iterations. Before deep-diving any endpoint, check its tier/status in the program's asset table. If it's No bounty or Out of scope, skip it.

13. **Spring Boot list-vs-detail auth asymmetry** — When probing Spring Data REST endpoints, test list and detail endpoints separately. GET list may be public while GET {id} is protected, or vice versa. Always check: GET list, GET {id}, POST, PUT {id}, DELETE {id}. The asymmetry (write protected, read public) is a strong signal auth was intended but the developer missed a method.

14. **OIDC E-Commerce Recon (Coolblue pattern)** — Modern e-commerce sites (Next.js + OIDC) have a standardized recon flow:
    - Check `.well-known/openid-configuration` for OIDC issuer info (scopes, grant types, claims, JWKS)
    - Look for custom scopes like `ucp:scopes:checkout_session`, `openid:customerid`, `openid:identityroleid` — these reveal business logic boundaries
    - Check grant_types for `token-exchange` — potential privilege escalation vector
    - Next.js: check `__NEXT_DATA__` in HTML, client-side route extraction
    - WebSocket endpoints: probe root for route discovery, check authenticated paths for 403 vs 426
    - User-Agent matters: CloudFront blocks curl default UA, allows Chrome UA
    - VPN for rate-limit bypass: WireGuard to EU IP can jump rate limits (0.3 -> 2 req/s)
    - Two-step login flows (email then password) require browser for CAPTCHA

15. **Data Weaponization Workflow** — Before submitting a data-exposure finding, try to USE the data:
    a. Internal IDs: try as auth headers, URL params, API keys, payment URLs
    b. S3 bucket paths: enumerate for config/.env/credentials; 403 vs 404 reveals existence
    c. GA accounts: shared across sites reveals business unit relationships
    d. Broken URL flags (validUrl false): check DNS, CNAME takeover, HTTP codes
    e. Testing/staging env names: probe for weaker security on discovered subdomains
    f. Legacy IDs: check if old IDs still work on third-party systems
    g. Internal codes: try as URL segments, query params, auth headers
    h. Mimir WebSocket probe: connect to wss://host/, send JSON with route field. 400=wrong route format, 403=needs auth

## Verification Checklist

- [ ] Loaded this skill before starting knowledge extraction
- [ ] Checked memory for existing knowledge about the target/class
- [ ] Queried PortSwigger Research for relevant articles
- [ ] Queried H1 Hacktivity API for real disclosed reports
- [ ] Checked Bugcrowd blog for methodology content
- [ ] Searched DEF CON archives for relevant talks
- [ ] Referenced GitHub repos from jhaddix/Nahamsec/Tomnomnom/KUGG as applicable
- [ ] Applied the Four Schools framework to the specific task
- [ ] Saved durable findings as skills (not memory)
- [ ] Only presented claims backed by verifiable evidence

## Impact Gate — Data Sensitivity Assessment (compact — full detail in `references/impact-gate.md`)

Before submitting any finding whose harm comes from exposed data, classify the data: **P1 credentials / P2 PII / P3 financial infrastructure** = submittable; **P4 business metadata / P5 operational / P6 public** = Informative. Three-question test: (1) is the data P1–P3? (2) does it enable a DIRECT action (login, access another account, process a payment)? (3) can you write a victim story where the harm is not "a competitor could see this"? The mechanism doesn't save you — the DATA does. Full P1–P6 table, mistake scenarios, self-diagnosis, and the shallow-conclusion trap / meta-analysis workflow: `references/impact-gate.md`.

## Architecture-Aware Hunting (compact — full detail in `references/architecture-aware-hunting.md`)

- **Two-layer auth trap**: a gateway accepting any token is NOT an auth bypass if the data/service layer independently validates. Token-accepting component ALSO serves protected data = finding; token forwarded to an independently-validating downstream = architecture.
- **The Title Test**: every finding title must end with an action an attacker can take ("IDOR on /api/invoices leading to view any user billing data" — not "Insecure Direct Object Reference").
- **Data-classification-first**: hunt in reverse — what PII/credentials does this system hold, where does it flow, are those endpoints protected? Not: find endpoint → probe → theorize impact.
- **VDP rules** and underutilized info sources: `references/architecture-aware-hunting.md`.

## Context Discipline — Think in Code (added Aug 12 2026, from mksglu/context-mode 19.7K★ + HN #1)

Large raw tool output (scan JSONs, page dumps, repo trees, recon lists) burns context fast. Rule: **the agent programs the analysis, it does not read the data.**

- Before reading a large blob into context, write a one-liner script that computes what you need and logs ONLY the result (counts, matches, unique values, top findings).
- One script that answers the question replaces 10-50 tool calls — ~100x context saving.
- Examples done right: `python3 -c` JSON parsing of scanner output (this sweep), `grep -oP` over JS bundles, `curl | jq` field extraction.
- Enforce where DATA goes (keep it out of context), not how the model writes its final answer — aggressive brevity prompts degrade reasoning (Moonshot kimi-k2.5 benchmark regression).

## Lesson Bank (MANDATORY)

After any finding, test result, or submission outcome from this methodology:
- Append a dated entry to `~/Dev/ATLAS-LEARNINGS/LESSONS.md` under the relevant section
- If MEMORY.md is near full, compress to a pointer (`See LESSONS.md`)
- Load `atlas-lesson-bank` skill for the full workflow
- Every rejection is data; every accepted report is a pattern worth keeping
