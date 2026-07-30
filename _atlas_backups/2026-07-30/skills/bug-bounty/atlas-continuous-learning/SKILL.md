---
name: atlas-continuous-learning
description: "Use when ATLAS needs to learn about a target, vulnerability class, or hacking methodology from curated resources. Consult PortSwigger, H1 Hacktivity, Bugcrowd blog, DEF CON, and top hackers (Tomnomnom, jhaddix, Nahamsec, KUGG) to extract knowledge systematically."
version: 2.3.0
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

## Resource Hub

### Primary Research Portals

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

### GitHub Methodology Repos

- **jhaddix/tbhm**: https://github.com/jhaddix/tbhm — The Bug Hunter's Methodology v4 (12+ sections: philosophy, discovery, mapping, auth, XSS, SQLi, file upload, CSRF, priv esc, mobile, IDOR)
- **jhaddix/domain**: https://github.com/jhaddix/domain — enumall.py subdomain enumeration script
- **jhaddix/pentest-bookmarks**: https://github.com/jhaddix/pentest-bookmarks — categorized penetration testing bookmarks
- **jhaddix/CSPReconGO**: https://github.com/jhaddix/CSPReconGO — CSP header domain extraction
- **jhaddix/awsScrape**: https://github.com/jhaddix/awsScrape — SSL cert scraping from AWS IP ranges
- **jhaddix/scripts**: https://github.com/jhaddix/scripts — mass.sh, bust.sh, header XSS generators
- **Nahamsec/lazyrecon**: https://github.com/nahamsec/lazyrecon — full automated recon pipeline
- **Nahamsec/bbht**: https://github.com/nahamsec/bbht — one-click bug bounty toolkit installer
- **Nahamsec/recon_profile**: https://github.com/nahamsec/recon_profile — bash aliases for recon
- **Nahamsec/Resources-for-Beginner-Bug-Bounty-Hunters**: https://github.com/nahamsec/Resources-for-Beginner-Bug-Bounty-Hunters — 12k-star curated learning list
- **Nahamsec/JSParser**: https://github.com/nahamsec/JSParser — JS endpoint extraction
- **Nahamsec/lazys3**: https://github.com/nahamsec/lazys3 — S3 bucket brute-forcer
- **Nahamsec/HostileSubBruteforcer**: https://github.com/nahamsec/HostileSubBruteforcer — subdomain bruteforce + takeover detection

### Tomnomnom Tool Ecosystem (Unix Pipe Philosophy)

**Core belief**: Every tool reads from stdin, writes to stdout. No tool is standalone — they're pipe fittings. Simple text processing with grep/sort/uniq is the universal API.

**Tool-building philosophy** (the meta-lesson from tomnomnom's entire approach):
- **Small > large**: Write 50 tiny tools (hacks/ repo), not one monolith. Each does ONE thing well.
- **Chain, don't wrap**: Don't build "the ultimate recon tool" — build pipe-fitting tools and chain them.
- **Text is the API**: JSON, HTML, URLs — they're all text. grep/sort/uniq/sed/awk process them all.
- **Read first, then write**: assetfinder reads crt.sh, httprobe reads responses, waybackurls reads archive.org, unfurl reads URLs. Data comes first; analysis second.
- **Dedupe always**: `anew` is tiny but critical — persistent recon without duplicates.
- **Don't fight tools**: If Burp is slow, write curl. If a website has auth, write a cookie manager. Build the tool that makes the specific task trivial.

**Concrete example — how a tool is born**: Tomnomnom noticed himself repeatedly grepping JSON responses for interesting fields. He wrote `gron` to flatten JSON to grep-friendly lines. He noticed himself repeatedly extracting parts of URLs. He wrote `unfurl`. Every tool came from a personal pain point, not a theoretical need.

**Lesson for AI agents**: When you find yourself repeating a 3+ step terminal sequence, that's a candidate for a bash function or mini-script. Save it as a skill reference/script file. Don't retype it — package it.

**Core tools** (github.com/tomnomnom):
- `assetfinder` — passive subdomain discovery (crt.sh, CertSpotter, etc.)
- `httprobe` — probe for working HTTP/HTTPS servers
- `waybackurls` — fetch historical URLs from Wayback Machine
- `meg` — breadth-first HTTP fetcher (one path × all hosts)
- `gf` — grep wrapper with named patterns (php-sources, takeovers, base64, etc.)
- `gron` — make JSON greppable (flatten to assignment statements)
- `unfurl` — extract URL components (domains, paths, keys, values)
- `qsreplace` — replace query string values for fuzzing
- `fff` — fire-and-forget HTTP fetcher (fast, resource-heavy)
- `comb` — cartesian product combiner
- `kxss` — XSS parameter discovery
- `html-tool` — HTML element extraction
- `tok` — wordlist tokenizer from text
- `anew` — deduplication append (for persistent recon)

**Hacks collection** (github.com/tomnomnom/hacks — 50+ mini tools):
`anti-burl`, `b64d`, `bbdb`, `bbinit`, `check-cnames`, `chromeredir`, `compres`, `cors-blimey`, `domaintree`, `ettu`, `filter-resolved`, `get-title`, `geteventlisteners`, `ghtool`, `gittrees`, `goreqs`, `gronval`, `html-comments`, `htmlattribs`, `inscope`, `jsb-inplace`, `jsstrings`, `kxss`, `lsinteresting`, `manyreqs`, `mirror`, `paster`, `perms`, `phpreqs`, `pupjs`, `remove-subdomains`, `sectxt-parser`, `strip-wildcards`, `structured-scopes`, `subs`, `tojson`, `tok`, `unimap`, `unisub`, `uresolve`, `urinteresting`, `urlteamdl`, `webpaste`

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

**Real example — Nutaku to AdultForce pivots (Jun 2026, 7 iterations):**\n1. Probed Nutaku gateway creds → gateway down\n2. Probed Nutaku _xd API → login protected, user endpoints 404\n3. Probed Nutaku OSAPI → root responds but REST paths 404\n4. Probed Nutaku signup, members pages → CSRF/recaptcha locked\n5. Searched Nutaku JS files → found AdultForce URL embedded\n6. Probed AdultForce → found /api/config publicly accessible (operational messages)\n7. Probed /api/site → 155 brand sites with ProBiller billing IDs, S3 paths, GA accounts discovered (real finding)

**Multi-target parallel recon (Jun 2026 — Skoda + Auth0 + Torfs):**
When the user wants to pick the best target from multiple platforms, run parallel recon across platforms simultaneously:
1. **List candidates** from Intigriti (sorted by last update), Bugcrowd (sorted by max payout), comparing scope, payout, competition level
2. **Shortlist** 1 target per platform, prioritizing: public programs > registered, fresh updates, low researcher count, tech stack match, availability of dedicated skills
3. **Probe both** in parallel on first iteration — surface recon on target A + surface recon on target B
4. **Pivot strategy**: When target A hits a blocker (credential wall, app-only, registration blocked), escalate to the next candidate on same platform while keeping target B active
5. **Continuous learning** between each iteration — research attack vectors discovered during probing (e.g., SFCC WebDAV, Auth0 Liquid SSTI, SAML bypass)
6. **Cross-pollinate** findings — if one target's recon reveals a technique applicable to another (e.g., OIDC scope analysis from Visma applied to Skoda's identity.vwgroup.io), apply it

**Real example — Visma → Auth0 → Torfs → Nexuzhealth (Jun 2026):**
1. Visma: Probed AI Assistant, OIDC, MCP server — blocked by credential requirements
2. Pivoted to Torfs (SFCC): Mapped full OCAPI/SLAS/SCAPI surface, registered account, extracted JWT with 26 scopes
3. Pivoted to Nexuzhealth: Mapped LiquidFiles, SimpleSAMLphp IdP, Jira dashboards, XERO PACS viewer
4. Each pivot was informed by the prior target's auth pattern (SAML/OIDC similarities)
5. Key learning: Multi-target parallel recon reveals patterns that single-target focus misses

**Real example — Visma documentation search (Jun 2026):**
1. Probed VismaOnline stage → login gated, no self-registration on identity server
2. Probed testing.maventa.com → registration blocked by valid org number requirement
3. Searched official Visma bug bounty docs → found getting-started PDFs on Azure blob storage (`vismabugbountyprod.z16.web.core.windows.net`)
4. PDFs revealed: student signup URL `https://admin.stage.vismaonline.com/Customer/StudentSignup.aspx?uilang=en` with training code `g004t`
5. PDFs also revealed: Developer Portal test app naming conventions, OAuth2 client_credentials test APIs, and scope documentation

**Where to find official program docs (when registration is blocked):**
- Search `site:<program-domain> getting started` or `site:<program-domain> test account`
- Search `<program-name> bug bounty getting started pdf` via web search
- Check the program's GitHub repos for README/SECURITY.md files with test instructions
- Check Azure blob storage: often `<company>bugbountyprod.z16.web.core.windows.net/*.pdf`
- Check program support knowledge bases (e.g. `support.maventa.fi`, `community.visma.com`)
- Check the program's developer portal for sandbox/test environment documentation
- Search for the program name on FireBounty, which sometimes links to test credentials docs

**Key rule**: Between each iteration, load the continuous learning skill (this one) or search PortSwigger/H1/PortSwigger for new technique ideas. Never repeat the same failed probe — the target has already told you something. Listen to what the target says about itself.

### Subdomain Discovery Services

- **C99 Subdomain Finder**: https://subdomainfinder.c99.nl/scans/YYYY-MM-DD/<domain> — 50+ passive data sources, finds 100-1000+ subdomains per scan. Has years of historical scans with compare mode. Essential for finding subdomains NOT behind the same WAF as the primary domain. Results include IP addresses for each subdomain.
- **SubDomainRadar.io**: https://subdomainradar.io/task/<uuid> — Alternative passive subdomain scanner for cross-referencing C99 results.

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

### KUGG (Christoffer Jerkeby) Resources

- **GitHub**: https://github.com/kugg
- **Blog**: https://www.jerkeby.se/newsletter/index.html
- **Notable research**: F5 BIG-IP iRule RCE, Log4Shell testing (log4shellverify), supply chain attack visualization (supplychaingraphs), prototype pollution, memory corruption/ROP, fuzzing methodology

### Notable Talks to Search For

- Nahamsec "Doing Recon Like a Boss" (LevelUp 2017)
- Nahamsec "It's the Little Things" (BSides Portland 2018)
- Nahamsec "Owning the Clout Through SSRF and PDF Generators" (DEF CON 27)
- jhaddix "Bug Bounty Methodology" (various)
- Tomnomnom "Bug Bounties With Bash" (various)
- James Kettle "HTTP/1.1 Must Die: The Desync Endgame" (DEF CON 33 / Black Hat 2025)
- All DEF CON 33 web security talks listed in resources

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

## Four Schools of Hacking Thought

### 1. Tomnomnom — Unix Pipe Philosophy

**Core belief**: Every tool reads from stdin, writes to stdout. No tool is standalone — they're pipe fittings. Simple text processing with grep/sort/uniq is the universal API.

**Tool-building philosophy** (the meta-lesson from tomnomnom's entire approach):
- **Small > large**: Write 50 tiny tools (hacks/ repo), not one monolith. Each does ONE thing well.
- **Chain, don't wrap**: Don't build "the ultimate recon tool" — build pipe-fitting tools and chain them with `|`.
- **Text is the API**: JSON, HTML, URLs — they're all text. grep/sort/uniq/sed/awk process them all. `gron` exists because flattening JSON to grep-able statements is more useful than parsing JSON in isolation.
- **Read first, then write**: assetfinder reads crt.sh, httprobe reads responses, waybackurls reads archive.org, unfurl reads URLs. Data first; analysis second.
- **Dedupe always**: `anew` is tiny but critical — persistent recon without duplicates.
- **Don't fight tools**: If Burp is slow, write curl. If a website has auth, write a cookie manager. Build the tool that makes the specific task trivial.
- **Identify friction, then automate**: Every tool came from a personal pain point. `gron` = tired of grepping JSON. `unfurl` = tired of writing URL regexes. `gf` = tired of mistyping grep patterns. Identify YOUR friction points and script them.

**Lesson for AI agents**: When you find yourself repeating a 3+ step terminal sequence, that's a candidate for a bash function or mini-script. Save it as a skill reference/script file. Don't retype it — package it.

**The six-phase workflow pipeline**:
```
Phase 1: Passive Subdomain Discovery
  assetfinder --subs-only target.com > subs.txt
  cat subs.txt | waybackurls | unfurl domains | sort -u >> subs.txt

Phase 2: Live Host Probing
  cat subs.txt | httprobe > alive.txt

Phase 3: Content Discovery
  meg — breadth-first scan (one path x all hosts)
  cat alive.txt | fff -S -o output/

Phase 4: Vulnerability Pattern Discovery
  gf takeovers / gf php-sources / gf sec
  kxss — XSS parameter discovery
  html-comments — extract HTML comments from responses

Phase 5: Parameter Fuzzing
  cat urls.txt | grep '=' | qsreplace '"><script>alert(1)</script>' > xss-test.txt

Phase 6: Response Analysis
  gron — flatten JSON for grepping
  compres — diff HTTP responses
```

**Key one-liners**:
```bash
# Complete passive recon (one pipeline)
assetfinder --subs-only target.com | httprobe | tee alive.txt | waybackurls | tee urls.txt

# XSS fuzzing pipeline
assetfinder target.com | waybackurls | grep '=' | qsreplace '"><script>alert(1)</script>'

# All unique subdomains from wayback data
assetfinder target.com | waybackurls | unfurl domains | sort -u

# JSON exploration for secrets
curl -s API_URL | gron | grep -i "token\|secret\|key"

# Open redirect check
cat urls.txt | grep -a -i '=http' | qsreplace 'http://evil.com'

# JS file extraction for API endpoints
waybackurls target.com | grep '\.js$' | tee js-files.txt

# Deduplicated persistent recon
subfinder -d target.com -all -silent | anew -q subs_monitor.txt

# Wordlist generation (feedback loop)
cat urls.txt paths.txt scripts.js | tok | sort -u | anew wordlist.txt
```

### 2. Nahamsec (Ben Sadeghipour) — Recon-Driven Hunting

**Core belief**: The more time on recon, the easier exploitation becomes. Follow the data to find forgotten assets.

**Workflow**:
```
1. Certificate Transparency logs (crt.sh, certspotter, crtndstry)
2. Passive Amass enumeration
3. Search engines / Google dorking
4. Shodan/Censys — internet-facing assets
5. GitHub recon — leaked creds, API keys, internal endpoints
6. Wayback Machine — historical endpoints, JS files
7. DNS bruteforce (massdns + HostileSubBruteforcer)
8. Subdomain takeover detection (CNAME analysis)
9. Live host probing (httprobe)
10. Directory bruteforcing (dirsearch/ffuf)
11. JS analysis (JSParser) — extract endpoints, params
12. S3 bucket enumeration (lazys3)
13. Port scanning + screenshots
14. Nuclei template scanning
15. HTML report generation
```

**Automation**: `lazyrecon.sh -d domain.com` ties it all together in one script.

### 3. jhaddix (Jason Haddix) — Systematic Methodology (TBHM)

**Core belief**: "Heat map" the application to identify risky behaviors. Find seed domains first, then systematically expand the attack surface.

**TBHM v4 workflow**:
```
1. Finding Seeds — Crunchbase (acquisitions), LinkedIn, Wikipedia, amass intel -asn <ASN>, metabigor
2. ASN Enumeration — bgp.he.net, Shodan, Censys
3. Subdomain Scraping — Amass, Subfinder, Assetfinder, Censys, crt.sh, VirusTotal, AlienVault OTX
4. Subdomain Bruteforcing — massdns + all.txt wordlist, shuffledns
5. Alteration Scanning — altdns, dnsgen (dev-, api-, admin- variants)
6. Port Scanning — masscan (full 65535), naabu, nmap
7. Screenshotting — Gowitness, EyeWitness
8. Content Discovery — dirsearch, ffuf, meg
9. Vulnerability Scanning — Nuclei templates
10. Subdomain Takeover — nuclei, subjack, can-i-take-over-xyz
11. Automation — Interlace, bash scripts
```

### 4. Modern 2025 ProjectDiscovery Pipeline

**Core toolchain**:
```bash
subfinder -d target.com -all | dnsx -a -resp | httpx -status-code -tech-detect -title | \
  tee alive.txt | katana -jc -kf all -aff | nuclei -t ~/nuclei-templates/
```

**Persistent recon** (cron-based):
```bash
#!/bin/bash
subfinder -d $1 -all -silent | anew -q subs_monitor.txt | notify
```

### 5. Triage Perspective — See the Report Through Their Eyes

**Core belief**: The difference between Informative and Triaged is understanding what Triage needs. Every finding should be evaluated from both the attacker's AND the triager's viewpoint before submission. This is not about writing reports — it's about a mindset shift: imagine you ARE the triage analyst who has 50 reports to review today. Which ones do you accept? Which do you reject?

**Key triage decision criteria** (from HackerOne analyst blog, Intigriti KB, Bugcrowd docs):

1. **Scope gate** — Is the asset AND vuln type explicitly in scope? This is #1 rejection reason (~30-40%).
2. **Reproducibility gate** — Can triage follow the steps and reproduce the finding WITHOUT your session/account? Vague steps = Needs More Info → eventual N/A.
3. **Vulnerability validity gate** — Is this a real security issue, or is it expected behavior? Self-XSS, missing headers, version disclosure, CORS `*` are well-known non-issues.
4. **Uniqueness gate** — Has this been submitted before? Speed matters for common bugs.
5. **Impact gate** — Is there REAL harm demonstrated, not theorized? Data accessed? Money lost? Accounts taken over?

**Victim Perspective Framework** (load `pre-submission-verification` skill and run PART 4): For every finding, ask:
- V1: Who is the specific human victim?
- V2: What must the victim DO for the attack to work?
- V3: What prerequisites must be true?
- V4: Can the attacker FORCE those prerequisites?
- V5: What is the ACTUAL harm to that victim?
- V6: Can you write a 3-sentence attack scenario story?

If V6 is impossible, the finding isn't developed enough to submit.

**The self-test**: Before submitting, ask "Why would triage reject this?" — then answer honestly. If you can find a reason, address it before submitting or don't submit at all.

**Key references**: Intigriti KB (kb.intigriti.com — Handling Submissions, Submission States, Triage Standards), HackerOne "View from the Other Side" blog, Bugcrowd "Understanding Substates" docs.

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

## Report Writing Best Practice (from Bugcrowd Top Hackers)

**Structure**:
1. Title: `[Vuln Type] in [Component]` — descriptive, includes impact
2. Summary: What the bug is, why it matters
3. Severity: CVSS score, risk rating
4. Steps to Reproduce: Numbered, copypasta-ready, no assumptions
5. PoC: Screenshots, video, HTTP request/response pairs
6. Impact: What attacker can achieve (data access, RCE, ATO)
7. Remediation: Suggested fix
8. References: CWE, CVE, similar writeups

**Quality self-assessment** (Brett Buerhaus):
1. Is it formatted clean and easy to read?
2. Does it contain everything the program owner needs?
3. Can someone reproduce the finding using ONLY the steps in the report?

**Key tips**: Faster reproduction = faster triage = higher acceptance. A well-written report can turn $100 into $1,000. Build reputation through consistent quality.

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

## Impact Gate — Data Sensitivity Assessment (Added Jun 2026 from AdultForce rejection)

Before submitting ANY finding where the harm comes from exposed data, run this data classification check. A technically valid vulnerability with low-value data will be Informative — the mechanism is irrelevant if the data doesn't matter.

### Data Value Classification

| Class | Examples | Impact | Safe to Submit? |
|-------|----------|--------|----------------|
| **P1 — Credentials** | API keys, tokens, passwords, client secrets that still work | Direct auth bypass | ✅ Yes |
| **P2 — PII** | Email addresses, password hashes, payment info, SSN, phone | Identity theft, fraud | ✅ Yes |
| **P3 — Financial Infrastructure** | Payment gateway account IDs, revenue figures, refund handling data | Billing fraud, competitor intel | ⚠️ Maybe — depends on whether the ID enables action vs. is just a label |
| **P4 — Business Metadata** | Internal IDs (ProBiller IDs), infrastructure paths (S3 URLs), account codes, organizational structure | Competitor intelligence | ❌ No — this is what triage calls "without significant and executable impact" |
| **P5 — Operational Data** | Maintenance messages, feature flags, public-facing configuration | None | ❌ No |
| **P6 — Public/Intended** | Thumbnails, game titles, public content | None | ❌ No |

### The 3-Question Test Before Submission

For every finding where the impact is "data exposed without auth":

1. **Is the data P1, P2, or P3?** (Credentials, PII, or truly actionable financial data?)
   - If yes → submit with impact demonstrated
   - If no → do NOT submit — it will be Informative

2. **Does the exposed data enable a DIRECT action?**
   - Can it be used to log in? (token, password)
   - Can it be used to access someone else's account? (user ID pattern, email)
   - Can it be used to process a payment? (billing token, card data)
   - If no → the data is "metadata" — do NOT submit

3. **Can you write a victim story where the harm is NOT "competitor could see this"?**
   - "A user's email was exposed" → PII, valid
   - "A billing ID was exposed and could be used to..." but if you can't finish that sentence, it's metadata
   - "A competitor could see the brand portfolio" → Informative (this is exactly what peaches said)

### When the Finding Mechanism Is Valid but the Data Isn't

Auth gap on business metadata → **Informative** (AdultForce)
Auth gap that leaks PII → **Accepted** (what we need to find next)
SSRF that resolves DNS → **Informative** (no data captured)
SSRF that reads cloud metadata → **Accepted** (credentials captured)

The lesson: the MECHANISM doesn't save you. The DATA does. Find the mechanism AND valuable data together.

### Common Mistake Scenarios (Same Pattern)

| Finding Type | Mistake | Why It Fails | Fix |
|-------------|---------|-------------|-----|
| Auth gap on internal API | Reporting the gap without checking if the data behind it is valuable | "Business metadata" → Informative | Verify the data is P1/P2/P3 before submitting |
| Auth bypass but authz blocks access | Proving the auth gate is broken (any token works) but the tool behind it still returns permission error, not data | "Endpoint reachability, not a practical authentication bypass" → N/A | Need a credential that passes BOTH layers, OR find a method/resource that skips authorization entirely. Auth bypass is real but impact chain incomplete |
| Hardcoded credential in APK | Reporting the credential without demonstrating what it unlocks | "Public client credentials" → Informative | Use the credential to access something — if nothing accessible, don't report |
| Open S3 bucket | Reporting the bucket without checking what's in it | "Public thumbnails" → Informative | List the bucket contents first. Empty or public assets → don't report |
| Leaked API key | Reporting the key without testing it against the API | "Defunct or scoped key" → Informative | Test `curl -H "Authorization: Bearer $KEY"` first. 401 → don't report |
| CORS misconfiguration | Reporting permissive CORS without demonstrating cross-origin data theft | "No demonstrated exploit" → Informative | If you can't steal user data cross-origin, don't report |
| Verbose error messages | Reporting stack traces without demonstrating exploitable information | "No sensitive data in error" → Informative | If the error doesn't contain tokens/PII/query strings, don't report |
| IDOR on non-sensitive data | Reporting cross-user access to metadata (game titles, preferences) | "No security or financial impact" → OOS (Nutaku Finding 2) | Only submit IDOR on PII or financial data |
| Version disclosure | Reporting old software version without a working CVE PoC | "No executable exploit" → Informative | Find and run the PoC first, or don't report |
| Weak password policy | Reporting password complexity rules without account compromise | "Best practices violation" → OOS/Informative | Never report this — standard exclusion across all programs |
| SSRF with DNS only | Reporting outbound DNS without capturing HTTP response | "Blind without impact" → Informative | Chain with a service that echoes request body (interact.sh, Burp Collaborator) |

### The Pre-Submission Self-Diagnosis

Before submitting, ask yourself:

> *"If I were peaches reading this, would I say 'this is business metadata' or 'this exposes credentials/PII'?"*

If the answer is "business metadata" — do NOT submit. Find a better endpoint or chain with something else.

### The Shallow Conclusion Trap — Meta-Analysis Workflow (Added Jun 2026)

When the user pushes back on a conclusion as "a bit lacking" or "not always true,"
the shallow-conclusion detector fired. Run the meta-analysis workflow BEFORE
presenting the refined conclusion:

1. Have I checked at least 5 different source types? (CVEs, disclosed reports,
   top hunter writeups, program rules, live tests)
2. Can I name specific counterexamples that contradict my initial theory?
3. Did I find a contradiction that forced me to refine?
4. Does the conclusion fit on a bumper sticker? If yes, it's probably too simple.

Reference: `references/meta-analysis-workflow.md` in this skill for the full
10+ iteration deep-dive process.

## Architecture-Aware Hunting (Added Jun 2026 from meta-analysis)

### The Two-Layer Auth Trap

Many modern APIs use a two-layer architecture:
- Layer 1 — Session/Gateway: Accepts connections, manages sessions, lenient auth
- Layer 2 — Data/Service: Enforces actual auth, validates tokens, serves data

Pattern: Layer 1 accepting any token is NOT an auth bypass if Layer 2 independently validates. Compare with CVE-2024-8954 (Composio) where the same broken check was the ONLY gate.

To distinguish a real auth bypass from architecture:
1. Does the token-accepting component ALSO serve protected data? Yes = finding
2. Does the token get forwarded to a downstream service that independently validates? Yes = architecture, not finding
3. Can you find a method/endpoint on Layer 1 that DOESNT forward to Layer 2? Maybe = check for direct data access

### The Title Test

Every finding must pass this test: The title must end with an action an attacker can take.

| Pass | Fail |
|------|------|
| Auth bypass leading to read customer PII from /api/users | Authentication Bypass in MCP Server |
| IDOR on /api/invoices leading to view any user billing data | Insecure Direct Object Reference |
| SSRF via webhook URL leading to read cloud metadata | Server-Side Request Forgery |
| CORS misconfig leading to exfiltrate user data cross-origin | CORS allows any origin |

If you can't write a title that ends with "leads to [action]", the finding isn't developed.

### The Data Classification First Principle

Reverse the order of hunting:
1. First: What PII, credentials, or actionable data does this system handle?
2. Second: Where does that data flow? Which endpoints serve it?
3. Third: Are those endpoints properly protected?

Not:
1. Find interesting endpoint > probe > find behavior > theorize impact > submit

### VDP-Specific Considerations

VDPs accept a wider range of findings than paid bounties but still require:
- Demonstrable security impact (not best-practice violations)
- Clear reproduction steps
- No prohibited testing (DoS, social engineering, physical)
- For no-payout VDPs: focus on high confidence findings with CLEAR impact
  - Exposed admin panels with default creds
  - Confirmed SSRF to metadata endpoints
  - PII leakage via IDOR
  - Subdomain takeover with active service
  - Publicly accessible internal tools/dashboards
- Avoid for VDPs: missing headers, version disclosure, theoretical issues, business metadata

### Information Sources We Underutilized

| Source | What It Finds | Why We Missed It |
|--------|--------------|------------------|
| Shodan/Censys | Exposed instances, dev servers, admin panels | We probed live domains directly instead of discovering shadow IT |
| GitHub dorking | Leaked creds, internal docs, config files | Limited use; didn't search for target in code comments/commits |
| Source code analysis | Auth annotations, middleware chains, config-guarded features | Only done for open-source targets; should check npm/PyPI for target SDKs |
| Nuclei templates | Known CVEs and misconfigs | Rarely ran; could find standing vulns faster |
| Wayback Machine diffs | Removed endpoints, historical JS with API routes | Used wayback URLs but didn't diff versions for security-relevant changes |
| C99 subdomain scans | 50+ data sources for subdomain discovery | Didn't use on most targets |
| OIDC well-known configs | Custom scopes, grant types, JWKS URLs | Only checked on Coolblue; should be standard recon step |
