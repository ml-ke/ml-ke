---
name: atlas-continuous-learning
description: "Use when ATLAS needs to learn about a target, vulnerability class, or hacking methodology from curated resources. Consult PortSwigger, H1 Hacktivity, Bugcrowd blog, DEF CON, and top hackers (Tomnomnom, jhaddix, Nahamsec, KUGG) to extract knowledge systematically."
version: 2.1.0
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
  - **Race conditions**: "Race condition: the atomic molecule" (Aug 2024)
  - **WebSocket**: "WebSocket Turbo Intruder: Unearthing the WebSocket Goldmine" (Sep 2025)
  - **CSS data exfiltration**: "Inline Style Exfiltration: leaking data with chained CSS conditionals" (Aug 2025)
  - **Email parser exploits**: "Splitting the email atom: exploiting parsers to bypass access controls" (Aug 2024)
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

- **OIDC / OAuth / SSO Attacks**: Check `.well-known/openid-configuration` first for weak grants (password, implicit). See `recon-to-exploitation` skill Phase 4.6 for full methodology. Hunt for: redirect_uri manipulation, CSRF on OAuth, state validation, CORS misconfiguration on OAuth-protected APIs.
- **SSRF**: Cloud metadata endpoints, DNS rebinding, CGNAT bypass (100.64.0.0/10)
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

### Tomnomnom site**: https://tomhudson.co.uk/

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

## How to Learn About a New Target

1. **Scope analysis**: Read program scope, identify in-scope/out-of-scope assets. Big scope = heavy recon; small scope = skip most recon.
2. **Seed discovery**: Use Crunchbase, LinkedIn, WHOIS, ASN lookups to find all related domains/subdomains
3. **Passive recon**: Certificate transparency (crt.sh, certspotter), search engines, Shodan/Censys, GitHub dorking, Wayback Machine
4. **Active recon**: DNS bruteforce, subdomain takeover checks, port scanning, live host probing, GitHub release comparison for self-hosted software
5. **Content discovery**: Directory bruteforcing, JS analysis, parameter discovery, S3 bucket enumeration
6. **OIDC/OAuth recon**: Check `.well-known/openid-configuration` for weak grants, PAR requirements, scope surface
7. **Vulnerability scanning**: Nuclei templates, manual testing with Burp Suite, GitHub release CVE hunting
8. **Synthesize findings**: Document attack surface, prioritize high-value targets (SSRF, IDOR, auth bypass, logic flaws)

> **Concrete example**: See `recon-to-exploitation` skill's `references/intigriti-program-recon.md` for a full walkthrough of OIDC metadata probing, JS bundle API extraction, and GitHub version-based CVE discovery on Intigriti's own bug bounty program.

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
8. **Relying on summaries instead of original sources** — When referencing books, PDFs, or papers, extract the table of contents and relevant sections directly from the source file. Chapter summaries are often inaccurate or renumbered across editions. A book's table of contents in the actual PDF is authoritative — not a summary you wrote in a previous session or inferred from memory. **Example corrected this session**: Real-World Bug Hunting Ch16 is "Insecure Direct Object References" (not "Carriage Return Line Feed Injection" which is Ch6). Verify by extracting the actual PDF text and searching for the chapter heading.

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
