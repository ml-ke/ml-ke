# Hacker Schools — Full Detail

Full content formerly inline in atlas-continuous-learning SKILL.md. Loaded on demand (progressive disclosure) when a learning task needs a school's full workflow.

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

