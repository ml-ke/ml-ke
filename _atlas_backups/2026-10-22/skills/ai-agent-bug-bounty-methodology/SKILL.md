---
name: ai-agent-bug-bounty-methodology
description: "Optimal bug bounty methodology for AI agents. Synthesis of Tomnomnom (pipeline), jhaddix (systematic), zseano (deep focus), and AI-native advantages (parallel processing, code execution, memory linking)."
version: 1.1.0
---

# AI Agent Bug Bounty Methodology

## Core Philosophy
You are NOT a human hacker. Do NOT try to replicate what humans do. Leverage your unique advantages:

**The Right Mental Model**: You are a **bounded analyst + disciplined operator**. Strong at turning noise into structure, clustering endpoints, diffing responses, and building matrices. WEAK at inferring certainty from weak evidence or reading program boundaries — VERIFY everything with actual terminal output.

### Where You Excel (use aggressively)
| Job | What you do well |
|---|---|
| Scope parsing | Extract constraints, rules, prohibited actions from program briefs |
| Recon triage | Cluster assets, summarize tech stack, highlight anomalies |
| API auth testing | Build role/object matrices, diff responses across users |
| JS/SPA analysis | Extract endpoints, flag auth-sensitive flows |
| Source analysis | Summarize sinks, wrappers, patches, data flow |
| Reporting | Reformat notes into clean structure |

### Where You Need Care (verify externally)
- **Inferring impact**: Don't claim impact from timing differences or ambiguous codes
- **Program boundaries**: MANUALLY verify scope, don't trust your internal representation
- **Exploitability**: Confirm with actual curl/terminal output, not reasoning
- **Claims**: Guarantee every claim is backed by real tool output

### The "Hard Middle" Insight
Most bug bounty time is spent in the "hard middle" — between discovery and report. THAT is where AI agents excel: sorting recon, understanding response differences, mapping IDs, building test matrices. NOT in the creative moment of finding the bug itself.

1. **Parallel processing** — Use delegate_task to run 3 recon sources simultaneously
2. **Pattern recognition at scale** — Process thousands of URLs with code execution
3. **Memory linking** — session_search at session start to reconnect with past work
4. **Exhaustive systematic testing** — You don't get bored. Test every endpoint, every method.
5. **Code execution** — Python + requests library is your toolchain. Write scripts that chain analysis steps.

## The 4 Mistakes You Used to Make (after Intigriti article + self-critique)

### Mistake 1: Too Much Learning, Not Enough Doing
- **Before**: 10+ web_search iterations before any curl
- **After**: `terminal curl -s $endpoint | jq .` FIRST, then research what you found

### Mistake 2: No Pipeline
- **Before**: Jumping between techniques without a repeatable sequence
- **After**: Follow the pipeline: Scope → Subdomains → Live hosts → URLs → Filter → Test → Chain. Record each output.

### Mistake 3: No Business Impact
- **Before**: Reporting technical findings without business context
- **After**: Always ask "What does the business lose?" before submitting

### Mistake 4: Not Understanding the Target
- **Before**: Probing endpoints without knowing the application
- **After**: Read the program page, understand features, learn the tech stack first

## Optimal Workflow Pipeline

### Phase 1: Scope Intake (5 mins)
- Read program page carefully
- Store in memory: scope domains, in-scope assets, out-of-scope rules, rate limits
- session_search for past work on this target

### Phase 2: Passive Recon (parallel via delegate_task)
```python
# Run 3 tasks simultaneously:
delegate_task(goal="crt.sh subdomain enumeration for target.com")
delegate_task(goal="Wayback Machine URL collection for target.com")
delegate_task(goal="Technology fingerprinting of target.com")
```

### Phase 3: Live Host Probing
```bash
cat subdomains.txt | while read h; do
  code=$(curl -sk -o /dev/null -w "%{http_code}" "https://$h" --connect-timeout 5)
  echo "$code $h"
done | grep -E "^[23]" > live_hosts.txt
```

### Phase 4: Endpoint & Parameter Filtering
```bash
# API endpoints
cat wayback_urls.txt | grep -iE "/api/|/v[0-9]/|graphql|rest|swagger" > api_endpoints.txt
# Parameters
cat wayback_urls.txt | grep "=" | sed 's/.*?//' | tr '&' '\n' | cut -d'=' -f1 | sort -u > params.txt
```

### Phase 5: Systematic Vulnerability Testing
For each vulnerability class, load the relevant skill:
- **Identity platforms** (Auth0, Okta, Azure B2C) → load `oauth-oidc-attacks` — then get the Management API token FIRST
- IDOR → load `idor-testing-methodology`
- Mass assignment → load `mass-assignment-method-tampering`
- OAuth → load `oauth-oidc-attacks`
- SAML → load `saml-attacks` or `saml-attack-techniques`
- Business logic → load `business-logic-flaws`
- JWT → load `jwt-attacks`
- SSRF → load `ssrf-testing`

### Phase 5B: Credential-First Workflow (Identity Platforms)
When you get Bugcrowd/Intigriti credentials for an identity platform:
1. **Get the API token FIRST** — before using the dashboard or testing manually
2. The Management API gives full access: users, templates, actions, connections
3. Use Python + requests (or execute_code) for API calls — avoids curl escaping issues
4. Test systematically: email templates → actions → connections → cross-tenant
5. For Auth0 specifically: the Management API token format is:
   ```python
   POST {tenant}/oauth/token
   {"client_id": "...", "client_secret": "...",
    "audience": "https://{tenant}/api/v2/", "grant_type": "client_credentials"}
   ```
6. Save the token to a file for the session — it lasts 24 hours

### Phase 6: Chaining
After finding ANY vulnerability, ask:
- "What does this unlock?"
- "Can I chain this with another finding?"
- "What's the access level after exploiting this?"

### Phase 7: Report
- Impact-first: "This allows X to happen to the business"
- PoC with actual terminal output, not description
- Chain if multiple issues found

## Key Commands (Tomnomnom-style pipes)

```bash
# Subdomain → live → URLs → params pipeline
echo "target.com" | \
  (curl -s "https://crt.sh/?q=%.target.com&output=json" | jq -r '.[].name_value') | \
  sort -u | \
  xargs -I@ sh -c 'curl -sk -o /dev/null -w "%{http_code} @\n" "https://@"' | \
  grep -E "^[23]" | \
  awk '{print $2}' | \
  xargs -I@ sh -c 'curl -s "http://web.archive.org/cdx/search/cdx?url=@/*&output=text&collapse=urlkey&fl=original"' | \
  sort -u > all_urls.txt

# Extract API endpoints from JS
curl -s $TARGET | grep -oP 'src="[^"]*\.js[^"]*"' | cut -d'"' -f2 | \
  xargs -I@ sh -c 'curl -s "https://$TARGET/@"' | \
  grep -oP '["'\'']/[a-zA-Z0-9_/]+api[a-zA-Z0-9_/]*["'\'']' | \
  sort -u
```

## The 6 Self-Identified Gaps (Jun 2026)

These gaps were identified during a 40-iteration methodology upgrade cycle. Check them before every session.

### Gap 1: Tools exist but aren't used
- Created scripts, pipeline tools — then never ran them on a real target
- **Fix**: Run the script BEFORE doing more research. Load and execute immediately.

### Gap 2: No target application
- Methodology upgrades without target application = zero findings
- **Fix**: Pick ONE target per work session. Run the full pipeline on it. Don't stop at methodology.

### Gap 3: No decision tree
- 10+ methodologies but no way to choose which applies
- **Fix**: Target tech stack → load relevant skills → test in priority order (OAuth first if Azure B2C, IDOR first if sequential IDs visible, etc.)

### Gap 4: Session search unused
- Memory linking is a unique AI agent advantage, but rarely used
- **Fix**: `session_search()` at the START of every session. Reconnect before acting.

### Gap 5: No feedback loop
- Learned from articles, not from actual results (submissions, rejections, acceptances)
- **Fix**: Submit a finding. Get rejected/accepted. Learn from the outcome. Update this skill.

### Gap 6: Meta > action
- More time spent building methodology than finding bugs
- **Fix**: The goal is REAL findings, not beautiful methodology. When in doubt, curl.

## Methodology Upgrade Cycle Pattern

When doing a methodology upgrade (40+ iteration cycles):

1. **Every 5 iterations: SWITCH** between deep dive and critique
   - Deep dive (5 iter): Research external sources — PortSwigger, H1 Hacktivity, top hacker talks
   - Critique (5 iter): Examine YOUR methodology against what you learned. Identify gaps. Save to skill.

2. **Save as you go** — don't batch all learnings at the end
   - After each deep dive block: create/patch skills
   - After each critique block: update memory with gap findings

3. **Apply to a real target after 20 iterations** — don't do 40 iterations of pure research
   - 20 iterations of learning → target application → 20 more iterations of refinement

## Discernment Framework: Good Knowledge vs Bad Knowledge

### Good Knowledge (trust these sources)
- **PortSwigger Web Security Academy** — Authoritative, practical, verified
- **OWASP Testing Guide (WSTG)** — Industry standard methodology
- **HackerOne Hacktivity** — Real disclosed reports with actual impact
- **Bugcrowd CrowdStream** — Validated findings from known researchers
- **Top hackers' own content** — Tomnomnom, jhaddix, STOK, NahamSec, InsiderPhD talks/videos
- **CVEs from reputable researchers** — Nicholas Carlini, 0xacb, etc.
- **Recent articles with named authors and publication dates** — Recent, verifiable
- **Peer-reviewed security papers** — Usenix WOOT, IEEE S&P, CCS

### Bad Knowledge (filter these out)
- **"I hacked Facebook in 5 minutes" clickbait** — Usually fake or extremely lucky
- **Republished Medium articles with no new content** — Same techniques rephrased
- **Automated scanner output claiming critical findings** — False positive central
- **Outdated payloads/techniques** — Anything 3+ years old unless classics
- **"Using AI to find 100 bugs a day" claims** — Usually auto-submission noise
- **Content with $ amounts but no PoC** — If they can't show the bug, it didn't happen

### Discernment Test: Before saving a technique, ask:
1. "Did I verify this with an actual curl/terminal call?" (NO = don't embed yet)
2. "Is this from a known/published source or random Medium blog?"
3. "Would this work against the specific target I'm testing NOW?"
4. "Is this using tools I can actually run?" (Some need Burp Pro, specific Go binaries, etc.)

### Penligent Framework: AI's Role in Bug Bounty (Apr 2026)
From the definitive guide on AI in bug bounty:

**AI is strong at:**
- Turning noise into structure (ingesting HTTP transcripts, JS bundles, CLI output, GraphQL introspection)
- Clustering similar endpoints, grouping parameters by semantics
- Extracting candidate object identifiers, comparing role-based responses
- Summarizing code you don't want to read line by line
- Turning rough notes into a report skeleton

**AI is weak at:**
- Inferring certainty from weak evidence (timing differences, ambiguous HTTP codes, frontend-only state changes)
- Reading program boundaries — the model speaks confidently about scope even with no durable internal representation
- Generating test plans that respect program rules — will violate them if prompt is careless
- **Fix**: READ THE BRIEF YOURSELF. Mark boundaries yourself. Then use AI to compress and organize.

**The "keep the tester in control" design principle:**
"If the model cannot see the exact request you are mutating, the exact response you got back, the exact role or session state attached to that response, and the exact boundaries of the program, then it is mostly guessing. Fluent guessing is the raw material of low-quality submissions."

## The Layered Skill Architecture

The skills are organized so the right one loads at the right time:

1. **Top-level** (always loaded): `ai-agent-bug-bounty-methodology` — the workflow itself
2. **Phase 4**: `recon-to-exploitation` — recon scripts and approach
3. **Phase 5**: Attack-type skills — `idor-testing`, `mass-assignment`, `oauth-oidc-attacks`, `saml-attacks`, `saml-attack-techniques`, `business-logic-flaws`, `jwt-attacks`, `ssrf-testing`, `chaining-methodology`
4. **Phase 5B (identity platforms)**: `oauth-oidc-attacks` references/auth0-okta-attack-surface.md — Auth0-specific Management API, Actions, email templates, Liquid SSTI
5. **Phase 7**: `pre-submission-verification`, `hackerone-submission-template`, `bugcrowd-submission-template`

### Skill Loading Decision Tree
```
Target has OAuth/OIDC? → load oauth-oidc-attacks
  Is it Auth0/Okta? → also load references/auth0-okta-attack-surface.md
  Is it Azure B2C? → also load references/azure-b2c-oauth-testing.md
Target has SAML? → load saml-attacks, saml-attack-techniques
Target is web app with user IDs visible? → load idor-testing
Target is e-commerce/fintech? → load business-logic-flaws
Target has JWT tokens? → load jwt-attacks
Target has SSRF-prone features? → load ssrf-testing
```
