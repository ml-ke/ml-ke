---
name: pre-submission-verification
category: bug-bounty
description: Pre-submission verification. Run BEFORE every submission — but which gates apply depends on the VULNERABILITY CLASS, not all checks apply universally.
---

# Pre-Submission Verification

Run this BEFORE every submission. The 4 universal gates apply to ALL classes.
Then apply the class-specific gate for your finding's type.

---

## PART -1 — VERDICT ASSIGNMENT (run first, before any gate)

Every candidate gets EXACTLY ONE verdict. Record candidates in
`~/Dev/REPORTS/<Target>/findings.json` — scaffold and validate with:

```bash
python3 ~/.hermes/scripts/findings_ledger.py --new <Target>          # scaffold ledger
python3 ~/.hermes/scripts/findings_ledger.py --check  <path>         # validate (exit 1 = invalid)
python3 ~/.hermes/scripts/findings_ledger.py --summary <path>        # triage view
```

| Verdict | Requires | Severity? | Submittable? |
|---|---|---|---|
| `confirmed` | full trace + pasted request/response + `reproduced: true` + named crossed boundary | yes | yes |
| `needs_validation` | ONE exact unresolved fact + a non-destructive validation plan | **NO** | **NO** |
| `rejected` | the refuting reason (source line, existing control, no impact) | NO | no |

> ⚠️ **Gotcha — this is our #1 historical failure mode.** Fireblocks MPC 004/005 and the
> Rapyd spec-only claims were `needs_validation` records written up as if confirmed: no raw
> PoC output, theorized impact, a live blocker papered over. **Severity on an unresolved
> claim is the tell.** If you cannot set `reproduced: true` and paste the output, the
> verdict is `needs_validation`, there is nothing to submit, and the correct action is to
> say so and stop — not to write the narrative anyway.
> Pattern source: cloudflare/security-audit-skill (findings.json + report-schema.json).

### Verification scope is TIERED — never exhaustive-by-default

Do NOT run every gate on every finding. Skills that turn validation checklists into
mandatory work are the single largest measured cost regression (arxiv 2608.11888:
excessive verification = 67 of 182 efficiency regressions — the biggest subcategory). Scale depth to risk:

- **Tier 0 — always (4 gates):** `Gate A0` boundary question, `U1` demonstrated impact,
  `U4` PoC reproducible, `U9` anti-AI detection. Cheap; every rejection we have maps here.
- **Tier 1 — by class:** the ONE class gate for this finding's type (C1–C7).
- **Tier 2 — only when the finding is genuinely about to be submitted AND is P1/P2 or a
  duplicate-collision risk:** Part 3 triage (T1–T5), Part 4 victim perspective, and the
  Part 5 self-fact-check gates (S1–S8, S4–S5 severity honesty).
- **Exit early:** if `Gate A0` fails, record `rejected` and stop. Do not spend Tier 1–2
  effort polishing something that is not a vulnerability.

Record which tiers you ran in the report's own notes. Skipping Tier 2 on a low-severity
finding is correct; skipping Tier 0 is never correct.

---

## PART 0 — PRE-DIAGNOSIS: Bug, Architecture Observation, or Vulnerability?

Run this BEFORE any other gate. Most of our rejections come from skipping this step.

### Gate A0 — The Security Boundary Question

**> "Does this finding let an attacker DO something the system was designed to prevent?"**

#### Stage 1: Action Test

Write a 5-word title that ends with what an attacker CAN DO:
- ✅ "Auth bypass → read customer PII" → VULNERABILITY. Proceed.
- ✅ "Missing auth → modify invoice data" → VULNERABILITY. Proceed.
- ❌ "Auth bypass → endpoint reachable" → ARCHITECTURE. Stop.
- ❌ "Broken auth → permission error" → ARCHITECTURE. Stop.
- ❌ "API accepts any token → can't access data" → ARCHITECTURE. Stop.

If the action after the arrow is NOT a concrete, direct harm (data read, data modify, code execute, account takeover, service disrupt), the finding is probably architecture, not vulnerability.

#### Stage 2: The Two-Layer Architecture Rule

If the finding involves a service that accepts arbitrary credentials at one layer but has a SECOND independent enforcement point at the data layer, it's architecture, not vulnerability. Examples:

| Front Layer | Backend Layer | What You Found | Verdict |
|-------------|---------------|----------------|---------|
| API Gateway / MCP Server (accepts any token) | TWG / Data Service (validates token separately) | "Gateway is lenient" | 🔍 Architecture observation. The security boundary is at the data layer, not the gateway. |
| Reverse Proxy / CDN | Origin Server | "CDN forwards all requests" | 🔍 Architecture. CDNs don't authenticate. |
| Session service | Authorization service | "Session token accepted" | 🔍 Architecture if authz is independent. |

**Genuine auth bypass (single layer):** The component that accepts arbitrary tokens ALSO serves the data. Example: CVE-2024-8954 (Composio) — x-api-key header with any value → full API access. One component, one auth check.

**False auth bypass (two layer):** The component that accepts arbitrary tokens is a gateway/proxy that passes requests to a backend that enforces its own auth. Example: Atlassian MCP Server — accepts any Bearer token for init/session, but the TWG backend independently validates. The MCP `init` endpoint being lenient is a design choice for protocol accessibility, not a vulnerability.

**Test:** Can you reach data WITHOUT passing through the second auth layer? If no, it's architecture.

#### Stage 3: The Kettle Test

From Shubham Shah (Assetnote): *"If you don't have an excellent understanding of fundamental application security attacks and weaknesses before you approach bug bounties, you are wasting your time."*

Ask yourself:
- "Is this a real security boundary that I've crossed, or did I just observe how the system is designed?"
- "Could the vendor fix this and say 'this was never a vulnerability, this is how our architecture works'?"
- "What is the SPECIFIC data I accessed? Not 'could access' — what DID I access?"

#### Stage 4: Compare Against Known Accepted Findings

| Finding | What Made It Accepted | Why It's Different From Architecture |
|---------|----------------------|--------------------------------------|
| CVE-2024-8954 (Composio) | Any x-api-key → full API access | Single auth layer. Broken check = data access. |
| CVE-2025-9485 (WordPress OAuth) | Forged JWT → admin login | Direct account takeover. No second auth layer. |
| CVE-2026-29000 (pac4j JWT) | alg:none JWT → authentication | Protocol implementation bug, not architecture. |
| Spring heapdump → secrets → ATO (Shubham Shah) | Exposed endpoint → creds → account takeover | Complete chain demonstrated end-to-end. |

Our rejected findings compared:
| Finding | Why Rejected | Gate A0 Stage That Caught It |
|---------|-------------|------------------------------|
| Atlassian MCP auth bypass | Two-layer architecture: TWG independently validates | Stage 2 — Two-Layer Rule |
| AdultForce /api/site config exposure | Data is P4 business metadata, not P1-P3 data | Stage 3 — Data Access Question |
| MPC 004 oracle | Oracle theorized, key never recovered | Stage 1 — Action Test |
| MPC 005 version downgrade | Proved constants exist, not exploit | Stage 1 — Action Test |

#### Stage 5 — Decision

| Result | Action |
|--------|--------|
| **Vulnerability** — clear action, single boundary, data accessed | Proceed to class-specific gates |
| **Architecture observation** — interesting behavior, no data crossed a boundary | Do NOT submit. Save as research notes. |
| **Bug** — feature doesn't work but no security boundary crossed | Do NOT submit. |
| **Unsure** — write the 3-sentence attack story. If you can't, it's not ready. |

---

## PART 0 — PRE-WRITING COMPLIANCE CHECK

Run this BEFORE writing a single word of the report. If you skip this phase,
you will write a report that violates program rules and have to rewrite it.

### Gate P1 — Program Rules Review

Go to the program's rules page. Extract these BEFORE drafting:

- [ ] **Required headers**: Does the program require `X-Bug-Bounty`, `X-HackerOne- Researcher`, custom User-Agent? Add to EVERY curl command in the PoC.
- [ ] **Rate limits**: Max requests/sec? Ensure PoC respects them.
- [ ] **Automated tooling**: Permitted or prohibited?
- [ ] **Account requirements**: Must use `@intigriti.me` email? Premium accounts available?
- [ ] **Reporting timeframe**: Must report within 24 hours of discovery?
- [ ] **Report format**: Clear textual description required? Video-only reports refused?
- [ ] **Scoring exceptions**: Does the program list vuln types that get automatically downgraded? (e.g. XSS with CSP → P4, Open Redirect → P4, Admin→SysAdmin PrivEsc → P4) This affects your payout expectations but NOT your decision to submit.
- [ ] **No-pivoting clause**: Does the program prohibit using a vulnerability to find another? If so, submit findings as standalone reports.
- [ ] **Cross-program scope check**: Is the domain scoped under THIS program or a different one? Verify you can submit before deep-diving. Findings on sibling domains may need a separate program account.

### Gate P2 — OOS Scan Per Finding Aspect

Scan EVERY aspect of your finding against the program's explicit OOS list.

Common traps that get reports rejected:

| Finding Aspect | Common OOS Rule | Action |
|---------------|----------------|--------|
| User enumeration | "Account enumeration" | **Remove entirely** from report |
| Brute force capability | "Rate limiting or brute force issues" | Reframe as secondary consequence, not primary finding |
| Missing cookie flags | "HttpOnly, SameSite and Secure Cookie flags" | Remove — known non-issue |
| Version disclosure | "Banner identification issues" | Remove unless you have a working CVE PoC |
| Error messages | "Descriptive error messages" | Remove unless you demonstrate executable exploit |
| IDOR without impact | "IDOR with no direct security or financial impact" | Must cross boundary AND show real harm |
| Information disclosure | "Information Disclosure without significant and executable impact" | Must demonstrate cross-boundary data access |

**Process:**
1. List every claim in your summary and impact sections
2. Check each against the program's OOS list
3. If a claim matches OOS, REMOVE it. Do not reframe or hide it.
4. If the CORE finding relies entirely on an OOS class (e.g., account enumeration is your only evidence), the finding may not be viable.
**Grep the FINAL report file for every OOS term before submission.** This is a killer — see the real Nutaku case below.

**Real case — Finding 2 (Nutaku favorites, Jun 2026)**: The finding was technically CWE-287 Improper Authentication (missing auth on a GET endpoint where the POST requires auth). But the program OOS list said IDOR with no direct security or financial impact. The description contained the word IDOR even though the finding was a different class. The triager's OOS scanner likely matched the keyword and rejected automatically. If we had framed it as CWE-287 Improper Authentication — unauthenticated read of user-scoped data with zero mentions of IDOR, the outcome might have been different.

The fix: grep the FINAL report for EVERY OOS term from the program rules. If you find a match, DELETE the sentence. Do not reframe — delete. Then re-run grep until clean. Common traps: IDOR, account enumeration, rate limiting, brute force, missing security header, information disclosure.

### Gate P3 — Report Structure Planning

Before drafting, decide the deliverable structure:

- **Folder convention**: `~/Dev/REPORTS/<Target>/<number>/REPORT.md` for the report,
  `~/Dev/REPORTS/<Target>/<number>/poc/<script>` for the PoC script.
  Working sandbox stays in `~/Dev/<target>/` — never mix sandbox artifacts with final reports.
- **PoC script**: Self-contained, runnable, includes ALL required headers.
  Avoid shell variable references for secrets (masking systems may intercept them).
  Pipe responses directly or use environment variables.
- **Report structure**: Summary, Steps to Reproduce (numbered), Request/Response pairs,
  Impact, CVSS, Remediation, References.

**Report structure additions for readability and trust:**

Three patterns emerged from Nutaku submission reviews (Jun 2026) that improve acceptance odds:

1. **Plain-English opener** — After the title/header, add a "What This Means in Simple Terms" section using a concrete real-world analogy (filing cabinet, storefront, bank vault, etc.). Lead with the analogy, not the technical detail. Triagers and program owners skim — make the first paragraph understandable to a non-technical business person.

2. **"What This Is NOT" section** — Before or within the Impact section, add an explicit limitations paragraph stating what the finding does NOT expose (e.g., "This finding does NOT expose passwords, credit card numbers, or personal data. It exposes business operational data — the configuration and structure of the company's digital infrastructure."). This builds trust, reduces overclaim risk, and prevents triage from rejecting on "theoretical impact" grounds because you've already drawn the line yourself.

3. **Business analogy for impact** — When describing impact, frame at least one bullet in pure business terms without technical jargon. Example: "A retailer publishing its entire supplier list, wholesale prices, and warehouse locations" rather than "exposes internal site configuration with business unit groupings."

**Intigriti submission form fields**:
The form has 9 fields that map to a well-structured report:

| Form Field | What Goes There | Char Limit |
|-----------|----------------|------------|
| Title | Vulnerability class + component | N/A |
| Asset | From the scope list dropdown | N/A |
| Endpoint | The specific URL path | N/A |
| Type | CWE category under appropriate group (Broken Authentication, Mobile, etc.) | N/A |
| Severity | CVSS vector string (use calculator) | N/A |
| Proof of Concept / description | Summary + Steps to Reproduce + technical detail | 30,000 |
| Impact | Concrete harm, victim perspective, business risk | 15,000 |
| Recommended solution | Remediation steps (optional) | 15,000 |
| IP address | Your testing IP (optional, "Fetch my IP" button) | N/A |

**Key rule**: Separate the PoC/description from the Impact. Triage reads Impact separately to decide severity — don't bury the impact in the description. The impact section must stand alone as a clear answer to "why does this matter?"

---

## PART 1 — UNIVERSAL GATES (all classes)

### Gate U1 — Impact must be demonstrated, not theorized

Does your PoC show REAL unauthorized access or real harm?
Or does it describe what "could potentially" happen?

**Pass**: Concrete request/response showing data you should not have, or action you should not be able to perform.
**Fail**: "An attacker could enumerate X" without showing any cross-boundary data. "This could lead to Y" without showing Y.

🔴 **Rejected example**: "An attacker could enumerate vault IDs to discover hidden wallets." — No cross-boundary data shown. No actual hidden data accessed.

🟢 **Valid example**: A single curl command that returns another user's invoices, with the response included.

### Gate U2 — VRT classifies severity, doesn't validate existence

Does your finding actually fit the VRT category you chose?
Or are you using the VRT severity table as evidence the finding is valid?

**Pass**: You identified the vulnerability class first (IDOR, SSRF, crypto weakness) and then used VRT to rate severity.
**Fail**: "Sequential IDs → VRT says Iterable Object Identifiers is P1 → this is IDOR." The VRT doesn't tell you what class your finding belongs to.

### Gate U3 — Understand the system architecture

Do you know how the system's authentication, authorization, and data flow work?
Can you describe the security boundary model?

**Pass**: You can explain: "Vaults are workspace-scoped. User A in workspace B should not access vault C." And your PoC crosses that boundary.
**Fail**: You assumed a field name implies a security boundary without checking server enforcement.

### Gate U4 — PoC is reproducible

Can triage copy-paste your steps and verify the finding?

**Pass**: Concrete curl commands, exact URL paths, example request/response bodies.
**Fail**: Vague steps like "enumerate IDs" without showing which API endpoint and what response you got.

### Gate U9 — Anti-AI Detection Gate (NEW — Added Jul 2026)

**Why this exists**: Google stopped accepting AI-generated reports in March 2026. Valid submission rates dropped from ~15% to below 5% due to AI slop. Triage teams now actively flag reports that look AI-written. A real bug with an AI-sounding report gets rejected.

**Before writing ANY report text, run through these checks:**

#### Check 1 — Write PoC FIRST, then describe what it proves
- Write the curl command / script
- Run it against the live server
- Capture the ACTUAL terminal output
- Write the description based on what actually happened

**Fail**: Writing "this could extract the key" without running it.
**Pass**: `$ python3 exploit.py` followed by actual output showing data.

#### Check 2 — Strip all structure templates
Delete every section header, bold label, numbered step, and template placeholder BEFORE writing. The only structure that should remain is:
- One paragraph: what you found
- The PoC command and its output (inline code block)
- One paragraph: what this means

**Fail**: Reports with "Summary", "Vulnerability Details", "Technical Analysis", "Impact Assessment", "Remediation" sections in sequence.
**Pass**: A narrative that reads like an experienced hunter describing what they found.

#### Check 3 — Delete all explanatory text
The triager knows what XSS, SSRF, IDOR, JWT, Paillier, and OAuth are. Do NOT explain them.

**Fail**: "Paillier is a public-key encryption scheme that is additively homomorphic..."
**Pass**: "The /v1/validate endpoint returns -9 for bad ciphertext, -13 for good but used nonce."

#### Check 4 — Every claim backed by a request/response pair
Read each sentence of your impact section. If it doesn't reference a concrete HTTP response or terminal output, delete it.

**Fail**: "An attacker could access customer data" — no customer data shown.
**Pass**: "GET /api/users returns: [actual response JSON with email, name, phone]"

#### Check 5 — AI vocabulary grep
Before finalizing, grep the report for these words. Delete any sentence containing them:
`crucial, pivotal, underscores, highlights, showcases, testament, landscape, underscores the importance, it is important to note, additionally, delve, intricate, interplay, tapestry, serves as, stands as, robust, groundbreaking, seamless`

**Fail**: "This vulnerability underscores the critical importance of proper access control."
**Pass**: "The endpoint returns data without checking the auth token."

#### Check 6 — Title matches what PoC proves
Read the title. Then read the PoC output. Do they match?

**Fail**: Title says "Key Recovery" but PoC only shows an oracle exists.
**Pass**: Title says "Error Oracle: /v1/validate leaks ciphertext validity via status codes"

#### Check 7 — Apply the humanizer skill
After writing the report in narrative form, load `skill_view(name='humanizer')` and run the Bug Bounty Report Humanization pass:
1. **Pass 1 — Strip**: Remove all templated structure
2. **Pass 2 — Rewrite**: Natural narrative, curl command inline, output inline, one impact paragraph
3. **Pass 3 — Audit**: Read aloud. Would you believe this is human-written?

**Note**: The humanizer skill now has a dedicated "Bug Bounty Report Humanization" section with before/after examples and the 8 specific triggers that flag reports. Use it.

#### Check 8 — The "Call Your Own Bluff" Test
Before submitting, read the report as if you're a triager who just rejected 50 AI-generated reports today. Ask:
- "Would I believe a person wrote this?"
- "Does the report prove the impact or just describe it?"
- "Is every sentence necessary, or is there padding?"

If any answer flags concern — rewrite that section in plain English. Short sentences. Specific data. Raw output.

---

## Where the rest of this skill lives (progressive disclosure)

Parts 2-8 are **Tier 1/Tier 2** — load them only when the tiering rule in PART -1 says to:

| File | Contains | Load when |
|---|---|---|
| `references/class-specific-gates.md` | C1-C7 class gates | Tier 1: always load the ONE gate for this finding's class |
| `references/triage-and-impact.md` | T1-T5 triage, Part 4 victim perspective | Tier 2: P1/P2 or duplicate-collision risk |
| `references/report-fact-check-and-postmortem.md` | S1-S8 self-fact-check, Final Gate, bug-vs-vulnerability, rejected-findings table, Parts 7-8 post-rejection analysis | Tier 2, plus always run `S7`/`S8` when a PoC was inherited or a subagent supplied the claim |

Rationale for the split (arxiv 2608.11888): excessive verification is the largest measured
cost regression skills cause, and it is caused by checklists presented as mandatory rather
than conditional. If you find yourself loading all three references on every finding, stop —
that is the failure mode this split exists to prevent.

## Lesson Bank (MANDATORY)

After every submission outcome (accepted / rejected / duplicate / informative) or every
finding that turned out to be `rejected` or `needs_validation`, append a dated entry to
`~/Dev/ATLAS-LEARNINGS/LESSONS.md` §01 with the reason and the wording that failed. Read
that section before starting a hunt or drafting a report — it holds the rejection patterns
and target intel that this skill exists to prevent repeating.
