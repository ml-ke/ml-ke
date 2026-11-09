# Triage Evaluation & Victim Impact (T1-T5, Part 4)

Provenance: split out of `SKILL.md` 2026-09-15. This is **Tier 2** work: load it only when
the finding is genuinely about to be submitted AND is P1/P2 or a duplicate-collision risk.

## PART 3 — TRIAGE EVALUATION FRAMEWORK

### The Triage Decision Tree

Every triager runs these checks in order:

```
Gate 1: IS IT IN SCOPE? → No? ❌ Out of Scope
Gate 2: IS IT REPRODUCIBLE? → No? ❌ Needs More Info
Gate 3: IS IT A VALID VULNERABILITY? → No? ❌ Informative
Gate 4: IS IT UNIQUE? → No? ❌ Duplicate
Gate 5: WHAT IS THE IMPACT? → Sets severity + payout
```

### Gate T1 — Scope Check

**What triage checks:**
- Is the asset explicitly listed or covered by wildcard?
- Is the vulnerability type explicitly excluded?
- Does the program have special rules?

**Pass**: Both asset AND vuln type are explicitly in-scope.
**Fail**: Either is OOS = immediate rejection.
**Case-by-case**: Some programs include a provision like "Rewards for high impact Vulnerabilities outside of the Scope of this Program might be considered on a case-by-case basis." If you have a strong finding on an OOS asset — one with clear, executable impact and data that directly relates to the program's in-scope properties — you can submit with an upfront **Disclosure section** that:
- Explicitly states the asset is technically outside the strict scope
- Cites the program's case-by-case provision
- Explains why the impact justifies consideration (data includes the program's own infrastructure, shared codebase with in-scope assets, etc.)
- Does NOT try to hide or reframe the scope issue

This is a long shot. Only attempt it when the impact is genuinely significant and the data directly relates to the program's own properties. The case-by-case provision exists for exceptional findings — don't use it as a loophole for weak findings on out-of-scope assets.

**Program compliance — 3 critical checks before submission:**

1. **Custom header requirements**: Some programs require `X-Bug-Bounty: <username>` or similar on ALL requests. Check the Rules of Engagement table. Add this header to every curl/PoC command in your report.

2. **OOS overlap check**: Your finding's FRAMING must not overlap with OOS vulnerability classes even if the root cause is different. For example:
   - Finding is CWE-798 hardcoded credentials (in scope) but you frame impact as "user enumeration" (OOS) → expect rejection. Frame as "auth bypass" instead.
   - Finding is missing auth on an endpoint (in scope) but you frame impact as "rate limiting" (OOS) → expect dismissal. Frame as "unauthorized data access" instead.
   - Check the FULL OOS list, not just the vuln class. Search for terms that overlap with how you describe your finding.

3. **Own-account-only rule**: If the program requires testing only against your own accounts, ensure the PoC only accesses data belonging to the researcher's account. Do not demonstrate cross-user access even if technically possible — submit the auth bypass and let triage ask for escalation.

4. **No-pivoting/post-exploitation rule**: If the program explicitly prohibits "using a vulnerability to find another," you must:
   - Submit each finding as a standalone report
   - NOT reference cross-finding chaining or escalation in your report
   - NOT submit findings discovered by exploiting another vulnerability (even if independently verified)
   - If multiple findings exist, submit the auth bypass first, and for subsequent findings disclose in the report text that they were discovered through independent testing, not by pivoting from the first finding.

#### T1a — Required Headers and Program-Specific Rules

Some programs require specific headers on ALL requests:
- Intigriti Nutaku: `X-Bug-Bounty: <username>`
- Other programs may require `X-HackerOne- Researcher`, custom rate limits, specific User-Agent

**Check before writing the report:**
1. Read the program's Rules of Engagement section for required headers
2. Add the header to EVERY curl command in the PoC — not just the first one
3. Check for rate limits (max req/sec) and ensure your PoC respects them
4. Check if automated tooling is permitted, and any User-Agent requirements

#### T1b — OOS List Scanning Per Finding Aspect

BEFORE writing the report, scan EVERY aspect of your finding against the program's OOS list. Common pitfalls:

| Finding Aspect | Common OOS Rule | Risk |
|---------------|----------------|------|
| User enumeration | "Account enumeration" | Report rejected if mentioned |
| Brute force capability | "Rate limiting or brute force issues" | May be rejected case-by-case |
| Missing cookie flags | "HttpOnly, SameSite and Secure Cookie flags" | Instant reject |
| Version disclosure | "Banner identification issues" | Instant reject |
| Error messages | "Descriptive error messages" | Instant reject unless proven exploitable |

**Process:**
1. List every claim in your impact section
2. Check each claim against the program's OOS list
3. If a claim matches an OOS item, REMOVE it from the report — do not reframe it
4. If the CORE finding relies entirely on an OOS class (e.g., account enumeration), the finding itself may need re-evaluation

### Gate T2 — Reproducibility Check

**Checklist:**
- [ ] Prerequisites listed FIRST
- [ ] Step-by-step numbered instructions
- [ ] Actual HTTP request/response pairs
- [ ] Screenshot/video for complex UI workflows
- [ ] Works on production, not local env

### Gate T3 — Vulnerability Validity Check

**Common invalid patterns:**
1. **Self-XSS** — Requires victim to paste JS into console.
2. **Missing security header** — Not a vuln without demonstrable impact.
3. **Version disclosure** — Not a vuln without unpatched CVE you can demonstrate.
4. **Rate limiting absence** — Not a vuln without demonstrated harm.
5. **Missing SPF/DMARC** — Not a vuln without demonstrated email spoofing.
6. **CSP misconfiguration** — Contributing factor, not standalone finding.
7. **Verbose error messages** — Not a vuln unless leaked info enables further exploitation.

### Gate T4 — Uniqueness Check

Search program's public disclosures, CVE/NVD, HackerOne Hacktivity. Accept duplicates happen — speed + depth is the differentiator.

### Gate T5 — Impact Assessment

| Factor | Informative | Triaged |
|--------|------------|---------|
| Impact shown? | Theorized | Demonstrated |
| POC exists? | Conceptual | Working |
| Exploit chain? | Partial | Complete |
| Data accessed? | Your own | Cross-boundary |
| Security controls? | Ignores them | Addresses/bypasses them |

---

## PART 4 — VICTIM PERSPECTIVE FRAMEWORK

Before submitting, imagine you're the **victim**. This flips the question from "can I exploit this?" to "could someone be harmed?"

### The Victim Questions

**V1 — Who is the victim?**
- **Fail**: "The server is the victim because it leaks info." — Servers aren't victims.
- **Pass**: "Any user with a valid session."

**V2 — What must the victim DO for the attack to work?**
- More actions = harder exploit.
- **Fail**: "Victim must paste JS into console." — Self-XSS.
- **Pass**: "Victim only needs to visit attacker's site while logged in."

**V3 — What prerequisites must be true?**
- List every prerequisite explicitly. 3+ unlikely prerequisites = unrealistic.
- **Fail**: "Victim must be authenticated admin + visiting attacker site + have disabled CSRF."
- **Pass**: "Victim just needs to be authenticated."

**V4 — Can the attacker FORCE the prerequisites?**
- If not forceable, it's phishing/social engineering, not a technical vulnerability.
- Programs typically exclude social engineering.

**V5 — What is the ACTUAL harm?**
- Not "information disclosure" but "leaks the victim's full name, email, phone, billing address."

**V6 — Is there a realistic attack scenario?**
- Write a 3-sentence story. If you can't, the finding isn't ready.

### Business Impact Framing

After running the Victim Perspective Framework, frame the impact for a **non-technical business reader**. Program owners and product managers review reports too.

**The analogy-first approach**: Lead with a real-world business analogy before explaining the technical mechanism. This ensures the business reader understands WHY it matters before they hit the technical detail.

**Pattern:**
1. **One-sentence plain-English description**: "An API endpoint that lists every brand's billing account ID and cloud storage location can be read by anyone with an internet connection."
2. **Business analogy**: "This is the equivalent of a retailer publishing its entire supplier list, wholesale prices, and warehouse locations."
3. **What this means for the business**: List concrete business risks in plain language (competitor intelligence, infrastructure targeting, billing system analysis).
4. **What this is NOT**: Explicitly state what the finding does NOT expose (no user passwords, no credit cards, no PII). This builds trust and prevents triage from dismissing on overclaim grounds.

**Avoid**: Pure technical impact descriptions without business translation. "Exposes ProBiller IDs" means nothing to a business reader. "Reveals which billing account each brand uses for payment processing" means something.

**Examples of business analogies that work:**
| Technical finding | Business analogy |
|-----------------|------------------|
| Unauthenticated API listing all sites with billing IDs | Filing cabinet with locks on the drawers but not the front door |
| IDOR on invoice endpoint | Customer A reading Customer B's receipts |
| SSRF to cloud metadata endpoint | Using the mailroom to read the CEO's mail |
| Missing auth on admin panel | Bank vault with a spinning lock but no guard at the door |

---
