---
name: pre-submission-verification
category: bug-bounty
description: Pre-submission verification. Run BEFORE every submission — but which gates apply depends on the VULNERABILITY CLASS, not all checks apply universally.
---

# Pre-Submission Verification

Run this BEFORE every submission. The 4 universal gates apply to ALL classes.
Then apply the class-specific gate for your finding's type.

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

---

## PART 2 — CLASS-SPECIFIC GATES

Apply the gate corresponding to your finding's vulnerability class.
Different classes have DIFFERENT rules — don't use the IDOR gate for crypto bugs.

### Gate C1 — Access Control (IDOR, BAC, PrivEsc, BOLA/BFLA)

**The boundary question**: What scopes access to this resource? (user? workspace? role? tenant?)
**The cross question**: Does your PoC access a resource owned by a DIFFERENT scope?

**Definition** (PortSwigger/Intigriti/OWASP): IDOR occurs when an application takes user-supplied input and uses it to retrieve an object **without performing sufficient authorization checks**. The key phrase is "belongs to a different entity."

**Testing methodology** (from 4 public IDOR writeups):
1. **Create two accounts** in different scopes (User A in Org/Workspace A, User B in Org/Workspace B)
2. Log in as User A, capture all requests with Burp
3. Log in as User B, capture all requests
4. Try User A's identifiers (IDs, UUIDs, emails) while authenticated as User B
5. If User B can access User A's resources → real IDOR
6. If you only tested within one scope → **not ready for submission**

**Rules**:
- Must CROSS a resource ownership boundary. Same-workspace enumeration is NOT IDOR regardless of ID format.
- Client-side UI hiding, field name conventions ("hidden", "private"), obfuscation are NOT access controls. Only server-side enforcement counts. **The server must first RESTRICT access before you can BYPASS it.**
- Sequential/incrementing IDs enable exploitation but don't create the vulnerability. The missing auth check creates it. UUIDs with missing auth would be equally broken.
- **"Always ask yourself: is this really an issue or is it intended behaviour?"** — Intigriti Hackademy. If the answer could be "the app was designed this way," it's likely a false positive.
- If your PoC requires "you already have access" to work, it's not an access control finding.

**Real rejected examples** (HackerOne): Reports #166849, #49499, #361133 — researchers misunderstood the application behavior.

**Test**: Create two accounts in different scopes. Access scope A's resources from scope B. If it works → real finding.

**Resources**:
- PortSwigger: https://portswigger.net/web-security/access-control/idor
- Intigriti: https://www.intigriti.com/researchers/hackademy/idor
- OWASP: https://owasp.org/www-community/attacks/Insecure_Direct_Object_References

### Gate C2 — Cryptographic Weaknesses

**Rules**:
- Theoretical weakness ≠ practical exploit. Academic paper ≠ bounty submission.
- RNG attacks need viable failure scenario (VM clone, fork, entropy exhaustion).
- Demonstrate the actual attack: key recovery, sig forgery, plaintext recovery.
- Implementation bugs (wrong data hashed) are easier to demonstrate than protocol design flaws.

**Test**: Can you run ONE script that proves the attack end-to-end?

### Gate C3 — SSRF

**Rules**:
- DNS resolution without HTTP response data is blind SSRF.
- Reaching a metadata endpoint and extracting creds is impact. Connection refused is not.
- AI agent SSRF: Check program stance — many consider UI:R + PR:L as mitigation.
- Novel bypass techniques > standard SSRF for acceptance.

**Test**: Capture the internal service's response. Verify via OOB callback if blind.

### Gate C4 — Credential / API Key Exposure

**Rules**:
- Verify key against LIVE API: must return 200 with usable data.
- Defunct keys, incomplete pairs, origin-restricted keys are NOT findings.
- Client-side-only keys are not findings unless cross-origin abuse is demonstrated.

**Test**: `curl -sL "https://api.target.com/endpoint" -H "Authorization: Bearer <key>"` — must return 200 with data.

### Gate C5 — Memory Safety

**Rules**:
- Crash alone = P4. Must show controlled exec or data corruption.
- Modern mitigations (CFG, CET, ASLR, stack cookies) affect exploitability.

**Test**: Show more than a crash — register control, memory corruption, or security property violation.

### Gate C6 — Business Logic

**Rules**:
- "An attacker could spam" needs rate limit, CAPTCHA, and cost analysis.
- Frame in business terms: revenue loss, compliance violation, user trust damage.

**Test**: Calculate actual dollar value or operational cost. If negligible → not a finding.

---

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

---

## PART 5 — REPORT SELF-FACT-CHECK (from bodyHash bug rewrite)

Before writing the final report, run these verifications against every claim.

### Gate S1 — Verify EVERY impact claim against the LIVE server

Do not write an impact claim based on reasoning alone. Every claim must be backed by a real HTTP response.

**Process:**
1. Write a test script exercising the exact scenario
2. Run it against the live server
3. Capture the actual response
4. Only then include the claim

🔴 **bodyHash report**: Claimed "JWT can be replayed with any arbitrary body." — Never tested. Actual: `401 "nonce already used"`. Claim was **false**.

🟢 **Correct**: Run the test first. Replay fails → delete the claim.

**Rule:** Every impact bullet must reference a concrete request/response pair.

### Gate S2 — Verify the ACTUAL scope of the bug

**Process:**
1. Check ALL versions, not just the one you first found
2. Check the published package, not just source
3. Check both source and compiled output
4. For npm packages: `npm view @scope/name versions --json` then `npm pack` each version and inspect

🔴 **bodyHash report**: Claimed "v19.1.0" — ALL 64 versions from v8.0.1 to v20.0.0 have the same bug.

🟢 **Correct**: `npm pack @fireblocks/ts-sdk@{v1,v2,...}` each version. ALL broken → "ALL published versions."

**Rule:** Check at least 3 versions spanning the package's lifetime. For npm packages, check the oldest, middle, and newest.

### Gate S3 — Check "escalation" paths end-to-end

**Process:**
1. Write the COMPLETE exploit chain as a single script
2. Run it start to finish
3. If it fails at any step → remove that escalation

🔴 **bodyHash**: "bodyHash bug → JWT replay → modify transaction." Step 2 failed (nonce tracking).

🟢 **Correct**: Step 1 ✓, Step 2 ✗ → No escalation. Report the bug as-is.

**Rule:** No escalation path until the full chain is demonstrated.

### Gate S4 — Severity must match VRT baseline honestly

**Process:**
1. Look up the VRT baseline for your category
2. If claiming higher, document the specific evidence
3. If evidence was unverified, severity claim is wrong

🔴 **bodyHash**: VRT baseline = P3. Claimed P2 based on unverified replay.

🟢 **Correct**: Classify → P3. Test escalation. If it works, include. If not, accept P3.

### Gate S5 — Write PoC FIRST, then describe what it proves

**Process:**
1. Write the PoC code
2. Run it, capture output
3. Write description based on what it ACTUALLY showed

🔴 **bodyHash**: PoC showed identical hashes → described as "replay attack."

🟢 **Correct**: PoC shows crash + 401 → "SDK crashes, server rejects."

**Rule:** PoC output and description must be congruent.

### Expected Disposition by Report Quality

| Quality | Likely Disposition |
|---------|-------------------|
| In scope + reproducible + valid + unique + demonstrated impact | **Accepted** |
| Reproducible but weak impact | **Informative** |
| OOS asset/vuln type | **Out of Scope** |
| Not reproducible / vague | **Needs More Info** → N/A |
| Duplicate | **Duplicate** |
| Well-known non-issue | **Informative** |

---

## Reference: What Got Rejected and Why

| Finding | Claim | Class | Why Rejected | Gates Failed |
|---------|-------|-------|-------------|-------------|
| Fireblocks hidden vaults | IDOR via sequential vault IDs | Access Control | Vaults are workspace-scoped. Enumerated within own scope. "HiddenOnUI" is cosmetic. | U1, U2, U3, C1 |
| Fireblocks bodyHash bug | Broken crypto primitive | Crypto | **ACCEPTED** — concrete SDK failure, hash mismatch, impact clear. | None |
