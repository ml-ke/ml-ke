# Report Fact-Check, Bug-vs-Vulnerability, and Post-Rejection Analysis

Provenance: split out of `SKILL.md` 2026-09-15. Contains Part 5 self-fact-check (S1-S8),
the Final Gate, Part 6 bug-vs-vulnerability, the rejection reference table, and Parts 7-8
post-rejection analysis. **Tier 2**, except `S7`/`S8` which are Tier 0-adjacent: run them
whenever a PoC was inherited or a subagent supplied the claim.

## PART 5 — REPORT SELF-FACT-CHECK

Before writing the final report, run these verifications against every claim.

### Gate S1 — Verify EVERY impact claim against the LIVE server

Do not write an impact claim based on reasoning alone. Every claim must be backed by a real HTTP response.

**CVE precondition ≠ exploitation**: A confirmed CVE version or signature pattern does NOT mean the exploit works. Example — CVE-2017-9822 (DNN deserialization RCE): Nuclei detected preconditions (DNNPersonalization cookie accepted, 404 handler active) but the actual gadget chain silently fails because the WPF assembly required for exploitation isn't loaded in the ASP.NET worker process. Always test the FULL exploit chain, not just the preconditions.

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

### Gate S5 — Self-review as the triage reviewer BEFORE submitting

Before writing the final report, explicitly adopt the reviewer's perspective. Ask:

**"How would I reject this finding?"**

Then systematically answer. This catches the most common rejection reasons before triage does.

**Process:**
1. Read the program's OOS list. For each OOS item, ask: "Could a reviewer argue this finding falls under this category?"
2. Read your own impact claims. For each, ask: "Is the impact demonstrated (request/response pair) or theorized ('could lead to...')?"
3. Read your own data sensitivity claims. For each, ask: "Would a reasonable person agree this data causes harm? Or is it business-operational data that doesn't cross a meaningful boundary?"
4. Ask: "What is the weakest part of this submission?" Fix it or remove it.
5. Ask: "If the developer comes back and says 'this is intentional design,' what evidence do I have that it's not?"
6. Write a **"Strengths and Limitations"** section in the report that explicitly addresses the honest weaknesses. This builds trust with the reviewer and prevents them from rejecting on a point you already acknowledged.

**Examples from AdultForce submission (Jun 2026):**
- Weakest point: ProBiller IDs are identifiers, not credentials. An attacker can't charge a card with them.
- Honest limitation acknowledged: "This finding does NOT expose user passwords, credit card numbers, or PII. It exposes business operational data."
- Evidence against "intentional design": The same auth gap pattern exists on a second endpoint (`/api/config`), making it a systemic issue rather than a deliberate choice.

**Rule:** If you cannot articulate a credible counter-argument and explain why it's wrong, the finding is not ready to submit.

### Gate S6 — Write PoC FIRST, then describe what it proves

**Process:**
1. Write the PoC code
2. Run it, capture output
3. Write description based on what it ACTUALLY showed

🔴 **bodyHash**: PoC showed identical hashes → described as "replay attack."

🟢 **Correct**: PoC shows crash + 401 → "SDK crashes, server rejects."

**Rule:** PoC output and description must be congruent.

### Gate S7 — Re-Run INHERITED PoCs Before Any Resubmission (NEW Aug 2026)

**Any PoC you did not write yourself this session MUST be re-run from scratch.** Do not trust prior reports' claims, prior build artifacts, or "it worked in my environment."

**The Fireblocks case (Aug 2026)**: Reports 004/005 claimed a "BAM Paillier oracle" that recovers keys in ~300 queries. Re-running the inherited PoC (`bam_attack_poc`) on the same repo HEAD it crashed at `generate_signature_proof` → `zero_knowledge_proof_status:Internal error`. The oracle NEVER worked. The companion `bam_crt_extraction` PoC passed only `REQUIRE(300 < 1000)` — hardcoded numbers, no actual λ extraction. Both were theater that got submitted as real findings (and rejected as AI-generated partly because the "proof" was invented).

**Process:**
1. `git status` — confirm you're on the SAME commit the report claims
2. Rebuild from clean (delete build dir, `cmake -B build && cmake --build build`)
3. Run the EXACT command the report says produced output
4. Compare: does stdout match the report's quoted output?
5. If it crashes/asserts/exits non-zero → the PoC does not prove the claim → **withdraw the finding, do NOT resubmit**

**Also verify PoC claims against SOURCE, not just runtime:**
- A test that calls a LOCAL buggy replica (`compute_buggy_seed`) proves nothing about the library — check the test actually invokes the library function
- A test asserting hardcoded numbers (`REQUIRE(300 < 1000)`) proves nothing — read what it asserts
- Grep the call sites: a PoC may pass while the real code path is never exercised

### Gate S8 — Verify SUBAGENT Claims Before Trusting Them (NEW Aug 2026)

When findings come from delegated/subagent work, the subagent's summary is a SELF-REPORT — verify the key claims yourself before submitting:

1. **Re-run the money commands**: the 2-3 curl commands behind the top finding, yourself, against the live target. Check status codes match.
2. **Spot-check the raw proof files exist** and contain real output (not empty files).
3. **Check for AI-vocab** in any report a subagent wrote (they inherit the same slop tendencies).
4. **Check the report wasn't padded** with claims the raw output doesn't support (subagents overclaim impact).

Verified examples (Aug 2026 multi-agent hunt): Etsy XML-RPC `system.listMethods` claim → re-ran curl myself, got the raw method list. Skoda swagger→Azure SSO claim → re-ran, got the 302 to login.microsoftonline.com with tenant ID. Both real. The Skoda user-enum claim → my first reproduction attempt got 405 (wrong session setup) — the agent's exact flow was needed; verify the FULL flow, not a shorthand variant.

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

## FINAL GATE — Two Mandatory Questions Before "Ready to Submit"

Before telling the user the report is ready, answer these two questions explicitly. The user will ask both — pre-empt them.

### Q1 — Does this conform to what the program requires?

**Run through every row:**

| Requirement | Source | Checked? |
|-------------|--------|----------|
| Asset in scope | Program's assets list | |
| Vuln type not OOS | Program's OOS list | |
| Required headers on ALL requests | Rules of Engagement | |
| Rate limit respected | Automated tooling rules | |
| @intigriti.me email if required | Account requirements | |
| Reported within 24hr | Reporting timeframe | |
| Clear textual description | Reporting requirements | |
| Own accounts only | Testing requirements | |

**Then grep the final REPORT.md for every OOS term** — OOS keywords in descriptions trigger automated rejection even if the finding is a different class. Common traps: "IDOR", "account enumeration", "rate limiting", "brute force", "missing security header". Remove every match, re-run grep until clean.

Fix: The `grep` command runs against the final REPORT.md. Delete matches — do not reframe or hide them. Re-run grep until clean.

Write a 2-3 sentence plain-English answer that:
- Identifies the security boundary the server SHOULD enforce
- Explains how the server FAILS to enforce it
- Describes what an attacker can DO because of that failure

**Example (favorite-games finding)**:
> "The POST endpoint requires authentication (401 without token). The GET endpoint does not — it returns any user's favorites to anyone. This means the server was designed to protect this data but forgot to check on the read path."

---

## PART 6 — BUG VS VULNERABILITY DECISION

Not every real, verifiable bug is a submittable security vulnerability. The distinction:

### The Core Question

> "Does this finding let an attacker do something the system was designed to prevent?"

**No → It's a Bug.** Don't submit. Save the analysis in your research notes.

**Yes → It's a Vulnerability.** Submit it (after passing all other gates).

### Decision Table

| It's a Bug (don't submit) | It's a Vulnerability (submit) |
|---------------------------|-------------------------------|
| SDK crashes on POST | Crash lets attacker bypass authentication |
| Hash computed incorrectly | Wrong hash lets attacker forge requests |
| Feature doesn't work | Broken behavior crosses a security boundary |
| Server returns an error | Error leaks exploitable information |
| Client-side only issue | Server-enforced restriction bypassed |
| Auto-generated code quality bug | Cryptographic protocol implementation flaw |

### The Server-Side Integrity Test

If the server-side security is intact — the server correctly validates, rejects, and logs bad input — the bug exists only in the client-side component. It's a quality issue, not a vulnerability.

**Real example — Fireblocks TS SDK bodyHash:**
- `crypto.update(bodyJson)` receives a raw JS Object instead of a string
- Every POST crashes (Node 22) or sends an invalid hash (Node 18)
- **But**: The server correctly rejects bad bodyHash with code -9
- **And**: The server correctly tracks nonces with code -13
- The server-side crypto is sound. The TS SDK just can't participate.
- → **Bug, not vulnerability. Not worth submitting.**

### When Framing Alone Can't Save a Finding

Gate C2 (Cryptographic Weaknesses) has a "narrative framing" note about leading with the broken mechanism instead of the symptom. But framing only works when there IS a security boundary being crossed. If the server-side enforcement is intact, no amount of reframing turns a bug into a vulnerability.

**Test**: "If the vendor fixed this bug tomorrow, would any security boundary still be crossed?" If the answer is no (because no boundary was ever crossed), the bug was never a vulnerability.

### The Credential-Access Gap Gate — NEW

A found credential (API key, secret, token) is NOT valuable unless you can demonstrate it unlocks something. Apply this gate BEFORE reporting any credential leak:

**Three-layer test:**
1. **Can you authenticate with it?** — Try the credential against the target API/system. 401 = dead end without a corresponding user token or additional factor.
2. **Does it unlock something?** — Even if you can authenticate, list what endpoints become accessible. If they're all public/PII-free, impact is low.
3. **Can a real attacker reach those endpoints?** — Are they network-accessible, or do they require being on an internal network/VPN?

**Real example (Nutaku, 2026)**:
- Finding: 224 OpenSocial OAuth 1.0 consumerSecrets leaked from a catalog API
- Gate 1: Could not authenticate to OSAPI with leaked secrets alone (OAuth 1.0 also requires user token)
- Gate 2: Could not find any endpoint that accepts consumerSecret as a standalone credential
- Gate 3: Some target endpoints (metaapi, userapi) returned 403 from researcher IP
- **Result: Finding dropped from High to Non-reportable.** Leaked credentials with no exploitable path = Informational at best.

**Lesson**: A credential leak is not a vulnerability unless you can demonstrate one of:
- A working authenticated API call using the leaked credential
- An endpoint that trusts the credential as proof of authorization
- A chain that combines the credential with another finding to achieve impact

---

## Reference: What Got Rejected and Why

| Finding | Claim | Class | Disposition | Why | Gates Failed |
|---------|-------|-------|-------------|-----|-------------|
| Nutaku hardcoded OAuth2 creds | CWE-798 in mobile APK | Credential Exposure | **Informative** | Triager: "You're basically logging into your own account. This does not pose any security risk." Mobile app creds talking to their own API is industry standard. No cross-boundary access demonstrated. RFC 6749 argument didn't overcome lack of harm. | C4, T5 |
| Nutaku missing auth on favorites | CWE-287 improper auth on GET endpoint | Access Control | **Out of Scope** | Program OOS: "IDOR with no direct security or financial impact." Exposed game titles only. Report used the keyword "IDOR" which matched the OOS rule. Could have been framed as CWE-287, but data sensitivity too low to overcome OOS even with correct framing. | P2 (keyword trap), T1 (scope) |

**Related reference**: `references/two-account-idor-proof-methodology.md` — two-account PoC approach, shell variable masking workarounds for writing PoC scripts, and the write/read asymmetry signal for finding missing authentication.
**Related reference**: `references/nutaku-rejection-analysis-jun2026.md` — full post-mortem of two Nutaku rejections with counter-argument viability analysis and the keyword grep trap that killed the second finding.

| Fireblocks bodyHash bug | Cryptographic integrity failure | Crypto | **Not submitted** | Real bug but server-side security intact. Quality issue, not vulnerability. | Part 6 decision |
| Fireblocks bodyHash (original version) | P2 replay attack (false claim) | Crypto | Draft caught in review | Replay claim was never tested. Nonce tracking blocks it (code -13). | S1, S3 |
| Fireblocks MPC 004 (Paillier oracle) | Key recovery in ~300 queries | Crypto | N/A (AI-generated) | Oracle confirmed but key recovery never demonstrated. Title overclaims PoC output. | C9, C10, S5, U1 |
| Fireblocks MPC 005 (Version + key rotation) | Permanent compromise surviving refresh | Crypto + Architecture | N/A (AI-generated) | Chained 3 undemonstrated findings to claim P3. PoC proves constants exist, not exploit. | U1, U4, S5, R2 |

**Related reference**: `references/mpc-rejection-case-studies-jun2026.md` — full analysis of both rejections with specific pipeline failures and what would have passed triage.

## Part 7 — Post-Rejection Analysis (Added Jun 2026 from MPC rejection feedback)

Every rejection is data. Run after every rejection.

### Gate R0 — Record the Lesson (MANDATORY — added Aug 2026)

Before anything else, append the outcome to the lesson bank:
- Write a dated entry to `~/Dev/ATLAS-LEARNINGS/LESSONS.md` §01 (what was rejected, why, which gate failed, what to do differently)
- Update the Track Record table in `h1-submission-lessons` skill if applicable
- If MEMORY.md is near full, compress to a pointer: `See LESSONS.md §01`
- Load `atlas-lesson-bank` skill for the full workflow

This applies to EVERY outcome: accepted, rejected, duplicate, informative, N/A, OOS. Every rejection is data; every acceptance is a pattern worth keeping.

### Gate R1 — Triage Gate That Failed

| Signal | What It Means | Fix |
|--------|--------------|-----|
| "Out of Scope" | Asset or vuln type OOS | Check program scope |
| "Not reproducible" / "lacks clear PoC" | No working PoC | Write PoC FIRST |
| "Informative" | Not a real vulnerability | Run Part 6 (Bug vs Vulnerability) |
| "Duplicate" | Someone else found it first | Speed, depth, or niche target |
| "AI-generated" / "automated" | Report reads like AI wrote it | See Gate R2 |

### Gate R2 — What Triggered "AI-Generated" Flag (2 rejections, Jun 2026)

Two crypto submissions rejected with "content appears to be low-quality or potentially automated (AI-generated)." Specific triggers:

1. **Long structured sections** — triage sees templated output
2. **Explaining basics** — "Paillier is a public-key encryption scheme..." assumes reviewer needs crypto 101
3. **No raw PoC output** — described what PoC DOES without showing what it PRODUCED
4. **Theoretical impact** — "300 queries would recover key" without actually doing it
5. **Over-broad titles** — "Key Recovery via..." without having recovered a key
6. **Content-padding headings** — Section titled "Exploitation" that only describes the setup

**Fix each (cross-reference Gate U9 — Anti-AI Detection Gate):**
1. Narrative paragraphs, not sectioned structure → **Gate U9 Check 2**
2. Delete all explanatory text → **Gate U9 Check 3**
3. Paste actual terminal output inline → **Gate U9 Check 4**
4. Demo it or don't claim it → **Gate U9 Check 1**
5. Title matches exactly what PoC proves → **Gate U9 Check 6**
6. Every heading delivers what it promises → **Gate U9 Check 2**

**After fixing, load and apply the humanizer skill** (`skill_view(name='humanizer')`, Bug Bounty Report Humanization section, 3-Pass Process). This is the rewrite engine that turns the fixed content into natural narrative form.

### Gate R3 — Honest Scope Check

Before resubmitting:
- Is this a vulnerability or expected behavior?
- If vendor fixed it, would a security boundary become uncrossable?
- Or was no boundary ever crossed? → Not a vulnerability.

### Gate C9 — Crypto PoC: Must Be End-to-End (NEW)

For crypto submissions specifically:

| Chain Step | Must Show | Our Failure |
|-----------|----------|-------------|
| Oracle exists | Different errors per input | ✅ Had this |
| Oracle extracts info | Script output showing leaked bits | ❌ Theorized only |
| Full key recovery | "λ = 0x..." in terminal | ❌ Never ran it |
| Escalation | Decrypted server share | ❌ Never ran it |

Missing any step = theorized impact = rejection.

### Gate C10 — Crypto PoC Format: Standalone Script (NEW)

The PoC must be a single self-contained script:

```bash
python3 exploit.py
# Output: Recovered Paillier private key: 0xDEADBEEF
```

Not "append to test file, build, run the suite." Not "here's the oracle setup, you could recover a key."

**Minimum viable crypto PoC**: A script that takes the target's public key and outputs a leaked private key bit or extracted plaintext. Terminal output must be included inline in the report.

### Gate U5 — PoC is reproducible for TRIAGE, not just for you

Before submitting, verify the reviewer can reproduce your finding with ZERO access to your environment.

**The fresh-clone test:**
```bash
git clone --depth 1 $TARGET_URL /tmp/fresh
cd /tmp/fresh
bash /path/to/poc.sh
```

If this fails, your PoC depends on your specific checkout, build artifacts, or environment — and triage will mark it "Not Reproducible."

**Checklist (all must be YES):**
- [ ] One `git clone` gets the target code
- [ ] One command runs the PoC (no build step)
- [ ] No `apt-get install` of special dependencies
- [ ] No appending to existing test files
- [ ] No custom branch or commit hash required
- [ ] PoC output (terminal text) clearly shows the vulnerability

**Why this matters:** Rejected reports 004 and 005 (Fireblocks MPC, Jun 2026) both passed our internal tests but failed for Bugcrowd triage. The PoCs required access to our local repo with appended test code, modified CMakeLists.txt, and incremental build artifacts. A fresh clone + one grep command would have worked. The appended-test-file approach did not.

## Part 8 — Post-Rejection: Counter-Argument or Move On (Added Jun 2026)

Every rejection is data. Before deciding to counter-argue or move on, run this framework.

### Gate D1 — Map the Rejection to a Gate

| Disposition | Which Gate Failed | Can You Fix It? |
|-------------|------------------|-----------------|
| Out of Scope | Gate T1 (Scope) | Only if you can reframe the vuln class without using OOS keywords — AND the new framing changes the class entirely. If the CORE issue is OOS, move on. |
| Informative | Gate T3 (Validity) or Gate T5 (Impact) | Counter-argument: demonstrate the missing impact that triage said was absent. If you can't, the finding was correctly Informative. |
| Duplicate | Gate T4 (Uniqueness) | Counter-argument: show your finding covers a scope the dupe didn't. Rarely works — move on. |
| Needs More Info | Gate T2 (Reproducibility) | Easy fix — provide clearer steps, curl commands, response examples. |
| AI-generated flag | Gate R2 | Fix the report format (narrative, not templated). No counter-argument — rewrite. |

### Gate D2 — Viability Test for Counter-Arguments

Before drafting a response to the triager, answer these questions:

**Q1 — Is the infrastructure still live?**
- Does the endpoint still exist? (curl test)
- Are the credentials/tokens still valid?
- If the answer is no, the finding is dead — do NOT counter-argue.

**Real example (Nutaku gateway creds, Jun 2026)**: The OAuth-front credentials were marked Informative. We considered counter-arguing, but `api.nutaku.net/gateway/v1` now returns Connection Refused. The entire gateway API has been taken down. There is no live endpoint to demonstrate impact against. Counter-argument has zero chance. → Move on.

**Q2 — Is the data sensitive enough?**
- Would a reasonable person agree the exposed data causes harm?
- Game titles? No. Game titles on an adult platform? Still no — the program explicitly excludes it.
- Email addresses, password hashes, payment info, credit cards? Yes.

**Q3 — Can you reframe the CLASS without changing the EVIDENCE?**
- If the program excludes IDOR but you have CWE-287, reframing might work
- If the program excludes hardcoded creds but you can show actual token reuse impacting another user, reframing works
- If the program excludes all information disclosure regardless of framing, reframing won't work

**Q4 — Is there new evidence you collected AFTER the rejection?**
- New endpoint found? New credential test? New data sample?
- If yes, submit a NEW report with the chain (don't counter-argue the old one)
- If no, the original triage was correct — move on

**Q5 — Write the 3-sentence attack story (from Gate V6)**
- V1: Who is the victim?
- V2: What must they do?
- V3: What is the actual harm?
- If you can't write this, the finding wasn't ready.

### Gate D3 — Decision Matrix

| Q1 (Live?) | Q2 (Sensitive?) | Q3 (Reframe?) | Q4 (New evidence?) | Action |
|-----------|----------------|--------------|-------------------|--------|
| Yes | Yes | Yes | — | **Counter-argue** — write a response addressing the specific gate that failed |
| Yes | Yes | No | — | **Submit NEW report** with the actual sensitive data demonstration |
| Yes | No | — | — | **Move on** — the program doesn't value this data class |
| No | — | — | — | **Move on** — dead infrastructure means no demonstration possible |
| — | — | Yes | Yes | **Submit NEW report** — the new evidence changes the finding entirely |

### Gate D4 — How to Write a Counter-Argument

If the matrix says counter-argue, write a concise response that:

1. **Quotes the triager's specific reason** (e.g., "You said 'IDOR with no direct security or financial impact'")
2. **Acknowledges their reasoning** ("I understand the data on this endpoint is low-sensitivity")
3. **Presents the NEW evidence** ("However, I've now discovered the same auth bypass applies to the user profile endpoint, which leaks email addresses")
4. **Requests re-opening**: "Please re-open for additional review."

**Do NOT**:
- Argue semantics without new evidence
- Quote RFCs or OWASP standards as authority without demonstrating impact
- Blame the triager or accuse them of misreading
- Submit a counter-argument that takes more than 4 sentences — triagers are busy

### Gate D5 — The Infrastructure Dead End Pattern (NEW)

This pattern killed a finding this session and will kill others:

**When the credential/token/endpoint is no longer accessible, the finding is dead.**

Signs:
- Connection refused on the API endpoint
- 404 on the old URL
- Authentication rejection that wasn't there before
- Service shutdown / deprecation notice

**Do NOT try to counter-argue a dead endpoint.** Even if the triager was wrong about the vulnerability class, you cannot demonstrate impact against infrastructure that doesn't respond. Submit a fresh report if a new live endpoint appears, but don't waste time contesting the rejection.

---

### Gate U6 — Data Sensitivity Gate (Added Jun 2026 from AdultForce rejection)

**A technically valid vulnerability with low-value data is Informative. The mechanism does not save you. The data does.**

Before submitting ANY finding where the harm comes from exposed data, classify the data:

| Class | Examples | Verdict |
|-------|----------|---------|
| P1 — Credentials | Working API keys, tokens, passwords | ✅ Submit |
| P2 — PII | Emails, password hashes, payment info | ✅ Submit |
| P3 — Actionable Financial | Billing tokens, card data, refund endpoints | ⚠️ Submit if demonstrable |
| P4 — Business Metadata | Internal IDs, infrastructure paths, org structure | ❌ Do NOT submit |
| P5 — Operational | Maintenance messages, feature flags | ❌ Do NOT submit |
| P6 — Public | Public content, thumbnails | ❌ Do NOT submit |

**The 3-Question Test:**

1. Is the data P1, P2, or P3? (credentials/PII/actionable)
2. Does the data enable a DIRECT action (login, access account, process payment)?
3. Can you write a victim story where the harm isn't "a competitor could see this"?

If any answer is NO, do not submit.

**The self-diagnosis**: Ask "Would peaches call this business metadata?" If yes, don't submit.

### Gate U7 — Standard Disclosure Terms check

Bugcrowd's Standard Disclosure Terms define two categories of rejections:

**Excluded (never rewardable, immediately invalid):**
- Physical testing (office access, tailgating)
- Social engineering (phishing, vishing)
- OOS targets
- UI/UX bugs, spelling mistakes
- DoS/DDoS

**Non-qualifying (low impact — don't report without chaining):**
- Descriptive error messages
- Banner disclosure
- Missing security headers
- Clickjacking
- CSRF on anonymous forms
- Weak Captcha
- Username enumeration
- Login brute force / account lockout
- SSL config issues (BEAST, BREACH, weak ciphers, missing HSTS/XXSS protection)

Before any submission, check:
1. Is the finding excluded outright? (physical, social, OOS, functional, DoS) → Don't submit.
2. Is the finding on the non-qualifying list? → Don't submit unless you can chain to higher impact.
3. Does the finding affect "the target's users, systems, or data security in a meaningful way"? → This is the standard for P4-P5 submissions. Be prepared to defend it.

For crypto/MPC findings specifically: the non-qualifying list is web-app-focused (missing headers, SSL config, CSRF). None of those apply to cryptographic implementation bugs. But the "impact" requirement still applies — be ready to explain why the finding matters to security posture.

### Gate U8 — Multi-Target Audit Format

After a structured hunt across multiple programs, compile findings using
`references/multi-target-audit-format.md` in this skill. Format requires:
- Previous mistake section per target
- Submittable/Not/Lead verdict per finding
- Methodology validation summary at the end

**Zero findings is an acceptable outcome.** If the corrected methodology prevented
N likely-rejected submissions, state that explicitly. The goal is quality submissions,
not volume. A session with zero false submissions is a success.
