# Impact Gate & Meta-Analysis (detail)

Detail moved out of SKILL.md for progressive disclosure (loaded on demand).

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

