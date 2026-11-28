# Multi-Target Audit Report Format (Added Jun 2026)

## When to Use

After completing a structured hunt across multiple programs (typically 2-3 targets
for a focused session). Single-target findings use the standard REPORT.md format.

## Report Structure

```
## Final Report: [N]-Target Audit

---

### Target 1: [Program Name] — [Methodology Applied]

**Previous mistake**: [What we would have done wrong before the corrected approach]

**Corrected approach**: [What we did differently]

**What we found**:

| Endpoint | Status | Data Returned | Vulnerability? |
|----------|--------|---------------|----------------|
| /path/1 | 200 | actual data | Yes/No/Lead |
| /path/2 | 403 | blocked | Expected |

**Verdict**: [Submittable or not, with reasoning]

**Lesson applied correctly**: [How the corrected methodology prevented a mistake]

---

### Target 2: [Program Name] — [Methodology Applied]

[Same structure as Target 1]

---

### Target N: [Program Name] — [Methodology Applied]

[Same structure]

---

## Methodology Validation

**What worked:**
1. [Technique that succeeded]
2. [Technique that succeeded]

**What still needs improvement:**
1. [Gap identified]
2. [Gap identified]

**Bottom line**: [X submittable findings, Z prevented false submissions]
```

## Key Rules

1. Every target must have a "Previous mistake" section — if you can't articulate
   what we would have done wrong before, you haven't applied the corrected methodology.
2. Every finding must explicitly state whether it's submittable or not. No maybes.
3. "Lead" means potential but needs more work — describe exactly what's needed.
4. If zero submittable findings, that's fine. State "correct methodology prevented
   N likely-rejected submissions."
5. Always include a "Lesson applied correctly" for each target to reinforce the
   corrected approach.
