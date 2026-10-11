# Report Writing Best Practice (detail)

Detail moved out of SKILL.md for progressive disclosure (loaded on demand).

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

