# OWASP Agentic Skills Top 10 (AST10) — the security axis

Source: https://owasp.org/www-project-agentic-skills-top-10/ (published April 2026, OWASP Foundation;
project leads Ken Huang, Hammad Atta, Fabio Cerullo, Aonan Guan, Bhavya Gupta, Niv Hoffman, Iftach Orr,
Akram Sheriff). Reputability: official standards body, peer-reviewed by >100 practitioners.

Why it exists here: the 26-smell taxonomy (arxiv 2607.01456) measures **quality** — does this skill
work and route correctly. AST10 measures **security** — can this skill or its distribution path hurt
the host. An audit that only runs one of the two has a blind half. AST10 is also the first formal
statement that a skill is not a plugin: it runs in the agent's full security context.

## Lifecycle map — where each risk lands

| Phase | Risks |
|---|---|
| Author & Publish | AST04 Insecure Metadata |
| Distribute & Install | AST02 Supply Chain |
| Load & Permission | AST01 Malicious Skills, AST07 Update Drift |
| Execute & Isolate | AST05 External Instructions, AST03 Over-Privileged, AST06 Weak Isolation |
| Detect & Govern | AST08 Poor Scanning, AST09 No Governance, AST10 Cross-Platform Reuse |

## The ten risks

- **AST01 Malicious Skills (Critical)** — looks legitimate, hides credential stealers, reverse
  shells, or prose that hijacks the agent.
- **AST02 Supply Chain Compromise (Critical)** — registries without provenance let attackers
  mass-upload, take over accounts, poison distribution.
- **AST03 Over-Privileged Skills (High)** — access far beyond need; a prompt injection into the
  skill becomes a huge blast radius.
- **AST04 Insecure Metadata (High)** — unvalidated/unsigned metadata enables brand impersonation,
  understated permissions, poisoned search.
- **AST05 Untrusted External Instructions (High)** — a skill that points the agent at external
  documents trusts mutable, unpinnable content that can be swapped for hostile instructions later.
- **AST06 Weak Isolation (High)** — skills run in the agent's full security context; with no
  sandbox, every skill is a potential full-system compromise.
- **AST07 Update Drift (Medium)** — without pinning or verification, a skill silently drifts to a
  vulnerable or freshly malicious version.
- **AST08 Poor Scanning (Medium)** — natural-language-plus-code blends defeat signature scanners,
  so malicious skills pass every automated check.
- **AST09 No Governance (Medium)** — no inventory, approval, audit, or revocation: a shadow-AI
  layer nobody can see or control.
- **AST10 Cross-Platform Reuse (Medium)** — porting a skill across platforms drops the source
  format's security metadata, opening exploitable gaps.

## Tier-0 checks (run on every newly installed / externally sourced skill)

Do not run the full list on a skill you just wrote — that is the excessive-procedure smell. Tier 0
is cheap and always applies; Tier 1 is already done for skills under `~/.hermes/skills` because
`skill_validator.py` enforces it mechanically.

1. **Provenance (AST02/AST07/AST10)** — can you name the source repo AND the resolved commit SHA or
   content hash? A skill installed from a moving branch cannot be re-verified later. For bridged
   copies, `metadata.atlas-source` + `atlas-source-sha256` must be present; run the drift reporter
   instead of trusting the copy (`opencode_skills_bridge.py --check`).
2. **Permissions vs function (AST03)** — read `allowed-tools:`. Flag any skill whose declared tools
   are broader than the work described (a formatting skill with `Bash(*)` is the classic shape).
3. **Metadata integrity (AST04)** — does the name/description claim a vendor or official standing
   the provenance does not support? Vendor-named directories with cookie-cutter bodies across many
   skills are the brand-impersonation shape.
4. **Routing intact (AST04/AST10)** — description still carries [what]+[when]+keywords after any
   port or reformat. A port that drops tags/category/related_skills is AST10 by definition.
5. **Static scan + one manual read of the prose** — the scan is a filter, not the verdict (AST08).

## Tier-1 checks (only for community/unknown-provenance or high-privilege skills)

- **Composition analysis (the modern attack shape).** Read the capability chain, not the individual
  calls. See below. This is the check that catches what every scanner misses.
- **External coupling (AST05)** — does the skill fetch instructions, tool definitions, or config
  from a URL at runtime? Mutable remote content is a standing invitation to be swapped later.
- **Isolation (AST06)** — what is the worst action this skill can reach if its prose is obeyed
  literally? Name the boundary before deciding severity.

## The composition check — why "zero findings" is not a clean bill of health

Three independent 2026 results say the same thing, and together they invalidate triage rules that
only look for obvious markers:

- **SkillDRE (arxiv 2609.32400)** evolved complete malicious skill packages through a two-stage
  red-team loop (scanner-guided + runtime-guided) against four victim models. Average attack
  success **45.28%**, and the final submitted skills received **no SkillScan findings** while largely
  preserving benign-task performance. A skill can be simultaneously scanner-clean and malicious.
- **Tencent Zhuque Lab** scanned 50,000+ ClawHub skills. Their case study is a skill that passed
  ClawHub's full stack — regex scanning, five injection-pattern detectors, an LLM safety evaluation,
  and VirusTotal — because every individual step looked reasonable and was honestly documented: a
  remote fetch, a multi-layer transport decoder presented as a binary-safe encoding requirement, and
  a deserialization call. **The attack was the composition, not any single step.** None of the pieces
  is a marker; the chain is the finding.
- **AST08 exists precisely because of this**: natural-language-plus-code blends defeat signature
  scanners, so malicious skills pass automated checks.

Practical rule: after a scanner returns zero findings, you have measured nothing about composition.
Spend the manual read on the **capability chain** — remote fetch + decoder + deserializer,
file-read + network-egress, or credential-path + network-egress. Two or more of those in one skill,
with a plausible-sounding justification in the prose, is the finding regardless of scanner verdict.

## Popularity is not a trust signal

- **SkillProbe** (Shanghai Jiao Tong, arxiv 2603.21019; 2,500 ClawHub skills): **>90% of
  high-download skills failed strict security audit** — the "popularity-security paradox". Download
  count does not predict safety, and high-risk skills cluster into one giant connected component,
  meaning cascade risk is systemic rather than sample-level.
- **Snyk ToxicSkills** (3,984 skills across OpenClaw, Claude Code, Cursor, VS Code): 280+ samples
  directly leaked credentials; every platform had substantial defects.
- **Silverfort** found ClawHub ranked skills by a download counter that an unauthenticated request
  could inflate. A malicious skill was pushed to rank #1 — and because agents preferentially install
  high-ranked skills when choosing tools autonomously, ranking manipulation equals automated mass
  poisoning of agents.

Consequence for this library: star counts and install counts are **discovery** signals only. Record
provenance (repo + resolved SHA) and read the content. Never let a popularity number substitute for
the Tier-0 checks, and never let an agent auto-install by rank.

## Supply-side signals worth recording

- 74.6% of the 50,000 skills (27,818) declared network-request permission; file-read + network-egress
  is a complete exfiltration path even when each half is individually justified.
- 15,427 developers produced the corpus, but the top 20 accounts published 12.9% of it, and one
  account shipped 955 skills in 90 days (~10.6/day) with sibling accounts naming-convention-linked.
  Bulk-template generation is the tell, not any single skill.
- **When auditing, weight the author's aggregate output and naming pattern** over the individual
  skill's polish. A professional-looking README is cheap; a coherent history is not.
- ClawHavoc (Feb 2026): 1,184 malicious skills, 247,693 confirmed installs, ~$2.3M stolen — at peak,
  5 of the top 7 downloads were malicious. Typosquatted brand names plus dual delivery (Markdown
  instructing credential theft, embedded shell deploying a stealer).

## Related

- SKILL.md Step 3 (security scan) — the operational entry point.
- `references/collection-level-seams.md` — collection-scale audit (Step 7).
- `references/skill-security-scanners.md` — scanner setup, schemas, and triage of false positives.
