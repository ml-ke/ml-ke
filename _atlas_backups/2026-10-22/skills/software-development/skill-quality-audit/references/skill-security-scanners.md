# Skill Security Scanners — Comparison & Triage (verified Oct 2026)

Live-tested against 224 installed OpenCode skills + the Hermes library. Two scanners
available; SkillSpector is the primary for static-only library scans.

## Scanner environment (read this before touching the venvs)

Three venvs, **one tool each** — the tools' pinned deps conflict at current versions:

| Path | Tool | Why separate |
|---|---|---|
| `~/.hermes/venvs/skillsec` | SkillSpector 2.9.6 (+`skill-scanner` shim) | primary static scanner |
| `~/.hermes/venvs/skillscanner` | cisco `skill-scanner` 0.3.3 | pins `typer==0.25.1`; SkillSpector pins `typer>=0.23,<0.24` |
| `~/.hermes/venvs/skillsec-snyk` | `snyk-agent-scan==0.5.16` | pins `rich==14.2.0`; SkillSpector 2.9.6 needs `rich>=14.3.0` |

Co-installing all three in one venv is **unsatisfiable** (that is why the old single-venv
recipe stopped working). `~/.hermes/venvs/skillsec/bin/skill-scanner` and
`.../bin/snyk-agent-scan` are two-line `sh` shims that exec the other venvs, so every
documented path still works and `skill_second_opinion.py` needs no change.

Build (idempotent):

```bash
# Pin the interpreter with uv — do NOT create these venvs from /usr/bin/python3.
uv python install 3.12
uv venv --python 3.12 ~/.hermes/venvs/skillsec
uv venv --python 3.12 ~/.hermes/venvs/skillscanner
uv venv --python 3.12 ~/.hermes/venvs/skillsec-snyk
uv pip install --python ~/.hermes/venvs/skillsec/bin/python \
  "skillspector @ git+https://github.com/NVIDIA/SkillSpector@v2.9.6" pyyaml
uv pip install --python ~/.hermes/venvs/skillscanner/bin/python "skill-scanner==0.3.3"
uv pip install --python ~/.hermes/venvs/skillsec-snyk/bin/python "snyk-agent-scan==0.5.16"
```

> ⚠️ Gotcha — **a distro `python3` upgrade orphans every venv built on it.** `/usr/bin/python3`
> follows the default version (3.12 → 3.14 on this box), while packages stay in
> `lib/python3.<old>/site-packages`. Every pinned tool then dies with `ModuleNotFoundError`
> *for a package pip says is installed* — and a scanner that cannot start looks exactly like a
> scanner that found nothing. Detect before trusting a scan: `python3
> ~/.hermes/scripts/venv_integrity_check.py --deep` (reports DEAD/MIGRATED per venv). Use a
> **non-editable** install from the pinned tag — the old `pip install -e` left a `.pth` that
> silently stopped resolving — and keep the venv interpreter uv-managed so a system upgrade
> cannot move it.

> ⚠️ Gotcha — **v2.9.6 JSON schema.** Per-skill findings are under `skills[].issues[]`
> (id, category, pattern, severity, confidence, location, finding) with `risk_score` /
> `risk_severity` / `finding_count` alongside. A parser reading `findings` sees an empty list and
> reports "clean" — that is a schema mismatch, not a clean library.

## Scanner comparison

| | NVIDIA SkillSpector | snyk-agent-scan |
|---|---|---|
| Version tested | 2.9.6 (pin the tag; record the resolved SHA) | 0.5.16 |
| Install | `uv pip install "skillspector @ git+…@v2.9.6"` (not on PyPI) | `uv pip install snyk-agent-scan==0.5.16` (own venv) |
| Static-only mode | `--no-llm` (no API keys) | no true static mode |
| Patterns | 68 patterns / 17 categories | prompt injection + malware payloads |
| Risk scoring | 0-100 + severity | issue list |
| MCP server startup | No | **Yes — starts stdio MCP servers during scan; CI mode needs `--dangerously-run-mcp-servers`** |
| JSON output | `--format json --output file.json` | `--json` |
| Recursion | `--recursive` (may only descend 1 level in some builds) | scans well-known paths only |

**Decision:** use SkillSpector `--no-llm` for scanning skill libraries (safe, no
network side effects). Use snyk only when MCP configs are trusted and you accept
server startup. Never run snyk in CI mode without explicit trust confirmation.

## Correct SkillSpector invocations

```bash
# single skill dir
skillspector scan ~/.config/opencode/skills/atlas-jwt-attacks --no-llm --format json

# whole library (recursive) — save JSON for triage
skillspector scan ~/.config/opencode/skills --recursive --no-llm --format json --output /tmp/scan.json

# Hermes library: categories are nested 2 levels (category/skill) — scan each category
for d in ~/.hermes/skills/*/; do
  skillspector scan "$d" --recursive --no-llm --format json --output "/tmp/scan-$(basename $d).json"
done
```

Notes:
- `--no-llm` mode: `risk_score` is populated, `risk_level`/`risk_severity` shows `?` — trust the score.
- Findings live under `skills[].issues[]` in the JSON (keys: id, category, pattern, severity, confidence, location, finding, explanation). `findings`/`risk_assessment` keys in the per-skill object are mostly empty in no-llm mode; read `issues`.
- On security content the scanner is aggressive: a clean-looking score is meaningful, a high score needs manual triage.

## Triage — expected false positives on offensive-security content

Live-verified against tob-* (Trail of Bits), atlas-* (ATLAS BB methodology), and anthropic-* skills:

| Scanner category | What triggers it | Verdict on BB/pentest skills |
|---|---|---|
| Data Exfiltration / External Transmission | any `curl`/HTTP request to a remote host | FP — that IS the methodology |
| Prompt Injection / Hidden instructions | BOM char (U+FEFF), zero-width chars, HTML comments | FP on docx/pptx/xlsx skills (BOM at file start) |
| Supply Chain / External Script Fetching | `curl <url> \| bash` of official installers (dl.google.com, railway) | FP if URL is the vendor's official installer |
| Rogue Agent / persistence | "cron", "startup", XML/plist boilerplate strings | FP on document skills (word "pList" triggers it) |
| Anti-Refusal | "don't apologize", "omit warnings" style guidance | FP — style guidance, not refusal bypass |
| MCP Least Privilege | no declared allowed-tools/permissions | Advisory — add `allowed-tools` if desired |
| Dependency version pinning (LOW) | `dep>=1.0` unpinned | Advisory — real but low priority |

**Real catches worth acting on:** verbatim instruction-override directive strings (the "ignore earlier directives" / "override prior guidance" class); `curl <attacker-controlled> | sh`; base64-encoded
commands with no explanation; `--dangerously-skip-permissions`; over-broad
`allowed-tools`; skills that read `$env` secrets and POST them somewhere.

## Triage script

Save as `~/hermes/scripts/skill_scan_triage.py` (re-runnable):

```python
#!/usr/bin/env python3
"""Triage SkillSpector JSON: print findings in the real-risk categories."""
import json, sys

d = json.load(open(sys.argv[1]))
target_cats = {"Prompt Injection", "Anti-Refusal", "Supply Chain",
               "System Prompt Leakage", "Rogue Agent"}
count = 0
for s in d["skills"]:
    for i in (s.get("issues") or []):
        if i.get("category") in target_cats:
            count += 1
            finding = (i.get("finding") or "").replace("\n", " ")[:140]
            expl = (i.get("explanation") or "")[:110]
            print(f"[{i.get('severity')}] {s['name']} | {i.get('category')} | {finding} | {expl}")
print(f"\nTOTAL in real-risk categories: {count}")
```

## Live scan results (Aug 10 2026)

- **OpenCode library (194 skills):** 102 flagged; severity 251 HIGH / 487 MEDIUM / 68 LOW / 6 CRITICAL. Categories: 247 MCP Rug Pull (noise on non-MCP skills), 107 Excessive Agency, 91 Privilege Escalation, 70 Data Exfiltration, 51 Dangerous Code Execution, 47 Rogue Agent, 19 Prompt Injection, 22 Supply Chain, 14 Memory Poisoning, 12 Anti-Refusal, 6 System Prompt Leakage. **Verdict: no genuine malicious skills.**
- **Hermes library (top-level only, 11 skills scanned):** 4 flagged (yuanbao 48, ai-agent-bug-bounty-methodology 31, jwt-attacks 23, ssrf-testing 22) — all methodology skills with curl commands, expected FPs. Note the recursion quirk: nested category skills weren't scanned; scan per-category.
- **snyk-agent-scan:** refused CI mode without `--dangerously-run-mcp-servers` (by design). Empty JSON output when MCP servers were declined. Not usable for unattended static scans — use SkillSpector.

## Key lesson

Scanner scores on security-tooling content are NOT a measure of maliciousness — they
measure how much curl/HTTP/encoded-payload content a skill contains. The genuinely
dangerous signal is *verbatim instruction-override text* and *unexplained
attacker-controlled execution*. Triage with the script above; act on real catches,
ignore category counts on security content.

## Cron injection-scanner sweep (run before attaching ANY skill to a cron job)

Hermes's cron scheduler scans the assembled prompt (job prompt + loaded skill bodies)
against `_CRON_SKILL_ASSEMBLED_PATTERNS = _CRON_THREAT_PATTERNS[:4]` in
`tools/cronjob_prompt_scan.py` → `_scan_cron_skill_assembled(assembled) -> (label, message)`
(`tools/threat_patterns.py` holds the sibling `scan_for_threats(content, scope)`). The
gate moved out of `tools/cronjob_tools.py` in 2026-09 — grepping the old file finds nothing,
and grepping `threat_patterns.py` for the assembled-skill function finds `None`. If an attached
skill contains a **literal instruction-override phrase** — common in security skills
that teach you to grep for injection strings — the whole job is silently BLOCKED with
a false `prompt_injection` hit. Verified Aug 2026: killed the weekly sweep 2 weeks
running. Full diagnosis + fix recipe: `hermes-maintenance` →
`references/cron-injection-scanner-false-positive.md`.

**Critical trap:** the `\s` in the patterns matches **newlines**, so line-based `grep`
MISSES phrases wrapped across lines. Always scan with Python `re` over full file content:

```python
import re, os
pats = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', 'prompt_injection'),
    (r'do\s+not\s+tell\s+the\s+user', 'deception_hide'),
    (r'system\s+prompt\s+override', 'sys_prompt_override'),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', 'disregard_rules'),
]
for root, dirs, files in os.walk('<skill_root>'):
    for f in files:
        if not f.endswith(('.md','.py','.sh','.txt','.json','.yaml','.yml')): continue
        data = open(os.path.join(root,f), encoding='utf-8', errors='replace').read()
        for pat, label in pats:
            for m in re.finditer(pat, data, re.IGNORECASE):
                print(f"[{label}] {os.path.join(root,f)}")
```

**Meta-pitfall:** your own patch/fix note can re-trigger the scanner if it *quotes* the
forbidden phrase — the Aug 2026 fix note itself tripped all 4 patterns on first draft.
After editing any security skill, re-run this sweep AND verify with the real scanner
(`tools.cronjob_prompt_scan._scan_cron_skill_assembled`) before attaching it to a cron job.

## Reporting a finding: separate capability context from confirmed concern (Oct 2026)

Skillstore's audit methodology (skillstore.io/security, Jul 2026) makes the distinction that
our triage was missing: publish **capability context** and **confirmed security concern** as
separate fields, and never let a trust signal substitute for another ("public" ≠ safe,
"audited" ≠ safe, "signed manifest" ≠ harmless). Practical form for this library:

- For each flagged skill, record the capability the finding is about (e.g. `curl` to a target
  host, `os.environ` read) **separately** from whether that capability is unexplained by the
  skill's prose. A finding is only a *concern* when the capability is undeclared or the
  composition is wrong; otherwise it is context.
- Bind every verdict to the artifact it describes: skill path + content hash + scanner version
  + resolved scanner SHA. A verdict without a hash cannot be re-verified next week.

## The tier above static scanning: dynamic, state-conditioned behaviour

SkillSentry (arxiv 2608.03485, code: github.com/nizhangli062-jpg/SkillSentry-…) tests skills in
an LLM-simulated "honey world" with decoy resources, infers the intended capability boundary,
and compares skill-enabled against **matched no-skill executions** before deciding — 99.50%
recall, and 92.95% F1 under semantics-preserving evasion vs 80.07% for the best static/LLM
baseline. The transferable rule for this pipeline: a skill that is benign under inspection can
misbehave only at a particular state, resource, or history, so **neither a clean static scan nor
an LLM read is evidence about conditional behaviour** — only a run with decoys and a matched
no-skill control is. Do not upgrade a "clean" verdict into a claim of safety on that evidence;
for high-privilege or third-party skills, prefer the paired-run question: what does the skill do
here that the same task without the skill does not?

## Optional third linter: agnix (`uvx agnix <root>`, no install)

457 rules across Claude Code / Codex / OpenCode / Cursor / Copilot configs (agent-sh/agnix,
MIT/Apache-2.0). Useful and different from `skill_validator.py` (frontmatter spec only), but its
FP profile on this library makes it a **reporting aid, not a gate**: 134/203 errors are
`metadata must be a map from string keys to string values` (our `metadata.hermes` carries nested
lists — Hermes-specific, not a defect here, but a genuine **portability** finding for anything
re-exported), and ~26 more errors are `Import target not found: @bugcrowdninja.com` — it parses
our own bug-bounty handles as import specifiers. Real signal it did produce: hard-coded
`/home/pro-g` paths and client-specific frontmatter fields. Read the rule name, not the count.

## Live scan results (Oct 6 2026, SkillSpector 2.9.6, 224 skills)

25 skills ≥50 risk, 11 ≥80, 888 findings (252 HIGH / 516 MEDIUM / 111 LOW / 6 CRITICAL). Top
categories unchanged: MCP Rug Pull 268 (RP1, noise on non-MCP skills), Excessive Agency 146,
Privilege Escalation 96, Data Exfiltration 64, Rogue Agent 55. All 6 CRITICAL-severity findings
are YARA matches on security *teaching* content (`nc attacker.com 4444`, `CobaltStrike`,
`coinhive.min.js`, `preg_replace('/.*/e'` inside a pentest reference or a YARA-authoring skill).
Full marker sweep (verbatim override text, non-vendor `curl|sh`, unexplained base64, undeclared
secret reads) over both libraries found **zero new genuine catches** — every hit is the documented
teaching-text/false-positive class. **8th consecutive all-false-positive run.**
