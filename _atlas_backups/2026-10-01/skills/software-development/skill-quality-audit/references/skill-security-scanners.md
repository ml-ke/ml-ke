# Skill Security Scanners — Comparison & Triage (verified Aug 2026)

Live-tested against 194 installed OpenCode skills + the Hermes library. Two scanners
available; SkillSpector is the primary for static-only library scans.

## Scanner comparison

| | NVIDIA SkillSpector | snyk-agent-scan |
|---|---|---|
| Version tested | 2.8.2 | 0.5.16 |
| Install | git clone + `pip install -e .` (NOT on PyPI) | `pip install snyk-agent-scan` |
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
against `_CRON_SKILL_ASSEMBLED_PATTERNS` in `tools/cronjob_tools.py`. If an attached
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
(`tools.cronjob_tools._scan_cron_skill_assembled`) before attaching it to a cron job.
