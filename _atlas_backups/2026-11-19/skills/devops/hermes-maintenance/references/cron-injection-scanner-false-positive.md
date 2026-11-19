# Cron Job Blocked by Injection Scanner (False Positive)

Verified Aug 2026: the weekly agent-skills sweep was silently killed for 2 weeks
(Aug 4 + Aug 11) by this exact failure mode. This is the diagnostic recipe.

## Symptom

- `hermes cron list` (or the cronjob tool) shows `last_status: error` for a job
  that has **skills attached**.
- The output file at `~/.hermes/cron/output/<job_id>/<timestamp>.md` contains:

```
**Status:** BLOCKED

The assembled prompt (user prompt + loaded skill content) tripped the cron
injection scanner and the agent was NOT run.

**Scanner result:** Blocked: prompt matches threat pattern 'prompt_injection'.
```

- `~/.hermes/logs/errors.log` shows the same WARNING from `cron.scheduler`.
- From the user's perspective the job just "does nothing" — no delivery, no
  report. Check `last_status` before assuming the scheduler is down.

## Root cause

A skill attached to the job contains a **literal instruction-override phrase**
(e.g. the "ignore earlier directives" class) — very common in security skills,
which legitimately teach you to grep for injection strings in untrusted skills.
Hermes's cron scanner (`tools/cronjob_tools.py::_scan_cron_skill_assembled`)
regex-matches the verbatim phrase in the assembled prompt (job prompt + loaded
skill bodies) and blocks the entire run as a suspected `prompt_injection`
payload. A tripwire meant to catch malicious skills trips on its own training
manual.

## The 4 assembled-prompt threat patterns

From `_CRON_SKILL_ASSEMBLED_PATTERNS` in `tools/cronjob_tools.py`:

```python
(r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
(r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
(r'system\s+prompt\s+override', "sys_prompt_override"),
(r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
```

**Critical trap:** the `\s` in these patterns matches **newlines**, so a
line-based `grep` will MISS a phrase wrapped across lines. Always scan with a
Python `re.finditer(..., re.IGNORECASE)` over the whole file content.

## Fix

1. **Rephrase, don't quote.** In the offending skill, describe the class instead
   of writing the literal string:
   - "ignore previous instructions" → "instruction-override directives (e.g.
     'ignore earlier directives')"
   - "system prompt override" → "system prompt breakout"
   - "do not tell the user" → paraphrase ("withhold X from the user")
   Watch your own patch: a pitfall note that *quotes* the forbidden phrase
   re-triggers the scanner. Verify after every edit.

2. **Verify with the REAL scanner** — don't just eyeball. From the hermes-agent
   repo root:

```python
import sys, json
sys.path.insert(0, '.')
import tools.cronjob_tools as cjt

with open('/home/pro-g/.hermes/cron/jobs.json') as f:
    data = json.load(f)
jobs = data if isinstance(data, list) else data.get('jobs', [])
job = next(j for j in jobs if (j.get('job_id') or j.get('id')) == '<JOB_ID>')
assembled = job.get('prompt', '')
for s in job.get('skills', []):
    # append each skill's SKILL.md content (paths vary by category)
    assembled += "\n\n" + open('<path/to/skill/SKILL.md>').read()

result = cjt._scan_cron_skill_assembled(assembled)
print(result or "CLEAN")
```

3. **Sweep the whole skill library** for the same false positives so no OTHER
   cron job breaks later (any job with skills attached is at risk, not just the
   one that failed):

```python
import re, os
pats = [ (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', 'prompt_injection'),
         (r'do\s+not\s+tell\s+the\s+user', 'deception_hide'),
         (r'system\s+prompt\s+override', 'sys_prompt_override'),
         (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', 'disregard_rules') ]
for root, dirs, files in os.walk('<skill_root>'):
    for f in files:
        if not f.endswith(('.md','.py','.sh','.txt','.json','.yaml','.yml')): continue
        data = open(os.path.join(root,f), encoding='utf-8', errors='replace').read()
        for pat, label in pats:
            for m in re.finditer(pat, data, re.IGNORECASE):
                print(f"[{label}] {os.path.join(root,f)}")
```

4. **Re-trigger the job** to catch up on the missed runs:
   `cronjob(action='run', job_id='...')` — it fires on the next scheduler tick
   and delivers its report normally. Check `agent.log` for
   `Running job '<name>'` to confirm it actually executed this time.

## Scope note

This false positive is specific to cron jobs with `skills:` attached. Bare
`script:`/`no_agent` jobs and the strict prompt scanner (`_scan_cron_prompt`)
are unaffected. The strict scanner's command-shape patterns (cat .env, rm -rf /,
authorized_keys, /etc/sudoers) are intentionally NOT in the assembled set
because security docs legitimately describe them in prose.
