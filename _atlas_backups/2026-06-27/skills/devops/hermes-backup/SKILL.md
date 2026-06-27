---
name: hermes-backup
description: >-
  System for backing up Hermes agent state — skills, memories, config, cron jobs,
  and scripts — to a durable GitHub repository. Covers snapshot backups, git sync,
  cron job setup, and recovery procedures.
category: devops
trigger: >-
  User asks about backup, preservation, git sync, self-preservation, disaster
  recovery, migrating Hermes state, or checking that their work won't be lost.
  Also load when creating or fixing cron jobs that archive system state.
---

# Hermes Backup & Self-Preservation

## Architecture (General Pattern)

Hermes state lives in `~/.hermes/` and must be backed up to a GitHub repo for durability across machines and sessions. The pattern is:

```
~/.hermes/
├── skills/          ← Tactical knowledge (procedural memory)
├── memories/        ← Persistent facts (MEMORY.md, USER.md)
├── cron/jobs.json   ← Scheduled job definitions
├── config.yaml      ← Settings
├── .env             ← API keys (sanitize before backup!)
└── scripts/         ← Runnable shell scripts

→ Snapshot to: <GitHub-repo>/_<name>_backups/<DATE>/
→ Also sync skills directly to repo for live use
→ Push to GitHub on every backup
```

## What to Back Up

| Item | Path | How |
|------|------|-----|
| Skills | `~/.hermes/skills/` | `rsync -a` excluding `.archive`, `.curator_backups`, `.curator_state` |
| Memory | `~/.hermes/memories/MEMORY.md` + `USER.md` | Direct copy (NOT a dummy placeholder) |
| Config | `~/.hermes/config.yaml` | Direct copy |
| Env (sanitized) | `~/.hermes/.env` | `grep -v 'export.*KEY='` to strip secrets |
| Cron jobs | `~/.hermes/cron/jobs.json` | Direct copy to a `jobs/` subdirectory |
| Scripts | `~/.hermes/scripts/` | `rsync -a` |

## What NOT to Back Up

- API keys / credentials (sanitize .env first)
- `.archive/`, `.curator_backups/`, `.curator_state/` inside skills
- `__pycache__/`, `.venv/`, `venv/`, `*.pyc`
- Temporary files, logs, locks

## Cron Job Patterns

Two complementary cron job types work best together:

### 1. Agent-Driven Snapshot (script-based, no_agent=false)
- Runs a Bash script that snapshots state to a dated directory
- Adds a MANIFEST.md listing what was backed up
- Commits and pushes the repo
- The cron prompt tells the agent to report results succinctly
- Scheduled once daily (e.g., 14:00)

### 2. Native Repo Sync (no_agent=true, watchdog mode)
- Runs the repo's own `scripts/sync.sh` (`git add -A && git commit && git push`)
- Catches changes that step 1 didn't touch (persona files, reports, docs)
- No LLM tokens consumed — just runs the script, delivers stdout
- Scheduled after step 1 (e.g., 20:00)
- Place the wrapper script in `~/.hermes/scripts/` (cron requires scripts there)

## Snapshot Directory Structure

```
<repo>/_<name>_backups/<YYYY-MM-DD>/
├── MANIFEST.md       ← What was backed up and file counts
├── config/
│   ├── config.yaml
│   └── env.safe      ← Sanitized .env
├── jobs/
│   └── jobs.json
├── memory/
│   ├── MEMORY.md
│   └── USER.md
├── scripts/
│   └── *.sh
└── skills/
    └── <category>/
        └── <skill>/
            └── SKILL.md + references/ + scripts/ + templates/
```

## Cleanup

- Auto-prune backups older than 30 days: `find <backup-dir> -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;`
- Keep the manifest in git permanently (small overhead, useful audit trail)

## Recovery (Restore from Backup)

```bash
# 1. Clone the repo
git clone git@github.com:<user>/<repo>.git ~/ProG/<Repo>

# 2. Restore skills
rsync -a ~/ProG/<Repo>/_<name>_backups/latest/skills/ ~/.hermes/skills/

# 3. Restore memory
cp ~/ProG/<Repo>/_<name>_backups/latest/memory/*.md ~/.hermes/memories/

# 4. Restore config
cp ~/ProG/<Repo>/_<name>_backups/latest/config/config.yaml ~/.hermes/

# 5. Restore scripts
cp ~/ProG/<Repo>/_<name>_backups/latest/scripts/*.sh ~/.hermes/scripts/
```

## Common Pitfalls

- **Dummy memory placeholder**: A naive script creates `memory/atlas-memory.txt` with just a date — ALWAYS copy the real `MEMORY.md` and `USER.md` instead.
- **Wrong cron path**: Cron jobs are stored at `~/.hermes/cron/jobs.json`, NOT `~/.hermes/cron.json` (old location).
- **Wrong target repo**: Backups should go to the persona's own repo, not a separate blog/reports repo.
- **Secrets leak**: Always sanitize `.env` before backing up. Use `grep -v 'KEY='` to strip credential lines.
- **Runaway append**: Appending memory snapshots to the same file every day creates unbounded growth. Write to dated backup files instead.
- **No git push verification**: Check exit code and have fallback handling for network failures.
- **Forgotten no_agent cron**: A native `git add -A && git push` job that needs no LLM reasoning should use `no_agent=true` to save tokens and run silently.
