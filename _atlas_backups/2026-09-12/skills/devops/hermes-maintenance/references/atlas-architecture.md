# ATLAS Backup Architecture

## Repos

| Repo | Remote | Purpose |
|------|--------|---------|
| `~/ProG/ATLAS/` | `git@github.com:BongweKE/ATLAS.git` | PRIMARY — Hermes snapshots go here under `_hermes_backups/` |
| `~/ProG/ml-ke/` | `git@github.com:ml-ke/ml-ke.git` | LEGACY — backup mirrored here under `_atlas_backups/` for historical continuity |

## Scripts

| Script | Location | Role |
|--------|----------|------|
| `atlas-sync.sh` | `~/.hermes/scripts/atlas-sync.sh` | Full Hermes state snapshot → ATLAS repo + legacy ml-ke mirror |
| `atlas-repo-sync.sh` | `~/.hermes/scripts/atlas-repo-sync.sh` | Thin wrapper that runs ATLAS repo's own `scripts/sync.sh` |
| `sync.sh` | `~/ProG/ATLAS/scripts/sync.sh` | Native `git add -A && commit && push` — catches persona/strategy/report changes |

## Cron Jobs

| Job | Schedule | Type | Description |
|-----|----------|------|-------------|
| ATLAS daily sync | `0 14 * * *` (daily 14:00 EAT) | script-based, no_agent=false | Runs atlas-sync.sh, reports backup status |
| ATLAS repo native sync | `0 20 * * *` (daily 20:00 EAT) | no_agent=true | Runs atlas-repo-sync.sh, silent unless output |

## Atlas Sync Script (atlas-sync.sh) Flow

1. Create `_hermes_backups/<YYYY-MM-DD>/` in ATLAS repo
2. Rsync skills (excluding `.archive`, `.curator_backups`, `.curator_state`)
3. Copy `config.yaml` + sanitized `.env`
4. Copy `MEMORY.md` + `USER.md` from `~/.hermes/memories/` (REAL files, not placeholder)
5. Copy `~/.hermes/cron/jobs.json`
6. Rsync `~/.hermes/scripts/`
7. Sync newer skills back to `~/ProG/ATLAS/skills/` (bidirectional freshness)
8. Write MANIFEST.md
9. Cleanup backups > 30 days
10. `git add _hermes_backups/ skills/ memory.md && commit && push`
11. Mirror to `~/ProG/ml-ke/_atlas_backups/<DATE>/` and push

## History

- **Before June 27, 2026**: Script backed up to ml-ke repo only, had dummy memory placeholder, referenced non-existent `~/.hermes/cron.json`, didn't push ATLAS repo
- **June 27, 2026**: Fixed — all issues resolved, two-tier cron architecture deployed, verified working
