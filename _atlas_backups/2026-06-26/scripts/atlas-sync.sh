#!/bin/bash
# ATLAS Daily Sync — backs up Hermes skills, memory, config, and scripts
# to the ATLAS repo and pushes to github.com:BongweKE/ATLAS.git
set -e

DATE=$(date -u +%Y-%m-%d)
HERMES_HOME=~/.hermes
ATLAS_DIR=~/ProG/ATLAS
BACKUP_DIR=$ATLAS_DIR/_hermes_backups/$DATE

mkdir -p "$BACKUP_DIR"/{skills,memory,config,scripts,jobs}

# --- Backup skills ---
if [ -d "$HERMES_HOME/skills" ]; then
  rsync -a --exclude='.archive' --exclude='.curator_backups' --exclude='.curator_state' \
    "$HERMES_HOME/skills/" "$BACKUP_DIR/skills/" 2>/dev/null
  SKILL_COUNT=$(find "$HERMES_HOME/skills" -name 'SKILL.md' | wc -l)
  echo "Skills backed up: $SKILL_COUNT skills"
fi

# --- Backup config ---
if [ -f "$HERMES_HOME/config.yaml" ]; then
  cp "$HERMES_HOME/config.yaml" "$BACKUP_DIR/config/"
  echo "Config backed up"
fi
if [ -f "$HERMES_HOME/.env" ]; then
  # Strip secrets for safe backup
  grep -v '^export.*KEY=' "$HERMES_HOME/.env" > "$BACKUP_DIR/config/env.safe" 2>/dev/null || true
  echo "Config .env (sanitized) backed up"
fi

# --- Backup memories (REAL MEMORY.md and USER.md) ---
for f in MEMORY.md USER.md; do
  if [ -f "$HERMES_HOME/memories/$f" ]; then
    cp "$HERMES_HOME/memories/$f" "$BACKUP_DIR/memory/"
    echo "Memory backed up: $f"
  fi
done

# --- Backup cron jobs ---
if [ -f "$HERMES_HOME/cron/jobs.json" ]; then
  cp "$HERMES_HOME/cron/jobs.json" "$BACKUP_DIR/jobs/"
  echo "Cron jobs backed up"
fi

# --- Backup scripts ---
if [ -d "$HERMES_HOME/scripts" ]; then
  rsync -a "$HERMES_HOME/scripts/" "$BACKUP_DIR/scripts/" 2>/dev/null
  echo "Scripts backed up"
fi

# --- Sync Hermes memory into ATLAS memory.md ---
if [ -f "$HERMES_HOME/memories/MEMORY.md" ]; then
  TIMESTAMP=$(date -u "+%Y-%m-%d %H:%M UTC")
  {
    echo ""
    echo "---"
    echo "## Hermes Memory Snapshot ($TIMESTAMP)"
    echo ""
    head -200 "$HERMES_HOME/memories/MEMORY.md"
  } >> "$ATLAS_DIR/memory.md"
fi

# --- Also update ATLAS skills if Hermes skills are newer ---
# Sync skills from ~/.hermes/skills/ to ~/ProG/ATLAS/skills/ (newer files only)
if [ -d "$HERMES_HOME/skills" ]; then
  rsync -a --exclude='.archive' --exclude='.curator_backups' --exclude='.curator_state' \
    --update "$HERMES_HOME/skills/" "$ATLAS_DIR/skills/" 2>/dev/null
  echo "ATLAS skills synced from Hermes"
fi

# --- Create manifest ---
cat > "$BACKUP_DIR/MANIFEST.md" << EOF
# ATLAS Hermes Backup - $DATE

## Contents
EOF

for dir in skills memory config scripts jobs; do
  count=$(find "$BACKUP_DIR/$dir" -type f 2>/dev/null | wc -l)
  echo "- $dir: $count files" >> "$BACKUP_DIR/MANIFEST.md"
done

# --- Cleanup backups older than 30 days ---
find "$ATLAS_DIR/_hermes_backups" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null

# --- Commit and push ATLAS repo ---
cd "$ATLAS_DIR"
git add _hermes_backups/ skills/ memory.md 2>/dev/null
if git diff --cached --quiet 2>/dev/null; then
  echo "No new changes to ATLAS repo"
else
  git commit -m "ATLAS auto-sync: $DATE"
  git push origin main 2>/dev/null && echo "Pushed to github.com:BongweKE/ATLAS.git" || echo "Push failed (network or auth)"
fi

# --- Also update ml-ke blog backups (legacy) ---
MLKE_DIR=~/ProG/ml-ke
if [ -d "$MLKE_DIR" ]; then
  MLKE_BACKUP=$MLKE_DIR/_atlas_backups/$DATE
  mkdir -p "$MLKE_BACKUP"
  cp -r "$BACKUP_DIR"/* "$MLKE_BACKUP/" 2>/dev/null
  cd "$MLKE_DIR"
  git add _atlas_backups/$DATE/ 2>/dev/null
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "ATLAS backup: $DATE" 2>/dev/null
    git push origin main 2>/dev/null && echo "Pushed to ml-ke repo (legacy)"
  fi
fi

echo "ATLAS sync complete: $DATE — backed up to BongweKE/ATLAS.git"
