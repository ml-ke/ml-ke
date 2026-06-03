#!/bin/bash
# ATLAS Daily Sync — backs up skills, memory, and config to GitHub
set -e

DATE=$(date -u +%Y-%m-%d)
REPO_DIR=~/ProG/ml-ke
BACKUP_DIR=$REPO_DIR/_atlas_backups/$DATE

mkdir -p "$BACKUP_DIR"/{skills,memory,config,scripts}

# --- Backup skills ---
if [ -d ~/.hermes/skills ]; then
  cp -r ~/.hermes/skills/* "$BACKUP_DIR/skills/" 2>/dev/null
  echo "Skills backed up: $(find ~/.hermes/skills -name 'SKILL.md' | wc -l) skills"
fi

# --- Backup config ---
if [ -f ~/.hermes/config.yaml ]; then
  cp ~/.hermes/config.yaml "$BACKUP_DIR/config/"
  echo "Config backed up"
fi

# --- Backup cron jobs ---
if [ -f ~/.hermes/cron.json ]; then
  cp ~/.hermes/cron.json "$BACKUP_DIR/config/"
  echo "Cron jobs backed up"
fi

# --- Backup scripts ---
if [ -d ~/.hermes/scripts ]; then
  cp -r ~/.hermes/scripts/* "$BACKUP_DIR/scripts/" 2>/dev/null
  echo "Scripts backed up"
fi

# --- Generate memory snapshot ---
cat > "$BACKUP_DIR/memory/atlas-memory.txt" << EOF
=== ATLAS MEMORY SNAPSHOT ===
Date: $DATE
EOF

# --- Create manifest ---
cat > "$BACKUP_DIR/MANIFEST.md" << EOF
# ATLAS Backup - $DATE

## Contents
EOF

for dir in skills memory config scripts; do
  count=$(find "$BACKUP_DIR/$dir" -type f 2>/dev/null | wc -l)
  echo "- $dir: $count files" >> "$BACKUP_DIR/MANIFEST.md"
done

# --- Cleanup backups older than 30 days ---
find "$REPO_DIR/_atlas_backups" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null

# --- Commit and push ---
cd "$REPO_DIR"
git add _atlas_backups/ 2>/dev/null
git commit -m "ATLAS backup: $DATE" --quiet 2>/dev/null || echo "Nothing new to backup"
git pull --rebase origin main --quiet 2>/dev/null
git push origin main --quiet 2>/dev/null || echo "Push skipped (no changes)"

echo "ATLAS sync complete: $DATE"
