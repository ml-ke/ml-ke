#!/bin/bash
# ATLAS Daily Sync — backs up Hermes state to GitHub (NO secrets!)
#
# SECURITY: Never pushes secrets to GitHub.
# - .env: Only whitelisted safe vars backed up (no API keys/tokens)
# - config.yaml: All credential fields stripped
# - Skills: Pre-commit scan catches residual secrets
# - Secrets: Encrypted with GPG to local archive (NOT pushed)
set -euo pipefail

DATE=$(date -u +%Y-%m-%d)
HERMES_HOME=~/.hermes
ATLAS_DIR=~/ProG/ATLAS
BACKUP_DIR=$ATLAS_DIR/_hermes_backups/$DATE
SECRETS_DIR=$ATLAS_DIR/_secrets
SKIP_GIT_PUSH=false
SECRETS_BACKUP_SUCCESS=false

mkdir -p "$BACKUP_DIR"/{skills,memory,config,scripts,jobs}
mkdir -p "$SECRETS_DIR"

# ============================================================================
# HELPER: Safe env backup — only whitelisted non-secret vars
# ============================================================================
backup_env_safe() {
  local src="$HERMES_HOME/.env"
  local dst="$1"

  # Whitelist of safe env vars (configuration only, no secrets)
  local safe_vars=(
    TERMINAL_MODAL_IMAGE
    TERMINAL_TIMEOUT
    TERMINAL_LIFETIME_SECONDS
    BROWSERBASE_PROXIES
    BROWSERBASE_ADVANCED_STEALTH
    BROWSER_SESSION_TIMEOUT
    BROWSER_INACTIVITY_TIMEOUT
    GATEWAY_ALLOW_ALL_USERS
    WEB_TOOLS_DEBUG
    VISION_TOOLS_DEBUG
    MOA_TOOLS_DEBUG
    IMAGE_TOOLS_DEBUG
    AGENT_BROWSER_EXECUTABLE_PATH
    TELEGRAM_HOME_CHANNEL
    TELEGRAM_HOME_CHANNEL_THREAD_ID
  )

  # Patterns that indicate a secret must never be backed up
  local secret_patterns=(
    'TOKEN'
    'KEY'
    'SECRET'
    'PASSWORD'
    'PASS'
    'AUTH'
    'CREDENTIAL'
    'API_KEY'
    'API_SECRET'
    'ACCESS_KEY'
    'SECRET_KEY'
    'BEARER'
    'JWT'
    'SIGNING'
  )

  local exclude_pattern
  exclude_pattern=$(IFS='|'; echo "${secret_patterns[*]}")

  {
    echo "# Hermes safe env vars (auto-generated $(date -u '+%Y-%m-%d %H:%M UTC'))"
    echo "# WARNING: Only non-secret configuration vars are included"
    echo "# Secrets must be restored from encrypted local archive"
    echo ""

    # Copy whitelisted safe vars (only actual values)
    for var in "${safe_vars[@]}"; do
      local val
      val=$(grep -E "^${var}=" "$src" 2>/dev/null | head -1 | sed 's/^[^=]*=//')
      if [ -n "$val" ]; then
        echo "$var=$val"
      fi
    done

    # Copy comments (non-secret documentation)
    grep '^#' "$src" 2>/dev/null

  } > "$dst"

  # Also append safe non-secret lines (non-export, non-credential)
  grep -v '^#' "$src" 2>/dev/null | grep -v '^\s*$' | grep -viE "$exclude_pattern" >> "$dst" 2>/dev/null || true

  local line_count
  line_count=$(wc -l < "$dst")
  echo "Safe env backed up ($line_count lines, secrets excluded)"
}

# ============================================================================
# HELPER: Strip credentials from config.yaml
# ============================================================================
sanitize_config() {
  local src="$1"
  local dst="$2"

  sed \
    -e 's/^\([[:space:]]*api_key:\) .*$/\1 ''''''/' \
    -e 's/^\([[:space:]]*session_key:\) .*$/\1 ''''''/' \
    -e 's/^\([[:space:]]*access_token_env:\).*$/\1 ''''''/' \
    -e 's/^\([[:space:]]*secret:\).*$/\1 ''''''/' \
    -e 's/^\([[:space:]]*password:\).*$/\1 ''''''/' \
    -e 's/^\([[:space:]]*bearer_token:\).*$/\1 ''''''/' \
    "$src" > "$dst"
  echo "Config backed up (credentials stripped)"
}

# ============================================================================
# HELPER: Pre-commit secret scan
# ============================================================================
scan_for_secrets() {
  local scan_dir="$1"
  local found_any=false
  local found_files=()

  echo "  Scanning for leaked secrets..."

  local patterns=(
    'sk-[A-Za-z0-9]{20,}'
    'sk-ant-[A-Za-z0-9]{20,}'
    'ghp_[A-Za-z0-9]{36,}'
    'gho_[A-Za-z0-9]{36,}'
    'ghu_[A-Za-z0-9]{36,}'
    'ghs_[A-Za-z0-9]{36,}'
    'ghr_[A-Za-z0-9]{36,}'
    'AKIA[A-Z0-9]{16}'
    '[0-9]{8,10}:[A-Za-z0-9_-]{35,}'
    'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'
    'Bearer\s+[A-Za-z0-9]{20,}'
    'type.*service_account'
  )

  for pattern in "${patterns[@]}"; do
    while IFS= read -r line; do
      found_files+=("$line")
      found_any=true
    done < <(grep -rnP "$pattern" "$scan_dir" --include='*.md' --include='*.json' --include='*.yaml' --include='*.yml' --include='*.py' --include='*.js' --include='*.sh' --include='*.txt' 2>/dev/null | grep -v 'xxx\|\.\.\.\|YOUR\|your\|<your\|sk-\.\.\.\|ghp_\.\.\.\|example\|placeholder\|test\|__pycache__\|node_modules\|atlas-sync\.sh\|_hermes_backups\|hermes-backup/SKILL\.md' || true)
  done

  if [ "$found_any" = true ]; then
    echo ""
    echo "  *** SECRETS DETECTED! Push ABORTED ***"
    echo "  Files with potential secrets:"
    printf '%s\n' "${found_files[@]}" | sort -u | while IFS= read -r line; do
      echo "    - $line"
    done
    echo ""
    echo "  Remove secrets from those files or add to .gitignore."
    echo "  Backup saved locally but NOT pushed."
    return 1
  fi

  echo "  * No secrets detected"
  return 0
}

# ============================================================================
# STEP 1: Backup skills
# ============================================================================
if [ -d "$HERMES_HOME/skills" ]; then
  rsync -a --exclude='.archive' --exclude='.curator_backups' --exclude='.curator_state' \
    "$HERMES_HOME/skills/" "$BACKUP_DIR/skills/" 2>/dev/null
  SKILL_COUNT=$(find "$HERMES_HOME/skills" -name 'SKILL.md' | wc -l)
  echo "Skills backed up: $SKILL_COUNT skills"
fi

# ============================================================================
# STEP 2: Backup config (sanitized)
# ============================================================================
if [ -f "$HERMES_HOME/config.yaml" ]; then
  sanitize_config "$HERMES_HOME/config.yaml" "$BACKUP_DIR/config/config.yaml"
fi
if [ -f "$HERMES_HOME/.env" ]; then
  backup_env_safe "$BACKUP_DIR/config/env.safe"
fi

# ============================================================================
# STEP 3: Backup memories (REAL files)
# ============================================================================
for f in MEMORY.md USER.md; do
  if [ -f "$HERMES_HOME/memories/$f" ]; then
    cp "$HERMES_HOME/memories/$f" "$BACKUP_DIR/memory/"
    echo "Memory backed up: $f"
  fi
done

# ============================================================================
# STEP 4: Backup cron jobs
# ============================================================================
if [ -f "$HERMES_HOME/cron/jobs.json" ]; then
  cp "$HERMES_HOME/cron/jobs.json" "$BACKUP_DIR/jobs/"
  echo "Cron jobs backed up"
fi

# ============================================================================
# STEP 5: Backup scripts
# ============================================================================
if [ -d "$HERMES_HOME/scripts" ]; then
  rsync -a "$HERMES_HOME/scripts/" "$BACKUP_DIR/scripts/" 2>/dev/null
  echo "Scripts backed up"
fi

# ============================================================================
# STEP 6: Encrypted secrets backup (LOCAL ONLY — never pushed)
# ============================================================================
if command -v gpg &>/dev/null; then
  GPG_KEY_ID=$(gpg --list-keys --keyid-format LONG 2>/dev/null | grep '^   ' | head -1 | awk '{print $1}' || true)

  if [ -n "$GPG_KEY_ID" ]; then
    if [ -f "$HERMES_HOME/.env" ]; then
      gpg --yes --quiet --recipient "$GPG_KEY_ID" --encrypt \
        --output "$SECRETS_DIR/env-$DATE.gpg" "$HERMES_HOME/.env" 2>/dev/null
      echo "Secrets: .env encrypted (local only, not pushed)"
    fi
    if [ -f "$HERMES_HOME/config.yaml" ]; then
      gpg --yes --quiet --recipient "$GPG_KEY_ID" --encrypt \
        --output "$SECRETS_DIR/config-$DATE.gpg" "$HERMES_HOME/config.yaml" 2>/dev/null
      echo "Secrets: config.yaml encrypted (local only, not pushed)"
    fi
    if [ -f "$HERMES_HOME/auth.json" ]; then
      gpg --yes --quiet --recipient "$GPG_KEY_ID" --encrypt \
        --output "$SECRETS_DIR/auth-$DATE.gpg" "$HERMES_HOME/auth.json" 2>/dev/null
      echo "Secrets: auth.json encrypted (local only, not pushed)"
    fi
    SECRETS_BACKUP_SUCCESS=true

    # Ensure _secrets/ is gitignored
    if [ -f "$ATLAS_DIR/.gitignore" ]; then
      if ! grep -q '_secrets/' "$ATLAS_DIR/.gitignore" 2>/dev/null; then
        echo '_secrets/' >> "$ATLAS_DIR/.gitignore"
        echo "  Added _secrets/ to .gitignore"
      fi
    fi

    # Cleanup old encrypted secrets (keep 90 days)
    find "$SECRETS_DIR" -name '*.gpg' -mtime +90 -delete 2>/dev/null || true
  else
    echo "WARNING: No GPG key found. Secrets not encrypted."
    echo "  Create one with: gpg --full-generate-key"
  fi
else
  echo "WARNING: gpg not installed. Secrets not encrypted."
fi

# ============================================================================
# STEP 7: Sync Hermes skills to ATLAS repo (with secret scan)
# ============================================================================
if [ -d "$HERMES_HOME/skills" ]; then
  if scan_for_secrets "$HERMES_HOME/skills"; then
    rsync -a --exclude='.archive' --exclude='.curator_backups' --exclude='.curator_state' \
      --exclude='.usage.json' --exclude='.usage.json.lock' --exclude='.bundled_manifest' \
      --update "$HERMES_HOME/skills/" "$ATLAS_DIR/skills/" 2>/dev/null
    echo "ATLAS skills synced from Hermes"

    # Double-check ATLAS skills after sync
    if ! scan_for_secrets "$ATLAS_DIR/skills" 2>/dev/null; then
      echo "  *** Reverting skills sync due to secrets in ATLAS skills ***"
      cd "$ATLAS_DIR" && git checkout -- skills/ 2>/dev/null || true
      SKIP_GIT_PUSH=true
    fi
  else
    echo "  *** Skipping skills sync due to secrets in Hermes skills ***"
    SKIP_GIT_PUSH=true
  fi
fi

# ============================================================================
# STEP 8: Update ATLAS memory.md (overwrite, never append)
# ============================================================================
if [ -f "$HERMES_HOME/memories/MEMORY.md" ]; then
  TIMESTAMP=$(date -u "+%Y-%m-%d %H:%M UTC")
  {
    echo "# ATLAS Memory"
    echo ""
    echo "## Hermes Memory Snapshot ($TIMESTAMP)"
    echo ""
    cat "$HERMES_HOME/memories/MEMORY.md"
  } > "$ATLAS_DIR/memory.md"
  echo "ATLAS memory.md updated with latest Hermes snapshot"
fi

# ============================================================================
# STEP 9: Create manifest
# ============================================================================
cat > "$BACKUP_DIR/MANIFEST.md" << EOF
# ATLAS Hermes Backup - $DATE

## Contents
EOF

for dir in skills memory config scripts jobs; do
  count=$(find "$BACKUP_DIR/$dir" -type f 2>/dev/null | wc -l)
  echo "- $dir: $count files" >> "$BACKUP_DIR/MANIFEST.md"
done

if [ "$SECRETS_BACKUP_SUCCESS" = true ]; then
  echo "- secrets: Encrypted with GPG (local only, not pushed)" >> "$BACKUP_DIR/MANIFEST.md"
fi

# ============================================================================
# STEP 10: Cleanup backups older than 30 days
# ============================================================================
find "$ATLAS_DIR/_hermes_backups" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null

# ============================================================================
# STEP 11: Commit and push (only if no secrets detected)
# ============================================================================
cd "$ATLAS_DIR"

if [ "$SKIP_GIT_PUSH" = true ]; then
  echo ""
  echo "*** SECURITY HOLD: Backup saved locally but NOT pushed to GitHub ***"
  echo "  Check flagged files, remove secrets, then:"
  echo "  cd ~/ProG/ATLAS && git add -A && git commit -m 'retry: $DATE' && git push"
  exit 0
fi

echo "Running pre-commit secret scan..."
if scan_for_secrets "$ATLAS_DIR" 2>/dev/null; then
  git add _hermes_backups/ skills/ memory.md 2>/dev/null

  if git diff --cached --quiet 2>/dev/null; then
    echo "No new changes to ATLAS repo"
  else
    git commit -m "ATLAS auto-sync: $DATE"
    if git push origin main 2>/dev/null; then
      echo "* Pushed to github.com:BongweKE/ATLAS.git"
    else
      echo "WARNING: Push failed (network or auth) -- changes committed locally"
    fi
  fi
else
  echo ""
  echo "*** SECURITY HOLD: Pre-commit scan found secrets. NOT pushing ***"
  echo "  Backup saved locally in $BACKUP_DIR"
  echo "  Files reverted."
  git checkout -- _hermes_backups/ skills/ memory.md 2>/dev/null || true
fi

# ============================================================================
# STEP 12: Legacy ml-ke blog mirror
# ============================================================================
MLKE_DIR=~/ProG/ml-ke
if [ -d "$MLKE_DIR" ] && [ "$SKIP_GIT_PUSH" = false ]; then
  MLKE_BACKUP=$MLKE_DIR/_atlas_backups/$DATE
  mkdir -p "$MLKE_BACKUP"
  rsync -a "$BACKUP_DIR/" "$MLKE_BACKUP/" 2>/dev/null

  cd "$MLKE_DIR"
  git add _atlas_backups/$DATE/ 2>/dev/null
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "ATLAS backup: $DATE" 2>/dev/null
    git push origin main 2>/dev/null && echo "* Pushed to ml-ke repo (legacy)" || true
  fi
fi

echo ""
echo "* ATLAS sync complete: $DATE"
echo "  GitHub: skills, memories, config (stripped), scripts, cron"
echo "  Local encrypted secrets: $SECRETS_DIR/"
