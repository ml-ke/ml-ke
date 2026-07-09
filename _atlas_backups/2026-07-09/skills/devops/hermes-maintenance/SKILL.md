---
title: Hermes Agent Maintenance
name: hermes-maintenance
description: Central system-administration skill for the Hermes agent environment. Covers Hermes state backup and recovery, sudo privilege escalation setup, and general Hermes housekeeping. Think of this as "how to keep the Hermes agent container healthy."
triggers:
  - User asks about backup, preservation, git sync, disaster recovery, migrating Hermes state
  - Task requires sudo or root access (apt install, dpkg, systemctl, editing /etc/)
  - Setting up passwordless sudo or SUDO_PASSWORD in .env
  - Creating or fixing cron jobs that archive system state
  - GitGuardian/detect-secrets alerts fire
---

# Hermes Agent Maintenance

This umbrella skill covers two critical administration domains for the
Hermes agent: **Backup & Self-Preservation** (keeping agent state durable)
and **Sudo Privilege Setup** (making sure the terminal tool can run
privileged commands).

---

## Section 1: Backup & Self-Preservation

### Secret Hygiene Philosophy

**GitHub gets ZERO secrets.** The backup system uses a layered defense:

1. **Whitelist-only .env backup** — Only pre-approved safe vars (TERMINAL_TIMEOUT,
   BROWSERBASE_PROXIES, etc.) are copied. Any var containing TOKEN, KEY, SECRET,
   PASSWORD, AUTH, CREDENTIAL, BEARER, JWT, or SIGNING is automatically excluded.

2. **Config credential stripping** — config.yaml is `sed`-filtered to blank out
   api_key, session_key, access_token_env, secret, password, and bearer_token
   fields before backup.

3. **Encrypted local secrets archive** — GPG-encrypted copies of the full .env,
   config.yaml, and auth.json are saved to a `_secrets/` directory that is NEVER
   pushed to GitHub.

4. **Pre-commit secret scan** — Before any `git push`, a regex-based scan checks
   for API keys (sk-*, ghp_*, AKIA*, Telegram tokens, JWT tokens, Bearer tokens,
   service account keys). If secrets are found, the push is aborted.

5. **No appended memory growth** — memory.md is overwritten (never appended) to
   prevent unbounded growth and accidental inclusion of stale secrets.

### What to Back Up

| Item | Path | Method | Secret-safe? |
|------|------|--------|-------------|
| Skills | `~/.hermes/skills/` | `rsync -a` excluding `.archive`, `.curator_backups`, `.curator_state`, `.usage.json`, `.bundled_manifest` | Scanned pre-push |
| Memory | `~/.hermes/memories/MEMORY.md` + `USER.md` | Direct copy | Yes (no secrets stored here) |
| Config (safe) | `~/.hermes/config.yaml` | Sanitized — credential fields blanked | **Yes — keys stripped** |
| Env (safe) | `~/.hermes/.env` | Whitelist-only — {TOKEN,KEY,SECRET,PASSWORD,etc.} excluded | **Yes — secrets filtered** |
| Cron jobs | `~/.hermes/cron/jobs.json` | Direct copy to `jobs/` subdirectory | Yes (no secrets stored here) |
| Scripts | `~/.hermes/scripts/` | `rsync -a` | Scanned pre-push |

### What NOT to Back Up to GitHub

| Item | Why | Alternative |
|------|-----|-------------|
| .env full | Contains API keys, tokens | GPG-encrypted to `_secrets/` |
| config.yaml raw | May contain credentials | Sanitized version goes to GitHub |
| auth.json | Cached auth tokens | GPG-encrypted to `_secrets/` |
| skills/.archive/ | Deleted/archived cruft | Excluded from rsync |
| skills/.curator_backups/ | Internal state | Excluded from rsync |
| `.usage.json` | Hermes internal metadata | Excluded from rsync |
| `__pycache__/`, `.venv/`, `venv/` | Build artifacts | Ignored by git |
| *.log, *.lock, *.pid | Runtime garbage | Ignored by git |

### Architecture (General Pattern)

```
~/.hermes/
├── skills/          ← Tactical knowledge (procedural memory)
├── memories/        ← Persistent facts (MEMORY.md, USER.md)
├── cron/jobs.json   ← Scheduled job definitions
├── config.yaml      ← Settings (credentials stripped before backup)
├── .env             ← API keys (NEVER backed up as-is to GitHub!)
└── scripts/         ← Runnable shell scripts

→ GitHub snapshot: <Repo>/_<name>_backups/<DATE>/   (all secrets stripped)
→ Local encrypted:  <Repo>/_secrets/<DATE>.gpg        (full credentials)
→ Also sync skills directly to repo for live use
→ Push to GitHub on every backup (pre-commit scan gates it)
```

### Cron Job Patterns

Two complementary cron jobs work best together:

**1. Agent-Driven Snapshot (script-based, no_agent=false)**
- Runs at e.g. 14:00 daily
- `bash ~/.hermes/scripts/atlas-sync.sh` — does ALL the work
- Reports results to the user via normal agent output

**2. Native Repo Sync (no_agent=true, watchdog mode)**
- Runs at e.g. 20:00 daily
- `bash ~/.hermes/scripts/atlas-repo-sync.sh` — `git add -A && commit && push`
- Catches non-Hermes changes (persona.md, reports, etc.)

### Secret Scan Patterns (pre-commit gate)

| Pattern | What it catches |
|---------|----------------|
| `sk-[A-Za-z0-9]{20,}` | OpenAI API keys |
| `sk-ant-[A-Za-z0-9]{20,}` | Anthropic API keys |
| `ghp_, gho_, ghu_, ghs_, ghr_` | GitHub tokens |
| `AKIA[A-Z0-9]{16}` | AWS access keys |
| `[0-9]{8,10}:[A-Za-z0-9_-]{35,}` | Telegram bot tokens |
| `eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}` | JWT tokens |
| `Bearer\s+[A-Za-z0-9]{20,}` | Bearer auth tokens |
| `type.*service_acc...` | Google service account keys |

If any match, the push is aborted immediately and the user is shown the offending files.

### Snapshot Directory Structure

```
<repo>/_<name>_backups/<YYYY-MM-DD>/
├── MANIFEST.md       ← What was backed up and file counts
├── config/
│   ├── config.yaml   ← Sanitized (credentials blanked)
│   └── env.safe      ← Whitelist-only safe vars
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

### Encrypted Secrets Archive (Local Only)

```
<repo>/_secrets/
├── env-<YYYY-MM-DD>.gpg       ← Full .env (encrypted)
├── config-<YYYY-MM-DD>.gpg    ← Full config.yaml (encrypted)
├── auth-<YYYY-MM-DD>.gpg      ← Full auth.json (encrypted)
└── _(gitignored — never pushed)_
```

To decrypt:
```bash
gpg --decrypt _secrets/env-<DATE>.gpg > ~/.hermes/.env
gpg --decrypt _secrets/config-<DATE>.gpg > ~/.hermes/config.yaml
```

To create a GPG key for encryption:
```bash
cat > /tmp/gpg-batch << 'EOF'
%echo Generating backup encryption key
Key-Type: RSA
Key-Length: 4096
Key-Usage: encrypt
Name-Real: Hermes Backup
Name-Email: hermes-backup@local
Expire-Date: 0
%no-protection
%commit
EOF
gpg --batch --gen-key /tmp/gpg-batch
```

### Cleanup

- **Backup snapshots**: Auto-prune older than 30 days: `find _hermes_backups -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;`
- **Encrypted secrets**: Auto-prune older than 90 days
- **Keep manifest** in git permanently (small overhead, useful audit trail)

### Recovery (Restore from Backup)

```bash
# 1. Clone the repo
git clone git@github.com:<user>/<repo>.git ~/ProG/<Repo>

# 2. Restore public state
rsync -a ~/ProG/<Repo>/_<name>_backups/latest/skills/ ~/.hermes/skills/
cp ~/ProG/<Repo>/_<name>_backups/latest/memory/*.md ~/.hermes/memories/
cp ~/ProG/<Repo>/_<name>_backups/latest/config/config.yaml ~/.hermes/
cp ~/ProG/<Repo>/_<name>_backups/latest/scripts/*.sh ~/.hermes/scripts/

# 3. Restore secrets (from local encrypted archive)
gpg --decrypt ~/ProG/<Repo>/_secrets/env-latest.gpg > ~/.hermes/.env
gpg --decrypt ~/ProG/<Repo>/_secrets/config-latest.gpg >> ~/.hermes/config.yaml
```

### Common Backup Pitfalls

- **Broken env sanitizer**: `grep -v 'export.*KEY='` only catches lines starting with `export` that contain `KEY=`. Real `.env` files often use `TOKEN`, `SECRET`, `PASSWORD` without `export` prefix. Always use a **whitelist** approach instead.
- **Dummy memory placeholder**: A naive script creates a placeholder file with just a date — ALWAYS copy the real `MEMORY.md` and `USER.md` instead.
- **Wrong cron path**: Cron jobs are stored at `~/.hermes/cron/jobs.json`, NOT `~/.hermes/cron.json`.
- **Runaway append**: Appending memory snapshots to the same file every day creates unbounded growth. **Overwrite** (never append) memory snapshots.
- **GPG key missing**: Without a GPG key, encrypted secrets backup is skipped. Create one with the batch method above.
- **False positives in secret scan**: The scanner may flag its own regex patterns in documentation files. The exclusion list in the script should cover its own scan patterns, historical backup snapshots, and this skill's documentation.
- **No git push verification**: Check exit code and have fallback handling for network failures.
- **Forgotten no_agent cron**: A native `git add -A && git push` job that needs no LLM reasoning should use `no_agent=true` to save tokens and run silently.

### What to Do After a GitGuardian Alert

1. **Don't panic** — The pre-commit scan prevents most leaks.
2. **Identify the leaked secret** — Check the GitGuardian email for the file + line.
3. **Rotate the credential** — Generate a new token/key immediately.
4. **Fix the root cause** — Either add the var to exclusion list, add the file to `.gitignore`, or strip the secret from the source skill/reference file.
5. **Check if it was pushed** — `git log --all -p | grep <secret-prefix>`
6. **Rewrite history if already pushed** — Use `git-filter-repo` to remove the file from all commits, then force-push. See `references/git-history-cleanup.md` for a complete recipe.

---

## Section 2: Sudo Privilege Setup

### The Mechanism

Hermes does **not** support interactive sudo password prompts. The terminal
tool runs non-interactively — there is no TTY to read a password from, and
`sudo -S` with piped stdin is blocked as a brute-force attack vector.

The solution is the **`SUDO_PASSWORD`** variable in `~/.hermes/.env`:

```bash
# ~/.hermes/.env
SUDO_PASSWORD=your_sudo_password_here
```

When `SUDO_PASSWORD` is set, the terminal tool reads it directly from the
environment and passes it to `sudo -S` internally. Every `sudo ...` command
then just works.

### Workflow

**1. Check if sudo access is available:**
```bash
sudo -n true 2>&1
# exit code 0  → passwordless sudo (NOPASSWD in sudoers)
# exit code 1  → needs password
```

**2. Check if SUDO_PASSWORD is already configured:**
```bash
grep "^SUDO_PASSWORD=" ~/.hermes/.env
```

If present and non-empty, sudo commands will work. If absent, configure it.

**3A. Option A: Set SUDO_PASSWORD in .env**

Ask once, explain the mechanism, do not nag:
```bash
echo 'SUDO_PASSWORD=the_password' >> ~/.hermes/.env
```
After this, all future `sudo` commands work automatically.

If the user demurs — **stop asking immediately.** Move to Option B or C.

**3B. Option B: Passwordless sudo via sudoers.d (recommended per Hermes docs)**

Use `pkexec` to write the sudoers file:
```bash
pkexec bash -c "echo 'pro-g ALL=(ALL) NOPASSWD: /usr/bin/apt, /usr/bin/apt-get, /usr/bin/dpkg, /usr/bin/systemctl' > /etc/sudoers.d/pro-g-apt"
```

Verify:
```bash
sudo -n apt-get update -qq   # should succeed silently
```

Note: `pkexec` is NOT always available (not on headless servers, minimal installs).

**3C. Option C: Paste commands for the user to run**

When the user won't provide the password and pkexec doesn't work:
```
I need sudo to proceed. Here are the exact commands — run them in a terminal:

sudo dpkg -i /tmp/package1.deb /tmp/package2.deb
sudo apt install -y <package>
```

### Sudo Pitfalls

- **Do not pipe passwords with `sudo -S`** — the terminal tool blocks this. Only the `.env` mechanism works.
- **Do not ask more than once.** If the user deflects, move to Option B or C immediately.
- **`pkexec` may work where `sudo` is blocked** on Ubuntu Desktop. Try it first.
- **`pkexec` is NOT available on headless/minimal installs.** Fall back to Option C.
- **Do not use `su`** — it also needs a TTY password prompt and will fail.
- **`SUDO_PASSWORD` goes in `.env`, not in `config.yaml`.** The config file has no secret-variable concept.
- **Environment variables in `.env` are loaded at agent startup.** The current session may need `/reload` to see it.
- **A cosmetic Hermes bug** may report sudo as "disabled" in `hermes status` even when passwordless sudo is correctly configured.

---

## Reference Files

- `references/git-history-cleanup.md` — Step-by-step git history rewrite with git-filter-repo and BFG for purging leaked secrets.
- `references/atlas-architecture.md` — ATLAS-specific backup architecture (repo locations, script roles, history). May be specific to a past persona.
