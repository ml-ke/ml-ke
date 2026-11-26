# Git History Cleanup for Secret Leaks

## When to Use This

GitGuardian (or another scanner) flagged a secret that was already pushed to a remote repo. You need to:

1. Remove the file containing the secret from ALL commits
2. Force-push the rewritten history
3. Rotate the compromised credential (the secret is still exposed until rotated)

## Prerequisites

- `git-filter-repo` installed (via `uv venv + pip`)
- Ability to force-push to the remote repo (may need maintainer access)
- A backup clone of the repo in case something goes wrong

## Step-by-Step

### 1. Backup the repo

```bash
git clone --mirror ~/ProG/<REPO> /tmp/<repo>-backup.git
```

Keeping a backup means you can recover if the rewrite goes wrong.

### 2. Verify what leaked

```bash
# Check which commits contain the leaked file
git log --all --oneline --name-only -- '**/env.safe'

# View the actual leaked content
git show <COMMIT_HASH>:<path/to/leaked-file>
```

### 3. Install git-filter-repo

```bash
cd /tmp && uv venv --python 3.11 .filtvenv
source .filtvenv/bin/activate
uv pip install git-filter-repo
```

Keep the venv active for the next step.

### 4. Remove the specific leaked file from history

```bash
cd ~/ProG/<REPO>
git filter-repo --path <relative/path/to/leaked/file> --invert-paths --force
```

- `--path <path>` — selects files matching this path
- `--invert-paths` — inverts the selection (REMOVES matching files instead of keeping them)
- `--force` — allows running on a non-bare repository with a working tree

This rewrites ALL commits, removing only that specific file. Other files in the same directory are unaffected.

**Multiple files**: Repeat `--path` per file, or use `--paths-from-file` for files list.

### 5. What filter-repo changes

- Removes the `origin` remote (for safety)
- Rewrites all commit SHAs
- Repacks the repo and discards old objects

### 6. Verify the cleanup

```bash
# Check the file is gone
git show HEAD:<path/to/leaked/file> 2>&1 | head -1
# Should say: fatal: path '...' does not exist in 'HEAD'

# Check history still looks sane
git log --oneline -5
```

### 7. Restore origin and force-push

```bash
git remote add origin git@github.com:<user>/<repo>.git
git push origin main --force
```

`--force` is required because all commit SHAs changed.

### 8. Clean up

```bash
rm -rf /tmp/<repo>-backup.git /tmp/.filtvenv
```

## Caveats

- **Everyone who cloned the repo must re-clone** — old clones still have the secret in their history
- **If other branches exist**, force-push each one: `git push origin <branch> --force`
- **If the repo is shared** (other collaborators), coordinate with them first
- **Force-push protection**: GitHub may reject force-pushes to protected branches. You may need to temporarily disable branch protection in Settings > Branches, or use `git push origin main --force --force-with-lease` (if allowed)
- **Open PRs will break**: All open PRs referencing old commits will need to be re-created

## Pattern: Remove All Env Files from History

```bash
git filter-repo --path-glob '*.env' --path env.safe --invert-paths --force
```

## Pattern: Remove All Backups from a Specific Date

```bash
git filter-repo --path _hermes_backups/2026-06-26/ --invert-paths --force
```

## When NOT to Use filter-repo

- If the secret was only in the latest commit and hasn't been pushed, just amend the commit: `git commit --amend` or `git rebase -i`
- If the repo has many collaborators and the secret was short-lived, consider rotation-only (no history rewrite) — rewrite is disruptive
- For very large repos (100K+ commits), BFG Repo-Cleaner (Java) is faster

## BFG Repo-Cleaner Alternative

If you have Java but not Python:

```bash
# Download BFG
wget -O /tmp/bfg.jar https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Remove file by name
java -jar /tmp/bfg.jar --delete-files env.safe ~/ProG/<REPO>
cd ~/ProG/<REPO>
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push origin --force --all
```
