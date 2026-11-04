#!/usr/bin/env bash
# Sync a corpus of markdown pages into a GitHub repository's wiki.
#
# Copy this into the corpus repo as scripts/push_wiki.sh and set OWNER_REPO.
# Safe to re-run: it copies every source page, removes pages whose source is gone,
# commits, and pushes.
#
#   scripts/push_wiki.sh            # sync and push
#   scripts/push_wiki.sh --dry-run  # show the diff, commit nothing
#
# Prerequisite (once, in the browser): open https://github.com/<owner>/<repo>/wiki and
# click "Create the first page". GitHub does not create the wiki git repo before that,
# so every clone attempt fails with "Repository not found" until it exists.

set -euo pipefail

OWNER_REPO="${OWNER_REPO:-owner/repo}"      # <-- set me
SRC_DIR="${SRC_DIR:-wiki}"                    # directory of source pages in the main repo
HOME_SOURCE="${HOME_SOURCE:-00-Home.md}"      # copied to Home.md as the wiki landing page
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/$SRC_DIR"
WORK="$(mktemp -d)"
DRY="${1:-}"

die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"
[ -d "$SRC" ] || die "no $SRC_DIR/ directory in $REPO_ROOT"

# Prefer SSH; fall back to reusing gh's credentials over HTTPS.
if git -c credential.helper='!gh auth git-credential' clone --depth 1 \
     "https://github.com/$OWNER_REPO.wiki.git" "$WORK/wiki" 2>/dev/null; then
  :
else
  die "could not clone https://github.com/$OWNER_REPO.wiki.git
GitHub only creates a wiki repository after its first page exists.
Open https://github.com/$OWNER_REPO/wiki, click 'Create the first page', save it, then re-run this script."
fi

cd "$WORK/wiki"
git config user.name  "$(git -C "$REPO_ROOT" config user.name)"
git config user.email "$(git -C "$REPO_ROOT" config user.email)"

# 1. copy every source page (wikis are a flat namespace: no directories)
shopt -s nullglob
for f in "$SRC"/*.md; do cp "$f" "./$(basename "$f")"; done
shopt -u nullglob

# 1b. the wiki landing page must be named Home.md
[ -f "$SRC/$HOME_SOURCE" ] && cp "$SRC/$HOME_SOURCE" ./Home.md

# 2. remove pages whose source no longer exists (keep GitHub's Home stub)
for f in ./*.md; do
  b="$(basename "$f")"
  [ "$b" = "Home.md" ] && continue
  if [ ! -f "$SRC/$b" ]; then echo "removing $b (no longer in $SRC_DIR/)"; rm -f "$f"; fi
done

git add -A
git diff --cached --quiet && { echo "wiki already in sync"; exit 0; }

echo "--- staged changes ---"
git diff --cached --stat
[ "$DRY" = "--dry-run" ] && { echo "(dry run: not committing)"; exit 0; }

git commit -m "Sync wiki from $OWNER_REPO ($(date -u +%Y-%m-%d))"
git push origin master 2>/dev/null || git push origin HEAD
echo "wiki pushed: https://github.com/$OWNER_REPO/wiki"
