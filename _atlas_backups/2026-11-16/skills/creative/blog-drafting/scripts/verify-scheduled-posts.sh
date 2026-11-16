#!/usr/bin/env bash
# Pre-flight verification for scheduled blog posts before the cron runs.
# Called from the cron script or manually before a batch commit.
#
# Usage: bash scripts/verify-scheduled-posts.sh [sched_dir] [repo_root]
# Default sched_dir: .scheduled ; default repo_root: parent of this script's dir.
# NOTE: this script ships inside the SKILL directory, so its own parent is NOT the
# blog repo. Pass the repo root explicitly when running it from the skill:
#   bash ~/.hermes/skills/creative/blog-drafting/scripts/verify-scheduled-posts.sh .scheduled ~/ProG/ml-ke

set -euo pipefail

SCHED_DIR="${1:-.scheduled}"
REPO_ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

if [ ! -d "$SCHED_DIR" ]; then
  echo "✅ No scheduled directory ($SCHED_DIR) — nothing to verify."
  exit 0
fi

POSTS=$(ls "$SCHED_DIR"/*.md 2>/dev/null || true)
if [ -z "$POSTS" ]; then
  echo "✅ No scheduled posts found — nothing to verify."
  exit 0
fi

echo "=== Verifying scheduled posts in $SCHED_DIR ==="
HAD_ERRORS=false

# Check 1: No post_url tags (these crash the build with future dates)
echo ""
echo "--- Check 1: post_url tags ---"
FOUND_POST_URL=$(grep -rn "post_url" "$SCHED_DIR"/ 2>/dev/null || true)
if [ -n "$FOUND_POST_URL" ]; then
  echo "❌ CRITICAL: post_url tags found (build will crash):"
  echo "$FOUND_POST_URL"
  HAD_ERRORS=true
else
  echo "✅ No post_url tags found"
fi

# Check 2: Cover WebP exists for every post
echo ""
echo "--- Check 2: Cover WebP images ---"
for POST in $POSTS; do
  SLUG=$(basename "$POST" | sed 's/^[0-9-]*-//; s/\.md$//')
  if [ -f "assets/img/cover-${SLUG}.webp" ]; then
    echo "✅ cover-${SLUG}.webp exists"
  elif [ -f "assets/blog/cover-${SLUG}.svg" ]; then
    echo "⚠️  cover-${SLUG}.webp missing — SVG found at assets/blog/cover-${SLUG}.svg, generate before publish"
  else
    echo "⚠️  No cover image found for: $SLUG (neither .webp nor .svg)"
  fi
done

# Check 3: Image path in front matter uses .webp, not .svg/.png
echo ""
echo "--- Check 3: Front matter image paths ---"
BAD_EXT=$(grep -rn "path:.*\\.png\\|path:.*\\.svg" "$SCHED_DIR"/ 2>/dev/null || true)
if [ -n "$BAD_EXT" ]; then
  echo "❌ Bad image paths found (must be .webp):"
  echo "$BAD_EXT"
  HAD_ERRORS=true
else
  echo "✅ All image paths use .webp"
fi

# Check 4: No cover: instead of image: path: (Chirpy won't render cover:)
echo ""
echo "--- Check 4: Front matter cover: format ---"
BAD_COVER=$(grep -rn "^cover:" "$SCHED_DIR"/ 2>/dev/null || true)
if [ -n "$BAD_COVER" ]; then
  echo "❌ CRITICAL: Posts using 'cover:' instead of 'image: path:' — covers will be MISSING:"
  echo "$BAD_COVER"
  HAD_ERRORS=true
else
  echo "✅ All posts use image: path: format"
fi

echo ""
if [ "$HAD_ERRORS" = true ]; then
  echo "❌ VERIFICATION FAILED — fix errors above before publishing"
  exit 1
else
  echo "✅ All checks passed — ready to publish"
  exit 0
fi
