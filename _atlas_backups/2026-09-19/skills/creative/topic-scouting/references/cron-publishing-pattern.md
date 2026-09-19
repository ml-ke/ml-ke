# Cron Publishing Pattern

## Directory Structure

Scheduled posts live at **`~/ProG/ml-ke/.scheduled/YYYY-MM-DD-slug.md`** (NOT `~/.hermes/blog_scheduler/`).

This is inside the blog repo so the cron can directly `mv` files into `_posts/` and `git push` without path confusion.

## The Setup (done once)

1. Write all posts to `~/ProG/ml-ke/.scheduled/YYYY-MM-DD-slug.md` (one file per day)
2. Stage cover images, WebP, and LQIP files in a single pre-cron commit
3. Do NOT commit `.scheduled/` files — they remain uncommitted for cron deployment
4. Set `deliver: origin,all` so the cron notifies both chat AND Telegram (@Pro_Grammar254)
5. Set `skills: [blog-drafting]` so the agent knows the blog conventions
6. Set `schedule: 05 11 * * *` (14:05 EAT = 11:05 UTC)

## The Cron Prompt Template

```
PUBLISH ONE BLOG POST — handles date rewriting

TODAY=$(date +%Y-%m-%d)
cd /home/pro-g/ProG/ml-ke

# Check for today's scheduled post
MATCH=$(ls .scheduled/${TODAY}-*.md 2>/dev/null | head -1)

if [ -n "$MATCH" ]; then
  mv "$MATCH" _posts/
  BASENAME=$(basename "$MATCH")
  git add "_posts/$BASENAME"
  git commit -m "Publish: $BASENAME"
  git pull --rebase origin main
  git push origin main
  echo "PUBLISHED: $BASENAME"
  exit 0
fi

# No match for today — take first remaining, rewrite its date
FIRST=$(ls .scheduled/*.md 2>/dev/null | head -1)
if [ -z "$FIRST" ]; then
  echo "All posts published! Nothing to do."
  exit 0
fi

BASENAME=$(basename "$FIRST")
OLD_DATE=$(echo "$BASENAME" | grep -oP '^\d{4}-\d{2}-\d{2}')

# Rewrite front matter date and rename file to today
sed -i "s/date: $OLD_DATE/date: $TODAY/" "$FIRST"
NEW_NAME=$(echo "$BASENAME" | sed "s/$OLD_DATE/$TODAY/")

mv "$FIRST" "_posts/$NEW_NAME"
git add "_posts/$NEW_NAME"
git commit -m "Publish (date-fixed): $NEW_NAME"
git pull --rebase origin main
git push origin main
echo "PUBLISHED (date-fixed): $NEW_NAME"
```

## What the Cron Does NOT Do

- Does NOT write content (posts are pre-drafted)
- Does NOT generate images (assets are pre-committed)
- Does NOT fact-check (done before staging)
- Does NOT ask for permission (publishes autonomously)
- On success: outputs success message (delivered to chat + Telegram)
- On git conflict: may need manual intervention

## Date Fixing

Posts in `.scheduled/` use fixed dates (e.g., 2026-06-09) but get published on whatever day the cron picks them up. The cron rewrites both:
- The front matter `date:` field
- The filename date prefix

Both are set to `$(date +%Y-%m-%d)` at publish time. The time is always `00:00:00 +0300` (midnight EAT) to prevent Jekyll from skipping the post as "future."

## Telegram Delivery

Use `deliver: origin,all` to notify both chat and Telegram. The user's Telegram handle is @Pro_Grammar254. This is the fallback channel if the user is offline when the cron fires.

## Tracking

The cron's `last_run_at` field shows when it last fired. If a day was missed (cron didn't fire for any reason), the catch-up logic publishes the post with the earliest remaining date, rewriting its date to today.
