# Cron Publishing Pattern

## The Setup (done once)

1. Write all posts to `~/.hermes/blog_scheduler/` — each file is `YYYY-MM-DD-slug.md`
2. Set `deliver: origin,all` so the cron notifies both chat AND Telegram (@Pro_Grammar254)
3. Set `skills: [blog-drafting]` so the agent knows the blog conventions
4. Set `schedule: 05 11 * * *` (14:05 EAT = 11:05 UTC)

## The Cron Prompt Template

```
PUBLISH ONE BLOG POST PER DAY

Today is $(date +%Y-%m-%d). The time is 14:05 EAT.

STEPS:
1. Check if .scheduled/ has a file matching today's date
2. If YES: mv to _posts/, git add, git commit -m "Publish: [title]", git pull --rebase, git push
3. If NO: publish the earliest remaining file in .scheduled/
4. If nothing remains: output "All posts published!" and exit

The post files, cover images, WebP, and LQIP are already fully prepared.
No content writing, image generation, or fact-checking needed — just git operations.
```

## What the Cron Does NOT Do

- Does NOT write content (posts are pre-drafted)
- Does NOT generate images (assets are pre-committed)
- Does NOT fact-check (done before staging)
- Does NOT ask for permission (publishes autonomously)
- On success: outputs success message (delivered to chat + Telegram)
- On git conflict: may need manual intervention

## Tracking

The cron's `last_run_at` field shows when it last fired. If a day was missed (user not online, cron did fire but unnoticed), the catch-up logic (step 3) publishes the post with the earliest remaining date.
