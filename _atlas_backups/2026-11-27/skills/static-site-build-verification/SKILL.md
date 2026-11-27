---
name: static-site-build-verification
description: Use when a built static-site page is wrong or 404s.
version: 1.0.1
author: Pro Grammar
license: MIT
metadata:
  hermes:
    category: web
    tags: [jekyll, static-site, front-matter, permalink, build-verification]
    related_skills: [diagnosing-bugs, seo-audit]
---

# Static-Site Build Verification

Static generators (Jekyll, Hugo, Eleventy, Astro) treat content errors as warnings: they log one line, publish a degraded page, and exit 0. Any conclusion drawn from "the build succeeded" is unfounded. Read the emitted file.

Load this when a published page has a missing title, wrong date, wrong position in the feed, a permalink that 404s, or vanished taxonomy — and before publishing a content change that touches front matter.

## When to Use

- A live page renders with no title, no date, no categories/tags, or an obviously wrong date.
- Something sits at the top of the feed that should not, or a series reads out of order.
- An internal link 404s, or a permalink changed shape after a content fix.
- You are about to commit a front matter change (new post, edited `date:`, new cover `alt`).
- A build or deploy reported success but the published page disagrees.

Skip it for prose-only edits that leave front matter untouched.

## Procedure

### 1. Read the live artifact before the source

```bash
curl -sI https://site/posts/slug/ | grep -iE 'HTTP/|last-modified|cache'
curl -sL https://site/posts/slug/ | grep -o '<title>[^<]*</title>'
```

Suspicious signals, in order of usefulness:

- `last-modified` equals the newest deploy — the artifact was rebuilt now, so anything odd in it is live state, not a stale cache.
- `<h1>` empty, or `og:title` falling back to the site title — the page had **no title at build time**.
- Date/categories/tags present on sibling pages but missing here — metadata was dropped wholesale, not mistyped.
- Emitted path changed shape (date-prefixed vs slug permalink) — links pointing at the other shape now 404.

Only then look at the source. If `git show origin/main:<path>` is clean and the live page is not, the defect lives in the build pipeline (parse error, plugin hook, duplicate output path); chasing the markdown will waste the session.

### 2. Reproduce the real build

Build with the production environment and flags, in a container if the host lacks the toolchain, writing to a **mounted host directory**:

```bash
rm -rf /tmp/site-out && mkdir -p /tmp/site-out
docker run --rm -v "$PWD":/src -v /tmp/site-out:/out -w /src -e JEKYLL_ENV=production ruby:3.3 \
  bash -lc 'bundle install --quiet >/dev/null 2>&1; bundle exec jekyll build -d /out 2>&1' \
  > /tmp/build.log 2>&1; echo "exit=$?"
```

A `-d` path inside the container that is not mounted vanishes with `--rm`.

### 3. Assert on the log and the artifact

```bash
grep -c 'YAML Exception' /tmp/build.log    # must be 0; the exit code will not carry this
ls /tmp/site-out/posts | grep '<slug>'     # wrong dir shape = front matter did not parse
```

For ordering, parse the rendered list rather than trusting dates in the source files — the extraction snippet is in `references/jekyll-failure-modes.md`.

### 4. Enumerate cause classes before fixing

- **Front matter parse failure** — one malformed scalar discards the entire block: title, date, taxonomy, and the permalink's shape.
- **Duplicate output path** — two sources rendering the same permalink; last writer wins, the loser is unreachable.
- **Future-dated content** — production builds skip it silently, and any cross-link to it breaks.
- **Plugin/hook mutation** — a `_plugins/*.rb` or build hook rewriting `data`.
- **Stale artifact** — the deploy published an older build; compare `last-modified` with the deploy timestamp before debugging anything else.

### 5. Fix the input, rebuild, re-read

Patch the source, rebuild, and read the same artifact again: title, date, taxonomy, emitted path, neighbours. A fix verified only against the source diff is unverified.

### 6. Protect URL identity whenever a permalink moves

A repair that restores correct front matter can change the output path (for example `/posts/2026-08-23-slug/` → `/posts/slug/`), silently 404ing every external link and anything indexed. Keep the old path alive with a stub: `templates/redirect-stub.html`, dropped in the repo root with `layout: null`, the old `permalink`, a `canonical` link and a `meta refresh` to the new URL. Verify both URLs return 200 and the stub carries the redirect.

### 7. Verify live, not just locally

- old URL 200 (stub) and new URL 200 (real page)
- tags/categories/archives list the item
- the feed page containing it shows the intended neighbours, in the intended order
- the top of the feed no longer shows the anomaly

### 8. Gate the failure class

Run the validator before any content commit, and wire it into the publish path so the cron/CI runs it too:

```bash
python3 ~/.hermes/skills/web/static-site-build-verification/scripts/check-frontmatter.py _posts .scheduled
```

It exits non-zero on unparseable YAML, a missing title, or a missing image path; missing `date`/taxonomy are warnings because the filename date usually covers them. Checking the whole corpus takes seconds — run it site-wide once you have found one broken item.

## Pitfalls

- **Quote every scalar containing `: ` or ` #`.** Unquoted, `alt: A shield: an eye` parses as a nested mapping and the whole front matter block is discarded — no build error, no failing exit code, just an untitled page dated from file mtime (checkout time on CI) that pins itself to the top of the feed forever.
- **Never conclude from the exit code or from a clean source file.** Both can be green while the published artifact is wrong; the artifact is the only witness.
- **Read the feed in both directions when a series must read in sequence.** A descending feed shows newest first, so an item sitting physically below its predecessor is later in reading order. Verify the neighbours, not the date strings.
- **Same-day items tie on date and fall back to path order.** When a sequel must sit above a same-day sibling in the feed, give it a later time-of-day instead of relying on filename order.
- **A wrong-but-live page still owns its URL and its search result.** Repairing it and abandoning the old path leaves dead inbound links; always leave the redirect stub.
- **Check the inbound links, not just the page.** An item whose permalink changed is usually linked from siblings under the old slug convention; grep the content tree for the slug and confirm each target resolves.
- **Series slot is a product decision, not just a date.** Before "fixing" placement, read the neighbours' forward links ("next in this series") and the editorial calendar: the correct slot is the one the predecessor promises.

## Support files

- `scripts/check-frontmatter.py` — YAML front matter validator (parse, required keys, offending line).
- `templates/redirect-stub.html` — permalink redirect stub.
- `references/jekyll-failure-modes.md` — Jekyll/Chirpy mechanics: date fallback, permalink derivation, duplicate slugs, build log line, reading rendered feeds.
