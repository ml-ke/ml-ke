# Jekyll / Chirpy failure modes

Mechanics behind the symptoms that static-site repair sessions chase. Assumes posts live in `_posts/` and the theme sets `permalink: /posts/:title/`.

## Front matter is all-or-nothing

A YAML exception anywhere in the block makes Jekyll discard the whole `data` hash. Nothing errors out; `jekyll build` exits 0.

Consequences of an empty `data`:

| Missing | Jekyll behaviour | What the page looks like |
|---|---|---|
| `title` | falls back to site title | empty `<h1>`, `og:title` = site name |
| `date` | falls back to **file mtime** (checkout time on CI) | item redated to the deploy time, pins to top of feed |
| `categories`/`tags` | none | no taxonomy links, absent from tag pages |
| permalink | derived from the file path instead of `:title` | `/posts/YYYY-MM-DD-slug/` instead of `/posts/slug/` |

That last row breaks links: every internal link written as `/posts/<slug>/` now 404s, silently.

Find it in the build log, not the source:

```bash
grep -n -A2 'YAML Exception' /tmp/build.log
# Error: YAML Exception reading /src/_posts/<file>.md: (<unknown>):
#   mapping values are not allowed in this context at line 8 column 46
```

## Reproduce locally

Host may have no Ruby. Container build, output to a mounted directory:

```bash
rm -rf /tmp/site-out && mkdir -p /tmp/site-out
cd <repo> && docker run --rm \
  -v "$PWD":/src -v /tmp/site-out:/out -w /src -e JEKYLL_ENV=production ruby:3.3 \
  bash -lc 'bundle install --quiet >/dev/null 2>&1; bundle exec jekyll build -d /out 2>&1' \
  > /tmp/build.log 2>&1; echo "exit=$?"
```

## Read the rendered order

Feed order is a property of the built HTML, not of the date strings. The home page renders 10 per page; the archives page renders every post in order.

```python
import re, datetime
h = open('/tmp/site-out/index.html').read()
seg = h.split('id="post-list"')[1].split('</main>')[0]
for a in re.findall(r'<article class="card-wrapper card">.*?</article>', seg, re.S)[:12]:
    href = re.search(r'href="([^"]+)"', a).group(1)
    t = re.search(r'class="card-title[^"]*">([^<]*)<', a)
    ts = re.search(r'data-ts="(\d+)"', a)
    when = datetime.datetime.fromtimestamp(int(ts.group(1)), datetime.UTC).date() if ts else '?'
    print(when, '|', href, '|', (t.group(1) if t else '(NO TITLE)'))
```

Same pattern against `archives/index.html` when the target is deeper than page 1; it lists every post, so three neighbours prove the slot. Pagination URLs are directory-style (`/page10/`), and `/page10` 301s.

## Other silent killers

- **Duplicate slug** — two `_posts` files with the same stem render the same destination; Jekyll logs `Conflict: The following destination is shared by multiple files` and the last writer wins. Intentional for a recurring series that keeps one permalink, fatal when it is not: the loser is unreachable.
- **Future dates** — production builds default to `future: false`; a future-dated post is skipped and nothing links to it. Use a date already in the past (midnight of the publish day), never the current time.
- **Ties on date** — items sharing a date order by path; a sequel that must sit above a same-day sibling needs a later time-of-day in `date:`.
- **Last-modified plugins** — one that shells out to `git log -1 -- <path>` happily reports a last-modified date for a page that renders no title, which makes a broken page look normal at a glance.

## Live-side probes

```bash
curl -sI https://site/posts/slug/ | grep -iE 'HTTP/|last-modified'
curl -sL https://site/posts/slug/ | grep -o '<title>[^<]*</title>'
curl -sL https://site/posts/slug/ | grep -o 'data-ts="[0-9]*"' | head -1   # posted date = mtime? compare with last-modified
```

`last-modified` on GitHub Pages is the artifact build time for every file; a page whose posted timestamp equals it was dated by mtime, i.e. its front matter did not parse.
