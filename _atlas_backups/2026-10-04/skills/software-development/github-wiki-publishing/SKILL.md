---
name: github-wiki-publishing
description: Use when building a multi-page wiki or docs collection.
version: 1.0.0
author: hermes-curator
license: MIT
metadata:
  hermes:
    tags: [docs, wiki, github, content-generation, subagents, mermaid, ci]
    related_skills: [subagent-driven-development, github-repo-management, architecture-diagram]
---

# Building and publishing a multi-page wiki

For a GitHub wiki, handbook or docs set of more than ~15 pages, especially a learning resource written
by several writers (subagents or humans) that must read like one book. Carries the generation pipeline,
the GitHub wiki mechanics, and the pitfalls that cost real time.

## When to use

- "Create a wiki / handbook / docs set" from repositories, notes, lessons or research.
- Turning existing repos into a structured resource for a target audience.
- Any content set written in parallel that must stay consistent and navigable.
- Publishing to a GitHub wiki, or to a `wiki/` or `docs/` folder in a repo.

## Non-negotiables

1. **The page list is data, and it is frozen before anyone writes.** Keep one source of truth (a Python
   dict / YAML / JSON) and generate everything from it: per-writer spec files, the committed
   `WIKI-MANIFEST.md`, section index pages, the home page, the sidebar. Hand-maintained indexes drift
   within one edit; generated ones cannot.
2. **One house style file, declared binding.** `STYLE.md` carries the page skeleton, the length budget,
   tone calibration with bad/good examples, mermaid safety rules, the link policy and the sanitiser
   rule. Every writer gets its absolute path and is told it is binding. Without this, parallel pages
   read like different books.
3. **Verify on disk, never from the writer's report.** A child's "wrote 9 pages, all good" is a claim.
   After each wave: list the expected filenames, diff against disk, count words, run the checker
   yourself. Require a done-file (one line per page: path + word count) so the check is scriptable.
4. **Generate assets before writers run.** Pages embed fixed asset filenames, so publish that list in
   each spec; otherwise writers invent image names that will never exist. See
   `references/html-to-png-infographics.md`.
5. **Never invent a link, version, SHA, price or quota.** Give writers a verified link bank file and
   permit documentation *roots*, not guessed deep paths. A confident deep link that 404s, or a pinned
   `uses: owner/action@<sha>` recalled from memory, is a defect that ships. Use major-version tags with
   a comment saying why, and write "as of <year>" or link the source for anything numeric.
6. **Sanitise anything harvested from private sources.** Real repos are excellent raw material and
   terrible copy. Generalise before publishing: no internal hostnames, service URLs, credentials,
   account IDs, customer data, vendor or client names, private paths. "A payments platform for small
   merchants" keeps the lesson and loses the identity.
7. **A deterministic checker gates the publish** — structure, links, assets, mermaid fences, filler
   words, secret patterns. Offline, stdlib-only, exit non-zero, run before push and in CI. Start from
   `templates/check_docs_site.py`.
8. **Publish from a prepared copy, and gate the rendered output.** Keep `wiki/` byte-identical so the
   GitHub-wiki sync keeps working, and build the docs site from a *copy*; a wiki-style link rewriter
   must key on each page's source filename (`src_uri`) and rewrite to `Page-Name.md` so the renderer
   resolves, validates and generates the URL itself. Then check the generated HTML — every local
   `href`/`src` must resolve — on top of `mkdocs build --strict`. A source-only checker cannot see a
   failed rewrite, and a green build exit proves nothing. Ship **directory URLs** so the host resolves
   `/` and `/Foo/` itself: a root path that exists only because a deploy script rewrites it is custom
   code on the critical path of your homepage, and it fails silently on whichever edge does not have
   that script. See `references/docs-site-hosting-and-domains.md`.

## Page skeleton (this user's shape for learning content)

Bite-sized and consistent beats long and inconsistent.

- **No H1** — the renderer supplies the page title; an H1 duplicates it (the landing page is the one
  exception, since it also acts as the repo README).
- Opens with a metadata blockquote:
  `> **Section 05 · Lesson 3** · Level: beginner · ~12 min · Prereq: [Git essentials](05-Git-Essentials)`.
- `## Why this matters` → body (600–1000 words of prose, hard ceiling ~1400) → `## Try it` (a runnable
  step) → `## Common mistakes` (3+ specific failures) → `## Key takeaways` (bullets) →
  `## Further learning` (verified links, one clause each).
- At least one diagram: **mermaid inline** (diffable, editable) for flows, lifecycles, schemas;
  a **PNG infographic** for layouts mermaid cannot express (ranked bars, comparisons, pyramids).
- Internal links use the page name with no extension and only point at manifest pages.
- Progression is explicit: every page names its prerequisite, every section has an index page that says
  what the reader can do by the end.

## Procedure

1. **Recon the sources.** Per repo: `AGENTS.md`, docs trees, CI/CD runbooks, `*-gotchas.md`,
   retrospectives, lesson banks. Then check the environment — `gh auth status`, `git config user.name`,
   and which asset tooling actually exists (do not assume matplotlib; see the infographics reference).
2. **Harvest into a digest file, not into context.** Concatenate the valuable docs into one scratch
   digest with per-file caps (head+tail), a banner per source, and a note that it is machine-assembled
   so writers grep headings instead of reading linearly. Parent context stays small; every writer is
   grounded in the same real material.
3. **Collect a verified link bank.** Run the searches once, save `{query: [{title, url}]}`, then emit
   `LINK-BANK.md`. Tell writers it is the only permitted source of URLs, plus named documentation roots.
4. **Write the manifest, then generate** the per-writer specs, the committed manifest, section indexes,
   the home page and `_Sidebar` / `_Footer`.
5. **Write `STYLE.md`** before dispatching anything.
6. **Generate assets and interactive tools**, and inspect two or three visually (byte size proves
   nothing about layout).
7. **Dispatch writers in waves** at the platform's parallel limit, one group per spec, splitting large
   sections into an `a`/`b` pair. Repeat the shared constraints in every task (checklist in
   `references/parallel-content-generation.md`) and forbid git commands.
8. **Verify each wave on disk**, run the checker, and fix the findings — including the checker's own
   false positives (`references/checking-generated-docs.md`).
9. **Commit and push the repo**, then sync the wiki (`references/github-wiki-mechanics.md` — it needs a
   one-time UI page before the wiki git repo exists).
10. **Report** what exists, the gate result, and any step still needing the user's hands.

## Pitfalls

- **The GitHub wiki git repo does not exist until a first page is created in the web UI.** Cloning
  `<repo>.wiki.git` before that returns `Repository not found` over both SSH and the `gh` credential
  helper, and there is no REST/GraphQL endpoint for wiki pages. Do not read that as an auth problem —
  walk the user through the one-click page creation, then push.
- **Wiki repos default to `master`** even when the main repo is on `main`. Push with a fallback
  (`git push origin master || git push origin HEAD`) rather than hardcoding one name.
- **Relative image paths do not resolve in a wiki page.** Reference assets by absolute
  `raw.githubusercontent.com/<owner>/<repo>/<branch>/assets/<file>.png` from a public repo, and keep the
  files committed so repo and docs-site renders still work.
- **A wiki is a flat namespace.** The filename becomes the URL and the title; `Home.md` is the landing
  page; `_Sidebar.md` and `_Footer.md` render on every page; showing a fenced example inside a page
  needs a 4-backtick outer fence.
- **A checker that scans raw text reports false positives** — shell `#` comments inside fenced blocks
  look like H1s, an inline `` ```mermaid `` in prose inflates fence counts, and `re.findall` with one
  capture group returns strings rather than tuples. Strip fenced blocks and inline spans before prose
  checks, keep secret scanning over the whole file, and fix the checker instead of mutilating good pages.
- **A child's summary is not evidence.** Re-derive filenames, word counts and completeness from the
  filesystem; a writer that reports 9 pages may have written 6 and summarised them.
- **Never let writers run git.** Concurrent `git add`/`commit` from parallel children races on the index
  and sweeps in half-finished pages. The parent commits once, after the wave verifies.
- **Do not commit generator output you have not just run.** Re-run the asset generator before committing
  and let CI diff the result; committed images from a now-broken generator stay invisible until someone
  edits the generator.
- **Publishing private-project lessons publicly is a leak, not a shortcut.** Run a sanitisation pass on
  the finished pages (grep for hostnames, keys, client names) before the first push, not after.
- **A link rewriter that has silently stopped matching is invisible.** When its page-name map comes back
  empty (e.g. it reads `File.name`, which dropped the `.md` extension in MkDocs 1.5+), the build still
  exits 0 and logs only an INFO line per link — so the site ships with every internal link broken. Assert
  the rewrite on a built page with `grep -o 'href="Page-Name[^"]*"' site/Page-Name.html`, and make the
  post-build link check a build step rather than something a human remembers to run.
- **Rewrite internal links to the page's source filename, never to a `.html` URL.** MkDocs resolves link
  targets against the source tree, so `.html` targets make the validator report every link as "not found
  among documentation files" (hundreds of false warnings); a `.md` target is validated properly and the
  renderer emits the real URL, so `use_directory_urls` still decides the shape.
- **Test every gate against a deliberate violation before trusting it.** An exemption regex that matched
  every page (`^\d\d-[A-Z]` against names like `01-The-Agent-Loop.md`) meant the checker reported
  "0 errors, 0 warnings" while checking nothing, and a hand-maintained "Total pages: N" claim in the
  manifest drifted from the tree. Add the negative test (break it on purpose, watch it fail) and verify
  derived counts against the tree inside the checker instead of trusting a prose header.
- **Never print an unbounded build log into context.** Capture it (`mkdocs build --strict > /tmp/build.log
  2>&1`) and reduce it to counts of distinct message shapes. MkDocs writes logs to stderr, so
  `cmd 2>&1 > file` sends stderr to the terminal and leaves the file empty.

## Support files

- `references/github-wiki-mechanics.md` — enabling, the one-time UI prerequisite, structure conventions,
  the sync pattern.
- `references/parallel-content-generation.md` — manifest shape, spec-file contract, writer task
  checklist, wave verification, how to fix a bad wave.
- `references/checking-generated-docs.md` — what the gate must cover and the false positives to design
  around.
- `references/docs-site-hosting-and-domains.md` — building the site from a copy, the wiki-link hook,
  build-time link validation, GitHub Pages enablement, and custom domains via Cloudflare (Workers
  custom domain vs Pages, route precedence, verification commands). General Cloudflare mechanics —
  token scopes, static-asset `html_handling`/`not_found_handling`, route precedence, verifying through
  the domain — live in the `cloudflare-edge-deployments` skill.
- `references/html-to-png-infographics.md` — HTML→PNG infographics with headless Chrome, plus the
  interactive single-file tool pattern and how to verify it.
- `scripts/html_to_png.sh` — one-off HTML→PNG render wrapper that fails on blank output.
- `templates/check_docs_site.py` — starting checker for a manifest + pages tree.

## Verification checklist

- [ ] Every manifest page exists; no orphans; no page outside the manifest.
- [ ] Checker exits 0 under `--strict` before commit.
- [ ] The rendered site is verified, not just built: every local link in the generated HTML resolves,
      and one real page's links were grepped in the output.
- [ ] Every gate has been shown to fail on a deliberate violation (negative test), and any count the
      checker or manifest asserts is derived from the tree.
- [ ] The published URL was opened (or fetched with `curl --resolve` when a fresh record has not
      propagated) and renders: search, nav, diagrams, and a custom 404.
- [ ] Two or three pages read end to end for voice, grounding and link validity (machine checks miss
      "technically valid but hollow").
- [ ] Assets render (visually inspected) and every embedded filename exists.
- [ ] No secret-shaped strings, no private hostnames or names, no fabricated deep links or SHAs.
- [ ] Repo pushed; wiki state reported honestly, including any step still needing the user.
