---
name: wiki-build-with-parallel-agents
description: Use when building a wiki or large docs set with agents.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [documentation, wiki, subagents, github, verification]
    related_skills: [hermes-agent-skill-authoring, github-repo-management, subagent-driven-development]
---

# Building a large wiki or docs set with parallel agents

For producing 50-200 interlinked pages (handbook, wiki, docs site) with subagents. The approach that
works: **one authoritative manifest, one binding style guide, one verified link bank, then fan out.**

## When to Use

- Producing a multi-section handbook, wiki, course or documentation set of more than ~30 pages.
- The content must be internally consistent and interlinked (a style guide and manifest exist or can be written).
- You can verify mechanically (structure checker, diagram parser, link check) before publishing.

Do not use this for a handful of pages — write those directly. It also assumes you may fan out to
subagents; a single-author pass needs only the style guide and the verification stack.

## Why this shape

- A manifest is the only way to keep 10 agents from inventing colliding filenames, duplicated pages or dangling links.
- A style guide is what makes 135 pages written over two hours read like one book.
- A link bank built from *real search results* is the highest-value anti-hallucination control: agents
  asked to "cite sources" invent deep URLs. Extract titles+URLs from real searches into a file and forbid
  every other URL (permit documentation roots only).
- Verification must be mechanical. Agents' self-reports ("all links resolve, structure validated") were
  broadly honest but never sufficient: every batch contained defects the agent believed it had fixed.

## Procedure

1. **Recon the sources first.** Harvest real material into one digest file (head+tail truncate large files,
   keep headings intact) that writers grep by heading. Grounded pages beat generated prose.
2. **Write the manifest as data**, not prose: a Python dict of `(filename, title, level, minutes,
   prereq, coverage_bullets, diagrams, assets, reads)` per page. Generate per-group *spec files* from it —
   each writer reads only its own spec plus the shared style guide.
3. **Write STYLE.md before any content.** Page skeleton, word budget, banned words, link rules, diagram
   syntax rules, image-embedding form, and an explicit sanitiser rule. State that it is binding.
4. **Pre-create every referenced asset** before writers run, so image URLs are never dangling and no agent
   is asked to invent an image.
5. **Fan out in waves** sized to the concurrency limit, one group per section or per 3-10 pages. Put the
   shared preamble (style path, spec path, output contract, link rules, sanitiser rule, "do not run git")
   in every task's `context` — children know nothing of your conversation.
6. **Verify mechanically, fix, then publish**: `gh repo create`, commit, push, then verify against the
   *live* artefact (see the stack below).

## Verification stack (run all four)

1. A structure/links/secrets checker in the repo (`scripts/check_wiki.py`) that also runs in CI.
2. **Real mermaid parsing** — `npm i mermaid jsdom`, then `mermaid.parse()` per block. Regex checks miss
   real parse errors; the real parser is the only way to know a diagram renders. In Node, jsdom's
   `navigator` needs `Object.defineProperty(globalThis, 'navigator', {value: dom.window.navigator})`.
3. **External link check** with curl in parallel, then re-test failures sequentially: batch failures are
   usually rate-limiting (429) or timeouts, not dead links.
4. **Sanitisation scan** of the published corpus for real project identifiers, hostnames, emails, IPs and
   secret-shaped strings. Expect legitimate placeholder hits — inspect each rather than trusting a count.

Code for all four: `references/validation-toolkit.md`.

## GitHub wiki mechanics (the traps that cost time)

- **The wiki git repo does not exist until someone saves a first page in the web UI.** Check with
  `git ls-remote https://github.com/OWNER/REPO.wiki.git HEAD`. No API creates wiki pages. If the browser
  session is not logged into GitHub you cannot do it: ask the user once, with the URL in the question
  (…/wiki → "Create the first page"), then push. Do not ask for their password.
- **`git -c credential.helper='!gh auth git-credential' clone …` applies to that command only.** The later
  `git push` is then unauthenticated and dies with "could not read Username for 'https://github.com'"
  (and `die` from a shell script hides the real cause if you swallow stderr). Persist the helper inside the
  clone: `git config credential.helper '!gh auth git-credential'`.
- Wiki repos push to branch `master`; pages share a flat namespace (no directories); `Home.md` is the
  landing page and `_Sidebar.md` / `_Footer.md` are special (and excluded from the page count).
- **Images cannot be repo-relative inside a wiki.** Commit the assets to the main repo and embed the
  `raw.githubusercontent.com` URL for each one in the markdown.
- Ship a `scripts/push_wiki.sh` that clones, copies `wiki/*.md` (plus `00-Home.md` → `Home.md`), prunes
  pages whose source disappeared, commits and pushes. Make the remote overridable (`WIKI_REMOTE=`) so the
  whole script can be tested against a local bare repo before the user relies on it.

## Defects to pre-empt in generated pages

- **Never generate index/landing pages from writer-facing spec text.** "Teach the reader to…" shipped as
  user-facing copy on 16 index pages. Pull one-line descriptions from each page's own
  `## Why this matters` first sentence instead.
- **Trim to a sentence or word boundary.** Naive character caps produced `tool-specific as` mid-word.
- **Exclude fenced code blocks from prose checks** (`#` comments look like H1 headings) and from banned-word
  scans; scan the whole file for secrets regardless.
- **Stripping inline code spans with `` `[^`\n]*` `` eats the ``` fences**, making fence-count checks
  silently always pass. Use a pattern that cannot match three backticks.
- **A warning that fires on every run trains people to ignore warnings.** A PNG rebuild diff across Chrome
  versions is informational: `continue-on-error: true` plus a printed diff, not `::warning::`.
- **Reconcile cross-page command drift** after agents finish: one page used a legacy CLI form while the
  reference page used the current one. Check flags against the installed CLI's `--help`, not either page.
- Duplicate `H1` inside a code example (an `AGENTS.md` template, a skill example) is not a violation —
  scan prose only.

## Scale expectations (measured)

18 subagents in 3 waves (max 10 concurrent), ~5-8 minutes per wave, produced 135 lesson pages (~134k words,
mean 1,040 words/page), 112 mermaid diagrams, 76 infographic embeds and 24 generated PNG assets. Agents
consistently ran 10-30% over the prose budget when asked to cover a dense list and said so; budget the page
count, not just the words, and expect two or three pages to need a trim afterwards.
