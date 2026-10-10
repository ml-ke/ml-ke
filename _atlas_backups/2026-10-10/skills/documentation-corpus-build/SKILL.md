---
name: documentation-corpus-build
description: "Use when building a multi-page docs corpus or wiki."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Documentation, Wiki, GitHub, Subagents, Verification, Publishing]
    related_skills: [writing-router, humanizer, subagent-driven-development, grounded-citations, github-repo-management]
---

# Documentation corpus build

Producing 20–200 pages of publishable prose — a wiki, handbook, curriculum, reference site — from a mix of private source trees and public research. The work is roughly 30% writing and 70% scaffolding, verification and publication mechanics. Do the scaffolding first: it is what makes fifteen parallel writers produce one coherent book instead of fifteen essays.

For delegating code implementation with review stages, use `subagent-driven-development` instead; this skill is about content corpora.

## When to Use

Load this when the deliverable is a body of pages rather than a single document:

- A GitHub repo wiki, documentation site, handbook, or onboarding guide.
- A curriculum or learning path with numbered sections, exercises and a capstone.
- A reference manual, runbook collection, or glossary set.
- Any request phrased as "make it comprehensive / bite-sized / well structured", with diagrams and further-reading links.
- Turning private-project lessons into public teaching material — step 4 is the one that matters there.

Not for: a single post or one-off document (`blog-drafting`, `docx`, `pdf`, `powerpoint`), or code implementation delegation (`subagent-driven-development`).

## 1. Recon: harvest to disk, never into context

- List the source trees, then extract headings only: `grep -n '^#' <file> | head -80`. Your context is the scarcest resource here; a 200KB corpus read directly will crowd out the work.
- Assemble ONE harvest file on disk: per-source heading blocks, each file head+tail truncated (~14KB cap) with an explicit middle-truncation marker. Writers read it from disk by path.
- Record every web search result (query → title, URL) into a link bank file. **Only that bank plus named documentation roots may be cited by anyone downstream.** A confident invented deep link that 404s is the most common defect in generated documentation.
- If the web-extract backend is search-only, fetch pages yourself with `curl -A "<browser UA>" -L` and convert with `pandoc -f html -t gfm --wrap=none`, then strip inline tags and nav-only lines before saving.
- Confirm tooling before you design around it: image renderer available, mermaid validator installable, `gh auth status`, and whether the target repo name is free (`gh api repos/<owner>/<name>` returning 404 means free).

## 2. Manifest first; generate every navigation artefact from it

- Keep the page list as DATA (dict/YAML): section → pages, each with filename, title, level, minutes, prerequisite, coverage bullets, required diagrams, image assets, source reads.
- Generate from that single structure: the per-writer spec files, the committed manifest, the section index pages, the home page, and the sidebar. Hand-maintained navigation drifts within a day and silently breaks links.
- Naming: `<section>-<lesson>-<Title>` per page, `<section>-<Title>` for a section index. Flat filenames — wikis have no hierarchy.
- Order sections the way a beginner should meet them, and put the exit ramp (capstones, exercises, learning path) in the plan from the start rather than bolting it on.
- Fixed asset filenames belong in the manifest *before* writers start, so pages reference images that will exist.

## 3. STYLE.md is the contract — write it before dispatching anyone

Every writer receives the same binding style file, and it must contain:

- The reader personas (two named archetypes with different starting knowledge) so writers calibrate depth.
- The page skeleton, verbatim: metadata line → "why this matters" → 2–5 body sections → a runnable "try it" → a specific "common mistakes" list → "key takeaways" → "further learning".
- A length budget (600–1000 prose words with a hard ceiling around 1300) and a forbidden-filler word list.
- The verified-links-only rule, the mermaid syntax limits, the exact image-embedding form, and the sanitisation section from step 4.
- Bite-sized wins: one idea per page, ~10 minutes of reading, every page cross-linked onward. Say plainly that depth belongs in the next page, not in more words here.

## 4. Sanitisation — non-negotiable when the sources are private

Public deliverables built from private repos leak by default: internal hostnames, service URLs, cloud project/service IDs, key names, customer or merchant data, vendor pricing, client and organisation names, private repo paths.

- State the rule in STYLE.md *and* repeat it in every writer's task context, listing preferred generalisations ("a Kenyan payments platform", "a multi-agent RAG research assistant", "a geospatial compliance platform").
- The lesson survives, the identity does not: keep the failure and the fix, drop the identifiers.
- Before publishing, grep the corpus for the real identifier list plus credential shapes (`sk-`, `ghp_`, `AKIA`, `BEGIN … PRIVATE KEY`, JWTs). Review hits rather than failing blindly — a case-insensitive word match on a project name will hit ordinary prose, and a genuine credential hit is always a defect.
- Never write anything resembling a real credential, including in examples: use `os.environ["SERVICE_API_KEY"]` or `sk-...REDACTED`.
- Case studies built from private docs are the highest-risk pages. Give them their own instruction block naming both what to generalise and what to preserve.

## 5. Assets: generate them, then look at them

Use `references/asset-generation.md` for the HTML→PNG route (works when matplotlib/PIL/graphviz are absent), the CSS skeleton and the formatting pitfall. Two rules regardless of renderer:

- Vision-check two or three renders before a hundred pages depend on them. Clipping, overflow and unreadable contrast are invisible in command output.
- Infographics live in the main repo and are embedded by absolute raw URL. Wiki pages cannot resolve repo-relative image paths.

## 6. Writer waves

- One writer per spec group, up to ten concurrent. Each gets: style-file path, its spec path, the reads list, the sanitisation rule, an output contract naming only its own filenames, and "do not run git".
- Require a done-file per group (`path<TAB>wordcount`). A returned summary is a claim, not evidence — verify filenames and word counts on disk.
- Results land only after you end your turn, so finish every independent task first (assets, tooling, repo creation, checkers), then dispatch, then stop.
- Give writers the full page manifest so cross-links point at real pages; the gate will catch the rest.
- Tell writers not to invent version numbers, prices, quotas or commit SHAs. Pin an action by its major tag and say a real pipeline should pin a verified SHA instead of fabricating one.

## 7. Gates before publication

Write one checker, run it until clean, then commit it and run it in CI. Full design and every parsing pitfall: `references/verification-gates.md`. The startable version is `scripts/check_docs_corpus.py`.

1. Every manifest page exists; no orphan pages.
2. Structure: metadata line, first section, closing sections, prose-length budget.
3. Every internal cross-link resolves to a real page.
4. Every embedded asset exists on disk.
5. Every mermaid block parses under the real parser (not a regex).
6. No credential shapes; filler words reported as warnings.
7. External links resolve (treat blank status as a failed request, 403/429 as unverifiable, 404 as a defect).
8. Spot-check one version-specific claim per section against the harvested source; a claim you cannot locate is a fabrication to remove or hedge.

A gate that fails listing forward links to pages you have not written yet is the gate working. Do not soften it to go green early.

## 8. Publish

`references/repo-and-wiki-publication.md` covers repo creation, the wiki initialisation trap (a wiki's git repo does not exist until its first page is created in the web UI), the sync script, and post-push verification. Ship the sync as a template (`templates/push_wiki.sh`) so the interactive step is the user's one click, not a hand-typed sequence.

## Pitfalls

- **Never invent a commit SHA, version, price or quota.** Writing one from memory into a workflow file is fabrication, and it is the failure mode reviewers check for first.
- Checker regexes bite in three specific ways: `findall` with ONE capture group returns strings, so unpacking raises "too many values"; a `#` inside a fenced code block is not an H1 (strip fences before prose checks); and a stripper that removes any two backticks as an inline span also eats fence markers, silently reporting zero problems. Details and safe patterns in the reference.
- Do not hand-maintain navigation: section indexes, the home page and the sidebar come from the manifest, or they lie.
- Removing a page means removing it from the manifest too, or the gate reports it missing forever.
- A page that summarises someone else's blog post adds nothing. Every page should carry at least one failure and its fix.
- If the corpus is being published publicly, decide `public` vs `private` deliberately and say which you chose in the final report — flipping it later is one command, but the content is already indexed.

## References

- `references/verification-gates.md` — checker design, regex and parsing pitfalls, mermaid validation recipe, external-link checking, sanitisation scan.
- `references/asset-generation.md` — HTML→PNG infographics via headless Chrome, CSS skeleton, formatting pitfall, visual QA.
- `references/repo-and-wiki-publication.md` — `gh` repo creation, GitHub wiki initialisation and sync mechanics, post-push verification, doc/code licence split.
- `templates/push_wiki.sh` — idempotent wiki sync script to copy and adapt.
- `templates/infographic-card.html` — starter card for the renderer.
- `scripts/check_docs_corpus.py` — generalised corpus checker; run it, do not retype it.
