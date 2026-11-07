# Parallel content generation

Writing 100+ consistent pages is a pipeline problem, not a writing problem.

## 1. Sources → digest

Read each source repo's orientation and lessons material (`AGENTS.md`, docs trees, CI/CD runbooks,
`*-gotchas.md`, retrospectives, lesson banks). Concatenate the valuable parts into one scratch digest:
per-file cap with head+tail truncation, a banner per source naming its path, and a note that the file is
machine-assembled so writers grep headings instead of reading linearly.

Why a digest: the parent context stays small, and every writer is grounded in the same real material
instead of inventing plausible filler.

## 2. Link bank

Run the topic searches once, save `{query: [{title, url}]}` as JSON, then emit `LINK-BANK.md` as
`- title — url` lines grouped by query. Tell writers: **this file plus the named documentation roots are
the only URLs you may use**; if a needed link is absent, link the docs root. This single rule removes the
most common defect in generated documentation.

## 3. Manifest as data

One record per page, carrying everything a writer needs:

```
(file, title, level, minutes, prereq,
 coverage=[...],     # what the page must teach, in order
 diagrams=[...],     # required mermaid, described by the parent, written by the child
 assets=[...],       # exact PNG filenames the page embeds
 reads=[...])        # digest / research files to ground it
```

Group pages so one writer can finish in a sitting (5–12 pages), and split a large section into `<NN>a` /
`<NN>b` groups. Generate from the manifest:

- `specs/G<NN>.md` — the per-group writer brief.
- `WIKI-MANIFEST.md` — the committed, numbered page list.
- Section index pages, the home page, `_Sidebar.md`, `_Footer.md`.

Regenerate after any edit that renumbers or adds pages. Never hand-edit generated navigation.

## 4. The spec file is the contract

Each spec opens with the reading order — `STYLE.md` in full (binding) → `LINK-BANK.md` → the listed
`Reads` → then the output contract:

- Write only these exact filenames; one page per file; 600–1000 words of prose.
- Follow the page skeleton; no H1; metadata blockquote first; closing `## Key takeaways` and
  `## Further learning`.
- Embed only the listed assets, in the exact raw-URL form.
- Internal links use page names, checked against the manifest.
- Restate the sanitiser rule with concrete examples of what to generalise.
- **Do not run git, do not commit, do not touch any other file** — other agents are writing concurrently.
- Write the done-file: one line per page, `path<TAB>wordcount`.

## 5. Writer task prompt checklist

Repeat all of it in every task; children know nothing of the parent conversation:

1. What the resource is and who the reader is (one line each).
2. Absolute paths: style file, spec file, link bank, manifest, done-file.
3. Workflow order: read → write → done-file → report.
4. Hard rules restated: no H1, word budget, required closing sections, listed assets only, banked URLs
   only, mermaid syntax limits.
5. The sanitiser rule.
6. "No git, no other files."
7. On finish, report files written, word counts, and **anything you could not ground** — that last item
   is how you learn where the source material is thin.

## 6. Wave discipline and verification

- Dispatch in waves at the platform's parallel limit, shaped by section so each child owns a coherent
  area and cross-links land in or next to it.
- Assets and tools land **before** the first wave (fixed filenames, visually checked).
- After each wave returns: list expected filenames and diff against disk, count prose words per page,
  run the checker, then read two or three pages end to end for voice and grounding.
- Treat every child claim as unverified until the filesystem agrees, and report stubborn gaps instead of
  papering over them.

## 7. Fixing a bad wave

Most defects are cheap to fix in place and expensive to re-dispatch: a stray H1, a filler word, a link to
an old page name, a wrong asset filename, a section that drifted off-spec. Patch the pages yourself. Re-
dispatch only for a wholly missing page or a group that clearly misread its spec.
