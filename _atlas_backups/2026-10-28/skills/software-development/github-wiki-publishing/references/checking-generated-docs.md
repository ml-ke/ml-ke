# Checking a generated docs tree

A deterministic checker is what makes a large generated set publishable. Offline, stdlib-only, exit
non-zero, run before every commit and in CI.

## What the gate must cover

1. **Manifest coverage, both directions** — every listed page exists; no page exists that the manifest
   does not know about.
2. **Structure** — no H1 in the body; the metadata blockquote is the first line; the closing sections
   are present; `## Try it` and `## Common mistakes` warn when missing.
3. **Length budget** — prose words within a range, computed with fenced code and tables removed, so a
   code-heavy page is not mis-scored.
4. **Internal links resolve** — every `[label](Page-Name)` maps to a real page, and any `.md` link is an
   error (diffable renderers never carry the extension).
5. **Embedded assets exist** — parse the asset URLs and stat the local files.
6. **Mermaid** — fences balanced, allowed diagram type on the first line, no init directives or click
   handlers, size warning past ~20 nodes.
7. **Filler words and secret-shaped strings** — the latter as errors and always scanned over the whole
   file, never just prose (`sk-…`, `ghp_…`, `AKIA…`, JWT-shaped, private-key headers).

## False positives to design around (all encountered in practice)

- **Shell comments are not headings.** A fenced block containing `# choose HTTPS` reads as an H1 when the
  raw text is scanned. Strip fenced blocks before heading, structure and word-count checks.
- **Inline code spans break fence counting.** Prose that mentions the mermaid fence opener in backticks
  inflates the count and reports "unbalanced fence". Strip inline spans (`` `[^`\n]*` ``) before counting.
- **`re.findall` with a single capture group returns strings, not tuples**, so `for label, target in …`
  raises "too many values to unpack". Use two capture groups, or `finditer` with `.group(n)`.
- **Generated navigation legitimately differs from the page skeleton.** Home may carry a title; an index
  page has no key takeaways. Detect and exempt those by name or pattern rather than loosening the rules
  for every page.
- **Alias renamed pages.** A link to `Home` resolves to `00-Home.md` in the source tree; keep an alias
  map so the checker does not report a broken link.

## False negatives to design around (worse than false positives)

- **An exemption rule that matches everything disables the gate silently.** An exempt set derived from
  `^\d\d-[A-Z]` matched every lesson page too (`01-The-Agent-Loop.md`), so structure, length, internal
  link, mermaid, filler-word *and* orphan checks were skipped for 138 pages — while the checker printed
  `0 errors · 0 warnings`, which reads as success. Identify generated pages by their **content** (the
  overview blockquote a section index opens with), not by the shape of their filenames, keep the exempt
  list short, and assert its size so it cannot silently grow to everything.
- **"Clean run" is not evidence the checks ran.** Print what each check examined — pages checked vs pages
  exempt, subjects per rule — and prove a rule applies by running it against a page that violates it on
  purpose. A gate that cannot fail is a pass-through.
- **Counts asserted in prose drift from the tree.** A hand-written `Total pages: N lessons + M section
  indexes` header was wrong for months and nothing contradicted it. Recompute asserted numbers inside the
  checker and fix header and checker in the same change.
- **A check scoped to a filename pattern stops covering renamed pages.** After a rename, re-run the checker
  and confirm the rule still looks at that page instead of assuming the rename was cosmetic.

## Working style

- Keep the checker in `scripts/` and wire it into CI with `--strict`, so warnings become failures once
  the tree is clean.
- When the checker is wrong, fix the checker in the same change as the page and say so — a checker that
  flags good pages teaches writers to ignore it.
- Warnings for taste (length, filler words, missing sections); errors for truth (missing page, broken
  link, missing asset, secret-shaped string).
- Reconcile the reported page count against `ls`. A silent discovery bug makes the gate look green over
  an empty tree — the worst possible failure for a quality gate.
