---
name: writing-router
description: "Route writing tasks to the right skill; humanize last."
version: 1.0.0
author: ATLAS
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [writing, routing, docs, blog, prose, editing, humanizer, pipeline]
    category: writing
    related_skills: [humanizer, blog-drafting, grounded-citations, research-paper-writing, documentation-corpus-build, github-wiki-publishing, writing-plans, docx, pdf, powerpoint, hodaripay-docs-workflow, hodaripay-adr, hodaripay-fintech-brs, hermes-agent-skill-authoring, topic-scouting, email-inbox-triage, himalaya, youtube-content, document-to-action-items, meeting-action-items, pre-submission-verification]
---

# Writing Router

## Overview

The single entry point for **any writing task**: blog posts, documentation, READMEs, release notes, specs, ADRs, papers, reports, emails, threads, decks. It answers two questions the other skills do not answer for you:

1. **Which skill(s) do I load for this writing task?** (the decision table below)
2. **In what order?** — content skill first, then `humanizer` as the mandatory final pass.

Do not use this skill as the writer itself. It routes, then gets out of the way. The actual craft lives in the target skill.

> ⚠️ **The 60-character rule.** Hermes renders skill descriptions in the system prompt truncated to `SKILL_PROMPT_DESC_LIMIT = 60` chars (57 + `...`). Everything after that is invisible when the agent picks a skill. A skill nobody can route to does not exist. When you author or fix a skill description, the trigger must be self-contained in the first 57 characters. Verify with: `python3 -c "import re,sys;print(len(re.search(r'description: ?(.*)',open(sys.argv[1]).read()).group(1)))" <SKILL.md>`.

## When to Use

Load this skill when the user asks for prose that a human will read:

- "write a blog post", "draft an article", "put together a roundup"
- "write documentation", "update the docs", "add a README", "document this API", "write release notes", "write a changelog entry"
- "draft an email", "write a thread", "turn this into tweets", "write up the findings"
- "edit this", "polish this", "tighten this", "make this read better"
- "write a spec / BRS / ADR / proposal / memo / paper"

**Don't use for:** code (that's the language's own skill), JSON/YAML/CSV output, or a plan you will execute yourself (use `writing-plans` — it is an implementation plan, not prose for a reader).

## The decision table

Match the task, load the skill, then finish with `humanizer`.

| Writing task | Load first | Fact/source layer | Final pass |
|---|---|---|---|
| ML Kenya blog post (ml.co.ke, Jekyll/Chirpy) | `blog-drafting` | `blog-drafting` fact-check protocol | `humanizer` |
| Find blog topics / gaps | `topic-scouting` | — | — |
| Kubernetes of docs: README, API reference, release notes, changelog, runbook | this skill's procedure (below) | `grounded-citations` | `humanizer` |
| Multi-page docs site / handbook / corpus | `documentation-corpus-build` | `grounded-citations` | `humanizer` |
| GitHub wiki / docs collection | `github-wiki-publishing` | `grounded-citations` | `humanizer` |
| YucanPay repo docs (hodaripay) | `hodaripay-docs-workflow` | repo source | `humanizer` |
| Architecture decision record | `hodaripay-adr` | repo source | `humanizer` |
| Business Requirements Spec (fintech) | `hodaripay-fintech-brs` | `choicebank-baas` / `smileid-kyc` | `humanizer` |
| ML paper (NeurIPS/ICML/ICLR) | `research-paper-writing` | `arxiv` | `humanizer` |
| Cited report, market/competitor research, briefing | `grounded-citations` | `grounded-citations` | `humanizer` |
| Word / PDF / deck deliverable | `docx` / `pdf` / `powerpoint` | — | `humanizer` on the prose **before** conversion |
| Obligations, deadlines, action items from documents | `document-to-action-items` | the document | — (mechanical) |
| Meeting notes → decisions, owners, tickets | `meeting-action-items` | the notes | — (mechanical) |
| Email: triage, draft, reply | `email-inbox-triage` | — | `humanizer` |
| Email: send from terminal | `himalaya` | — | `humanizer` on the body |
| Social thread / repurposing long-form | `xurl` (mechanics) + procedure below | source article | `humanizer` |
| YouTube video → summary, thread, blog | `youtube-content` | transcript | `humanizer` |
| Authoring or fixing a SKILL.md | `hermes-agent-skill-authoring` | — | `humanizer` on prose; `skill-quality-audit` for the structure |
| Bug bounty / vuln report | `pre-submission-verification` | PoC output | `humanizer` (**bug-bounty mode — mandatory**) |
| Implementation plan for you to execute | `writing-plans` | — | not required (internal artifact) |

If two rows fit, load the more specific one; if the task spans rows (e.g. "write docs for this repo and blog about it"), run the rows as separate passes, not one merged draft.

## The mandatory pairing rule

**`humanizer` is the last pass on every human-facing prose deliverable.** Load it after the content skill has produced a factually complete draft, never before.

Why the order matters:

1. **Facts before voice.** Polishing an unverified draft means re-polishing after the facts change. `blog-drafting` and `grounded-citations` verify first; `humanizer` runs on the verified text.
2. **`humanizer` is an editor, not a writer.** It removes AI tells (inflated significance, rule-of-three, em-dash overuse, AI vocabulary, tailing negations) and adds voice. It does not gather sources or fix structure.

**Load `humanizer` for:** blog posts, docs, READMEs, release notes, reports, papers, emails, threads, decks, PR descriptions, skill prose, bug bounty reports.

**The only exceptions** (do not run `humanizer`): machine-consumed output (JSON/YAML/CSV/SQL), code and code comments, data tables with no prose, internal plans you will execute yourself, and verbatim quotations or legal text.

For bug bounty reports specifically, the order is fixed: `pre-submission-verification` gates → `humanizer` (its "Bug Bounty Report Humanization" section) → submit. See that section for the 8 triggers that get reports flagged.

## The pipeline (order to actually run)

1. **Scope** — one sentence: who reads this, and what must they be able to do after reading it.
2. **Gather** — load the source layer (`grounded-citations`, `blog-drafting` fact-check, repo source, PoC output). No claims without a source.
3. **Structure** — load the format skill (`blog-drafting` / `docx` / `research-paper-writing` / ...). Fix the skeleton before writing prose.
4. **Draft** — write it complete and factually correct. Do not self-censor for style yet.
5. **Verify** — run the format skill's own checks (cross-links, `post_url` tags, code execution, image paths, CVE/secondary-source checks).
6. **`humanizer`** — final pass. Always. Then deliver.
7. **Record** — if anything was learned (a rejection, a new source, a routing miss), append it to `~/Dev/ATLAS-LEARNINGS/LESSONS.md` per `atlas-lesson-bank`.

## Procedure: generic docs with no dedicated skill

For READMEs, API references, release notes, changelogs and runbooks that have no format skill:

1. Read the actual source (code, diff, `git log`, the API). Never document from memory.
2. Lead with the reader's task, not the component's history: "To rotate a key, POST /keys/rotate" beats "The key subsystem was refactored".
3. One claim per sentence; every command must be copy-paste runnable and actually executed once.
4. Release notes / changelogs: bullet the user-visible change + the migration action. No adjectives.
5. Run `humanizer` before publishing.

## Procedure: social thread from long-form

1. Pull the source article's three strongest concrete facts (numbers, names, outcomes).
2. Hook tweet = the single most surprising fact, stated plainly. No "🧵" preamble, no "Let's dive in".
3. One idea per tweet, under 280 chars; end on a concrete action or the open question.
4. Run `humanizer` on the whole thread — thread hooks are the highest-density AI-tell surface.
5. Post mechanics: `xurl`.

## Common Pitfalls

1. **Writing with no skill loaded.** You already know how to write; the skills carry the facts, the format and the pitfalls (`blog-drafting` alone encodes ~12 build-breaking gotchas). Load the skill and follow it.
2. **Running `humanizer` first.** Editing before the facts are verified wastes the pass and can launder unverified claims into confident-sounding prose.
3. **Skipping `humanizer` because "it's internal".** PR descriptions, commit messages, tickets and skill bodies are read by humans. Only machine-consumed output is exempt.
4. **Merging two rows into one draft.** A post that is also the docs page serves neither reader. Split the passes.
5. **Describing from memory.** Documentation written without reading the source gets the details wrong and is not caught by any lint. Read the source first.
6. **Ignoring the 60-char rule when you create skills.** A new skill whose trigger is in chars 58+ is unroutable — `skill_manage(action='create')` rejects descriptions over 60 chars for exactly this reason.

## Verification Checklist

- [ ] Exactly one format skill loaded for the deliverable (or one per pass when the task spans rows)
- [ ] Facts/sources gathered before prose was written
- [ ] The format skill's own verification steps were run (build checks, code execution, cross-links, source checks)
- [ ] `humanizer` was applied as the final pass on human-facing prose
- [ ] Any lesson learned was appended to `~/Dev/ATLAS-LEARNINGS/LESSONS.md`

## References

- `references/kimi-and-writing-skill-research.md` — the sources behind this router (Kimi's creative-writing skills page, the open-source writing-skill landscape, the blader/humanizer provenance) and what was adopted vs rejected.
- `humanizer` — the final-pass editor (29 AI-tell patterns + bug-bounty mode).
- `skill-quality-audit` — description/trigger auditing, the `SKILL_PROMPT_DESC_LIMIT` check.
- `atlas-lesson-bank` — durable lesson recording.
