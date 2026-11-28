# Sweeping a whole blog corpus for AI-writing tells

Procedure for humanizing many already-published posts. Use it when the ask is "fix all
posts" rather than "humanize this draft" — the per-post skill (`humanizer`) is the
instruction set for each edit; this is the orchestration around it.

## Why wave-based

A single pass over 100+ posts cannot be verified by hand, and a subagent's self-report is
not evidence. Batching into waves of 10 keeps each commit small enough to review and
revert, and makes the verification step finite.

## Procedure

1. **Audit first, edit second.** Run `scripts/check-prose-style.py --json _posts/*.md` over
the corpus, then rank. Rank by severity (`emoji*3 + prose_em_dashes + titlecase*2 + 2*len(ai_vocab)`),
worst first — if the sweep is interrupted, the biggest offenders are already fixed.
   Save the ranked list to a worklist file; it is the queue for every wave.
2. **Gate the corpus by month** before committing to the plan. A per-month breakdown
   (`posts / failing / avg prose em dashes`) tells you whether you are fixing a legacy
tail or a live regression. If recent months are already clean, the mandate is working and
you are only backfilling.
3. **Verify the detector before trusting it at scale.** Sample the flagged headings. A
   regex that fires on 75 posts is worth 30 seconds of confirmation — `## The Mathematical
   Foundation` is a real tell; a heading that only *looks* title-case (proper nouns) is not.
4. **Waves of 10 parallel subagents**, one post each. Give each child: the hard invariants,
   the emoji policy, the full tell list from `humanizer`, and the two scripts to self-verify.
   Dispatch the *next* wave only after verifying and committing the current one.
5. **Verify independently, every post.** Run `scripts/verify-wave.sh <wave files>`. Never
   commit on a child's word. A wave that fails verification gets repaired before commit.
6. **Commit per wave** with the measured before/after per post in the message. One wave =
   one revertable commit.
7. **Re-run the full corpus audit at the end** to confirm the count moved.

## Hard invariants for every post

- YAML front matter byte-identical (including the post title — it is not a section heading).
- Fenced code blocks and inline `code spans` byte-identical.
- No URL, link target, number, statistic, date, proper name or table row data changed.
- No claims/sections added; only a heading-restating one-liner may be deleted.
- Reference blocks exempt at ANY heading level — do not rewrite their separators to hit the band.

## Pitfalls found in practice (all of these happened)

- **A gate false positive makes children edit CONTENT, not style.** The title-case regex fires on any
  all-capitalised heading, so it flagged `## Deep Learning Indaba` (a real conference) as a tell. A
  child renamed it to `## The Indaba` to go green - changing a real name to satisfy a broken check.
  Keep a `TITLE_CASE_ALLOWLIST` of genuine proper-noun headings in the corpus (currently
  `Data Science Nigeria`, `Deep Learning Indaba`, `ONNX Runtime Mobile`) and tell every child:
  **if a heading is a genuine proper noun, report the false positive - never rename it.**
  The same failure mode recurs for every over-eager rule: reference-block variants, flag emoji,
  technical vocabulary. Whenever a child reports contorting content to pass, suspect the gate first.
- **Substring checks must strip markup.** The allowlist initially matched nothing because `findall`
  returns the heading *including* its `## ` prefix. Compare the heading text, not the raw match.
- **Checker excluded only `## References`, not `### References`.** Posts using the h3 form
  had their reference separators counted toward the band, which pushed them over and caused
  a child to rewrite house formatting to go green. Exclude at any heading level and revert
  such rewrites.
- **Emoji policy is ambiguous unless you spell it out.** "Remove emoji but don't touch data"
  is contradictory when the emoji *are* the data. State it explicitly: decorative emoji
  deleted; data-encoding emoji transliterated (`check`→Yes, `cross`→No, `red`→Critical,
  `ballot`→No); never deleted; **never blanked**. A blanked status column is data loss.
- **`verify-post-facts.py` failed on posts with zero em dashes** (`0 -> 0` reads as "no
  reduction"). Title-case-only and emoji-only jobs are legitimate, so the em-dash check must
  be skipped when the original had none.
- **"No edit applied" is a legitimate outcome**, not a failure — a post may already be in
  band. Only treat it as suspicious when the style gate still fails.
- **Children over-edit when allowed to.** Some sentence-cased headings the checker never
  flagged "for consistency". Defensible, but it enlarges the diff — decide whether that is
  in scope before the sweep, not after.
- **A metric that UNDER-counts is worse than one that over-counts.** The title-case regex required
  3+ consecutive capitalised words, so `Common Pitfalls`, `Key Takeaways` and `Series Navigation`
  scored as clean. Eight waves' worth of "title-case: 0" passes were therefore weaker than they
  looked, and 20 already-processed posts needed a mop-up pass. Fixing the detector *mid-sweep*
  invalidates every earlier pass, so calibrate the detector against a sample of the real corpus and
  spot-check what it flags BEFORE the first wave, not after wave 8.
- **Flag by word, not by whole heading.** Any heading-level pattern needs a new allowlist entry the
  moment a new proper noun appears (`South America`, `Amazon Neptune`, `REST API`, `Lacuna Fund`,
  `Install Kserve`). Title case means *an ordinary word capitalised after the first word* - test each
  word after position 0 against a `PROPER_WORDS` set (acronyms, brands, places, technical terms)
  instead. That one rule absorbs new proper nouns without maintenance and still catches
  `Python Implementation` (Implementation) while ignoring `Install Kserve` and `REST API`.
- **Some flagged headings are already correct.** Test the detector's expectations, not just its
  output: `Getting Started` genuinely IS a tell ("Getting started" is the sentence case), so a
  test asserting otherwise is the bug. Verify each regression case by hand before "fixing" the code.
- **Do not build the title-case test out of a regex.** Three successive regexes under-counted or
  over-counted, each differently: `3+ consecutive capitals` missed `Key Takeaways`; a `2+` version
  missed headings with lowercase connectors (`Queue-Based Architecture with Redis + Celery`) and
  trailing punctuation (`What We'd Do Differently (The Honest Retrospective)`); loosening it further
  then flagged 669 headings, including `2. Tool call frequency` and `Sharding strategies for large
  KGs`, because it read a numbering prefix as position 0 and treated mixed-case acronyms (`LoRA`,
  `KGs`, `FB15k-237`) as ordinary capitalised words. Extract every heading and test word by word:
  strip a leading enumeration (`^\d+[.)]`, `^Step \d+:`) first, then flag only when a word after
  position 0 is capitalised, is not in `PROPER_WORDS`, and is not an acronym (`sum(c.isupper()) >= 2`).
  That final form passed 28 hand-checked regression cases.
- **Over-flagging is the more dangerous failure here.** A gate that under-counts makes you sign off
  early; a gate that over-counts makes children sentence-case proper nouns and numbered list items -
  editing real content to please a broken check, the exact failure this doc warns about. Always sample
  what the detector flags and classify a share of it by hand before dispatching a wave on it.
- **Fixing the detector mid-sweep invalidates earlier passes - budget for a mop-up.** When the
  title-case test was corrected after wave 8, posts already reported as "0 title-case" failed again.
  Re-running the corrected gate over every committed post and dispatching a focused mop-up pass
  (headings only, no prose changes) is the recovery; it is cheap if the mop-up prompt is narrow.
- **A child's self-report of "titlecase=0" only ever means "0 under the detector that child ran".**
  Children also flagged gaps I had not: headings ending in `)` and headings with lowercase
  connectors both slipped the old pattern. Treat such a report as a useful bug report, not noise.
## Script the mechanical part, delegate the judgement part

Sentence-casing Title Case headings is mechanical: lowercase ordinary words, keep proper nouns.
Across 63 posts and 485 headings a script did it in seconds, where subagent waves had spent eight
rounds and still needed mop-ups. Reserve subagents for what needs judgement - em-dash reconstruction,
AI-vocabulary, restructuring - and script the rest.

The guard that makes it safe to script: **lowercase a word only when you can prove it is ordinary**,
by checking its lowercase form actually occurs as a word in the corpus prose. A proper noun like
`Nairobi` never appears as `nairobi`, so it is never touched.

Proof-first is necessary but NOT sufficient. Words like `android`, `claude`, `swahili`, `trivy`,
`lagos`, `crew`, `sonnet` all appear lowercase somewhere (URLs, code blocks, prose) yet are proper
nouns in a heading. Enumerate those explicitly in `PROPER_WORDS`; the corpus-frequency test cannot
distinguish `act` the verb from `Act` in "EU AI Act".

Three checks make the result trustworthy:
- **The fixer must honour the heading allowlist too.** It was rewriting `Deep Learning Indaba` to
  `Deep learning Indaba` because it consulted `PROPER_WORDS` but not `TITLE_CASE_ALLOWLIST`. A
  detector that ignores a heading does not stop a fixer from damaging it - guard both.
- **Assert the blast radius.** `git diff -U0 | grep -v '^[+-]#\{2,4\} '` must be empty: heading lines
  only. This proves no prose, code, link or front matter moved.
- **Review the distinct word list, not the diff.** 485 heading edits are unreadable; the ~585
  distinct words they lowercased can be scanned in one pass, and that is where the proper nouns show
  up. Words left capitalised because they are hyphenated or unprovable (`Cyber-Tiered`) show as
  leftover failures - hand those to the next pass.
- **"Don't rename proper nouns" is too narrow an instruction - children will REWORD instead.**
  Told not to rename a flagged proper noun, children obeyed the letter and reordered or shortened the
  heading to move it out of the flagged position. Real losses followed: `South America: Brazil Hedges
  With Two Supercomputers` became `South America: two supercomputers` ("Brazil" gone),
  `Europe: Sovereignty's Chinese Paradox` became `...sovereignty's China paradox` ("Chinese" silently
  changed to "China"), and `The Flutterwave case: ...` lost its framing. State the rule as "leave the
  heading byte-identical and report it" - and afterwards diff every heading line the wave touched,
  not just the gate result. A passing gate hides a deleted word.
- **A proper-noun list will always lag the corpus; expect to extend it.** Each wave surfaced new
  misses that children had to work around: `Chinese`, `Brazil`, `Russian`, `Astra`, `Danske`,
  `Russia`, `MENA`, `China`, `Asia`. Since a missing entry causes content damage rather than a
  harmless flag, seed the list generously with countries, nationalities and brand names up front.
- **Possessives hide proper nouns from a word list.** `Kenya's` does not match `kenya`, so a heading
  that is already correct reads as title case. Strip a trailing `'s` before the proper-noun lookup.
- **Reverting damage is cheap once the detector is fixed.** After adding the missing names and the
  possessive rule, the six reworded headings were restored from `HEAD` and re-cased by the script -
  five of those files ended up with no heading change at all. Fix the cause, then re-derive the
  edit from the original rather than hand-patching the damaged text.
## The title-case test, in full

Six separate bugs surfaced before it was right. The working rule (in check-prose-style.py, shared by
`is_tell_word()` with the fixer so the two cannot drift) is: scan **every** heading, strip a leading
enumeration (`2. `, `Step 5: `), then flag only when a word after position 0 is capitalised and NONE
of these hold:

- it is in `PROPER_WORDS`, or its possessive form is (`Kenya's` -> `kenya`)
- it is an acronym by `sum(c.isupper()) >= 2` (REST, LoRA, KGs, SafeInstr)
- it contains a digit - an identifier, not a word (S3, K3, Core42, A24, `R$23`, GPT-5)
- it is a single character (`A` from `A*`)
- it is the pronoun I or a contraction of it (I'd, I'm, I've - always capitalised in English)
- the heading contains an allowlisted proper-noun phrase (substring, not exact match, so
  `Data Science Nigeria (DSN)` is covered)

The order matters: `PROPER_WORDS` overrides everything, so a dictionary word that is also a name
(`china`, `act`, `crew`, `kit`, `lite`, `sonnet`) stays protected.

Seed `PROPER_WORDS` generously with countries, nationalities and brands. A missing entry causes
**content damage** (children reword the heading) where a false flag only causes noise, so the cost
of over-inclusion is far lower than the cost of omission.

## Scripting vs delegating

Run `scripts/sentencecase-headings.py` for headings - see its docstring. It needs one thing from
you: proof-of-ordinariness is necessary but not sufficient, so review the *distinct words it
lowercased*, not the diff, and confirm afterwards that no non-heading line changed.

## Reference-like sections to exempt (measured on this corpus)

`## References` (82 posts), `## Related posts` (40), `## Further Reading` (10), `### References`
(5), `## Further reading` (1). Exclude all of them, at any heading level, case-insensitive.

## Verify before reporting

Run the third-party checks, not the child's summary:

```bash
python3 scripts/check-prose-style.py  _posts/<slug>.md   # in band?
python3 scripts/verify-post-facts.py  _posts/<slug>.md   # undamaged?
```

Plus per-wave: `scripts/verify-wave.sh $(cat worklist-wave.txt)`.
