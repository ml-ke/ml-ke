# AI Crime Watch — Series State

version: 1
status: published
cycle: 1
issue: 2
due: 2026-09-12
draft: _ai-crime-watch/cycles/cycle-1/draft.md
published: 2026-09-09-ai-crime-watch-issue-1
updated: 2026-09-09 16:40 EAT

## How this file works

Machine-parseable top block ONLY (status / cycle / issue / due / draft) — it
drives the finisher gate script `~/.hermes/scripts/ai_crime_watch_signal.py`
which prints `PENDING <issue> <due>` when status=pending, else `IDLE`.
Never put timestamps or prose in the top block (the signal must be
deterministic). Everything below is the human log.

- `status`: idle | pending | published | skipped
- `cycle`: current research cycle number (opened by an engine fire or the kickoff session)
- `issue`: NEXT issue number to use (increments when an issue is published)
- `due`: polish deadline YYYY-MM-DD (only meaningful when status=pending)

Who writes what: the engine (bi-weekly cron) opens/closes cycles; the
finisher (daily monitor-gated cron) continues pending cycles; this session
kicked off cycle 1 manually.

## Cycle log

- cycle 1 (→ issue 1): opened 2026-09-09 by kickoff session. Research +
  draft produced in-session; first-ever engine cron fire is ~2026-09-23
  14:41 EAT (created 2026-09-09). Finisher job created same day (daily
  16:30 EAT, monitor-gated — sleeps while idle).
- cycle 1 verdict (2026-09-09): NOT READY at kickoff. In-window
  (Aug 19–Sep 9) crime×AI material is thin and the strong August anchors
  were already used by the blog (INTERPOL 55% → Aug 11 AI Update spotlight;
  Operation Jackal + $442B + Sumsub Kenya 10% → Aug 26 deepfake post; EU AI
  Act name-drops). Uniqueness map + ranked candidates + draft skeleton in
  `cycles/cycle-1/`. status=pending, due 2026-09-12; finisher job
  8ea5a3a5de4d wakes daily 16:30 EAT until publish or skip. Engine job
  d75da864fce0 next fire ~2026-09-23 14:41 EAT (opens cycle 2; supersedes
  this cycle if still pending after due).
- cycle 1 PUBLISHED (2026-09-09, finisher first wake 16:30 EAT): finisher
  verified the Taiwan voice-clone romance-scam indictment (Sep 2, CNA +
  Taipei Times bodies) — the fresh in-window anchor the kickoff sweep missed
  (its Sep 1–9 queries surfaced nothing). Anchor: 57 indicted, NT$900M
  (US$28M+), 20,000+ victims, bespoke voice AI trained on 22 employees'
  voices. Angle differentiated from Aug 26 corporate-deepfake post
  (consumer romance-scam lane + enforcement milestone + pillar-5 research
  edge: Deepfakes We Missed / ViKing). Post:
  `_posts/2026-09-09-ai-crime-watch-issue-1.md`, cover
  `assets/img/cover-ai-crime-watch-issue-1.webp`. All ready-gate checks
  passed (concrete: Sep 2 anchor ≤21 days; unique: no Taiwan/romance-scam
  coverage in _posts; every claim 2+ body-level sources). Next engine fire
  (~Sep 23) opens cycle 2 for issue 2.
