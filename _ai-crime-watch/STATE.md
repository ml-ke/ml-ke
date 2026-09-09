# AI Crime Watch — Series State

version: 1
status: idle
cycle: 1
issue: 1
due: 2026-09-13
draft: _ai-crime-watch/cycles/cycle-1/draft.md
published: none
updated: 2026-09-09 14:45 EAT

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
  14:4x EAT (created 2026-09-09). Finisher job created same day (daily
  16:30 EAT, monitor-gated — sleeps while idle).
