---
name: atlas-lesson-bank
description: "ATLAS durable memory system. Master lessons live in ~/Dev/ATLAS-LEARNINGS/LESSONS.md, not just in the 2,200-char MEMORY.md. Use to recall past lessons, store new ones, and keep memory compressed."
version: 1.0.0
metadata:
  hermes:
    tags: [memory, lessons, recall, knowledge-base, durable, compression]
    category: note-taking
    related_skills: [skill-quality-audit]
---

# ATLAS Lesson Bank

## Why this exists

MEMORY.md is bounded (2,200 chars) and injected frozen into every system prompt. It fills up and the memory tool starts rejecting writes. Durable lessons must live in a FILE, not in memory. Memory keeps only pointers + the most critical facts.

## The system

- **Master file**: `~/Dev/ATLAS-LEARNINGS/LESSONS.md` — append-only, categorized (01 Bug Bounty, 02 OpenCode, 03 Skills, 04 Operations). Each entry: 1-5 lines, dated, sourced.
- **Weekly snapshots**: `~/Dev/ATLAS-LEARNINGS/YYYY-MM-DD.md` — full reports from the weekly learning sweep cron (job 3ded7d48e350, Tue 15:00 EAT).
- **Memory (MEMORY.md)**: compressed pointers only. When memory is full, move detail into LESSONS.md and keep the pointer.
- **Sessions DB**: `session_search` for conversation-level recall — the lesson bank is for distilled knowledge, session_search is for "what did we do in that session".

## When to read the lesson bank

1. At the start of any complex task (5+ tool calls expected) — skim LESSONS.md for relevant past lessons
2. When starting a bug bounty hunt or writing a report — section 01 has rejection patterns and target intel
3. When using OpenCode — section 02 has model availability and workflow
4. When memory seems thin or a memory write fails — read the file, then compress memory
5. Weekly learning sweep — read before, append after

## When to append to the lesson bank

- After every bug bounty submission outcome (accepted/rejected/duplicate) — record the lesson
- After every weekly learning sweep — record what was learned
- After every integration discovery (new tool, new model availability, new workflow)
- After user corrections — record what was corrected
- After any task where "I'll remember this next time" comes to mind

## How to append

Use `patch` or `write_file` on `~/Dev/ATLAS-LEARNINGS/LESSONS.md`. Find the right category section and add a dated entry:

```
### Topic (YYYY-MM-DD)
- One-line lesson with enough context to be useful later
- Include source/repo/URL where relevant
```

Keep entries tight (1-5 lines). The file grows over time — that's fine, it's on disk not in context.

## How to compress memory (when memory tool errors "would exceed the limit")

1. Read LESSONS.md to confirm the content is already captured there
2. Replace the verbose memory entry with a short pointer: `See LESSONS.md §02 for OpenCode details`
3. Only keep in MEMORY.md facts that are needed EVERY session (target creds, cron IDs, critical workflow rules)
4. Retry the memory write

## Rules

- NEVER delete a lesson from LESSONS.md to make room — the file is the durable store, memory is the cache
- Prefer `patch` (targeted) over `write_file` (whole-file) for appends to LESSONS.md
- If LESSONS.md grows past ~1000 lines, archive old sections to `~/Dev/ATLAS-LEARNINGS/archive/` with dates
- Every weekly sweep cron run MUST: read LESSONS.md first, append new lessons at the end
