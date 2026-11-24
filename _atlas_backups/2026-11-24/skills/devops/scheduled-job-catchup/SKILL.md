---
name: scheduled-job-catchup
description: Use when a scheduled job missed runs and must be caught up.
version: 1.0.1
author: hermes-agent
license: MIT
metadata:
  hermes:
    category: devops
    tags: [cron, catch-up, audit, recovery, scheduler]
    related_skills: [hermes-maintenance, blog-drafting, atlas-lesson-bank]
---

# Scheduled job catch-up

"Catch up on anything you missed" is an audit, not a re-run. The output the user wants is: what was
actually missed, why, what was done about each one, and the evidence that the repair landed. Nothing
else.

## When to Use

- The user says "catch up on tasks", "anything you missed", or asks about a daily post, weekly report
  or snapshot that never arrived.
- A cron job shows a non-`ok` status, or a dated artifact the schedule implies is simply absent.
- You are about to re-fire a job whose last completed run is in doubt.

## Procedure

### 1. Build the missed-run inventory (two sources, both)

```bash
# status log: recent fires, including the ones that left no output file
sqlite3 ~/.hermes/cron/executions.db \
  "select job_id, started_at, status, substr(coalesce(error,''),1,80) \
   from executions where started_at > date('now','-10 days') order by started_at"

# artifact log: one file per completed fire
ls -lat ~/.hermes/cron/output/*/ | head -40
```

Also list the job definitions (`cronjob(action='list')`) so you can see each job's schedule and
`last_run_at` — a job whose `next_run_at` has advanced past a slot that `last_run_at` never covered has
a hole even if every recorded status is `ok`.

`executions.db` is a recent-window log and prunes; older gaps survive only in `cron_incidents` and in
whatever artifact the job writes (git commits, dated reports, appended lesson banks). Do not conclude
"nothing was missed" from a clean DB.

### 2. Classify each miss before touching anything

| Status signature | What it means | Default repair |
|---|---|---|
| no row, no output file | scheduler/offline gap; the fire never happened | do the work by hand, or re-fire labelled as catch-up |
| `failed` + `TimeoutError: … idle for 0s (limit 600s)` | the run stalled mid-way, possibly after writing its artifact | inspect the artifact first, then decide |
| `failed` + provider 401/402/429 | the LLM step died; the job's script may have finished | verify the script's artifact, re-fire only a missing report step |
| `unknown` + `Scheduler restarted after this execution's owner exited before finishing` | killed by a restart; usually produced nothing | re-run |
| `completed` but no artifact | reported success and did nothing | treat as a defect, not a success |

### 3. Verify the artifact before believing the status

A status is not an artifact. For every candidate miss, check the thing the job was supposed to produce:

```bash
ls -l --time-style=full-iso <artifact-dir> | tail -20    # mtime vs the missing date
cd <repo> && git status --short && git log --oneline -5   # staged/uncommitted work survives a failed run
```

A `failed` publish run can leave a finished, verified artifact uncommitted in the working tree; a killed
run usually leaves nothing. Backfill only what is genuinely absent. **Re-check mtimes immediately before
you report** — a backfill running in parallel makes an earlier audit read "nothing was lost" when that
was false when it was taken.

### 4. Repair by hand, not by re-firing publish jobs

Re-firing a publish or queue job treats *today* as missing, emits a second item for today, and leaves the
historical gap exactly where it was. Do the missing item by hand.

Research, learning and reporting jobs are the exception: they are safe to re-fire, and the fire should
carry the missed date as transient context so the output is filed under the week it belonged to:

```
cronjob(action='run', job_id='<id>', prompt='Catch-up fire: the <date> run never happened. Produce the
<period> report now and file the snapshot under <date>.')
```

### 5. Verify the repair the way the job's own gate would

Re-read the artifact; do not trust the repairing step's exit code. Run the same gate the job runs (its
style/lint check, its build, its verification script), and when the job appends to a shared log confirm
the append landed in the right section and under the right date.

### 6. Report in this order

The missed items and why each was missed, what was done for each, the evidence, then the next scheduled
fire. Offer the decision and the reason; do not narrate the audit.

## Pitfalls

- **A catch-up run's own claims about artifacts are claims.** A late-firing learning sweep asserting
"files exist for those dates, nothing was lost" can be true when written and false an hour earlier — it
read a directory a concurrent backfill had just filled. Re-derive it from mtimes and git instead of
inheriting the conclusion.
- **Do not re-fire to "refresh" a job whose artifact already exists.** Duplicate work plus a second dated
snapshot is harder to explain than a gap.
- **A job can be green and useless.** `completed` with no artifact means the run did not do its work;
audit the artifact, not the colour.
- **Weekly jobs caught up late: one catch-up per missed week, labelled.** Running the Tuesday sweep on
Sunday and again on Tuesday is correct when a week is genuinely missing; unlabelled, the two reports look
like a duplicated run.
- **Paused jobs are not missed jobs.** Check `enabled`/`paused_at` first; a deliberate pause is state, not
a gap.
