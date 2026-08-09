---
name: linux-performance-diagnosis
description: Diagnose why a Linux desktop/server is slow — memory pressure, swap thrashing, thermal throttling, process hogs, Docker container bloat, crash loops. Use when user asks "why is the system slow", "check what's making it slow", "do another performance check". Produces a metric table BEFORE → AFTER each fix so the user sees real impact.
---

# Linux Performance Diagnosis

The user runs a resource-constrained desktop (i7-1065G7 8-core, 11GB RAM) with Docker stacks (GitLab, Supabase, etc.) and browsers with many tabs. They want **measured diagnosis + fixes with before/after numbers**, and sign-off before destructive actions.

## Baseline snapshot (run first, one command)
```bash
free -h; swapon --show; uptime
ps aux --sort=-%cpu | head -15
ps aux --sort=-%mem | head -15
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count
```

## Read the signals
- **Load average ≫ core count** + `kswapd0` at high %CPU → memory pressure/swap thrashing (not CPU-bound)
- **Swap full + available RAM low** → find what's holding RAM: per-app RSS sums
- **CPU freq stuck near minimum (400 MHz) + rising thermal_throttle counters** → hardware overheating; the fix is heatsink clean/repaste, not software. Throttle counters in the 10⁴–10⁵ range over hours = chronic.
- **Swap still shows usage after freeing RAM** → normal; kernel reclaims lazily, not a problem
- **`docker stats`** — container-level memory; a crash-looping container (`Restarting (1)`) churns CPU every ~60s

## Per-app aggregation (browsers are usually the whale)
```bash
ps aux | grep brave | grep -v grep | awk '{sum+=$6; n++} END {printf "%d procs, %.0f MB\n", n, sum/1024}'
```
- Chromium-family: 1 renderer per tab, ~150–900 MB each. 35+ renderers = multi-GB.
- Two browsers (Brave + Chrome) simultaneously = pure redundancy, free GBs by closing one.

## Common fixes on this box (measured 2026-06)
- **GitLab Puma 8 workers ≈ 6.1 GB** → cap to 2 (`puma['worker_processes'] = 2` in container gitlab.rb + `gitlab-ctl reconfigure`) → freed ~4 GB, load 260 → 1.0
- **Stop unused Docker stacks** (`docker compose down` in project dir) → e.g. yucan stack freed ~2.2 GB
- **Kill idle GUI apps** (gnome-software ~460 MB) — `pkill -9 <name>`; user is fine with this
- **Prevent auto-start of unused containers**: `docker update --restart=no <container>` — user rarely uses Docker, prefers manual `docker start`
- **Docker container k3s/nemoclaw**: check if the k3s server is inside a Docker container (`/proc/<pid>/cgroup` or PPID chain → containerd-shim → `docker ps`). Stopping the container stops k3s; it has `restart: unless-stopped` by default → `docker update --restart=no` first.

## Crash-loop diagnosis (container `Restarting (1)` every ~60s)
- `docker logs <c> --tail 15` — read the fatal error; it repeats each cycle
- Compose-file placeholder secrets are a top cause: `GOTRUE_DB_DATABASE_URL: "postgres://user:***@db:5432/postgres"` — the `***` was never filled in
- Verify the DB-side role: `docker exec <db> psql -U postgres -t -c "SELECT rolpassword IS NULL FROM pg_authid WHERE rolname='<role>';"` — role exists but `rolpassword IS NULL` = no password set at all
- **Red herring to avoid**: `psql` connecting to `localhost` inside the DB container SUCCEEDS with any password because pg_hba has `host all all 127.0.0.1/32 trust`. The container-to-container path (`db:5432`) hits the `host all all all scram-sha-256` line and FAILS. Always test the real network path, not localhost.
- User verdict on unused stacks: stop them (`docker compose down` in project dir) and fix later when actually needed — don't sink time fixing a dev stack nobody is using.

## Reporting pattern (user expects this)
- Table of metric BEFORE → AFTER for every fix (RAM used, free, swap, load, CPU freq)
- Call out the single biggest consumer explicitly ("Brave = 86% of used RAM")
- End with "what's left" (e.g. hardware throttling needs physical repaste)
- Ask before killing user-facing apps (browser tabs, IDE); apply container/system fixes with their ok

## Pitfalls
- `sudo` prompts for password in non-interactive shells — use `sudo -n` or work without sudo where possible
- Docker containers share the host PID namespace — `ps aux` on host shows container processes; `docker exec <c> ps aux` shows the same PIDs
- RSS double-counts shared memory across processes — fine for ranking, not exact totals
- `docker exec` fails on crash-looping containers ("is restarting") — read config/logs via `docker inspect` / `docker logs` instead
- Thermal throttle counters persist across the session — compare deltas, not absolutes, when re-checking after a fix
- GitLab `gitlab-ctl reconfigure` does a phased Puma restart — old workers linger ~10-15s; wait then re-check worker count

## References
- references/gitlab-puma-tuning.md — exact steps to cap GitLab Puma workers in Docker
