# GitLab Puma Worker Tuning in Docker

## Problem
GitLab's default Puma configuration uses 8+ workers, each consuming ~800 MB RSS. On a system with 12 GB RAM, Puma alone can consume 6+ GB, forcing heavy swapping and thermal throttling.

## Fix

### 1. Edit gitlab.rb inside the container

```bash
# Find the commented-out settings
docker exec gitlab-local grep -n 'puma\[.worker_processes.\]' /etc/gitlab/gitlab.rb

# Uncomment and set to 2 workers, 4 threads
docker exec gitlab-local sed -i 's/^# puma\["worker_processes"\] = 2/puma["worker_processes"] = 2/' /etc/gitlab/gitlab.rb
docker exec gitlab-local sed -i 's/^# puma\["min_threads"\] = 4/puma["min_threads"] = 4/' /etc/gitlab/gitlab.rb
docker exec gitlab-local sed -i 's/^# puma\["max_threads"\] = 4/puma["max_threads"] = 4/' /etc/gitlab/gitlab.rb
```

### 2. Reconfigure GitLab

```bash
docker exec gitlab-local gitlab-ctl reconfigure
```

Wait ~10 seconds for the phased restart to complete (old workers die off gradually).

### 3. Verify

```bash
# Check generated puma.rb confirms the setting
docker exec gitlab-local grep 'workers' /var/opt/gitlab/gitlab-rails/etc/puma.rb

# Count running workers
docker exec gitlab-local ps aux | grep 'puma.*worker' | wc -l
# Expected: 3 (1 master + 2 workers)

# Check memory impact
docker stats --no-stream gitlab-local
```

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Puma workers | 8+ | 2 |
| Container RAM | ~4.9 GB | ~2.2 GB |
| Host free RAM | 1.0 GB | 5.5 GB |
| Load average | 260+ | 1.0 |
| CPU frequency | 869 MHz (throttled) | 2,300 MHz |

## Prevent Auto-Start

If GitLab shouldn't start on boot:

```bash
docker update --restart=no gitlab-local
```

To start manually later:

```bash
docker start gitlab-local
```
