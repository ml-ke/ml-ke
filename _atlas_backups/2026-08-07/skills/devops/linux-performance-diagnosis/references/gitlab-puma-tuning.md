# GitLab Puma Tuning in Docker (verified 2026-06)

GitLab CE in Docker defaults to ~8 Puma workers × ~790 MB RSS ≈ 6.1 GB. On an 11 GB box that alone triggers swap thrashing (load 260+, kswapd0 at 40% CPU, CPU throttled to 400 MHz).

## Cap Puma to 2 workers
```bash
# 1. Uncomment/set in the container's gitlab.rb
docker exec gitlab-local sed -i 's/^# puma\[.worker_processes.\] = 2/puma["worker_processes"] = 2/' /etc/gitlab/gitlab.rb
docker exec gitlab-local sed -i 's/^# puma\[.min_threads.\] = 4/puma["min_threads"] = 4/' /etc/gitlab/gitlab.rb
docker exec gitlab-local sed -i 's/^# puma\[.max_threads.\] = 4/puma["max_threads"] = 4/' /etc/gitlab/gitlab.rb

# 2. Apply
docker exec gitlab-local gitlab-ctl reconfigure   # ~8s, phased Puma restart

# 3. Verify the generated config (NOT gitlab.rb — the source of truth is puma.rb)
docker exec gitlab-local cat /var/opt/gitlab/gitlab-rails/etc/puma.rb | grep -E 'workers|threads'
# expect: workers 2 / threads 4, 4

# 4. Wait ~15s for phased restart, then count on host
ps aux | grep 'puma.*cluster worker' | wc -l   # expect ~2-3 (master + workers)
```

Measured result: container RAM 4.86 GiB → 2.24 GiB, host load 260 → 1.0, CPU freq 869 MHz → 2300 MHz.

## Prevent GitLab auto-start (user rarely uses Docker)
```bash
docker update --restart=no gitlab-local
# start manually when needed:
docker start gitlab-local
```

## Notes
- `gitlab-ctl reconfigure` output ends with "gitlab Reconfigured!" when successful
- Puma master PID stays; only worker count changes
- GitLab is installed only in Docker on this box (no /etc/gitlab on host, no gitlab-ctl binary on host)
