---
name: linux-system-triage
description: "Systematic Linux performance triage: diagnose high load, memory pressure, thermal throttling, and resource hogs, trace processes across container boundaries, and apply targeted fixes."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [troubleshooting, performance, system-administration, docker, process-tracing]
    related_skills: [systematic-debugging]
---

# Linux System Triage

## Overview

When a Linux system is slow, random restarts and vague cleanups waste time. This skill provides a structured approach to:

1. **Assess** — gather the right metrics in the right order
2. **Identify** — find the actual cause, not the symptom
3. **Fix** — apply the targeted correction
4. **Verify** — confirm the fix worked and the system recovered

## When to Use

- User says "system is slow" / "what's eating resources"
- High load average unexplained
- Apps responding slowly
- Thermal throttling suspected
- RAM exhausted, swap thrashing
- Investigating unknown processes (Docker containers, k3s, etc.)

## Quick Triage Commands

Run these in order. Each informs the next.

```bash
# 1. Memory overview
free -h

# 2. Swap status (high swap usage = memory pressure)
swapon --show

# 3. System load + uptime
uptime

# 4. CPU thermal throttling counters
for c in /sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count; do echo "$c: $(cat $c 2>/dev/null)"; done

# 5. Current CPU frequency (compared to max MHz from lscpu)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null | awk '{printf "%.0f MHz\n", $1/1000}'
lscpu | grep -E 'Model name|CPU max MHz'

# 6. Top CPU consumers
ps aux --sort=-%cpu | head -15

# 7. Top memory consumers
ps aux --sort=-%mem | head -15

# 8. Disk usage
df -h /
```

## Classic Red Flags

### 1. kswapd0 at high CPU

`kswapd0` is the kernel swap daemon. If it's consuming 20%+ CPU, the system is **constantly swapping** — RAM is exhausted. Check `free -h` for low available RAM + high swap usage.

**Fix:** Find the memory hog(s) via `ps aux --sort=-%mem`, reduce or eliminate them.

### 2. Thermal Throttling

Check `/sys/devices/system/cpu/cpu*/thermal_throttle/` counters. If counts are > 0, the CPU has been forced to slow down (often to 400-800 MHz vs 3-4 GHz max).

**Root causes:**
- Sustained high CPU load from multiple heavy services
- Physical heatsink / dust issue (hardware fix needed)
- Overheating because RAM thrashing → CPU busy → heat buildup

**Software fix:** Reduce CPU-intensive services first. If throttling persists after load drops, it's a hardware cooling issue.

### 3. System Load >> CPU count

Load average is a queue metric. If load is 260 on an 8-core system, that's 260 processes waiting for CPU. Typically caused by:
- kswapd0 thrashing (all processes blocked on I/O waiting for swap)
- Thermal throttling (CPU running at 1/4 speed, can't keep up)

### 4. GitLab Puma memory hog

GitLab Puma defaults to 8+ workers, each ~800MB RSS. On a 12GB system, Puma alone can consume 6+ GB. Target: 2 workers.

See reference: `references/gitlab-puma-tuning.md`

## Process Tree Investigation

When you see processes that look like they belong to a service but aren't managed by systemd, trace their parent chain to find the container or orchestration layer.

```bash
# Trace a process up the tree
PID=<pid>
while [ $PID -ne 1 ] && [ -d /proc/$PID ]; do
  cat /proc/$PID/status | grep -E 'Name|Pid|PPid'
  echo "  CMD: $(cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' ')"
  PID=$(cat /proc/$PID/status | grep PPid | awk '{print $2}')
  echo "  ↑ parent"
done

# Check if it's inside a Docker container
# The shim process will have -namespace moby -id <container_id>
ps aux | grep containerd-shim | grep -v grep

# Match container ID from process tree to Docker
docker ps --no-trunc | grep <container_id>
```

**Common patterns:**
- `containerd-shim-runc-v2 -namespace moby -id <hash>` → inside a Docker container
- `k3s server` spawned by `/bin/k3s init` from a containerd-shim → k3s runs inside a container (e.g., nemoclaw cluster)
- `puma` workers with paths like `/var/opt/gitlab/` → GitLab running in a container

## Docker Container Management

### Diagnose

```bash
# See resource usage per container (no —no-stream flag = one-shot)
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Inspect restart policy
docker inspect <name> --format '{{.Name}} {{.HostConfig.RestartPolicy.Name}}'

# See config, ports, and image
docker inspect <name> --format '{{.Name}} {{.Config.Image}} {{.State.Status}} {{range $p, $conf := .NetworkSettings.Ports}}{{$p}} -> {{(index $conf 0).HostPort}} {{end}}'
```

### Change Restart Policy

```bash
# Prevent auto-start at boot
docker update --restart=no <container>

# Or allow manual restart only
docker update --restart=unless-stopped <container>
```

### Stop / Prevent Auto-Start Chain

If Docker itself starts at boot (systemd):
```bash
systemctl is-enabled docker   # → "enabled"
systemctl is-enabled docker.socket
```
The chain is: **systemd → Docker daemon → restart policy → container**. Breaking the restart policy (above) stops the container from starting even when Docker auto-starts.

## Fix → Verify Pattern

After applying any fix, always verify:

```bash
# Wait for the service to settle (5-10 seconds)
sleep 10

# Re-check the metrics
free -h
uptime
ps aux --sort=-%mem | head -8

# Confirm the specific fix
# e.g., for Puma:
ps aux | grep 'puma.*worker' | wc -l

# e.g., for Docker restart policy:
docker inspect <name> --format '{{.HostConfig.RestartPolicy.Name}}'
```

## Section 6: Legacy APT Dependency Resolution

Some packages — especially Electron-based apps, older games, or unmaintained .deb releases — depend on **gconf2**, **libgconf-2-4**, or other GNOME2-era libraries removed from Ubuntu 24.04 (Noble) and later. These packages still exist in the **Ubuntu 22.04 (Jammy)** archive and can be installed manually.

### When to Use

`apt install` fails with:
```
Depends: gconf2 but it is not installable
Depends: gconf-service but it is not installable
Depends: libgconf-2-4 but it is not installable
```

### Workflow

**1. Diagnose the full dependency tree:**
```bash
apt-get --just-print install <package> 2>&1 | tail -30
```

**2. Check the archive for the missing packages:**
```bash
curl -sL "https://archive.ubuntu.com/ubuntu/pool/universe/g/gconf/" | grep -oP 'href="[^"]*\.deb"' | sort -u
```

**3. Download all missing packages (same version to avoid conflicts):**
```bash
cd /tmp
base="https://archive.ubuntu.com/ubuntu/pool/universe/g/gconf"
for pkg in \
  gconf2-common_3.2.6-7ubuntu2_all.deb \
  libgconf-2-4_3.2.6-7ubuntu2_amd64.deb \
  gconf-service_3.2.6-7ubuntu2_amd64.deb \
  gconf-service-backend_3.2.6-7ubuntu2_amd64.deb \
  gconf2_3.2.6-7ubuntu2_amd64.deb; do
  wget -q "$base/$pkg" -O "/tmp/$pkg"
done
```

**4. Check for transitive dependency mismatches:**
```bash
dpkg -l | grep -E "libdbus-glib|ucf|psmisc|python3"
# Also check libldap version — Jammy ships libldap-2.5-0, Noble has libldap2 v2.6.x
# If mismatch, download the Jammy version too:
# wget "http://archive.ubuntu.com/ubuntu/pool/main/o/openldap/libldap-2.5-0_2.5.20+dfsg-0ubuntu0.22.04.1_amd64.deb"
```

**5. Install all packages together (circular dependency: gconf-service ↔ gconf-service-backend):**
```bash
sudo dpkg -i /tmp/gconf2-common_3.2.6-7ubuntu2_all.deb \
             /tmp/libgconf-2-4_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf-service-backend_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf-service_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf2_3.2.6-7ubuntu2_amd64.deb
```

**6. Install the target package:**
```bash
sudo apt install -y <target-package>
```

### Pitfalls

- **Circular deps**: gconf-service ↔ gconf-service-backend. Install together in one dpkg call.
- **Architecture**: Download `_amd64.deb`, not `_i386.deb`.
- **Breaks/Conflicts**: Check `dpkg-deb --info <deb>` for Conflicts/Breaks fields.
- **Held broken packages**: Run `sudo apt --fix-broken install` after dpkg if apt marks them.
- **Removed in later releases**: Some legacy packages may not exist in Jammy either. Try AppImage or Flatpak.
- **Sudo required**: If the terminal tool cannot run sudo, see the **hermes-maintenance** skill (Section 2: Sudo Privilege Setup).

**Reference**: `references/balena-etcher-gconf-fix.md` — concrete worked example of resolving gconf deps for balena-etcher-electron on Ubuntu 24.04.

---

## Pitfalls

- **Docker stats shows container memory, not host memory** — Puma workers tracked via `ps aux` on the host show higher RSS than Docker stats because Docker counts shared pages once.
- **`docker exec` with `--no-stdin`** — only use `-i` if you need stdin. For config changes, `docker exec <name> sed ...` works fine.
- **GitLab reconfigure restarts Puma** — it uses phased restarts, so old workers coexist with new ones for ~10 seconds. Wait before counting workers.
- **`which k3s` may return nothing even though k3s is running** — it's inside a container, not on the host PATH. Trace the process tree instead.
- **Thermal throttle counters persist across reboots?** No, they reset on boot. Check uptime — if counters are >0 on a fresh boot, throttling is active NOW.
- **High load + low CPU usage** — usually means processes are blocked on I/O (swap). Check `kswapd0` CPU and swap usage.
