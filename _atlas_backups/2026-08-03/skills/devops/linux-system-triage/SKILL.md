---
name: linux-system-triage
description: "Systematic diagnostics for Linux servers: internal performance triage (CPU, memory, thermal, processes) AND external site/infrastructure availability (DNS, domain registry, nameserver health, hosting, deployments). One umbrella for any 'something is wrong' scenario."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [troubleshooting, performance, system-administration, docker, process-tracing, dns, availability]
    related_skills: [systematic-debugging]
---

# Linux System Triage

## Overview

When a Linux system is slow or a website won't load, random restarts and vague cleanups waste time. This skill provides a structured approach to:

1. **Assess** — gather the right metrics in the right order
2. **Identify** — find the actual cause, not the symptom
3. **Fix** — apply the targeted correction
4. **Verify** — confirm the fix worked and the system recovered

Sections 1–6 cover **internal server health** (CPU, memory, thermal, processes, Docker, APT). Section 7 covers **external infrastructure diagnostics** (DNS, domain registry, nameservers, hosting, CI/CD deployments).

## When to Use

- User says "system is slow" / "what's eating resources"
- User says "site X is not loading" / "deployments aren't reflecting"
- High load average unexplained
- Apps responding slowly
- Thermal throttling suspected
- RAM exhausted, swap thrashing
- Investigating unknown processes (Docker containers, k3s, etc.)
- DNS resolution failures, domain expiry, nameserver health issues

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

## Section 7: External Site & Infrastructure Diagnostics

Investigate bottom-up (DNS → hosting → deployment) — an early layer failure cascades and makes higher layers look broken.

### 7.1 Direct Site Check

```bash
# Fast HTTP check
curl -svo /dev/null "https://example.com" 2>&1 | grep -E "< HTTP/|location|Could not resolve|Connection refused"

# Full content check
curl -sL "https://example.com" 2>/dev/null | head -50

# Timeout detection
curl -s --connect-timeout 5 -m 10 "https://example.com" >/dev/null 2>&1
echo "Exit code: $?"  # 6=DNS failure, 7=connection refused, 28=timeout
```

**Symptoms and first conclusions:**
- `curl: (6) Could not resolve host` → **DNS failure** (Layer 2)
- `curl: (7) Connection refused` → **Server down / not listening**
- `curl: (28) Connection timed out` → **Network path blocked** (firewall, routing)
- HTTP 200 but wrong content → **Parked page / DNS pointing elsewhere**
- HTTP 301/302 → **Check the redirect target**

### 7.2 DNS Resolution

Check every record type:

```bash
dig +short example.com A           # IPv4
dig +short example.com AAAA        # IPv6
dig +short example.com CNAME       # Canonical name (for subdomains)
dig +short example.com NS          # Nameservers
dig +short example.com SOA         # Start of authority — proves zone exists
dig +short example.com MX          # Mail (if relevant)
```

**Diagnostic rules:**
- **Empty A + AAAA** → domain has no IP mapping
- **Empty NS** → nameserver delegation is broken at the registry
- **Empty SOA** → authoritative nameservers don't have a zone configured (REFUSED)
- **A/AAAA resolves but wrong IP** → DNS points to old hosting or parked page
- **CNAME set** → must resolve to a hostname that has IPs (cannot coexist with other records at apex)

Also check propagation across resolvers:

```bash
dig @8.8.8.8 example.com A
dig @1.1.1.1 example.com A
dig @ns1.example-ns.com example.com A
```

### 7.3 Domain Registry / WHOIS

```bash
whois example.com | grep -E "Name Server|Expir|Status|Registrar|Creation Date"
```

**What to look for:**
- **Domain expired?** Check expiration date and current status (active/redemption/held)
- **Nameservers correct?** Compare registry vs. provider
- **Last changed date?** Recent change may indicate incomplete migration
- **Registry DNS diagnostics** — `.ke` domains have built-in DNS health check at kenic.or.ke (KENIC RDAP shows JSON report: `tldDns`, `soaStatus`, and final `error` field)

### 7.4 Nameserver Health

If the registry reports nameservers but they refuse or don't respond:

```bash
# Resolve nameserver hostnames to IPs
dig +short ns1.provider.com A

# Query each nameserver directly for the domain's SOA
dig @ns1.provider.com example.com SOA
dig @ns2.provider.com example.com SOA

# Check if the nameserver is even reachable
nc -zv ns1.provider.com 53 2>&1
```

**REFUSED vs. NXDOMAIN vs. timeout:**
| Response | Meaning |
|----------|---------|
| `REFUSED` | Nameserver received query but has no zone for this domain |
| `NXDOMAIN` | Zone exists but specific record doesn't |
| `timeout` / no response | Nameserver not reachable or not running |
| Valid SOA | Zone is healthy. Problem is elsewhere. |

### 7.5 Hosting Provider Check

Verify the hosting platform directly (bypassing the custom domain):

**GitHub Pages:**
```bash
# Pages config exists?
curl -s "https://api.github.com/repos/org/repo" | python3 -c "import sys,json; print('has_pages:', json.load(sys.stdin).get('has_pages', 'N/A'))"

# Direct Pages URL (bypasses custom domain)
curl -svo /dev/null "https://org.github.io/repo/" 2>&1 | grep -E "< HTTP/|location"
# 301 redirect to custom domain = Pages is live but redirecting
# 200 with content = Pages serving without custom domain
```

**Cloudflare Pages / Netlify / Vercel:**
```bash
curl -svo /dev/null "https://project.pages.dev" 2>&1 | grep "< HTTP/"
```

### 7.6 Deployment Pipeline

```bash
# GitHub Actions — latest runs
curl -s "https://api.github.com/repos/org/repo/actions/runs?per_page=3" | python3 -c "
import sys, json
for r in json.load(sys.stdin).get('workflow_runs', []):
    print(f\"  {r['status']:10s} conclusion={r.get('conclusion') or '-':10s} sha={r['head_sha'][:8]} created={r['created_at'][:19]} name={r['name']}\")
"
```

**Key checks:**
- Is the latest workflow run `completed` with `conclusion=success`?
- Were there warnings? (Node.js version deprecation — usually non-fatal)
- Was an artifact produced?

### 7.7 Verification After Fix

After making DNS changes, verify propagation:

```bash
dig +short example.com A
dig @8.8.8.8 +short example.com A
curl -sL "https://example.com" | grep -c "expected site keyword"
curl -svo /dev/null "https://example.com" 2>&1 | grep -i "ssl\|certificate\|TLS"
```

**GitHub Pages TLS:** Once DNS points A records to 185.199.108.153–185.199.111.153, GitHub auto-provisions a Let's Encrypt certificate within minutes. If you see TLS errors, wait 5–10 minutes.

**Comprehensive one-shot verification after user fix:** When the user says "check back in N hours" after applying a fix, create a comprehensive checker script that tests every layer (registry → DNS NS → A records → hosting direct → custom domain). Schedule it as a `no_agent=true` one-shot cron job, typically 3 hours out. See `scripts/check-all-milestones.sh` for a template (edit DOMAIN and EXPECTED_NS).

**Reference**: `references/ml-co-ke-case-study.md` — worked example of diagnosing a `.ke` domain with registry-level diagnostics (KENIC RDAP "SHOW DNS DATA" feature).

---

## Pitfalls

- **Docker stats shows container memory, not host memory** — Puma workers tracked via `ps aux` on the host show higher RSS than Docker stats because Docker counts shared pages once.
- **`docker exec` with `--no-stdin`** — only use `-i` if you need stdin. For config changes, `docker exec <name> sed ...` works fine.
- **GitLab reconfigure restarts Puma** — it uses phased restarts, so old workers coexist with new ones for ~10 seconds. Wait before counting workers.
- **`which k3s` may return nothing even though k3s is running** — it's inside a container, not on the host PATH. Trace the process tree instead.
- **Thermal throttle counters persist across reboots?** No, they reset on boot. Check uptime — if counters are >0 on a fresh boot, throttling is active NOW.
- **High load + low CPU usage** — usually means processes are blocked on I/O (swap). Check `kswapd0` CPU and swap usage.
- **Actions API is rate-limited** without authentication. For heavy debugging, use `gh` CLI or browser.
- **301 redirect from GitHub Pages to custom domain** means Pages IS working — the problem is DNS, not Pages.
- **The hosting provider's own domain** (e.g., `org.github.io/repo/`) may return a different result from the custom domain. This is diagnostic gold — compare them.
- **Registry-level DNS diagnostics** (like KENIC's "SHOW DNS DATA") are authoritative because they test from the TLD's perspective. REFUSED means DNS provider problem, not registry.
- **Don't confuse warnings with errors.** GitHub Actions Node.js deprecation warnings are yellow, not red. Builds succeed despite them.
