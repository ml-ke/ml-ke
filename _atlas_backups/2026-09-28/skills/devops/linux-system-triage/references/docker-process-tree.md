# Tracing Processes Across Container Boundaries

## When to Use

When `ps aux` shows a process that:
- Isn't managed by systemd (`systemctl status <name>` returns "not-found")
- References paths that look like they belong in a container (`/var/opt/gitlab/`, etc.)
- Is `k3s server` but `which k3s` returns nothing
- Runs alongside `containerd-shim-runc-v2` processes

## The Technique

Trace the parent PID chain from the suspicious process up to PID 1 (init/systemd) to find where it originates.

### Step 1: Find the Process

```bash
ps aux --sort=-%cpu | head -10
# Note the PID of the suspicious process
```

### Step 2: Walk the Parent Chain

```bash
PID=<target_pid>
while [ $PID -ne 1 ] && [ -d /proc/$PID ]; do
  cat /proc/$PID/status | grep -E 'Name|Pid|PPid'
  echo "  CMD: $(cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' ')"
  echo "  EXE: $(readlink -f /proc/$PID/exe 2>/dev/null)"
  PID=$(cat /proc/$PID/status | grep PPid | awk '{print $2}')
  echo "  ↑ parent (PPID=$PID)"
done
```

### Step 3: Identify the Container

If a process in the chain is `containerd-shim-runc-v2 -namespace moby -id <container_hash>`, the target process is running **inside a Docker container**.

```bash
# Find the Docker container by hash prefix
docker ps | grep <hash_prefix>

# Or inspect with full hash
docker inspect <full_hash> --format '{{.Name}} {{.Config.Image}}'
```

### Step 4: Confirm What Runs Where

```bash
# Processes inside the container
docker exec <name> ps aux | head

# vs processes visible from the host
ps aux | grep <process_name>
```

## Real Example: k3s in NemoClaw Container

This is the exact trace from a session where k3s was suspected to be running standalone on the host:

```
PID 2980 (k3s server)
  → PPID 2769 (/bin/k3s init)
    → PPID 2724 (/usr/bin/containerd-shim-runc-v2 -namespace moby -id 9dd0d89b...)
      → PPID 1 (/sbin/init)

Matched to Docker container:
  docker ps | grep 9dd0d89b
  → openshell-cluster-nemoclaw (nemoclaw-cluster:0.0.36)
```

**What this means:** k3s is NOT installed on the host. It runs inside the NemoClaw Docker container as the Kubernetes backbone for NVIDIA OpenShell sandbox orchestration.

## Pitfalls

- **`which k3s` returns empty** — does NOT mean k3s isn't running. It may be running inside a container.
- **Host `ps aux` shows container processes** — Docker containers expose processes to the host's process list. This doesn't mean they're host-native services.
- **`systemctl is-enabled <name>` returns "not-found"** for a process that's clearly running → it's inside a container.
- **`/proc/<pid>/exe` may be `(deleted)`** — the binary was removed/replaced after start. The cmdline in `/proc/<pid>/cmdline` is more reliable.
- **Multiple containerd-shim processes = multiple containers** — count them to estimate how many containers are running.
