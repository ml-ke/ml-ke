---
name: large-artifact-downloads
description: Fetch multi-GB files from slow or flaky hosts, verified.
---

# Fetching large artifacts reliably (firmware, ISOs, datasets, model weights)

Applies whenever the job is "pull a multi-GB file from a mirror/CDN and know when it actually
arrived intact" — including long background downloads the agent must monitor and cancel safely.

## 1. Measure before committing to the wait
```bash
curl -sSIL -A "Mozilla/5.0" "$URL" | grep -iE "content-length|accept-ranges"
```
Sample ~60 s of the real transfer and compute MB/min + ETA. "It is downloading" is not progress:
a host can move 3 MB/min while looking healthy, turning a 25-minute job into 7 hours. Report the
measured ETA to the user instead of a guess.

## 2. If one connection is slow and `Range` is advertised, parallelise
Single connections to ROM/mirror hosts commonly sit at ~3 MB/min. Split the file across `nproc`
concurrent `Range: bytes=start-end` requests (~14x measured on the same host), concatenate in
order, then hash the assembled file. Ready to run:
`scripts/parallel_range_download.py <base-url> <manifest.txt>` — manifest lines are
`<filename> <expected-md5|->`; it skips files already present and matching, so it is safe to
re-run after an interruption.

Always leave more than one download running concurrently only when the host tolerates it; a
FUS/CDN host and a mirror host are independent, the same host is not.

## 3. Verify before consuming, and again after moving it
The publisher's md5/sha256 is the acceptance test — compare it, and keep the result in a manifest
next to the files so re-runs are cheap. An unverified 5 GB blob is not a deliverable.
- **Find the manifest the publisher actually ships; never guess its filename.** Hosts commonly publish
  one `SHA256SUMS` / `checksums.txt` covering a whole directory instead of a per-file
  `<artifact>.sha512sum`, even when a companion install script still asks for the per-file name. A 404
  on the guessed name is **not** evidence of a corrupt download — fetch the directory listing or the
  aggregated manifest, verify against that, and only then conclude anything about integrity.
- **Re-verify at the destination.** A multi-GB copy onto a phone, remote host or second disk can
  silently not happen, and that failure is indistinguishable from a permissions error at the *next*
  step — which sends the user off debugging the wrong thing entirely. Hash the file where it landed
  (e.g. on-device `sha256sum`) and compare against the host value before telling anyone to proceed.

## Pitfalls that cost real time
- **Preallocated/sparse downloads make `ls` and `stat` lie.** A downloader that reserves the full
  size shows the final size immediately and `du -b` (apparent size) reports it too. Track real
  progress with `du -B1 <file>` / `du -sh` over time, or the downloader's own byte counter.
- **Compute progress against the real `Content-Length`, never a hand-rounded total.** Mixing GiB and
  MB or eyeballing the expected size yields nonsense percentages — a healthy ~78 MB/min transfer was
  reported as "3% complete, stalled" purely because the total was mistyped. Take the byte count from
  step 1, sample the size twice about a minute apart, and re-measure before calling a download stuck.
- **Never stop a download with `pkill -f <pattern>`.** The agent's own command line contains that
  pattern, so pkill matches itself and SIGTERMs the agent's shell — the turn dies with exit -15
  and the download is orphaned. Kill by PID, or make the pattern self-excluding:
  `ps -eo pid,cmd | grep -E "exampl[e]\.com" | awk '{print $1}'`. This bites repeatedly under time
  pressure; reach for the PID form first.
- **A privileged step needed mid-download belongs at the START, not the end.** Install udev rules,
  USB permissions, or credentials up front (check whether the host already has a mechanism before
  asking the user — this environment reads `SUDO_PASSWORD` from `~/.hermes/.env`, so
  `PW=$(grep -m1 '^SUDO_PASSWORD=' ~/.hermes/.env | cut -d= -f2-); echo "$PW" | sudo -S <cmd>`
  needs no interruption). A vendor-wide `TAG+="uaccess"` rule grants access only to the locally
  logged-in user, so it can still fail over SSH with no seat.
- **Don't let a slow download block the turn.** Run it with `background=true, notify=true`, and use
  the wait to do independent work (fetch the other artifacts, snapshot state, write the plan).

## Reporting the wait
State the measured rate, the percentage complete, the ETA, and exactly what you will do when it
lands. Never claim a file is ready before its checksum matches.
