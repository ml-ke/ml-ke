---
name: mvt-android-compromise-assessment
description: Check an Android phone for spyware over ADB with MVT.
---

# MVT Android compromise assessment (ADB-connected device)

## Version reality check first
`mvt-android check-adb` **was REMOVED** upstream (confirmed on MVT 2026.9.7 — the command
prints "REMOVED: Check an Android device over ADB"). Do not promise an ADB-only MVT run.
Live Android collection is now: **AndroidQF (separate binary) → `mvt-android check-androidqf`**.

Available commands on 2026.x: `check-androidqf`, `check-bugreport`, `check-backup`,
`check-intrusion-logs`, `check-iocs`, `download-iocs`, `version`.
Global flags must precede the subcommand: `mvt-android --disable-indicator-update-check check-androidqf ...`.
Putting the flag after the subcommand is a hard error (`No such option`).

## Step 1 — IOCs (needed for any matching)
```bash
timeout 600 mvt-android download-iocs   # writes ~/.local/share/mvt/indicators/*.stix2
ls ~/.local/share/mvt/indicators/ | wc -l   # ~19 bundles, ~11.5k unique indicators
```
Without this, MVT checks nothing meaningful and prints a banner nagging you to run it.

## Step 2 — collect with AndroidQF (NOT MVT)
Download the host binary from `mvt-project/androidqf` releases and **verify sha256 against
the release `checksums.txt`** before running. Flags are minimal: `-c ip:port -o OUT -m MODULE -l -v -f`.

### PITFALL: modules prompt interactively and silently die on non-TTY stdin
Three modules ask questions via promptui; with `< /dev/null` they abort with
`failed to make selection for ... : ^D` and **skip the module**, while the run continues
and looks fine:
- `backup` — "Would you like to take a backup of the device?" (needs on-device authorization anyway)
- `intrusion_logs` — "Would you like to download Intrusion Logs from the device?"
- `packages` — "download copies of all apps or only non-system ones?" → **this one is fatal to
the assessment: no `packages.json` is written at all** (IOC matching then has nothing to run on).

Run AndroidQF under a **PTY** and answer with arrow keys (`\x1b[B` = down) + Enter.
Prompt order/items for `-m packages`: Download = [All, Only non-system packages, Do not download any],
then (if not "none") Remove = [Remove trusted apps copies, Keep all].
"Do not download any" is the fast, complete option — the on-device collector still returns
md5/sha1/sha256/sha512 for every APK, so hash IOC matching still works; only APK copies,
certificate/signer verification and AQFFiles coverage are lost.

### Module order (v1.8.3) and cost
backup, intrusion_logs, packages, getprop, dumpsys, processes, services, bugreport, files,
settings, selinux, environment, root_binaries, mounts, logcat, logs, temp.
The `files` module (whole-filesystem listing) can run 15+ minutes with no output — run it
separately/last so it cannot stall the high-value modules. `bugreport` is worth its ~3 min
(20 MB zip → `check-bugreport` runs 16 more modules over it).

### PITFALL: the collector's package filters can fail → hundreds of bogus MEDIUM alerts
On some devices (seen on Samsung Android 10) AndroidQF's collector logs
`Failed to get packages filtered by '-d'/'-s'/'-3': exit status 1`, so **no package gets the
`system=True` flag** and `packages.json` reports 0 system / N non-system. MVT then emits one
MEDIUM per package: *"Found a non-system package installed via adb or another method"* —
hundreds of them, flagging `com.android.systemui`, `com.android.settings`, `com.android.phone`.
**Always verify before reporting:** compare the flagged set against live
`adb shell pm list packages -s` / `-3`. On a real case, 336 of 343 flags were system apps and
only 6 were genuine sideloads. Report the artifact, not the count.

### PITFALL: hash coverage can silently collapse
The same failures mean few APK paths are resolved — one case had hashes for only 12 of 61
third-party packages (and **none** for the sideloads, i.e. the apps that matter). MVT's hash
IOC matching is then largely untested. Close it yourself: `adb shell pm path <pkg>` →
`adb shell sha256sum '<path>'` for every third-party package, then intersect with the hashes
scraped from the STIX2 bundles (`file:hashes.…`). State the coverage number (e.g. "61/61
third-party APKs hashed, 0 matches vs 3,461 IOC hashes") — it is far stronger evidence than a
bare "no findings".

### Other real-device noise to triage rather than report
- `dumpsys_accessibility` flags every **declared** service; check
  `settings get secure enabled_accessibility_services` — often `null` (none active). List the
  notable third-party declarers instead (password managers, fitness bands).
- `settings` module: `send_security_reports = 0` / `samsung_errorlog_agree = 0` are
  anti-forensics *heuristics*; a privacy-minded owner sets these. Verify the value live and ask.
- MVT's closing banner ("MVT produced N MEDIUM / 0 HIGH / 0 CRITICAL alerts") is the cleanest
  source for severity totals — a `grep -i critical` hits that banner, so read it, don't count.
- `ro.boot.veritymode` empty + `verifiedbootstate=orange` + `flash.locked=0` on an
  Exynos/Samsung device means **no verified boot**: ADB-visible checks can all look stock
  (confirm the kernel via `/proc/version` — a stock `dpi@SWDI…` string is reassuring) while a
  pre-boot modification stays invisible. Report that limit explicitly; the fix is reflashing
  official firmware. Check `/data/adb` with a control test first — it exists on stock AOSP and
  proves nothing.

## Step 3 — analyse
```bash
mvt-android --disable-indicator-update-check check-androidqf -o <CASE>/mvt-results <CASE>/acq
mvt-android --disable-indicator-update-check check-bugreport -o <CASE>/mvt-results-br <CASE>/acq/bugreport.zip
```
A full device run is ~104 MB of artefact and produces `*/_detected.json` files only for
modules that have findings — absence of a file means clean, not skipped (verify against the
`Running module ...` lines in `command.log`).

## Step 4 — triage the alerts (do NOT report MVT output verbatim)
Known noise on modern Android:
- **`mounts` HIGH "system partition mounted rw"** — false positive on overlayfs. Confirm with
  live `mount | grep overlay` (expect `(ro,...)` + `lowerdir=`), a write test into `/system`
  (expect *Read-only file system*), and `ro.boot.verifiedbootstate=green` /
  `ro.boot.vbmeta.device_state=locked` / `ro.boot.veritymode=enforcing`.
- **`dumpsys_appops` MEDIUM for `com.android.shell`** — that is your own ADB session (timestamps
  match your collection window).
- **`dumpsys_adb_state` LOW trusted-host-key** — the examiner's host key.
- **`dumpsys_battery_daily` MEDIUM uninstalls** — daily buckets are cumulative, so N rows ≈ 1 event.
  Cross-check install/uninstall dates in `mvt-results/timeline.csv`.
- Any `frida|magisk|xposed|substrate|cydia` string in `dumpsys.txt` is usually an app's
  anti-tamper `queriesPackages=[...]` probe list, not installed tooling — grep the context line.

Always corroborate with an independent script: parse the STIX2 bundles yourself and intersect
package names + hashes with `packages.json`. Also re-hash 2-3 APKs on-device (`adb shell
sha256sum '<apk path>'`) to prove acquisition integrity.

## Step 5 — Android 16 Intrusion Logs (AAPM devices)
**Check whether logs can even exist before promising this artefact.** Intrusion Logging only
produces logs starting 24 h *after* the "Intrusion logging" toggle was enabled, and Google does
not backfill, so on a first assessment it is usually empty. The retrieval screen says so itself:
*"Logs will become available 24 hours after you turn on intrusion logging"* and *"No logs available
for this Google Account"*. If that is what you see, report IL as **unavailable** rather than pending —
it cannot cover any pre-enablement period, ever.

AndroidQF's `intrusion_logs` module only works interactively and pulls from
`/sdcard/Download/Intrusion Logging/`. Equivalent manual flow:
```bash
adb -s <ip:port> shell am start -n com.google.android.gms/.intrusiondetection.ui.retrieval.IntrusionDetectionRetrievalActivity
# ON DEVICE: scroll down -> Access Logs -> Download and Decrypt (per listed device)
adb -s <ip:port> pull "/sdcard/Download/Intrusion Logging/" <CASE>/intrusion_logs/
mvt-android check-intrusion-logs -o <CASE>/mvt-results-il <CASE>/intrusion_logs/
```
Always requires a human finger on the phone — get consent/timing before launching.

### Driving an on-device settings screen over ADB
- Read UI state authoritatively with `adb shell uiautomator dump /sdcard/ui.xml && adb pull` and
  grep `text=` / `checked=` — do **not** infer toggle state from a screenshot alone.
- `adb shell input swipe x y1 x y2` scrolls, but a swipe ending in the bottom gesture zone can
  trigger the home gesture instead; and a drag over a Switch may flip it. Never leave the device
  in a changed state you cannot account for: re-read the setting and tell the user explicitly if
  attribution is ambiguous.
- The screen reverts to the previous sub-page on re-open, so capture the toggle on first landing.

## Coverage gaps to state in every report
ADB-only collection never sees: network IOCs (domains/IPs need pcap/mitmproxy), SMS/MMS/call
logs or app data (need `adb backup` with on-device tap, or root), and the AQFFiles IOC-path
module unless the slow `files` module completes. Say so explicitly rather than implying the
device is "fully" checked.

## Device hygiene findings worth raising (non-malware)
- `settings get global adb_wifi_enabled` = 1 → wireless debugging left on with a trusted host key.
- `ro.build.version.security_patch` older than the current month by 2+ months.
- Third-party apps holding READ_SMS/RECEIVE_SMS (loan/fintech apps), READ_CALL_LOG on odd apps.
- Uninstalled-but-retained data (`pm list packages -u`) of banking/2FA apps.
- Extra profiles: `pm list users` (Private space / DualApps 999) — check `pm list packages --user <n>`
  for cloned apps that could hide a second instance.

## Reporting
Write to `~/Dev/REPORTS/<device>/<date>/MVT-REPORT.md` with: verdict, device+boot-integrity
profile, artefact table with sha256, per-finding severity table including an explicit
"FALSE POSITIVE" assessment, independent corroboration, coverage gaps, and copy-paste next steps.
Include the `1970-01-01` timeline-timestamp caveat when quoting `timeline.csv`.
