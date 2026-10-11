---
name: android-device-firmware-update
description: Use when updating or flashing Android firmware from Linux.
---

# Android firmware update / flash, with data preservation (Linux host)

Covers: deciding between a stock reflash and a custom ROM, identifying the true hardware variant,
obtaining official firmware, preserving user data before anything destructive, and the pre-write
safety gate. Pair with `mvt-android-compromise-assessment` when the update follows a spyware check.

## Order of operations
1. Identify the hardware variant (below) — before choosing OR downloading any firmware.
2. Establish the software ceiling for that model+region.
3. Preserve data and tell the user exactly what cannot be preserved.
4. Build/obtain the tooling, then flash behind the safety gate.
5. Re-read the integrity props afterwards to confirm what actually changed.

## Step 0 — Identify the hardware variant first (non-negotiable)
The SoC is decisive and the build fingerprint can lie (it reflects whatever firmware is flashed).
```bash
for p in ro.board.platform ro.hardware ro.hardware.chipname ro.product.board ro.csc.sales_code ro.csc.country_code; do
  printf '%-28s ' $p; adb -s <serial> shell getprop $p; done
```
- `universal9810` / `samsungexynos9810` = Exynos; `sdm845` = Snapdragon. **Never flash another
  variant's firmware** — a mismatched-SoC bootloader cannot boot the device.
- Compare `ro.boot.bootloader` against `ro.build.PDA` / `ro.build.fingerprint`. A bootloader string
  naming a *different model variant* than the running build (e.g. a Korean `N960N…` string while the
  build and SoC are the international Exynos `N960F…`) is physically impossible as a real match and
  means **mixed/partial firmware** — flash a COMPLETE package (BL+AP+CP+CSC), never a bare AP.
- Record the tamper state in the same pass: `ro.boot.verifiedbootstate` (`orange` = bootloader
  unlocked), `ro.boot.flash.locked`, `ro.boot.veritymode` (empty = dm-verity not enforcing),
  `ro.boot.warranty_bit` (`1` = Knox fused), kernel string in `/proc/version` (a vendor build host
  string such as `dpi@SWDI…` confirms a stock kernel), and a write test into `/system`.
- An unlocked bootloader + verity off means **no ADB-visible check can vouch for the system image**.
  Say that instead of calling the device clean; a full stock flash is what closes it.

## Step 1 — Establish the ceiling for this model + region
- Find the vendor's support end and its LAST official build; it is often years old, and the device
  may be far behind even that.
- **Query the firmware service per CSC — never assume your device's CSC has the newest build.** On a
  single model, verified: one CSC stopped at an Aug-2021 build while another received the final
  Feb-2023 build. Run `check-update` across several plausible CSCs and pick deliberately.
- Custom ROM reality check before promising one: whether the port is still maintained (upstream might
  mark it "no longer maintained" while an unofficial thread is still actively built), and the ROM's
  stated base requirement — these often require **latest stock firmware**, which makes a stock flash a
  prerequisite rather than an alternative. Expect to lose vendor features and possibly VoLTE/VoWiFi
  (carrier-dependent), and expect a wipe: format `/system` + `/data` + `/cache` is typical.

## Step 2 — Tooling: use the maintained FUS client
`topjohnwu/samloader-rs` covers the whole job on Linux: `check-update`, `download`,
`flash` (Odin protocol — no Heimdall needed), `reboot-download` (no button combo),
`detect`, `dump-pit` / `print-pit`, `verify-md5`, `fix-usb` (installs udev rules; needs sudo once).
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
git clone --depth 1 https://github.com/topjohnwu/samloader-rs && cd samloader-rs
cargo build --release -p samloader      # NOT -p samloader2>&1 — that typo names a nonexistent package
./target/release/samloader check-update -m <MODEL> -r <CSC>
./target/release/samloader download -m <MODEL> -r <CSC> -d <dir> -j 8
```
- **Build pitfall:** HEAD fails to compile on Linux — `samloader/src/actions.rs` calls
  `libc::geteuid()` without the crate declared. Add `libc = "0.2"` to `samloader/Cargo.toml` and rebuild.
- `cargo build -p <name>` glued to a redirect (`-p samloader2>&1`) silently becomes package
  `samloader2` → "package ID specification did not match any packages". Put a space before `2>&1`.
- Older clients are dead ends: the legacy Python `samloader` and `SamFirm.js` both call a retired FUS
  endpoint and return HTTP 403. Vendor FOTA URLs (`…/firmware/<CSC>/<MODEL>/version.xml`) also 403 any
  non-Samsung client, UA spoofing included, and firmware mirror sites are JS/Cloudflare-walled — go
  straight to the maintained client rather than fighting those.
- A `download` preallocates the full file size up front, so `ls -la` shows the final size immediately
  while `du` shows real progress. Track `du`, not `ls`, and verify with `verify-md5` before flashing.

## Step 3 — Preserve data before anything destructive
Recipes and hard limits: `references/data-preservation-before-wipe.md`. Reusable exporter:
`scripts/export_sms_content_provider.py <serial> <outdir>`.
Always: pull the user's `/sdcard` directories, generate a sha256 manifest of the local backup, and
state plainly what CANNOT be preserved (see the reference — call logs are one).

### Build the restore kit in the same pass
A flash destroys state that cannot be re-imaged afterwards, so assemble the way back *before* writing:
- **No Nandroid without root.** `/dev/block/*` is `root:root 0600` and `ro.debuggable=0` blocks
  `adb root`, so an unrooted device cannot produce a partition-level image — never imply a "full
  backup" exists. The restore medium is the destination ROM's zip + recovery, plus the official
  firmware package as the route back to stock.
- **The kit is a directory, not a memory:** firmware zip, ROM/recovery/GApps artefacts
  (hash-verified), the official `.pit`, a pre-flash device-state snapshot (getprop dump + partition
  sizes), every script used, a `sha256sum` manifest of the lot, and a README giving the procedure per
  target. Park it outside the working directory so routine cleanup cannot delete it.
- **Trim re-creatable bulk** — an extracted firmware tree and `system.img.lz4` regenerate from the zip
  they came from (~9 GB reclaimed in one case); keep the zip itself.
- **Say what is unrecoverable:** an overwritten OS is gone as a bit-image, and a stock restore stays
  gated on the partition-size decision in `references/custom-rom-and-partition-sizing.md`. State both
  instead of implying the kit covers them.

## Step 4 — Pre-write safety gate (all three must pass)
```bash
samloader detect                         # a download-mode device is actually visible
samloader dump-pit /tmp/d.pit && samloader print-pit /tmp/d.pit   # real partition layout
samloader verify-md5 <firmware>          # integrity of what you are about to write
```
Treat the write as unverified until those three pass on THIS device — they are cheap and each one
fails loudly instead of half-writing a partition. If a write aborts with a partition-too-small error,
or the destination is a custom ROM, work from `references/custom-rom-and-partition-sizing.md`.
```bash
samloader flash -p <PARTITION> <file>    # explicit partition
samloader flash -f <file>                # auto-match partition from filename
samloader reboot-download                # enter download mode over ADB when USB ADB is alive
```

## Standing guardrails
- **Never** start a wipe or flash without a completed backup, a hash manifest, and the user's explicit
  confirmation of what will be lost. "Update it" is not consent to destroy data.
- Cross-CSC flashing changes the device's CSC and carrier configuration — disclose that before doing it.
- **Do not re-lock a bootloader** that has ever carried non-stock partitions unless the user accepts
  hard-brick risk; staying unlocked is the safer default and costs nothing already tripped (Knox).
- Flash with the device charged and the user available: download-mode entry, on-device authorization
  prompts and recovery-stage wipes need hands on the phone.
- Firmware choice is a user decision with real consequences (feature loss, wipe, carrier behaviour) —
  present the ceiling and the trade-offs, get the pick, then proceed.
- **Serial-qualify every command when more than one device is attached** (`adb -s <serial>`, AndroidQF
  `-c <serial>`). A wireless target (`ip:port`) plus a USB device (hex serial) in one session means a
  bare `adb shell` picks one arbitrarily and can mix two devices into one dataset — record the target
  serial next to the case/backup directory so a mix-up stays detectable.
- **One download-mode session per boot:** after an aborted or failed flash, further
  `print-pit`/`dump-pit`/flash attempts fail with `Unexpected handshake response` even though the
  device still enumerates as `04e8:685d`. The fix is a physical re-entry (hold Vol Down + Power ~7 s to
  exit, then back to download mode) — do not burn time retrying the handshake.
- **Working style for this user:** they ask for executive decisions on long hardware jobs — take the
  reversible/diagnostic path yourself and proceed, reporting the decision and its reason rather than
  asking. Interrupt only for physical actions (button combos, recovery taps, unlocking) and genuinely
  irreversible choices (`--repartition`, bootloader writes, re-locking). Deliver the report in chat
  *and* to Telegram with the documents attached.
- Run long jobs from a script file (`bash x.sh`) rather than a giant inline one-liner or heredoc:
  oversized inline payloads get mangled or rejected, and a mangled parse inside a pipeline returns
  *empty or wrong* output instead of erroring — sanity-check a parser against known-good output before
  trusting an empty result. Likewise `pkill -f <pattern>` matches the command line that issued it and
  kills your own shell; kill by PID or bracket the pattern.
- Observe on-device work you cannot otherwise see through `df -h /data` deltas — app data dirs are
  unreadable from adb, and `run-as` requires a debuggable build.
