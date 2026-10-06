---
name: samsung-firmware-flash-linux
description: Update or reflash an EOL Samsung phone from Linux.
---

# Updating / reflashing a Samsung phone from Linux

For an abandoned (EOL) Samsung device, or one whose firmware is inconsistent (unlocked
bootloader, cross-flashed partitions). Covers fetching **official** firmware, the region trap,
and installing a maintained custom ROM, plus how to preserve data first.

## 1. Establish the device's real state BEFORE choosing firmware
```bash
adb shell getprop ro.board.platform ro.hardware ro.product.model   # SoC decides the firmware family
adb shell getprop ro.build.fingerprint ro.build.version.security_patch ro.build.PDA
adb shell getprop ro.boot.verifiedbootstate ro.boot.flash.locked ro.boot.veritymode
adb shell getprop ro.csc.sales_code                                  # region, decides which CSC to query
```
- **SoC is the trap.** Model strings come from the *flashed* firmware, so they can lie on a
  cross-flashed device. `ro.board.platform`/`ro.hardware` is hardware truth: e.g.
  `universal9810`/`samsungexynos9810` = Exynos (SM-N960F) — a Korean `N960N…` bootloader string
  cannot be real on Exynos hardware (different SoC), so it indicates a mixed/partial flash.
- `verifiedbootstate=orange` + `flash.locked=0` + empty `ro.boot.veritymode` = **unlocked, no
  verified boot**. ADB checks can all look stock (confirm kernel via `/proc/version` — a stock
  `dpi@SWDI…` hostname is reassuring) but a pre-boot modification stays invisible. Reflashing
  official firmware is the only real fix.
- `ro.boot.warranty_bit=1` = Knox permanently tripped (Samsung features already gone).

## 2. Fetch official firmware with samloader-rs (the only client still working)
Both `samloader` (Python, nlscc) and `samfirm` / SamFirm.js (npm) now die with
`403` on `fota-cloud-dn.ospserver.net/.../version.xml` — Samsung locked that endpoint down.
`topjohnwu/samloader-rs` speaks the current protocol AND flashes:

```bash
export PATH="$HOME/.cargo/bin:$PATH"      # rustup via: curl https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
git clone --depth 1 https://github.com/topjohnwu/samloader-rs && cd samloader-rs
# UPSTREAM BUG (HEAD, 2026-09): samloader/src/actions.rs uses libc::geteuid() but libc is not a
# declared dependency -> E0433. Fix: add `libc = "0.2"` to samloader/Cargo.toml, then:
cargo build --release -p samloader       # NOTE: `-p samloader2>&1` glues the redirect on; keep the space
```
Subcommands: `check-update`, `download`, `flash`, `reboot-download` (no button combo needed),
`detect`, `dump-pit`, `print-pit`, `verify-md5`, `fix-usb` (adds udev rules; needs sudo).

**The region trap:** query several CSCs — the *same* model gets different final builds per region.
Real example: SM-N960F `DBT` (its own CSC) ended at `N960FXXU9FVH1` (Aug-2021) while `BTU` got the
final surprise `N960FXXSAFWB3` (patch 2023-02-01). Query BTU/DBT/XEF/EVR before concluding what
the ceiling is.
```bash
./target/release/samloader check-update -m SM-N960F -r BTU   # -> PDA/CSC/MODEM/PDA
./target/release/samloader download -m SM-N960F -r BTU -d ~/firmware -j 8
```
Downloads are **sparse/preallocated**: `ls` shows the full target size immediately and `du -b`
lies. Track real progress with `du -B1 <file>` (or `du -sh`) over time — FUS throttles around
~100 MB/min in practice, so a 5 GB package takes ~45 min.

## 3. Which target?
- **Official final** = the newest build any CSC ever got. Keeps Samsung features; for a 2018
  device this is still years behind today.
- **Custom ROM** = only route to *current* patches. LineageOS's official `crownlte` support is
  unmaintained, but unofficial exynos9810 builds (S9/S9+/Note9) are actively maintained by
  `krazey90` and hosted on **images.krazey.de** (threads on XDA are Cloudflare/bunny-walled to
  curl and to headless browsers — read them through `https://r.jina.ai/<full-thread-url>`).
  Artifact naming: `lineage-<ver>-<date>-unofficial-<device>.zip`,
  `...-recovery-<device>.img` (heimdall/fastboot), `...-recovery-<device>.tar` (Odin), plus
  MindTheGapps in the same folder. Verify the thread's published md5s after download.
- ROM requirements, in its own words: **latest stock Samsung firmware**, **Lineage Recovery
  only** (no TWRP), then `Format /system + /data + /cache → Flash → Reboot`. So the stock flash
  is step one even when the destination is a custom ROM.
- Tell the user what they lose: Samsung features, possible VoLTE/VoWiFi loss (carrier-dependent),
  unofficial-build risk.
- Post-flash bootloader decision: **stay unlocked** (safer, avoids bricking on a non-stock
  partition) vs re-lock (green verified boot, but risky).
- DevBase-style ROMs state "BL update not necessary if you're already on a Q bootloader" —
  i.e. any Android-10-era bootloader can run these ROMs, but a *mismatched* bootloader is a
  reason to flash the full stock package first.

## 4. Preserve data first (non-rooted, Android 10-era)
- **Files:** plain `adb pull` of `/sdcard/{DCIM,Download,Pictures,Samsung,Telegram,...}`, then a
  local sha256 manifest.
- **SMS: `adb backup` will NOT capture it** when the messaging app lacks ALLOW_BACKUP
  (check `dumpsys package <pkg> | grep -i flag` — Samsung Messages has no ALLOW_BACKUP flag).
  The shell user CAN read the provider directly instead:
  ```bash
  adb shell content query --uri content://sms \
    --projection _id,thread_id,address,date,date_sent,type,read,body
  ```
  Parse by splitting records on `^Row: N ` and treating everything after ` body=` as the message
  text, so commas/newlines inside bodies don't corrupt the fields.
- **Call logs are properly locked down:** `content://call_log/calls` raises SecurityException for
  uid 2000 (shell lacks READ_CALL_LOG). No root, no export — say so rather than implying it worked.
- `pm list packages -u` reveals **uninstalled-but-retained** apps (their data survives), and if
  the owner removed banking/2FA apps for security reasons the retained data should be purged.

## 5. Flash sequence (verify each step on the real device)
```bash
./target/release/samloader fix-usb                 # once, if USB permissions bite (sudo)
adb reboot download                                # or: samloader reboot-download
./target/release/samloader detect                  # wait for the download-mode device
./target/release/samloader flash -f BL.tar.md5 AP.tar.md5 CP.tar.md5 CSC.tar.md5   # auto-match, or -p PARTITION file
```
Use `CSC` for a clean wipe, `HOME_CSC` to preserve data. Unzip the firmware package first
(`verify-md5` the tar.md5s). Then flash recovery (`-p RECOVERY recovery.img`), `adb reboot
recovery`, wipe, and `adb sideload` the ROM zip. Expect a few on-screen taps in recovery — it
has no shell.

### One download-mode session per boot
`adb reboot download` gives a working Odin session **once**. After an aborted/failed flash, further
`print-pit`/`dump-pit`/flash attempts fail with `Unexpected handshake response` even though the
device still enumerates as `04e8:685d`. The fix is a physical re-entry (hold Vol Down + Power ~7 s
to exit, then reboot to download mode again). Don't burn time retrying the handshake.

## 6. When the stock flash aborts: "X partition is too small for given file"
That is **pre-flight validation** — nothing was written, and a device that boots afterwards proves it
(compare `ro.boot.bootloader`, `ro.build.version.security_patch`, `ro.boot.verifiedbootstate` against
a pre-flash snapshot). It means the device was **repartitioned** by a previous owner (common on a
second-hand unlocked device). Diagnose with real numbers before deciding:

| What you need | How to get it |
|---|---|
| Device's real partition sizes | `adb shell df -h /system /vendor /odm /cache` — `/proc/partitions` and `/sys/class/block/*/size` are **blocked** for the shell user on Android 10+ |
| Firmware's required size | read the LZ4 frame header of `system.img.lz4` inside the AP package: magic `0x184D2204`, and if FLG bit 3 is set the 8 bytes at offset 6 are the uncompressed size |
| Official layout | the `.pit` ships **inside the CSC tar** (`tar -xf CSC*.tar.md5 --wildcards '*.pit'`), then `heimdall print-pit --file <pit>` (`heimdall-flash` from apt) |
| Custom ROM's requirement | `unzip -p ROM.zip system.transfer.list` → line 2 is the block count × 4096 B |

PIT parsing gotcha: in heimdall's dump each entry is `Partition Block Count:` **then** `Partition Name:`.
Pairing a name with the *following* count silently shifts every partition by one (a bogus
"675 MB SYSTEM" sent me down the wrong path). Parse line-by-line as a state machine, not with a
loose regex over the whole file.

**Stock can be too big while LineageOS fits easily.** Real case: stock system.img 4.43 GiB vs a 4.2 GiB
partition (aborts) while LineageOS needed only **1.897 GiB** → no repartitioning required. Check the
ROM's transfer list before considering `--repartition`, which is the one operation that can hard-brick.
You can also skip the stock flash entirely for a LineageOS destination: the ROM only needs a Q-era
bootloader (`ro.boot.bootloader` from any Android-10 stock build), and the ROM ships its own vendor
image. Skipping it also avoids the most brick-prone write there is — the bootloader — which is a
reasonable trade: a mismatch between the bootloader string and `PDA` is cosmetic if the device boots.

## 7. LineageOS install over sideload (what actually works)
1. `flash -p RECOVERY lineage-*-recovery-<device>.img` — partition-only, no stock parts needed.
2. `adb reboot recovery`. In recovery the device shows as **`unauthorized`** (or vanishes from
   `adb devices`) because `/data` is still FDE-encrypted so recovery's adbd cannot read the
   authorized-keys file. That is normal and **not** an error.
3. User: `Factory reset -> Format data/factory reset`, then `Apply update -> Apply from ADB`.
4. `adb sideload ROM.zip`. **Sideload mode needs no RSA authorization**, which is why it works on
   encrypted /data. Success looks like `Total xfer: 1.00x`.
5. **Wait ~60 s after the transfer before doing anything else.** `1.00x` means bytes delivered, not
   install finished; a second sideload fired immediately gets `adb: failed to read command: Success`
   and the recovery shows `Failed: broken pipe`. Then GApps (its own `Apply from ADB` selection),
   then `Reboot system now`.
6. A watcher loop that auto-sideloads must require the device to **leave and re-enter** sideload state
   between pushes — polling for `adb devices` containing `sideload` alone will fire into the
   still-writing session.

Recovery menu order matters for hand-held instructions: `Reboot system now` is the **first** item, so
"select Apply update" is easily mis-clicked. Say "press Volume Down once, then Power".

### After a data wipe, ADB is gone — plan for it
Formatting `/data` erases the ADB keys and resets Developer Options, so the freshly booted system is
invisible to adb and shows only as MTP (`04e8:6860`). MTP storage also lists **empty** until the phone
is unlocked. To verify the install you need the user to enable USB debugging
(Settings -> About phone -> tap Build number ×7 -> System -> Developer options -> USB debugging).
Verify by measurement: `ro.lineage.version`, `ro.build.version.security_patch`,
`ro.crypto.type` (should flip `block` → `file`), `getenforce`, and `df -h /system` to see the space freed.

## Reporting to the user
State plainly: what was verified vs assumed, the ETA of any download, and which steps need their
hands (unlocking, recovery taps). Never claim a flash succeeded without re-reading the device
state afterwards (`ro.build.version.security_patch`, `ro.boot.verifiedbootstate`).
