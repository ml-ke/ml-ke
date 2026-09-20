# Custom-ROM install and partition sizing (when the stock flash aborts)

## The abort, and what it does not mean
`ERROR: <PARTITION> partition is too small for given file.` is **pre-flight validation** — nothing was
written. Prove it rather than asserting it: boot the device and compare `ro.boot.bootloader`,
`ro.build.PDA`, `ro.build.version.security_patch` and `ro.boot.verifiedbootstate` against the pre-flash
snapshot. The cause is that a previous owner **repartitioned** the device (common on second-hand,
unlocked hardware).

## Get real numbers before deciding anything
| What | How |
|---|---|
| Device's actual partition sizes | `adb shell df -h /system /vendor /odm /cache` — `/proc/partitions` and `/sys/class/block/*/size` are blocked for the shell user on Android 10+ |
| Firmware's required size | LZ4 frame header of `system.img.lz4` inside the AP package: magic `0x184D2204`, and if FLG bit 3 is set the 8 bytes at offset 6 hold the uncompressed size |
| Official layout | the `.pit` ships inside the CSC tar (`tar -xf CSC*.tar.md5 --wildcards '*.pit'`), then `heimdall print-pit --file <pit>` (`heimdall-flash` from apt) |
| Custom ROM's requirement | `unzip -p ROM.zip system.transfer.list` → line 2 is the block count × 4096 B |

**PIT parsing gotcha:** in heimdall's dump each entry is `Partition Block Count:` and *then*
`Partition Name:`. Pairing a name with the *following* count silently shifts every partition by one
entry — a bogus "675 MB SYSTEM" sent one diagnosis down the wrong path. Parse line-by-line as a state
machine, never with a loose regex over the whole file, and sanity-check the parsed total against the
device's `df` before trusting it.

## Resolution: check the custom ROM before even considering --repartition
Stock can be far too large while a custom ROM fits easily. Measured case: stock `system.img` 4.43 GiB
against a 4.2 GiB partition (aborts) while LineageOS needed only **1.897 GiB** → no repartitioning
required. `--repartition` is the one operation that can hard-brick, so it is a user decision, never an
automatic fallback. Skipping the stock flash entirely for a custom-ROM destination also avoids the most
brick-prone write there is — the bootloader; a bootloader/PDA string mismatch is cosmetic when the
device boots and the ROM ships its own vendor image.

## LineageOS-style sideload install — what actually works
1. `samloader flash -p RECOVERY lineage-*-recovery-<device>.img` — partition-only, no stock parts needed.
2. `adb reboot recovery`.
3. In recovery the device shows as **`unauthorized`** (or drops off `adb devices`) because `/data` is
   still FDE-encrypted, so recovery's adbd cannot read the authorized-keys file. That is normal and
   **not** an error.
4. User: `Factory reset → Format data/factory reset`, then `Apply update → Apply from ADB`.
5. `adb sideload ROM.zip`. Sideload needs no RSA authorization, which is why it works against an
   encrypted `/data`; success reads `Total xfer: 1.00x`.
6. **Wait ~60 s after the transfer before the next action.** `1.00x` means bytes delivered, not install
   finished; firing the next sideload immediately gives `adb: failed to read command: Success` and
   `Failed: broken pipe` in recovery. Then GApps (its own `Apply from ADB` selection), then
   `Reboot system now`.
7. A watcher loop that auto-sideloads must require the device to **leave and re-enter** sideload state
   between pushes — polling for `sideload` in `adb devices` alone fires into the still-writing session.
8. Recovery menu order matters for hand-held instructions: `Reboot system now` is the **first** item, so
   "select Apply update" is easily mis-clicked. Say "Volume Down once, then Power".

### After the data wipe, ADB is gone — plan for it
Formatting `/data` erases the ADB keys and resets Developer Options, so the freshly booted system is
invisible to adb and appears only as MTP (`04e8:6860`); MTP storage also lists empty until the phone is
unlocked. The user must re-enable USB debugging (Settings → About phone → tap Build number ×7 → System
→ Developer options → USB debugging). Then verify by measurement: `ro.lineage.version`,
`ro.build.version.security_patch`, `ro.crypto.type` (flips `block` → `file`), `getenforce`, and
`df -h /system` to see the space freed.
