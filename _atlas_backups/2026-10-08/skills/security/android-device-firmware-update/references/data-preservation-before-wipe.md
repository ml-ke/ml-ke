# Preserving an Android device's data before a wipe or flash

Run this before any destructive step, and report to the user exactly what survived and what did not.

## 1. Pull the user-visible storage
```bash
adb -s <serial> shell "ls /sdcard/"                 # discover what exists first
adb -s <serial> shell "du -sh /sdcard/DCIM /sdcard/Download /sdcard/Pictures /sdcard/Documents 2>/dev/null"
for d in DCIM Pictures Download Documents Music Movies Podcasts Samsung Telegram; do
  adb -s <serial> pull "/sdcard/$d" "$DEST/" 2>&1 | tail -1   # missing dirs error harmlessly
done
```
Then hash-manifest the local copy so the backup is provable:
```bash
cd "$DEST" && find . -type f -print0 | sort -z | xargs -0 sha256sum > BACKUP-MANIFEST.sha256
```
Windows-style leftovers (`/sdcard/Telegram`, `/sdcard/DualApp`) often persist with the app already
uninstalled — they are data, not evidence of an installed app; check `pm list packages` before
calling them suspicious.

## 2. SMS — exportable, no root
`adb shell` holds READ_SMS, so the provider is readable. Use `scripts/export_sms_content_provider.py`.
Verify the count looks plausible and tell the user the real number (e.g. "2,502 messages, 2,398
inbox / 104 sent, <date> to <date>") rather than "SMS backed up".

## 3. Call logs — NOT exportable this way
`content query --uri content://call_log/calls` fails with
`SecurityException: … requires android.permission.READ_CALL_LOG or WRITE_CALL_LOG` for uid 2000.
Options are an installed app (e.g. an SMS/call-log backup app writing XML to `/sdcard`, which can also
restore on a custom ROM) or root. Either way it is a user decision — never silently skip it and never
imply it was captured.

## 4. `adb backup` is a trap for messaging data
`adb backup -apk -shared -all …` silently skips every package whose flags lack `ALLOW_BACKUP`, so a
"full backup" can contain zero messages. Check before relying on it:
```bash
adb -s <serial> shell "dumpsys package com.samsung.android.messaging | grep -i ALLOW_BACKUP"
```
No `ALLOW_BACKUP` in the flags line = excluded from the backup. It also requires the device unlocked
and a confirmation tap, and produces a `.ab` archive you still need an extractor for — prefer the
file pull + content-provider export above.

## 5. App data and everything else
- App-private data needs `adb backup` (allowBackup permitting), root, or a vendor transfer tool.
- Storage encryption state matters for what any of this can recover:
  `getprop ro.crypto.state` / `ro.crypto.type`.
- Record the device's identity alongside the backup (`ro.build.fingerprint`, serial, date) so a future
  session can prove which device a backup came from.

## 6. Report to the user
State: bytes/files preserved, the hash manifest path, message counts with date range, and an explicit
list of what could NOT be preserved plus the route that would capture it. "Backed up" without numbers
is how a user loses data while believing it is safe.
