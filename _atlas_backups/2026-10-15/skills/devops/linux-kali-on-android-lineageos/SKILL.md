---
name: linux-kali-on-android-lineageos
description: Run Linux/Kali on an Android phone, rootless or NetHunter.
---

# Linux (Kali) on an Android phone running LineageOS

Decide the tier first — they are different products with different requirements:

| Tier | Needs | Gives | Doesn't give |
|---|---|---|---|
| **Rootless Kali (NetHunter rootless)** | Termux only, no root | nmap(-sT), sqlmap, metasploit, hydra, hashcat(CPU), aircrack-ng tooling, KeX desktop | monitor mode, packet injection, tcpdump/ARP spoof, HID, raw sockets |
| **Full NetHunter** | root (Magisk) + a NetHunter **kernel** + usually an **older LineageOS base** | injection, external Wi-Fi adapters (OTG), HID, capture | nothing much, but costs ROM currency + brick risk |

Say this out loud to the user rather than implying "Kali" means injection out of the box. On a
device whose ROM is current (e.g. LineageOS 23.2 / Android 16), NetHunter kernels usually exist
only for older branches (LOS 20 era). Check `nethunter.kali.org/kernels.html` and GitHub for the
*codename* (e.g. `crownlte` for Note 9, `starlte`/`star2lte` for S9/S9+) before promising anything.

## Termux install (the pitfalls)
- Get the APK from **F-Droid** (`https://f-droid.org/repo/com.termux_<versionCode>.apk`). The
  versionCode is on the package page; GitHub releases lag behind F-Droid (e.g. GitHub's latest was
  v0.118.3 while F-Droid ships 0.119.x).
- **Verify the download before installing.** A truncated pull installs as
  `INSTALL_PARSE_FAILED_NOT_APK: Failed to load asset path .../base.apk`. If `curl -w` prints no
  http/size line, suspect a silent failure; validate with `unzip -t` and check the MIME type
  (`application/vnd.android.package-archive`) and that it's ~115 MB for the universal build.
- `adb install -r <apk>` works with USB debugging on LineageOS; no extra toggle needed.

## Pre-push the rootfs instead of downloading it on the phone
Kali's rootless installer **reuses a rootfs archive already present in the CWD** (it asks
"Existing image file found. Delete and download a new one?" — answer N). So:
1. On the host: fetch `https://kali.download/nethunter-images/current/rootfs/kali-nethunter-rootfs-<variant>-arm64.tar.xz`
   (+ `.sha512sum`). Variants: **full 1.64 GiB**, minimal 130 MB, nano 180 MB. The host has Cloudflare
   in front — it does support `accept-ranges`, but a single connection ran ~78 MB/min, which is fine.
2. `adb push` it to `/sdcard/Download/`. Apps cannot be written into Termux's data dir from adb, so
   shared storage is the only route in.
3. On the phone: `termux-setup-storage` (tap Allow) then a script that copies it into `$HOME`, fetches
   the installer (`https://offs.ec/2MceZWr` → Kali GitLab raw), and runs it with answers piped:
   `printf '1\nN\nN\n' | ./install-nethunter-termux` (1 = full, N = keep existing archive, N = don't delete).

Usage after install: `nethunter` (user), `nethunter -r` (root in container), `nethunter kex passwd` +
`nethunter kex &` for the desktop (needs the NetHunter-KeX app as the VNC client).

## Plain distros without Kali branding
`pkg install proot-distro && proot-distro install debian` for general Linux. Note the official
`proot-distro` has **no kali plugin** (its `distro-plugins/kali.sh` 404s) — but it can install any
rootfs tarball from a URL (`proot-distro install --name kali <url>`).

## Rootless NetHunter on this phone: what actually breaks (verified crownlte/LOS 23.2)

The stock rootless install leaves three things broken. All three are fixed on disk — the chroot
files are owned by the Termux app uid, so you can edit them from `adb shell`/Termux without proot.

1. **`nethunter` / `nh` hangs forever.** Upstream starts the container as
   `sudo -u kali /bin/bash`, and *any* setuid user switch (`sudo -u`, `su -`) inside proot never
   returns on this ROM: the process spins in **kernel** time (utime 0, stime climbing, state `R`)
   and **cannot be killed even with SIGKILL** — only a reboot clears it, and each attempt adds
   another 100% CPU burner. Fix: in `$PREFIX/bin/nethunter` set `start="/bin/bash --login"`
   (proot `-0` already presents fake-root, so nothing is lost). Keep a `nethunter.orig` backup.
2. **`nethunter kex start` hangs.** `/usr/bin/kex` calls interactive `vncpasswd` when
   `~/.vnc/passwd` is missing, and `vncpasswd` on a pipe dies with `getpassword error: ...`.
   Create the password non-interactively (`-f` reads stdin, min 6 chars):
   `printf 'yourpass\n' | vncpasswd -f > ~/.vnc/passwd && chmod 600 ~/.vnc/passwd`.
   Do it for both `$HOME=/root` and `$HOME=/home/kali`, since the file lives per-home.
   Patch `/usr/bin/kex` to always use display `:1` (port 5901, the KeX app default — upstream
   picks `:2`/5902 whenever `whoami` is root, which inside proot is always) and `-localhost yes`
   (the phone is on Wi-Fi/cellular; KeX connects over 127.0.0.1 anyway).
3. **KeX desktop is a single flat colour.** `$HOME/.vnc` in the kali home is a symlink to
   `.config/tigervnc`, which has no `xstartup`, so vncserver falls back to
   `/etc/X11/Xtigervnc-session`; and even the copied `startxfce4` xstartup leaves a black screen
   because `xfce4-session` starts but never spawns `xfwm4`/`xfdesktop`/`xfce4-panel` under proot.
   Write an xstartup that starts the pieces directly (dbus-launch once, then
   `xfsettingsd --daemonize`, `xfwm4 --replace`, `xfdesktop`, `xfce4-panel`, `wait`).

**Container has no DNS.** The rootfs ships a systemd-resolved `resolv.conf` pointing at a
nameserver unreachable from the phone, so every `apt-get` fails with `Temporary failure resolving`.
Write working nameservers into `<rootfs>/etc/resolv.conf` (`getent hosts http.kali.org` proves it).
`xfce4-terminal`, `xterm` are not in the rootfs — install them or the desktop has no terminal.

## Driving the phone from the host when nothing is debuggable
`adb root` is refused ("disabled by system setting") and Termux is not debuggable, so
`run-as` and direct reads of `/data/data/com.termux` are out. Reliable pattern:
- Write a script, `adb push` it to `/sdcard/Download/`, then type it into Termux:
  `am start -n com.termux/.app.TermuxActivity` → `input text "bash%s/sdcard/Download/x.sh"`
  (`%s` = space) → `input keyevent 66`. The script must redirect its own output to `/sdcard`,
  which adb *can* read.
- Long runs: make the script re-exec itself detached (`( nohup bash "$0" --worker >log 2>&1 & )`)
  and invoke it with a plain `bash <path>` — `/sdcard` is not executable, so `./x.sh` and bare
  `nohup x.sh` fail with `Permission denied`.
- `am force-stop com.termux` resets a wedged session (it does not kill the unkillable procs above).
- Always `unset LD_PRELOAD` at the top of any script that invokes `proot` directly: the
  termux-exec plugin makes proot fail with `execve("/usr/bin/env"): No such file or directory`
  plus `can't chmod .../usr/tmp/proot-*`. (The shipped `nethunter` launcher already unsets it.)
- `input keystroke KEYCODE_WAKEUP` + a `swipe` unlocks; set
  `settings put system screen_off_timeout 1800000` and `svc power stayon true` or long tests
  keep dying on the lock screen.
- To poke display `:1` from a *separate* `nh -r` session, export `XAUTHORITY=/home/kali/.Xauthority`
  or X clients fail with `Authorization required, but no authorization protocol specified`.

## Verifying the desktop without the app
`adb forward tcp:15901 tcp:5901`, then speak RFB from Python: read the 12-byte banner, send
`RFB 003.008\n`, pick VncAuth, answer the 16-byte challenge with two DES-ECB blocks computed via
`openssl enc -des-ecb -nopad -K <key> -provider legacy -provider default` (the key is the password
padded to 8 bytes with each byte bit-reversed). Then read a FramebufferUpdate and count distinct
pixel values: **1 colour = the desktop never painted**; dozens = xfce is really up. This proves
the password and the session independently of the KeX app.

## The NetHunter (com.offsec.nethunter) app without root
The GUI app is root-only: on a Magisk-less phone it retries `CheckForRoot.isRoot` every ~0.5s and
spams `E ShellExecuter` stack traces. Nothing in the rootless workflow needs it — KeX
(`com.offsec.nethunter.kex`) is a standalone bVNC client. Its saved connection is
`127.0.0.1:1` (display :1 = port 5901) and it stores the VNC password per connection, so a wrong
saved password shows up in the server log as `Authentication error: Authentication failed`, and
an un-painted desktop shows up as a black canvas.

## Installing tooling / a second Hermes inside the chroot (the pitfalls that cost hours)

- **Any `dpkg`/`apt` run can spawn `/usr/lib/cnf-update-db`, which spins forever in kernel time**
  (state `R`, ~100% CPU per instance, unaffected by SIGKILL, only a reboot clears it) and starves
  everything else on the phone. Neutralise the trigger once — copy it aside and replace the script
  with `#!/bin/sh` + `exit 0` at `<rootfs>/usr/lib/cnf-update-db`. The man-db / kali-menu triggers
  are harmless.
- **An `apt-get update` that stops right after `Hit: ... InRelease` is a blackholed mirror**, not a
  broken config. `http.kali.org` is a redirector; force IPv4 (`-o Acquire::ForceIPv4=true`) and
  retry — on a non-starved phone the same command finished in 3 s.
- **uv/pip installs fail with `failed to hardlink file ... .l2s.*: Operation not permitted`** —
  proot's `--link2symlink` rewrites `link()` into symlinks, which breaks uv's hardlink cache.
  `export UV_LINK_MODE=copy` is the fix (this is what made the Hermes install succeed).
- Useful packages: `git`, `gh`, `ripgrep` (all in Kali's repo once `update` works). When a package
  truly is not in the index, the GitHub `.deb` + `dpkg -i` route works and is fast (13 MB in ~9 s).
  Remember a `.deb` install runs dpkg triggers — see the cnf stub above.
- Hermes Agent in the chroot (root install → code in `/usr/local/lib/hermes-agent`, launcher
  `/usr/local/bin/hermes`, data in `$HERMES_HOME`):
  `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-setup --non-interactive --skip-browser --skip-computer-use --hermes-home /home/kali/.hermes`
  Since the patched `nethunter` launcher runs with `HOME=/home/kali`, `hermes` resolves its data
  automatically in a normal session; from `nethunter -r` export `HERMES_HOME=/home/kali/.hermes`.
- Cloning a Hermes instance onto the phone: `$HERMES_HOME/SOUL.md` is the always-loaded identity
  file (project context files like `AGENTS.md`/`.hermes.md` are cwd-scoped and wrong for this),
  plus `memories/MEMORY.md` and `memories/USER.md` for continuity, model settings via
  `hermes config set model.provider|model.default|model.base_url`, and the key in `$HERMES_HOME/.env`.
  Ship it as `tar czf` → `adb push` → extract straight into the chroot path from Termux (the chroot
  files are writable from Termux without proot). Leave `cron/` behind when the second instance is
  meant to have different duties. Verify with `hermes doctor` and one real `hermes chat -q` call.

## Wireless debugging (adb over Wi-Fi) on this phone
- Pair once from the Wireless debugging screen: `adb pair <ip>:<pairing-port> <code>`, then connect
  on the *connect* port — a different number, found with
  `nmap -Pn -p 30000-65500 --open <ip>` followed by `adb connect <ip>:<port>`.
- **The port changes every reboot, and adbd does not listen until the device is unlocked.** After a
  reboot: unlock the phone, wait ~1 min, rescan, reconnect. Assume this when a task needs reboots.

## What rootless genuinely cannot do
No kernel modules, no raw sockets, no `iptables` NAT, no tun/tap inside the container, and no injection
regardless of adapters — that is the Android sandbox, not a configuration error. State it up front so
the user doesn't spend an evening debugging a capability that requires the kernel tier.
