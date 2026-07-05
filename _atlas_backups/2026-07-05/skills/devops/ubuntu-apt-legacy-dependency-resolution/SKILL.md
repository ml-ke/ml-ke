---
name: ubuntu-apt-legacy-dependency-resolution
description: >-
  Resolve APT dependency failures on modern Ubuntu (24.04+) where the target
  package depends on obsolete libraries (gconf2, libgconf-2-4, etc.) that were
  removed from the current release. Covers diagnosis, sourcing from older
  Ubuntu archive, and safe dpkg installation with circular-dependency handling.
trigger: >-
  When `apt install` fails with "Depends: X but it is not installable" and the
  missing package is a known-removed library (not a typo or missing repo).
---

# Ubuntu APT Legacy Dependency Resolution

Some packages — especially Electron-based apps like balena-etcher, older games, or unmaintained .deb releases — still depend on **gconf2**, **gconf-service**, **libgconf-2-4**, or other GNOME2-era libraries removed from Ubuntu 24.04 (Noble) and later.

These packages still exist in the **Ubuntu 22.04 (Jammy)** archive and can be installed manually to satisfy the dependency chain.

## Workflow

### 1. Diagnose the real dependency tree

Do not just run `sudo apt install` and read the error. Use the non-root simulation to see all deps at once:

```bash
apt-get --just-print install <package> 2>&1 | tail -30
```

This shows every missing dependency without needing sudo. Look for lines like:
```
Depends: gconf2 but it is not installable
Depends: gconf-service but it is not installable
Depends: libgconf-2-4 but it is not installable
```

### 2. Verify the missing packages exist in an older release

Check the archive directory for the source package:

```bash
# List all .deb files for the source package
curl -sL "https://archive.ubuntu.com/ubuntu/pool/universe/g/gconf/" | grep -oP 'href="[^"]*\.deb"' | sort -u
```

Pick the **same version** (e.g. `3.2.6-7ubuntu2`) across all sub-packages to avoid version mismatches.

### 3. Download the missing packages

Download from `archive.ubuntu.com` (not the live repos):

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

### 4. Check local deps are already satisfied

Before installing, check that the legacy packages' own dependencies are present on the current release:

```bash
dpkg -l | grep -E "libdbus-glib|ucf|psmisc|python3"
```

Common deps of gconf packages (`libdbus-glib-1-2`, `ucf`, `psmisc`, `python3`) are usually present on Noble — verify once.

**Watch for library version mismatches.** The gconf packages from Jammy may depend on specific library versions that aren't on Noble. For example, `gconf-service-backend` depends on `libldap-2.5-0 (>= 2.5.4)` — but Noble ships `libldap2` v2.6.10 instead. In that case, download the matching old library from the Jammy archive too:

```bash
# Find the right package
curl -sL "http://archive.ubuntu.com/ubuntu/pool/main/o/openldap/" | grep -oP 'href="libldap-2\.5-0[^"]*\.deb"' | head -5

# Download and install
wget -q "http://archive.ubuntu.com/ubuntu/pool/main/o/openldap/libldap-2.5-0_2.5.20+dfsg-0ubuntu0.22.04.1_amd64.deb" -O /tmp/libldap-2.5-0_amd64.deb
sudo dpkg -i /tmp/libldap-2.5-0_amd64.deb
```

The Jammy versions are binary-compatible with Noble's libc6, libgnutls30, and libsasl2 — these are typically already installed.

### 5. Install all legacy packages together

**Critical: circular dependencies** — `gconf-service` and `gconf-service-backend` depend on each other. Install them all in a single `dpkg -i` command:

```bash
sudo dpkg -i /tmp/gconf2-common_3.2.6-7ubuntu2_all.deb \
             /tmp/libgconf-2-4_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf-service-backend_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf-service_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf2_3.2.6-7ubuntu2_amd64.deb
```

### 6. Install the target package

```bash
sudo apt install -y <target-package>
```

The apt resolver should now find all deps satisfied.

## Pitfalls

- **Circular deps**: `gconf-service` ↔ `gconf-service-backend`. Installing them one-at-a-time with `dpkg -i` fails. Always pass all interrelated packages in one call.
- **Architecture mismatch**: Download `_amd64.deb` for x86_64 systems, not `_i386.deb`.
- **Breaks/Conflicts**: Check `dpkg-deb --info <deb>` for `Conflicts` and `Breaks` fields. The legacy packages may declare breaks against newer package versions — if so, `dpkg --ignore-depends` may be needed.
- **Held broken packages**: If apt marks them as held, run `sudo apt --fix-broken install` after dpkg.
- **Timeout on downloads**: `archive.ubuntu.com` is rate-limited. Use `wget -q` with retries (`--tries=3`) for unstable connections.
- **Removed in later releases**: Some legacy packages may not exist in Jammy either (e.g. very old GTK1 libs). In that case, the AppImage or Flatpak version of the target app is the better path.
- **Sudo requirement blocked?**: The `dpkg -i` and `apt install` commands in this workflow require `sudo`. If the terminal tool cannot run sudo interactively, see the `hermes-sudo-setup` skill (devops category) for the SUDO_PASSWORD .env mechanism and fallback flows.

## References

- `references/balena-etcher-gconf-fix.md` — concrete example of resolving gconf deps for balena-etcher-electron on Ubuntu 24.04
