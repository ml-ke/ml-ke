# Balena Etcher - gconf dependency fix on Ubuntu 24.04

## Error

```
$ sudo apt install balena-etcher-electron
...
The following packages have unmet dependencies:
 balena-etcher-electron : Depends: gconf2 but it is not installable
                          Depends: gconf-service but it is not installable
                          Depends: libgconf-2-4 but it is not installable
```

Both `balena-etcher` and `balena-etcher-electron` (v1.7.9) have this issue. It affects all versions in the balena apt repo on Noble.

## Root cause

The gconf packages (`gconf2`, `gconf-service`, `libgconf-2-4`, `gconf2-common`, `gconf-service-backend`) were removed from Ubuntu 24.04 (Noble). They are legacy GNOME2-era config libraries that were deprecated years ago in favour of GSettings/dconf.

Balena has not updated their .deb packaging to remove these dependencies.

## Fix

### 1. Download all 5 gconf packages from Jammy (22.04)

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

### 2. Install all at once (circular dep: gconf-service ↔ gconf-service-backend)

```bash
sudo dpkg -i /tmp/gconf2-common_3.2.6-7ubuntu2_all.deb \
             /tmp/libgconf-2-4_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf-service-backend_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf-service_3.2.6-7ubuntu2_amd64.deb \
             /tmp/gconf2_3.2.6-7ubuntu2_amd64.deb
```

**If dpkg reports missing `libldap-2.5-0`**, download it from Jammy too:

```bash
wget -q "http://archive.ubuntu.com/ubuntu/pool/main/o/openldap/libldap-2.5-0_2.5.20+dfsg-0ubuntu0.22.04.1_amd64.deb" -O /tmp/libldap-2.5-0_amd64.deb
sudo dpkg -i /tmp/libldap-2.5-0_amd64.deb
sudo dpkg --configure -a   # finish configuring the gconf packages
```

Then proceed to step 3.

### 3. Install balena-etcher-electron

```bash
sudo apt install -y balena-etcher-electron
```

## Notes

- All 5 gconf packages use version `3.2.6-7ubuntu2` (last Jammy version). The versions on Jammy are binary-compatible with Noble's libc6 (2.35+), libdbus, libglib2.0, and libxml2.
- Base deps (`ucf`, `libdbus-glib-1-2`, `psmisc`, `python3`) are all present stock on Noble — no additional downloads needed.
- The repo at `https://dl.cloudsmith.io/public/balena/etcher/deb/ubuntu` was already configured; only the gconf deps were missing.
