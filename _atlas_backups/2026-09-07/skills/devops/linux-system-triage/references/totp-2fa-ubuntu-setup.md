# TOTP 2FA on Ubuntu Desktop (sudo + GDM login)

Researched 2026-06 for the user's Ubuntu 24.04 desktop (GDM, password sudo, Android phone). Recommendation: **`libpam-google-authenticator`** (PAM TOTP module) + **Aegis Authenticator** (free Android app, F-Droid). Alternative phone app: **Ente Auth** (E2EE cloud sync across devices). Avoid Authy (proprietary/cloud) and Google Authenticator (no export/backup lock-in).

## Packages
```bash
sudo apt install libpam-google-authenticator   # Ubuntu universe, candidate 20191231-2build1
```
Hardware-key alternative: `libpam-u2f` / `pamu2fcfg` (FIDO2) — only if the user has a YubiKey-style token.

## Setup sequence (lockout-safe — test sudo BEFORE GDM)
1. `google-authenticator` — run in user's terminal (needs the QR to scan into Aegis). Save the recovery codes offline; they are the ONLY way back in if the phone is lost.
2. **sudo first**: add `auth required pam_google_authenticator.so` to `/etc/pam.d/sudo` (above the `@include common-auth` line is typical).
   - Test in a fresh shell: password + 6-digit code both required.
3. **GDM second** (only after sudo works): add the same line to `/etc/pam.d/gdm-password`.
4. Future-proof: same line in `/etc/pam.d/sshd` if SSH is ever enabled.

## Critical PAM semantics
- Use **`required`**, NOT `requisite`, for at least the first auth module — `required` keeps prompting through the chain so an attacker can't tell which factor was wrong (timing/error side-channel). google-authenticator docs recommend both password and OTP be `required`.
- Make TOTP `required` alongside `pam_unix.so` (password) so BOTH factors are demanded.

## Pitfalls
- **Lockout risk**: enabling GDM 2FA before sudo 2FA works can lock the user out of the whole desktop session. Always test sudo path first in a separate shell.
- Keep recovery codes offline (paper / encrypted vault), not on the same phone.
- `libpam-google-authenticator` is universe/community-supported — fine for a desktop, don't build a server policy on it without testing.
- Pair with a strong passphrase — a 7-digit numeric password (as the user used 2026-06) + TOTP is still weak against a local attacker who can capture the phone screen or if any login surface is exposed.
