#!/bin/bash
# harden-host-sudoers.sh — remediate F1/F2 from host-compromise-review-2026-09-14.md
# Run: bash ~/.hermes/scripts/harden-host-sudoers.sh
# Safe order: evidence first, then remove, then validate. Idempotent.
set -uo pipefail

PW=$(grep '^SUDO_PASSWORD' /home/pro-g/.hermes/.env | cut -d= -f2-)
printf '#!/bin/sh\necho "%s"\n' "$PW" > /tmp/hermes-askpass.sh
chmod 700 /tmp/hermes-askpass.sh
export SUDO_ASKPASS=/tmp/hermes-askpass.sh
unset SUDO_PASSWORD
S="sudo -A"
EV=/root/incident-2026-09-14
DATE=$(date +%Y%m%d-%H%M%S)

echo "=== 0. pre-flight: agent privilege path still works? ==="
$S id -u >/dev/null 2>&1 || { echo "FATAL: no sudo — aborting, nothing changed"; exit 1; }
echo "sudo OK (uid $($S id -u))"

echo
echo "=== 1. preserve evidence before deleting anything ==="
$S mkdir -p "$EV"
$S cp -a /etc/sudoers.d/hermes-atlas "$EV/sudoers-hermes-atlas.$DATE.evidence" 2>/dev/null
$S cp -a /home/hermes-atlas "$EV/hermes-atlas-home.$DATE.evidence" 2>/dev/null
$S cp -a /etc/sudoers.d/pro-g.bak "$EV/pro-g.bak.$DATE.evidence" 2>/dev/null
$S grep -E 'hermes-atlas' /etc/passwd /etc/group /etc/shadow > "$EV/account.$DATE.txt" 2>/dev/null
$S ls -la "$EV" 2>/dev/null

echo
echo "=== 2. F1: remove the NOPASSWD:ALL sudoers rule for hermes-atlas ==="
if $S test -e /etc/sudoers.d/hermes-atlas; then
  $S rm -f /etc/sudoers.d/hermes-atlas && echo "removed /etc/sudoers.d/hermes-atlas"
else
  echo "already absent"
fi

echo
echo "=== 3. F1: remove the dormant service account + its home ==="
if $S id hermes-atlas >/dev/null 2>&1; then
  $S userdel -r hermes-atlas 2>/dev/null || $S userdel hermes-atlas
  echo "removed user hermes-atlas"
else
  echo "already absent"
fi
$S groupdel hermes-atlas 2>/dev/null && echo "removed group hermes-atlas" || true

echo
echo "=== 4. F2: neutralise the inert NOPASSWD:ALL copy ==="
if $S test -e /etc/sudoers.d/pro-g.bak; then
  $S rm -f /etc/sudoers.d/pro-g.bak && echo "removed /etc/sudoers.d/pro-g.bak (evidence kept in $EV)"
else
  echo "already absent"
fi

echo
echo "=== 5. validate sudoers parses (must NOT lock you out) ==="
$S visudo -c

echo
echo "=== 6. post-flight verification ==="
echo -n "hermes-atlas account: "; $S getent passwd hermes-atlas >/dev/null && echo "STILL PRESENT" || echo "gone"
echo -n "hermes-atlas sudo:    "; $S sudo -l -U hermes-atlas 2>&1 | tail -1
echo "remaining NOPASSWD rules:"; $S grep -rn 'NOPASSWD' /etc/sudoers /etc/sudoers.d/ 2>/dev/null | sed 's/^/  /'
echo -n "pro-g sudo still works: "; $S id -un

echo
echo "=== 7. exposure reminder (not changed by this script) ==="
echo "  docker ports still published to 0.0.0.0:"
ss -tln | awk 'NR==1 || $4 !~ /^127\./ && $4 !~ /^\[::1\]/ {print "   ", $4}' | grep -E ':5434|Local' || true
echo "  Fix in compose (127.0.0.1:PORT:PORT) and: docker compose up -d"

rm -f /tmp/hermes-askpass.sh
echo
echo "DONE. Evidence: $EV"
