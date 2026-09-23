#!/usr/bin/env bash
# check-all-milestones.sh
# Comprehensive one-shot verification of ALL checkpoints for a domain.
# Designed for cron/no_agent delivery — reports every milestone's current
# status regardless of whether it was already reported before.
#
# Intended use: after user applies a fix (DNS change, router reboot, etc.),
# schedule this as a one-shot cron job to check if everything resolved.
#
# Usage: edit DOMAIN and EXPECTED_NS at the top, then run or schedule.

DOMAIN="ml.co.ke"
EXPECTED_NS="freehosting"

echo "=== $DOMAIN — Full Milestone Check ==="
echo ""

# M1: Registry/WHOIS nameservers
RDAP_NS=$(curl -s "https://whois.kenic.or.ke/domain/$DOMAIN" 2>/dev/null | python3 -c "
import sys,json
try:
    d = json.load(sys.stdin)
    nss = [e.get('ldhName','') for e in d.get('nameservers', []) if 'ldhName' in e]
    print(','.join(nss))
except:
    pass
" 2>/dev/null || echo "unreachable")
echo "M1 | Registry NS: $RDAP_NS"
if echo "$RDAP_NS" | grep -qi "$EXPECTED_NS"; then
  echo "    ✅ Registry nameservers set to $EXPECTED_NS"
else
  echo "    ❌ Registry nameservers NOT yet $EXPECTED_NS"
fi

# M2: DNS NS propagation
CURRENT_NS=$(dig +short "$DOMAIN" NS 2>/dev/null | tr '[:upper:]' '[:lower:]' | paste -sd, || echo "unreachable")
echo "M2 | DNS NS: $CURRENT_NS"
if echo "$CURRENT_NS" | grep -qi "$EXPECTED_NS"; then
  echo "    ✅ NS propagated to $EXPECTED_NS"
else
  echo "    ❌ NS NOT propagated yet"
fi

# M3: A records
A_RECORDS=$(dig +short "$DOMAIN" A 2>/dev/null | paste -sd, || echo "unreachable")
echo "M3 | A records: $A_RECORDS"
if [ -n "$A_RECORDS" ]; then
  echo "    ✅ A records resolving ($A_RECORDS)"
else
  echo "    ❌ No A records resolving"
fi

# M4: GitHub Pages direct access
GH_HTTP=$(curl -svo /dev/null "https://ml-ke.github.io/ml-ke/" 2>&1 | grep -oP '(?<=HTTP/2 )\d+' || echo "000")
echo "M4 | GitHub.io: HTTP $GH_HTTP"
if [ "$GH_HTTP" = "200" ]; then
  echo "    ✅ GitHub Pages direct returns 200"
else
  echo "    ❌ GitHub Pages direct returns HTTP $GH_HTTP"
fi

# M5: Site live at custom domain
CURL_HTTP=$(curl -svo /dev/null "https://$DOMAIN" 2>&1 | grep -oP '(?<=HTTP/2 )\d+' || echo "000")
echo "M5 | $DOMAIN: HTTP $CURL_HTTP"
if [ "$CURL_HTTP" = "200" ]; then
  echo "    ✅ $DOMAIN IS LIVE! 🎉"
else
  echo "    ❌ $DOMAIN returns HTTP $CURL_HTTP"
fi
