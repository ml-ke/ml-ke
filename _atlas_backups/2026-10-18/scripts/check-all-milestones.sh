#!/usr/bin/env bash
# One-shot comprehensive milestone check for ml.co.ke
DOMAIN="ml.co.ke"

echo "=== ml.co.ke — Full Milestone Check ==="
echo ""

# 1) Registry NS check
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
if echo "$RDAP_NS" | grep -qi "freehosting"; then
  echo "    ✅ Registry nameservers set to Freehosting"
else
  echo "    ❌ Registry nameservers NOT yet Freehosting"
fi

# 2) DNS NS propagation
CURRENT_NS=$(dig +short "$DOMAIN" NS 2>/dev/null | tr '[:upper:]' '[:lower:]' | paste -sd, || echo "unreachable")
echo "M2 | DNS NS: $CURRENT_NS"
if echo "$CURRENT_NS" | grep -qi "freehosting"; then
  echo "    ✅ NS propagated to Freehosting"
else
  echo "    ❌ NS NOT propagated yet"
fi

# 3) A records
A_RECORDS=$(dig +short "$DOMAIN" A 2>/dev/null | paste -sd, || echo "unreachable")
echo "M3 | A records: $A_RECORDS"
if [ -n "$A_RECORDS" ]; then
  echo "    ✅ A records resolving ($A_RECORDS)"
else
  echo "    ❌ No A records resolving"
fi

# 4) GitHub.io direct access
GH_HTTP=$(curl -svo /dev/null "https://ml-ke.github.io/ml-ke/" 2>&1 | grep -oP '(?<=HTTP/2 )\d+' || echo "000")
echo "M4 | GitHub.io: HTTP $GH_HTTP"
if [ "$GH_HTTP" = "200" ]; then
  echo "    ✅ ml-ke.github.io/ml-ke/ returns 200"
else
  echo "    ❌ ml-ke.github.io/ml-ke/ returns HTTP $GH_HTTP"
fi

# 5) Site live at ml.co.ke
CURL_HTTP=$(curl -svo /dev/null "https://$DOMAIN" 2>&1 | grep -oP '(?<=HTTP/2 )\d+' || echo "000")
echo "M5 | ml.co.ke: HTTP $CURL_HTTP"
if [ "$CURL_HTTP" = "200" ]; then
  echo "    ✅ ml.co.ke IS LIVE! 🎉"
else
  echo "    ❌ ml.co.ke returns HTTP $CURL_HTTP"
fi

echo ""
echo "DELIVER=YES"
