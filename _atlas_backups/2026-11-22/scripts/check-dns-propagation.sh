#!/usr/bin/env bash
# Check DNS propagation for ml.co.ke and report milestones
set -euo pipefail

DOMAIN="ml.co.ke"
STATE_FILE="/home/pro-g/.hermes/scripts/.dns-milestones-${DOMAIN//./-}.state"
EXPECTED_NS="freehosting"

# Initialize state file
touch "$STATE_FILE"

# Mark a milestone as reached
mark_milestone() {
    local name="$1"
    if ! grep -qx "$name" "$STATE_FILE" 2>/dev/null; then
        echo "$name" >> "$STATE_FILE"
        return 0  # New milestone
    fi
    return 1  # Already reported
}

# Load current state
CURRENT_NS=$(dig +short "$DOMAIN" NS 2>/dev/null | tr '[:upper:]' '[:lower:]' | paste -sd, || true)

# Check RDAP/registry for nameserver changes
RDAP_NS=$(curl -s "https://whois.kenic.or.ke/domain/$DOMAIN" 2>/dev/null | python3 -c "
import sys,json
try:
    d = json.load(sys.stdin)
    nss = [e.get('ldhName','') for e in d.get('nameservers', []) if 'ldhName' in e]
    print(','.join(nss))
except:
    pass
" 2>/dev/null || true)

# Check A records
A_RECORDS=$(dig +short "$DOMAIN" A 2>/dev/null | paste -sd, || true)

# Check if site loads (HTTP 200)
SITE_OK=0
CURL_HTTP=$(curl -svo /dev/null "https://$DOMAIN" 2>&1 | grep -oP '(?<=HTTP/2 )\d+' || echo "000")
if [ "$CURL_HTTP" = "200" ]; then
    SITE_OK=1
fi

# Also check redirect from github.io
GH_REDIRECT_OK=0
GH_HTTP=$(curl -svo /dev/null "https://ml-ke.github.io/ml-ke/" 2>&1 | grep -oP '(?<=HTTP/2 )\d+' || echo "000")
if [ "$GH_HTTP" = "200" ]; then
    GH_REDIRECT_OK=1
fi

echo "NS=$CURRENT_NS"
echo "RDAP_NS=$RDAP_NS"
echo "A_REC=$A_RECORDS"
echo "SITE_OK=$SITE_OK"
echo "GH_DIRECT=$GH_REDIRECT_OK"
echo "CURL_HTTP=$CURL_HTTP"

# Check milestones
NEW_MILESTONE=0
OUTPUT=""

# Milestone 1: Registry updated
if echo "$RDAP_NS" | grep -qi "$EXPECTED_NS" && mark_milestone "registry_updated"; then
    NEW_MILESTONE=1
    OUTPUT+="🏁 MILESTONE 1: Registry nameservers updated to Freehosting!\n  Registry NS now: $RDAP_NS\n  (Waiting for DNS propagation…)\n"
fi

# Milestone 2: DNS NS records propagated
if echo "$CURRENT_NS" | grep -qi "$EXPECTED_NS" && mark_milestone "dns_ns_propagated"; then
    NEW_MILESTONE=1
    OUTPUT+="🏁 MILESTONE 2: DNS nameservers propagated!\n  Resolving NS: $CURRENT_NS\n  (Waiting for A records…)\n"
fi

# Milestone 3: A records resolving
if [ -n "$A_RECORDS" ] && mark_milestone "a_records_resolving"; then
    NEW_MILESTONE=1
    OUTPUT+="🏁 MILESTONE 3: A records now resolving!\n  Resolving to: $A_RECORDS\n  (Waiting for site to load…)\n"
fi

# Milestone 4: GitHub.io direct access works
if [ "$GH_REDIRECT_OK" -eq 1 ] && mark_milestone "gh_direct_working"; then
    NEW_MILESTONE=1
    OUTPUT+="🏁 MILESTONE 4: GitHub.io direct access working!\n  https://ml-ke.github.io/ml-ke/ returns HTTP 200\n"
fi

# Milestone 5: Site loads at ml.co.ke
if [ "$SITE_OK" -eq 1 ] && mark_milestone "site_loading"; then
    NEW_MILESTONE=1
    OUTPUT+="🎉 MILESTONE 5: ml.co.ke IS LIVE!\n  https://ml.co.ke returns HTTP 200\n  The blog is back online!\n"
fi

if [ "$NEW_MILESTONE" -eq 1 ]; then
    echo "DELIVER=YES"
    echo -e "$OUTPUT"
fi
