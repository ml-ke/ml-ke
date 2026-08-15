#!/bin/bash
# AI Agent Bug Bounty Pipeline
# Tomnomnom-inspired: pipe small tools, one thing at a time
# Each phase produces a file. Next phase consumes it.
# Usage: ./bug-bounty-pipeline.sh target.com

set -e
DOMAIN="$1"
DIR="recon/$DOMAIN"
mkdir -p "$DIR"

echo "[+] Phase 1: Passive subdomain gathering"
curl -s "https://crt.sh/?q=%25.$DOMAIN&output=json" | jq -r '.[].name_value' | sed 's/\*\.//g' | sort -u > "$DIR/crtsh.txt"
echo "[+] Found $(wc -l < "$DIR/crtsh.txt") from crt.sh"

echo "[+] Phase 2: Historical URLs"
# waybackurls equivalent - fetch from Wayback Machine
curl -s "http://web.archive.org/cdx/search/cdx?url=*.$DOMAIN&output=json&fl=original&collapse=urlkey" | jq -r '.[] | .[0]' 2>/dev/null | grep -v "^\[\|\]\|^$" > "$DIR/wayback.txt" || true
echo "[+] Found $(wc -l < "$DIR/wayback.txt") Wayback URLs"

echo "[+] Phase 3: Live host probing"
# httprobe equivalent
cat "$DIR/crtsh.txt" | while read host; do
  curl -sk -o /dev/null -w "%{http_code} %{url_effective}\\n" "https://$host" 2>/dev/null &
done | sort -u > "$DIR/live.txt"
echo "[+] Live hosts saved"

echo "[+] Phase 4: Endpoint analysis"
cat "$DIR/wayback.txt" | grep -iE 'api|graphql|rest|v[0-9]|admin|internal|private|swagger|docs|health|status|callback|webhook' > "$DIR/api_endpoints.txt"
echo "[+] $(wc -l < "$DIR/api_endpoints.txt") API-related endpoints"

echo "[+] Phase 5: Parameter extraction"
cat "$DIR/wayback.txt" | grep '=' | sed 's/.*?//' | tr '&' '\\n' | cut -d'=' -f1 | sort -u > "$DIR/params.txt"
echo "[+] $(wc -l < "$DIR/params.txt") unique parameters"

echo "=== RECON COMPLETE ==="
echo "Results in $DIR/"
ls -la "$DIR/"