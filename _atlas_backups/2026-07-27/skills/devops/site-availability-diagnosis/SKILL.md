---
name: site-availability-diagnosis
description: Multi-layered systematic methodology for diagnosing why a website isn't loading. Covers DNS resolution, domain registry/WHOIS, nameserver health, hosting provider checks, and deployment pipeline investigation.
---

# Site Availability Diagnosis

Trigger when a user reports "site X is not loading" or "deployments aren't reflecting."

## Methodology: The Six Layers

Investigate bottom-up (DNS → hosting → deployment) — an early layer failure cascades and makes higher layers look broken.

### Layer 1: Direct Site Check

```bash
# Fast HTTP check
curl -svo /dev/null "https://example.com" 2>&1 | grep -E "< HTTP/|location|Could not resolve|Connection refused"

# Full content check
curl -sL "https://example.com" 2>/dev/null | head -50

# Timeout detection
curl -s --connect-timeout 5 -m 10 "https://example.com" >/dev/null 2>&1
echo "Exit code: $?"  # 6=DNS failure, 7=connection refused, 28=timeout
```

**Symptoms and first conclusions:**
- `curl: (6) Could not resolve host` → **DNS failure** (Layer 2)
- `curl: (7) Connection refused` → **Server down / not listening**
- `curl: (28) Connection timed out` → **Network path blocked** (firewall, routing)
- HTTP 200 but wrong content → **Parked page / DNS pointing elsewhere** (Layer 2 or 3)
- HTTP 301/302 → **Check the redirect target** and follow it

### Layer 2: DNS Resolution

Check every record type that matters:

```bash
dig +short example.com A           # IPv4
dig +short example.com AAAA        # IPv6
dig +short example.com CNAME       # Canonical name (for subdomains)
dig +short example.com NS          # Nameservers
dig +short example.com SOA         # Start of authority — proves zone exists
dig +short example.com MX          # Mail (if relevant)
```

**Diagnostic rules:**
- **Empty A + AAAA** → the domain has no IP mapping. Site won't load.
- **Empty NS** → nameserver delegation is broken at the registry.
- **Empty SOA** → the authoritative nameservers don't have a zone configured for this domain (REFUSED response).
- **A/AAAA resolves but wrong IP** → DNS points to old hosting or a parked page.
- **CNAME set** → must resolve to a hostname that resolves to IPs (CNAME cannot coexist with other record types at the apex).

Also check propagation across different resolvers:

```bash
# Check via Google DNS vs. system resolver
dig @8.8.8.8 example.com A
dig @1.1.1.1 example.com A

# Check authoritative nameservers directly
dig @ns1.example-ns.com example.com A
```

### Layer 3: Domain Registry / WHOIS

Check the domain registration health and nameserver delegation:

```bash
# General WHOIS (install whois first or use web-based)
whois example.com | grep -E "Name Server|Expir|Status|Registrar|Creation Date"

# .ke domains — use KENIC RDAP: https://whois.kenic.or.ke/
# Search by domain name, expand the "Nameservers" section
```

**What to look for:**
- **Domain expired?** Check expiration date and current status (active/redemption/held)
- **Nameservers correct?** Compare what the registry lists vs. what your provider told you
- **Last changed date?** If recently changed, a nameserver migration may be incomplete
- **Registry DNS diagnostics** — `.ke` domains have a built-in DNS health check (click "SHOW DNS DATA" on the KENIC RDAP page). It shows if nameservers are responding or refusing.

The **"SHOW DNS DATA"** feature on Kenya's KENIC WHOIS is invaluable — it returns a JSON report showing:
- Whether each nameserver is registered at the TLD (`tldDns: true/false`)
- Whether each nameserver returns an SOA for the domain (`soaStatus`)
- `REFUSED` = The nameserver received the query but said "I don't serve this zone"
- `OK` = The nameserver responded with valid SOA data
- The final `error` field tells you exactly which step failed (e.g., `"Step 3 failed: no SOA record found"`)

### Layer 4: Nameserver Health

If the registry reports nameservers but they refuse or don't respond:

```bash
# Resolve nameserver hostnames to IPs
dig +short ns1.provider.com A

# Query each nameserver directly for the domain's SOA
dig @ns1.provider.com example.com SOA
dig @ns2.provider.com example.com SOA

# Check if the nameserver is even reachable
nc -zv ns1.provider.com 53 2>&1
```

**REFUSED vs. NXDOMAIN vs. timeout:**
| Response | Meaning |
|----------|---------|
| `REFUSED` | Nameserver received query but has no zone for this domain. Zone was deleted or not configured. |
| `NXDOMAIN` | Zone exists but the specific record doesn't (or the domain itself doesn't exist in the zone). |
| `timeout` / no response | Nameserver is not reachable or not running. |
| Valid SOA | Zone is healthy. Problem is elsewhere. |

**Common causes of REFUSED:**
1. The DNS zone was deleted at the provider (billing issue, manual deletion)
2. The nameservers were changed at the registrar but the new provider hasn't been configured
3. The DNS hosting account was suspended

### Layer 5: Hosting Provider Check

Verify the hosting platform directly (bypassing the custom domain):

**GitHub Pages:**
```bash
# Pages config exists?
curl -s "https://api.github.com/repos/org/repo" | python3 -c "import sys,json; print('has_pages:', json.load(sys.stdin).get('has_pages', 'N/A'))"

# Direct Pages URL (bypasses custom domain)
curl -svo /dev/null "https://org.github.io/repo/" 2>&1 | grep -E "< HTTP/|location"
# 301 redirect to custom domain = Pages is live but redirecting
# 200 with content = Pages serving without custom domain
# 404 = Pages not deployed or wrong URL format
```

**Cloudflare Pages / Netlify / Vercel:**
```bash
# Check the *.pages.dev / *.netlify.app / *.vercel.app URL directly
curl -svo /dev/null "https://project.pages.dev" 2>&1 | grep "< HTTP/"
```

### Layer 6: Deployment Pipeline

If the hosting platform says content is deployed but the site isn't loading, check the CI/CD:

```bash
# GitHub Actions — latest runs (rate-limited without auth)
curl -s "https://api.github.com/repos/org/repo/actions/runs?per_page=3" | python3 -c "
import sys, json
for r in json.load(sys.stdin).get('workflow_runs', []):
    print(f\"  {r['status']:10s} conclusion={r.get('conclusion') or '-':10s} sha={r['head_sha'][:8]} created={r['created_at'][:19]} name={r['name']}\")
"

# Check specific job steps
curl -s "https://api.github.com/repos/org/repo/actions/runs/RUN_ID/jobs" | python3 -c "
import sys, json
for j in json.load(sys.stdin).get('jobs', []):
    print(f\"  Job: {j['name']} status={j['status']} conclusion={j.get('conclusion')}\")
    for s in j.get('steps', []):
        print(f\"    Step: {s['name']} status={s['status']} conclusion={s.get('conclusion')}\")
"
```

**Key checks:**
- Is the latest workflow run `completed` with `conclusion=success`?
- Are there warnings? (e.g., Node.js version deprecation — usually non-fatal)
- Was an artifact produced? Check the Actions run page for "Artifacts" section
- Does the deploy step link to the right environment URL?

**Unreliable method:** `git commit --allow-empty` does NOT trigger GitHub Actions. To re-trigger, modify a tracked file.

## Verification After Fix

### Ad-hoc checks

After making DNS changes, verify propagation:

```bash
# Check from your system
dig +short example.com A

# Check from Google DNS (to confirm global propagation)
dig @8.8.8.8 +short example.com A

# Check the site is reachable and serves the right content
curl -sL "https://example.com" | grep -c "expected site keyword"
echo "Exit: $?"

# Check TLS certificate (GitHub Pages auto-provisions after DNS resolves)
curl -svo /dev/null "https://example.com" 2>&1 | grep -i "ssl\\|certificate\\|TLS"
```

GitHub Pages TLS auto-provisioning: Once DNS points A records to 185.199.108.153–185.199.111.153, GitHub automatically provisions a Let's Encrypt certificate for the custom domain within minutes. If you see TLS errors, wait 5–10 minutes and retry.

### Comprehensive one-shot verification (after user applies a fix on their end)

When the user says "the NS is okay now, let me try something on my end" and asks to check back later, **do not** start a long-running incremental watcher. Instead, schedule a **one-shot comprehensive check** that reports ALL milestones regardless of prior state.

**Why:** The incremental watch pattern (state-file-based, only reports new milestones) is designed for long-running propagation monitoring. But when the user has made a manual fix (router reboot, DNS change, local network change) they want to know "did everything resolve?" — not "what changed since last check?"

**Pattern:**
1. Create a comprehensive checker script that tests every layer independently (registry → DNS NS → A records → hosting direct → custom domain)
2. Schedule it as a `no_agent=true` one-shot cron job, typically 3 hours out (to allow for propagation + local DNS cache expiry)
3. Deliver to `origin,all` so it reaches both CLI and Telegram
4. Each milestone gets ✅ or ❌ so the user can see at a glance what passed and what failed

See `scripts/check-all-milestones.sh` in this skill for a working example (edit DOMAIN and EXPECTED_NS, then schedule with cronjob).

## Pitfalls

- **The Actions API is rate-limited** without authentication. For heavy debugging, use `gh` CLI if available, or use the browser to view Actions page.
- **DNS is the most likely root cause** for "deployments successful but site not loading." Don't spend time debugging the deployment pipeline until you've confirmed DNS resolves.
- **301 redirect from GitHub Pages to custom domain** means Pages IS working — the problem is DNS not resolving the custom domain.
- **The hosting provider's own domain** (e.g., `org.github.io/repo/`) may return a different result from the custom domain. This is diagnostic gold — compare them.
- **Registry-level DNS diagnostics** (like KENIC's "SHOW DNS DATA") are authoritative because they test from the TLD's perspective. If they report REFUSED, the problem is definitely at the DNS provider, not the registry.
- **Don't confuse warnings with errors.** GitHub Actions Node.js deprecation warnings (Node.js 20 → 24) are yellow warnings, not red failures. The build succeeds despite them.
- **Incremental watch ≠ comprehensive check.** When a user says "check back in N hours" after applying a fix, use a comprehensive one-shot that tests ALL layers — not a state-file watcher that only reports new milestones.
