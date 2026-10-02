# Case Study: ml.co.ke DNS Failure (Jul 2026)

## Background

A Jekyll + Chirpy blog deployed via GitHub Actions → GitHub Pages with a custom domain `ml.co.ke`. Deployments were succeeding (build + deploy steps both green, artifact produced), but the site wasn't loading in browsers.

## Layer-by-Layer Investigation

### Layer 1: Direct Site Check

```
$ curl -svo /dev/null https://ml.co.ke 2>&1 | grep "< HTTP/"
curl: (6) Could not resolve host: ml.co.ke
```

**Signal:** DNS failure — move to Layer 2 immediately.

### Layer 2: DNS Resolution

```
$ dig +short ml.co.ke ANY
→ Empty (all record types)

$ dig +short ml.co.ke A      → empty
$ dig +short ml.co.ke AAAA   → empty
$ dig +short ml.co.ke CNAME  → empty
$ dig +short ml.co.ke NS     → empty
$ dig +short ml.co.ke SOA    → empty

$ dig ml.co.ke ANY @8.8.8.8
→ status: SERVFAIL
→ EDE: REFUSED from nameservers at [158.69.211.95, 49.12.105.164, 57.128.250.247]
→ EDE: No Reachable Authority
```

**Signal:** Nameservers exist but refuse queries for this domain. Proceed to Layer 3 to find them.

### Layer 3: Domain Registry / WHOIS

Used the KENIC RDAP browser interface at https://whois.kenic.or.ke/

**Findings:**
- Domain: ml.co.ke — **active** (not expired)
- Registered: 2024-06-29 | Expires: 2027-06-29
- Last changed: **2026-07-01** (something changed around this date)
- Registrar: Truehost Kenya (code: TK)
- Nameservers registered at TLD:
  - `ns1.cloudoon.com` (57.128.250.247 — OVH Poland)
  - `ns2.cloudoon.net` (49.12.105.164 — Hetzner Germany)
  - `ns3.cloudoon.org` (158.69.211.95 — OVH Canada)

Clicked **"SHOW DNS DATA"** on the RDAP page → received a JSON diagnostic:

```json
{
  "domain": "ml.co.ke",
  "success": false,
  "error": "Step 3 failed: no SOA record found",
  "nameserverStatus": {
    "ns1.cloudoon.com": {
      "tldDns": true,
      "registry": true,
      "soaStatus": {"57.128.250.247": "REFUSED"}
    },
    "ns2.cloudoon.net": {
      "tldDns": true,
      "registry": true,
      "soaStatus": {"49.12.105.164": "REFUSED"}
    },
    "ns3.cloudoon.org": {
      "tldDns": true,
      "registry": true,
      "soaStatus": {"158.69.211.95": "REFUSED"}
    }
  }
}
```

**Signal:** All 3 nameservers are correctly registered at the .ke TLD (`tldDns: true`, `registry: true`) but every one returns `REFUSED` — they don't have a zone for ml.co.ke configured.

### Layer 4: Nameserver Health

```
$ dig @ns1.cloudoon.com ml.co.ke SOA
→ REFUSED

$ dig @57.128.250.247 ml.co.ke SOA
→ REFUSED
```

Confirmed: The Cloudoon DNS hosting account either had the zone deleted, was suspended, or was never configured. The nameservers are alive and responding — but they refuse to serve ml.co.ke.

### Layer 5: Hosting Provider (GitHub Pages)

```
$ curl -svo /dev/null https://ml-ke.github.io/ml-ke/ 2>&1 | grep "< HTTP/"
< HTTP/2 301
< location: https://ml.co.ke/
```

**Signal:** GitHub Pages IS serving content. The 301 redirect to ml.co.ke confirms that Pages is configured and the CNAME is set correctly. The site would work if DNS resolved.

```
$ curl -s "https://api.github.com/repos/ml-ke/ml-ke" | python3 -c "..."
has_pages: True
```

Confirmed: Pages is enabled on this repo.

### Layer 6: Deployment Pipeline

```
$ curl -s "https://api.github.com/repos/ml-ke/ml-ke/actions/runs?per_page=5" | ...
ID=29914115274 status=completed conclusion=success sha=a646704
ID=29824532983 status=completed conclusion=success sha=05855a3
...
```

All recent workflow runs successful. The latest run (ID 29914115274) had:
- Build step: 28s ✅
- Deploy step: 37s ✅
- Artifact: 3.77 MB (github-pages)
- Push by BongweKE to main

**Only finding:** 2 warnings about Node.js 20 deprecation — non-fatal.

## Root Cause

**Cloudoon DNS hosting has no zone for ml.co.ke.** All 3 nameservers return REFUSED. The domain is registered at Truehost Kenya with Cloudoon nameservers, but Cloudoon's DNS control panel needs a zone with A records pointing to GitHub Pages IPs.

Likely trigger: The `last changed` date of 2026-07-01 suggests something was modified around that time — possibly a nameserver change, account issue, or zone deletion.

## Resolution Path

### First attempt: Configure zone at Cloudoon
User needed access to Cloudoon DNS control panel to add A records:
```
ml.co.ke  A  185.199.108.153
ml.co.ke  A  185.199.109.153
ml.co.ke  A  185.199.110.153
ml.co.ke  A  185.199.111.153
```

### Second attempt: Migrate to Freehosting nameservers
User switched nameservers at the registry (KENIC/Truehost) from Cloudoon to **Freehosting**. A DNS propagation watch was set up (cron job, every 5 min, incremental milestones via state file).

**Milestones tracked:**
1. Registry NS updated to Freehosting ✅ (confirmed at the registry level)
2. DNS NS propagation (waiting for DNS to reflect Freehosting)
3. A records resolving
4. GitHub Pages direct access 200
5. Site live at ml.co.ke 200

**Status as of 22 Jul 2026 17:11 EAT:** M1 (registry) done. NS propagation still pending.

### Third attempt: Suspect local router caching
User: "Since the NS is okay, the rest is in the router. Let me turn off the router then you'll let me know in three hours if all checkpoints work."

A one-shot comprehensive check was scheduled for 20:55 EAT (3 hours out) using `scripts/check-all-milestones.sh` — reports ALL 5 milestones with ✅/❌ regardless of prior state, delivered to CLI + Telegram.

## Key Lessons

1. **Deployments succeeding ≠ site working.** Always check DNS first when the build is green but the site is dark.
2. **The GitHub Pages direct URL** (`org.github.io/repo/`) bypasses the custom domain. If it 301s to the custom domain, Pages is configured correctly — the problem is DNS.
3. **Registry-level DNS diagnostics** (KENIC's "SHOW DNS DATA") are the most authoritative test. They test from the TLD perspective and catch REFUSED responses that local dig queries might not surface clearly.
4. **Node.js deprecation warnings** in GitHub Actions are yellow, not red. Don't let them distract from the real issue.
5. **The "last changed" date** in WHOIS data is a valuable clue — correlate it with when the user first noticed the problem.
6. **Incremental watch ≠ comprehensive check.** When the user applies a manual fix and asks for a status-check, provide a full milestone report — not just new milestone notifications.
7. **Local network/routing** can mask DNS resolution even after the authoritative side is fixed. Router caches, ISP DNS, or local resolvers may need to be flushed or rebooted.
