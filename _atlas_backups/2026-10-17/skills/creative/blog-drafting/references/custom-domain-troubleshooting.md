# Custom Domain Troubleshooting (ml.co.ke)

## DNS State Timeline

| Date | State | Observed |
|------|-------|----------|
| Pre-Jul 2026 | Working blog at ml.co.ke | Chirpy blog served |
| ~Jun 2026 | Sedo parked page (DNS pointed to domain parking) | Registrar default landing page with ads |
| Jul 11, 2026 | **DNS name not resolving** (`Could not resolve host: ml.co.ke`) | No response at all from DNS |

The domain has progressively degraded: from working blog → parked page → no DNS resolution. The Actions API is now the **only** reliable way to verify deployments.

## Symptom: Homepage Shows Sedo Parked Page

```
<!DOCTYPE html><html...><title>ml.co.ke - ml Resources and Information.</title>
<script async src='https://euob.iseaskies.com/sxp/i/...'></script>
```

**Root cause:** The custom domain `ml.co.ke` DNS records do not point to GitHub
Pages. The domain registrar's default parking page (Sedo) is served instead.
The `CNAME` file in the repo sets `ml.co.ke`, but DNS resolution ignores it
because the A/AAAA records at the registrar level still point to the domain
parking service.

## Symptom: DNS Name Not Resolving (worse than parked)

```
$ curl -s https://ml.co.ke
curl: (6) Could not resolve host: ml.co.ke
```

This state is **worse** than a parked page — the domain name simply doesn't
resolve to any IP at all. Possible causes:
- DNS records were removed entirely (not just pointed to a parking service)
- Domain registration may have lapsed or nameservers changed
- Aliyun/Namecheap registrar may have dropped A/AAAA records

**When this happens, all DNS-based verification methods fail:**
- `curl https://ml.co.ke` → "Could not resolve host"
- `dig +short ml.co.ke` → empty
- Homepage canary check → silent failure (no content to grep)
- Direct permalink check → silent failure (no route to host)

**The only working verification method:** GitHub Actions API (see below).

## Detection

```bash
# Does the domain serve blog content or a parked page?
curl -s https://ml.co.ke | grep -c "jekyll-theme-chirpy\|ml-ke\|posts" || \
  echo "Parked/offline or unresolving DNS — use Actions API instead"

# Check DNS resolution explicitly
curl -svo /dev/null "https://ml.co.ke" 2>&1 | grep "Could not resolve"
# If this matches, DNS is completely broken — skip all homepage checks

# Check GitHub Pages build status — this works regardless of DNS
curl -s "https://api.github.com/repos/ml-ke/ml-ke/actions/runs?per_page=1" | \
  python3 -c "import sys,json; r=json.load(sys.stdin)['workflow_runs'][0]; print(r['status'], r.get('conclusion','-'), r['head_sha'][:8])"

# Check if ml-ke.github.io works (may 404 due to CNAME enforcement)
curl -s -o /dev/null -w "%{http_code}" "https://ml-ke.github.io/"
```

## Fix: Point DNS to GitHub Pages

GitHub Pages serves from these IP addresses (configure as A/AAAA records):

```
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

Alternatively, configure a CNAME from `www.ml.co.ke` → `ml-ke.github.io`
and set up a `www` redirect in the domain settings.

After updating DNS, verify with:

```bash
dig +short ml.co.ke | grep -c "185.199" && echo "✅ Points to GitHub Pages" || echo "❌ Still not GitHub Pages"
```

Note: DNS propagation can take 5–30 minutes. GitHub Pages will auto-renew the
TLS certificate once DNS resolves correctly.

## Impact on Cron Publishing

When the custom domain is broken (parked or unresolving), cron jobs that publish
posts STILL work correctly:

1. ✅ Post is committed to the repo and pushed to GitHub
2. ✅ GitHub Actions build runs and completes with "success"
3. ❌ Site is not accessible at `https://ml.co.ke` until DNS is fixed

**The cron should NOT report this as a deployment failure.** The post is
published and the build passed. Report the DNS issue as a separate known
concern.

**Verification strategy when DNS is broken:** Use the Actions API as the
primary verification method. Do not attempt homepage curl checks — they will
silently fail and produce false negatives.
