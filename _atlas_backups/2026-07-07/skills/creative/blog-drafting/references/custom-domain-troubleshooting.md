# Custom Domain Troubleshooting (ml.co.ke)

Real debugging history: `ml.co.ke` resolves to a Sedo parked page instead of the
Chirpy blog hosted on GitHub Pages.

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

**Detection:**

```bash
# Does the domain serve blog content or a parked page?
curl -s https://ml.co.ke | grep -c "jekyll-theme-chirpy\|ml-ke\|posts" || echo "Parked/offline"

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

When the custom domain is parked, cron jobs that publish posts STILL work
correctly:

1. ✅ Post is committed to the repo and pushed to GitHub
2. ✅ GitHub Actions build runs and completes with "success"
3. ❌ Site is not accessible at `https://ml.co.ke` until DNS is fixed

**The cron should NOT report this as a deployment failure.** The post is
published and the build passed. Report the DNS issue as a separate known
concern.
