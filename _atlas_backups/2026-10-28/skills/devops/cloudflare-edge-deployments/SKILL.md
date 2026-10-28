---
name: cloudflare-edge-deployments
description: Use when deploying to Cloudflare or attaching a domain.
version: 1.0.0
author: hermes-curator
license: MIT
metadata:
  hermes:
    tags: [cloudflare, workers, pages, dns, deployment, wrangler]
    related_skills: [deployed-change-verification, github-wiki-publishing]
---

# Deploying to Cloudflare (Workers, static assets, custom domains)

For any task that puts something on a Cloudflare hostname: publishing built output as a Worker,
attaching a subdomain, or working out why a hostname serves the wrong site. The decisions here are
driven by the scopes the available credential actually holds, so read the scopes before choosing a
mechanism.

## When to use

- "Deploy this to <subdomain>" / "put this behind Cloudflare".
- A built site or static directory needs a public URL.
- A hostname resolves and returns 200 but shows the wrong project.
- A deploy works locally but not from CI (or vice versa).

## 1. Read the scopes first — they pick the mechanism

```bash
wrangler whoami          # prints the account id, the account name and the scope list
```

- **No `dns_records:edit` scope?** You cannot create a CNAME/A/TXT through the API. That does not block
  you: a Workers **custom domain** provisions the DNS record and certificate server-side (see 3).
- **Cloudflare Pages custom domains want the CNAME created first** and sit at
  `verification_data.error_message: "CNAME record not set"` indefinitely without DNS write, or until
  someone adds the record in the dashboard. Prefer the Workers path when DNS write is unavailable.
- An insufficient scope returns `{"success": false, "errors": [{"code": 10000, "message":
  "Authentication error"}]}` even for a plain `GET` you expect `zone:read` to cover. Treat code 10000 as a
  scope problem, not as a wrong URL, wrong account id or bad token format — re-read `wrangler whoami`
  before retrying.
- `zone:read` is enough to list zones and routes and to resolve `zone_name` for a route.
- Prefer a tool that already holds a wider grant (an MCP server with its own OAuth, a logged-in
  `wrangler`) over asking the user for a broad token. Ask for a narrowly scoped token only when nothing
  else can do the job, and say which scope you need.
- Never assume an API-only path is impossible without testing it: a Workers custom domain is documented
  as self-provisioning and behaves that way in practice, which is the difference between finishing the
  task and asking the user for a DNS paste.

## 2. Serving built output: Workers static assets

Start from `templates/wrangler-static-assets.jsonc`. The settings that bite:

- **`html_handling` — prefer the URL shape the asset layer serves natively.** The goal is a served URL
  that equals the canonical/sitemap URL with no redirect hop. Reach it by choosing the *URL shape*: emit
  directory URLs (`/Foo/` + `Foo/index.html`, i.e. MkDocs `use_directory_urls: true`), and leave
  `html_handling` at its default. Then `/` resolves to `index.html` and `/Foo/` to `Foo/index.html` in the
  asset layer itself — no script runs, nothing to go stale.
- **Flat `.html` URLs force the fragile path.** If the build only emits `Page.html` (MkDocs with
  `use_directory_urls: false`), the default `auto-trailing-slash` **307-redirects** every `/Page.html` to
  `/Page` — an extra hop per link plus a served URL that disagrees with the canonical. The apparent fix is
  `"html_handling": "none"`, whose side effect is that `/` **stops** mapping to `/index.html`, so the
  root now depends on custom Worker code. That dependency is the defect: wherever the script does not run
  — an edge still holding the previous deploy, a stale or absent binding — the homepage falls through to
  the asset layer and 404s **while `/index.html` still works**, and it will not reproduce from any edge
  that already has the new version. Keep only when the URL shape is externally fixed, and treat the
  script as a fallback rather than the design:

```js
// cloudflare/worker.js — FALLBACK ONLY. Prefer directory URLs and no script: this
// rewrite puts custom code on the critical path of "/", and wherever it does not
// run the home page 404s while /index.html still works.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/") url.pathname = "/index.html";
    const response = await env.ASSETS.fetch(new Request(url.toString(), request));
    if (response.status !== 404) return response;
    const miss = await env.ASSETS.fetch(new Request(new URL("/404.html", url.origin).toString(), request));
    // A cached 404 is sticky and invisible: the path can exist a minute later and
    // the cached miss keeps being served. Never let a miss be cached.
    const headers = new Headers(miss.headers);
    headers.set("cache-control", "no-store");
    return new Response(miss.body, { status: 404, headers });
  },
};
```

- **Never let a miss be cached.** Return 404s with `cache-control: no-store`: a missing path that got
  cached stays 404 long after the path exists, which looks like a broken deploy and is invisible in the
  config. Add it both to the asset-layer 404 and to anything that re-serves one.
- **`not_found_handling`** — `"404-page"` serves your own `404.html`; without it a miss returns an empty
  body, which looks broken to everyone but you.

Assets-only (no `main`) is simpler and worth preferring; add `main` only for real request logic.

Deploy from a committed config, not from a directory you happen to have open:

```bash
wrangler deploy              # prefer the installed binary
npx --yes wrangler@4 deploy  # fallback for a machine that has never run wrangler
```

In a deploy script, fail early and loudly when the site builder is missing (`command -v mkdocs ||` print
the pip line and exit 1) — otherwise a PATH problem reads as a Cloudflare problem. Read the triggers
`wrangler deploy` prints: if your custom domain or route is not in that list, the config did not take
effect.

## 3. Custom domains and routes

```jsonc
"routes": [
  { "pattern": "host.example.com", "custom_domain": true },
  { "pattern": "host.example.com/*", "zone_name": "example.com" }
]
```

- The `custom_domain: true` entry creates the DNS record and the certificate. No DNS write scope needed.
- **Routes beat custom domains, and a wildcard route steals the hostname.** An existing
  `*.example.com/*` route pointing at another Worker will answer *your* hostname with 200 and a valid
  certificate — the URL "works" while serving the wrong site, which is the misleading failure mode.
  Diagnose by listing both sides:

  ```bash
  TOKEN=...; ZONE=...; ACC=...
  curl -s -H "Authorization: Bearer $TOKEN" "https://api.cloudflare.com/client/v4/zones/$ZONE/workers/routes"
  curl -s -H "Authorization: Bearer $TOKEN" "https://api.cloudflare.com/client/v4/accounts/$ACC/workers/domains"
  ```

  Then add the more specific `host.example.com/*` route to your Worker (the longest pattern wins) and keep
  BOTH entries declared in the config so a later redeploy does not drop the hostname back to the wildcard.
- A Worker and a Pages project can both claim the same hostname; only one serves. Check
  `/workers/domains` before debugging anything DNS-shaped.
- Delete an abandoned claim rather than leaving two half-live copies:
  `DELETE /accounts/<id>/pages/projects/<name>`.

## 4. Verify through the domain, not through the config

A freshly created record is often absent from the local resolver, so pin the edge IP:

```bash
dig +short @1.1.1.1 host.example.com
curl --resolve host.example.com:443:<edge-ip> -o /dev/null -w '%{http_code}' https://host.example.com/
curl -sv --resolve host.example.com:443:<edge-ip> https://host.example.com/ 2>&1 | grep -i 'subject:\|issuer:'
```

Check all of: the certificate subject/issuer, `/` and a deep path both 200 **with no redirect**, a real
asset, `sitemap.xml` if the generator emits one, and a nonsense path returning your custom 404 body.
Then confirm content identity by fetching a string that only exists in the build you just deployed —
status 200 on a stale host looks identical to success.

Fetch `/` from **every** address the hostname resolves to, not one:

```bash
for ip in $(dig +short @1.1.1.1 A host.example.com) $(dig +short A host.example.com | head -1); do
  printf '%s / => %s\n' "$ip" "$(curl -s -o /dev/null -w '%{http_code}' --resolve host.example.com:443:$ip https://host.example.com/)"
done
```

A user reporting a broken path you cannot reproduce is landing somewhere you are not — another colo, a
cached response, or a version you do not have. Do not close it as "works for me": find what varies and
remove the dependency that varies (for a root path, that means serving it in the layer that cannot be
stale, not adding another rewrite).

## 5. CI deploys: a skipped step reports success

- Guard a deploy job on a secret with an env-var + flag step; `if: secrets.X != ''` is not valid at job or
  step level:

  ```yaml
  - name: Is a token configured?
    id: cf
    env:
      TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
    run: |
      # Length only, never the value. "unset" and "set to an empty string" are
      # identical to the shell, and this line is the only thing that can tell
      # them apart. Keep it in the guard permanently.
      echo "secret lengths: token=${#TOKEN}"
      if [ -n "$TOKEN" ]; then echo "enabled=true" >> "$GITHUB_OUTPUT"; else
        echo "enabled=false" >> "$GITHUB_OUTPUT"
        echo "::notice::no Cloudflare token — skipping the domain deploy"; fi
  - if: steps.cf.outputs.enabled == 'true'
    uses: cloudflare/wrangler-action@v3
    with:
      apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
      accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
      wranglerVersion: "4" # v3 of the action installs wrangler 3.x, which cannot read a .jsonc config
      command: deploy
  ```

- **Pin `wranglerVersion: "4"` — the action's default wrangler 3.x cannot read `wrangler.jsonc`.** The
  deploy then runs with no `main` and no assets and dies with `Missing entry-point: The entry-point should
  be specified via the command line (e.g. wrangler deploy path/to/script) or the main config field`, which
  reads exactly like a broken config file and is not one; the tell is the wrangler version line in the log
  (`wrangler 3.90.0 (update available 4.x)`). The same config that deploys fine from a local `wrangler` 4
  works in CI once pinned, so a local-vs-CI mismatch on a config-shaped error means check the tool version
  before editing the config.

- **The consequence is the real lesson: with no secret the job is green and simply skips**, so the
  canonical host serves the previous build while any other host (a Pages mirror, a second deploy path)
  advances. Nothing on a dashboard says so. After every content change, fetch a newest-commit-only string
  from **each** published surface, or run the deploy by hand — the surface you did not deploy is the one
  that goes stale.
- **Read the step conclusion, never the job's.** A step whose `if:` is false reports `skipped`, and
  skipped counts neither way, so the job concludes `success` while the deploy never happened — the job
  name is not evidence. Assert on the specific step:

  ```bash
  gh run view <id> --json jobs \
    -q '.jobs[] | select(.name|test("deploy")) | .steps[] | "\(.name) -> \(.conclusion)"'
  ```

- **A secret that exists can still hold nothing, and nothing warns you.** `gh secret list` prints a name
  and a timestamp whether the value is a real token or an empty string, and `gh secret set` stores an
  empty or whitespace-only paste with exit status **0**. So a secret can be listed, be "set", and still
  read as unset in the runner — which looks exactly like nobody configured it. The length print above is
  the only visible symptom: if it reads 0 while the secret is listed, the value is empty; store it again.
- **A value captured with `read -s` can be whitespace-only.** bash keeps a trailing CR as data, so a
  token copied from a GUI clipboard (CRLF) can arrive as `\r` — non-empty to the shell, empty after any
  trim, and the emptiness check passes. Strip `[:space:]`, reject anything shorter than a plausible token
  (Cloudflare's are ~40 chars), report the character count you read, and re-prompt rather than storing
  garbage. Run `scripts/ci-secret-setup.sh` instead of hand-typing `gh secret set` — it does all of that
  and then triggers a run to prove the runner sees the value.
- Upload the built output as a workflow artifact in the build job and download it in the deploy job
  rather than rebuilding in both.

## Pitfalls

- **Don't debug a 200 as if it were a 404.** A wildcard route or a second claim makes the hostname answer
  with someone else's content; compare the body, not the status.
- **Don't add request logic to fix a URL-shape problem.** Rewriting `/` in a Worker script turns the
  homepage into custom code that can be stale, unbound or absent, and the failure only shows on some
  edges. Change the URLs the generator emits (directory URLs) or the asset-layer handling instead;
  reach for a script only when the shape is externally fixed.
- **Don't test DNS through the local resolver right after creating a record.** It lags; `dig @1.1.1.1` or
  the Cloudflare DoH endpoint is authoritative enough to continue with.
- **Don't leave the deploy path undocumented.** A Worker deployed once from a laptop and never wired into
  CI drifts silently; put the command in a `scripts/` file, reference it from the README, and name the
  canonical URL there when there is more than one host.
- **Don't ask for a broad token reflexively.** Enumerate what the current scopes can do first; a Workers
  custom domain is usually enough to finish without one.

## Support files

- `scripts/ci-secret-setup.sh` — store CI secrets without the empty-value trap (whitespace-stripping,
  length sanity check, character-count readout, retry), then trigger a workflow and assert a named step's
  conclusion instead of trusting the job.
- `templates/wrangler-static-assets.jsonc` — starter config for serving a built directory as a Worker on
  a custom domain: native URL handling (no `html_handling` override, no `main`), a real 404 page, and both
  route entries so a redeploy cannot drop the hostname back to a wildcard route.
- Docs-site specifics (build-from-a-copy shape, wiki link rewriting, GitHub Pages mirror, drift between
  published surfaces): the `github-wiki-publishing` skill, in its docs-site hosting and domains
  reference.
