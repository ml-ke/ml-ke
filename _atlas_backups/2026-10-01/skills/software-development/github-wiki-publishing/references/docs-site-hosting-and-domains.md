# Publishing the docs site: build shape, links, hosting, domains

Depth for the publishing half of the skill: how to render a `wiki/` tree into a real docs site without
breaking the GitHub-wiki sync, how to make link rewriting trustworthy, and how to attach a custom domain
when the API credentials you hold cannot write DNS.

## The build shape that keeps the wiki working

- **Build from a prepared copy so `wiki/` stays byte-identical** (the wiki sync copies `wiki/` verbatim):
  `rm -rf site-src && mkdir -p site-src && cp wiki/*.md site-src/ && cp wiki/00-Home.md site-src/index.md`,
  point `docs_dir` at `site-src`, and `exclude_docs` the wiki-only chrome (`_Sidebar.md`, `_Footer.md`,
  `00-Home.md`). Commit the script, not the copy — gitignore `site-src/` and `site/`.
- **`use_directory_urls: false`** keeps flat `Page.html` URLs. Relative links between pages then work
  under any path prefix, including a project Pages path like `/repo-name/`.
- **Prefer the generator's directory URLs (`use_directory_urls: true`) when a static host serves the site.**
  They are the default, they are what a static host resolves natively (`/` -> `index.html`, `/Foo/` ->
  `Foo/index.html`) with no redirect and no rewrite, and the canonical URL the generator writes is the URL
  actually served. Choose flat URLs only when something else pins the shape (an existing inbound link set,
  a host that cannot do directory indexes). The link rewriter does not care: it hands the renderer a source
  filename and the renderer generates the URL for whichever shape is configured.
- **One nav, one source of truth.** The site nav, the wiki `_Sidebar.md`, the section index pages and
  the manifest are four renderings of the same page list; when a new page is added, all four need the
  row, and the checker should fail when one is missing (orphan page = error).

## Rewriting wiki-style links (the build hook)

- Map page names from **`f.src_uri`** — `os.path.basename(src)[:-3]` — never `f.name`, which has carried
  no extension since MkDocs 1.5. Skip non-documentation files, and add the aliases the wiki needs
  (`Home`/`00-Home` -> `index.md`).
- **Rewrite to the source filename** (`Page-Name.md`), not to a `.html` URL: MkDocs then validates the
  target against the file list and generates the URL itself (honouring `use_directory_urls`). Rewriting
  to `.html` is what produces hundreds of "target is not found among documentation files" warnings.
- Strip chrome links (`_Sidebar`, `_Footer`) down to their label rather than linking to a page that
  does not exist in the site build.
- Leave unknown targets untouched so the validator judges them; do not guess a target.

## Validation that actually catches a broken rewrite

1. `mkdocs.yml`:

   ```yaml
   validation:
     links:
       not_found: warn
       unrecognized_links: warn
       absolute_links: warn
     anchors: warn
   ```

2. `mkdocs build --strict` in the build script, so a warning fails the build.
3. A **post-build link check** over the generated HTML: scan every `*.html` for `href`/`src`, skip
   `http`/`https`/`mailto`/`tel`/`data`/`#`, resolve each relative target against the file that contains
   it, and require a real file (or a directory containing `index.html`). Also check `og:image`-style
   absolute asset references. When `site_url` carries a path prefix, strip that prefix before resolving
   absolute targets, or every absolute link looks broken.

Operational notes that cost real time:

- MkDocs logs to **stderr**: `cmd > log 2>&1`. `cmd 2>&1 > log` silently leaves the log empty.
- Never let the build log reach the agent's context — it runs to thousands of lines. Redirect to a file,
  then reduce to counts of distinct message shapes (normalise page paths and targets out of the text) and
  print only the summary.
- Grep one built page for a known link before believing the build: `grep -o 'href="Page-Name[^"]*"'
  site/Page-Name.html` shows `.html` when the rewrite worked and the bare page name when it did not.

## Hosting on GitHub Pages

- `actions/configure-pages` with `enablement: true` fails with *Resource not accessible by integration*
  when the repo's default workflow permissions are read-only. Create the site once out of band and re-run
  the workflow: `gh api -X POST repos/OWNER/REPO/pages -f build_type=workflow` (then confirm
  `https_enforced`).
- `gh api repos/OWNER/REPO/pages` answers 404 until the site exists — read that as "not created yet", not
  as an auth failure.
- A workflow that deploys on `push` + `workflow_dispatch` and touches a `paths:` filter will not re-run
  when you only edit a file outside that filter; use `gh run rerun <id> --failed` after fixing config.

## Attaching a custom domain (Cloudflare)

Pick the mechanism by what your credentials can actually do:

1. **Workers custom domain — works without any DNS write scope.** `wrangler deploy` with

   ```jsonc
   {
     "name": "<site>",
     "compatibility_date": "<date>",
     "assets": { "directory": "./site", "not_found_handling": "404-page" },
     "routes": [{ "pattern": "host.example.com", "custom_domain": true }]
   }
   ```

   Cloudflare provisions the DNS record and the certificate server-side, so a token with Workers scopes
   and `zone:read` is enough. Verify with an API check of the account's worker domains, then fetch the
   host.

   **Serve the built URLs the host resolves natively — directory URLs first.** The goal is a served URL
   that equals the canonical URL with no redirect hop, and the way to reach it is the *URL shape*, not a
   script: emit directory URLs (`use_directory_urls: true`) and let the asset layer do the mapping. `/`
   then resolves to `index.html` and `/Foo/` to `Foo/index.html` by itself, nothing custom runs, and the
   canonical MkDocs writes is the URL actually served. Flat `.html` output forces the fragile alternative:
   `"html_handling": "none"` to stop the 307-redirects, which also stops `/` mapping to `/index.html`, so
   a Worker script has to remap the root and the homepage's critical path is now custom code. Where that
   script does not run — an edge still holding the previous deploy, a stale or absent binding — the
   homepage falls through to the asset layer and 404s **while `/index.html` still works**. That symptom is
   unreproducible from any edge that already has the new version, so it reads as "works for me" to the
   person who deployed it. Keep the fallback only when the URL shape is externally fixed, and if you do,
   return misses with `cache-control: no-store`:

   ```js
   export default {
     async fetch(request, env) {
       const url = new URL(request.url);
       if (url.pathname === "/") url.pathname = "/index.html";
       const response = await env.ASSETS.fetch(new Request(url.toString(), request));
       if (response.status !== 404) return response;
       const miss = await env.ASSETS.fetch(new Request(new URL("/404.html", url.origin).toString(), request));
       return new Response(miss.body, { status: 404, headers: miss.headers });
     },
   };
   ```

   Verify the served shape, not just the status: `/` and a deep page path must each be 200 **with no
   redirect**, plus `sitemap.xml`, the social image, and a nonsense path returning the custom 404 body.
   `html_handling` changes are only visible through the domain, so re-fetch after redeploying — and fetch
   `/` on *every* address the hostname resolves to (`dig +short @1.1.1.1 A <host>`, then `curl --resolve`
   each): one healthy edge proves the config, not the fleet. When a reported broken path will not
   reproduce, do not add another rewrite — move the responsibility somewhere that cannot be stale.
2. **Cloudflare Pages + custom domain** — only if the token can write DNS. Without it the domain sits at
   `verification_data.error_message: "CNAME record not set"` forever, because Pages wants the CNAME
   created by hand (`dns_records:edit`).
3. Ask the user for one DNS record — last resort, one paste in their dashboard.

Traps:

- **Routes beat custom domains, and a wildcard route steals the hostname.** An existing
  `*.example.com/*` route to another Worker will serve *your* hostname with a 200 and a valid
  certificate, so the URL "works" while showing the wrong site. List them
  (`GET /zones/<zone_id>/workers/routes`, `GET /accounts/<account_id>/workers/domains`) and add the more
  specific `host.example.com/*` -> your Worker: the longest pattern wins.
- **Verify through the domain, not through the config you just wrote.** A freshly created record is not in
  the local resolver yet, so pin the edge IP: `curl --resolve host:443:<edge-ip> https://host/`. Check the
  certificate with `curl -sv … | grep -i 'subject:\|issuer:'`, a deep link for 200, and a nonsense path to
  see the custom 404.
- **A deploy job guarded on a missing secret reports success while skipping.** With no Cloudflare token
  secret the job is green, the canonical domain keeps serving the previous build, and the CI-deployed
  mirror advances — silent divergence, which is worse than a failure because every dashboard says fine.
  Guard with a step that exports a flag (`if: steps.<id>.outputs.enabled == 'true'`; `if: secrets.X != ''`
  is not valid at job or step level) and prints a `::notice::` when skipping. Then, after every content
  change, fetch a string that exists ONLY in the newest commit from **each** published surface before the
  deploy is called done — the surface you did not deploy is the one that goes stale.
- **One host, one pipeline.** Delete the abandoned project (`DELETE /accounts/<id>/pages/projects/<name>`)
  instead of leaving two half-live copies of the same content; if you keep a hosted mirror, say in the
  README which URL is canonical.

## Site plumbing worth adding once

- **A 404 override built from `config.site_url`** absolute links is useful on both the canonical domain and
  a mirror under a path prefix (the theme's default 404 links assume the publish path). Add those targets
  to the post-build link check so a renamed page cannot rot the 404 silently.
- **Link previews:** most themes emit only `description` and canonical, so add an `extrahead` override with
  `og:*` / `twitter:*`, reading a social image URL from `extra` in the site config. Generate the card
  (1200x630) with the same HTML->PNG renderer as the infographics and inspect it visually.
- **Per-page description:** set `page.meta["description"]` in the build hook — from the metadata blockquote
  (level, reading time) plus the first sentence of `## Why this matters` — when markdown metadata is not
  enabled in the config.
- **Analytics stays opt-in:** an empty `extra.analytics_code` plus an `on_post_page` hook that injects the
  script only when it is set means the published default makes zero third-party requests, and switching
  it on is a one-line config change rather than a code change.
