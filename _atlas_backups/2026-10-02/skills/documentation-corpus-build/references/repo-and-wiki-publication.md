# Repo and wiki publication

## Create the repo

```bash
gh auth status                                    # confirm account and scopes
gh api repos/<owner>/<name> --jq .full_name       # 404 means the name is free
gh repo create <owner>/<name> --public --description "..."
gh api -X PATCH repos/<owner>/<name> -f description="longer description"
gh api -X PUT repos/<owner>/<name>/topics -f names[]=docs -f names[]=wiki -f names[]=ci-cd
```

- `gh repo create <name>` without `--source`/`--clone` creates the remote only and adds no git remote locally: `git remote add origin git@github.com:<owner>/<name>.git` before the first push, or the push fails with "Could not read from remote repository".
- `--enable-wiki` is deprecated: wikis are enabled by default. The useful flag is `--disable-wiki`.
- Confirm what you got: `gh api repos/<owner>/<name> --jq '{private, has_wiki, default_branch}'`.
- SSH needs `-o StrictHostKeyChecking=accept-new -o BatchMode=yes` (or a warm known_hosts) in unattended runs, and `gh` may authenticate git over HTTPS rather than SSH — check `gh auth status` for "Git operations protocol".

## The wiki initialisation trap

A repository's wiki is a **separate git repository** (`<owner>/<repo>.wiki.git`) with a flat page namespace, and **it does not exist until the first page is created through the web UI**. Until then every clone and push fails with `Repository not found` — over SSH and HTTPS, authenticated or not — and there is no API for creating wiki pages.

So plan for exactly one interactive step: the user opens `https://github.com/<owner>/<repo>/wiki`, clicks "Create the first page", saves anything, and from then on everything is scriptable. Two ways to handle it:

1. Ship the sync as a script (`templates/push_wiki.sh`) and hand the user one command to run after their click.
2. If the agent browser is logged into GitHub in that session, do the click yourself — check for a signed-in state first (an unauthenticated browser shows only the sign-in page), and do not promise it before checking.

Do not present this as an agent failure or the wiki being broken; it is a GitHub mechanic, and the one-click step is the fix.

## Page layout rules the wiki imposes

- The landing page must be `Home.md`. If your sources are numbered, copy the home source to `Home.md` during sync.
- `_Sidebar.md` and `_Footer.md` are injected into every page; they are ordinary markdown files in the wiki repo. A sidebar listing every section and page is the difference between a wiki and a pile of files.
- No directories: filename = page title = URL. Cross-link by page name with no `.md`.
- Do not start a page with an `# H1` — the wiki renders the title itself, so an H1 duplicates it.
- Mermaid fences render in wiki pages. Images must be absolute URLs (raw GitHub URLs from the main repo work); relative paths do not resolve.
- Keep the sources in a `wiki/` directory in the main repo and sync outward: that way the corpus has reviewable diffs, one source of truth, and a CI gate.

## Sync script shape

The script must be idempotent because it runs repeatedly as the corpus grows:

1. Clone (or pull) `<repo>.wiki.git`.
2. Copy every source page in; copy the home source to `Home.md`.
3. `git rm` pages whose source no longer exists.
4. `git add -A`, then exit early if `git diff --cached --quiet` — nothing to push.
5. Commit and push. Support `--dry-run` so the user can inspect the diff first.

Handle the init trap with a clear error message rather than a stack trace, so a re-run after the first page just works. `templates/push_wiki.sh` is a working version.

## Verify after pushing

- `curl -sS -o /dev/null -w "%{http_code} %{size_download}" -L <raw asset URL>` → 200 with a plausible size means every embedded image will render.
- `gh run list --repo <owner>/<repo> --limit 3` → confirm the CI gate ran; `gh run view <id> --log-failed` → read the actual reason rather than assuming. A gate failing only on forward links to pages not yet written is correct behaviour.
- Re-run the external link check: own-repo asset URLs stop 404ing once the main repo is pushed.

## Doc and code licensing

Corpora mix prose and tooling, and a single MIT file misrepresents the prose. Split it: `LICENSE` (MIT) for `scripts/` and `tools/` with an explicit scope note, and `LICENSE-CONTENT.md` for the prose and generated images under CC BY 4.0 with the attribution line you want people to use. State the split in the README and the contributing guide, plus a line that the content is not legal, financial or security advice for the reader's specific system.

## Final report to the user

State plainly: the repo URL, what is verified (gate status, mermaid parse count, link-check result), what is deliberately deferred, whether the repo is public or private, and the single interactive step they still own. Never present a red gate as green, and never claim the wiki is live before the first page exists.
