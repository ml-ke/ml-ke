# GitHub wiki mechanics

## Enabling and verifying

- New repos created with `gh repo create` already have the wiki enabled; `--enable-wiki` is deprecated
  (the CLI warns and proceeds).
- Verify or flip it over the API — `gh` has no wiki subcommand:

  ```bash
  gh api repos/<owner>/<repo> --jq '.has_wiki'                 # true/false
  gh api -X PATCH repos/<owner>/<repo> -f has_wiki=true        # enable
  ```

## The one-time prerequisite that blocks automation

GitHub creates the wiki's git repository **only after the first page exists, created through the web
UI**. Before that:

- `git clone git@github.com:<owner>/<repo>.wiki.git` → `ERROR: Repository not found`.
- `git -c credential.helper='!gh auth git-credential' clone https://github.com/<owner>/<repo>.wiki.git`
  → `remote: Repository not found`.
- No REST or GraphQL endpoint exists for wiki pages; the API exposes `has_wiki` and nothing more.

So: `https://github.com/<owner>/<repo>/wiki` → **Create the first page** → save → then push. Treat it as
one interactive step for the user, not as an auth problem to debug. An unauthenticated browser session
cannot do it either (GitHub shows "Sign in" instead of the wiki).

## Page-name and structure conventions

- **Flat namespace.** `05-Checks-That-Actually-Matter.md` becomes `/wiki/05-Checks-That-Actually-Matter`;
  the filename is the page title and the URL. There are no directories.
- **`Home.md` is the landing page.** Name the generated home page `00-Home.md` and copy it to `Home.md`
  during sync, so repo numbering stays consistent.
- **`_Sidebar.md` and `_Footer.md`** are injected into every page — use them for the navigation tree and
  a licence/verify-note footer. Exact filenames only.
- **No H1 in the body**: the wiki renders the title above the content.
- **Links** are bare page names — `[Context engineering](02-Context-Engineering)`. Never link `.md`, and
  `[[wikilinks]]` do not render on GitHub.
- **Images must be absolute URLs.** A committed asset referenced as
  `raw.githubusercontent.com/<owner>/<repo>/<branch>/assets/x.png` works for a public repo and keeps one
  asset copy for wiki, repo and docs-site renders.
- **Numbering is the navigation.** Section prefix + lesson number, globally unique, with the section
  index page (`01-Foundations`) sitting beside the lessons (`01-The-Agent-Loop`).

## Sync pattern

The wiki is a separate git repo. Treat the main repo's `wiki/` as source of truth and the wiki as a
rendering. A sync script should:

1. Clone `https://github.com/<owner>/<repo>.wiki.git`, failing with the UI instruction above rather
   than a bare stack trace.
2. Copy every `wiki/*.md` in, then copy `00-Home.md` → `Home.md`.
3. Delete wiki pages whose source no longer exists, skipping `Home.md`.
4. `git add -A`, exit cleanly when nothing changed (idempotent re-runs).
5. Commit with a dated message and push with a `master`-then-`HEAD` fallback.
6. Support `--dry-run` that prints `git diff --cached --stat` and stops.

Run the docs checker over `wiki/` in the same CI workflow, so a sync can never publish a page with
broken links or a missing asset.

## What to keep in the main repo

Page sources, the asset generator, the interactive tools, the manifest and the checker all belong in
main. That way a reader without a GitHub account (or an offline reader, or a later docs-site migration)
still has everything, and any page can be reviewed as a normal pull request instead of an unreviewable
wiki edit.
