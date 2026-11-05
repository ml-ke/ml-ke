# Publishing a skill suite without leaking your machine

Depth for the disclosure rules in SKILL.md: separating generated registries by
what they reveal, auditing the artifact before it ships, purging a leak from
already-pushed history, and the npm publish path.

## The disclosure rule

Anything produced by scanning the local machine describes the local machine:
installed-skill inventories, host lists, account or target lists, dependency
snapshots. Committing one publishes the private project names inside it. Two
defences have to hold at once, because either alone leaks:

1. Do not commit generated machine-specific files.
2. Do not let the packaging step pick them up off disk.

`.gitignore` covers only the first, and only for files git does not already
track.

## Two registries, split by disclosure

When a validator needs to resolve cross-references against a wider corpus, split
the registry rather than choosing between leaking and not validating:

```
catalog/known-skills.public.txt   COMMITTED   curated allowlist, public names only
catalog/known-skills.local.txt    GITIGNORED  generated snapshot of this machine
```

- Build the public allowlist from what the repo itself references, intersected
  with the local corpus, minus private name families. It stays small and it is
  reviewable.
- Build the local snapshot from a live scan of every runtime's skill root.
- Have the loader merge both, tolerating a missing local file. CI and a fresh
  clone then validate against the allowlist; a developer machine validates
  against everything installed.
- Keep the private-name exclusion list in a gitignored file (for example
  `catalog/.private-prefixes`, one prefix per line). A public script that
  hardcodes the names it filters has published them.

### Prove the allowlist stands alone

This is the check that matters, and it is easy to skip because the repo passes
locally where both files exist:

```bash
mv catalog/known-skills.local.txt /tmp/ksl.bak
python3 scripts/validate.py          # must pass
node --test tests/*.test.mjs         # must be 0 failures
python3 scripts/gen-known-skills.py --check
mv /tmp/ksl.bak catalog/known-skills.local.txt
```

## Audit the publish artifact

Never eyeball the tarball. Assert on it:

```bash
npm pack --dry-run --json | python3 -c "
import json,sys
fs=[f['path'] for f in json.load(sys.stdin)[0]['files']]
print('files:',len(fs))
print('leaks:',[f for f in fs if 'local' in f or f.startswith(('.agents','vendor','node_modules'))])
"
```

List explicit file paths in `files`, not directories. A directory entry ships
everything inside it at pack time, including generated files that `.gitignore`
keeps out of git but which are still on disk.

## Purge a leak from pushed history

Deleting the file at the tip does not unpublish it. Confirm the scope first —
this greps content across every reachable commit, which `git log` on a path does
not:

```bash
for pat in private-name-1 private-name-2; do
  echo "$pat: $(git grep -li "$pat" $(git rev-list --all) 2>/dev/null | wc -l) commit(s)"
done
```

Remove a whole file from every revision:

```bash
export FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/leaked-file' --prune-empty -- --all
rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now
git push --force origin main
```

To redact a string from a file that must stay, use `--tree-filter` with a helper
script rather than an inline `sed`. Backticks and nested quotes in an inline
filter are re-parsed by a subshell and corrupt the command; a script file avoids
the whole class of quoting bug. Match a backtick with `\x60` so no literal
backtick appears on the command line:

```sh
#!/bin/sh
[ -f catalog/CATALOG.md ] && sed -i 's/, \x60private-skill-name\x60//' catalog/CATALOG.md
exit 0
```

```bash
git filter-branch --force --tree-filter 'sh /path/to/helper.sh' -- --all
```

Afterwards re-run the scan and `git fetch` before trusting it: verify against
`$(git rev-list origin/main)`, not just the local refs.

Rewriting history is only safe on a repo that is yours and that nobody has
cloned or based work on. On a brand-new public repo it is the correct fix; on a
shared one, removing the exposed data is the fix and history is left alone.

## npm publish

- Check the name is free: `npm view <name> version` exits 404 when it is.
- npm takes no GitHub credential on the CLI, but the web flow signs in *with*
  GitHub, which is the closest thing to linking the accounts:
  `npm login --auth-type=web` prints a URL; the account is created or linked on
  the npm site via "Sign in with GitHub".
- Run that interactively in the user's own terminal. If the browser step is not
  completed the flow degrades to a `Username:` prompt, so kill it and re-run
  rather than feeding it credentials.
- After the first publish, configure a **Trusted Publisher** in the package
  settings pointing at the repo, and publish from CI over OIDC. That removes the
  long-lived token entirely.
- `LICENSE` must remain verbatim for GitHub to detect the licence; carve-outs for
  vendored or curated third-party work belong in `NOTICE.md`.
