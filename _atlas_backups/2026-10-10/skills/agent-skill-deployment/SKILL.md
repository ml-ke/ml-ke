---
name: agent-skill-deployment
description: "Use when deploying or publishing multi-runtime agent skills."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [skills, deployment, opencode, antigravity, vibe, portable-skills]
    related_skills: [hermes-agent-skill-authoring, hermes-maintenance, find-skills]
---

# Agent Skill Deployment (Hermes, OpenCode, Antigravity, Vibe)

Four agent CLIs run on this machine and each one has its own skill directory and
its own loader. There is no shared registry: a skill authored for Hermes is
invisible to the others until you put a copy where they look. This skill is the
map of where each runtime looks, and the procedure for installing one source of
truth into all of them without letting the copies drift.

## When to Use

- The user asks to add, improve, or install skills for OpenCode, Antigravity CLI,
  or Mistral Vibe — not only Hermes.
- You are authoring a `SKILL.md` that has to load in more than one runtime.
- A skill works in one agent but "isn't showing up" in another.
- You need to prove a deployed copy still matches the source it came from.
- You are asked to build a *suite* of related skills for all the agents at once.

**Don't use for:** authoring skills inside the `hermes-agent` package tree
(`hermes-agent-skill-authoring`), backing up Hermes state or sudo setup
(`hermes-maintenance`), or finding a prebuilt third-party skill to install
(`find-skills`).

## Runtime Map

| Runtime | What it loads | Where | Config that points there |
|---------|---------------|-------|--------------------------|
| Hermes Agent | `~/.hermes/skills/**/SKILL.md` | nested category dirs allowed | none — auto-discovered |
| OpenCode | `~/.config/opencode/skills/<name>/SKILL.md` | flat, one dir per skill | `opencode.json(c)` → `"skills": {"paths": [...]}` |
| Antigravity CLI | `~/.gemini/antigravity-cli/plugins/<plugin>/skills/<name>/` | flat, inside a plugin | `plugins/<plugin>/plugin.json` + `import_manifest.json` |
| Mistral Vibe | `~/.vibe/skills/<name>` | flat | `~/.vibe/config.toml` |
| Shared hub | `~/.agents/skills/<name>` | flat | `.skill-lock.json` (written by `npx skills`) |

The two Antigravity plugins on this machine (`opencode-skills`, `hermes-skills`)
are **copies** of the OpenCode and Hermes libraries, not symlinks. Copying a skill
into `~/.config/opencode/skills/` therefore does *not* make it visible to
Antigravity — you must also place it in a plugin's `skills/` directory.
Vibe's own directory is built from symlinks into `~/.agents/skills/`.

Full per-runtime detail (config keys, verification commands, the import-manifest
shape, the existing Hermes→OpenCode bridge): `references/runtime-map.md`.

## Authoring for Portability

Write to the **intersection** of what every runtime reads, not to Hermes' full
vocabulary.

- Top-level frontmatter: only `name`, `description`, `license`, `compatibility`,
  `metadata`. OpenCode reads those five and ignores the rest; anything Hermes- or
  platform-specific belongs *inside* `metadata`, never as a new top-level key.
- `name` must equal the directory name and match `^[a-z0-9]+(-[a-z0-9]+)*$`.
- This portable shape applies to skills that ship to other runtimes. A skill that
  will live only in `~/.hermes/skills/` should instead follow Hermes' peer shape:
  top-level `version`, `author`, `license` plus `metadata.hermes.{tags,
  related_skills}` — Hermes' own lint flags the portable shape as missing metadata.
- Hermes renders only the **first 57 characters** of `description` in its system
  prompt skill index. The opening clause must carry the trigger on its own; the
  rest of the sentence only helps once another runtime or a human reads it.
- Deep material goes in `references/<topic>.md`, reusable artifacts in
  `templates/`, runnable helpers in `scripts/`. Keep `SKILL.md` under ~500 lines.
- Paths differ per runtime, so a skill must never hardcode its own install
  location or reference a sibling by absolute path.

## The One-Source-Of-Truth Pattern

Do not edit four copies. Keep one git repo as the source and deploy from it:

```
<suite>/
├── AGENTS.md            # authoring spec: hard rules, structure, style
├── skills/<name>/SKILL.md   (+ references/ templates/ scripts/)
├── scripts/validate.py  # enforces the open standard + repo rules
└── deploy.py            # installs skills/ into every runtime
```

Procedure:

1. **Author** every skill under `skills/` in the repo. Fix the full set of skill
   names up front — siblings cross-reference each other by exact name, and a
   validator that checks `related-skills` resolve will fail on a half-built set.
2. **Pre-create a valid stub `SKILL.md`** for each planned name before fanning the
   work out to parallel authors. A stub keeps the repo's name set complete so each
   author can run the validator and get a clean pass instead of a cascade of
   "related-skill not in repo". Authors then read the stub and overwrite it (a
   plain `write_file` refuses to clobber an unread existing file).
3. **Validate**: a script that checks name/dir match, the name regex, description
   length, portable-field-only frontmatter, no raw emoji, size, and that every
   `references/…` link resolves.
4. **Deploy** with a script that copies into Hermes, OpenCode, and the Antigravity
   plugin, and symlinks into Vibe. Copy is safer for the three that already host
   real directories; symlink is what Vibe's own directory already uses.
5. **Register the Antigravity plugin** once: write `plugins/<plugin>/plugin.json`
   and add a matching entry to `plugins/import_manifest.json` (an unregistered
   plugin directory is not picked up).
6. **Drift-check** after any edit: recompute each source file's hash and compare
   against a manifest written at deploy time. Provide `--check` that exits
   non-zero so it can gate a commit or a cron.

A generalised deployer to copy from: `templates/deploy_skills.py`.

## Verifying a Deployment

The source of truth for "did it land" is the file on disk at the runtime's own
path — not the deploy script's exit code.

```bash
# Hermes (nested categories are fine)
find ~/.hermes/skills -name SKILL.md -path '*<suite>*'
# OpenCode
ls ~/.config/opencode/skills/<name>/
# Antigravity plugin
ls ~/.gemini/antigravity-cli/plugins/<plugin>/skills/<name>/
# Vibe (expect a symlink pointing back at the source)
readlink -f ~/.vibe/skills/<name>
```

Then confirm the runtime can actually see it: the current session's skill loader
is initialised at startup, so a newly deployed skill will **not** appear in
`skills_list` until a new session. Verify by reading the file, not by listing
skills in the session that deployed it.

## Vetting An Upstream Skill Repo

When the user supplies research pointing at repositories to learn from or adopt,
verify each one exists and has traction before building on its architecture.
LLM-generated research routinely cites repos that do not exist:

```bash
gh api repos/<owner>/<repo> --jq '.full_name + " " + (.stargazers_count|tostring)'
```

Confirm the repo's actual layout too — the marketingskills-style architecture
(`skills/<name>/SKILL.md` plus a shared context doc under `.agents/`) is the
reference pattern to model a suite on, and it is worth checking what a cited repo
really contains rather than trusting the summary of it.

## Packaging and Publishing a Suite Publicly

A suite that ships to other people has a second obligation beyond loading
correctly: it must credit what it did not write and never vendor it.

- **Registry, not vendoring.** Keep curated third-party skills in a
  `registry/sources.json` file with `repo`, `license`, `ref`, `why`, `include`.
  An installer script shallow-clones each source and copies only the named
  skills into the runtimes at install time. Add the clone cache to `.gitignore`;
  committing other people's trees is both a licensing problem and a review
  burden.
- **Exclude the overlap.** If a curated source has a skill your suite already
  authors, leave it out. Two skills competing for one trigger make routing
  worse, not better. Enforce it in a test.
- **Generate attribution, do not hand-write it.** A script renders `CREDITS.md`
  and a machine-readable `installed.json` from the registry, with a `--check`
  mode that exits non-zero when the committed files are stale. Hand-maintained
  credits drift the moment a source is added.
- **Keep `LICENSE` verbatim.** Appending a carve-out paragraph to the MIT text
  stops GitHub's licence detector recognising it. Put the carve-out in
  `NOTICE.md` and reference that from the README.
- **Guard every cited skill name with a test.** A catalog or router doc that
  names a skill you did not install makes an agent silently do nothing. Parse
  the backticked tokens out of the docs, allowlist the non-skill tokens, and
  assert each one resolves against the installed corpus. This catches drift that
  no amount of careful writing prevents.
- **Self-contained CLI beats a build step.** A single dependency-free `.mjs`
  with a `bin` entry works under `npx` with no install, no bundler and no
  lockfile. Add `files` to `package.json` so the tarball contains only what
  ships, and verify with `npm pack --dry-run`.
- **Never commit a generated artifact that describes your machine.** Anything
  produced by scanning the local system — an inventory of installed skills, a
  host list, an account or target list — publishes the private project names
  inside it the moment the repo is public. Commit a curated allowlist instead
  and keep the generated snapshot gitignored, with the validator merging both so
  CI validates against the allowlist and a developer machine validates against
  everything it has.
- **Prove the allowlist stands alone.** Move the gitignored file aside and run
  the full validator and test suite. If anything fails, the committed allowlist
  is incomplete and the public repo does not build from a fresh clone.
- **Read exclusion lists from a gitignored file.** A filter script that
  hardcodes the private names it is meant to hide republishes them in the clear.

Full procedure, with the commands for auditing the publish artifact, verifying a
fresh clone, and purging a leak from git history:
`references/publishing-disclosure.md`.

## Common Pitfalls

1. **Assuming one deployment covers every runtime.** There is no shared skill
   registry. Deploy Hermes, OpenCode, the Antigravity plugin, and Vibe separately;
   Antigravity in particular only sees what is inside its own plugin directories.
2. **Adding a plugin directory but not registering it.** Antigravity resolves
   plugins through `import_manifest.json`; a bare directory with skills in it is
   invisible.
3. **Shipping Hermes-only frontmatter.** A `platforms:` or custom top-level key
   is dropped or rejected elsewhere. Put non-portable data under `metadata`.
4. **Burying the trigger past character 57.** Hermes truncates the description in
   its routing index, so a description that only becomes specific later never
   routes.
5. **Editing a deployed copy.** The next deploy overwrites it and the runtimes
   silently diverge in between. Edit the source repo and re-deploy.
6. **Fanning out authors onto an incomplete name set.** Validators that check
   sibling references fail on every skill until the last one lands. Stub the
   whole set first.
7. **Trusting a deploy exit code as proof.** Read the file at the target path; a
   copy can land in the wrong directory and still report success.
8. **Writing a description longer than 60 characters on create.**
   `skill_manage(action='create')` refuses it (the index shows 57 + "...");
   create with a short trigger and extend the description afterwards with
   `patch`, which carries no such limit.
9. **Grepping the CLI's skill listing to prove a skill is present.** Hermes'
   `skills list` table truncates the name column to about 14 characters, so a
   grep for a full skill name returns nothing and looks like a failed deploy.
   Use a prefix short enough to survive truncation, or ask the loader directly.
10. **`node --test tests/` on modern Node.** Node 26 treats the directory as a
    module and fails with MODULE_NOT_FOUND; bare `node --test` auto-discovers
    into any cloned repo under the tree. Use an explicit glob:
    `node --test tests/*.test.mjs`.
11. **Writing YAML lists as flow sequences.** `related-skills: [a, b]` is valid
    YAML that PyYAML reads correctly and a scalar-only reader silently turns
    into the string `"[a, b]"`. If you re-implement frontmatter parsing, handle
    `[` ... `]` explicitly.
12. **Moving the suite directory.** Every symlinked deployment (Vibe, and any
    external-skill symlinks) breaks, and any script using
    `Path(__file__).parent` needs a `.parent` added if it moves a level deeper.
    Re-run the deployer after any move and confirm with the drift check.
13. **Unquoted colons in `description`.** `description: Do X: then Y` is a YAML
    error (`mapping values are not allowed here`). Quote any description
    containing `: `.
14. **Tests that assume an installed environment.** A CLI test asserting
    `doctor` succeeds fails on CI where nothing is installed yet. Make the exit
    code part of the test, not an implicit precondition.
15. **Assuming a new commit unpublishes a leaked file.** Deleting it from the
    tip leaves the blob reachable in every earlier commit, and `git log -- <path>`
    on the tip says nothing while `git grep $(git rev-list --all)` still finds
    it. History has to be rewritten and force-pushed; see
    `references/publishing-disclosure.md`.
16. **Listing a directory in `package.json` `files`.** A directory entry ships
    everything inside it, including generated local artifacts that `.gitignore`
    hides from git but which still exist on disk at pack time. List explicit file
    paths and assert on `npm pack --dry-run --json`.
17. **Adding a path to `.gitignore` to untrack it.** Ignoring a path does not
    untrack a file git already knows about; run `git rm --cached` as well, or it
    keeps being committed.

## Verification Checklist

- [ ] Each skill's directory name equals its `name` frontmatter field.
- [ ] Frontmatter uses only portable top-level keys.
- [ ] Description trigger fits in the first 57 characters.
- [ ] Every runtime's target path contains the skill (read the file, not the log).
- [ ] Vibe entry resolves via `readlink -f` to the source directory.
- [ ] Antigravity `plugin.json` exists and the plugin is in `import_manifest.json`.
- [ ] Drift check passes against the deploy manifest after the final edit.
- [ ] Curated sources are recorded with a real upstream licence and are not
      vendored into the repo.
- [ ] Generated attribution (`CREDITS.md`) is in sync with the registry.
- [ ] Every skill named in the catalog, router and docs actually resolves
      against the installed corpus (guarded by a test).
- [ ] No committed file is generated from a scan of the local machine; the
      committed allowlist alone passes the validator with the local file absent.
- [ ] `npm pack --dry-run --json` lists no local-only artifact, and no private
      name is reachable anywhere in history.

## References

- `references/runtime-map.md` — per-runtime discovery paths, config keys,
  verification commands, the Antigravity import-manifest shape, and the existing
  Hermes→OpenCode bridge script.
- `templates/deploy_skills.py` — a generalised cross-runtime deployer to copy.
- `references/publishing-disclosure.md` — splitting generated registries by
  disclosure, auditing the publish artifact, purging a leak from git history,
  and the npm publish path.
