# Runtime map — per-runtime detail

Companion to the Runtime Map table in SKILL.md. Paths are for a Linux desktop
install with the user `pro-g`; substitute your own paths when reusing this.

## Hermes Agent

- Discovery: `~/.hermes/skills/**/SKILL.md`, recursive — nested category dirs are
  fine (`business-development/<name>/`, `mlops/inference/<name>/`).
- No config file needed; the loader is initialised at session start, so a newly
  deployed skill appears in a **new** session, not the one that deployed it.
- Only the first 57 characters of `description` are rendered into the system-prompt
  skill index (`SKILL_PROMPT_DESC_LIMIT = 60`, truncated to `57 + "..."`).
- `skill_manage(action='create')` writes here and **refuses descriptions over 60
  chars**. `action='patch'` carries no such limit — create short, then extend.
- Curator state sits alongside the skills: `.bundled_manifest`, `.usage.json`,
  `.curator_ledger.jsonl`, `.curator_backups/`. Do not hand-edit those.

## OpenCode

- Discovery: `~/.config/opencode/skills/<name>/SKILL.md` — flat, one directory
  per skill.
- Registered in `~/.config/opencode/opencode.jsonc`:
  `"skills": { "paths": ["<absolute path to the skills dir>"] }`.
- Reads five frontmatter fields only: `name`, `description`, `license`,
  `compatibility`, `metadata`. Anything else is outside its schema.
- `permission.skill` in the same config gates skills by glob (`allow` / `ask` /
  `deny`) — relevant when adding skills that execute commands or burn resources.
- `~/.config/opencode/skills-REF/` holds `SPECIFICATION.md` (the open-standard
  spec text) and `template-SKILL.md` — the authoritative local copy of the rules.
- Project-local skills can also live under a `.agents/skills/` directory in the
  working tree.

## Antigravity CLI

- Discovery: `~/.gemini/antigravity-cli/plugins/<plugin>/skills/<name>/`.
- A plugin needs **both**:
  1. `<plugin>/plugin.json`
     (`$schema`, `name`, `description`), and
  2. an entry in `~/.gemini/antigravity-cli/import_manifest.json`:

     ```json
     { "imports": [ { "name": "<plugin>", "source": "local-install",
                       "importedAt": "<ISO8601>", "components": ["installed"] } ] }
     ```

  A plugin directory without the manifest entry is not picked up.
- Built-in skills ship at `~/.gemini/antigravity-cli/builtin/skills/`
  (`automation`, `antigravity_guide`, `plugin`, ...) and use the same
  `name` / `description` / `metadata` frontmatter.
- The `opencode-skills` and `hermes-skills` plugin directories are **copies** of
  the OpenCode and Hermes libraries. They do not follow later edits to those
  libraries; treat them as a third, separate deploy target.
- Binary: `~/.local/bin/agy`. Settings: `~/.gemini/antigravity-cli/settings.json`
  (`trustedWorkspaces`, `toolPermission`).

## Mistral Vibe

- Discovery: `~/.vibe/skills/<name>` — flat, and the existing entries are
  **symlinks** pointing into `~/.agents/skills/`.
- Config: `~/.vibe/config.toml` (theme, active model, providers, `[tools.bash]`
  allowlist).
- The bash allowlist is restrictive by default: a skill that needs a command to
  run must have that command added to `[tools.bash].allowlist` first.

## Shared hub — `~/.agents/skills/`

- Flat library maintained by the `npx skills` CLI (the open agent-skills package
  manager). Skills install here first and are then linked into each runtime.
- `.skill-lock.json` records `source`, `sourceType`, `sourceUrl`, `skillPath`,
  `skillFolderHash`, `installedAt` per skill — the provenance trail for anything
  installed from a GitHub repo (`npx skills add <owner>/<repo>[@skill]`).
- `lastSelectedAgents` lists the runtimes the CLI offers to wire up
  (`hermes-agent`, `mistral-vibe`, `opencode`, ...).

## Existing bridge script

`~/.hermes/scripts/opencode_skills_bridge.py` is a worked example of the
Hermes → OpenCode direction:

- A curated `(hermes-relative-path, opencode-name)` list, so only chosen skills
  cross over.
- Rewrites frontmatter down to OpenCode's five fields, preserving the source's
  routing metadata inside `metadata`.
- Stamps provenance — `metadata.atlas-source`, `-source-sha256`, `-synced` — which
  makes a stale or tampered copy detectable (supply-chain / update-drift hygiene).
- Modes: `--list`, `--dry-run`, `--check` (read-only drift report, always exit 0),
  `-v`.
- Re-execs itself under an interpreter that has PyYAML rather than failing, since
  the bundled `python3` on PATH may lack it.
