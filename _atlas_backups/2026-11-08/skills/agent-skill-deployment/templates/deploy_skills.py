#!/usr/bin/env python3
"""Generalised cross-runtime skill deployer — copy into a skill-suite repo.

One source of truth (`skills/`) installed into every agent runtime that should
see it, plus a manifest so drift is detectable.

Set up: edit TARGETS for the runtimes you actually run, and PLUGIN_JSON if you
want an Antigravity plugin. Then:

    python3 deploy_skills.py            # install / update every target
    python3 deploy_skills.py --check    # read-only drift report (exit 1 on drift)
    python3 deploy_skills.py --dry-run
    python3 deploy_skills.py --target hermes opencode

Copy vs symlink: use copy for runtimes that already host real directories
(Hermes, OpenCode, the Antigravity plugin) and symlink for the one whose own
library is built from symlinks (Vibe). Never hand-edit a deployed copy — edit
`skills/` and re-run this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "skills"
MANIFEST = ".skill-deploy-manifest.json"
HOME = Path.home()

# ---------------------------------------------------------------------------
# EDIT ME: one entry per runtime. `mode` is "copy" or "symlink".
# ---------------------------------------------------------------------------
TARGETS = {
    "hermes": {
        "path": HOME / ".hermes/skills/<category>",
        "mode": "copy",
    },
    "opencode": {
        "path": HOME / ".config/opencode/skills",
        "mode": "copy",
    },
    "antigravity": {
        "path": HOME / ".gemini/antigravity-cli/plugins/<plugin>/skills",
        "mode": "copy",
        "plugin_dir": HOME / ".gemini/antigravity-cli/plugins/<plugin>",
    },
    "vibe": {
        "path": HOME / ".vibe/skills",
        "mode": "symlink",
    },
}

PLUGIN_JSON = {
    "$schema": "https://antigravity.google/schemas/v1/plugin.json",
    "name": "<plugin>",
    "description": "<suite> agent store.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_files(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}


def source_skills() -> list[Path]:
    return sorted(d for d in SRC.iterdir() if d.is_dir() and (d / "SKILL.md").is_file())


def install_one(skill: Path, root: Path, mode: str, dry: bool) -> str:
    dest = root / skill.name
    if mode == "symlink" and dest.is_symlink() and dest.resolve() == skill.resolve():
        return "ok"
    if dry:
        return "update" if (dest.exists() or dest.is_symlink()) else "new"
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)
    if mode == "symlink":
        os.symlink(skill.resolve(), dest)
    else:
        shutil.copytree(skill, dest)
    return "ok"


def write_manifest(root: Path, dry: bool) -> None:
    if dry:
        return
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "source": str(SRC),
        "synced": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "skills": {s.name: {"skill_md_sha256": sha256(s / "SKILL.md")} for s in source_skills()},
    }
    (root / MANIFEST).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def register_plugin(dry: bool) -> None:
    """Write plugin.json and add the import_manifest.json entry Antigravity needs."""
    pd = TARGETS.get("antigravity", {}).get("plugin_dir")
    if not pd:
        return
    if dry:
        print(f"  would write {pd / 'plugin.json'}")
        return
    pd.mkdir(parents=True, exist_ok=True)
    pj = pd / "plugin.json"
    if not pj.is_file():
        pj.write_text(json.dumps(PLUGIN_JSON, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {pj}")
    manifest = pd.parent / "import_manifest.json"
    if not manifest.is_file():
        return
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    imports = data.setdefault("imports", [])
    if any(i.get("name") == PLUGIN_JSON["name"] for i in imports):
        return
    imports.append({
        "name": PLUGIN_JSON["name"],
        "source": "local-install",
        "importedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "components": ["installed"],
    })
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  registered {PLUGIN_JSON['name']} in {manifest}")


def do_deploy(names: list[str], dry: bool) -> int:
    skills = source_skills()
    if not skills:
        print("no skills found", file=sys.stderr)
        return 1
    print(f"deploying {len(skills)} skills to: {', '.join(names)}" + ("  [dry-run]" if dry else ""))
    for tname in names:
        t = TARGETS[tname]
        root: Path = t["path"]
        print(f"\n[{tname}] {root}  ({t['mode']})")
        counts: dict[str, int] = {}
        for s in skills:
            try:
                status = install_one(s, root, t["mode"], dry)
            except OSError as e:
                print(f"  ! {s.name}: {e}")
                counts["error"] = counts.get("error", 0) + 1
                continue
            counts[status] = counts.get(status, 0) + 1
        write_manifest(root, dry)
        print(f"  {counts}")
        if tname == "antigravity":
            register_plugin(dry)
    return 0


def do_check(names: list[str]) -> int:
    skills = source_skills()
    problems = 0
    for tname in names:
        root: Path = TARGETS[tname]["path"]
        print(f"\n[{tname}] {root}")
        if not root.is_dir():
            print("  ! target directory missing")
            problems += 1
            continue
        for s in skills:
            dest = root / s.name
            if not (dest.exists() or dest.is_symlink()):
                print(f"  ! missing: {s.name}")
                problems += 1
            elif dest.is_symlink():
                if dest.resolve() != s.resolve():
                    print(f"  ! wrong symlink target: {s.name}")
                    problems += 1
            elif dir_files(s) != dir_files(dest):
                print(f"  ! drifted: {s.name}")
                problems += 1
    if problems:
        print(f"\n{problems} drift issue(s) — run this script without --check to fix")
        return 1
    print("\nall targets in sync")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="read-only drift report")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", nargs="+", choices=sorted(TARGETS), default=sorted(TARGETS))
    args = ap.parse_args()
    if not SRC.is_dir():
        sys.exit(f"missing {SRC}")
    return do_check(args.target) if args.check else do_deploy(args.target, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
