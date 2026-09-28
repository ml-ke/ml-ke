#!/usr/bin/env python3
"""ATLAS skill-root seam check — detect shadowed / divergent skill copies.

Why: OpenCode loads skills from SEVEN locations and dedupes by `name`, with the
global config root winning:
  project  .opencode/skills, .claude/skills, .agents/skills   (walk up to worktree)
  global   ~/.config/opencode/skills  >  ~/.claude/skills  >  ~/.agents/skills
  + skills.paths from opencode.json(c) (additive, not restrictive)
`gh skill install` writes to ~/.agents/skills by default, so a library can grow a
second, SHADOWED copy of the same skill names. Shadowed copies are never loaded —
but edits made to them look successful and change nothing (real case: a
choicebank-baas copy edited 5 days after the loaded one, silently inert).

This is a collection-seam failure (arxiv 2609.13321: "regime gating" — aliases /
duplicate copies; noncanonical routes 0/32 -> 15/32 in the controlled test).

Exit code 0 always: a reporting tool for the weekly pipeline.

Usage:
  python3 ~/.hermes/scripts/skill_root_seam_check.py
  python3 ~/.hermes/scripts/skill_root_seam_check.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

HOME = pathlib.Path.home()

# Load order per OpenCode docs; earlier entries win a name collision.
ROOTS = [
    ("opencode-global", HOME / ".config/opencode/skills"),
    ("claude-global", HOME / ".claude/skills"),
    ("agents-global", HOME / ".agents/skills"),
]


def skill_name(skill_md: pathlib.Path) -> str:
    m = re.search(r"^name:\s*['\"]?([^'\"\n]+)", skill_md.read_text(errors="ignore")[:2000], re.M)
    return m.group(1).strip() if m else skill_md.parent.name


def content_hash(d: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    maps: list[tuple[str, dict[str, pathlib.Path]]] = []
    for label, root in ROOTS:
        if not root.is_dir():
            continue
        found = {}
        for skill_md in root.rglob("SKILL.md"):
            if ".archive" in skill_md.parts or "node_modules" in skill_md.parts:
                continue
            found.setdefault(skill_name(skill_md), skill_md.parent)
        maps.append((label, found))

    if not maps:
        print("[skip] no skill roots found")
        return 0

    print("=== skill-root seam check ===")
    for label, m in maps:
        print(f"  {label:18} {len(m):4} skills")

    owners: dict[str, list[tuple[str, pathlib.Path]]] = {}
    for label, m in maps:
        for name, d in m.items():
            owners.setdefault(name, []).append((label, d))

    dupes = {n: v for n, v in owners.items() if len(v) > 1}
    divergent, identical = [], []
    for name, copies in sorted(dupes.items()):
        hashes = {content_hash(d) for _, d in copies}
        if len(hashes) == 1:
            identical.append(name)
        else:
            entry = []
            for label, d in copies:
                skill_md = d / "SKILL.md"
                entry.append({
                    "root": label,
                    "path": str(d),
                    "mtime": skill_md.stat().st_mtime,
                    "loaded": label == copies[0][0],
                })
            divergent.append({"name": name, "copies": entry, "winner": copies[0][0]})

    print(f"\n  names in more than one root : {len(dupes)}  (shadowed, since the first root wins)")
    print(f"  byte-identical duplicates   : {len(identical)}")
    print(f"  DIVERGENT duplicates        : {len(divergent)}  <-- review these")

    for d in divergent:
        print(f"\n  ! {d['name']} (loaded from {d['winner']})")
        for c in sorted(d["copies"], key=lambda c: c["mtime"]):
            tag = "LOADED " if c["loaded"] else "shadowed"
            import datetime
            ts = datetime.datetime.fromtimestamp(c["mtime"]).strftime("%Y-%m-%d %H:%M")
            print(f"      {tag} {ts}  {c['path']}")
        newer_shadow = max((c for c in d["copies"] if not c["loaded"]), key=lambda c: c["mtime"], default=None)
        if newer_shadow and newer_shadow["mtime"] > max(c["mtime"] for c in d["copies"] if c["loaded"]):
            print("      -> the SHADOWED copy is newer: edits there are silently inert. Sync into the loaded root.")

    if args.json:
        print("\nJSON:")
        print(json.dumps({"roots": {l: len(m) for l, m in maps}, "duplicates": len(dupes),
                          "identical": len(identical), "divergent": divergent}, indent=1))

    print("\n  Verdict: duplicates are shadowed (deduped by name, first root wins) — they cost"
          "\n  discovery noise and silently swallow edits, not runtime routing. Prune with"
          "\n  `gh skill` bookkeeping in mind, or keep them and let this check catch drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
