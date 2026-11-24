#!/usr/bin/env python3
"""skill_reference_check.py — resolve the file pointers inside every SKILL.md.

Pipeline step 8 of the ATLAS weekly skill sweep. A skill's procedural value is
carried by the paths it names ("run `scripts/x.sh`", "see
`references/y.md`"); when a pointer does not resolve, the instruction becomes a
hunt, and the agent burns context searching for a file that was renamed,
moved into a sibling skill, or archived by the curator. SkillSeam (arxiv
2609.13321) measures that as the "dangling anchor" seam: +64% loaded tokens and
-3.1pp accuracy.

What it checks (skill-relative, the spec's own directories only):

    `references/...`  `scripts/...`  `templates/...`

`assets/` is intentionally NOT queued — skills routinely use it to name paths
inside a *project* the skill operates on (e.g. a blog repo's `assets/img/`),
not inside the skill directory, which makes it a false-positive factory.

Verdicts, in resolution order:
  OK          resolves relative to the skill directory
  MIS-SCOPED  absent locally, but a file with that basename exists elsewhere in
              the live library -> the pointer text is wrong, the capability is
              not. This is the common real bug (cross-skill reference written
              as if local).
  ARCHIVED    only exists under `.archive/` (curator prune). Point at it as
              archived or `hermes curator restore` — never leave it silent.
  DANGLING    nowhere on disk: an aspirational pointer, or the file was lost.

Always exits 0 — a reporter, not a gate. Reads only; never edits a skill.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HOME = pathlib.Path.home()
LIVE_ROOTS = [HOME / ".hermes/skills", HOME / ".config/opencode/skills"]
ARCHIVE_DIRNAME = ".archive"
REQUIRED_DIRS = ("references", "scripts", "templates")

REF = re.compile(r"`((?:\./)?(?:references|scripts|templates)/[^`\s'\"]+)`")
TEMPLATE_TOKEN = re.compile(r"[<>{}$*]|YYYY|\.\.\.|\bN\b")
EXCLUDED_DIRS = {".archive", ".hub", ".git", ".gitignore", "__pycache__"}


def bodies(root: pathlib.Path):
    """Yield (skill_md_path, text) for discoverable skills under root."""
    for path in sorted(root.rglob("SKILL.md")):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        try:
            yield path, path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def basename_index(bases: list[pathlib.Path]) -> dict[str, list[pathlib.Path]]:
    """basename -> paths, over the given skill trees (live roots or archive trees)."""
    index: dict[str, list[pathlib.Path]] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for sub in REQUIRED_DIRS:
            for d in base.rglob(sub):
                if not d.is_dir():
                    continue
                if any(part in EXCLUDED_DIRS for part in d.relative_to(base).parts):
                    continue
                for f in d.iterdir():
                    if f.is_file():
                        index.setdefault(f.name, []).append(f)
    return index


def rel_to_roots(p: pathlib.Path) -> str:
    for root in LIVE_ROOTS:
        try:
            return root.name + "/" + p.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--root", action="append", help="override skill root (repeatable)")
    args = ap.parse_args()

    roots = [pathlib.Path(r).expanduser() for r in args.root] if args.root else LIVE_ROOTS
    live_index = basename_index(roots)
    arch_index = basename_index([r / ARCHIVE_DIRNAME for r in roots])

    rows = []
    counts = {"OK": 0, "MIS-SCOPED": 0, "ARCHIVED": 0, "DANGLING": 0, "PROJECT-REL": 0}
    for root in roots:
        if not root.is_dir():
            continue
        for skill_md, text in bodies(root):
            seen = set()
            for raw in REF.findall(text):
                ref = raw.lstrip("./").rstrip(".,:;)")
                if ref in seen or TEMPLATE_TOKEN.search(ref):
                    continue
                seen.add(ref)
                local = skill_md.parent / ref
                if local.exists():
                    counts["OK"] += 1
                    if args.verbose:
                        rows.append({"skill": skill_md.parent.name, "ref": ref, "verdict": "OK",
                                     "resolved": rel_to_roots(local)})
                    continue
                name = pathlib.PurePosixPath(ref).name
                top = ref.split("/", 1)[0]
                ships_dir = (skill_md.parent / top).is_dir()
                elsewhere = [p for p in live_index.get(name, []) if p != local]
                archived = arch_index.get(name, [])
                if not ships_dir and top != "references":
                    # `scripts/x` / `templates/x` in a skill that ships no such dir
                    # names a path in the PROJECT under work (repo runner, plugin
                    # root), not a bundle file. Informational, never queued.
                    verdict, where = "PROJECT-REL", []
                elif elsewhere:
                    verdict, where = "MIS-SCOPED", [rel_to_roots(p) for p in elsewhere]
                elif archived:
                    verdict, where = "ARCHIVED", [rel_to_roots(p) for p in archived]
                else:
                    verdict, where = "DANGLING", []
                counts[verdict] = counts.get(verdict, 0) + 1
                rows.append({"skill": skill_md.parent.name, "skill_path": rel_to_roots(skill_md),
                             "ref": ref, "verdict": verdict, "resolved": where})

    if args.json:
        print(json.dumps({"counts": counts, "findings": rows}, indent=2))
        return 0

    print("=== skill reference check (intra-skill pointers) ===")
    print(f"roots: {', '.join(str(r) for r in roots)}")
    print(f"pointers checked: {sum(counts.values())}  |  " +
          "  ".join(f"{k}={v}" for k, v in counts.items()))
    for verdict in ("MIS-SCOPED", "ARCHIVED", "DANGLING"):
        hits = [r for r in rows if r["verdict"] == verdict]
        if not hits:
            continue
        print(f"\n== {verdict} ({len(hits)}) ==")
        for r in sorted(hits, key=lambda x: (x["skill"], x["ref"])):
            suffix = "  ->  " + ", ".join(r["resolved"][:2]) if r["resolved"] else ""
            print(f"  {r['skill']:<42} {r['ref']}{suffix}")
    print("\nMIS-SCOPED = pointer text is wrong, capability exists elsewhere (qualify the path).")
    print("ARCHIVED   = only in .archive/ (label it `ARCHIVED: curator restore first`).")
    print("DANGLING   = nowhere on disk (write the file or drop the claim).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
