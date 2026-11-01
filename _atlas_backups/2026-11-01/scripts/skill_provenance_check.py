#!/usr/bin/env python3
"""skill_provenance_check.py — hash-bound provenance check for third-party skills.

WHY THIS EXISTS
---------------
The Hermes->OpenCode bridge only covers the atlas-* skills it generates: it stamps
`atlas-source-sha256` and can report drift with --check. Everything else installed
into a skill root (Hermes Skills Hub installs, `gh skill install`, third-party
suites, hand-copied folders) has NO drift detection at all.

That gap is the "single-approval persistent trust model" defect (arxiv 2604.02837):
a skill is reviewed and approved ONCE, then trusted forever while its bytes change.
arxiv 2607.13987 puts it in the skill lifecycle as the *evolution* phase; OWASP
AST10 calls it AST07 Update Drift.

Manifests that record a per-skill `skill_md_sha256` (the convention used for the
curated third-party suites synced into ~/.config/opencode/skills) turn approval
into something checkable: re-hash the file and compare.

WHAT IT IS NOT
--------------
It is a REPORTER, not a gate: security tooling in a cron pipeline must not fail
the run. Exit code is always 0. A DRIFTED result means "this file changed after it
was recorded — read the diff before trusting it", not "this skill is malicious".
A clean result means "unchanged since recording", not "audited".

USAGE
  python3 ~/.hermes/scripts/skill_provenance_check.py                 # report
  python3 ~/.hermes/scripts/skill_provenance_check.py --json          # machine form
  python3 ~/.hermes/scripts/skill_provenance_check.py --only DRIFTED  # triage view

Interpreter note: never rely on `python3` from PATH here — Hermes tool upgrades
prepend a bundled interpreter without PyYAML (see LESSONS.md "Bundled-interpreter
regression class"). This script is stdlib-only on purpose, so any Python 3.8+ runs
it, and the cron wrapper should still call /usr/bin/python3 explicitly.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_ROOTS = [
    Path.home() / ".config/opencode/skills",
    Path.home() / ".hermes/skills",
]
MANIFEST_GLOBS = ("*skills-manifest.json", "*manifest.json", "manifest.json")

# Fields a manifest may use to record the content hash of a skill's SKILL.md.
HASH_FIELDS = ("skill_md_sha256", "sha256", "skill_sha256", "content_sha256")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def find_manifests(roots):
    """Yield (manifest_path, root) for every manifest that looks provenance-shaped."""
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for pat in MANIFEST_GLOBS:
            for mf in sorted(root.glob(pat)):
                rp = mf.resolve()
                if rp in seen or not mf.is_file():
                    continue
                seen.add(rp)
                yield mf, root


def skills_from_manifest(doc):
    """Normalise the several manifest shapes into {name: {field: value}}.

    Accepts: {"skills": {...}}, {"skills": [...]}, or a bare {name: {...}} map.
    """
    if isinstance(doc, dict) and isinstance(doc.get("skills"), dict):
        return doc["skills"]
    if isinstance(doc, dict) and isinstance(doc.get("skills"), list):
        out = {}
        for item in doc["skills"]:
            if isinstance(item, dict):
                name = item.get("name") or item.get("skill") or item.get("id")
                if name:
                    out[name] = item
        return out
    if isinstance(doc, dict):
        # bare map, but only if the values look like records
        if all(isinstance(v, dict) for v in doc.values()) and doc:
            return doc
    return {}


def resolve_skill_md(root: Path, name: str, meta: dict):
    """Locate a skill's SKILL.md, tolerating flat AND category-nested layouts.

    Returns (path_or_None, candidates). Same lesson as SkillSpector's `--recursive`
    gotcha: a flat-only resolver reports every category-nested skill as MISSING,
    which reads exactly like deletion. Roots here are mixed:
      ~/.config/opencode/skills/<name>/SKILL.md          (flat)
      ~/.hermes/skills/<category>/<name>/SKILL.md        (nested; `suite` names it)
    """
    rel = meta.get("path") or meta.get("dir")
    if rel:
        cand = root / rel
        cand = cand if cand.name == "SKILL.md" else cand / "SKILL.md"
        if cand.is_file():
            return cand, [cand]

    candidates = []
    flat = root / name / "SKILL.md"
    if flat.is_file():
        candidates.append(flat)
    suite = meta.get("suite")
    if suite:
        nested = root / str(suite) / name / "SKILL.md"
        if nested.is_file() and nested not in candidates:
            candidates.append(nested)
    if not candidates:
        # bounded fallback: name must match the immediate parent directory
        for depth_glob in (f"*/{name}/SKILL.md", f"*/*/{name}/SKILL.md"):
            for hit in root.glob(depth_glob):
                if hit.is_file() and hit not in candidates:
                    candidates.append(hit)
            if candidates:
                break
    if len(candidates) == 1:
        return candidates[0], candidates
    if len(candidates) > 1:
        return None, candidates  # ambiguous — refuse to pick
    return None, []


def check_root(root):
    records = []
    for mf, mroot in find_manifests([root]):
        try:
            doc = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            records.append({
                "status": "BAD_MANIFEST",
                "skill": mf.name,
                "manifest": str(mf),
                "detail": f"unreadable manifest: {exc}",
            })
            continue
        table = skills_from_manifest(doc)
        if not table:
            continue
        for name, meta in table.items():
            meta = meta if isinstance(meta, dict) else {}
            want = next((meta[f] for f in HASH_FIELDS if meta.get(f)), None)
            skill_md, candidates = resolve_skill_md(mroot, name, meta)
            rec = {
                "skill": name,
                "manifest": str(mf),
                "path": str(skill_md or (candidates[0] if candidates else "")),
                "recorded": want,
            }
            if len(candidates) > 1:
                rec.update(status="AMBIGUOUS",
                           detail=f"{len(candidates)} copies match name: "
                                  + ", ".join(str(c) for c in candidates))
            elif skill_md is None:
                rec.update(status="MISSING",
                           detail=f"no <name>/SKILL.md under {mroot}")
            elif not want:
                rec.update(status="UNRECORDED",
                           detail="manifest records no hash for this skill")
            else:
                got = sha256_file(skill_md)
                if got == want:
                    rec.update(status="OK", actual=got)
                else:
                    rec.update(status="DRIFTED", actual=got,
                               detail=f"recorded {want[:12]}… on-disk {got[:12]}…")
            records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=None,
                    help="skill root to check (repeatable; default: known roots)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--only", default=None,
                    help="show only this status (OK/DRIFTED/MISSING/UNRECORDED/BAD_MANIFEST)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.root] if args.root else DEFAULT_ROOTS
    all_records = []
    checked_roots = []
    for root in roots:
        recs = check_root(root)
        if recs:
            checked_roots.append((root, len(recs)))
        all_records.extend(recs)

    summary = {}
    for r in all_records:
        summary[r["status"]] = summary.get(r["status"], 0) + 1

    if args.json:
        print(json.dumps({"roots": [str(r) for r, _ in checked_roots],
                          "summary": summary, "records": all_records}, indent=2))
        return 0

    print("=== skill provenance check (hash-bound) ===")
    if not checked_roots:
        print("  no provenance manifest found under: "
              + ", ".join(str(r) for r in roots))
        print("  -> nothing is hash-tracked here; 'no findings' is UNMEASURED, not clean.")
        return 0
    for root, n in checked_roots:
        print(f"  root {root}  ({n} manifest-tracked skills)")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(summary.items())) or "  none")

    interesting = [r for r in all_records if r["status"] != "OK"]
    if args.only:
        interesting = [r for r in interesting if r["status"] == args.only.upper()]
    if interesting:
        print()
        for r in interesting:
            print(f"  [{r['status']}] {r['skill']}  ({r['manifest']})")
            if r.get("detail"):
                print(f"      {r['detail']}")
    print()
    print("A DRIFTED skill changed after review — read the diff before trusting it.")
    print("An OK skill is unchanged since recording, NOT audited.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
