#!/usr/bin/env python3
"""
ATLAS Skill Validator — audits SKILL.md frontmatter against the Agent Skills spec.

Checks (both libraries):
  1. Frontmatter parses as YAML
  2. `name` field exists
  3. name matches ^[a-z0-9]+(-[a-z0-9]+)*$ (spec regex)
  4. name matches the containing directory name (opencode requirement — mismatch silently breaks discovery)
  5. description exists and is 1-1024 chars
  6. SKILL.md body is under 5000 words (spec)
  7. no NUL/binary bytes in the file

Usage:
  python3 ~/.hermes/scripts/skill_validator.py ~/.config/opencode/skills
  python3 ~/.hermes/scripts/skill_validator.py ~/.hermes/skills --hermes
  python3 ~/.hermes/scripts/skill_validator.py <dir> --json     # machine-readable

Exit code 0 = clean, 1 = issues found.
"""
import argparse
import json
import os
import re
import sys

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_DESC = 1024
MAX_WORDS = 5000


def find_skills(root: str):
    """Yield (skill_dir, skill_name, sk.md path) for every SKILL.md under root."""
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden dirs and node_modules
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]
        if "SKILL.md" in filenames:
            yield dirpath, os.path.basename(dirpath), os.path.join(dirpath, "SKILL.md")


def parse_frontmatter(path: str):
    """Return (frontmatter_dict, body_text, error_str)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return None, "", f"unreadable: {e}"
    if "\x00" in content:
        return None, "", "contains NUL bytes"
    if not content.startswith("---"):
        return None, "", "no YAML frontmatter (does not start with ---)"
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, "", "malformed frontmatter (--- not closed)"
    fm_text, body = parts[1], parts[2]
    try:
        import yaml
    except ImportError:
        return None, "", "PyYAML not installed (pip install pyyaml)"
    try:
        fm = yaml.safe_load(fm_text)
    except Exception as e:
        return None, "", f"YAML parse error: {e}"
    if not isinstance(fm, dict):
        return None, "", f"frontmatter is not a mapping (got {type(fm).__name__})"
    return fm, body, None


def audit_skill(skill_dir: str, skill_name: str, sk_path: str, hermes_mode: bool):
    """Return list of (severity, message)."""
    issues = []
    fm, body, err = parse_frontmatter(sk_path)
    if err:
        issues.append(("error", err))
        return issues

    name = fm.get("name")
    if not name:
        issues.append(("error", "missing `name` field"))
    else:
        if not isinstance(name, str):
            issues.append(("error", f"name is not a string: {name!r}"))
        else:
            if not NAME_RE.match(name):
                issues.append(("error", f"name {name!r} fails regex {NAME_RE.pattern}"))
            if not hermes_mode and name != skill_name:
                issues.append(("error", f"name {name!r} != directory {skill_name!r} (breaks discovery)"))
            elif hermes_mode and name != skill_name:
                issues.append(("warn", f"name {name!r} != directory {skill_name!r}"))

    desc = fm.get("description")
    if not desc:
        issues.append(("error", "missing `description` field"))
    else:
        if not isinstance(desc, str):
            issues.append(("warn", f"description is not a string: {type(desc).__name__}"))
        elif not (1 <= len(desc) <= MAX_DESC):
            issues.append(("error", f"description length {len(desc)} outside 1-{MAX_DESC}"))

    word_count = len(body.split())
    if word_count > MAX_WORDS:
        issues.append(("warn", f"body {word_count} words > {MAX_WORDS} (spec recommends under {MAX_WORDS})"))

    if hermes_mode:
        # Hermes-specific: metadata.hermes.tags recommended
        meta = fm.get("metadata", {})
        if isinstance(meta, dict) and isinstance(meta.get("hermes"), dict):
            if "tags" not in meta["hermes"]:
                issues.append(("warn", "no metadata.hermes.tags"))
        else:
            issues.append(("warn", "no metadata.hermes block"))

    return issues


def main():
    ap = argparse.ArgumentParser(description="Audit SKILL.md frontmatter")
    ap.add_argument("root", help="skills root directory")
    ap.add_argument("--hermes", action="store_true", help="Hermes library mode (relaxed name check, hermes metadata check)")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--quiet", action="store_true", help="only print skills with issues")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"error: not a directory: {args.root}", file=sys.stderr)
        sys.exit(2)

    results = {}
    total_skills = 0
    for skill_dir, skill_name, sk_path in find_skills(args.root):
        total_skills += 1
        issues = audit_skill(skill_dir, skill_name, sk_path, args.hermes)
        if issues:
            results[sk_path] = issues

    errors = sum(1 for issues in results.values() for sev, _ in issues if sev == "error")
    warns = sum(1 for issues in results.values() for sev, _ in issues if sev == "warn")

    if args.json:
        print(json.dumps({
            "root": args.root,
            "total_skills": total_skills,
            "skills_with_issues": len(results),
            "errors": errors,
            "warnings": warns,
            "issues": {k: [{"severity": s, "message": m} for s, m in v] for k, v in results.items()},
        }, indent=2))
    else:
        for path, issues in sorted(results.items()):
            if args.quiet and not any(s == "error" for s, _ in issues):
                continue
            print(f"\n{path}")
            for sev, msg in issues:
                print(f"  [{sev.upper()}] {msg}")
        print(f"\n=== {args.root} ===")
        print(f"total skills: {total_skills}")
        print(f"skills with issues: {len(results)}")
        print(f"errors: {errors} | warnings: {warns}")
        if not results:
            print("CLEAN")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
