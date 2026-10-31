#!/usr/bin/env python3
"""
ATLAS Skill Spec Lint — the mechanical defect layer between skill_validator.py
(frontmatter + length) and skillspec scanners (security).

Gap this fills: arxiv 2608.08453 ("What Keeps Agent Skills from Being Reusable?",
138,133 SKILL.md files / 20,556 repos) found 91.8% carry >=1 defect across 31
static checks in 7 categories, and that the dominant failures are PACKAGING, not
attacks. skill_validator.py only covers spec-frontmatter; this covers the rest of
the cheap mechanical ones, so the checks stop depending on someone re-reading a
skill by hand.

Checks (id, severity, source finding)
  routing-no-trigger        WARN  desc gives no when-to-use guidance (52.3% of corpus)
  routing-budget            INFO  desc longer than the 57-char routing window
  name-as-h1                WARN  body repeats the skill name as its H1 (44.3%)
  inline-example-bloat      WARN  worked examples crammed inline instead of references/ (32.1%)
  prohibited-content        WARN  changelog / version history / TODO left in the body
  portability-abs-path      WARN  literal /home/<user> or /Users/<user> paths
  persona-scope             WARN  body claims to redefine the agent or outrank higher instructions
  no-validation             INFO  no verify/validate language anywhere in the body

Usage:
  python3 ~/.hermes/scripts/skill_spec_lint.py ~/.hermes/skills
  python3 ~/.hermes/scripts/skill_spec_lint.py ~/.config/opencode/skills --json
  python3 ~/.hermes/scripts/skill_spec_lint.py <root> --check inline-example-bloat
  python3 ~/.hermes/scripts/skill_spec_lint.py <root> --strict   # exit 1 on any WARN

Exit code 0 = report only (default), 1 = --strict and at least one WARN.
Report-only by default on purpose: a lint that gates a cron job wedges the
pipeline the first time the corpus shifts.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

_YAML_CANDIDATES = ("/usr/bin/python3", "/usr/bin/python3.12", "/usr/bin/python3.13")


def ensure_yaml():
    """Re-exec under an interpreter that has PyYAML, once.

    Same failure class as skill_validator.py / opencode_skills_bridge.py: Hermes
    prepends its own bundled python to PATH and it ships no PyYAML, so the lint
    would report zero findings for every skill -- indistinguishable from a clean
    library. Guarded by an env flag so the re-exec can never loop.
    """
    try:
        import yaml  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_ATLAS_YAML_REEXEC") == "1":
        return
    import shutil
    import subprocess

    me = os.path.abspath(sys.executable)
    for cand in _YAML_CANDIDATES + (shutil.which("python3.12"), shutil.which("python3.13")):
        cand = (cand or "").strip()
        if not cand or not os.path.isfile(cand) or os.path.abspath(cand) == me:
            continue
        try:
            if subprocess.run([cand, "-c", "import yaml"], capture_output=True, timeout=30).returncode == 0:
                os.execve(cand, [cand, *sys.argv], dict(os.environ, _ATLAS_YAML_REEXEC="1"))
        except (OSError, subprocess.SubprocessError):
            continue


TRIGGER_RE = re.compile(
    r"(?i)\b(use (this|when|for|it)|use\s+when|when (the )?(user|you|a|an|asked|working)|"
    r"invoke (this|when)|load (this|when)|apply (this|when)|trigger|for (any|all|tasks?)\b)"
)
CHANGELOG_RE = re.compile(r"(?im)^#{1,4}\s*(changelog|version history|release notes|revision history)\b")
# Require a literal leading "v": without it, ordinary numbered methodology headings
# ("### 1.1 API Documentation Audit") match and every well-structured skill is flagged.
CHANGELOG_ENTRY_RE = re.compile(r"(?im)^#{2,4}\s+v\d+(\.\d+)+\b")
# A marker is only debris when it stands alone at line start. Bare "XXX" is a URL
# placeholder and "grep TODO" is a technique — both tripped the first version.
STANDALONE_TODO_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?(?:TODO|FIXME)\s*[:\-]|<!--\s*(?:TODO|FIXME)")
ABS_PATH_RE = re.compile(r"(/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/)")
PERSONA_RE = re.compile(
    r"(?i)\b(you are now\b|your new (role|persona) is|from now on you (are|will|must)\b|"
    r"redefin(e|ing) your (persona|identity)|your real (task|purpose) is|"
    r"(supersede|overrule|outrank)\w*\s+(the\s+)?(higher|system|operator)\s+instructions)"
)
VALIDATION_RE = re.compile(r"(?i)\b(verify|verif(y|ication)|validat|check|confirm|re-?run|assert|test)\w*\b")

FENCE_RE = re.compile(r"```")


def find_skills(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "node_modules")
        if "SKILL.md" in filenames:
            yield dirpath, os.path.basename(dirpath), os.path.join(dirpath, "SKILL.md")


def parse_frontmatter(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return None, "", f"unreadable: {e}"
    if "\x00" in content:
        return None, "", "contains NUL bytes"
    if not content.startswith("---"):
        return None, "", "no YAML frontmatter"
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, "", "malformed frontmatter"
    try:
        import yaml
    except ImportError:
        return None, "", "PyYAML not installed"
    try:
        fm = yaml.safe_load(parts[1])
    except Exception as e:
        return None, "", f"YAML parse error: {e}"
    if not isinstance(fm, dict):
        return None, "", "frontmatter is not a mapping"
    return fm, parts[2], None


def lint(name, fm, body):
    """Return list of (check_id, severity, detail)."""
    out = []
    desc = fm.get("description")
    desc = desc if isinstance(desc, str) else ""

    # --- Routing metadata -------------------------------------------------
    if desc and not TRIGGER_RE.search(desc):
        out.append(("routing-no-trigger", "WARN",
                    "description states mechanics but no when-to-use trigger"))
    if len(desc) > 60:
        out.append(("routing-budget", "INFO",
                    f"description {len(desc)} chars; only the first 57 are visible when routing"))

    # --- Body content -----------------------------------------------------
    heads = [h.strip() for h in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", body)]
    for h in heads[:3]:
        if h.strip().strip("#").lower() == name.lower():
            out.append(("name-as-h1", "WARN", f"H1 repeats the skill name ({h!r}) — pure token cost"))
            break

    fences = len(FENCE_RE.findall(body)) // 2
    fence_chars = sum(len(m) for m in re.findall(r"```[\s\S]*?```", body))
    body_chars = max(len(body), 1)
    if fences >= 8 or (fences >= 4 and fence_chars / body_chars > 0.45):
        out.append(("inline-example-bloat", "WARN",
                    f"{fences} fenced blocks occupying {100 * fence_chars // body_chars}% of body; move worked examples to references/"))

    # --- Prohibited content ----------------------------------------------
    # Prose only: a template block that SHOWS a changelog shape, or a grep pattern
    # containing TODO, is the skill's content, not leftover authoring debris.
    prose = re.sub(r"```[\s\S]*?```", " ", body)
    if CHANGELOG_RE.search(prose) or len(CHANGELOG_ENTRY_RE.findall(prose)) >= 2:
        out.append(("prohibited-content", "WARN", "changelog / version-history section in the body"))
    todos = len(STANDALONE_TODO_RE.findall(prose))
    if todos:
        out.append(("prohibited-content", "WARN", f"{todos} leftover TODO/FIXME marker line(s) in the body"))

    # --- Portability ------------------------------------------------------
    abs_paths = sorted(set(ABS_PATH_RE.findall(body)))
    if abs_paths:
        home = os.path.expanduser("~").rstrip("/") + "/"
        ours = [p for p in abs_paths if p == home]
        note = "this host's home (breaks on any other machine)" if ours else "other user (may be a doc example)"
        out.append(("portability-abs-path", "WARN",
                    f"literal home path(s) {abs_paths[:3]} — {note}; use ~ or a skill-relative path"))

    # --- Persona & scope --------------------------------------------------
    # Fenced blocks are how a security skill QUOTES hostile text as teaching
    # material (godmode lists trigger words); only prose counts as an instruction.
    for m in PERSONA_RE.finditer(prose):
        snippet = prose[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        out.append(("persona-scope", "WARN", f"persona/scope override language: ...{snippet.strip()}..."))
        break
    else:
        if PERSONA_RE.search(body):
            out.append(("persona-scope", "INFO",
                        "persona language appears only inside a code block — teaching text, not an instruction"))

    # --- Verification presence (informational) ----------------------------
    if not VALIDATION_RE.search(body):
        out.append(("no-validation", "INFO", "body never asks for a verification step"))

    return out


def main():
    ap = argparse.ArgumentParser(description="Mechanical defect lint for SKILL.md files")
    ap.add_argument("root")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any WARN is present")
    ap.add_argument("--check", action="append", help="only report this check id (repeatable)")
    ap.add_argument("--top", type=int, default=40, help="max skills listed per check")
    args = ap.parse_args()

    ensure_yaml()
    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    findings = []
    total = 0
    for skill_dir, dirname, sk in find_skills(root):
        total += 1
        fm, body, err = parse_frontmatter(sk)
        if err:
            findings.append({"skill": dirname, "path": sk, "check": "unparseable",
                             "severity": "ERROR", "detail": err})
            continue
        name = fm.get("name") if isinstance(fm.get("name"), str) else dirname
        for cid, sev, detail in lint(name, fm, body):
            if args.check and cid not in args.check:
                continue
            findings.append({"skill": name, "path": sk, "check": cid,
                             "severity": sev, "detail": detail})

    by_check = defaultdict(list)
    for f in findings:
        by_check[f["check"]].append(f)
    sev_counts = Counter(f["severity"] for f in findings)
    clean = total - len({f["skill"] for f in findings if f["severity"] in ("WARN", "ERROR")})

    if args.json:
        print(json.dumps({"root": root, "skills": total, "clean": clean,
                          "severity": dict(sev_counts),
                          "by_check": {k: len(v) for k, v in by_check.items()},
                          "findings": findings}, indent=2))
    else:
        print(f"=== skill spec lint: {root}")
        print(f"  skills scanned: {total}   clean: {clean}   "
              f"WARN: {sev_counts.get('WARN', 0)}   INFO: {sev_counts.get('INFO', 0)}   "
              f"ERROR: {sev_counts.get('ERROR', 0)}")
        order = ["unparseable", "persona-scope", "portability-abs-path", "prohibited-content",
                 "routing-no-trigger", "name-as-h1", "inline-example-bloat",
                 "no-validation", "routing-budget"]
        for cid in order:
            hits = by_check.get(cid) or []
            if not hits:
                continue
            print(f"\n  [{cid}]  {len(hits)}")
            for f in hits[:args.top]:
                print(f"    - {f['skill']}: {f['detail']}")
            if len(hits) > args.top:
                print(f"    ... {len(hits) - args.top} more (--top to raise)")
        if not findings:
            print("\n  no findings")
        print("\n  Read the rule name, not the count: some of these are legitimate in context")
        print("  (a security skill may quote hostile phrasing as an example). Fix packaging,")
        print("  not prose, and re-run after edits.")

    if args.strict and sev_counts.get("WARN"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
