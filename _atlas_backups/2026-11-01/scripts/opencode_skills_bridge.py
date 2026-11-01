#!/usr/bin/env python3
"""
ATLAS Hermes→OpenCode Skill Bridge.

Syncs curated Hermes methodology skills into OpenCode's skill library so coding
sessions can load IDOR/JWT/SAML/SSRF/api-hacking methodology directly.

Why: OpenCode scans ~/.config/opencode/skills/<name>/SKILL.md and only reads 5
frontmatter fields (name, description, license, compatibility, metadata). The
Hermes library (~/.hermes/skills, 130+ skills) is invisible to it. This bridge
copies the security methodology skills that are useful during code review /
pentest automation, rewriting frontmatter to OpenCode's rules:

  - name MUST match directory name and ^[a-z0-9]+(-[a-z0-9]+)*$
  - description MUST be 1-1024 chars (truncated if longer)
  - long material stays in references/ (copied on demand — progressive disclosure)

Usage:
  python3 ~/.hermes/scripts/opencode_skills_bridge.py            # sync all curated skills
  python3 ~/.hermes/scripts/opencode_skills_bridge.py --list     # show what would sync
  python3 ~/.hermes/scripts/opencode_skills_bridge.py --dry-run  # show changes without writing

Config: the CURATED list below. Add new Hermes skills there as they become useful
for coding-side work. Namespace prefix "atlas-" avoids collisions with the 177
installed skills (tob-*, flutter-*, etc).
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# Interpreters to try when the active python lacks PyYAML.
_YAML_CANDIDATES = ("/usr/bin/python3", "/usr/bin/python3.12", "/usr/bin/python3.13")


def ensure_yaml():
    """Re-exec under an interpreter that has PyYAML, once.

    Hermes ships a bundled python on PATH that lacks PyYAML; when a tool
    upgrade repoints `python3` this bridge refuses to run, which silently
    stops the whole skill pipeline. Re-exec instead of failing. Guarded by an
    env flag so the re-exec can never loop.
    """
    try:
        import yaml  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_ATLAS_YAML_REEXEC") == "1":
        return
    me = os.path.abspath(sys.executable)
    for cand in _YAML_CANDIDATES + (shutil.which("python3.12"), shutil.which("python3.13")):
        cand = (cand or "").strip()
        if not cand or not os.path.isfile(cand) or os.path.abspath(cand) == me:
            continue
        try:
            if subprocess.run([cand, "-c", "import yaml"], capture_output=True, timeout=30).returncode == 0:
                env = dict(os.environ, _ATLAS_YAML_REEXEC="1")
                os.execve(cand, [cand, *sys.argv], env)
        except (OSError, subprocess.SubprocessError):
            continue

# Hermes skill name (as it appears under ~/.hermes/skills) -> OpenCode name
# Use the skill's directory-relative path under ~/.hermes/skills so nested
# categories (mlops/...) resolve correctly.
CURATED = [
    # bug-bounty methodology (the crown jewels for security work)
    ("bug-bounty/atlas-continuous-learning", "atlas-continuous-learning"),
    ("api-bug-bounty-methodology", "atlas-api-bug-bounty"),
    ("bug-bounty/api-hacking-methodology", "atlas-api-hacking"),
    ("bug-bounty/recon-to-exploitation", "atlas-recon-to-exploitation"),
    ("bug-bounty/idor-testing-methodology", "atlas-idor-testing"),
    ("jwt-attacks", "atlas-jwt-attacks"),
    ("oauth-oidc-attacks", "atlas-oauth-oidc-attacks"),
    ("saml-attacks", "atlas-saml-attacks"),
    ("ssrf-testing", "atlas-ssrf-testing"),
    ("mass-assignment-method-tampering", "atlas-mass-assignment"),
    ("business-logic-flaws", "atlas-business-logic"),
    ("chaining-methodology", "atlas-chaining"),
    ("bug-bounty/pre-submission-verification", "atlas-pre-submission-verification"),
    ("bug-bounty/bugcrowd-vrt", "atlas-bugcrowd-vrt"),
    # source audit (directly useful when OpenCode reviews code)
    ("software-development/source-code-security-audit", "atlas-source-code-audit"),
    ("software-development/requesting-code-review", "atlas-code-review"),
    ("software-development/skill-quality-audit", "atlas-skill-quality-audit"),
    # apk/recon
    ("bug-bounty/supabase-self-hosted-studio", "atlas-supabase-studio"),
    # system design theory + agent orchestration principles (Aug 16 2026)
    ("software-development/system-design-theory", "atlas-system-design"),
    ("autonomous-ai-agents/opencode-principles", "atlas-opencode-principles"),
    ("autonomous-ai-agents/antigravity-principles", "atlas-antigravity-principles"),
]

# Frontmatter fields OpenCode recognizes. Keep only these + our metadata.
OPENCODE_FIELDS = ("name", "description", "license", "compatibility", "metadata")
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_DESC = 1024

HERMES_ROOT = os.path.expanduser("~/.hermes/skills")
OPENCODE_ROOT = os.path.expanduser("~/.config/opencode/skills")


def resolve_hermes(path):
    """Resolve a Hermes skill path (may be category/nested)."""
    candidates = [
        os.path.join(HERMES_ROOT, path, "SKILL.md"),
        os.path.join(HERMES_ROOT, path + "/SKILL.md"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.dirname(c)
    # fallback: search by leaf name
    leaf = os.path.basename(path)
    for dirpath, dirnames, filenames in os.walk(HERMES_ROOT):
        if "SKILL.md" in filenames and os.path.basename(dirpath) == leaf:
            return dirpath
    return None


def parse_frontmatter(path):
    try:
        import yaml
    except ImportError:
        print("PyYAML required", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception as e:
        print(f"  ! frontmatter parse error in {path}: {e}", file=sys.stderr)
        fm = {}
    return (fm if isinstance(fm, dict) else {}), parts[2]


def build_opencode_skill(src_dir, oc_name, synced_at):
    """Copy a Hermes skill to OpenCode layout with rewritten frontmatter."""
    import yaml

    src_skill = os.path.join(src_dir, "SKILL.md")
    fm, body = parse_frontmatter(src_skill)

    desc = str(fm.get("description") or "").strip()
    if len(desc) > MAX_DESC:
        desc = desc[: MAX_DESC - 3].rstrip() + "..."

    new_fm = {"name": oc_name, "description": desc}
    for field in OPENCODE_FIELDS:
        if field in ("name", "description"):
            continue
        if field in fm:
            new_fm[field] = fm[field]

    # Provenance block (OWASP Agentic Skills Top 10):
    #   AST02 supply chain  — the copy names the source it came from.
    #   AST07 update drift  — source content hash + sync time make a stale
    #       copy detectable instead of silently divergent.
    #   AST10 cross-platform reuse / AST04 insecure metadata — the source's
    #       routing metadata (metadata.hermes: tags, category, related_skills)
    #       is PRESERVED. Rewriting frontmatter down to the target platform's
    #       5 fields used to drop it, which is exactly the documented
    #       "porting drops the source format's security metadata" failure.
    with open(src_skill, "rb") as fh:
        src_sha256 = hashlib.sha256(fh.read()).hexdigest()

    meta = new_fm.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    meta["atlas-source"] = os.path.relpath(src_dir, HERMES_ROOT)
    meta["atlas-source-sha256"] = src_sha256
    meta["atlas-synced"] = synced_at
    new_fm["metadata"] = meta

    # build YAML frontmatter manually to preserve formatting control
    fm_lines = ["---", f"name: {oc_name}"]
    # description: always block scalar — inline breaks on ': ' or '#' in text
    fm_lines.append("description: >-")
    for ln in desc.split("\n"):
        fm_lines.append(f"  {ln}" if ln else "  ")
    if "license" in new_fm:
        fm_lines.append(f"license: {new_fm['license']}")
    if "compatibility" in new_fm:
        fm_lines.append(f"compatibility: {new_fm['compatibility']}")
    meta_yaml = yaml.safe_dump(
        {"metadata": meta}, default_flow_style=False, allow_unicode=True, sort_keys=False
    ).rstrip("\n")
    fm_lines.extend(meta_yaml.split("\n"))
    fm_lines.append("---")
    new_content = "\n".join(fm_lines) + "\n\n" + body.lstrip("\n")

    return new_content


def _strip_synced(text):
    """Drop the atlas-synced timestamp line so a re-run is not read as drift."""
    return "\n".join(
        ln for ln in text.split("\n") if not ln.strip().startswith("atlas-synced:")
    )


def check_drift(curated, synced_at):
    """Report copies whose on-disk content no longer matches their source.

    Read-only. Always exits 0 — this is a reporter, not a gate.
    """
    drifted, missing, ok = [], [], 0
    for rel, oc_name in curated:
        src_dir = resolve_hermes(rel)
        dst_skill = os.path.join(OPENCODE_ROOT, oc_name, "SKILL.md")
        if src_dir is None:
            missing.append((oc_name, "source skill not found under ~/.hermes/skills"))
            continue
        if not os.path.isfile(dst_skill):
            missing.append((oc_name, "not synced into the OpenCode library"))
            continue
        try:
            expected = _strip_synced(build_opencode_skill(src_dir, oc_name, synced_at))
            with open(dst_skill, "r", encoding="utf-8", errors="replace") as fh:
                actual = _strip_synced(fh.read())
        except Exception as e:  # noqa: BLE001 - never let the reporter crash the run
            drifted.append((oc_name, f"unreadable: {e}"))
            continue
        if expected != actual:
            drifted.append((oc_name, "content differs from source"))
        else:
            ok += 1
    return ok, drifted, missing


def sync(curated, dry_run=False, verbose=False, synced_at=None):
    synced_at = synced_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    changes = []
    for rel, oc_name in curated:
        src_dir = resolve_hermes(rel)
        if src_dir is None:
            changes.append(("MISSING", rel, oc_name))
            continue
        if not NAME_RE.match(oc_name):
            changes.append(("BADNAME", rel, oc_name))
            continue
        dst_dir = os.path.join(OPENCODE_ROOT, oc_name)
        dst_skill = os.path.join(dst_dir, "SKILL.md")
        new_content = build_opencode_skill(src_dir, oc_name, synced_at)

        # references/ scripts/ templates/ — copy on demand (progressive disclosure)
        copied_refs = []
        for sub in ("references", "scripts", "templates"):
            src_sub = os.path.join(src_dir, sub)
            if os.path.isdir(src_sub):
                dst_sub = os.path.join(dst_dir, sub)
                if dry_run:
                    copied_refs.append(sub)
                    continue
                if os.path.isdir(dst_sub):
                    shutil.rmtree(dst_sub)
                shutil.copytree(src_sub, dst_sub)
                copied_refs.append(sub)

        if dry_run:
            status = "UPDATE" if os.path.isfile(dst_skill) else "NEW"
            changes.append((status, rel, oc_name))
            if verbose:
                print(f"  {status}: {oc_name} ({', '.join(copied_refs) or 'no refs'})")
            continue

        os.makedirs(dst_dir, exist_ok=True)
        with open(dst_skill, "w", encoding="utf-8") as f:
            f.write(new_content)
        changes.append(("SYNCED", rel, oc_name))
        if verbose:
            print(f"  SYNCED: {oc_name} <- {rel} (refs: {', '.join(copied_refs) or 'none'})")

    return changes


def main():
    ensure_yaml()
    ap = argparse.ArgumentParser(description="Bridge Hermes skills to OpenCode")
    ap.add_argument("--list", action="store_true", help="show curated mapping")
    ap.add_argument("--dry-run", action="store_true", help="show what would change")
    ap.add_argument(
        "--check",
        action="store_true",
        help="read-only: report OpenCode copies that have drifted from their Hermes source",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.list:
        for rel, oc in CURATED:
            src = resolve_hermes(rel)
            print(f"  {oc:42} <- {rel}  [{'FOUND' if src else 'MISSING'}]")
        return

    synced_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.check:
        ok, drifted, missing = check_drift(CURATED, synced_at)
        print("\n=== bridge drift check ===")
        print(f"  in sync : {ok}")
        print(f"  drifted : {len(drifted)}")
        print(f"  missing : {len(missing)}")
        for name, why in drifted:
            print(f"  ! DRIFT   {name} — {why}")
        for name, why in missing:
            print(f"  ! MISSING {name} — {why}")
        if not drifted and not missing:
            print("  verdict: all curated copies match their sources")
        return

    changes = sync(CURATED, dry_run=args.dry_run, verbose=args.verbose, synced_at=synced_at)
    print(f"\n=== bridge {'dry-run' if args.dry_run else 'sync'} complete ===")
    for status, rel, oc in changes:
        print(f"  {status:8} {oc}")
    missing = [c for c in changes if c[0] == "MISSING"]
    if missing:
        print(f"\n! {len(missing)} source skills not found — check HERMES_ROOT paths:")
        for _, rel, _ in missing:
            print(f"    {rel}")


if __name__ == "__main__":
    main()
