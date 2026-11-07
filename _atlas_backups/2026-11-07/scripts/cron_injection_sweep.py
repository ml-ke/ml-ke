#!/usr/bin/env python3
"""
Cron injection gate sweep — the exact check Hermes applies to an assembled cron
prompt, run on demand over skills you just edited.

Why a script: the naive version (`from tools.cronjob_prompt_scan import
_scan_cron_skill_assembled; print(_scan_cron_skill_assembled(text))`) DUMPS the whole
assembed text back into the agent's context — 247k chars on one run — and the useful
byte (pass/fail) is the trailing empty string. This prints the verdict only.

What it checks: `_CRON_SKILL_ASSEMBLED_PATTERNS` = the four patterns Hermes applies to
skill content in a cron prompt (prompt_injection, deception_hide, sys_prompt_override,
disregard_rules). Exfiltration and command-shape patterns are deliberately excluded from
that scope because skill prose legitimately describes commands. Invisible unicode is
stripped and reported, not blocked.

Usage:
  python3 ~/.hermes/scripts/cron_injection_sweep.py <file-or-dir> [...]
  python3 ~/.hermes/scripts/cron_injection_sweep.py ~/.hermes/skills -q   # quiet summary

Exit 0 = clean, 1 = a blocking pattern matched (do not attach the skill to a cron job).
"""
import argparse
import os
import sys

HERMES_REPO = os.path.expanduser("~/.hermes/hermes-agent")
if HERMES_REPO not in sys.path:
    sys.path.insert(0, HERMES_REPO)


def collect(paths):
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]
                files.extend(os.path.join(dirpath, f) for f in filenames if f == "SKILL.md")
        else:
            print(f"  ! not found: {p}", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description="Exact cron injection-gate sweep over skill files")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-q", "--quiet", action="store_true", help="one line per file, no header")
    args = ap.parse_args()

    try:
        from tools.cronjob_prompt_scan import _scan_cron_skill_assembled
    except Exception as e:  # import failure must not look like a pass
        print(f"CANNOT RUN — {type(e).__name__}: {e}")
        print("The gate lives in ~/.hermes/hermes-agent/tools/cronjob_prompt_scan.py")
        return 2

    files = collect(args.paths)
    if not files:
        print("CANNOT RUN — no SKILL.md files matched")
        return 2

    bad = []
    total = 0
    for f in sorted(files):
        try:
            text = open(f, "r", encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"  ! unreadable {f}: {e}")
            bad.append((f, "unreadable"))
            continue
        total += len(text)
        _cleaned, err = _scan_cron_skill_assembled(text)
        if err:
            bad.append((f, err))
            if not args.quiet:
                print(f"  BLOCKED {f}\n      {err[:300]}")

    if not args.quiet:
        print(f"=== cron injection sweep: {len(files)} file(s), {total} chars")
    if bad:
        print(f"  VERDICT: BLOCKED — {len(bad)} file(s) match a cron-gate pattern")
        for f, err in bad:
            print(f"    {f}: {err[:200]}")
        return 1
    print(f"  VERDICT: CLEAN — {len(files)} file(s) / {total} chars, no gate pattern matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
