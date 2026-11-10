#!/usr/bin/env python3
"""ATLAS venv-integrity check — catches ORPHANED virtualenvs.

Why: a distro `python3` upgrade silently orphans every venv created against the
old interpreter. Debian/Ubuntu's `/usr/bin/python3` is a symlink that follows the
default version (3.12 -> 3.14 here), while each venv's *packages* live in
`lib/python3.<minor>/site-packages`. After the upgrade:

  * `venv/bin/python` reports the NEW version but `sys.path` no longer contains
    the version dir the packages were installed into, so every pinned tool dies
    with `ModuleNotFoundError` — even though `pip list` in the old dir shows it
    installed, and even though the console-script shim is still present.
  * If the old interpreter binary was removed outright, `venv/bin/python` is a
    dangling symlink and the venv is completely dead.

Observed live (2026-10-06): `~/.hermes/venvs/skillsec` broke this way —
`skillspector` (the weekly supply-chain scan) exited with
`ModuleNotFoundError: No module named 'skillspector'` after the 3.12 -> 3.14
upgrade. The scan step had been reporting "nothing to show" by *failing to
start*, which is exactly the "a pipeline step can still exit 0 doing nothing"
failure class. Same box: pipx `mvt` (Android spyware scanner), pipx `shared`,
uv tool `mistral-vibe`, and ~40 project venvs were orphaned in the same event.

Usage:
  ~/.hermes/scripts/venv_integrity_check.py            # scan default roots
  ~/.hermes/scripts/venv_integrity_check.py --roots ~/ProG --roots ~/Dev
  ~/.hermes/scripts/venv_integrity_check.py --json     # machine-readable

Exit code is always 0 — this is a reporter, never a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

HOME = pathlib.Path.home()

# High-value roots: the ones cron jobs and ATLAS scripts actually depend on first.
DEFAULT_ROOTS = [
    HOME / ".hermes/venvs",
    HOME / ".local/share/uv/tools",
    HOME / ".local/share/pipx/venvs",
    HOME / ".hermes",
]

# Directories that are never worth walking into.
SKIP = {".cache", "node_modules", ".git", "site-packages", "__pycache__", ".venv-cache"}


def find_venvs(roots: list[pathlib.Path], max_depth: int = 4) -> list[pathlib.Path]:
    """Find every directory containing a pyvenv.cfg."""
    found: list[pathlib.Path] = []

    def walk(d: pathlib.Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(os.scandir(d))
        except (PermissionError, FileNotFoundError):
            return
        for e in entries:
            if not e.is_dir(follow_symlinks=False) or e.name in SKIP:
                continue
            p = pathlib.Path(e.path)
            if (p / "pyvenv.cfg").is_file():
                found.append(p)
                continue  # do not descend into a venv
            walk(p, depth + 1)

    for r in roots:
        r = r.expanduser()
        if r.is_dir():
            walk(r, 0)
    return sorted(set(found))


def cfg_version(cfg: pathlib.Path) -> str:
    """Return the version recorded at creation, e.g. '3.12.3'."""
    try:
        for line in cfg.read_text(errors="replace").splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("version", "version_info"):
                return v.strip()
    except OSError:
        pass
    return ""


def check_one(venv: pathlib.Path) -> dict:
    """Classify one venv. Never raises."""
    py = venv / "bin/python"
    rec = {"path": str(venv), "recorded": cfg_version(venv / "pyvenv.cfg"),
           "actual": None, "status": "unknown", "detail": "", "pinned_ok": []}

    if not py.exists():  # dangling symlink or missing interpreter
        rec["status"] = "DEAD"
        rec["detail"] = "bin/python missing or dangling (interpreter was removed)"
        return rec

    try:
        actual = subprocess.run(
            [str(py), "-c",
             "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
            capture_output=True, text=True, timeout=25,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "DEAD"
        rec["detail"] = f"interpreter will not run: {exc}"
        return rec
    rec["actual"] = actual

    try:
        paths = subprocess.run(
            [str(py), "-c",
             "import json,sys;print(json.dumps(sys.path))"],
            capture_output=True, text=True, timeout=25,
        ).stdout
        search_path = json.loads(paths) if paths.strip() else []
    except Exception:  # noqa: BLE001
        search_path = []

    site = [p for p in search_path if "site-packages" in p or "dist-packages" in p]
    has_site = any(pathlib.Path(p).is_dir() and any(pathlib.Path(p).iterdir())
                   for p in site)

    # Console scripts whose target module will not import = the observable symptom.
    for script in sorted((venv / "bin").glob("*")):
        if script.name.startswith(".") or not os.access(script, os.X_OK):
            continue
        if script.suffix in (".pyc",) or script.is_dir():
            continue
        try:
            head = script.read_text(errors="replace")[:2000]
        except OSError:
            continue
        if "from " not in head and "import " not in head:
            continue  # shell shim (e.g. the skillsec scanner wrappers)
        rec["pinned_ok"].append(script.name)

    rec_version = rec["recorded"].rsplit(".", 1)[0] if rec["recorded"] else ""
    act_version = ".".join(actual.split(".")[:2])

    if not site:
        rec["status"] = "DEAD"
        rec["detail"] = "interpreter exposes no site-packages directory"
    elif not has_site:
        rec["status"] = "DEAD"
        rec["detail"] = "no interpreter version dir holds any packages"
    elif rec_version and act_version and rec_version != act_version:
        # A mismatch is only a defect if the packages are unreachable. A venv that
        # was re-pointed or reinstalled under the new interpreter keeps working and
        # merely leaves a stale lib/python<old>/ directory behind — that is a
        # MIGRATED, not a broken, venv. Do not report it as a failure.
        rec["status"] = "MIGRATED"
        rec["detail"] = (f"created under {rec['recorded']}, now runs {actual} with "
                         "usable packages; stale lib/python"
                         f"{rec_version}/site-packages may remain")
    else:
        rec["status"] = "OK"
    return rec


def imports_ok(venv: pathlib.Path, modules: list[str]) -> list[str]:
    """Return the subset of `modules` that FAIL to import in `venv`."""
    py = venv / "bin/python"
    if not py.exists() or not modules:
        return []
    code = ("import importlib,sys\n"
            "bad=[]\n"
            "for m in %r:\n"
            "    try: importlib.import_module(m)\n"
            "    except Exception: bad.append(m)\n"
            "print(','.join(bad))" % (modules,))
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True,
                             text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        return list(modules)
    return [m for m in out.split(",") if m]


# Modules the weekly pipeline depends on, keyed by venv path suffix.
PIPELINE_IMPORTS = {
    ".hermes/venvs/skillsec": ["skillspector", "yaml"],
    ".hermes/venvs/skillscanner": ["skill_scanner"],
    ".hermes/venvs/skillsec-snyk": ["agent_scan"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", action="append", default=None,
                    help="root directory to scan (repeatable)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--deep", action="store_true",
                    help="also import-verify the pipeline tool modules")
    args = ap.parse_args()

    roots = [pathlib.Path(r) for r in args.roots] if args.roots else DEFAULT_ROOTS
    venvs = find_venvs(roots, args.max_depth)
    results = [check_one(v) for v in venvs]

    if args.deep:
        for rec in results:
            for suffix, mods in PIPELINE_IMPORTS.items():
                if rec["path"].endswith(suffix):
                    bad = imports_ok(pathlib.Path(rec["path"]), mods)
                    rec["import_failures"] = bad
                    if bad and rec["status"] == "OK":
                        rec["status"] = "BROKEN"
                        rec["detail"] = ("interpreter is fine but imports fail: "
                                         + ", ".join(bad))

    bad = [r for r in results if r["status"] not in ("OK",)]

    if args.json:
        print(json.dumps({"scanned": len(results), "problem": len(bad),
                          "venvs": results}, indent=1))
        return 0

    print(f"venvs found: {len(results)}   problems: {len(bad)}")
    for r in bad:
        print(f"  [{r['status']:<8}] {r['path']}")
        print(f"      {r['detail']}")
    if not bad:
        print("  all interpreters match their recorded version and expose packages")
    print("\nRule: never trust `python3` on PATH for a cron-dependency venv —"
          "\n      pin the interpreter (uv-managed or a versioned path). A venv"
          "\n      that cannot construct its own sys.path is dead, not empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
