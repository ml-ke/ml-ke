#!/usr/bin/env python3
"""ATLAS second-opinion skill scanner (bounded, LLM-assisted).

Why: a single static scanner's verdict is not evidence. Across 61,990 registry
skills, three scanners disagreed on 23,702 and post-adjudication sensitivity
ranged 21.67%-61.06% (arxiv 2609.17274), while the only scanner with an LLM in
the loop reached 100% injection detection and found latent risk in >17% of a
sampled public registry (arxiv 2609.14079). So the weekly pipeline keeps the
static SkillSpector sweep AND runs a second, LLM-assisted pass — but only over
skills that CHANGED, to keep cost and runtime bounded.

Uses cisco-ai-defense/skill-scanner (PyPI `skill-scanner`) backed by DeepSeek.
VirusTotal is disabled (no key; no third-party upload of our own content).

Usage:
  python3 ~/.hermes/scripts/skill_second_opinion.py                 # changed in 7d, <=10 skills
  python3 ~/.hermes/scripts/skill_second_opinion.py --days 30 --limit 25
  python3 ~/.hermes/scripts/skill_second_opinion.py --path ~/.hermes/skills/software-development/skill-quality-audit --force
  python3 ~/.hermes/scripts/skill_second_opinion.py --all           # ignore the mtime cache (full rescan)
Exit code is always 0 — this is a reporting tool, not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

HOME = pathlib.Path.home()
VENV = HOME / ".hermes/venvs/skillsec/bin"
SCANNER = VENV / "skill-scanner"
STATE = HOME / ".hermes/state/skill_second_opinion.json"
ENV_FILE = HOME / ".hermes/.env"
DEFAULT_ROOTS = [HOME / ".hermes/skills"]

# Severity order for the compact printout.
ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def load_key() -> tuple[str, str, str]:
    """Return (api_key, base_url, model) from the environment or ~/.hermes/.env."""
    key = os.environ.get("SKILLSCAN_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key and ENV_FILE.is_file():
        for line in ENV_FILE.read_text(errors="ignore").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return (
        key,
        os.environ.get("SKILLSCAN_BASE_URL", "https://api.deepseek.com"),
        os.environ.get("SKILLSCAN_MODEL", "deepseek/deepseek-chat"),
    )


def find_changed(roots: list[pathlib.Path], days: int, limit: int, cache: dict) -> list[pathlib.Path]:
    """Skill directories whose newest file is younger than `days`, newest first."""
    cutoff = time.time() - days * 86400
    found: list[tuple[float, pathlib.Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for skill_md in root.rglob("SKILL.md"):
            if ".archive" in skill_md.parts or "node_modules" in skill_md.parts:
                continue
            try:
                newest = max(
                    (f.stat().st_mtime for f in skill_md.parent.rglob("*") if f.is_file()),
                    default=skill_md.stat().st_mtime,
                )
            except OSError:
                continue
            if newest < cutoff:
                continue
            d = skill_md.parent
            if cache.get(str(d)) == newest:
                continue  # already scanned at this revision
            found.append((newest, d))
    found.sort(reverse=True)
    return [d for _, d in found[:limit]]


def scan_one(skill_dir: pathlib.Path, key: str, base: str, model: str, timeout: int) -> dict | None:
    env = dict(os.environ, SKILLSCAN_API_KEY=key, SKILLSCAN_BASE_URL=base, SKILLSCAN_MODEL=model)
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "out.json"
        cmd = [
            str(SCANNER), "scan",
            "--path", str(skill_dir),
            "--platform", "generic",
            "--no-vt",
            "--format", "json",
            "--output", str(out),
        ]
        try:
            subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"timeout": True}
        if not out.is_file():
            return None
        try:
            return json.loads(out.read_text())
        except json.JSONDecodeError:
            return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action="append", default=None, help="scan this skill dir directly (repeatable)")
    ap.add_argument("--roots", action="append", default=None, help="skill roots to search (default ~/.hermes/skills)")
    ap.add_argument("--days", type=int, default=7, help="only skills changed in the last N days (default 7)")
    ap.add_argument("--limit", type=int, default=10, help="max skills to scan per run (default 10)")
    ap.add_argument("--timeout", type=int, default=300, help="per-skill timeout seconds (default 300)")
    ap.add_argument("--all", action="store_true", help="ignore the mtime cache")
    ap.add_argument("--force", action="store_true", help="with --path: scan even if not recently changed")
    args = ap.parse_args()

    if not SCANNER.is_file():
        print(f"[skip] {SCANNER} not installed — pip install skill-scanner into {VENV.parent}")
        return 0
    key, base, model = load_key()
    if not key:
        print("[skip] no DEEPSEEK_API_KEY / SKILLSCAN_API_KEY found — LLM second opinion unavailable")
        return 0

    cache: dict = {}
    if STATE.is_file() and not args.all:
        try:
            cache = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            cache = {}

    if args.path:
        targets = [pathlib.Path(p).expanduser().resolve() for p in args.path]
    else:
        roots = [pathlib.Path(r).expanduser() for r in (args.roots or DEFAULT_ROOTS)]
        targets = find_changed(roots, args.days, args.limit, cache)

    if not targets:
        print(f"[ok] no changed skills in the last {args.days}d (cache hit or nothing new) — nothing to second-opinion")
        return 0

    print(f"=== second-opinion scan: {len(targets)} skill(s), model={model}, VT=off ===")
    total = 0
    rows: list[tuple[str, str, str, str]] = []
    for d in targets:
        res = scan_one(d, key, base, model, args.timeout)
        if res is None:
            print(f"  [!] {d.name}: scanner produced no output")
            continue
        if res.get("timeout"):
            print(f"  [!] {d.name}: timed out after {args.timeout}s")
            continue
        findings = []
        for rep in res.get("reports", []):
            findings.extend(rep.get("llm_findings", []))
        total += len(findings)
        print(f"  {d.name}: {len(findings)} finding(s) — {res.get('summary')}")
        for f in sorted(findings, key=lambda x: ORDER.get(x.get("severity", "info"), 9)):
            rows.append((f.get("severity", "?"), d.name, f"{f.get('category')}: {f.get('title')[:88]}", (f.get("recommendation") or "")[:150]))
        cache[str(d)] = max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()), default=0)

    if rows:
        print("\n--- findings (triage: security-tooling content is FP-prone; act only on unpinned/unsigned fetches, unexplained encoded blobs, credential collection, or instruction-override text) ---")
        for sev, name, title, rec in sorted(rows, key=lambda r: ORDER.get(r[0], 9)):
            print(f"[{sev:8}] {name}: {title}")
            if rec:
                print(f"           -> {rec}")
    print(f"\nTOTAL LLM FINDINGS: {total}")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(cache, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
