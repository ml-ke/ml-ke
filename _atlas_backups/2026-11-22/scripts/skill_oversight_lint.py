#!/usr/bin/env python3
"""skill_oversight_lint.py — find oversight-loosening directives in skills.

Grounding: arxiv 2610.37734 "When Skills Tell Agents Not to Ask" (Nov 2026),
the first large-scale study of oversight-loosening directives in agent skills
(1.53M skills analysed):

  - ~1 in 17 skills (~88,000) contains a loosening directive;
  - 70.1% apply UNCONDITIONALLY rather than only in a specific context (CI,
    unattended cron, sandboxed dir);
  - skills that run shell code, install packages or use sudo have 1.7-1.8x the
    odds of carrying one, and do NOT compensate with more tightening language.

Why this matters here: one line in a widely loaded skill reaches every agent
that loads it. A loosening directive that is correct for an unattended cron
("don't wait for approval") is a defect when the same skill is loaded into an
interactive session — the checkpoint silently disappears.

What it does NOT do: it does not judge intent. Security skills legitimately
describe `--yolo`, `curl | bash`, and "bypass WAF" as *targets*. The action
queue is therefore (unconditional hit) AND (skill is not itself an
oversight-mode skill) AND (hit is not inside a quoted/description context) —
and every entry still needs a human read.

Usage:
  skill_oversight_lint.py                       # both libraries, human report
  skill_oversight_lint.py --root PATH           # one root
  skill_oversight_lint.py --json                # machine-readable
  skill_oversight_lint.py --verbose             # print every hit + context

Exit codes: 0 = report produced (findings or not), 2 = usage/IO error.
Never exits non-zero on findings: this is a review aid, not a gate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HOME = pathlib.Path.home()
DEFAULT_ROOTS = [HOME / ".hermes/skills", HOME / ".config/opencode/skills"]

# Roots that are never loaded by a harness, so directives inside them cannot
# fire. Including them invents findings against retired copies of live skills.
SKIP_PARTS = {".archive", ".hub", ".git", "node_modules", "quarantine", ".obsidian"}

# (label, regex). Kept narrow on purpose: broad patterns ("proceed") drown the
# signal — 2610.37734 found the directives are mostly *permission flags*.
PATTERNS = [
    ("approval-off-flag", re.compile(
        r"--yolo\b|--dangerously-skip-permissions|--dangerously-bypass-approvals"
        r"|--no-verify\b|danger-full-access|approvals\.mode\s*[:=]?\s*(off|false)", re.I)),
    ("do-not-ask", re.compile(
        r"\b(?:do not|don't|never)\s+(?:ask|confirm|wait for (?:the )?(?:user|approval|permission)"
        r"|check in|pause)\b", re.I)),
    ("without-approval", re.compile(
        r"\bwithout\s+(?:asking|approval|confirmation|checking in|user input|human)\b", re.I)),
    ("skip-gate", re.compile(
        r"\b(?:skip|omit|bypass)\s+(?:the\s+)?(?:approval|verification|review|gate|checkpoint"
        r"|confirmation|human)\b", re.I)),
    ("auto-approve", re.compile(
        r"\bauto[- ]approve|\bautoapprove|\bunattended\b.*\b(?:approval|permission)", re.I)),
]

# Conditional markers: a hit this close to one of these is *scoped*, which
# 2610.37734 found to be the exception (29.9%) and the acceptable form.
CONDITIONAL = re.compile(
    r"\b(?:if|when|whenever|only|unless|for|in|during|inside)\b[^.\n]{0,80}?"
    r"\b(?:ci|unattended|cron|scheduled|sandbox|sandboxed|isolated|container"
    r"|headless|batch|non-interactive|explicit(?:ly)?|asked|requested|user says"
    r"|docker|ephemeral|vm)\b", re.I)
NEGATION = re.compile(
    r"\b(?:do not|don't|never|not)\s+(?:use|run|pass|add|apply|enable|invoke)"
    r"|\bmust not\b|\bforbidden\b|\bprohibit", re.I)
# "Documents the flag" is not "instructs the flag". Skills like codex/claude-code
# must be able to *warn* about --yolo without being queued. Warning vocabulary
# within the same window demotes a hit to informational (same class as NEGATION).
WARNING = re.compile(
    r"\b(?:dangerous|disables?|disable|grants?|avoids?|risk|risky|caution|warning|unsafe"
    r"|not recommended|no sandbox|fastest, most|only in|isolated environments"
    r"|auto-approves|obviously|minefield)\b", re.I)

# A skill whose PURPOSE is oversight-reduction is scoped by its own name or
# description. Narrow on purpose: "bypass" alone matches SAML/XSS/WAF-bypass
# security skills, which are about a TARGET's controls, not the agent's.
OVERSIGHT_SKILL = re.compile(
    r"unhinged|--yolo|\byolo mode\b|no-approval|skip-permissions|unrestricted operation"
    r"|bypass (?:all )?(?:command )?approvals?|approvals?\s*(?:mode\s*)?off", re.I)

CAPABILITY = re.compile(
    r"\bsudo\b|\b(?:npm|pip|apt|brew|uv)\s+install\b|curl[^\n|]{0,80}\|\s*(?:ba)?sh"
    r"|\bos\.environ\b|\.env\b", re.I)


def frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return ""
    parts = text.split("---", 2)
    return parts[1] if len(parts) > 2 else ""


def walk(root: pathlib.Path):
    for p in root.rglob("SKILL.md"):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        yield p


def scan_file(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = frontmatter(text)
    desc = ""
    m = re.search(r"(?m)^description:\s*(.*)", fm)
    if m:
        desc = m.group(1).strip()
        if desc in (">", ">-", "|", "|-", ""):
            tail = fm[m.end():]
            desc = " ".join(l.strip() for l in tail.splitlines()[:12]
                            if l.strip() and not re.match(r"^\w[\w-]*:", l))
    hits = []
    for label, pat in PATTERNS:
        for mm in pat.finditer(text):
            start = max(0, mm.start() - 160)
            end = min(len(text), mm.end() + 160)
            ctx = re.sub(r"\s+", " ", text[start:end])
            window = text[max(0, mm.start() - 120): mm.end() + 120]
            scoped = bool(CONDITIONAL.search(window))
            # Loop-prevention ("do not ask again for the token you already
            # provided") is scoped to ONE named action. If the sentence carries
            # a backticked command, the directive cannot reach a different
            # checkpoint, so it is informational. Global "don't ask" is not.
            action_scoped = bool(re.search(r"`[^`]{2,60}`", window))
            negated = (bool(NEGATION.search(ctx)) or bool(WARNING.search(ctx))
                       or action_scoped)
            line_no = text[: mm.start()].count("\n") + 1
            hits.append({
                "label": label,
                "line": line_no,
                "match": mm.group(0)[:60],
                "context": ctx[:240],
                "scoped": scoped,
                "negated": negated,
            })
    name = path.parent.name
    is_oversight = bool(OVERSIGHT_SKILL.search(name) or OVERSIGHT_SKILL.search(desc))
    capability = bool(CAPABILITY.search(text))
    unconditional = [h for h in hits if not h["scoped"] and not h["negated"]]
    return {
        "skill": name,
        "path": str(path),
        "description": desc[:200],
        "oversight_skill": is_oversight,
        "capability_bearing": capability,
        "hits": hits,
        "unconditional": unconditional,
        "root": str(path.parent.parent),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", help="skill root (repeatable)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    roots = [pathlib.Path(r) for r in args.root] if args.root else DEFAULT_ROOTS
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        print("no skill roots found", file=sys.stderr)
        return 2

    results = []
    for root in roots:
        for p in walk(root):
            results.append(scan_file(p))

    # Tier 1: a skill whose PURPOSE is to loosen oversight. The risk is not the
    # directive itself (that is the point) but an unscoped description that lets
    # the planner load it on its own initiative. 2610.37734's mechanism list is
    # permission flags; an oversight skill is the same mechanism pre-packaged.
    tier1 = [r for r in results if r["oversight_skill"]
             and not re.search(r"explicit|only when the user|user asks|user says", r["description"], re.I)]
    # Tier 2: unconditional directive inside a capability-bearing skill that is
    # not itself about oversight — the 1.7-1.8x-odds population.
    tier2 = [r for r in results if r["unconditional"] and not r["oversight_skill"]
             and r["capability_bearing"]]
    rest = [r for r in results if r["unconditional"] and r not in tier1 and r not in tier2]

    if args.json:
        def slim(r):
            return {"skill": r["skill"], "path": r["path"],
                    "hits": [{"line": h["line"], "label": h["label"], "match": h["match"]}
                             for h in r["unconditional"]],
                    "capability_bearing": r["capability_bearing"]}
        print(json.dumps({
            "roots": [str(r) for r in roots],
            "skills_scanned": len(results),
            "skills_with_hits": sum(1 for r in results if r["hits"]),
            "tier1_unscoped_oversight_skills": [slim(r) for r in tier1],
            "tier2_unconditional_capability": [slim(r) for r in tier2],
            "informational": [slim(r) for r in rest],
        }, indent=2))
        return 0

    print(f"roots: {', '.join(str(r) for r in roots)}")
    print(f"skills scanned: {len(results)}  |  with any loosening directive: "
          f"{sum(1 for r in results if r['hits'])}")
    print(f"TIER 1 (unscoped oversight skill): {len(tier1)}   "
          f"TIER 2 (unconditional + capability): {len(tier2)}   "
          f"informational: {len(rest)}\n")

    if tier1:
        print("== TIER 1 — oversight skill with no explicit-invocation scope ==")
        for r in tier1:
            print(f"! {r['skill']}: \"{r['description'][:100]}\"")
            print(f"    {r['path']}")
            for h in r["unconditional"][:3]:
                print(f"    L{h['line']} [{h['label']}] {h['match']}")
        print()

    if tier2:
        print("== TIER 2 — unconditional directive in a capability-bearing skill ==")
        for r in sorted(tier2, key=lambda x: (-len(x["unconditional"]), x["skill"])):
            print(f"! {r['skill']} [{len(r['unconditional'])}]")
            print(f"    {r['path']}")
            for h in r["unconditional"][:4]:
                print(f"    L{h['line']} [{h['label']}] {h['match']}")
                if args.verbose:
                    print(f"        ...{h['context']}...")
        print()

    print(f"{len(rest)} further skills have unconditional hits (documented flags, "
          f"loop-prevention guidance) — run --verbose to read them.")

    if args.verbose:
        print("--- scoped / negated hits (informational) ---")
        for r in results:
            inf = [h for h in r["hits"] if h["scoped"] or h["negated"]]
            if inf:
                print(f"{r['skill']}: {len(inf)} (scoped/negated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
