#!/usr/bin/env python3
"""ATLAS skill-routing collision check — find skills the planner cannot tell apart.

Why: at library scale the planner selects from DESCRIPTIONS ONLY. Hermes renders the
skill index as `- <name>: <description>` after truncating the description to
`57 chars + "..."` (agent/skill_utils.py::SKILL_PROMPT_DESC_LIMIT = 60), so every
token past char 57 is invisible at routing time. Two skills whose visible prefixes
say the same thing are a coin flip for the planner — the failure is silent (the
wrong body loads, the work still "succeeds").

Evidence:
  * SkillRouter (arxiv 2603.22455, Alibaba): the decisive routing signal is the BODY,
    not the metadata — removing the body costs 29-44pp accuracy and 91.7% of
    cross-encoder attention sits on the body field. At ~80K skills you need
    retrieve+rerank; at our scale (~300) there is no retrieval step, so the
    description prefix IS the whole selection signal and it must discriminate
    against the nearest neighbour, not just describe the topic.
  * "Skill Is Not Document" (arxiv 2606.03565): near-duplicate skills are the hard
    case for routing; multi-skill routing needs "minimal yet sufficient".
  * arxiv 2609.13321 (SkillSeam): overlapping lanes / synonymous aliases are a
    measurable collection defect (ownership conflicts 0/16 -> 14/16).

Scope rules learned the hard way (same class as the provenance checker's first
version, which reported 21 nested skills as MISSING):
  * Only DISCOVERABLE skills count. `.archive/`, `.hub/quarantine/`, `.git/` and any
    other dot-directory are never loaded — including them makes the tool invent
    collisions between a live skill and its own retired copy.
  * The agent dedupes by `name` and the first root wins, so the same name in two
    roots is ONE routing candidate (the seam checker reports those separately).

A collision is a PAIR, not a defect in either file: the fix is to make each
description lead with what separates it from its neighbour.

Exit code 0 always: a reporting tool for the weekly pipeline.

Usage:
  python3 ~/.hermes/scripts/skill_routing_collision_check.py
  python3 ~/.hermes/scripts/skill_routing_collision_check.py --json
  python3 ~/.hermes/scripts/skill_routing_collision_check.py --threshold 0.5 --top 15
"""
from __future__ import annotations

import argparse
import difflib
import itertools
import json
import pathlib
import re
import sys

HOME = pathlib.Path.home()

# Load order: earlier roots win a name collision (opencode.json + ~/.agents).
ROOTS = [
    ("opencode-global", HOME / ".config/opencode/skills"),
    ("hermes", HOME / ".hermes/skills"),
]

PREFIX = 57  # rendered description budget (SKILL_PROMPT_DESC_LIMIT 60 -> 57 + "...")

STOP = set(
    """a an the for and or of to in on at by with from into over under is are was were be been
    being this that these those it its as if then than when while use uses using used your you
    their there here what which who whom how why not no but so such via per each any all some
    can may should must will would could also more most other own same very just only about
    across between during before after above below up down out off again further once""".split()
)


def parse_description(text: str) -> str:
    """Extract `description:` from YAML frontmatter, incl. block scalars.

    Inline `description: a: b # c` is legal YAML only when quoted; block scalars
    (`description: >-`) are what generated skills should emit. Handle both, and
    never crash on a malformed skill — a parse failure is reported, not fatal.
    """
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    fm = text[3:end] if end != -1 else text[3:]
    m = re.search(r"(?m)^description:\s*(.*)$", fm)
    if not m:
        return ""
    first = m.group(1).strip()
    if first[:1] in (">", "|"):
        # block scalar: consume following indented lines
        lines = []
        for line in fm[m.end():].splitlines()[1:]:
            if line.strip() == "":
                lines.append("")
                continue
            if not line.startswith((" ", "\t")):
                break
            lines.append(line.strip())
        joined = " ".join(x for x in lines if x)
        return re.sub(r"\s+", " ", joined).strip()
    return first.strip().strip("\"'")


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1}


def collect(root: pathlib.Path) -> dict[str, dict]:
    """name -> record for ONE root. Dot-dirs excluded: never loaded."""
    found: dict[str, dict] = {}
    if not root.is_dir():
        return found
    for p in sorted(root.rglob("SKILL.md")):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue  # .archive / .hub/quarantine / .git — never loaded
        name = p.parent.name
        if name in found:
            found[name]["shadowed_in"].append(str(p))
            continue
        desc = parse_description(p.read_text(encoding="utf-8-sig", errors="replace"))
        found[name] = {
            "name": name,
            "root": root.name,
            "path": str(p),
            "description": desc,
            "prefix": desc[:PREFIX],
            "shadowed_in": [],
        }
    return found


def score(a: dict, b: dict) -> tuple[float, float, float, int]:
    ta, tb = tokens(a["prefix"]), tokens(b["prefix"])
    tj = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ta2, tb2 = tokens(a["description"]), tokens(b["description"])
    fj = len(ta2 & tb2) / len(ta2 | tb2) if (ta2 | tb2) else 0.0
    pr = difflib.SequenceMatcher(None, a["prefix"].lower(), b["prefix"].lower()).ratio()
    # Where the two rendered prefixes stop agreeing. A discriminator at char 3 is
    # harmless ("Bugcrowd ..." vs "HackerOne ..."); one at char 48 is inside the
    # budget but competing with 48 chars of boilerplate for the planner's attention.
    diverge = 0
    for x, y in zip(a["prefix"].lower(), b["prefix"].lower()):
        if x != y:
            break
        diverge += 1
    return tj, fj, pr, diverge


def verdict(jac: float, diverge: int, plen: int) -> str:
    """HIGH = indistinguishable at selection time. INFO = discriminates early, fine."""
    if jac >= 0.95 and diverge >= plen:
        return "HIGH (identical description)"
    if diverge >= 20:
        return "HIGH (discriminator deep in budget)"
    if diverge >= 8:
        return "MEDIUM"
    return "INFO (discriminates early)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="token Jaccard on the 57-char prefix (default 0.55)")
    ap.add_argument("--prefix-ratio", type=float, default=0.85,
                    help="difflib ratio on the 57-char prefix (default 0.85)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    report = {}
    all_pairs = []
    for root_name, root in ROOTS:
        # One root = one routing surface: mixing roots invents collisions between a
        # skill and its own bridged copy in the other runtime (atlas-X vs X).
        skills = collect(root)
        no_desc = sorted(s["name"] for s in skills.values() if not s["description"])
        pairs = []
        for a, b in itertools.combinations(skills.values(), 2):
            if not a["description"] or not b["description"]:
                continue
            tj, fj, pr, div = score(a, b)
            if tj >= args.threshold or pr >= args.prefix_ratio:
                plen = min(len(a["prefix"]), len(b["prefix"]))
                pairs.append({
                    "root": root_name, "a": a["name"], "b": b["name"],
                    "prefix_jaccard": round(tj, 3),
                    "full_jaccard": round(fj, 3),
                    "prefix_ratio": round(pr, 3),
                    "diverge_at": div,
                    "verdict": verdict(tj, div, plen),
                    "prefix_a": a["prefix"], "prefix_b": b["prefix"],
                    "a_shadowed": bool(a["shadowed_in"]), "b_shadowed": bool(b["shadowed_in"]),
                })
        pairs.sort(key=lambda p: (p["verdict"].startswith("INFO"), -p["prefix_jaccard"], -p["prefix_ratio"]))
        report[root_name] = {"skills": len(skills), "missing_description": no_desc,
                             "collisions": pairs}
        all_pairs.extend(pairs)

    if args.json:
        print(json.dumps({"threshold": args.threshold, "roots": report}, indent=2))
        return 0

    print("=== skill routing collision check ===")
    print(f"  description budget: {PREFIX} chars (SKILL_PROMPT_DESC_LIMIT 60 -> 57 + '...')")
    for root_name, root in ROOTS:
        r = report[root_name]
        if not r["skills"]:
            continue
        print(f"\n-- {root_name}  ({r['skills']} discoverable skills, dot-dirs excluded)")
        if r["missing_description"]:
            print(f"   NO DESCRIPTION (unroutable): {len(r['missing_description'])} -> "
                  f"{', '.join(r['missing_description'][:6])}")
        high = [p for p in r["collisions"] if not p["verdict"].startswith("INFO")]
        print(f"   candidate pairs: {len(r['collisions'])}  actionable: {len(high)}")
        for p in high[: args.top]:
            print(f"\n   ! [{p['verdict']}] {p['a']}  <->  {p['b']}   jac={p['prefix_jaccard']} "
                  f"diverge@{p['diverge_at']}")
            print(f"       A: {p['prefix_a']}")
            print(f"       B: {p['prefix_b']}")
    print("\n  A collision is a PAIR defect: lead each description with the token that")
    print("  separates it from its neighbour (the object/target), not a shared template.")
    print("  Everything past char %d is invisible to the planner at selection time." % PREFIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
