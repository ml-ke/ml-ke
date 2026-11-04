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
  * Skill++ (arxiv 2611.00544): routing interference is a LIFELONG problem, not a
    static one. A similar NEW skill can displace an incumbent even though the
    incumbent and the router are both unchanged — 731 skills over ten sessions
    selected all required skills for 76.3% of queries after rewriting only the
    INCOMING skill's name+description, versus 36.4% under the original metadata.
    Two consequences encoded here: (a) a static pair count cannot detect this, so
    `--baseline` diffs this week's pairs against the last saved snapshot and
    reports NEW/ESCALATED/RESOLVED rather than a total; (b) the cheap fix is to
    rewrite the *incoming* description (what the body supports / how it differs
    from its closest predecessor / what it excludes), never to rename incumbents.
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
import datetime
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


# --- trajectory (Skill++ arxiv 2611.00544) ---------------------------------
# A static pair total is not the metric. Routing interference appears when a NEW
# skill lands next to an incumbent, and disappears when a description is fixed;
# both are changes, and only a snapshot diff can see them. Report deltas.

def pair_key(p: dict) -> str:
    a, b = sorted((p["a"], p["b"]))
    return f"{p['root']}|{a}|{b}"


def load_baseline(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc.get("pairs", {}) if isinstance(doc, dict) else {}


def _rank(p: dict) -> int:
    v = str(p.get("verdict", ""))
    return 3 if v.startswith("HIGH") else 2 if v == "MEDIUM" else 1


def diff_baseline(pairs: list[dict], base: dict) -> dict:
    """Compare this run's pairs against a snapshot.

    `new` is the Skill++ signal: a skill added (or a description edited) this week
    that now competes with an incumbent. `escalated` is the same thing arriving by
    a slower route. `resolved` counts pairs that were real and are now fixed —
    without it the tool only ever reports debt, and a library the agent is actively
    repairing still looks like a library going bad.
    """
    cur = {pair_key(p): p for p in pairs}
    new = [p for k, p in cur.items() if k not in base]
    escalated = [p for k, p in cur.items()
                 if k in base and _rank(p) > _rank(base[k])]
    fixed = [p for k, p in cur.items()
             if k in base and _rank(p) < _rank(base[k])]
    resolved = [k for k in base if k not in cur]
    return {"new": new, "escalated": escalated, "fixed": fixed, "resolved": resolved}




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="token Jaccard on the 57-char prefix (default 0.55)")
    ap.add_argument("--prefix-ratio", type=float, default=0.85,
                    help="difflib ratio on the 57-char prefix (default 0.85)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--root", action="append", default=None,
                    help="audit these roots instead of the defaults (repeatable; "
                         "used for synthetic tests and for third-party libraries)")
    ap.add_argument("--baseline", default=str(HOME / ".hermes/state/skill-routing-baseline.json"),
                    help="snapshot to diff against (Skill++ trajectory metric)")
    ap.add_argument("--save-baseline", action="store_true",
                    help="write the current actionable pairs to --baseline after reporting")
    ap.add_argument("--no-baseline", action="store_true",
                    help="ignore any saved snapshot (plain static report)")
    args = ap.parse_args()

    report = {}
    all_pairs = []
    roots = [(pathlib.Path(r).expanduser().name, pathlib.Path(r).expanduser()) for r in args.root] \
        if args.root else ROOTS
    for root_name, root in roots:
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

    bpath = pathlib.Path(args.baseline).expanduser()
    base = {} if args.no_baseline else load_baseline(bpath)
    delta = diff_baseline(all_pairs, base) if base else None
    actionable = [p for p in all_pairs if not p["verdict"].startswith("INFO")]

    if args.save_baseline:
        snap = {"saved": datetime.datetime.now().isoformat(timespec="seconds"),
                "skills": {name: r["skills"] for name, r in report.items()},
                "pairs": {pair_key(p): {"root": p["root"], "a": p["a"], "b": p["b"],
                                        "verdict": p["verdict"],
                                        "prefix_jaccard": p["prefix_jaccard"],
                                        "diverge_at": p["diverge_at"]}
                          for p in actionable}}
        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        out = {"threshold": args.threshold, "roots": report,
               "baseline": {"path": str(bpath), "saved_pairs": len(base)},
               "delta": None if delta is None else {
                   "new": [pair_key(p) for p in delta["new"] if not p["verdict"].startswith("INFO")],
                   "escalated": [pair_key(p) for p in delta["escalated"]],
                   "fixed": [pair_key(p) for p in delta["fixed"]],
                   "resolved": delta["resolved"]}}
        print(json.dumps(out, indent=2))
        return 0

    print("=== skill routing collision check ===")
    print(f"  description budget: {PREFIX} chars (SKILL_PROMPT_DESC_LIMIT 60 -> 57 + '...')")
    for root_name, root in roots:
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

    if delta is not None:
        fresh = [p for p in delta["new"] if not p["verdict"].startswith("INFO")]
        print(f"\n-- trajectory since snapshot ({bpath.name}, {len(base)} pairs)")
        print(f"   NEW: {len(fresh)}   ESCALATED: {len(delta['escalated'])}   "
              f"FIXED: {len(delta['fixed'])}   GONE: {len(delta['resolved'])}")
        for p in fresh[: args.top]:
            print(f"   + [{p['verdict']}] {p['a']}  <->  {p['b']}   jac={p['prefix_jaccard']}")
            print(f"       rewrite the NEW description: say what its body supports, how it "
                  f"differs from '{p['b']}', and what it excludes")
        for p in delta["fixed"]:
            print(f"   - fixed: {p['a']}  <->  {p['b']}  ({p['verdict']})")
    elif not args.no_baseline and not args.save_baseline:
        print(f"\n-- trajectory: no snapshot at {bpath} — run with --save-baseline to start one")
    if args.save_baseline:
        print(f"\n   snapshot written: {bpath}  ({len(actionable)} actionable pairs)")

    print("\n  A collision is a PAIR defect: lead each description with the token that")
    print("  separates it from its neighbour (the object/target), not a shared template.")
    print("  Everything past char %d is invisible to the planner at selection time." % PREFIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
