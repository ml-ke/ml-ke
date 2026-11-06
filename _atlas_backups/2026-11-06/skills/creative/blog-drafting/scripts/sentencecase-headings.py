#!/usr/bin/env python3
"""Sentence-case Title Case headings across the blog corpus.

Run from the blog repo root (needs _posts/ and a git HEAD to diff against).

    python3 sentencecase-headings.py            # dry run
    python3 sentencecase-headings.py --apply    # write
    SC_EXCLUDE=/tmp/inflight.txt python3 sentencecase-headings.py --apply

Why a script and not a subagent: sentence-casing a heading is mechanical. In one sweep this
replaced ~500 heading edits across 63 posts that eight waves of subagents had not cleared, and it
is deterministic. Reserve subagents for em-dash reconstruction, AI vocabulary and restructuring.

The safety rule: a word is lowercased ONLY when it is provably an ordinary word - its lowercase
form occurs in the corpus prose or in the system dictionary - and the detector agrees it is a tell.
Proper nouns are therefore never touched by default.

Proof-of-ordinariness is necessary but NOT sufficient: 'android', 'claude', 'swahili', 'trivy',
'crew', 'sonnet' all appear lowercase somewhere (URLs, code, prose) yet are proper nouns in
headings. They must be enumerated in PROPER_WORDS inside check-prose-style.py.

ALWAYS review the distinct words it lowercased (not the diff) before committing, and assert that
no non-heading line changed.
"""
import collections
import importlib.util
import os
import pathlib
import re
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CHECKER = SCRIPT_DIR / "check-prose-style.py"
spec = importlib.util.spec_from_file_location("check_prose_style", CHECKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

POSTS = pathlib.Path("_posts")
DICT = pathlib.Path("/usr/share/dict/american-english")


def build_lower_used():
    """Words the corpus or the system dictionary actually uses in lowercase (= provably ordinary)."""
    used = set()
    for p in POSTS.glob("*.md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        used.update(re.findall(r"[a-z][a-z'-]{1,}", text))
    if DICT.exists():
        used.update(w.strip() for w in DICT.read_text(errors="replace").split()
                    if w.strip() and w.strip().islower())
    return used


lower_used = build_lower_used()


def fix_heading(heading):
    """Return (new_heading, [changed_words]). heading is the text after the hashes."""
    # A heading carrying an allowlisted proper-noun phrase is protected outright. The detector
    # ignores these, but the fixer must too, or it rewrites 'Deep Learning Indaba' to
    # 'Deep learning Indaba' - renaming a real conference to satisfy a style rule.
    low = heading.lower()
    if any(phrase in low for phrase in m.TITLE_CASE_ALLOWLIST):
        return heading, []
    changed, out = [], []
    for i, w in enumerate(heading.split()):
        # The detector decides what counts as a tell; the fixer acts only on its verdict, so the
        # two can never drift apart. Position 0 is never touched.
        if i == 0 or not m.is_tell_word(w):
            out.append(w)
            continue
        core = w.strip("*_`\"'()[]:,.?!").lower()
        if core not in lower_used:
            out.append(w)
            continue
        out.append(w[0].lower() + w[1:])
        changed.append(w)
    return " ".join(out), changed


def skip_zones(lines):
    """Line indexes that must never be edited: front matter, code fences, reference sections."""
    fm_end = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                fm_end = i
                break
    ref_at = None
    for i, ln in enumerate(lines):
        if re.match(r"^#{2,6}\s+(?:References|Related posts|Further [Rr]eading)\s*$", ln):
            ref_at = i
            break
    return fm_end, ref_at


def main(apply=False):
    exclude = set()
    ex = os.environ.get("SC_EXCLUDE")
    if ex:
        exclude = {pathlib.Path(x.strip()).name
                   for x in pathlib.Path(ex).read_text().split() if x.strip()}
    stats = collections.Counter()
    report = []
    for p in sorted(POSTS.glob("*.md")):
        if p.name in exclude:
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        fm_end, ref_at = skip_zones(lines)
        in_fence = False
        edited = False
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or i <= fm_end or (ref_at is not None and i >= ref_at):
                continue
            h = re.match(r"^(#{2,4})(\s+)(.+?)\s*$", ln)
            if not h:
                continue
            hashes, gap, body = h.groups()
            num = m._HEADING_NUMBERING_RE.match(body)
            prefix, rest = (body[:num.end()], body[num.end():]) if num else ("", body)
            new_rest, changed = fix_heading(rest)
            if changed:
                lines[i] = f"{hashes}{gap}{prefix}{new_rest}"
                edited = True
                stats["headings"] += 1
                report.append((p.name, body, f"{prefix}{new_rest}"))
        if edited and apply:
            p.write_text("\n".join(lines), encoding="utf-8")
            stats["files"] += 1
    print(f"headings to fix: {stats['headings']}   files applied: {stats['files']}")
    for fn, before, after in report[:25]:
        print(f"  {fn[:34]:34s}  {before[:44]:44s} -> {after[:40]}")
    if len(report) > 25:
        print(f"  ... and {len(report) - 25} more")
    return report


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
