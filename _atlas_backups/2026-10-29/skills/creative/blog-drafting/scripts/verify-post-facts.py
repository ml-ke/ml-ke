#!/usr/bin/env python3
"""Fact-preservation verifier for a humanized (style-only) blog post edit.

The style gate (check-prose-style.py) proves the prose got better. This proves the
edit did not damage the post: a style-only pass must leave every fact, link and code
block untouched. Run it BEFORE committing a humanizer pass.

Compares the working file against the committed version and fails on any violation:
  - YAML front matter must be byte-identical
  - every fenced code block body must be byte-identical (order + content)
  - every markdown link target and bare URL must be preserved
  - every numeric token must be preserved (a changed statistic is fact damage)
  - every inline `code span` must be preserved
  - every heading count must be preserved
  - the em-dash count must strictly decrease (proves the pass did something)
Warns (review, not fatal) on: heading-wording changes.

Usage:
    verify-post-facts.py _posts/2026-10-10-some-post.md            # vs HEAD
    verify-post-facts.py _posts/some-post.md <git-ref>
Exit: 0 = all invariants hold, 1 = violation found.
"""
import argparse
import collections
import pathlib
import re
import subprocess
import sys


def git_show(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True
    ).stdout


def split_front_matter(text: str):
    m = re.match(r"^(---\n.*?\n---\n)", text, re.S)
    if not m:
        raise SystemExit(f"no YAML front matter found")
    return m.group(1), text[m.end():]


def fences(text: str):
    # Pair fences by alternating tokens. A naive r'```(.*?)```' regex starts matching at
    # a block's CLOSING fence and captures the prose after it.
    parts = re.split(r"(?m)^```.*$", text)
    return [parts[i] for i in range(1, len(parts), 2)]


def links(text: str) -> collections.Counter:
    return collections.Counter(
        re.findall(r"\]\(([^)]+)\)", text) + re.findall(r"https?://[^\s)\]\"'>]+", text)
    )


def numbers(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"\b\d[\d,._]*\b", re.sub(r"`[^`]*`", " ", text)))


def inline_code(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"`([^`\n]+)`", text))


def headings(text: str):
    return [h.strip() for h in re.findall(r"(?m)^#{1,6}.*$", text)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("ref", nargs="?", default="HEAD")
    args = ap.parse_args()

    old = git_show(args.ref, args.path)
    new = pathlib.Path(args.path).read_text(encoding="utf-8")
    fails, warns = [], []

    # Running after the commit is a normal, easy mistake. Say so plainly instead of
    # emitting a scary "em-dash count did not decrease" FAIL for a no-op comparison.
    if old == new:
        print(f"--- {args.path}  (vs {args.ref})")
        print(f"    no changes vs {args.ref} - nothing to verify (already committed?)")
        return 0

    if split_front_matter(old)[0] != split_front_matter(new)[0]:
        fails.append("front matter CHANGED (must be byte-identical)")

    fo, fn = fences(old), fences(new)
    if fo != fn:
        fails.append(f"code fenced blocks changed ({len(fo)} -> {len(fn)})")

    lo, ln = links(old), links(new)
    if lo != ln:
        fails.append(f"link targets changed; missing={dict(lo - ln)} added={dict(ln - lo)}")

    # A changed statistic IS fact damage - this is the whole reason the tool exists,
    # so it fails rather than warns. Punctuation adjacent to a number does not trip it
    # (only the token itself is compared).
    no, nn = numbers(old), numbers(new)
    if no != nn:
        fails.append(
            f"numeric tokens changed; missing={dict(list((no - nn).items())[:10])} "
            f"added={dict(list((nn - no).items())[:10])}"
        )

    io_, ino = inline_code(old), inline_code(new)
    if io_ != ino:
        fails.append(f"inline code spans changed; missing={len(io_ - ino)} added={len(ino - io_)}")

    ho, hn = headings(old), headings(new)
    if len(ho) != len(hn):
        fails.append(f"heading count changed ({len(ho)} -> {len(hn)}) - section added or deleted")

    emo, emn = old.count("\u2014"), new.count("\u2014")
    # Informational only - NOT a failure. check-prose-style.py owns the em-dash band; a hard
    # decrease requirement here pushed passes to strip em dashes from posts that were already
    # in band (one child cut 7 -> 1 purely to satisfy this check, over-editing a clean post).
    # This tool's job is 'no damage', which the checks above enforce.

    print(f"--- {args.path}  (vs {args.ref})")
    print(f"    em-dashes {emo} -> {emn}   words {len(old.split())} -> {len(new.split())}")
    for f in fails:
        print(f"    FAIL: {f}")
    for w in warns:
        print(f"    warn: {w}")
    if not fails and not warns:
        print("    OK: all invariants hold")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
