#!/usr/bin/env python3
"""Diff every quoted output block in a post against the real stdout of the code block above it.

Usage: python3 check-quoted-output.py _posts/2026-10-04-slug.md

Why: verify-post-code.py proves the blocks RUN; this proves the numbers/figures quoted in the
post match what the blocks actually print. Blocks are paired by alternation in file order
(tagged ```python fence -> next untagged fence), which is the house layout: code block then its
output. Exits non-zero on any mismatch.
"""
import os
import re
import subprocess
import sys
import tempfile


def blocks(src):
    lines = re.sub(r"\{%\s*(raw|endraw)\s*%\}", "", src).split("\n")
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^```(\w*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        lang, body, j = m.group(1), [], i + 1
        while j < len(lines) and not re.match(r"^```\s*$", lines[j]):
            body.append(lines[j])
            j += 1
        out.append((lang, "\n".join(body)))
        i = j + 1
    return out


def main(path):
    pairs = blocks(open(path).read())
    runs = [b for l, b in pairs if l in ("python", "py", "bash", "sh")]
    quoted = [b for l, b in pairs if l == ""]
    print(f"{len(runs)} runnable blocks, {len(quoted)} quoted-output blocks")
    ok = True
    for n, (code, expected) in enumerate(zip(runs, quoted), 1):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            tmp = fh.name
        proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True)
        os.unlink(tmp)
        got, want = proc.stdout.strip(), expected.strip()
        if got == want:
            print(f"block {n}: stdout matches the quoted output byte-for-byte")
        else:
            ok = False
            print(f"block {n}: MISMATCH\n--- expected ---\n{want[:800]}\n--- got ---\n{got[:800]}")
    print("PROSE-OUTPUT CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
