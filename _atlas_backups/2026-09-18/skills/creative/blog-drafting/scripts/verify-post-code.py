#!/usr/bin/env python3
"""Verify Python code blocks in a drafted blog post actually run.

Usage:
  python3 scripts/verify-post-code.py _posts/YYYY-MM-DD-slug.md
  # ML blocks (numpy/sklearn) — run with the deps available:
  uv run --with=numpy --with=scikit-learn python3 scripts/verify-post-code.py POST.md

For every ```python fenced block (including ones wrapped in {% raw %}/{% endraw %}):
  - writes it to a temp file
  - py_compile-checks it (syntax errors reported without executing)
  - runs it and prints the block's own stdout so you can diff it against any
    output quoted in the post body (fact-checking protocol: "code claims
    actually run?" — verify the EXACT blocks, not a re-typed copy)

Exits non-zero if any block fails to compile or run.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
content = path.read_text(encoding="utf-8")

# Strip Liquid raw wrappers so the fence regex sees plain blocks
content = re.sub(r"\{%\s*raw\s*%\}", "", content)
content = re.sub(r"\{%\s*endraw\s*%\}", "", content)

blocks = re.findall(r"```python\n(.*?)```", content, re.S)
if not blocks:
    print(f"No ```python blocks found in {path}")
    sys.exit(0)

fail = 0
for i, code in enumerate(blocks, 1):
    print(f"\n===== block {i} ({len(code)} chars) =====")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        tmp = fh.name
    try:
        comp = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp],
            capture_output=True, text=True, timeout=60,
        )
        if comp.returncode != 0:
            print(f"SYNTAX ERROR:\n{comp.stderr}")
            fail += 1
            continue
        run = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True, timeout=180,
        )
        if run.returncode != 0:
            print(f"RUNTIME ERROR (exit {run.returncode}):\n{run.stderr}")
            fail += 1
        else:
            print("stdout:")
            print(run.stdout.rstrip())
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)

print(f"\n{'FAILED: ' + str(fail) + ' block(s)' if fail else 'OK: all blocks ran'}")
sys.exit(1 if fail else 0)
