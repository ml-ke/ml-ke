#!/usr/bin/env python3
"""Validate front matter in static-site content files (Jekyll/Hugo markdown).

Why this exists: Jekyll does NOT fail the build when front matter is invalid
YAML. It logs one line among thousands ("Error: YAML Exception reading <file>")
and then publishes the item with NO title, NO taxonomy and NO date. With no
date, Jekyll falls back to the file mtime, which on CI is the checkout time —
so the item jumps to the top of the feed and stays there, untitled.

The classic trigger is an unquoted scalar containing ": " (colon + space):

    image:
      alt: A shield from four quadrants: an AI eye, a CI/CD arrow

`alt: A shield ... quadrants: an AI eye` parses as a nested mapping -> boom.

Usage:
    python3 check-frontmatter.py _posts .scheduled      # dirs or files
    python3 check-frontmatter.py --strict _posts        # suppress warnings

Exit codes: 0 = clean, 1 = at least one hard error (unparseable, no title).
"""

import sys
import os
import re

REQUIRED = ("title",)  # a blank title publishes an unnamed card at the top of the feed
WARN_KEYS = ("date", "categories", "tags")

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.S)

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML not available — run with: uv run --with=pyyaml python3 " + sys.argv[0])


def iter_files(targets):
    for target in targets:
        if os.path.isdir(target):
            for name in sorted(os.listdir(target)):
                if name.endswith((".md", ".markdown")):
                    yield os.path.join(target, name)
        elif os.path.isfile(target):
            yield target


def check(path, strict):
    """Return (errors, warnings) for one file."""
    errors, warnings = [], []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    if not text.startswith("---"):
        errors.append("no front matter block (file does not start with '---')")
        return errors, warnings

    match = FRONT_MATTER.match(text)
    if not match:
        errors.append("front matter block is not closed with a '---' line")
        return errors, warnings

    block = match.group(1)
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        detail = str(exc).replace("\n", " ")
        if mark is not None:
            # +2: block starts on line 2 of the file (line 1 is the opening ---)
            lineno = mark.line + 2
            lines = text.splitlines()
            src = lines[lineno - 1] if 0 <= lineno - 1 < len(lines) else ""
            errors.append(
                "invalid YAML at line %d col %d: %s\n      line: %s"
                % (lineno, mark.column + 1, detail, src.strip())
            )
            errors.append(
                'hint: quote any value that contains \': \' — alt: "A shield: an eye"'
            )
        else:
            errors.append("invalid YAML: " + detail)
        return errors, warnings

    if not isinstance(data, dict):
        errors.append("front matter is not a mapping")
        return errors, warnings

    for key in REQUIRED:
        if not data.get(key):
            errors.append(
                "missing/empty '%s' — the item publishes with a blank title" % key
            )

    for key in WARN_KEYS:
        if not data.get(key):
            warnings.append(
                "missing '%s' — the generator falls back to the filename date; "
                "the item renders without taxonomy" % key
            )

    image = data.get("image")
    if isinstance(image, dict):
        if not image.get("path"):
            warnings.append("image block has no 'path'")
        elif not image.get("alt"):
            warnings.append("image block has no 'alt' text")
    elif image:
        errors.append("'image' must be a block with path:/alt:, not a scalar")

    if data.get("title") and isinstance(data["title"], str) and data["title"].endswith(" "):
        warnings.append("title has trailing whitespace")

    if strict:
        warnings = []

    return errors, warnings


def main(argv):
    strict = "--strict" in argv
    targets = [a for a in argv if not a.startswith("--")] or ["_posts"]
    files = list(iter_files(targets))
    if not files:
        sys.exit("no markdown files found in: " + ", ".join(targets))

    hard = 0
    soft = 0
    for path in files:
        try:
            errors, warnings = check(path, strict)
        except Exception as exc:  # noqa: BLE001 - report, keep scanning
            print("ERROR %s: could not read (%s)" % (path, exc))
            hard += 1
            continue
        for msg in errors:
            print("ERROR %s: %s" % (path, msg))
            hard += 1
        for msg in warnings:
            print("WARN  %s: %s" % (path, msg))
            soft += 1

    print("\n%d file(s) checked — %d error(s), %d warning(s)" % (len(files), hard, soft))
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
