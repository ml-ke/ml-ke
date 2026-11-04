#!/usr/bin/env python3
"""Starter checker for a manifest + pages docs tree.

Copy into scripts/ and adjust PAGES_DIR / ASSETS_DIR / MANIFEST / HOME_PAGE / ALIASES.
Stdlib only, offline, exit non-zero on error, safe to wire into CI with --strict.

The three false positives that always bite (fenced code read as headings, inline code spans breaking
fence counting, single-capture-group regexes returning strings) are already handled here — keep them
handled when you extend it. See references/checking-generated-docs.md for the reasoning.

    python3 scripts/check_docs_site.py [--strict]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(ROOT, "wiki")           # where the page sources live
ASSETS_DIR = os.path.join(ROOT, "assets")
MANIFEST = os.path.join(ROOT, "WIKI-MANIFEST.md")
HOME_PAGE = "00-Home.md"                         # may keep a title; it is also the repo README source
# Links that resolve to a differently named page (the wiki renders the landing page as "Home").
ALIASES = {"Home": HOME_PAGE, "_Sidebar": "_Sidebar.md", "_Footer": "_Footer.md"}

BANNED = ["delve", "leverage", "seamless", "game-changer", "supercharge"]
SECRETS = [
    (r"\bsk-[A-Za-z0-9]{16,}", "API-key-shaped string"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}", "GitHub-token-shaped string"),
    (r"\bAKIA[0-9A-Z]{12,}", "AWS-key-shaped string"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
    (r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.", "JWT-shaped string"),
]
MERMAID_OK = {"flowchart", "graph", "sequencediagram", "statediagram-v2", "erdiagram",
              "classdiagram", "gantt", "timeline", "pie", "journey", "gitgraph"}

ASSET_URL = re.compile(r"assets/([A-Za-z0-9._-]+\.png)")
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")   # two groups: unpacking needs both
FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]*`")


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, page, msg):
        self.errors.append(f"{page}: {msg}")

    def warn(self, page, msg):
        self.warnings.append(f"{page}: {msg}")


def manifest_pages(path):
    """Page filenames from markdown table rows: | 1 | Title | `file.md` | level | min |"""
    out = set()
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*\d+\s*\|.*?`([A-Za-z0-9._-]+\.md)`", line)
        if m:
            out.add(m.group(1))
    return out


def prose_words(text):
    t = FENCE.sub("", text)                                  # code is not prose
    t = re.sub(r"^\s*\|.*$", "", t, flags=re.M)             # tables are not prose
    t = re.sub(r"^\s*[-#>*\d]+[.)]?\s*", "", t, flags=re.M)
    return len([w for w in re.split(r"\s+", t) if w.strip()])


def check_page(path, name, exempt, report):
    text = open(path, encoding="utf-8").read()
    prose = FENCE.sub("", text)                              # never scan code as prose

    if name != HOME_PAGE:
        for i, line in enumerate(prose.splitlines(), 1):
            if re.match(r"^#(?!#)\s", line):
                report.error(name, f"H1 in prose (line {i}) — the renderer supplies the title")
                break

    if name not in exempt:
        if not text.startswith("> **"):
            report.error(name, "first line must be the metadata blockquote")
        for required in ("## Key takeaways", "## Further learning"):
            if required not in text:
                report.error(name, f"missing section: {required}")
        for optional in ("## Try it", "## Common mistakes"):
            if optional not in text:
                report.warn(name, f"missing section: {optional}")
        words = prose_words(text)
        if words < 400:
            report.error(name, f"{words} words of prose (minimum 400)")
        elif words > 1400:
            report.warn(name, f"{words} words of prose (maximum 1400)")

    for label, target in LINK.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:", "images/")) or target in ALIASES:
            continue
        if target.endswith(".md"):
            report.error(name, f"internal link must use the page name, not a filename: {target}")
        elif not os.path.exists(os.path.join(PAGES_DIR, target + ".md")):
            report.error(name, f"link to missing page: [{label}]({target})")

    for asset in ASSET_URL.findall(text):
        if not os.path.exists(os.path.join(ASSETS_DIR, asset)):
            report.error(name, f"embedded asset missing: assets/{asset}")

    fences = INLINE.sub("", text)                            # inline ` ```mermaid ` is not a fence
    blocks = re.findall(r"```mermaid\n(.*?)```", fences, re.S)
    if fences.count("```mermaid") != len(blocks):
        report.error(name, "unbalanced ```mermaid fence")
    for block in blocks:
        first = block.strip().splitlines()[0].strip().lower() if block.strip() else ""
        if first.split()[0] not in MERMAID_OK:
            report.error(name, f"unsupported mermaid type: {first[:40]}")
        if "%%{init" in block or re.search(r"^\s*click\s", block, re.M):
            report.error(name, "mermaid uses an init directive or click handler")

    low = prose.lower()
    for word in BANNED:
        if re.search(r"\b" + re.escape(word), low):
            report.warn(name, f"filler word: '{word}'")
    for pattern, what in SECRETS:                            # scan the WHOLE file, not just prose
        found = re.search(pattern, text)
        if found:
            report.error(name, f"{what} ({found.group(0)[:12]}...)")


def main():
    strict = "--strict" in sys.argv
    if not os.path.isdir(PAGES_DIR):
        sys.exit(f"no pages directory: {PAGES_DIR}")
    report = Report()
    pages = sorted(f for f in os.listdir(PAGES_DIR) if f.endswith(".md"))
    # Generated navigation pages legitimately differ from the page skeleton.
    exempt = {os.path.basename(HOME_PAGE), "_Sidebar.md", "_Footer.md"}
    exempt |= {f for f in pages if re.match(r"^\d\d-[A-Z]", f) and not re.match(r"^\d\d-[a-z]", f)}

    for page in pages:
        check_page(os.path.join(PAGES_DIR, page), page, exempt, report)

    listed = manifest_pages(MANIFEST)
    for missing in sorted(listed - set(pages)):
        report.errors.append(f"manifest lists {missing} but the file does not exist")
    for orphan in sorted(set(pages) - listed - exempt):
        report.errors.append(f"{orphan} is not in the manifest (orphan page)")

    print(f"checked {len(pages)} pages, {len(listed)} in manifest: "
          f"{len(report.errors)} errors, {len(report.warnings)} warnings")
    for err in report.errors:
        print("  ERROR   " + err)
    for warn in report.warnings:
        print("  warning " + warn)
    if report.errors or (strict and report.warnings):
        return 1
    print("docs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
