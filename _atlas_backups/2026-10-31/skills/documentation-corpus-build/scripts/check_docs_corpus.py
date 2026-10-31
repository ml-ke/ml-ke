#!/usr/bin/env python3
"""Check a documentation corpus before publishing.

Adapt CONFIG, then run it as a local gate and in CI:

    python3 scripts/check_docs_corpus.py            # report, exit 1 on errors
    python3 scripts/check_docs_corpus.py --strict   # also fail on warnings

Checks: manifest coverage both ways, page structure, prose length, internal links, embedded
assets, mermaid blocks, credential shapes and filler words. Offline (no network access).
"""
import os
import re
import sys

# ---------------------------------------------------------------- CONFIG
PAGES_DIR = "wiki"                     # source pages, relative to the repo root
ASSETS_DIR = "assets"                  # generated images
MANIFEST = "WIKI-MANIFEST.md"          # page list; table rows carry `filename.md`
EXEMPT_PREFIXES = ("_Sidebar", "_Footer")   # generated files exempt from the page skeleton
LANDING_PAGE = "00-Home.md"            # the one page allowed an H1
# Pages that exist in the published wiki under a different name than in the source dir.
ALIASES = {"Home": "00-Home.md", "_Sidebar": "_Sidebar.md", "_Footer": "_Footer.md"}
# Only URLs matching this may be embedded as images; other absolute URLs are treated as links.
ASSET_URL = re.compile(r"raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+/assets/([A-Za-z0-9._-]+)")
REQUIRED_SECTIONS = ("## Key takeaways", "## Further learning")
EXPECTED_SECTIONS = ("## Try it", "## Common mistakes")
MIN_WORDS, MAX_WORDS = 400, 1400
BANNED = ["delve", "leverage", "seamless", "game-changer", "supercharge", "empower",
          "in today's fast-paced"]
SECRET_PATTERNS = [
    (r"\bsk-[A-Za-z0-9]{16,}", "looks like an API key"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}", "looks like a GitHub token"),
    (r"\bAKIA[0-9A-Z]{12,}", "looks like an AWS access key id"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
    (r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.", "looks like a JWT"),
    (r"\bwhsec_[A-Za-z0-9]{16,}", "looks like a webhook signing secret"),
]
MERMAID_OK = {"flowchart", "graph", "sequencediagram", "statediagram-v2", "erdiagram",
              "classdiagram", "gantt", "timeline", "pie", "journey", "gitgraph"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(ROOT, PAGES_DIR)
ASSETS = os.path.join(ROOT, ASSETS_DIR)
# Two capture groups: findall with ONE group returns strings, and unpacking them raises.
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
# Strip inline code spans only — a blanket backtick sweep also eats ``` fences and hides problems.
INLINE_SPAN = re.compile(r"(?<!`)`{1,2}(?!`)[^`\n]*?`{1,2}(?!`)")
FENCE = re.compile(r"```.*?```", flags=re.S)

errors = []
warnings = []


def err(page, msg):
    errors.append(f"{page}: {msg}")


def warn(page, msg):
    warnings.append(f"{page}: {msg}")


def manifest_pages():
    out = set()
    path = os.path.join(ROOT, MANIFEST)
    if not os.path.exists(path):
        print(f"warning: no {MANIFEST} - skipping manifest coverage")
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*\d+\s*\|.*?`([A-Za-z0-9._-]+\.md)`", line)
        if m:
            out.add(m.group(1))
    return out


def exempt_pages():
    """Generated navigation pages: exempt from the lesson skeleton."""
    out = {LANDING_PAGE}
    for f in os.listdir(PAGES):
        if f.endswith(".md") and (f.startswith(EXEMPT_PREFIXES) or not re.match(r"^\d\d-", f)):
            out.add(f)
    return out


def prose_words(text):
    stripped = FENCE.sub("", text)
    stripped = re.sub(r"^\s*\|.*$", "", stripped, flags=re.M)      # tables are not prose
    stripped = re.sub(r"^\s*[-#>*\d]+[.)]?\s*", "", stripped, flags=re.M)
    return len([w for w in re.split(r"\s+", stripped) if w.strip()])


def check_page(path, name, exempt):
    text = open(path, encoding="utf-8").read()
    prose = FENCE.sub("", text)          # a comment inside a fence is not a heading

    if name != LANDING_PAGE:
        for i, line in enumerate(prose.splitlines()):
            if re.match(r"^#(?!#)\s", line):
                err(name, f"H1 in prose (line {i + 1}): the published page renders the title already")
                break

    if name not in exempt:
        if not text.startswith("> **"):
            err(name, "first line must be the metadata blockquote starting with '> **'")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                err(name, f"missing required section: {section}")
        for section in EXPECTED_SECTIONS:
            if section not in text:
                warn(name, f"missing expected section: {section}")
        words = prose_words(text)
        if words < MIN_WORDS:
            err(name, f"only {words} words of prose (minimum {MIN_WORDS})")
        elif words > MAX_WORDS:
            warn(name, f"{words} words of prose (target max {MAX_WORDS})")
        if "```mermaid" not in text and ".png" not in text:
            warn(name, "no diagram and no infographic")

    for label, target in LINK.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:")) or target in ALIASES:
            continue
        if target.endswith(".md"):
            err(name, f"internal link uses a filename, not a page name: {target}")
            continue
        if not os.path.exists(os.path.join(PAGES, target + ".md")):
            err(name, f"link to missing page: [{label}]({target})")

    for asset in ASSET_URL.findall(text):
        if not os.path.exists(os.path.join(ASSETS, asset)):
            err(name, f"embedded asset does not exist: {ASSETS_DIR}/{asset}")

    fences = INLINE_SPAN.sub("", text)
    blocks = re.findall(r"```mermaid\n(.*?)```", fences, flags=re.S)
    if fences.count("```mermaid") != len(blocks):
        err(name, "unbalanced mermaid fence (opened but never closed)")
    for block in blocks:
        first = block.strip().splitlines()[0].strip().lower() if block.strip() else ""
        kind = first.split()[0] if first else ""
        if kind not in MERMAID_OK:
            err(name, f"mermaid block starts with unsupported type: {first[:40]!r}")
        if "%%{init" in block or re.search(r"^\s*click\s", block, flags=re.M):
            err(name, "mermaid uses an init directive or click handler")
        if len(re.findall(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*[\[(]", block, flags=re.M)) > 20:
            warn(name, "mermaid diagram looks large (over 20 nodes)")

    low = prose.lower()
    for word in BANNED:
        if re.search(r"\b" + re.escape(word), low):
            warn(name, f"filler word: {word!r}")
    for pattern, what in SECRET_PATTERNS:
        m = re.search(pattern, text)          # secrets are checked in code blocks too
        if m:
            err(name, f"{what}: {m.group(0)[:12]}... - never ship anything resembling a credential")


def main():
    strict = "--strict" in sys.argv
    if not os.path.isdir(PAGES):
        sys.exit(f"no {PAGES_DIR}/ directory under {ROOT}")
    manifest = manifest_pages()
    exempt = exempt_pages()
    pages = sorted(f for f in os.listdir(PAGES) if f.endswith(".md"))

    for page in pages:
        check_page(os.path.join(PAGES, page), page, exempt)

    for missing in sorted(manifest - set(pages)):
        errors.append(f"manifest lists {missing} but the file does not exist")
    for orphan in sorted(set(pages) - manifest - exempt):
        errors.append(f"{orphan} is not in {MANIFEST} (orphan page)")

    print(f"checked {len(pages)} pages - {len(manifest)} in manifest - "
          f"{len(errors)} errors - {len(warnings)} warnings")
    for e in errors:
        print(f"  ERROR   {e}")
    for w in warnings:
        print(f"  warning {w}")
    if errors or (strict and warnings):
        return 1
    print("corpus OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
