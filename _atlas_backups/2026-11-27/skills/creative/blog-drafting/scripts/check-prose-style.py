#!/usr/bin/env python3
"""Prose-style gate for ML Kenya posts (the humanizer final pass, made measurable).

A mandate that is not measured gets skipped. This measures the tells that actually
regressed in production (prose em-dash overuse, residual AI vocabulary, title-case
headings, emoji) and exits non-zero when a post is off the house band, so the daily
blog cron can gate on it instead of trusting that the pass happened.

Scope: BODY PROSE only. Excluded by design (house formatting, see blog-drafting's
'Final pass' section): YAML front matter, fenced code blocks, inline `code spans`,
the '## References' and '## Related posts' sections.

Usage:
    check-prose-style.py _posts/2026-10-10-some-post.md
    check-prose-style.py _posts/*.md            # audit a batch
    check-prose-style.py --json _posts/x.md
Exit: 0 = all in band, 1 = at least one post off band.
"""
import argparse
import json
import pathlib
import re
import sys

# Measured on the live lane (Sep-Oct 2026): posts that passed the humanizer pass run
# 0-6 prose em dashes. 8 gives slack without letting a regression through.
MAX_EM_DASHES = 8
MAX_TITLE_CASE_HEADINGS = 0
MAX_EMOJI = 0

# HARD FAILURES: phrases/words with no legitimate use in this blog's prose.
# Kept deliberately narrow and high-precision - a gate that fires on real technical
# vocabulary gets ignored, and then it protects nothing.
AI_VOCAB = [
    "delve", "tapestry", "testament", "testament to", "underscore", "underscoring",
    "pivotal", "vibrant", "seamless", "myriad", "plethora", "ever-evolving",
    "rapidly evolving", "in conclusion", "at its core", "stands as", "serves as",
    "dive into", "it's worth noting", "here's what you need to know",
    "the future looks bright", "exciting times", "game-changer", "paradigm",
]
# REVIEW ONLY - never a failure on its own. These are legitimate technical nouns or
# ordinary English in this blog and only become tells in specific uses:
#   harness   = test/eval harness          landscape = loss landscape
#   robust    = robust estimator           leverage  = leverage ratio (finance)
#   actually/only/not just = ordinary usage
REVIEW_ONLY = [
    "actually", "harness", "robust", "landscape", "leverage", "comprehensive",
    "unlock", "valuable", "enhance", "streamline", "cutting-edge", "realm",
    "not just", "only",
]

# Headings capitalised because they are PROPER NOUNS, not because of the AI title-case tell.
# A regex cannot distinguish 'Deep Learning Indaba' (a real conference) from 'The Mathematical
# Foundation' (a genuine tell), so the corpus's genuine names are listed here. Without this, a
# pass contorts real names to satisfy the gate - one renamed '## Deep Learning Indaba' to
# '## The Indaba' purely to clear a false positive.
TITLE_CASE_ALLOWLIST = {
    "data science nigeria",
    "deep learning indaba",
    "onnx runtime mobile",
    "qubit by qubit",
    "african masters of machine learning",
}

HEADING_ANY_RE = re.compile(r"(?m)^#{2,4}\s+(.+?)\s*$")

# Kept for reference in docs/tests; titlecase_hits() no longer uses a pre-filtered pattern.
HEADING_RE = re.compile(r"(?m)^#{2,4}\s+(?:[A-Z][\w-]+\s+){1,}[A-Z][\w-]+\s*$")

# Words that are legitimately capitalised anywhere in a heading (proper nouns, acronyms, brand and
# product names). Title case means an ORDINARY word capitalised after the first position - so
# 'Python Implementation' is a tell (Implementation) while 'Install Kserve' and 'REST API' are not.
# Matching whole headings missed the general case and needed a new entry every time a proper noun
# appeared; matching words generalises.
PROPER_WORDS = {
    "api", "rest", "graphql", "grpc", "http", "cli", "sdk", "gpu", "cpu", "tpu", "aws", "gcp",
    "openai", "anthropic", "mistral", "microsoft", "google", "meta", "nvidia", "amazon", "apple",
    "docker", "compose", "kubernetes", "helm", "argocd", "prometheus", "grafana", "evidently",
    "python", "pytorch", "tensorflow", "scikit-learn", "scikit", "cnn", "rnn", "gnn", "llm",
    "africa", "african", "kenya", "kenyan", "nigeria", "nigerian", "america", "latin", "europe",
    "south", "north", "east", "west", "central", "southern", "sub-saharan", "middle",
    "indaba", "onnx", "runtime", "mobile", "lacuna", "fund", "kserve", "masakhane", "zindi",
    "neptune", "cassandra", "mongo", "redis", "postgres", "elasticsearch", "langfuse", "ollama",
    "celery", "kafka", "rabbitmq", "pandas", "numpy", "fastapi", "django", "flask", "airflow",
    "notion", "linear", "vercel", "cloudflare", "supabase", "firebase", "stripe", "mpesa",
    "hugging", "face", "dijkstra", "github", "gitlab", "slack", "discord", "zoom", "jira",
    "transformers", "huggingface",
    "android", "claude", "copilot", "sonnet", "chatgpt", "wikipedia", "lagos", "nairobi",
    "lelapa", "stakpak", "trivy", "swahili", "china", "asia", "act", "crew", "lite", "kit",
    "alertmanager", "flutterwave", "safaricom", "mpesa", "maize", "kaggle", "juypter",
    "chinese", "brazil", "brazilian", "russia", "russian", "astra", "danske", "mena", "gbp",
    "european", "german", "germany", "french", "india", "indian", "japan", "japanese", "korea",
    "kimi", "core42", "accra", "sber", "a24", "gemini", "omni", "alibaba", "yandex", "moonshot",
    "anthropic", "openai", "telus", "safaricom", "equity", "ncba", "cohere", "aleph", "databricks",
    "qubit", "hermitian", "gulf", "nollywood", "dijkstra", "juspay", "synapse", "vera", "rubin",
    "alice", "mermaid", "pagerduty", "interpol", "europol", "uniswap", "binance", "coinbase",
    "vienna", "terp", "gu", "badnets", "evermemos", "autoresearchclaw",
}


# Leading enumeration on a heading ('2. Tool call frequency', 'Step 5: Run It', 'Strategy 1: ...')
# is NOT title case - the first real word after it is position 0. Without this strip, every
# numbered list heading reads as a tell and children sentence-case genuinely correct headings.
_HEADING_NUMBERING_RE = re.compile(r"^(?:\d+[.)]|[A-Z][a-z]+\s+\d+[:.)])\s*")


def _is_acronym(w):
    """True for all-caps AND mixed-case acronyms (REST, LoRA, KGs, SafeInstr).

    A plain .isupper() test only catches the first form, so mixed-case proper terms like LoRA
    and KGs were flagged as ordinary capitalised words.
    """
    return sum(c.isupper() for c in w) >= 2


def _is_identifier(w):
    """True when a token carries a digit: S3, K3, Core42, A24, GPT-5, R$23, GB200.

    These are product names, versions or currency amounts, never an ordinary English word
    capitalised mid-heading. Without this, children reworded correct headings to dodge false
    flags - '### Local MLflow server with PostgreSQL and S3' was rewritten to lead with 'S3'.
    """
    return any(c.isdigit() for c in w)


def is_tell_word(w):
    """True when `w`, appearing AFTER position 0 of a heading, is an ordinary capitalised word.

    Single source of truth for both the detector and the sentence-case fixer. When the two had
    separate rules they drifted: the fixer lowercased words the detector never flagged, and left
    capitalised words it did.
    """
    bare = w.strip("*_`\"'()[]:,.?!")
    core = bare.lower()
    core_base = re.sub(r"['\u2019]s$", "", core)
    if not core or len(core) <= 1 or core in PROPER_WORDS or core_base in PROPER_WORDS:
        return False
    if _is_acronym(bare) or _is_identifier(bare):
        return False
    if core in {"i", "i'd", "i'm", "i've", "i'll"}:
        return False
    return w[:1].isupper()


def titlecase_hits(text):
    """Return headings that are genuine Title Case (an ordinary word capitalised after position 0).

    Scans EVERY heading, not a pre-filtered regex. A regex cannot express this test: title case
    survives lowercase connectors ('Queue-Based Architecture with Redis + Celery') and trailing
    punctuation ('What We'd Do Differently (The Honest Retrospective)').
    """
    hits = []
    for raw in HEADING_ANY_RE.findall(text):
        name = raw.strip().rstrip("#").strip()
        name = _HEADING_NUMBERING_RE.sub("", name)
        if not name or name.lower() in TITLE_CASE_ALLOWLIST:
            continue
        # substring match too: 'Data Science Nigeria (DSN)' carries a suffix the exact form misses
        if any(ph in name.lower() for ph in TITLE_CASE_ALLOWLIST):
            continue
        words = name.split()
        for w in words[1:]:
            if is_tell_word(w):
                hits.append(name)
                break
    return hits
# Includes regional-indicator flag emoji (U+1F1E6-1F1FF) used decoratively on region headings -
# they sit outside the main pictographic ranges and were being missed entirely.
EMOJI_RE = re.compile(r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]")
FENCE_RE = re.compile(r"(?m)^```")
# Reference-like sections must be excluded at ANY heading level, under any of the names the
# corpus actually uses: '## References' (82 posts), '## Related posts' (40), '## Further
# Reading' (10), '### References' (5), '## Further reading' (1). Missing a variant counts its
# separators toward the em-dash band and pushes a post over for no real reason.
SPLIT_RE = re.compile(
    r"(?mi)^#{2,6}\s+(?:References|Related posts|Further Reading)\s*$"
)


def body_prose(text: str) -> str:
    """Article prose: front matter, code fences, inline code and ref blocks removed."""
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    # pair fences by alternating tokens; naive regex pairs from the CLOSING fence
    parts = FENCE_RE.split(text)
    text = "".join(parts[i] for i in range(0, len(parts), 2))
    text = re.sub(r"`[^`\n]*`", " ", text)
    m = SPLIT_RE.search(text)
    if m:
        text = text[: m.start()]
    return text


def audit(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    prose = body_prose(raw)
    low = prose.lower()
    vocab = {w: low.count(w) for w in AI_VOCAB if low.count(w)}
    review = {w: low.count(w) for w in REVIEW_ONLY if low.count(w)}
    em = prose.count("\u2014")
    title_case = len(titlecase_hits(prose))
    emoji = len(EMOJI_RE.findall(prose))
    failures = []
    if em > MAX_EM_DASHES:
        failures.append(f"prose em dashes {em} > {MAX_EM_DASHES} (humanizer pattern 14)")
    if title_case > MAX_TITLE_CASE_HEADINGS:
        failures.append(f"title-case headings {title_case} (pattern 17)")
    if emoji > MAX_EMOJI:
        failures.append(f"emoji {emoji} (pattern 18)")
    if vocab:
        failures.append(f"AI vocabulary {vocab} (pattern 7)")
    return {
        "file": str(path),
        "words": len(prose.split()),
        "prose_em_dashes": em,
        "title_case_headings": title_case,
        "emoji": emoji,
        "ai_vocabulary": vocab,
        "review_only": review,
        "pass": not failures,
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = [audit(p) for p in args.paths]
    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if all(r["pass"] for r in results) else 1

    for r in results:
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"[{flag}] {pathlib.Path(r['file']).name}  "
              f"words={r['words']} prose_em_dashes={r['prose_em_dashes']} "
              f"titlecase={r['title_case_headings']} emoji={r['emoji']}")
        for f in r["failures"]:
            print(f"         - {f}")
        if r["review_only"]:
            print(f"         (review, not a failure): {r['review_only']}")

    bad = [r for r in results if not r["pass"]]
    if bad:
        print(f"\n{len(bad)}/{len(results)} post(s) off the house band. "
              f"Run the humanizer pass (skill_view humanizer) before committing.")
        return 1
    print(f"\nAll {len(results)} post(s) in band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
