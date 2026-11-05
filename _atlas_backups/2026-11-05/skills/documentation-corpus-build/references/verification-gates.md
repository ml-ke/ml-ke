# Verification gates for a documentation corpus

One checker, run before every publish and in CI. It is cheap, deterministic and offline except for the optional link check. Start from `scripts/check_docs_corpus.py` and adapt the config block rather than writing a new one.

## The eight checks and what each catches

| # | Check | Catches | Severity |
|---|-------|---------|----------|
| 1 | Manifest ↔ filesystem both ways | orphan pages, promised pages never written | error |
| 2 | Structure (metadata line, first section, closing sections, prose budget) | writers drifting from the house style | error |
| 3 | Internal cross-links resolve | renames, deleted pages, links by filename instead of page name | error |
| 4 | Embedded assets exist on disk | pages pointing at images nobody generated | error |
| 5 | Mermaid parses under the real parser | a diagram that renders as an error box in the browser | error |
| 6 | No credential shapes; filler words | a pasted key, hype vocabulary creeping in | error / warning |
| 7 | External links resolve | invented deep URLs, link rot | error on 404 |
| 8 | Spot-check version-specific claims | confident fiction (a command, price or quota that does not exist) | manual |

## Parsing pitfalls that make a checker silently useless

**`findall` with one capture group returns strings, not tuples.** `re.findall(r"\[([^\]]*)\]\(([^)]+)\)")` is fine because it has two groups, but a pattern with a single group returns a list of strings and `for label, target in ...` raises `too many values to unpack`. Use two groups, or `finditer` with `group(1)`.

**A `#` inside a fenced code block is not an H1.** Shell comments, YAML comments and markdown examples inside fences all look like headings. Compute `prose = re.sub(r"```.*?```", "", text, flags=re.S)` and run heading, length and filler-word checks on `prose`; run secret and link checks on the full text (a secret in a code block is still a secret).

**A naive inline-code stripper eats the fence markers.** `re.sub(r"`[^`\n]*`", "", text)` matches the first two backticks of a fence, turning a three-backtick mermaid fence into a one-backtick fragment — your fence count silently becomes zero and the balance check always passes. Strip only one-or-two-backtick spans:

```python
fences = re.sub(r"(?<!`)`{1,2}(?!`)[^`\n]*?`{1,2}(?!`)", "", text)
blocks = re.findall(r"```mermaid\n(.*?)```", fences, flags=re.S)
if fences.count("```mermaid") != len(blocks):
    error("unbalanced mermaid fence")
```

**Word counts must exclude fences and tables**, or a page full of code always looks long and a table-heavy page always looks short. Strip fenced blocks and table rows, then count whitespace-separated tokens.

**Exempt generated pages.** Home, section indexes, `_Sidebar` and `_Footer` do not follow the lesson skeleton. Keep an explicit exemption set (and an alias map, e.g. `Home` → `00-Home.md`) or the checker drowns real defects in false positives on the pages you generate.

## Validating mermaid properly

A regex cannot tell you whether a diagram renders. Parse every block with the real library:

```bash
mkdir -p mmcheck && cd mmcheck && npm init -y >/dev/null && npm install mermaid jsdom --silent
node validate_mermaid.mjs ../path/to/wiki
```

```js
// validate_mermaid.mjs — parses every mermaid block, prints per-file failures
import fs from 'node:fs'; import path from 'node:path';
import { JSDOM } from 'jsdom';
const dir = process.argv[2];
const dom = new JSDOM('<!doctype html><html><body></body></html>', { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
// navigator is a getter-only property on globalThis in current Node — plain assignment throws
Object.defineProperty(globalThis, 'navigator', { value: dom.window.navigator, configurable: true });
const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.md'))) {
  const blocks = [...fs.readFileSync(path.join(dir, f), 'utf8').matchAll(/```mermaid\n([\s\S]*?)```/g)];
  for (const [i, m] of blocks.entries()) {
    try { await mermaid.parse(m[1].trim()); }
    catch (e) { console.log(`FAIL ${f} [${i + 1}] ${String(e.message).split('\n')[0]}`); }
  }
}
```

Keep the corpus's mermaid to a small allow-list of diagram types, ban init directives and `click` handlers, quote labels containing punctuation, and cap diagrams at roughly twenty nodes. Those four rules prevent nearly every parse failure.

## External link checking

Extract unique markdown link targets across all pages, then check them in parallel (a thread pool of ~12 is enough), with a browser user agent and a timeout:

```bash
curl -sSL -o /dev/null -w "%{http_code}" -A "Mozilla/5.0 … Chrome/126 Safari/537.36" --max-time 25 --retry 1 "$url"
```

Interpreting the results matters more than collecting them:

- **2xx/3xx** — fine.
- **blank status** — the request itself failed (TLS, timeout, redirect loop). Retry once before calling it a defect.
- **403/429** — the site blocks non-browser clients. Unverifiable, not a defect; do not chase it.
- **404** — a real defect: an invented or dead deep link. Replace it with the documentation root rather than guessing another path.
- **Own-repo raw asset URLs 404 until the first push.** Expected before publication; re-run the check after pushing.

Because the link bank is the only permitted source, a 404 usually means someone typed a plausible path from memory rather than copying a verified URL.

## Sanitisation scan

Grep the corpus for the real identifier list and for credential shapes, and report the count rather than only the first hit:

```bash
# identifiers: every real project, client, vendor, hostname and platform you harvested from
grep -rniE "projectname|clientname|\.railway\.app|workers\.dev|\.neon\.tech|supabase\.co|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" .
# credential shapes
grep -rnoE "(sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{20,}\.|-----BEGIN [A-Z ]*PRIVATE KEY-----)" .
```

Expect false positives from case-insensitive word matches on short project names — read the hits. Any credential-shaped hit is a defect, always.

## Spot-checking claims

Pick one version-specific or numeric claim per section and search the harvested source for it. If you cannot find it, the writer invented it: delete it, hedge it, or attribute it. This is a two-minute check that catches the failure mode readers punish hardest.
