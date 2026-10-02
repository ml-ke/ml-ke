# Validation toolkit

Copy-paste starting points for the four-layer verification described in SKILL.md.

## 1. Structure, links, assets and secrets checker (`scripts/check_wiki.py`)

Dependency-free, offline, safe in CI. Returns non-zero on errors; `--strict` also fails on warnings.

| Check | Catches |
|---|---|
| Filename in the manifest, both directions | orphans and missing pages (a renamed page silently breaks links) |
| No H1 in prose | the wiki renders the page title; a duplicate H1 is visible noise |
| First line is the metadata blockquote | pages that lost their header while being edited |
| Required sections present (`Try it`, `Common mistakes`, `Key takeaways`, `Further learning`) | pages that drifted from the template |
| Prose word count in band | bloated pages nobody finishes |
| Internal links resolve to real pages | the most common defect across 135 pages |
| Embedded asset URLs exist on disk | broken images after an asset is renamed |
| Mermaid fences balanced, allowed types, no `%%{init}`, no `click`, <= 20 nodes | diagrams that fail to render |
| Banned hype words | house voice drift |
| Secret-shaped strings anywhere, including code fences | a credential in the published corpus |

Implementation details that were bugs first:

```python
# prose = page minus fenced code blocks, so shell comments (# ...) are not read as headings
prose = re.sub(r"```.*?```", "", text, flags=re.S)

# strip inline code spans for fence counting, but NEVER match three backticks
fences = re.sub(r"(?<!`)`{1,2}(?!`)[^`\n]*?`{1,2}(?!`)", "", text)

# link extraction must skip images and tolerate a bare page-name link
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
# findall with ONE group returns strings, not tuples -> use two groups or finditer
```

Secret patterns worth having: `sk-[A-Za-z0-9]{16,}`, `gh[pousr]_[A-Za-z0-9]{20,}`, `AKIA[0-9A-Z]{12,}`,
`eyJ[A-Za-z0-9_-]{20,}\.`, `whsec_[A-Za-z0-9]{16,}`, `-----BEGIN [A-Z ]*PRIVATE KEY-----`, and a
`postgres(ql)?://user:pass@` shape.

Manifest coverage check: a page is exempt from the lesson skeleton if it is an index page, the home page,
`_Sidebar` or `_Footer` — otherwise generated index pages fail their own gate.

## 2. Real mermaid validation (`validate_mermaid.mjs`)

```bash
npm init -y && npm install mermaid jsdom
node validate_mermaid.mjs ./wiki
```

```js
import fs from 'node:fs';
import path from 'node:path';
import { JSDOM } from 'jsdom';

const dir = process.argv[2] || './wiki';
const dom = new JSDOM('<!doctype html><html><body></body></html>', { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
// jsdom's navigator is a getter on globalThis in modern Node:
Object.defineProperty(globalThis, 'navigator', { value: dom.window.navigator, configurable: true });

const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });

let total = 0, bad = 0;
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.md')).sort()) {
  const text = fs.readFileSync(path.join(dir, f), 'utf8');
  const blocks = [...text.matchAll(/```mermaid\n([\s\S]*?)```/g)].map(m => m[1]);
  for (let i = 0; i < blocks.length; i++) {
    total++;
    try { await mermaid.parse(blocks[i].trim()); }
    catch (e) { bad++; console.log(`FAIL ${f} [${i + 1}] ${String(e.message).slice(0, 200)}`); }
  }
}
console.log(`parsed ${total} blocks, ${bad} failures`);
process.exit(bad ? 1 : 0);
```

## 3. External link check

```python
from concurrent.futures import ThreadPoolExecutor
import subprocess

def check(u):
    r = subprocess.run(["curl", "-sSL", "-o", "/dev/null", "-w", "%{http_code}",
                        "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
                        "--max-time", "25", "--retry", "1", u],
                       capture_output=True, text=True, timeout=70)
    return u, r.stdout.strip()

with ThreadPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(check, urls))
```

Interpretation:

- Batch failures are usually rate limits. **Re-test failures sequentially with a delay** before calling a
  link dead: 20 of 60 "failures" in one run were 429s or timeouts that resolved to 200.
- A permanent 429/403 from a bot-hostile site (some security vendors and OWASP pages) is not a broken link.
- Own-asset raw URLs 404 until the repo is pushed: check them after the first push, not before.

## 4. Sanitisation scan

Scan the published corpus for the *identities* of the source material, not just secrets: project names,
product names, vendor names, client and org names, internal hostnames, deployment-platform subdomains,
account/service/project IDs, personal paths, emails and IPv4s.

```python
TOKENS = ["<project>", "<vendor>", "<client>", ".railway.app", "workers.dev", "/home/<user>"]
SEC = re.compile(r"(sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|"
                 r"eyJ[A-Za-z0-9_-]{20,}\.|whsec_[A-Za-z0-9]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
```

Expect benign hits and inspect them: `postgres://user:***@localhost:5432/app` in a secrets-hygiene page is
correct, and your own raw-asset URLs are not leaks. Allow-list your own repo URL rather than guessing.

## 5. Post-publish verification

Never report "deployed" on the strength of a successful push.

- `git ls-remote` or clone the target, then count the pages that actually landed.
- `curl -o /dev/null -w '%{http_code}'` a sample of live URLs: home, one per section, and the last page.
- Load the live page in a browser and confirm visually that images resolve, tables render and navigation is
  present. This caught nothing in the final run, which is the point of doing it.
- Re-run the checker against the published copy, not just the source tree, when the two can drift (a wiki
  sync that prunes or renames pages).
