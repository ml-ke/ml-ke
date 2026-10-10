# Infographics and interactive tools

## Pick the right medium

- **Mermaid, inline in the page** — flows, sequences, state machines, ER diagrams, timelines. Diffable,
  reviewable, editable forever, and rendered natively by GitHub. Use it for anything structural.
- **PNG infographic** — what mermaid cannot express: ranked bars, comparisons, ladders, pyramids, cost
  breakdowns, emphasised tables. Generated once, embedded by URL.
- **Single-file HTML tool** — what the reader should *interact* with: a prompt builder, a context-budget
  calculator, a CI pipeline simulator. No CDN, no build step, no tracking; it must run from `file://`.

## Do not assume a plotting stack

Chart libraries are often absent. Check before choosing:

```bash
for m in matplotlib PIL numpy; do python3 -c "import $m" 2>/dev/null || echo "$m MISSING"; done
for b in google-chrome chromium chromium-browser rsvg-convert inkscape dot convert; do command -v "$b"; done
```

A headless Chrome is frequently present where matplotlib is not, and it renders arbitrary HTML/CSS —
enough for every infographic a docs set needs. Batch every image through one committed generator script
(`scripts/build_infographics.py`) so regenerating all assets is one command and CI can diff the result.

## Render recipe

```bash
CHROME=$(command -v google-chrome || command -v chromium || command -v chromium-browser)
"$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1600,900 --force-device-scale-factor=1 \
  --screenshot=/abs/out.png file:///abs/in.html
```

`scripts/html_to_png.sh` wraps this for one-offs.

Pitfalls:

- **A failed render still writes a file** (a few KB, or blank). Treat anything under ~5 KB as failure and
  surface it, or you commit empty images and only notice once they are embedded in a published page.
- **Pass absolute `file:///…` paths** for input and output: Chrome resolves relative paths against its own
  working directory, not your shell's.
- **`--window-size` is the whole canvas** — there is no auto-height. Design each card to a fixed viewport
  and size the page so nothing is clipped at the bottom.
- **Do not `%`-format a CSS string.** CSS contains `100%` and bare `%` inside values, so `"…%s…" % x`
  raises "not enough arguments for format string". Use `.replace("__W__", str(w))` placeholder tokens.
- **Inspect the render visually.** Load two or three PNGs into the vision tool and ask about overflow,
  clipping and contrast. A byte count proves nothing about layout.

## Design rules that survive the medium

- One idea per card: title, a one-line subtitle saying why it matters, body, footnote with the repo URL.
- Dark theme, small palette, high contrast, ~1600×900, large type — these are read scaled-down inside a
  wiki page.
- Build a small CSS + helper layer once (`.card`, `.node`, `.bar`, `.arrow`, a table helper) and reuse it.
  Consistency across twenty images comes from the helpers, not from discipline.
- Put real alt text on every embedded image, and keep the pedagogical sentence in the surrounding prose
  rather than baked into the image, so the lesson stays greppable and translatable.

## Interactive tool pattern

- One self-contained file: inline CSS, inline `<script>`, no external requests, works from `file://`.
- Defaults that teach — a filled example, a preloaded defect, a pre-set budget — plus a reset control.
- Label placeholder numbers as placeholders (prices, quotas) and state that the tool models a concept
  rather than running real code.
- **Verify in a real browser, not by reading the source.** Open the file, drive the interaction
  programmatically, and assert on the resulting DOM:

  ```js
  (() => { const s = document.getElementById('defect'); s.value = 'logic';
           document.getElementById('run').click();
           return { stages: document.querySelectorAll('.stage').length }; })()
  ```

  Then read the result element on the following call, since animations settle asynchronously. This
  catches the silent-JS-error class — a tool that renders but never updates — in two tool calls.
- Index every tool from one `tools/index.html` with a one-line description and the wiki section it
  belongs to.
