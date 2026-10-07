# Generating infographics without a plotting library

When matplotlib, PIL, numpy and graphviz are all absent, render HTML/CSS to PNG with headless Chrome. No dependency install, crisp typography, and the source stays editable as a real file in the repo.

## The command

```bash
google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1600,900 --force-device-scale-factor=1 \
  --screenshot=assets/05-pipeline-gates.png file:///abs/path/card.html
```

- `--window-size` *is* the canvas: design a fixed-size card and let the content fit. Auto-height screenshots are inconsistent across versions.
- `--hide-scrollbars` matters; without it a scrollbar lands in the image.
- `--no-sandbox` is required in containers.
- Treat a PNG larger than ~5KB as the smoke test for "it actually rendered"; a blank or near-empty file means a CSS or path problem, not a layout problem.
- Enumerate candidate binaries: `google-chrome`, `chromium`, `chromium-browser`.

## Structure the generator as data + one loop

One Python file with a `CARDS` dict of `name -> html` and a single build loop beats twenty bespoke commands: you re-run the whole set after every styling change, in seconds.

```python
CARDS = {}
CARDS["05-pipeline-gates"] = page(title="Gates, cheapest first", tag="Section 5",
                                   sub="...", body=table([...]), foot=FOOT)
```

Helpers worth writing once: `page()` (title, tag, subtitle, body, footer), `card()`, `table()`, `bars()` (horizontal labelled bars), `nodes()` (a row of nodes with arrows), and `arrow()`. Then each infographic is 5–20 lines of composition.

## The pitfall that costs ten minutes every time

**Do not `%`-format a CSS string.** CSS contains `100%`, and both `body{width:%(w)dpx}`-style formatting and a stray `%` raise misleading errors (`not enough arguments for format string`, or an unsupported format character). Build the CSS with explicit token replacement instead:

```python
CSS = """body{width:__W__px;height:__H__px} .fill{height:100%}"""
CSS = CSS.replace("__W__", str(W)).replace("__H__", str(H))
```

Related: inside f-strings a literal `%` for a bar width is fine (`f"width:{pct}%"`) — the problem is only `%`-formatting applied to a large stylesheet.

## CSS skeleton that renders well at 1600x900 (dark theme)

- Body: fixed width/height, `background:#0d1117`, `color:#e6edf3`, generous padding, `display:flex; flex-direction:column` with a `gap`.
- A title block with an optional pill "tag" (section number), then a one-line subtitle in muted grey that states the takeaway — the subtitle is where the teaching happens.
- A `.body` wrapper with `flex:1; justify-content:center` so cards sit vertically centred rather than hugging the top.
- Cards: `background:#161b22; border:1px solid #30363d; border-radius:12px`. Accent borders for emphasis (teal = do this, amber = caution, red = this bites).
- Font sizes: 34px title, 17px subtitle, 17px card headings, 14px card body. Anything smaller is unreadable in a wiki column.
- Accent palette that survives GitHub's dark and light themes: teal `#2dd4bf`, blue `#58a6ff`, amber `#d29922`, red `#f85149`, purple `#bc8cff`, green `#3fb950`.

A starter card is in `templates/infographic-card.html`.

## Visual QA

Load two or three PNGs into a vision check before letting the corpus reference them. Look specifically for: text overflowing a card, clipped bottom rows, elements overlapping, and low-contrast muted text. Layout bugs are invisible in the build log and embarrassing in a published page.

## Embedding

Name every asset in the manifest before writers start, then embed by absolute URL so the same markup works in a wiki, a README and a PDF export:

```
![Short real alt text](https://raw.githubusercontent.com/<owner>/<repo>/<branch>/assets/<file>.png)
```

After the first push, curl one asset URL and confirm 200 — that single check tells you every page's images will resolve.
