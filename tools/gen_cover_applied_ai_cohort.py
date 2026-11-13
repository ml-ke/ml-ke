#!/usr/bin/env python3
"""Cover for the Google Africa Applied AI Lab cohort post.

Metaphor: a graticule (map grid) with ten tiles pinned to it, sorted into four
columns by the constraint each company removes. No reuse of a sibling cover.
"""
import html

W, H = 1200, 630
BG1, BG2 = "#0d1117", "#1a1a2e"

CLUSTERS = [
    ("RAILS", "#6bcf7f", "who the customer is", ["Smile ID", "Termii", "Vendy"]),
    ("VOICE", "#00d2ff", "how they speak", ["AethexAI", "Intron", "Viamo"]),
    ("FIELD", "#ffd93d", "what the field does", ["Agrails", "Tolbi"]),
    ("EDGE", "#a78bfa", "where compute sits", ["Crane AI Labs", "Satlyt"]),
]


def esc(s):
    return html.escape(s, quote=False)


parts = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" role="img" '
    f'aria-label="Ten startup tiles pinned to a map grid, grouped in four columns by the constraint each removes">'
)
parts.append(
    f'<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
    f'<stop offset="0" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/>'
    f'</linearGradient></defs>'
)
parts.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')

# graticule: horizontal lines plus a soft meridian curve at the right
for y in range(60, 620, 45):
    parts.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#ffffff" stroke-opacity="0.035"/>')
for x in range(0, 1200, 60):
    parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#ffffff" stroke-opacity="0.022"/>')
for i, cx in enumerate((260, 520, 780, 1040)):
    parts.append(
        f'<path d="M {cx} 40 C {cx + 40} 200, {cx - 40} 430, {cx} 600" fill="none" '
        f'stroke="#ffffff" stroke-opacity="0.045" stroke-width="1"/>'
    )

# left: title block
parts.append(f'<text x="70" y="150" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="15" letter-spacing="3" fill="#00d2ff">FIRST COHORT, NOVEMBER 2026</text>')
parts.append(f'<text x="70" y="215" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="42" font-weight="700" fill="#f0f6fc">Ten African startups,</text>')
parts.append(f'<text x="70" y="265" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="42" font-weight="700" fill="#f0f6fc">one constraint map</text>')
parts.append(f'<text x="70" y="310" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="18" fill="#8b949e">Sorted by what each company takes off a</text>')
parts.append(f'<text x="70" y="336" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="18" fill="#8b949e">customer&#39;s plate, not by model size</text>')
parts.append(f'<text x="70" y="392" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="14" fill="#6e7681">thousands of applications</text>')
parts.append(f'<text x="70" y="416" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="14" fill="#6e7681">45+ countries &#183; 10 teams &#183; demo day in Accra</text>')

# right: four cluster columns
col_x = [520, 685, 850, 1015]
tile_w, tile_h = 135, 40
for (name, color, caption, startups), x in zip(CLUSTERS, col_x):
    parts.append(f'<text x="{x}" y="150" font-family="Inter, Helvetica, Arial, sans-serif" '
                 f'font-size="13" letter-spacing="2" fill="{color}">{esc(name)}</text>')
    parts.append(f'<line x1="{x}" y1="162" x2="{x + tile_w}" y2="162" stroke="{color}" '
                 f'stroke-opacity="0.45" stroke-width="2"/>')
    for i, s in enumerate(startups):
        y = 190 + i * 66
        parts.append(
            f'<rect x="{x}" y="{y}" width="{tile_w}" height="{tile_h}" rx="6" '
            f'fill="{color}" fill-opacity="0.10" stroke="{color}" stroke-opacity="0.55"/>'
        )
        parts.append(
            f'<circle cx="{x + 15}" cy="{y + tile_h / 2}" r="4" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + 28}" y="{y + 26}" font-family="Inter, Helvetica, Arial, sans-serif" '
            f'font-size="15" fill="#e6edf3">{esc(s)}</text>'
        )
    parts.append(
        f'<text x="{x}" y="{190 + len(startups) * 66 + 16}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif" font-size="12.5" fill="#8b949e">'
        f'{esc(caption)}</text>'
    )

# bottom rule + footer
parts.append(f'<line x1="70" y1="530" x2="1130" y2="530" stroke="#ffffff" stroke-opacity="0.08"/>')
parts.append(f'<text x="70" y="560" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="14" fill="#6e7681">identity &#183; verification &#183; payments &#183; speech &#183; '
             f'index insurance &#183; offline inference &#183; orbit</text>')
parts.append(f'<text x="1130" y="596" text-anchor="end" font-family="Inter, Helvetica, Arial, sans-serif" '
             f'font-size="13" fill="#484f58">ml-ke.github.io</text>')
parts.append("</svg>")

out = "\n".join(parts)
with open("assets/blog/cover-google-africa-applied-ai-lab-cohort.svg", "w") as f:
    f.write(out + "\n")
print("wrote", len(out), "chars")
