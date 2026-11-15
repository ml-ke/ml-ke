"""Cover for 2026-11-14: one hard label (a single lit slot) versus a teacher's
ten-number target, with the top-3 truncation cut marked."""
import math

W, H = 1200, 630
F = 'Inter, Helvetica, Arial, sans-serif'
BG = '#0d1117'
CYAN, GREEN, PURPLE, GREY = '#00d2ff', '#6bcf7f', '#a78bfa', '#8b949e'

e = []
e.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-label="A single lit slot labelled hard label beside a ten-number teacher '
         f'target, with the top-3 cut marked at 0.685">')
e.append('<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
         '<stop offset="0" stop-color="#0d1117"/><stop offset="1" stop-color="#1a1a2e"/>'
         '</linearGradient></defs>')
e.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')

# grid
for y in range(60, 630, 45):
    e.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#ffffff" stroke-opacity="0.035"/>')
for x in range(0, 1200, 60):
    e.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#ffffff" stroke-opacity="0.022"/>')

# header
e.append(f'<text x="70" y="150" font-family="{F}" font-size="15" letter-spacing="3" fill="{CYAN}">'
         'KNOWLEDGE DISTILLATION, MEASURED</text>')
e.append(f'<text x="70" y="212" font-family="{F}" font-size="42" font-weight="700" fill="#f0f6fc">'
         'One label holds ten numbers</text>')
e.append(f'<text x="70" y="258" font-family="{F}" font-size="18" fill="{GREY}">'
         'A teacher&#39;s probability vector is a target you can ship per row.</text>')
e.append(f'<text x="70" y="284" font-family="{F}" font-size="18" fill="{GREY}">'
         'A hard label is one number, and the other nine are zeros you invented.</text>')

# ---- left: one-hot, one lit slot
x0, y0, slot_w, slot_h, gap = 70, 350, 18, 120, 8
e.append(f'<text x="{x0}" y="{y0-16}" font-family="{F}" font-size="13" letter-spacing="2" fill="{GREEN}">'
         'HARD LABEL</text>')
e.append(f'<line x1="{x0}" y1="{y0-6}" x2="{x0+190}" y2="{y0-6}" stroke="{GREEN}" '
         'stroke-opacity="0.45" stroke-width="2"/>')
for i in range(10):
    x = x0 + i * (slot_w + gap)
    filled = (i == 2)
    fill = GREEN if filled else 'none'
    op = '0.22' if filled else '0.35'
    e.append(f'<rect x="{x}" y="{y0+20}" width="{slot_w}" height="{slot_h}" rx="4" fill="{fill}" '
             f'fill-opacity="{op}" stroke="{GREEN if filled else "#30363d"}" stroke-opacity="0.8"/>')
    if filled:
        e.append(f'<text x="{x+slot_w/2}" y="{y0+20+slot_h/2+5}" text-anchor="middle" '
                 f'font-family="{F}" font-size="13" fill="#f0f6fc">1</text>')
e.append(f'<text x="{x0}" y="{y0+165}" font-family="{F}" font-size="13" fill="#6e7681">'
         'one lit slot, nine zeros</text>')
e.append(f'<text x="{x0}" y="{y0+186}" font-family="{F}" font-size="13" fill="#6e7681">'
         'entropy 0.00 bits</text>')

# ---- arrow
e.append(f'<path d="M 330 430 L 470 430" stroke="{PURPLE}" stroke-width="2" stroke-opacity="0.8" '
         'marker-end="url(#ar)"/>')
e.append(f'<text x="332" y="418" font-family="{F}" font-size="12" fill="{PURPLE}">same row,</text>')
e.append(f'<text x="332" y="452" font-family="{F}" font-size="12" fill="{PURPLE}">ten numbers</text>')

# ---- right: teacher target, ten graded bars
p = [0.42, 0.27, 0.14, 0.06, 0.04, 0.03, 0.02, 0.01, 0.006, 0.004]
tot = sum(p)
p = [v / tot for v in p]
hmax, bw, bg2 = 120, 40, 12
tx = 520
e.append(f'<text x="{tx}" y="{y0-16}" font-family="{F}" font-size="13" letter-spacing="2" fill="{CYAN}">'
         'TEACHER TARGET, T=2</text>')
e.append(f'<line x1="{tx}" y1="{y0-6}" x2="{tx+560}" y2="{y0-6}" stroke="{CYAN}" '
         'stroke-opacity="0.45" stroke-width="2"/>')
base = y0 + 20 + slot_h
for i, pi in enumerate(p):
    h = max(3.0, pi * hmax / max(p))
    x = tx + i * (bw + bg2)
    keep = i < 3
    e.append(f'<rect x="{x}" y="{base-h:.1f}" width="{bw}" height="{h:.1f}" rx="3" '
             f'fill="{CYAN}" fill-opacity="{0.75 if keep else 0.22}" '
             f'stroke="{CYAN}" stroke-opacity="{0.7 if keep else 0.25}"/>')
# truncation cut after bar 3
cut = tx + 3 * (bw + bg2) - bg2 / 2
e.append(f'<line x1="{cut:.1f}" y1="{base-130}" x2="{cut:.1f}" y2="{base+8}" stroke="{PURPLE}" '
         'stroke-width="2" stroke-dasharray="5 5"/>')
e.append(f'<text x="{cut+8:.1f}" y="{base-118}" font-family="{F}" font-size="12" fill="{PURPLE}">'
         'top-3 cut</text>')
e.append(f'<text x="{tx}" y="{base+28}" font-family="{F}" font-size="13" fill="#6e7681">'
         '1.34 bits per row &#183; 400 rows</text>')

e.append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         'markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#a78bfa"/></marker></defs>')

# ---- bottom band
e.append('<line x1="70" y1="540" x2="1130" y2="540" stroke="#ffffff" stroke-opacity="0.08"/>')
e.append(f'<text x="70" y="572" font-family="{F}" font-size="15" fill="#e6edf3">'
         f'hard labels <tspan fill="{GREEN}">0.617</tspan>'
         ' &#183; distilled <tspan fill="#00d2ff" fill-opacity="1">0.693</tspan>'
         f' &#183; top-3 <tspan fill="{PURPLE}">0.685</tspan>'
         ' &#183; top-1 <tspan fill="#ff6b6b">0.495</tspan></text>')
e.append(f'<text x="70" y="598" font-family="{F}" font-size="13" fill="#6e7681">'
         '1,000 rows distilled at T=2: 0.693 &#183; 4,000 rows on hard labels: 0.679</text>')
e.append(f'<text x="1130" y="596" text-anchor="end" font-family="{F}" font-size="13" '
         'fill="#484f58">ml-ke.github.io</text>')
e.append('</svg>')

svg = "\n".join(e)
assert ' & ' not in svg
open('assets/blog/cover-distillation-target-logit-budget.svg', 'w').write(svg)
print("wrote", len(svg), "bytes")
