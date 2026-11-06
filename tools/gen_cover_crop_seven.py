import os

W, H = 1200, 630
out = []
a = out.append

a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
a('<defs>')
a('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">')
a('<stop offset="0" stop-color="#0d1117"/><stop offset="1" stop-color="#1a1a2e"/>')
a('</linearGradient>')
a('<linearGradient id="slab" x1="0" y1="0" x2="1" y2="1">')
a('<stop offset="0" stop-color="#16324a"/><stop offset="1" stop-color="#0e2233"/>')
a('</linearGradient>')
a('</defs>')
a(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')

# faint grid
for x in range(0, W, 40):
    a(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#ffffff" stroke-opacity="0.03"/>')
for y in range(0, H, 40):
    a(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#ffffff" stroke-opacity="0.03"/>')

# title
a('<text x="60" y="86" font-family="Helvetica, Arial, sans-serif" font-size="40" font-weight="700" fill="#f0f6fc">Seven Countries, One Set of Weights</text>')
a('<text x="60" y="126" font-family="Helvetica, Arial, sans-serif" font-size="23" fill="#8b98a5">the accuracy was field-validated. whose field?</text>')

# ---- left: weight slab with parameter blocks
a('<g>')
a('<rect x="60" y="186" width="330" height="250" rx="14" fill="url(#slab)" stroke="#00d2ff" stroke-opacity="0.55" stroke-width="2"/>')
a('<text x="84" y="220" font-family="Helvetica, Arial, sans-serif" font-size="16" letter-spacing="3" fill="#00d2ff">SHARED WEIGHTS</text>')
for r in range(6):
    for c in range(9):
        x = 84 + c * 33
        y = 238 + r * 26
        op = 0.25 + ((r * 9 + c) % 5) * 0.11
        a(f'<rect x="{x}" y="{y}" width="26" height="19" rx="3" fill="#00d2ff" fill-opacity="{op:.2f}"/>')
a('<text x="84" y="424" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#c9d1d9">one checkpoint · seven deployments</text>')
a('</g>')

# accuracy stamp on the slab
a('<g transform="rotate(-8 300 470)">')
a('<rect x="196" y="446" width="212" height="62" rx="8" fill="none" stroke="#ffd93d" stroke-width="3" stroke-dasharray="9 6"/>')
a('<text x="212" y="478" font-family="Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#ffd93d">96% accuracy*</text>')
a('<text x="212" y="497" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#ffd93d">measured somewhere else</text>')
a('</g>')

# ---- border line with gates
a('<line x1="470" y1="170" x2="470" y2="520" stroke="#8b98a5" stroke-width="3" stroke-dasharray="12 10" stroke-opacity="0.8"/>')
a('<text x="470" y="160" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" letter-spacing="2" fill="#8b98a5">BORDER</text>')
codes = ["KE", "MA", "IT", "TN", "GH", "TZ", "RW"]
for i, code in enumerate(codes):
    y = 196 + i * 46
    a(f'<rect x="452" y="{y}" width="36" height="26" rx="4" fill="#0e2233" stroke="#6bcf7f" stroke-width="1.5"/>')
    a(f'<text x="470" y="{y+18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" font-weight="700" fill="#6bcf7f">{code}</text>')

# ---- right: seven stacked class-mix tiles
a('<text x="560" y="200" font-family="Helvetica, Arial, sans-serif" font-size="16" letter-spacing="3" fill="#ffd93d">LOCAL CLASS MIX (PRIORS)</text>')
mixes = [
    [0.62, 0.08, 0.14, 0.09, 0.04, 0.03],
    [0.48, 0.12, 0.18, 0.11, 0.07, 0.04],
    [0.70, 0.06, 0.10, 0.07, 0.03, 0.04],
    [0.55, 0.09, 0.16, 0.10, 0.06, 0.04],
    [0.66, 0.07, 0.12, 0.08, 0.04, 0.03],
    [0.50, 0.11, 0.19, 0.10, 0.06, 0.04],
    [0.74, 0.05, 0.09, 0.06, 0.03, 0.03],
]
colors = ["#3b4a5a", "#6bcf7f", "#00d2ff", "#a78bfa", "#ff6b6b", "#ffd93d"]
x0, width = 560, 560
for i, mix in enumerate(mixes):
    y = 222 + i * 42
    a(f'<text x="{x0}" y="{y+16}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#8b98a5">{codes[i]}</text>')
    x = x0 + 44
    total_w = width - 44
    acc = 0.0
    for j, p in enumerate(mix):
        w = p * total_w
        a(f'<rect x="{x+acc:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{colors[j]}" fill-opacity="0.9"/>')
        acc += w
    a(f'<text x="{x0+width+8}" y="{y+17}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#c9d1d9">{mix[4]*100:.0f}%</text>')

a('<text x="560" y="540" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#c9d1d9">the red segment is one disease class; its precision moves with the mix</text>')
a('<text x="560" y="566" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#c9d1d9">same weights, same images, different base rate</text>')

a('<text x="1140" y="600" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#8b98a5">ml-ke.github.io</text>')
a('</svg>')

path = "assets/blog/cover-crop-model-seven-countries.svg"
with open(path, "w") as fh:
    fh.write("\n".join(out) + "\n")
print(path, os.path.getsize(path), "bytes")
