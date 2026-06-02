"""Convert all SVG blog covers to PNG using cairosvg."""
import subprocess, sys, os

SVG_DIR = "/home/pro-g/ProG/ml-ke/assets/blog"
CAIRO = "/tmp/svg_venv/bin/python3"

svg_files = sorted(f for f in os.listdir(SVG_DIR) if f.endswith(".svg"))
print(f"Found {len(svg_files)} SVGs to convert")

for svg in svg_files:
    png = svg.replace(".svg", ".png")
    svg_path = os.path.join(SVG_DIR, svg)
    png_path = os.path.join(SVG_DIR, png)
    
    cmd = [
        CAIRO, "-c",
        f"import cairosvg; cairosvg.svg2png(url='{svg_path}', write_to='{png_path}')"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size = os.path.getsize(png_path)
        print(f"  {svg} → {png} ({size:,} bytes)")
    else:
        print(f"  FAIL: {svg} — {result.stderr.strip()}")
