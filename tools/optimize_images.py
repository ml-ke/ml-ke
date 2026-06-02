"""
Hydroxide — Image optimization pipeline for ML Kenya blog.
Resizes, compresses to WebP, generates LQIP placeholders.
"""
import subprocess, sys, os, base64, re

VENV_PYTHON = "/tmp/svg_venv/bin/python3"
BLOG_DIR = "/home/pro-g/ProG/ml-ke"
OLD_DIR = f"{BLOG_DIR}/assets/BlogPhotos"
NEW_DIR = f"{BLOG_DIR}/assets/blog"
OUT_DIR = f"{BLOG_DIR}/assets/img"  # WebP + LQIP output
LQIP_DIR = f"{OUT_DIR}/lqip"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LQIP_DIR, exist_ok=True)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    if r.returncode != 0:
        print(f"  FAIL: {' '.join(cmd[:3])} — {r.stderr.strip()[:120]}")
    return r.returncode == 0

def optimize_image(src_path, basename, out_name=None):
    """Resize → WebP → LQIP in one pass using venv Python + Pillow."""
    if out_name is None:
        out_name = basename.rsplit('.', 1)[0]
    
    webp_path = f"{OUT_DIR}/{out_name}.webp"
    lqip_path = f"{LQIP_DIR}/{out_name}.txt"
    
    if os.path.exists(webp_path) and os.path.exists(lqip_path):
        return webp_path, lqip_path  # already done
    
    python_code = f"""
import os, sys
from PIL import Image
import base64

src = {repr(src_path)}
webp = {repr(webp_path)}
lqip_path = {repr(lqip_path)}
out_name = {repr(out_name)}

img = Image.open(src).convert("RGB")

# Resize: max 1200px wide, maintain aspect ratio
w, h = img.size
if w > 1200:
    ratio = 1200.0 / w
    w, h = 1200, int(h * ratio)
img = img.resize((w, h), Image.LANCZOS)

# Save as WebP (quality 85)
img.save(webp, "WEBP", quality=85)
webp_size = os.path.getsize(webp)

# Generate LQIP — tiny 20px-wide blurred thumbnail encoded as base64
lqip_img = img.copy()
lqip_w, lqip_h = 20, max(1, int(20 * h / w))
lqip_img = lqip_img.resize((lqip_w, lqip_h), Image.LANCZOS)

# Save tiny JPEG as bytes, then base64
from io import BytesIO
buf = BytesIO()
lqip_img.save(buf, "JPEG", quality=30)
b64 = base64.b64encode(buf.getvalue()).decode()
with open(lqip_path, "w") as f:
    f.write(b64)

original_size = os.path.getsize(src)
print(f"  {{out_name}}: {{original_size//1024}}KB → {{webp_size//1024}}KB WebP ({{'%.0f' % ((1-webp_size/original_size)*100)}}% smaller)  LQIP={{len(b64)}}B")
"""
    
    r = subprocess.run([VENV_PYTHON, "-c", python_code], capture_output=True, text=True)
    if r.returncode == 0:
        sys.stdout.write(r.stdout)
    else:
        print(f"  FAIL: {out_name} — {r.stderr.strip()[:200]}")
    
    return webp_path, lqip_path


# === Process old BlogPhotos if they exist ===
old_mapping = {}
if os.path.exists(OLD_DIR):
    print("=== Processing old BlogPhotos (massive 5-6MB originals) ===")
    for fname in sorted(os.listdir(OLD_DIR)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Map to chirpy-friendly names
            slug = fname.rsplit('.', 1)[0]
            slug_map = {
                "2ndOrder": "second-order-optimization",
                "GNNs": "gnn-fundamentals",
                "gradient-descent": "gradient-descent",
                "KGpythoN4j": "neo4j-tutorial",
                "Knowledge-Graphs": "kg-fundamentals",
                "MBMnLR": "momentum-adaptive-lr",
                "PagerankCommDetecN4j": "graph-algorithms-neo4j",
            }
            out_name = slug_map.get(slug, slug.lower().replace('_', '-'))
            src = os.path.join(OLD_DIR, fname)
            optimize_image(src, fname, out_name)
            old_mapping[slug] = out_name

# === Process new SVG-to-PNG covers (9 images) ===
print("\n=== Processing new blog covers (PNG 50-72KB) ===")
new_mapping = {}
for fname in sorted(os.listdir(NEW_DIR)):
    if fname.lower().endswith('.png') and fname != '.gitkeep':
        slug = fname.rsplit('.', 1)[0]
        src = os.path.join(NEW_DIR, fname)
        optimize_image(src, fname, slug)
        new_mapping[slug] = slug

# === Summary ===
print("\n=== Summary ===")
webp_files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith('.webp'))
total_orig = 0
total_webp = 0
for wf in webp_files:
    name = wf.rsplit('.', 1)[0]
    wsize = os.path.getsize(os.path.join(OUT_DIR, wf))
    # Find original
    for d in [d for d in [OLD_DIR, NEW_DIR] if os.path.exists(d)]:
        for f in os.listdir(d):
            if f.startswith(name) or name in f:
                total_orig += os.path.getsize(os.path.join(d, f))
                break
    total_webp += wsize
    print(f"  {wf}: {wsize//1024}KB")

print(f"\nTotal original: {total_orig//1024//1024} MB → Total WebP: {total_webp//1024} KB")
print(f"Overall reduction: {(1-total_webp/total_orig)*100:.0f}%")
print(f"\nLQIP placeholders generated in {LQIP_DIR}/")
