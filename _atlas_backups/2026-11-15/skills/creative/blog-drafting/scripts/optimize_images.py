#!/usr/bin/env python3
"""
Hydroxide — Image optimization pipeline for ML Kenya blog.
Run from repo root: python3 tools/optimize_images.py

Steps:
  1. Reads SVGs from assets/blog/ and old PNGs from assets/BlogPhotos/
  2. Resizes to max 1200px wide
  3. Converts to WebP quality 85
  4. Generates LQIP (20px WebP base64 placeholder)
  5. Output to assets/img/ and assets/img/lqip/

Requires: Pillow (pip install Pillow) in the /tmp/svg_venv venv.
"""
import os, base64, subprocess, sys
from io import BytesIO
from PIL import Image

BLOG_DIR = os.path.expanduser("~/ProG/ml-ke")
VENV_PYTHON = "/tmp/svg_venv/bin/python3"

def resize_to_webp(src_path, out_name):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    if w > 1200:
        ratio = 1200.0 / w
        w, h = 1200, int(h * ratio)
    img = img.resize((w, h), Image.LANCZOS)
    
    out_dir = os.path.join(BLOG_DIR, "assets", "img")
    lqip_dir = os.path.join(out_dir, "lqip")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(lqip_dir, exist_ok=True)
    
    webp_path = os.path.join(out_dir, f"{out_name}.webp")
    img.save(webp_path, "WEBP", quality=85)
    
    # LQIP
    lqip = img.resize((20, max(1, int(20 * h / w))), Image.LANCZOS)
    buf = BytesIO()
    lqip.save(buf, "WEBP", quality=20)
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    lqip_path = os.path.join(lqip_dir, f"{out_name}.txt")
    with open(lqip_path, "w") as f:
        f.write(f"data:image/webp;base64,{b64}")
    
    orig_size = os.path.getsize(src_path)
    webp_size = os.path.getsize(webp_path)
    pct = (1 - webp_size / orig_size) * 100
    print(f"  {out_name}: {orig_size//1024}KB → {webp_size//1024}KB WebP ({pct:.0f}% smaller)  LQIP={len(b64)}B")

if __name__ == "__main__":
    # Process from assets/blog/ and assets/BlogPhotos/
    for dirname, glob_pattern in [("assets/blog", "*.svg"), ("assets/BlogPhotos", "*.png")]:
        src_dir = os.path.join(BLOG_DIR, dirname)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(('.svg', '.png', '.jpg', '.jpeg')):
                continue
            src = os.path.join(src_dir, fname)
            out_name = os.path.splitext(fname)[0].lower().replace('_', '-')
            resize_to_webp(src, out_name)
