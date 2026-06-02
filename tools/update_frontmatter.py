"""Regenerate LQIPs as WebP format and update all post frontmatter."""
import os, base64, subprocess
from io import BytesIO
from PIL import Image

POSTS_DIR = "/home/pro-g/ProG/ml-ke/_posts"
IMG_DIR = "/home/pro-g/ProG/ml-ke/assets/img"
LQIP_DIR = f"{IMG_DIR}/lqip"
VENV = "/tmp/svg_venv/bin/python3"

os.makedirs(LQIP_DIR, exist_ok=True)

# === Regenerate ALL LQIPs as WebP base64 ===
lqip_cache = {}

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.endswith('.webp'):
        continue
    name = fname[:-5]
    src = os.path.join(IMG_DIR, fname)
    
    img = Image.open(src).convert("RGB")
    w, h = img.size
    lqip_w, lqip_h = 20, max(1, int(20 * h / w))
    lqip_img = img.resize((lqip_w, lqip_h), Image.LANCZOS)
    
    buf = BytesIO()
    lqip_img.save(buf, "WEBP", quality=20)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/webp;base64,{b64}"
    lqip_cache[name] = data_uri
    
    with open(os.path.join(LQIP_DIR, f"{name}.txt"), "w") as f:
        f.write(data_uri)

print(f"Generated {len(lqip_cache)} WebP LQIPs")

# === Front matter mapping for ALL posts ===
# Key: post filename (without _posts/) → {img_path, lqip_key}
updates = {
    # Old optimization series
    "2025-11-20-intro-to-gradient-descent.md": {
        "img": "/assets/img/gradient-descent.webp",
        "lqip": "gradient-descent"
    },
    "2025-11-20-momentum-adaptive-learning-rates.md": {
        "img": "/assets/img/momentum-adaptive-lr.webp",
        "lqip": "momentum-adaptive-lr"
    },
    "2025-11-24-second-order-optimization-methods.md": {
        "img": "/assets/img/second-order-optimization.webp",
        "lqip": "second-order-optimization"
    },
    # Old KG series
    "2025-11-24-knowledge-graphs-fundamentals.md": {
        "img": "/assets/img/kg-fundamentals.webp",
        "lqip": "kg-fundamentals"
    },
    "2025-12-01-building-knowledge-graph-neo4j-python.md": {
        "img": "/assets/img/neo4j-tutorial.webp",
        "lqip": "neo4j-tutorial"
    },
    "2025-12-02-graph-algorithms-neo4j.md": {
        "img": "/assets/img/graph-algorithms-neo4j.webp",
        "lqip": "graph-algorithms-neo4j"
    },
    "2025-12-04-graph-neural-networks-fundamentals.md": {
        "img": "/assets/img/gnn-fundamentals.webp",
        "lqip": "gnn-fundamentals"
    },
    # New KG series (my posts)
    "2026-06-01-kg-embeddings.md": {
        "img": "/assets/img/cover-kg-embeddings.webp",
        "lqip": "cover-kg-embeddings"
    },
    "2026-06-01-gnn-knowledge-graph-reasoning.md": {
        "img": "/assets/img/cover-kg-gnn.webp",
        "lqip": "cover-kg-gnn"
    },
    "2026-06-01-kg-production.md": {
        "img": "/assets/img/cover-kg-production.webp",
        "lqip": "cover-kg-production"
    },
    "2026-06-01-kg-llm-rag.md": {
        "img": "/assets/img/cover-kg-llm.webp",
        "lqip": "cover-kg-llm"
    },
    # AI Hacking series
    "2026-06-01-prompt-injection-llm-security.md": {
        "img": "/assets/img/cover-ai-prompt-injection.webp",
        "lqip": "cover-ai-prompt-injection"
    },
    "2026-06-01-jailbreaking-llms.md": {
        "img": "/assets/img/cover-ai-jailbreak.webp",
        "lqip": "cover-ai-jailbreak"
    },
    "2026-06-01-data-poisoning-model-backdoors.md": {
        "img": "/assets/img/cover-ai-data-poisoning.webp",
        "lqip": "cover-ai-data-poisoning"
    },
    "2026-06-01-insecure-agent-design.md": {
        "img": "/assets/img/cover-ai-agent-security.webp",
        "lqip": "cover-ai-agent-security"
    },
    "2026-06-01-supply-chain-ai-attacks.md": {
        "img": "/assets/img/cover-ai-supply-chain.webp",
        "lqip": "cover-ai-supply-chain"
    },
}

# === Update front matter === 
for fname, vals in updates.items():
    fpath = os.path.join(POSTS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {fname}")
        continue
    
    with open(fpath) as f:
        content = f.read()
    
    lqip_val = lqip_cache[vals["lqip"]]
    img_path = vals["img"]
    
    # Build the new image block
    new_img_block = f"""image:
  path: {img_path}
  lqip: {lqip_val}"""
    
    # Replace existing image block or add new one
    import re
    
    # Check if image: block exists
    has_image = re.search(r'^image:', content, re.MULTILINE)
    
    if has_image:
        # Replace existing image block (from "image:" line until next frontmatter key or end)
        content = re.sub(
            r'^image:\n(?:  .*\n?)*',
            new_img_block + '\n',
            content,
            flags=re.MULTILINE
        )
    else:
        # Add image block before last ---
        content = content.rstrip()
        if content.endswith('---'):
            content = content[:-3] + new_img_block + '\n---\n'
    
    with open(fpath, 'w') as f:
        f.write(content)
    
    print(f"  UPDATED: {fname}")

print("\nDone! All post front matter updated with WebP + LQIP.")
