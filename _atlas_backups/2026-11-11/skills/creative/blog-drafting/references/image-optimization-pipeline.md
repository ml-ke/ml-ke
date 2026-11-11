# Image Optimization Reference

## Hydroxide Pipeline Results (June 2026)

Ran on 7 old BlogPhotos (5-6 MB each at 2752×1536) + 9 new SVG-to-PNG covers.

### Old Images

| Image | Before | After (WebP) | Reduction |
|-------|--------|-------------|-----------|
| gradient-descent.png | 5,243 KB | 60 KB | 99% |
| 2ndOrder.png | 6,209 KB | 106 KB | 98% |
| MBMnLR.png | 6,005 KB | 128 KB | 98% |
| Knowledge-Graphs.png | 5,123 KB | 59 KB | 99% |
| KGpythoN4j.png | 5,961 KB | 116 KB | 98% |
| PagerankCommDetecN4j.png | 6,011 KB | 121 KB | 98% |
| GNNs.png | 5,072 KB | 62 KB | 99% |

### New SVG Covers

| Image | Before (PNG) | After (WebP) | Reduction |
|-------|-------------|-------------|-----------|
| cover-kg-embeddings | 54 KB | 15 KB | 72% |
| cover-kg-gnn | 71 KB | 17 KB | 75% |
| cover-kg-llm | 60 KB | 17 KB | 70% |
| cover-kg-production | 52 KB | 13 KB | 75% |
| cover-kg-neo4j | 68 KB | 17 KB | 74% |
| cover-ai-prompt-injection | 68 KB | 16 KB | 76% |
| cover-ai-jailbreak | 51 KB | 14 KB | 72% |
| cover-ai-data-poisoning | 50 KB | 13 KB | 73% |
| cover-ai-agent-security | 61 KB | 17 KB | 72% |
| cover-ai-supply-chain | 53 KB | 13 KB | 75% |

**Totals:** ~40 MB → 813 KB (98% reduction)

## Chirpy LQIP Support

The Chirpy theme v7+ renders `image.lqip` in front matter as a CSS background placeholder. The value must be a complete data URI. The theme uses it as a low-res preview while the real image loads from the network.

### Supported format

```yaml
image:
  path: /assets/img/slug.webp
  alt: Description
  lqip: data:image/webp;base64,UklGRnIAAABX...  # Full data URI
```

### LQIP generation

```python
from PIL import Image
from io import BytesIO
import base64

img = Image.open("source.webp").convert("RGB")
w, h = img.size
thumb = img.resize((20, max(1, int(20 * h / w))), Image.LANCZOS)

buf = BytesIO()
thumb.save(buf, "WEBP", quality=20)
b64 = base64.b64encode(buf.getvalue()).decode()
data_uri = f"data:image/webp;base64,{b64}"
```

## Tools Reference

All tools live at `~/ProG/ml-ke/tools/`:

| Tool | Purpose | When to Run |
|------|---------|-------------|
| `generate_covers.py` | Generate SVG cover images from spec | Before writing new series |
| `convert_svgs.py` | Convert SVGs to PNG (via cairosvg) | After generating SVGs |
| `optimize_images.py` | Resize → WebP → LQIP for ALL images | Before deployment |
| `update_frontmatter.py` | Rewrite all _posts/ front matter | After optimization |
| `add_alt_text.py` | Fix alt text after mass front matter update | After update_frontmatter |

### Full pipeline command

```bash
cd ~/ProG/ml-ke
python3 tools/generate_covers.py
python3 tools/convert_svgs.py
python3 tools/optimize_images.py
python3 tools/update_frontmatter.py
python3 tools/add_alt_text.py
git add -A
git commit -m "Update covers and optimize images"
git push origin main
```
