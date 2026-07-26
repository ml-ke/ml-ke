# Chirpy Image Troubleshooting Reference

Real debugging history from getting images to work on ml.co.ke (Jekyll + Chirpy v7.5).

## Symptom: Images Show as Broken (404 or DNS error — `https://assets/img/...`)

**Root cause:** `baseurl` in `_config.yml` is `"/"` instead of `""`. Chirpy prepends `site.baseurl` to media paths. With `baseurl: "/"`, a path like `/assets/img/...` becomes `//assets/img/...` — protocol-relative URL where browser interprets `assets` as the hostname.

**Fix:** `baseurl: ""` produces clean `/assets/img/...` paths.

## Symptom: Homepage Shows Blurry LQIP But Not Real Image

Chirpy v7.5 uses `data-src` for lazy loading with IntersectionObserver:

```html
<div class="preview-img blur">
  <img data-src="/assets/img/post.webp" alt="..." data-lqip="true"
       src="data:image/webp;base64,...">
</div>
```

JavaScript in `home.min.js` swaps `data-src` → `src` on page load:
```javascript
function on() { this.src = this.getAttribute("data-src"); }
document.querySelectorAll('article img[data-lqip="true"]').forEach(t => on.call(t));
```

**Troubleshoot:**
1. Hard refresh (Ctrl+Shift+R) — browser cached old `//assets/` URLs
2. `baseurl` must be `""` not `"/"`
3. Verify WebP exists: `curl -s -o /dev/null -w "%{http_code}" https://ml.co.ke/assets/img/slug.webp`
4. Scroll — IntersectionObserver triggers near viewport

## LQIP Front Matter Format

```yaml
image:
  path: /assets/img/post-slug.webp
  alt: Description
  lqip: data:image/webp;base64,UklGRnIAAABXRUJQVlA...
```

The `lqip` value is a COMPLETE data URI (including `data:image/webp;base64,` prefix).

## Quick Verification

```bash
curl -s -o /dev/null -w "%{http_code}" https://ml.co.ke/assets/img/gradient-descent.webp
curl -s https://ml.co.ke | grep -oP 'data-src="[^"]+\.webp"' | head -3
curl -s https://ml.co.ke | grep 'src="//assets'  # Should return NOTHING
```
