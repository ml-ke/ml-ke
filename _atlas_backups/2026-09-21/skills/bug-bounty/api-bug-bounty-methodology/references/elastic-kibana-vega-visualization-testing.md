# Vega Visualization Testing — Live Session Notes

## Environment
- Target: cloud.elastic.co (Kibana 9.4.2)
- Vega spec editor accessible from: Visualize Library → Create → Vega
- Bundled `vega-interpreter` version: 2.2.1

## Vega Spec Tests Performed

### Test 1 — Basic Expression
Confirmed: Vega expressions work. Simple `signal` → `tooltip` rendered "Hello from Vega".

### Test 2 — Tooltip HTML Injection
Tested tooltip with `<b>bold</b>` in data value. HTML was **escaped** (rendered as literal `<b>bold</b>`). 
- `vega_tooltip.js` line 70: `el.innerHTML = createTooltipContent(value, _.escape, 2)`
- lodash `_.escape` escapes `& < > " '` 
- No bypass found with standard HTML injection

### Test 3 — Expression Access
Signal expression `"if(1, 'yes', 'no')"` evaluated to "yes" — basic expressions work.

### Test 3b — Function Reference Probe
`testerr()` caused an error — undefined function references are blocked/rejected, not silently ignored.

### Test 4 — Same-Origin SSRF via URL Data Source
```json
{
  "name": "table",
  "url": "/api/status"
}
```
**Result:** SUCCESS — loaded Kibana status (version 9.4.2).

### Test 4b — Path Traversal SSRF
```json
{
  "name": "table",
  "url": "../api/status"
}
```
**Result:** SUCCESS — relative path traversal also works within same origin.

### Test 4c — Internal API Loading
| URL | Result |
|-----|--------|
| `/api/status` | ✅ Loaded (version: 9.4.2) |
| `/api/features` | ✅ Loaded |
| `../api/status` | ✅ Loaded (path traversal works) |
| `/api/console/proxy?method=GET&path=%2F` | ❌ Loading failed (post-only endpoint) |
| `/internal/search/ok?q=test` | ❌ Loading failed (needs auth headers) |

### Test 5 — External URL Data Source
```json
{
  "name": "table",
  "url": "https://httpbin.org/get"
}
```
**Result:** ❌ "External URLs are not enabled. Add vis_type_vega.enableExternalUrls: true to kibana.yml"

## Key Findings
1. Vega same-origin SSRF works — can read any same-origin Kibana API
2. Path traversal (`../`) works — can reach APIs at different path levels
3. External URLs are blocked by default (`enableExternalUrls: false`)
4. No expression sandbox escape found in basic testing
5. Tooltip HTML properly escaped by lodash

## Test Spec Templates

### SSRF probe (same-origin)
```json
{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "width": 400, "height": 200, "padding": 5,
  "data": [{"name": "table", "url": "/api/status"}],
  "marks": [{
    "type": "text",
    "from": {"data": "table"},
    "encode": {
      "enter": {
        "x": {"value": 200}, "y": {"value": 100},
        "text": {"value": "loaded"},
        "fontSize": {"value": 14}, "align": {"value": "center"},
        "fill": {"value": "black"}
      }
    }
  }]
}
```

### Expression sandbox test
```json
{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "width": 400, "height": 200, "padding": 5,
  "signals": [{"name": "probe", "update": "if(1, 'yes', 'no')"}],
  "data": [{"name": "table", "values": [{"x": 100, "y": 100}]}],
  "marks": [{
    "type": "symbol",
    "from": {"data": "table"},
    "encode": {
      "enter": {
        "x": {"value": 100}, "y": {"value": 100},
        "size": {"value": 400}, "fill": {"value": "steelblue"},
        "tooltip": {"signal": "probe"}
      }
    }
  }]
}
```

## References
- vega-interpreter CSP mode: vega.github.io/vega/usage/interpreter/
- Vega expression functions: vega.github.io/vega/docs/expressions/
- Past Vega CVEs: CVE-2025-68385, CVE-2025-59840, CVE-2025-25017, CVE-2025-25009
