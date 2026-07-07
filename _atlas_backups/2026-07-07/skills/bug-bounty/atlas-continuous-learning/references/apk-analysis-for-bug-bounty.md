# APK Analysis for Bug Bounty

When a target has a mobile app, the APK is frequently the most valuable recon source. App developers embed API endpoints, auth credentials, client IDs/secrets, and internal infrastructure addresses that never appear in the web frontend.

## Workflow

### Phase 1: Extraction

```bash
# 1. Extract the APK (it's just a ZIP)
unzip -o target.apk -d extracted/

# 2. Check structure
ls extracted/
# Key files:
#   classes.dex, classes2.dex, classes3.dex  — compiled DEX bytecode (primary source)
#   AndroidManifest.xml                       — permissions, package name, intent filters
#   resources.arsc                            — compiled string resources
#   assets/                                   — bundled JSON, fonts, config files
#   lib/                                      — native .so libraries (harder to analyze)
#   res/                                      — compiled XML layouts, drawables
```

### Phase 2: URL and Endpoint Discovery

```bash
# Extract ALL URLs from DEX files
strings classes.dex classes2.dex classes3.dex 2>/dev/null | \
  grep -E 'https?://[a-z0-9.-]+' | \
  sort -u

# Filter to target-related domains, exclude CDNs/analytics
strings classes.dex classes2.dex classes3.dex 2>/dev/null | \
  grep -E 'https?://[a-z0-9.-]+' | \
  grep -vE 'android\.com|google\.com|github\.com|facebook\.com|twitter\.com|amazonaws\.com|gstatic\.com|firebase|schema\.org|okhttp' | \
  sort -u
```

### Phase 3: Auth Model Discovery

Search for class names that follow API model naming patterns:

```bash
# Find model classes (JSON serialization targets)
strings classes.dex classes2.dex classes3.dex 2>/dev/null | \
  grep -i 'Login\|Auth\|Token\|Credential\|Session\|Register' | \
  grep -v 'google\|firebase\|facebook\|play-services' | \
  sort -u

# Find endpoint path patterns
strings classes.dex classes2.dex classes3.dex 2>/dev/null | \
  grep -oP '/[a-z]+/[a-z]+/[a-zA-Z0-9_/.-]*' | \
  grep -viE 'google|android|java|kotlin|firebase|play|gms|androidx|com|org|io|net|app|lib|util' | \
  sort -u

# Find API-related class names (often named Gateway*, Api*, Service*, Client*)
strings classes.dex classes2.dex classes3.dex 2>/dev/null | \
  grep -i 'Gateway\|ApiClient\|RestClient\|HttpService' | \
  sort -u
```

### Phase 4: Credential Extraction

**Plaintext strings**: Some apps embed keys directly in the DEX:
```bash
# Search for common key patterns
strings classes.dex classes2.dex classes3.dex 2>/dev/null | \
  grep -iE 'api.?[_-]?key|api_key|client.?id|client.?secret|app.?id|app.?secret|bearer|password' | \
  grep -v 'google\|firebase\|facebook\|play-services\|moengage\|okhttp' | \
  sort -u
```

**Obfuscated values**: When keys don't appear as plaintext:
- Check `assets/` JSON/config files
- Check `res/values/` strings.xml (or obfuscated resource files)
- Check `lib/*/lib*.so` native libraries with `strings`
- The values may be constructed at runtime from string fragments (harder to extract without full decompilation)

### Phase 5: Live Infrastructure Testing

For every discovered endpoint, probe it systematically:

```bash
# 1. Test the base URL
curl -sv https://discovered-api.target.com/

# 2. If it returns 404/405, try common API paths
for path in /v1 /v2 /api /health /login /auth /users /config; do
  curl -s -o /dev/null -w "%{http_code}" https://discovered-api.target.com$path
done

# 3. If you found auth model classes, try the matching endpoint
# The DEX often reveals the expected request body format via error messages
curl -sv -X POST https://gateway.target.com/v1/authapi/user/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'

# 4. Read error responses — they reveal the expected structure
# e.g. 422 → {"errors":["clientSecret must not be null","clientId must not be null","provider must not be null"]}
# This reveals the auth model has clientId, clientSecret, provider fields
```

### Phase 6: Format Iteration

When an endpoint responds with 422 (Unprocessable Entity) instead of 404:
- The endpoint EXISTS and is the correct one
- The request body format is wrong
- The error message tells you the EXACT fields expected
- Iterate by trying: JSON vs form-encoded, different field names, wrapped vs flat objects

```bash
# Try multiple body formats
curl -sv -X POST https://gateway.target.com/v1/authapi/user/login \
  -H "Content-Type: application/json" \
  -d '{"clientId":"x","clientSecret":"y","provider":"z","username":"u","password":"p"}'

# Try with Android User-Agent (some endpoints check for it)
curl -sv -X POST ... -H "User-Agent: NutakuAndroid/3.0"
```

## Real-World Example: Nutaku Android App

From the APK at `~/Dev/nutaku/nutakuclientlatest.apk`:

**Discovered endpoints** (not visible in web frontend):
- `https://gateway-api.nutaku.net/v1/authapi/user/login` — OAuth2 login
- `https://gateway-api.nutaku.net/v1/authapi/user/social-login` — Social login
- `https://metaapi.nutaku.net` — Metadata API (403)
- `https://vendor-gateway-api.nutaku.net/` — Vendor gateway (404)
- `https://sbox-osapi.nutaku.com/social_android/rest/` — Sandbox OpenSocial
- `https://sbox-mobileapi.nutaku.com/` — Sandbox mobile API
- `https://stage-api.gateway.nutaku.net/v1/` — Staging gateway
- `https://stage-metaapi.nutaku.net` — Staging metadata API
- `https://cdn-updater.nutaku.net/android/live/` — App update CDN

**Auth model** (from 422 error response):
```json
{"errors":["clientSecret must not be null","clientId must not be null","provider must not be null"]}
```

**App package**: `com.project.nutaku`
**Native libs**: `libnutaku.so` (minimal — just C++ runtime)
**Build**: Gradle 8.13, Kotlin 2.1.0

## Key Insight

The error response is often the most valuable output. A 422 (Unprocessable Entity) tells you:
1. The endpoint EXISTS (correct path)
2. The endpoint is PROCESSING your request (correct method)
3. The error lists the EXACT required fields

This is information leakage — but usable. Iterate on the format until you get a 400 (still wrong format) or 401 (correct format, wrong credentials) or 200 (success).
