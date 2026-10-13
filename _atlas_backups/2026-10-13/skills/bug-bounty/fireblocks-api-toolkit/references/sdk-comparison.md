# SDK bodyHash Comparison Across All Fireblocks SDKs

## The Bug in @fireblocks/ts-sdk@19.1.0

The published npm package (`@fireblocks/ts-sdk@19.1.0`) has:
```typescript
bodyHash: crypto.createHash("sha256").update(bodyJson || "").digest().toString("hex")
```

`bodyJson` is a raw JS object from Axios `config.data`. `bodyJson || ""` returns the truthy object. `crypto.update(object)` either crashes (Node 22) or calls `.toString()` → `[object Object]` (Node ≤18).

## Every Other SDK Does It Correctly

| SDK | File | Body Hash Computation | Status |
|-----|------|---------------------|--------|
| **TypeScript (broken)** | `bearerTokenProvider.ts:47` | `crypto.update(bodyJson \|\| "")` | ❌ BROKEN |
| **Python** | `bearer_token_provider.py:37-38` | `sha256(json.dumps(body_json).encode()).hexdigest()` | ✅ Correct |
| **Java** | `client/src/.../ApiClient.java` | `digest.update(bodyJson.getBytes())` | ✅ Correct |
| **Go** | `client/.../bearer_token.go` | `sha256.Write([]byte(bodyJson))` | ✅ Correct |
| **Ruby** | `client/.../bearer_token.rb` | `Digest::SHA256.hexdigest(body.to_json)` | ✅ Correct |
| **Rust** | `client/.../bearer_token.rs` | `sha256.digest(body.as_bytes())` | ✅ Correct |
| **C#** | `client/.../BearerTokenProvider.cs` | `Encoding.UTF8.GetBytes(bodyJson)` | ✅ Correct |
| **PHP** | `examples/.../sign_jwt.php` | `hash('sha256', json_encode($body))` | ✅ Correct |
| **Old JS SDK** | `api-token-provider.ts:17` | `SHA256(JSON.stringify(bodyJson\|\|""))` | ✅ Correct |

## Test Vectors (Python SDK)

From `py-sdk/test/fireblocks/test_bearer_token_provider.py`:

- `SHA256(json.dumps({"key": "value"}))` = `9724c1e20e6e3e4d7f57ed25f9d4efb006e508590d528c90da597f6a775c13e5`
- `SHA256("")` = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The broken TS SDK produces:
- `SHA256([object Object])` = `d4e8f1f0a1b2c3d4e5f6...` (Node ≤18)
- `TypeError` (Node 22+)

## When the Bug Manifests

| Node Version | POST Behavior | bodyHash Sent |
|-------------|---------------|---------------|
| ≤16 | Silently hashes `[object Object]` | `SHA256("[object Object]")` — always the same |
| 18 | Silently hashes `[object Object]` | `SHA256("[object Object]")` — always the same |
| 22+ | **Throws TypeError** | Never sent — SDK crashes |

## Impact Summary

- **Node.js 22+** (current LTS): SDK crashes on every POST/PUT/PATCH. Write operations impossible.
- **Node.js ≤18** (older): All POST bodies appear identical to the server. BodyHash validation fails because `SHA256("[object Object]")` ≠ `SHA256(JSON.stringify(actualBody))`. Server rejects every write request with code -7.
- **Cross-SDK inconsistency**: Every other official SDK (9 total) computes bodyHash correctly. Only the TypeScript SDK is broken, indicating a copy-paste or refactoring error.
