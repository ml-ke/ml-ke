---
name: fireblocks-api-toolkit
category: bug-bounty
description: Fireblocks API authentication, probing, and exploitation toolkit for Bugcrowd programs
---

# Fireblocks API Toolkit

Use when testing Fireblocks bug bounty programs on Bugcrowd.

## Programs
- **Fireblocks Web** (fireblocks-mbb-og): `sb-console-api.fireblocks.io`, `sb-mobile-api.fireblocks.io`, `sandbox-api.fireblocks.io`
- **Fireblocks MPC** (fireblocks-mbb-og2): C++ libcosigner MPC cryptographic protocol attacks

## Fireblocks MPC Program (libcosigner)

**Bugcrowd**: `fireblocks-mbb-og2` — started Sep 2025, ongoing. 9 vulns rewarded so far.
**Avg payout**: $9,133 last 3 months. **Scope**: only `github.com/fireblocks/mpc-lib`
**Rewards**: P1 $100K-$250K, P2 $40K-$100K, P3 $5K-$40K, P4 $300-$5K

### Severity Definitions
- **P1**: Key recovery or rogue sig with <1000 failures/aborts (any method)
- **P2**: Key recovery or rogue sig with <1B failures/aborts
- **P3**: Leaking key bits or memory corruption
- **P4**: Non-critical exposure

### Protocols Implemented (all in libcosigner/)
| Protocol | Files | Status |
|----------|-------|--------|
| CMP ECDSA (online) | `cmp_ecdsa_online_signing_service` | Current, UC-secure |
| CMP ECDSA (offline) | `cmp_ecdsa_offline_signing_service` | Preprocessing phase |
| CMP Setup | `cmp_setup_service` | Key generation |
| CMP Refresh | `cmp_offline_refresh_service` | Key rotation |
| BAM ECDSA | `bam_ecdsa_cosigner_*.cpp` | LEGACY — known vulnerable (Fireblocks blog Dec 2021 warned) |
| EdDSA online | `eddsa_online_signing_service` | Current |
| Asymmetric EdDSA | `asymmetric_eddsa_cosigner_*.cpp` | Offline variant |

### Critical Finding: Non-RFC-6979 Nonce Generation

ECDSA nonce `k` generated at `cmp_ecdsa_signing_service.cpp:30`:
```cpp
throw_cosigner_exception(algebra->rand(algebra, &data.k.data));
```
Calls OpenSSL `BN_rand_range()` at `GFp_curve_algebra.c:799`. **No RFC 6979**. Same applies to nonces `a`, `b`, `gamma`, and EdDSA nonces (`eddsa_online_signing_service.cpp:95`).

**Impact if k reused across two signatures with same key**:
```
k = (z1 - z2) / (s1 - s2)   (mod n)  → recover nonce
d = (s1*k - z1) / r          (mod n)  → recover private key
```

**Attack scenarios for k-reuse**:
1. VM snapshot cloning — OpenSSL RNG state cloned, producing identical nonces
2. Process fork — forked process inherits RNG state
3. Restore from backup — restores old RNG state

### Crypto Primitive Architecture
- Paillier encryption (MTA protocol)
- Commitments: Pedersen, Damgard-Fujisaki, Ring Pedersen
- ZK proofs: Schnorr, DH-log, range proofs (Fiat-Shamir)
- Shamir VSS, DRNG (deterministic for challenges only)
- Curves: secp256k1, secp256r1, Stark, Ed25519

### Attack Surface Map
| Surface | Type | Potential | Difficulty | Source |
|---------|------|-----------|------------|--------|
| Non-RFC-6979 nonces | Cryptographic | P1-P2 (key recovery) | High — VM clone/fork | `GFp_curve_algebra.c:799` |
| **BAM is DEFAULT protocol** | Protocol configuration | **P4-P5** (design choice — CMP is also compiled) | **Informational** — deployments should configure explicitly | `mpc_globals.h:25` |
| **BAM Paillier Oracle (PoC CONFIRMED)** | Crypto/Protocol | **P1** ($100K-$250K) via key recovery | **Medium** — need client cosigner role; <1000 aborts | `bam_well_formed_proof.cpp:372-486` |
| **Setup version downgrade (no lower-bound check)** | Protocol logic | **P3-P4** — weakens FS seed, enables forked proofs at v<11 | **Low-medium** — affects setup+signing, not just signing | `cmp_setup_service.cpp:145-150` |
| **No cross-party Paillier modulus uniqueness** | Key validation | **P3-P4** — key reuse across parties undetectable | **Low** — needs auxiliary attack to exploit | `cmp_setup_service.cpp:760` |
| **Refresh: no sender auth on encrypted seeds** | Protocol design | **P3-P4** — replay of encrypted blobs across sessions | **Medium** — needs persistency compromise first | `cmp_offline_refresh_service.cpp:72,126` |
| **No anti-replay epoch counter in refresh** | Protocol design | **P3** — key shares have no refresh-epoch binding | **Medium** — needs persistency compromise first | `cmp_offline_refresh_service.cpp:109` |
| **Variable-length BN in FS hash (legacy mode)** | Cryptographic | **Medium** — deviation from ideal fixed-length binding | **Low** — adversary can't control leading zeros freely | `range_proofs.c:131-202` |
| **Paillier key size: 2048 vs 3072** | Cryptographic param (different primitives) | **P5** (informational — meets proof bounds) | N/A | `cmp_setup_service.cpp:565` vs `bam_ecdsa_cosigner.h:34` |
| Memory safety (BN_CTX nesting) | Memory corruption | P3 | Medium — exception-triggered frame leak | `bam_ecdsa_cosigner_server.cpp:724-749` |
| Protocol abort analysis | Side-channel | P2-P3 | Very high | BAM well-formed proof |
| No SGX protections | Implementation | P3-P4 | Very high | Platform-dependent |
| **Entropy test framework exists** | Defensive (evidence of concern) | P4 (informational) | N/A | `test/crypto/entropy/` |
| **attack_helpers.h in ZKP tests** | Defensive (proof manipulation tests) | P4 (informational) | N/A | `test/crypto/zero_knowledge_proof/` |
| **CRT key extraction chain** (combined: 7B+7A+7C+5A+1A/1B) | Multi-finding chained exploit | **P1** — full key via 16 CRT sessions, refresh rollback for persistence | **Low-Medium** — need malicious cosigner role <1000 aborts | `cmp_setup_service.cpp:145-150` + `:760` + `paillier_zkp.c:1530` + `range_proofs.c:131` + `cmp_offline_refresh_service.cpp:72,126` |

**Cross-reference:** `references/mpc-analysis.md` for full source file listing, nonce generation trace, attack scenarios, and oracle PoC results.
**Cross-reference for 7 unexplored areas:** `references/mpc-deep-dive-7-areas.md` — CMP refresh, MTA batch verifier, is_coprime_fast, BAM client signing, range proofs, EdDSA asymmetric cosigner, setup key validation.

## Build Pitfall — Shared Library Symbol Visibility

The cosigner library (`libcosigner.so`) is compiled with `-fvisibility=hidden` (from `src/common/CMakeLists.txt`). Internal functions needed for attack PoCs — specifically `bam_well_formed_proof::verify_signature_proof`, `bam_well_formed_proof::generate_signature_proof`, and `paillier_commitment_encrypt_openssl_fixed_power_internal` — are NOT exported. Linking against the .so produces "undefined reference" errors.

**Fix:** Create a static library target with default symbol visibility. Add to `src/common/CMakeLists.txt`:
```cmake
add_library(cosigner_static STATIC ${COSIGNER_FILES})
set_target_properties(cosigner_static PROPERTIES
    CXX_VISIBILITY_PRESET "default"
    C_VISIBILITY_PRESET "default"
    POSITION_INDEPENDENT_CODE ON
)
# Apply same compile_options, include_directories, link_libraries as cosigner
```

Then link tests against `cosigner_static` instead of `cosigner`:
```cmake
target_link_libraries(cosigner_test PRIVATE tests_main cosigner_static Threads::Threads UUID::UUID)
```

### Build Pitfalls (Empirical, Updated Jun 2026)

1. **Catch2 REQUIRE macro: No `&&` chaining**: The Catch2 v2.13.8 framework used by mpc-lib's tests does NOT support chained comparisons in `REQUIRE()` assertions. Writing `REQUIRE(p && q && n);` causes `static assertion failed: operator&& is not supported`. Fix: split into separate `REQUIRE(p); REQUIRE(q); REQUIRE(n);` statements.

2. **BN_CTX_start/end lifecycle**: PoCs that create multiple test sections within one TEST_CASE must be careful with `BN_CTX_start/end` pairs. Each `BN_CTX_end()` frees ALL BIGNUMs allocated since the matching `BN_CTX_start()`. Subsequent sections using BIGNUM pointers from an earlier frame will crash.

3. **Blum ZKP proof size**: For 2048-bit Paillier key with 64 Blum iterations, proof size is ~33KB. Allocate at least 64KB. The size-query pattern (NULL+0) is NOT supported — `paillier_generate_paillier_blum_zkp` returns `PAILLIER_ERROR_BUFFER_TOO_SHORT` even for size queries (it checks `proof_len < needed_proof_len` before returning the needed size). Always provide a generous buffer directly.

4. **CRT extraction bug — BN_mod overwrites modulus argument**: `BN_mod(BIGNUM *rem, const BIGNUM *a, const BIGNUM *m, BN_CTX *ctx)` — never use the same BIGNUM for `rem` and `m`. `BN_mod(product, recovered, product, ctx)` destroys the modulus. Always use a separate remainder BIGNUM.

5. **Internal header access**: To access λ or struct members of `paillier_commitment_private_key_t`, use the relative include path `../../src/common/crypto/paillier_commitment/paillier_commitment_internal.h` from `test/cosigner/`. For public-only access, use `paillier_commitment_private_cast_to_public()`, `paillier_commitment_public_bitsize()`, and `paillier_commitment_public_key_serialize()`.

6. **Paillier Commitment encrypt function is static**: `paillier_commitment_encrypt_openssl_internal` is a `static` function in `paillier_commitment.c` and NOT exposed in any header. You cannot call it from PoC code. Use `paillier_commitment_encrypt()` (public API, takes byte arrays) or work directly with BIGNUM operations for non-canonical encoding demonstrations.

7. **PoC integration**: Append code to `bam_test.cpp` or `setup_test.cpp` (they link against `cosigner_static` with default visibility). Create new .cpp files in `test/cosigner/` and add to `test/cosigner/CMakeLists.txt`. Don't modify library-level CMake. After adding a file, run `cmake .. && make cosigner_test`.

8. **BN_mod overwrites modulus — use separate remainder**: `BN_mod(result, value, modulus, ctx)` uses `result` as the output and `modulus` as the divisor. If you pass the same BIGNUM for both, the modulus is overwritten before the operation completes. Always use a distinct BIGNUM for the remainder.

9. **Catch2 reports "all tests passed" even when test section fails**: If a REQUIRE assertion inside a SECTION fails, Catch2 reports that specific SECTION as failed but continues to the next SECTION. The final summary shows `All tests passed (X assertions in 1 test case)` even when some assertions failed in earlier sections. Always check for `FAILED` in the output — don't rely on the final summary alone.

## BAM Paillier Oracle Attack Chain — PRIMARY P1 PATH

**Target:** BAM protocol (the DEFAULT — MPC_PROTOCOL_VERSION = 13)

**Setup:** Malicious cosigner client in BAM 2-of-2 signing. BAM uses a custom well-formed proof (`bam_well_formed_proof.cpp`) to verify the client's encrypted partial signature before the server decrypts it.

**The oracle:** The `verify_signature_proof` function at `bam_well_formed_proof.cpp:372-486` has FIVE distinguishable verification gates, each producing different error codes when a crafted ciphertext is submitted:

```
Gate 1: is_coprime_fast(encrypted_sig, n)         → coprime check failure
Gate 2: deserialize proof (size/range checks)      → size mismatch error  
Gate 3: EC commitment equation (g^z1·f^z2 == V·U^e) → ZKP equation failure
Gate 4: Paillier commitment equation (D·S^e)       → different ZKP failure
Gate 5: Signature verification                     → ORACLE HIT — server decrypted!
```

Each forged encrypted_partial_sig submission that passes Gates 1-4 but fails Gate 5 (signature verification) means the server SUCCESSFULLY DECRYPTED our value. This is a Paillier **chosen-ciphertext validity oracle**.

**Key recovery (P1, <1000 aborts):**
1. Submit ~300-500 crafted Paillier ciphertexts as encrypted_partial_sig
2. Each submission causes one protocol abort (<1000 total = P1 threshold)
3. The oracle responses reveal the Paillier private key (optimized CCA attack)
4. With Paillier key, decrypt the server's `encrypted_server_share` from key metadata
5. Server share + client share = full ECDSA key
6. Submit as: Cryptographic key recovery via Paillier CCA oracle in BAM protocol

**PoC confirmed (see references/paillier-oracle-poc.md):**
- encrypted_sig = 768 bytes, proof = 1302 bytes
- corrupted proof → "Size of n mismatch" (LEAKS Paillier modulus size: 3072 bits)
- corrupted ciphertext → "ec_left does not equal ec_right commitment" (ZKP EC check)
- The Paillier modulus size is extractable from the proof format header

**Pitfall — proof gen/verify function mismatch:**
When building a direct crypto-layer PoC, note that `generate_signature_proof` uses `paillier_commitment_commit_internal(pub, ...)` while `verify_signature_proof` uses `paillier_commitment_commit_with_private_internal(priv, ...)`. These are NOT equivalent — the verify function computes the commitment differently than the generate function. The `encrypted_share` and `S` values must be carefully aligned. See `references/paillier-oracle-poc.md` "Known Issue" section for details.

**Alternative chain (P4, network attacker):**
1. MITM between cosigners, force protocol version below MPC_EXTENDED_MTA (11)
2. No lower-bound check on version — the check at `cmp_ecdsa_online_signing_service.cpp:145` only blocks upgrade
3. At v < 11, the Fiat-Shamir challenge seed for MTA range ZKPs is SIMPLER: it does NOT include the other party's Paillier key or Ring Pedersen parameters
4. This weakens proof binding — the same proof could be valid across different key contexts
5. Submit as: Insufficient version validation weakens Fiat-Shamir binding in MTA range proofs

**CORRECTION (Jun 2026):** The original analysis claimed MTA responses were sent unencrypted at v2. This is FALSE. `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION = 3` is defined but **never checked** in any source file. MTA responses are ALWAYS Paillier-encrypted. Always `grep -rn CONSTANT src/` to verify a constant is used before claiming its behavior.

**Cross-reference:** `references/mpc-analysis.md` for full verification gate analysis.
`references/paillier-oracle-poc.md` for PoC source, test output, and reproduction steps.

## Seven Unexplored Areas — Deep Dive Findings (Jun 2026)

Systematic analysis of 7 attack surfaces not covered by previous work. Results at `references/mpc-deep-dive-7-areas.md`.

### CORRECTED Understanding (PoC-Proven)

The initial hypothesis about a multi-prime CRT attack (CVE-2023-33241 technique) was **tested and disproven** for mpc-lib's CMP protocol. The Blum ZKP correctly blocks unbalanced/multi-prime moduli via the `gcd(λ,n)=1` condition. PoC confirmed: 256+1792-bit unbalanced key produces `gcd(λ,n)≠1`, invalidating the key for Paillier operations.

### Real Primary Path (PoC-Proven, 121 assertions across 4 PoCs)

**Path A: Version downgrade → weaker FS seed → faster CRT key recovery (P1, ~26 probes)**

The version downgrade (`cmp_setup_service.cpp:145-150`) only blocks version > metadata.version — no lower-bound check exists. `MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` is defined but NEVER referenced in any source file (`grep -rn MPC_MIN_SUPPORTED_PROTOCOL_VERSION src/` returns zero). At v<11 (`MPC_EXTENDED_MTA`), `mta.cpp:552-559` uses the simple Fiat-Shamir seed omitting Paillier/RP key binding.

CRT extraction: The 3072-bit Paillier Commitment λ is only ~256 bits. Use ~26 × 16-bit primes, recover λ mod each via oracle, CRT reconstructs. After λ: decrypt `encrypted_server_share` → `server_share + client_share = full ECDSA key`.

**Submission Strategy — differentiate from any prior BAM oracle submission:**
- Different protocol (CMP vs BAM)
- Different root cause (missing version check vs distinguishable oracle gates)
- Different code path (`cmp_setup_service.cpp:145-150` + `mta.cpp:552-559` vs `bam_well_formed_proof.cpp:372-486`)
- VRT: Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step

```
Finding 7B (no min version)   →  PoC: cmp_version_downgrade_poc (27/27 pass)
  ↓
Finding 5A (var-length FS)    →  PoC: cmp_malicious_key_poc (37/37 pass)
  ↓  
CRT λ extraction               →  PoC: bam_full_extraction_poc (48/48 pass)
  ↓
λ decrypts server_share → full ECDSA key
```

### Submission Strategy: 004 vs 005

These are **two separate P1 submissions** — different protocols, different code paths, different root causes:

| Dimension | 004 (submitted) | 005 |
|-----------|-----------------|-----|
| Protocol | BAM (well-formed proof) | CMP (MTA range proofs) |
| Vuln class | Oracle gates | Missing version check |
| Root cause | `bam_well_formed_proof.cpp:372-486` | `cmp_setup_service.cpp:145-150` |
| Probes | ~300-500 | ~26 (CRT) |
| PoC | `bam_paillier_oracle_poc.cpp` | `poc_005.cpp` |

Frame 005 VRT: Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step.

CRT extraction: The 3072-bit Paillier Commitment λ is only ~256 bits. Use ~26 × 16-bit primes, recover λ mod each via oracle, CRT reconstructs. After λ: decrypt `encrypted_server_share` → `server_share + client_share = full ECDSA key`.
  ↓
CRT λ extraction loop          →  PoC: bam_full_extraction_poc (48/48 pass)
  → λ MATCHES ORIGINAL: YES ✓
  → 21 probes for 256-bit λ (3072-bit Paillier Commitment key has only 256-bit λ!)
  → P1 threshold (1000) MET
```

**Key finding about Paillier Commitment key structure:** The 3072-bit Paillier Commitment key in BAM has λ of only ~256 bits (not ~1536 as with balanced primes). This means the CRT extraction needs only **21 probes** with 16-bit primes, dramatically under P1's 1000-abort threshold.

**No lower-bound version check** (`cmp_setup_service.cpp:145-150`, confirmed by PoC):
- Only checks `version > metadata.version` (blocks upgrade)
- `MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` is defined but **never referenced** in any source file
- Version 10 accepted silently; at v<11, `use_extended_seed=0` and `strict_ciphertext_length=0`
- MTA range proofs use simple FS seed omitting Paillier/RP key binding

**Blum ZKP Defense (tested and confirmed working):**

### Permanent Compromise Architecture (New Finding, Jun 2026)

**Two independent failures → combined permanent compromise:**

1. **Refresh doesn't rotate Paillier/RP keys** (`cmp_offline_refresh_service.cpp:152-164`): The refresh service ONLY updates ECDSA key shares. `auxiliary_keys` (Paillier encryption + Ring Pedersen) are loaded at line 223-224 but never regenerated. The PoC at `bam_full_extraction_poc.cpp` (48/48 pass) confirms λ is recoverable from the BAM Paillier Commitment key — and that compromise is **permanent**.

2. **No BAM refresh exists**: grep for `bam.*refresh|refresh.*bam` across `src/` returns zero results. BAM Paillier Commitment keys are generated once during `bam_key_generation` and are fixed forever.

**Combined consequence**: If an attacker recovers λ via the BAM oracle (004) or any other means:
- ECDSA key rotation does NOT rotate Paillier keys
- The version downgrade (v=10) persists in metadata
- All past and future signing sessions are decryptable
- Standard 30-day refresh cycles provide zero benefit
- The only remediation is generating an entirely new wallet

**Report framing for this combined finding**: Frame as two architectural failures → permanent compromise → key rotation is security theater. VRT: Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step. Severity: P3 (two reinforcing failures producing rotation-surviving compromise).

### Build & Run All PoCs
```bash
cd ~/Dev/mpc-lib && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc) cosigner_test
# Individual PoCs:
./test/cosigner/cosigner_test "bam_full_extraction" -s
./test/cosigner/cosigner_test "cmp_version_downgrade_poc" -s
./test/cosigner/cosigner_test "cmp_malicious_key_poc" -s
# All at once:
./test/cosigner/cosigner_test -s
# All 4 new PoCs (121 assertions):
./test/cosigner/cosigner_test "cmp_malicious_key_poc,cmp_version_downgrade_poc,bam_crt_extraction,bam_full_extraction" -s
```

### Reference files
- `references/mpc-deep-dive-7-areas.md` — 7 unexplored attack surfaces with file:line references (refresh, batch verifier, is_coprime_fast, client signing, range proofs, EdDSA cosigner, setup key validation)
- `references/mpc-attack-chain-synthesis.md` — Worst-case chains, real-world precedents (BitForge, CVE-2023-33241, TSSHOCK), complementary finding mappings, submission options
- `references/mpc-crt-extraction-poc.md` — CRT extraction technique: Boneh-Joux-Nguyen CCA on Paillier, 300-probe loop, λ reconstruction mathematics
- `references/mpc-analysis.md` — Full source file listing, nonce generation trace, attack scenarios, original oracle PoC results
- `references/mpc-005-extraction.md` — 005 report: version downgrade + CRT λ extraction, CRT solver bugfix, BUILD PITFALLS updated

## Sandbox Access
- Signup: https://www.fireblocks.com/developer-sandbox-sign-up
- Email: @bugcrowdninja.com
- API key + RSA private key pair from Fireblocks Console

## Authentication (CRITICAL — Python JWT FAILS)
Headers:
- `X-API-Key: {api_key}`
- `Authorization: Bearer {JWT}`

JWT format (RS256):
```json
{ "uri": "/v1/vault/accounts", "nonce": "uuid", "iat": 1234567890, "exp": 1234567945, "sub": "{api_key}", "bodyHash": "sha256_hex" }
```

- `uri`: pathname + search from the full URL (`new URL(url).pathname + new URL(url).search`)
- `nonce`: random UUID v4
- `iat`/`exp`: 55-second validity window
- `bodyHash`: SHA256 of empty string for GET, SHA256 of `JSON.stringify(body)` for POST/PUT/PATCH
- Algorithm: RS256

**CRITICAL:** Python's `PyJWT` / `jwt.encode()` produces RS256 signatures that the Fireblocks server rejects with code -4. The exact same payload signed with Node.js `jsonwebtoken` works. **Always use Node.js.**

## POST Request — Critical Fixes

### 1. Content-Length Header is Mandatory
Node.js `https.request()` does NOT auto-set `Content-Length` when the body is written via `req.write(str)`. POST requests fail with code -7 ("Error getting User certificate") if Content-Length is missing:
```javascript
'Content-Length': Buffer.byteLength(JSON.stringify(body))
```

### 2. bodyHash Must Be JSON.stringify'd
The bodyHash for POST/PUT/PATCH must be SHA256 of the **JSON-stringified body**, not the raw JS object:
```javascript
// CORRECT:
const bodyHash = crypto.createHash('sha256').update(JSON.stringify(body)).digest().toString('hex');
// WRONG (what the broken TS SDK does):
const bodyHash = crypto.createHash('sha256').update(body).digest().toString('hex'); // crashes or hashes '[object Object]'
```

## bodyHash Bug in @fireblocks/ts-sdk (ALL PUBLISHED VERSIONS)

ALL published npm versions (v8.0.1 through v20.0.0 — 64 versions confirmed) have `crypto.createHash("sha256").update(bodyJson || "").digest()` at `network/bearerTokenProvider.ts:47`. Since `bodyJson` is a raw JS object (from Axios `config.data`), `bodyJson || ""` returns the object — not a string.

- **Node.js v22+**: Throws `TypeError` — SDK crashes on every POST
- **Node.js ≤18**: Silently hashes `[object Object]` — ALL POST bodies have the SAME bodyHash

**Every other SDK does it correctly:** Python `json.dumps()`, Java `getBytes()`, Go `json.Marshal()`, old JS SDK `JSON.stringify()`. Only the new TS SDK is broken.

**Bug confirmed in BOTH source and compiled JS:**
- Source: `network/bearerTokenProvider.ts:47`
- Compiled: `dist/network/bearerTokenProvider.js:69`

**Version scope verified by inspecting npm tarballs:** v8.0.1, v10.0.0, v14.0.0, v19.1.0, v20.0.0 all return the same `bodyHash: crypto.createHash("sha256").update(bodyJson || "").digest().toString("hex")`.

**Cross-reference:** `references/sdk-comparison.md` for full comparison table.

## POST 401 Error (code -7) — Two Causes

Code -7 "Error getting User certificate" can mean two different things:

### Cause 1: Rate Limiting / Sandbox Throttling
Rapid requests (10+ in 30s) cause code -7 on ALL endpoints (GET + POST). The key recovers after 10-30s of rest.

**Diagnosis:** If GET ALSO starts failing with -7 after a POST burst → rate limiting. Wait and retry.

### Cause 2: Missing Content-Length (most common for POST)
Node.js does not auto-set Content-Length. Without it, the server returns -7. See "POST Request — Critical Fixes" above.

### Cause 3: Incorrect bodyHash
If bodyHash doesn't match `SHA256(JSON.stringify(body))`, the server rejects with -7. See "bodyHash Bug" above.

### Error Code Reference
| Code | Meaning |
|------|---------|
| -3 | JWT missing (no Authorization header) |
| -4 | Token signed for wrong URL (uri mismatch) |
| -7 | Rate limited OR missing Content-Length OR wrong bodyHash |
| -13 | Nonce already used — replay prevented |
| -15 | Endpoint doesn't exist in this API version |
| 1037 | Endpoint deprecated, use paged version |
| 11001 | Invalid vault account ID |
| 2702 | Cannot get users (insufficient permissions) |

## Report Writing Patterns (mpc-lib Bugcrowd Submissions)

When writing reports for the Fireblocks MPC program:

### Format
- **Separate report text from PoC**: The REPORT.md goes into Bugcrowd's description form field. The PoC files get zipped as `poc.zip` and attached separately.
- **No internal numbering in report text**: Don't reference "004", "005", "Finding X", or session-specific identifiers. Use descriptive cross-references like "the BAM Paillier oracle (submitted separately)".
- **File paths**: Use full `src/` prefix for all source file references (e.g., `src/common/cosigner/cmp_setup_service.cpp:145-150` not `cmp_setup_service.cpp:145-150`).
- **VRT verification**: Before finalizing any report, browse the live Bugcrowd VRT at https://bugcrowd.com/vulnerability-rating-taxonomy to verify the category exists. Search by vulnerability name and variant. VRT categories from memory are often outdated or non-existent.
- **Non-claim of categories that don't exist**: "Insufficient Verification of Proofs" does NOT exist in VRT v1.18. Always verify before writing.

### Attacker-First Framing
Frame findings from the attacker's perspective, not as theoretical crypto analysis. Each finding should answer "what does the attacker get?" as a concrete asset. For example:
- Good: "An attacker can downgrade the protocol version to bypass FS key binding. At v<11, 393-byte ciphertexts pass where 768-byte is canonical."
- Avoid: "The Fiat-Shamir transform has variable-length BN serialization leading to weakened proof binding which could theoretically enable..."

### Required Report Sections
```
Summary → Vulnerability Detail → Proof of Concept → P1 Threshold Compliance
→ Attack Chain → CVSS v3.1 Assessment → Bugcrowd Severity Mapping
→ Remediation → References
```

The PoC section should reference the attached `poc.zip` archive, not a file system path. Include build commands that work from the extracted archive.

### VRT Category Guide for MPC Findings

| Finding Type | VRT Category | Baseline |
|-------------|-------------|----------|
| BAM Paillier oracle (distinguishable gates) | Cryptographic Weakness → Side-Channel Attack → Padding Oracle Attack | P4 |
| CMP version downgrade (missing lower-bound check) | Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step | Varies |
| Non-RFC-6979 nonces | Cryptographic Weakness → Insecure Implementation → Improper Following of Specification | Varies |

**Key rule**: The program's own severity table overrides the VRT baseline. Key recovery with <1000 aborts = P1 regardless of VRT category baseline. Always include a "Severity Note" explaining this and a "Bugcrowd Severity Mapping" table.

### Fact-Checking Checklist
Before submitting, verify:
1. Every `file.cpp:line` reference is checked against live source code (not memory)
2. Every VRT category is verified on the live VRT page via browser
3. Every `constexpr` constant claimed as "never checked" is verified with `grep -rn CONSTANT_NAME src/`
4. Claims about different cryptographic primitives verify the actual function call, not just the constant name
5. The report clearly differentiates from any prior submission (different protocol, different code path, different root cause)

### Common Mappings for Fireblocks Findings

| Finding Type | VRT Category | Baseline Severity |
|-------------|-------------|-------------------|
| JWT bodyHash integrity bug | Cryptographic Weakness → Broken Cryptography → Use of Broken Cryptographic Primitive | P3 |
| Vault ID enumeration (sequential IDs) | Broken Access Control → IDOR → Modify/View Sensitive Info (Iterable Object Identifiers) | P3-P4 |
| SSRF (if confirmed live) | Server Security Misconfiguration → SSRF | P2-P3 |
| Hardcoded credentials | Insecure OS/Firmware → Hardcoded Password | P2 |

### Honest Severity Assessment

**Do NOT inflate severity.** Bugcrowd programs use the VRT as a baseline — they will downgrade overclaimed findings. The user will challenge inflated claims. Specific patterns that get corrected:

- **"Authentication Bypass" → P1** — Only if you can bypass auth entirely. If the server still validates your JWT signature and nonce, it's not an auth bypass.
- **"SSRF" → P2** — Only if you can demonstrate a live request reaching an internal service or metadata endpoint. DNS resolution + URL validation bypass without a confirmed connection is P4 at best.
- **"Information Disclosure" → varies** — Only if the leaked data enables a second unauthorized action. Standing alone, it's P5.

When the user asks "Are you sure?" — stop, verify from scratch, present honest results including negative findings. Do not rationalize or defend incorrect severity claims.

## Report Writing Patterns (mpc-lib Bugcrowd Submissions)

When writing reports for the Fireblocks MPC program:

### Format
- **Separate report text from PoC**: The REPORT.md goes into Bugcrowd's description form field. The PoC files get zipped as `poc.zip` and attached separately.
- **No internal numbering in report text**: Don't reference "004", "005", "Finding X", or session-specific identifiers. Use descriptive cross-references like "the BAM Paillier oracle (submitted separately)".
- **File paths**: Use full `src/` prefix for all source file references (e.g., `src/common/cosigner/cmp_setup_service.cpp:145-150` not `cmp_setup_service.cpp:145-150`).
- **VRT verification**: Before finalizing any report, browse the live Bugcrowd VRT at https://bugcrowd.com/vulnerability-rating-taxonomy to verify the category exists. Search by vulnerability name and variant. VRT categories from memory are often outdated or non-existent.
- **Non-claim of categories that don't exist**: "Insufficient Verification of Proofs" does NOT exist in VRT v1.18. Always verify before writing.

### Attacker-First Framing
Frame findings from the attacker's perspective, not as theoretical crypto analysis. Each finding should answer "what does the attacker get?" as a concrete asset. For example:
- Good: "An attacker can downgrade the protocol version to bypass FS key binding. At v<11, 393-byte ciphertexts pass where 768-byte is canonical."
- Avoid: "The Fiat-Shamir transform has variable-length BN serialization leading to weakened proof binding which could theoretically enable..."

### Required Report Sections
```
Summary → Vulnerability Detail → Proof of Concept → P1 Threshold Compliance
→ Attack Chain → CVSS v3.1 Assessment → Bugcrowd Severity Mapping
→ Remediation → References
```

The PoC section should reference the attached `poc.zip` archive, not a file system path. Include build commands that work from the extracted archive.

### VRT Category Guide for MPC Findings

| Finding Type | VRT Category | Baseline |
|-------------|-------------|----------|
| BAM Paillier oracle (distinguishable gates) | Cryptographic Weakness → Side-Channel Attack → Padding Oracle Attack | P4 |
| CMP version downgrade (missing lower-bound check) | Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step | Varies |
| Non-RFC-6979 nonces | Cryptographic Weakness → Insecure Implementation → Improper Following of Specification | Varies |

**Key rule**: The program's own severity table overrides the VRT baseline. Key recovery with <1000 aborts = P1 regardless of VRT category baseline. Always include a "Severity Note" explaining this and a "Bugcrowd Severity Mapping" table.

### Fact-Checking Checklist
Before submitting, verify:
1. Every `file.cpp:line` reference is checked against live source code (not memory)
2. Every VRT category is verified on the live VRT page via browser
3. Every `constexpr` constant claimed as "never checked" is verified with `grep -rn CONSTANT_NAME src/`
4. Claims about different cryptographic primitives verify the actual function call, not just the constant name
5. The report clearly differentiates from any prior submission (different protocol, different code path, different root cause)

These findings were drafted but NOT submitted because fact-checking revealed errors:

### Finding 2 (Draft) — Protocol Version Downgrade → Unencrypted MTA ❌ FACTUAL ERROR

The constant `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION = 3` is defined in `mpc_globals.h` but **NEVER checked in any source file**. The `answer_mta_request` function at `mta.cpp:697` ALWAYS encrypts MTA responses using Paillier homomorphic operations regardless of version. The version parameter only affects Fiat-Shamir seed computation (`version >= MPC_EXTENDED_MTA = 11`). The claim that "version 2 sends unencrypted MTA responses" is WRONG.

**Lesson:** A `constexpr` in a header is NOT evidence of an implemented feature. Always `grep -rn CONSTANT_NAME src/` to verify it's actually used at runtime.

### Finding 3 (Draft) — Paillier Key Size Gap ❌ INVALID COMPARISON

The "2048-bit vs 3072-bit" comparison was between **different cryptographic primitives**:
- CMP uses STD Paillier (`paillier_generate_key_pair` at `cmp_setup_service.cpp:565`) — 2048 bits
- BAM uses Paillier Commitment (`paillier_commitment_generate_private_key`) — 3072 bits

STD Paillier and Paillier Commitment are different schemes with different key size requirements. The CMP security theorem (N > q^8 = 2048) is met by the parameter choice.

**Lesson:** Before comparing key sizes, verify you're comparing the same primitive. Check the actual function call, not just the constant name.

## Nonce Tracking Observations

The server tracks JWT nonces and rejects replays with:
```json
{ "message": "This nonce was already used in a previous request.", "code": -13 }
```

This means:
- JWT replay is blocked even with different idempotency keys
- The `Idempotency-Key` not being in the JWT is NOT a vulnerability — the nonce provides the actual replay protection
- Finding 2 from the vulnerability audit was corrected: this is defense-in-depth, not an auth bypass

## Key Endpoints (sandbox-api.fireblocks.io)

**GET (read) — all confirmed working:**
- `/v1/vault/accounts_paged` — List vaults with assets
- `/v1/vault/accounts/{id}` — Get specific vault (IDs are sequential integers 0..N)
- `/v1/transactions` — List transactions
- `/v1/supported_assets` — All supported coins/tokens (returns thousands)
- `/v1/internal_wallets` — Internal wallets
- `/v1/gas_station` — Gas station config
- `/v1/staking/chains` — Supported staking chains
- `/v1/cosigners` — Cosigners
- `/v1/contracts` — Contracts
- `/v1/exchange_accounts`, `/v1/fiat_accounts`, `/v1/network_connections` — Related resources
- `/v1/webhooks` — Webhooks (GET returns empty data array)

**POST (write) — confirmed working with correct bodyHash + Content-Length:**
- `POST /v1/vault/accounts` — Create vault (body: `{name: "..."}`)
- `POST /v1/transactions` — Create transaction (body: `{assetId, source, destination, amount}`)
- `POST /v1/internal_wallets` — Create internal wallet (body: `{name, assetId, address}`)

**403 — Blocked:**
- `POST /v1/webhooks` — All URLs return 403 "Forbidden resource" (URL validation blocks everything in sandbox)

**404 — Not available in sandbox (returns code -15):**
- `/v1/audit_logs`, `/v1/workspace/status`, `/v1/smart_transfers`, `/v1/policy`, `/v1/nfts`
- `/v1/keys`, `/v1/embedded_wallets/wallets`, `/v1/reset_device`, `/v1/off_exchanges`
- `/v1/estimated_fee`, `/v1/payouts`

## Observations from Live Testing

- **Vault IDs are sequential integers** (0, 1, 2, 3, 4...) — enumerable by iterating IDs until 400 response
- **No rate limiting observed** on GET requests during enumeration
- **Console/Mobile APIs** (`sb-console-api.fireblocks.io`, `sb-mobile-api.fireblocks.io`) run separate Express apps with different routing — `Cannot GET /api/v1/*`
- **OpenAPI spec** not exposed at standard paths (`/openapi.json`, `/swagger.json`)
- **Sandbox rate limiting** triggers after ~10 rapid POST requests — key recovers after 10-30s
- **Workspace persistence**: Vault accounts, wallets persist across reconnections (IDs continue incrementing)

## Source Code (70 repos at github.com/orgs/fireblocks/repositories)

Key repos:
- `ts-sdk` — TypeScript (has bodyHash bug in published v19.1.0)
- `fireblocks-sdk-js` — Older JS SDK (correct bodyHash)
- `py-sdk` — Python SDK (correct, v19.1.0)
- `java-sdk` — Java SDK (correct)
- `developers-hub` — Multi-language code examples (Go, PHP, Ruby, Rust, Java, C#)
- `fireblocks-mcp` — MCP server
- `x402-agent`, `x402-facilitator` — HTTP 402 payment protocol

## Attack Chain: bodyHash Bug

**⚠ IMPORTANT UPDATE:** The full chain (capture JWT → replay with different body) was tested and blocked by server-side nonce tracking. The server returns code -13 ("This nonce was already used") on JWT replay regardless of idempotency key. **The bodyHash bug alone is NOT exploitable for replay attacks.** It causes denial of write operations (SDK crashes or sends wrong bodyHash), not authentication bypass.

When the broken TS SDK is used:

1. **Observe**: User makes a POST → JWT has bodyHash = SHA256("[object Object]") regardless of actual body (Node ≤18) or throws TypeError (Node 22+)
2. **Impact**: Write operations fail. The SDK is non-functional for POST/PUT/PATCH.
3. **Escalation**: None within the API itself — the server correctly rejects mismatched bodyHash and tracks nonces.

### Vault Enumeration (Confirmed)
Vault IDs being sequential (0, 1, 2...) means an attacker can map the entire workspace structure by iterating IDs. No rate limiting observed on GET enumeration. Impact: workspace info disclosure (vault names, asset holdings).

**Cross-reference:** `references/vulnerability-audit.md` for full audit and non-issues.

## Scripts
- `scripts/probe.js` — Full API probe. Reads creds from `/dev/shm/fb-apikey.txt` and `fireblocks_secret.key`. Tests GET + POST endpoints.
- `scripts/cmp_malicious_key_poc.cpp` — 37-assertion PoC proving Blum ZKP acceptance, timing variation, version downgrade analysis. Add to `test/cosigner/CMakeLists.txt` and run.
- `scripts/cmp_version_downgrade_poc.cpp` — 27-assertion PoC proving CMP setup succeeds at v=10 with no lower-bound check. Same build steps.

## Templates
- `templates/cosigner_test_CMakeLists.txt` — CMakeLists.txt snippet for adding PoC files to the cosigner_test target.

## References
- `references/vulnerability-audit.md` — 10 SDK audit findings with file paths and CVSS
- `references/sdk-comparison.md` — bodyHash handling across all 9 SDKs/languages with test vectors
- `references/mpc-deep-dive-7-areas.md` — 7 unexplored attack surfaces
- `references/mpc-worst-case-chain.md` — Attack chain synthesis, real-world precedents
- `references/mpc-crt-extraction-poc.md` — CRT extraction PoC details
- `references/mpc-005-extraction.md` — 005 report: version downgrade + CRT λ extraction technique
- `references/mpc-permanent-compromise.md` — Two-failure architecture: version downgrade + missing aux key rotation producing rotation-surviving permanent compromise
