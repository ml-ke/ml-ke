---
name: mpc-crypto-audit
category: bug-bounty
description: Systematic methodology for auditing multi-party computation (MPC) cryptographic libraries — finding missing/weak zero-knowledge proofs, version-gated crypto strength, oracle patterns in verification functions, and protocol abort side-channels.
---

# MPC Cryptographic Audit Methodology

Use when auditing any MPC/threshold-signature library for cryptographic vulnerabilities. This skill generalizes findings across Fireblocks mpc-lib, GG18/GG20, Lindell17, BitGo TSS, and other threshold ECDSA implementations.

## The Core Pattern: Missing/Wrong ZK Proofs

The most dangerous MPC vulnerabilities come from **missing or weakened zero-knowledge proofs**. The BitGo case study (2023) is the definitive example: BitGo patched a missing range proof, but the Paillier well-formedness proof was still missing, allowing Fireblocks researchers to **exploit the same class again**. The two missing proofs were independent — fixing one left the other exploitable.

**The two-proof minimum rule**: Any MPC protocol using Paillier encryption needs BOTH:
1. **Range proof** — proves encrypted values are within the correct bounds (not huge)
2. **Paillier well-formedness proof** — proves the Paillier modulus n is a valid biprime (not a product of many small primes)

If only one is present, the protocol is vulnerable.

### ZKP Checklist for Every MPC Implementation

For each ZKP in the protocol, verify:

| Check | What It Proves | If Missing |
|-------|---------------|------------|
| Schnorr proof | Knowledge of secret key for public key | Anyone can claim any public key |
| Paillier Blum ZKP | n is a Blum integer (p×q, p≡3 mod 8, q≡7 mod 8) | Multi-prime modulus passes (CVE-2023-33241) |
| Range proof (MTA values) | Encrypted values are bounded (not huge) | CCA oracle via crafted ciphertexts |
| Range proof (Paillier factors) | p,q are large enough | Small factors pass, attacker factors n |
| Ring Pedersen ZKP | Prover knows λ such that s = t^λ mod n | Commitment scheme weakened |
| DH-log consistency | R = G^k matches across all participants | Rogue-key attack |
| Large factors proof | Paillier primes have no small factors | CRT factoring (6ix1een attack) |

### The "Missing Uniqueness Check" Gap

Standard trap: Paillier/Ring Pedersen public keys are validated per-party but **never compared across parties**. This means:
- Two parties can submit the SAME key (key reuse attack)
- No mechanism detects this
- If the attacker knows the factorization of the reused key, they decrypt messages intended for the other party

Search pattern: grep for `deserialize_auxiliary_keys`, `verify_setup_proofs`, `generate_setup_proofs` — check if there's any cross-party comparison of moduli.

## Nonce Generation — RFC 6979 Gap

ECDSA and EdDSA nonces in many MPC implementations are generated via OpenSSL `BN_rand_range()` rather than RFC 6979 deterministic nonces. This creates a potential key-recovery vector if the RNG state is cloned (VM snapshot, process fork, backup restore).

**Checklist for nonce audit:**
- [ ] Are nonces generated via `BN_rand_range()` or a similar CSPRNG-only function?
- [ ] Is there a deterministic nonce alternative (RFC 6979) available?
- [ ] Is the RNG seeded from a trusted entropy source?
- [ ] Are nonces generated per-signature-round (not reused across rounds)?
- [ ] For EdDSA: does it use RFC 8032's deterministic nonce or random nonces?

**Search commands:**
```bash
grep -rn "rand(" src/ | grep -i "k\.data\|nonce\|scalar"  # Nonce generation sites
grep -rn "BN_rand_range\|RAND_bytes" src/                   # RNG call sites
```

See `references/mpc-lib-source-analysis.md` for a concrete case study with exact file paths and line numbers from the Fireblocks mpc-lib codebase.

## Version-Gated Crypto Strength (Unique Danger Pattern)

Some MPC libraries (Fireblocks mpc-lib being the prime example) make cryptographic strength **version-dependent** through the protocol version field. This is uniquely dangerous because:

1. **No lower-bound version check**: `if (version > metadata.version)` only blocks upgrade, not downgrade
2. **ZKP seed computation changes with version**: Simple seed (v<11) omits Paillier/RP key binding from Fiat-Shamir hash
3. **Ciphertext validation changes**: `strict_ciphertext_length` only enforced at high versions

**Audit pattern**: Search for `MPC_EXTENDED_MTA`, `version >=`, `use_extended_seed` across the codebase. Any cryptographic decision gated on a claimable version number is a downgrade vector.

**Search commands:**
```bash
grep -rn "version >=\\|version <\\|use_extended_seed\\|strict_ciphertext_length" src/
grep -rn "MPC_MIN_SUPPORTED" src/  # Is it actually referenced anywhere?
```

### Range Proof ZKP Extended Seed Never Used

The MTA verifier (mta.cpp) correctly switches between simple/extended seed based on `_version`. But the RANGE PROOF ZKP verification functions (`range_proof_exponent_zkpok_verify`, `range_proof_diffie_hellman_zkpok_verify`) receive `use_extended_seed` as a parameter — and the signing flow hardcodes it to 0 in 5 places even at version >= 11. See "Adjacency Hunting" section below for exact locations.

## Oracle Pattern: Distinguishable Verification Gates

Any verification function that **returns different errors for different failure conditions** is a potential oracle. The classic pattern is a multi-gate verification function:

```
Gate 1: cheap pre-filter (coprime check, size check)
Gate 2: format check (deserialization, length)
Gate 3: equation check 1 (EC commitment)
Gate 4: equation check 2 (Paillier commitment)  
Gate 5: ECDSA verification (the actual crypto)
```

Each gate produces a different error → attacker learns which gate caught their crafted input → this IS the oracle.

**Audit pattern**: For any `verify_*` function, check if different failure modes produce distinguishable error messages or exception types. If yes, trace whether the distinguishment leaks information about the internal state.

**Fix**: All verification failures should produce the SAME error message ("verification failed") regardless of which check triggered the failure.

## Protocol Abort Side-Channels

In MPC protocols, aborting a session causes the protocol to fail. Each abort leaks information through:
1. **Error message content** (see Oracle pattern above)
2. **Timing variation** (different gates take different time)
3. **State retained** (partial protocol state may reveal information)

P1 severity threshold for key recovery: <1000 aborts. Always check whether the exploitation path requires fewer than 1000 protocol aborts.

## CRT Key Extraction (6ix1een Attack)

Attack from CVE-2023-33241 (Makriyannis et al., ACM CCS 2024). Works when a malicious party can submit a Paillier modulus with small factors and the protocol has no (or weakened) Blum ZKP.

**Extraction technique:**
1. Construct n = p₁ × p₂ × ... × p₁₆ × q (16 small primes + 1 large)
2. During signing, submit ciphertext cⱼ = Enc(n/pⱼ mod n²)
3. When challenge e ≡ 0 (mod pⱼ), verification equation has a special case
4. Oracle reveals information about the honest party's key
5. CRT reconstructs full key from 16 congruences

**Mitigations to verify:**
- Blum ZKP that checks n is a valid biprime (NOT multi-prime)
- Cross-party modulus uniqueness check
- Small-prime product check (gcd(n, product_of_small_primes) == 1)
- Range proof for encrypted values

## Memory Safety Patterns in Crypto C Code

MPC libraries mix C (crypto primitives) and C++ (protocol logic). Key patterns to audit:

### BN_CTX Frame Management

The `BN_CTX_start/end` pattern in OpenSSL is error-prone in C++ exception contexts:
- C functions use `goto cleanup` for error paths
- C++ wrappers use RAII `bn_ctx_frame` or `BN_CTX_guard`
- Cross-boundary exceptions (C function called from C++) can leave frames open
- Nested BN_CTX frames corrupt the BIGNUM pool

**Search commands:**
```bash
grep -rn "BN_CTX_start" src/  # Every frame needs a corresponding end
grep -rn "BN_CTX_end" src/
```

### Fiat-Shamir Hash Truncation (Unique to mpc-lib)

The `generate_mta_range_zkp_seed` function (mta.cpp:128-130) computes the Fiat-Shamir challenge seed for MTA range proofs. The hash includes proof.A (a Paillier ciphertext, serialized as `2 × |n_Paillier|` bytes) but uses `BN_num_bytes(proof.S)` as the byte count instead of `BN_num_bytes(proof.A)`. Since Paillier n is always larger than Ring-Pedersen n (2048 vs 1024 bits in the default configuration), only the first fraction of proof.A is bound by the hash:

- proof.A: 512 bytes (2048-bit Paillier, ciphertext in Z_{n²})
- proof.S: 128 bytes (1024-bit Ring-Pedersen)
- FS challenge only reads 128/512 = 25% of proof.A

**Audit pattern**: Look for any case where `SHA256_Update` or similar uses a BIGNUM's byte count (`BN_num_bytes`) from one variable to hash a different variable. This is a common copy-paste bug in Fiat-Shamir implementations.

**Proving the collision**: To demonstrate impact beyond a size mismatch, build a 3-phase PoC:
1. Size verification — confirm serialized proof.A > proof.S using actual key generation
2. FS seed collision — replicate the exact SHA256 hash and show two different proof.A values produce identical seeds
3. Challenge equivalence — feed the seed into the library's DRNG and show the derived `e` values are identical

See `references/fs-hash-truncation-poc.md` for a complete 3-phase PoC blueprint with exact SHA256 and DRNG replication code.

### Missing Key Value Binding in Extended Seed (mpc-lib-specific)

The `generate_mta_range_zkp_extended_seed` function (mta.cpp:83-113) accepts `ring_pedersen_verifier`, `public_key_prover`, and `public_key_verifier` parameters but NEVER includes their actual values in the SHA256 hash. The parameters are only used for BN padding size computation (`BN_bn2binpad` width), not for value binding.

This means the "extended" seed only provides fixed-size BN encoding (preventing length-extension malleability) but does NOT actually bind the Fiat-Shamir challenge to the specific cryptographic keys. Two setups with same-size keys produce identical FS challenges for identical proof values.

### Adjacency-Hunting: Same Pattern in Parallel Code Paths

After finding a validated vulnerability (like the MTA Fiat-Shamir seed weakness that was accepted as a duplicate), systematically search for the same class of issue in adjacent code paths. This is the highest-probability source of new findings because the same developer patterns and assumptions propagate.

**Methodology:**
1. Identify the ROOT CAUSE pattern of your validated finding (e.g., "protocol version gates cryptographic parameters")
2. Search for ALL code paths that implement the same pattern (e.g., every `verify_*` function that receives `version` or `use_extended_seed`)
3. Check each for the same mistake — is the parameter respected in ALL paths?
4. Also check the GENERATE side — the prover may also hardcode the weaker mode

**Case study from this session (Fireblocks mpc-lib):**

The MTA finding (duplicate) showed that the extended FS seed is only used when `_version >= MPC_EXTENDED_MTA` in the MTA verifier (mta.cpp:552-559, 988-995). The simple seed omits key parameters.

Adjacency check → the RANGE PROOF ZKPs (`range_proofs.c`) have their OWN seed computation (`genarate_zkpok_seed_internal`) with its OWN `use_extended_seed` parameter. Searching for `use_extended_seed` across the signing flow reveals it's hardcoded to 0 in **5 places**, even at version >= 11:

| Location | Function | Line | Hardcoded? |
|----------|----------|------|-----------|
| mta.cpp (generate) | `range_proof_paillier_exponent_zkpok_generate` | 669, 672 | `/*use_extended_seed=*/0` |
| mta.cpp (generate) | `range_proof_diffie_hellman_zkpok_generate` | 658, 661 | `/*use_extended_seed=*/0` |
| cmp_ecdsa_signing_service.cpp (verify) | `range_proof_exponent_zkpok_verify` | 176 | `/*use_extended_seed=*/0` |
| cmp_ecdsa_offline_signing_service.cpp (verify) | `range_proof_diffie_hellman_zkpok_verify` | 143 | `/*use_extended_seed=*/0` |

Contrast with `cmp_setup_service.cpp:231` where `use_extended_seed` IS properly derived from version.

**Key insight for impact assessment:** The range proof ZKP's extended mode only changes encoding fixedness (variable vs fixed-length BN encoding), NOT key binding. So this finding is the same class of issue but lower impact than the MTA case. The adjacency check is valuable even when the second finding is lower severity — it prevents overlooked vulnerabilities and tells you the full extent of the developer's pattern.

**Search commands for adjacency hunting:**
```bash
# Find all functions that receive version and make cryptographic decisions
grep -rn "version >=\\|version <\\|use_extended_seed\\|strict_ciphertext" src/

# Find all verify functions with distinguishable errors as potential oracles
grep -rn "throw.*VERIFICATION_FAILED\\|throw.*INVALID_PARAMETERS" src/ | grep -v test/

# Check if the generate-side AND verify-side agree on parameters
# If generate uses extended but verify doesn't, that's a different class of bug
grep -rn "range_proof.*generate\|range_proof.*verify" src/ | grep -v test/

### Missing Cross-Party Modulus Uniqueness Check

In setup, each party's Paillier and Ring-Pedersen public keys are deserialized per-party but never compared ACROSS parties. If two parties submit the same Paillier modulus (e.g., the same n), the protocol has no mechanism to detect this. An attacker who knows the factorization of a shared modulus can decrypt MTA messages intended for any party using that modulus.

**Search commands:**
```bash
grep -rn "deserialize_auxiliary_keys" src/  # Where keys are loaded per-party
grep -rn "players_info" src/common/cosigner/cmp_setup_service.cpp  # Check for cross-party comparison
# There should be a loop comparing all moduli — if missing, it's a finding
```

### Constant-Time vs Variable-Time Operations

Search for comments like `// WARNING: This function doesn't run in constant time` — these indicate intentional timing side-channels. Verify they're not used in security-critical verification paths.

## Honest Assessment: When to Stop Escalating

The most important skill in MPC auditing is knowing when a finding ISN'T exploitable. Not every real bug is a reportable vulnerability. This section captures the decision framework.

### The Exploitability Assessment

Before claiming a known attack technique applies, verify each precondition:

| Known Attack | Preconditions | If NOT met → |
|-------------|---------------|-------------|
| **TSSHOCK α-shuffle** (binary FS challenges) | FS challenge bits are independent (1 bit each), dlnproof iterations are low enough to brute-force | Full-scalar SHA256 challenges (256-bit) make α-shuffle infeasible regardless of dlnproof iteration count |
| **GG18/20 CCA** (missing Blum ZKP) | No Paillier Blum ZKP (or weakened). Attacker can choose N with small factors. | If `paillier_generate_paillier_blum_zkp` exists and is verified, attack is blocked |
| **Lindell17 abort** (t1 = e·β' + γ) | Implementations deviate from paper, mishandling abort states | If aborts are caught by C++ scope guards and all partial state is cleaned up, leak is blocked |
| **Paillier CRT key recovery** (6ix1een) | Malicious party can embed small factors in N without detection | Requires either missing Blum ZKP or weakened version-gated proof acceptance |
| **Fiat-Shamir key binding** | Missing key modulus values in hash allows proof replay across contexts | The algebraic verification still binds proofs to group elements — seed weakness alone doesn't enable proof forgery |
| **Nonce reuse / RNG cloning** | Deterministic nonces (RFC 6979) NOT used, VM snapshot or fork possible | If `BN_rand_range` from system CSPRNG is used (not deterministic), preconditions require physical RNG compromise |

### The Six-Question Honest Check

After finding any potential vulnerability, ask:

1. **Does the PoC output match the title?** — If the title says "Key Recovery" but the terminal shows "different errors for different inputs," the claim and evidence diverge. Fix the title, not the evidence.
2. **Is every step of the attack chain demonstrated?** — List each step. If any step shows "theorized" instead of "terminal output," that step isn't real.
3. **Have I searched for code-level defenses?** — Before claiming a protocol-level attack, search for ZKPs, scope guards, and checks that might prevent it. `grep -rn "blum\|zkp\|verify\|abort\|scope" src/common/cosigner/`
4. **What's the minimum prerequisite count?** — 0-1 prerequisites = good finding. 2+ unlikely prerequisites = academic paper, not submission.
5. **Is this a known CVE that's already fixed?** — Check the commit history for relevant patches. If the fix is already in the codebase, the finding is invalid.
6. **If the vendor fixed this tomorrow, would a security boundary that was previously crossed become uncrossable?** — If the answer is "no, no boundary was ever being crossed," the finding was never a vulnerability (Part 6 of pre-submission-verification).

### The Threshold Decision

| PoC Output | Title Should Say | Report At | Action |
|------------|-----------------|-----------|--------|
| Oracle exists (different errors) | "Distinguishable errors in ZKP verification" | P4-P5 | Report as information disclosure, NOT key recovery |
| Key material extracted (λ output) | "Key recovery via oracle" | P1-P3 | Full submission |
| Source code confirms weakness (no PoC) | "Missing security control in [component]" | P4-P5 | Submit only if demonstrable impact exists |
| Source analysis only (nothing demonstrable) | Nothing — it's research, not a finding | N/A | Save as skill reference, don't submit |

### The PoC-Claim Congruence Test

Before submitting ANY crypto finding, write the PoC and check: does the terminal output contain what I need for the submission title?

```
Bad title: "Paillier Chosen-Ciphertext Key Recovery via Distinguishable Verification Gates"
PoC output: "ec_left does not equal ec_right commitment"
→ Title claims key recovery. PoC shows gate failure. MISMATCH. Fix title.

Good title: "BAM Well-Formed Proof Verification Has Distinguishable Error Channels"
PoC output: "ec_left does not equal ec_right commitment" + "Size n mismatch 383 != 384"
→ Title matches PoC output. MATCH. Ready to write report.
```

**Rule**: The PoC's last printed line before "ALL PASS" or "DONE" defines what you can claim. Nothing more.

## PoC Build Infrastructure

When building PoCs against MPC libraries:

1. **Use the existing test framework** — it already links against all crypto libs and has setup/teardown infra
2. **Static library > shared library** — shared libs often hide internal symbols
3. **Separate BN_CTX per library call** — never share a BN_CTX between test harness and library function
4. **No chained `&&` in Catch2 REQUIRE** — Catch2 doesn't support `REQUIRE(a && b)`, use separate `REQUIRE(a); REQUIRE(b);`
5. **BN_CTX_get allocates within most recent start frame** — crossing frames frees earlier allocations

### Value-Prefix Collision Technique

When proving a hash truncation bug (where only the first N bytes of a BIGNUM are hashed), create two BNs with identical prefixes using:

```cpp
// 1. Serialize BN_1 to fixed-width buffer
std::vector<uint8_t> buf(PROOF_A_BYTES);
BN_bn2binpad(proof_A_1, buf.data(), PROOF_A_BYTES);

// 2. XOR the tail (bytes beyond the hash window)
for (int i = HASH_WINDOW; i < PROOF_A_BYTES; i++)
    buf[i] ^= 0xFF;

// 3. Re-ensure positive (MSB == 0)
buf[PROOF_A_BYTES - 1] &= 0x7F;

// 4. Deserialize as BN_2
BN_bin2bn(buf.data(), PROOF_A_BYTES, proof_A_2);
```

This guarantees `proof_A_1` and `proof_A_2` share the first `HASH_WINDOW` bytes (identical FS input) but differ mathematically (different Paillier decryption result).

### Replicating Internal Hash Functions

When the hash function is `static inline` (not accessible from test code), replicate it exactly:

1. Declare the SALT as a local constant (read from source)
2. Copy the SHA256_Init/Update/Final sequence verbatim
3. Use `std::vector<uint8_t> n(BN_num_bytes(bn))` + `BN_bn2bin(bn, n.data())` for BIGNUM serialization (matching the variable-length encoding)
4. For the fixed-size case (extended seed): use `BN_bn2binpad(bn, vec.data(), vec.size())` with `hash_bn`-style padding

### Replicating DRNG Challenge Derivation

Feed the seed into the library's DRNG, then sample `e` the same way the verifier does:

```cpp
drng_t* rng;
drng_new(seed, SHA256_DIGEST_LENGTH, &rng);
const BIGNUM* q = algebra->order_internal(algebra);
do {
    drng_read_deterministic_rand(rng, e_out, sizeof(scalar));
    BN_bin2bn(e_out, sizeof(scalar), e_bn);
} while (BN_cmp(e_bn, q) >= 0);
drng_free(rng);
```

See `templates/cosigner_test_CMakeLists.txt` for the build configuration snippet to add new PoC files.

## Current PoC State (June 2026)

All PoCs are in `test/cosigner/` and registered in CMakeLists.txt (except bam_full_extraction_poc). Build and run:

```bash
cd ~/Dev/mpc-lib/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc) cosigner_test
```

### PoC Status Table (June 2026)

| PoC | Assertions | Status | Submission Status | Notes |
|-----|-----------|--------|-------------------|-------|
| `cmp_version_downgrade_poc` | 27/27 | ALL PASS | **Duplicate** (Fiat-Shamir binding) | Confirmed research valid. Setup succeeds at v=10. FS seed at v<11 omits key binding. |
| `cmp_malicious_key_poc` | 37/37 | ALL PASS | Not submitted alone | Blum ZKP defense works. Version downgrade allows weaker proof reuse. |
| `bam_crt_extraction_poc` | 9/9 | ALL PASS | Not submitted alone | 300-500 probes needed (under P1 1000 threshold). |
| `mta_range_zkp_hash_truncation_extended` | 47/47 | ALL PASS | **Duplicate** (Fiat-Shamir binding) | Only 128/512 bytes (25%) of proof.A hashed in FS seed. |
| `bam_attack_poc` | 7/8 | 1 FAIL | **Rejected** (AI-generated) | Oracle exists but key recovery NOT demonstrated. See rejection case studies reference. |
| `bam_full_extraction_poc` | — | NOT BUILT | — | File exists but NOT registered in CMakeLists.txt. |

**Submission outcomes:**
- MTA Fiat-Shamir binding → **Duplicate** (valid research, found by someone else first)
- Paillier oracle (004) → **N/A** (AI-generated/not reproducible) — oracle confirmed but key recovery never demonstrated
- Version downgrade + aux key rotation (005) → **N/A** (AI-generated/not reproducible) — chained 3 undemonstrated findings

**Key lesson**: The submissions that failed did so because they claimed impact beyond what the PoC demonstrated. The submission that succeeded (as a duplicate) did so because the PoC precisely matched the claim. See `pre-submission-verification` skill's Part 7 (Post-Rejection Analysis) and `references/mpc-rejection-case-studies-jun2026.md`.

### Key Negative Finding (Important)

The MPC library defines `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION = 3` in `mpc_globals.h:13` but **never checks it anywhere in the source code**. This means:

- The version downgrade to v=2 for unencrypted MTA is **not possible** — encryption is always applied regardless of version
- The version downgrade's actual impact is limited to FS seed weakening (v<11 omits key binding), not MTA plaintext leakage
- Do NOT claim "unencrypted MTA at v=2" in a submission — it's not supported by the code
- `templates/cosigner_test_CMakeLists.txt` — CMakeLists.txt snippet for adding PoC files to the cosigner_test target. Includes the required include paths and link dependencies.

## References

- `references/fs-hash-truncation-poc.md` — 3-phase PoC blueprint for Fiat-Shamir hash truncation bugs: size verification → FS seed collision → challenge equivalence, with Catch2 gotchas and exact SHA256/DRNG replication code
- `references/mpc-lib-source-analysis.md` — Concrete source audit findings (nonce generation code paths, DRNG patterns, protocol architecture, BAM legacy status) from Fireblocks mpc-lib
- `references/range-proof-extended-seed-hardcode.md` — Adjacency-hunting finding: `use_extended_seed=0` hardcoded in 5 range proof ZKP locations (both generate and verify sides) even at version >= 11
- `references/mpc-real-world-precedents.md` — BitForge, TSSHOCK, CVE-2023-33241/33242 case studies
- `references/poc-assessment-june2026.md` — PoC status table and build/run instructions
- `fireblocks-api-toolkit/references/mpc-attack-chain-synthesis.md` — Full attack chain synthesis with real-world precedents
- `fireblocks-api-toolkit/references/mpc-deep-dive-7-areas.md` — 7 unexplored attack surfaces analysis
- `fireblocks-api-toolkit/references/paillier-oracle-poc.md` — Working Paillier CCA oracle PoC
- `fireblocks-api-toolkit/references/cmp-malicious-key-poc.md` — Blum ZKP defense verification PoC

## Related Skills

- `fireblocks-api-toolkit` — Specific to Fireblocks MPC program, has PoC scripts and full source analysis
- `attack-chain-synthesis` — Cross-layer chain framework with Chain F for MPC layer
