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
grep -rn "version >=\|version <\|use_extended_seed\|strict_ciphertext_length" src/
grep -rn "MPC_MIN_SUPPORTED" src/  # Is it actually referenced anywhere?
```

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

## Templates
- `templates/cosigner_test_CMakeLists.txt` — CMakeLists.txt snippet for adding PoC files to the cosigner_test target. Includes the required include paths and link dependencies.

## References

- `references/fs-hash-truncation-poc.md` — 3-phase PoC blueprint for Fiat-Shamir hash truncation bugs: size verification → FS seed collision → challenge equivalence, with Catch2 gotchas and exact SHA256/DRNG replication code
- `references/mpc-lib-source-analysis.md` — Concrete source audit findings (nonce generation code paths, DRNG patterns, protocol architecture, BAM legacy status) from Fireblocks mpc-lib
- `fireblocks-api-toolkit/references/mpc-attack-chain-synthesis.md` — Full attack chain synthesis with real-world precedents
- `fireblocks-api-toolkit/references/mpc-deep-dive-7-areas.md` — 7 unexplored attack surfaces analysis
- `fireblocks-api-toolkit/references/paillier-oracle-poc.md` — Working Paillier CCA oracle PoC
- `fireblocks-api-toolkit/references/cmp-malicious-key-poc.md` — Blum ZKP defense verification PoC
- `fireblocks-api-toolkit/references/mpc-real-world-precedents.md` — BitForge, TSSHOCK, CVE-2023-33241/33242 case studies

## Related Skills

- `fireblocks-api-toolkit` — Specific to Fireblocks MPC program, has PoC scripts and full source analysis
- `attack-chain-synthesis` — Cross-layer chain framework with Chain F for MPC layer
