# Fiat-Shamir Hash Truncation — PoC Technique

## The Pattern

Many MPC implementations compute the Fiat-Shamir challenge `e = H(transcript)` using a hash of proof values. A common bug is passing the WRONG byte count to `SHA256_Update` — hashing one proof value using another proof value's `BN_num_bytes()`.

This happens because proof values are serialized with different sizes (Paillier ciphertexts in Z_{n²} are 2×|n|, while Ring-Pedersen parameters are |n|). Copy-paste errors in the hash computation use `BN_num_bytes(proof.X)` to hash `proof.Y`, truncating the hash window.

## How to Prove It (3-Phase PoC)

### Phase 1: Size Verification
Generate real keys (or mock with correct sizes) and show:
- `serialized_size(proof.A) > serialized_size(proof.S)` 
- `BN_num_bytes(proof.A) > BN_num_bytes(proof.S)` after deserialization
- The ratio: `BN_num_bytes(proof.S) / BN_num_bytes(proof.A)` is the fraction actually hashed

### Phase 2: FS Seed Collision
Replicate the EXACT SHA256 hash structure from the buggy function. For the mpc-lib example (`generate_mta_range_zkp_seed`, mta.cpp:115-154):

```
SHA256_Init
SHA256_Update(MTA_ZKP_SALT)
SHA256_Update(aad)
SHA256_Update(message)
SHA256_Update(commitment)

// BUG: n allocated with BN_num_bytes(proof.A), hashed with BN_num_bytes(proof.S)
std::vector<uint8_t> n(BN_num_bytes(proof_A));
BN_bn2bin(proof_A, n.data());
SHA256_Update(&ctx, n.data(), BN_num_bytes(proof_S));  // <-- WRONG SIZE

SHA256_Update(proof_Bx, 33)

// Rest use their OWN BN_num_bytes correctly
n.resize(BN_num_bytes(proof_By));
BN_bn2bin(proof_By, n.data());
SHA256_Update(&ctx, n.data(), BN_num_bytes(proof_By));
// ... same for E, F, S, T

SHA256_Final(seed)
```

**Create two proof.A values** with the same first `|proof.S|` bytes but different tail bytes:
```cpp
std::vector<uint8_t> buf(PROOF_A_BYTES);
BN_bn2binpad(proof_A_1, buf.data(), PROOF_A_BYTES);
for (int i = PROOF_EFST_BYTES; i < PROOF_A_BYTES; i++)
    buf[i] ^= 0xFF;
BN_bin2bn(buf.data(), PROOF_A_BYTES, proof_A_2);
```

**Assert**: BUGGY seed is identical, CORRECT seed differs.

### Phase 3: Challenge Equivalence
The FS seed feeds a DRNG, which generates `e` by sampling `< order(curve)`:

```cpp
drng_t* rng;
drng_new(seed, SHA256_DIGEST_LENGTH, &rng);
do {
    drng_read_deterministic_rand(rng, e_out, sizeof(scalar));
    BN_bin2bn(e_out, sizeof(scalar), e_bn);
} while (BN_cmp(e_bn, q) >= 0);
```

**Assert**: `e_buggy_1 == e_buggy_2` (identical challenge), `e_correct_1 != e_correct_2`.

## Key Headers for mpc-lib PoCs

When writing PoCs that need to replicate internal hash computations:

| Purpose | Header | Path |
|---------|--------|------|
| SHA256 | `openssl/sha.h` | system |
| BIGNUM | `openssl/bn.h` | system |
| DRNG | `crypto/drng/drng.h` | `<project>/include/` |
| EC Algebra | `crypto/elliptic_curve_algebra/elliptic_curve256_algebra.h` | `<project>/include/` |
| Catch2 | `tests/catch.hpp` | `<project>/include/tests/` |
| Internal Paillier | `paillier_internal.h` | NOT accessible from test (in `src/`) |
| Internal Ring-Pedersen | `ring_pedersen_internal.h` | NOT accessible from test (in `src/`) |

**Workaround**: Use public key generation + size constants instead of internal struct access. The key sizes are:
- `PAILLIER_KEY_SIZE = sizeof(elliptic_curve256_scalar_t) * 8 * 8 = 2048` bits
- `RING_PEDERSEN_KEY_SIZE = sizeof(elliptic_curve256_scalar_t) * 8 * 4 = 1024` bits

## Catch2 Gotchas

- `REQUIRE(a && b)` — **DO NOT USE**. Catch2's `&&` operator triggers a static_assert failure. Use separate `REQUIRE(a); REQUIRE(b);`
- `BN_CTX_get` returns pointers, not bools — check with `REQUIRE(ptr != nullptr)` not `REQUIRE(ptr)`
- Section-scoped variables are re-created per SECTION. Use `BN_CTX_get` from the test-case-scoped `BN_CTX*` (allocated before any SECTION) to avoid double-free.

## Drill for Future Audits

When auditing any MPC implementation's Fiat-Shamir:

1. **Find the seed/hash function** — grep for `SHA256_Update`, `SHA256_Init`, or domain separation strings
2. **Check each BN's byte count** — verify that `SHA256_Update(ptr, BN_num_bytes(X))` actually uses `X`'s byte count, not a different variable's
3. **Check for dead key parameters** — if a function takes key parameters but never hashes them, the extended seed doesn't bind to keys
4. **Build a 3-phase PoC** — size verification → seed collision → challenge equivalence
