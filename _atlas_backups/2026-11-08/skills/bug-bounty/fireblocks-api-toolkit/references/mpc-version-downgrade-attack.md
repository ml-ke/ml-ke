# CMP Version Downgrade + CRT λ Extraction Attack

**Source**: `github.com/fireblocks/mpc-lib`
**Program**: `fireblocks-mbb-og2`

## Vulnerability

The CMP protocol version check (`src/common/cosigner/cmp_setup_service.cpp:145-150`,
`src/common/cosigner/cmp_ecdsa_online_signing_service.cpp:145-149`,
`src/common/cosigner/cmp_ecdsa_offline_signing_service.cpp:98-104`)
only prevents VERSION UPGRADES — it never validates a minimum version.

`MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` (`include/cosigner/mpc_globals.h:12`) is defined but
NEVER referenced in any source file. A `grep -rn MPC_MIN_SUPPORTED_PROTOCOL_VERSION src/`
returns only the definition.

## What Version < 11 Changes

At v<11 (`MPC_EXTENDED_MTA`), `src/common/cosigner/mta.cpp:552-559` changes the Fiat-Shamir
seed computation for MTA range proofs:

- **Extended seed** (v>=11, `mta.cpp:83-113`): hashes ring_pedersen->n, prover_paillier_n,
  and verifier_paillier_n in addition to all proof values
- **Simple seed** (v<11, `mta.cpp:115-155`): omits ALL three key parameters from the hash

Also at v<11, `strict_ciphertext_length = 0` (`mta.cpp:988`), allowing non-canonical
ciphertext encodings.

## CRT λ Extraction

The 3072-bit Paillier Commitment key (BAM protocol) has λ ≈ 256 bits (not ~1536 as with
balanced primes — one factor is much smaller than the other).

### Algorithm
1. Generate ~26 × 16-bit primes (range: 32771 to 65521, skipping non-primes)
2. For each prime pᵢ, use the oracle to determine λ mod pᵢ
3. CRT reconstructs λ from congruences

### CRT Solver (OpenSSL BIGNUM)
```cpp
BIGNUM* result = BN_new();
BIGNUM* M = BN_new(), *M_orig = BN_new();
BN_one(M);
for (auto& c : congruences) BN_mul(M, M, c.first, ctx);
BN_copy(M_orig, M);  // preserve M for mod operations
for (auto& c : congruences) {
    BIGNUM* Mi = BN_new(), *inv = BN_new(), *term = BN_new();
    BN_div(Mi, nullptr, M_orig, c.first, ctx);  // Mi = M / mi
    BN_mod_inverse(inv, Mi, c.first, ctx);       // inv = Mi⁻¹ mod mi
    BN_mul(term, c.second, Mi, ctx);
    BN_mod_mul(term, term, inv, M_orig, ctx);
    BN_mod_add(result, result, term, M_orig, ctx);
    BN_free(Mi); BN_free(inv); BN_free(term);
}
```

### Key: BN_mod doesn't overwrite modulus
CRITICAL: `BN_mod(rem, a, m, ctx)` — use a SEPARATE BIGNUM for `rem` and `m`.
`BN_mod(product, recovered, product, ctx)` destroys the product value.

## PoC Verification

Run the full extraction:
```bash
cd ~/Dev/mpc-lib && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc) cosigner_test

# Version downgrade proof (27 assertions)
./test/cosigner/cosigner_test "cmp_version_downgrade_poc" -s

# Full λ extraction (48 assertions)
./test/cosigner/cosigner_test "bam_full_extraction" -s

# Blum ZKP + timing analysis (37 assertions)
./test/cosigner/cosigner_test "cmp_malicious_key_poc" -s
```

Expected: All assertions pass. λ match: YES ✓, ~26 probes.

## P1 Threshold

All variants under 1000:
- v<11 (downgrade, weakened FS): ~17 probes
- v>=11 (full binding): ~26 probes
- 1-bit modulus: 256 probes
