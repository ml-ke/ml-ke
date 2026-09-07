# 005 — CRT Paillier λ Extraction Technique

**Report:** `~/Dev/REPORTS/Fireblocks-MPC/005/REPORT.md`
**PoC:** `~/Dev/REPORTS/Fireblocks-MPC/005/poc/poc_005.cpp`

## Vulnerability

The CMP protocol version check (`cmp_setup_service.cpp:145-150`, `cmp_ecdsa_online_signing_service.cpp:145-149`, `cmp_ecdsa_offline_signing_service.cpp:98-104`) only prevents UPGRADES — it never validates a minimum version. `MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` is defined at `mpc_globals.h:12` but NEVER checked in any `src/` file.

At version < 11 (`MPC_EXTENDED_MTA`), the Fiat-Shamir challenge for MTA range proofs omits three critical key parameters (`mta.cpp:552-559`):
- ❌ Ring Pedersen public modulus
- ❌ Prover's Paillier modulus
- ❌ Verifier's Paillier modulus

The simple seed (`generate_mta_range_zkp_seed`, `mta.cpp:115-155`) hashes only: salt, aad, response message, commitment, and proof values A, Bx, By, E, F, S, T. The extended seed (`generate_mta_range_zkp_extended_seed`, `mta.cpp:83-113`) additionally hashes all three key parameters.

## CRT Extraction

The 3072-bit Paillier Commitment private key λ is only ~256 bits (unbalanced factors: p≈256-bit, q≈2816-bit). Recovery:

1. Generate ~26 × 16-bit primes (e.g., 32771, 32779, 32783... from 32771 to 65521)
2. For each prime pᵢ, use the oracle to determine λ mod pᵢ
3. CRT reconstructs λ from the system of congruences

CRT algorithm (Garner):
```
Given: x ≡ a₁ (mod m₁), ..., x ≡ aₖ (mod mₖ)
M = ∏mᵢ
For each i:
  Mᵢ = M / mᵢ
  invᵢ = Mᵢ⁻¹ (mod mᵢ)
  x = Σ(aᵢ × Mᵢ × invᵢ) (mod M)
```

Since ∏pᵢ > 2²⁵⁶, and λ has 256 bits, the result uniquely determines λ.

## PoC Build Pitfalls (mpc-lib-specific)

- **Catch2 `&&` restriction**: `REQUIRE(p && q)` causes static assertion. Use `REQUIRE(p); REQUIRE(q);` instead.
- **Internal headers**: Use relative path `../../src/common/crypto/paillier_commitment/paillier_commitment_internal.h` from `test/cosigner/`.
- **CRT bug**: `BN_mod(rem, a, m, ctx)` — never pass the same BIGNUM as `rem` and `m`. `BN_mod(product, recovered, product, ctx)` destroys the modulus.
- **Blum ZKP size**: For 2048-bit keys, proof is ~33KB. Allocate 64KB buffer — the size-query pattern (NULL+0) is NOT supported.
- **Test fixture extern**: Extern declarations of Catch2 TestSetup don't work across .cpp files. Append PoC code to bam_test.cpp or setup_test.cpp.

## Report Structure

```
Summary → Vulnerability Detail → PoC → P1 Threshold → Attack Chain
→ CVSS → Bugcrowd Severity Mapping → Remediation → References
```

VRT: Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step (Varies baseline).
Program rules override: key recovery with <1000 aborts = P1 ($100K-$250K).
