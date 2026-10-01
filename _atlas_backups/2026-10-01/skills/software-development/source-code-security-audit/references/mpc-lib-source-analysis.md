# mpc-lib Source Analysis — June 2026

Concrete findings from source code audit of `github.com/fireblocks/mpc-lib` (commit cloned June 2026).

## 1. Nonce Generation — No RFC 6979

All ECDSA and EdDSA nonces are generated via OpenSSL `BN_rand_range()`, which depends on the CSPRNG state. There is no deterministic nonce (RFC 6979) implementation.

### Code Paths

| Algorithm | File | Line | Code |
|-----------|------|------|------|
| CMP ECDSA offline signing | `src/common/cosigner/cmp_ecdsa_signing_service.cpp` | 30 | `algebra->rand(algebra, &data.k.data)` |
| CMP ECDSA offline signing | same file | 31-33 | `data.a`, `data.b`, `data.gamma` via same `rand()` |
| CMP ECDSA setup service | `src/common/cosigner/cmp_setup_service.cpp` | 524 | `algebra->rand(algebra, &temp_data.k.data)` |
| BAM ECDSA server signing | `src/common/cosigner/bam_ecdsa_cosigner_server.cpp` | 659 | `algebra->rand(algebra, &persistant_sig_data.k.data)` |
| BAM ECDSA client signing | `src/common/cosigner/bam_ecdsa_cosigner_client.cpp` | 637 | `algebra->rand(algebra, &k.data)` |
| EdDSA online signing | `src/common/cosigner/eddsa_online_signing_service.cpp` | 95 | `_ed25519->rand(_ed25519.get(), &k.data)` |
| Asymmetric EdDSA server | `src/common/cosigner/asymmetric_eddsa_cosigner_server.cpp` | 154 | `_ctx->rand(_ctx.get(), &k.data)` |
| Asymmetric EdDSA client | `src/common/cosigner/asymmetric_eddsa_cosigner_client.cpp` | 75 | `_ctx->rand(_ctx.get(), &k.data)` |
| BAM well-formed proof | `src/common/cosigner/bam_well_formed_proof.cpp` | 287 | `algebra->rand(algebra, &gamma.data)` |
| BAM key generation | `src/common/cosigner/bam_ecdsa_cosigner.cpp` | 286 | `algebra->rand(algebra, &private_share.data)` |

### The BN_rand_range Implementation

File: `src/common/crypto/GFp_curve_algebra/GFp_curve_algebra.c:788-808`

```c
elliptic_curve_algebra_status GFp_curve_algebra_rand(GFp_curve_algebra_ctx_t *ctx, elliptic_curve256_scalar_t *res)
{
    BIGNUM *tmp = BN_new();
    // ... null checks ...
    if (!BN_rand_range(tmp, EC_GROUP_get0_order(ctx->curve)))
        goto cleanup;
    ret = BN_bn2binpad(tmp, *res, sizeof(elliptic_curve256_scalar_t));
    // ... cleanup ...
}
```

The `rand` function pointer is set to `ec_rand` which delegates to `GFp_curve_algebra_rand` for all three curve types: secp256k1, secp256r1, and Stark (lines 1349, 1379, 1409).

### Attack Scenarios for Nonce Reuse

In ECDSA: if `k` is reused for two different messages `z1`, `z2`:
```
k = (z1 - z2) / (s1 - s2)  (mod n)
d = (s1*k - z1) / r         (mod n)
```

Viable failure scenarios:
- **VM snapshot clone**: Cloud VM cloned → RNG state identical → same k
- **Process fork**: Forked cosigner shares RNG state
- **Restore from backup**: RNG state restored to previous point
- **Entropy exhaustion**: Low entropy on boot → biased/predictable nonces

## 2. DRNG Module — Correctly Labeled

File: `include/crypto/drng/drng.h:22-23`

```c
/* This module implements deterministic pseudo random number generator,
   it should be used only for sampling deterministic randomness
   For true randomness you should use openssl RAND_bytes function
   or sgx_read_rand if used inside SGX */
```

The DRNG is used exclusively for Fiat-Shamir challenges in ZK proofs (schnorr.c, diffie_hellman_log.c, range_proofs.c, damgard_fujisaki_zkp.c, pedersen.c, ring_pedersen.c) — this is the correct usage pattern. The comment clearly separates deterministic from true randomness.

## 3. RNG Usage Patterns Across the Library

Three distinct RNG sources are used:

| Source | Used For | Security Level |
|--------|----------|----------------|
| `BN_rand_range()` | ECDSA nonces, Paillier randomness, ZK blindings, Shamir coeffs | CSPRNG (OpenSSL) |
| `RAND_bytes()` | Commitments (salt, nonce), permutations, ring_pedersen gamma | CSPRNG (OpenSSL) |
| `drng_read_deterministic_rand()` | Fiat-Shamir challenges only | Deterministic (seeded) |

The key observation: `BN_rand_range()` calls BN_rand_range which uses the default OpenSSL RNG method (AES-256-CTR-DRBG or similar, seeded from /dev/urandom). This is NOT RFC 6979 — it's pure CSPRNG without the deterministic safeguard.

## 4. Protocol Architecture

```
libcosigner/
├── CMP ECDSA (UC-secure, Canetti et al. 2020)
│   ├── cmp_setup_service         → Key generation
│   ├── cmp_ecdsa_offline_signing → Preprocessing
│   ├── cmp_ecdsa_online_signing  → Online signature
│   ├── cmp_offline_refresh_service → Key refresh
│   └── cmp_key_persistency       → Key storage
├── BAM ECDSA (Legacy — known vulnerabilities)
│   ├── bam_ecdsa_cosigner_client/server
│   ├── bam_well_formed_proof
│   └── bam_key_persistency
├── EdDSA Ed25519
│   ├── eddsa_online_signing_service
│   └── asymmetric_eddsa_cosigner_client/server
└── Crypto Primitives
    ├── paillier/ (encryption, ZKP)
    ├── commitments/ (Pedersen, DF, Ring Pedersen)
    ├── zero_knowledge_proof/ (Schnorr, DH-log, range proofs)
    ├── GFp_curve_algebra/ (secp256k1, secp256r1, Stark)
    ├── ed25519_algebra/
    ├── shamir_secret_sharing/
    └── drng/
```

## 5. BAM Legacy Protocol

Fireblocks blogged (Dec 2021) about vulnerabilities discovered in their "legacy MPC algorithm" and urged migration to MPC-CMP. The BAM code (`bam_ecdsa_*`) is still present in the repository alongside CMP. BAM is the older protocol with known issues.

Source: https://www.fireblocks.com/blog/vulnerabilities-discovered-and-patched-in-legacy-mpc-algorithm-fireblocks-urges-move-to-mpc-cmp (published 2021-12-14, updated 2026-05-11)

## 6. Security Audit History

`SECURITY.md` states: "The code has been audited by NCC."

## 7. Known External Reports

Reddit post (early 2026) titled "Critical Vulnerabilities in Fireblocks MPC Custody Solution: Nonce Reuse, Side-Channel, and Protocol Abort Issues" — specifically cited:
- ECDSA implementation not using RFC 6979 for deterministic nonce generation
- Reliance on OpenSSL's CSPRNG without additional safeguards
