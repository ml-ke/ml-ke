# MPC PoC Assessment — June 2, 2026

All PoCs are in `~/Dev/mpc-lib/test/cosigner/` and registered in CMakeLists.txt (except bam_full_extraction_poc).

## Build

```bash
cd ~/Dev/mpc-lib/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc) cosigner_test
```

## Run All PoCs

```bash
cd ~/Dev/mpc-lib/build

./test/cosigner/cosigner_test "cmp_version_downgrade_poc" -s
# → 27/27 PASS

./test/cosigner/cosigner_test "cmp_malicious_key_poc" -s
# → 37/37 PASS

./test/cosigner/cosigner_test "bam_crt_extraction" -s
# → 9/9 PASS

./test/cosigner/cosigner_test "mta_range_zkp_hash_truncation_extended" -s
# → 47/47 PASS

./test/cosigner/cosigner_test "[bam_attack]" -s
# → 7/8 PASS (1 expected failure: coprime check blocks non-coprime ciphertext)
```

Total: 120/121 assertions pass across 5 PoCs.

## Results per Finding

### Finding 1: Version Downgrade (CMP)
- **PoC**: `cmp_version_downgrade_poc.cpp` (79 lines)
- **Status**: ALL PASS (27/27)
- **What it proves**: CMP setup succeeds at v=10 (no lower-bound check enforces MPC_MIN_SUPPORTED_PROTOCOL_VERSION=2). At v<11, the simple FS seed is used instead of the extended seed, omitting ring_pedersen n, prover Paillier n, and verifier Paillier n from the Fiat-Shamir hash.
- **What it does NOT prove**: The constant `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION=3` is defined in `mpc_globals.h:13` but **never referenced in source code**. Encryption is always applied regardless of version. Do NOT claim "unencrypted MTA at v=2."
- **Submittability**: P3 standalone (weak FS binding at v<11). Escalation to P1 requires chaining with oracle.

### Finding 2: Paillier CCA Oracle (BAM)
- **PoC**: `bam_attack_poc.cpp` (182 lines)
- **Status**: 7/8 PASS, 1 FAIL
- **The 1 failure**: Line 110 — `generate_signature_proof()` throws "Internal error" when the ciphertext is not coprime with n. This is the CORRECT behavior of the coprime check (Gate 1). The test needs a refined ciphertext that passes Gate 1 but triggers Gate 4.
- **What it proves**: Three distinguishable error channels exist:
  - Gate 2 (proof deserialize): LEAKS Paillier modulus size (384 bytes = 3072 bits)
  - Gate 4 (EC commitment equation): Confirms ciphertext reached the crypto layer
  - Gate 1 (coprime check): Blocks non-coprime values correctly
- **Submittability**: P2-P3. Oracle exists but full key recovery PoC needs refinement.

### Finding 3: CRT Key Extraction (BAM)
- **PoC**: `bam_crt_extraction_poc.cpp` (119 lines)
- **Status**: ALL PASS (9/9)
- **What it proves**: 300-500 probes needed for key recovery (under P1's 1000 threshold, confirmed at lines 118-119).
- **Critical caveat**: The Blum ZKP (`paillier_zkp.c:690`, check `gcd(λ,n)=1`) blocks multi-prime moduli. The CRT attack (CVE-2023-33241) requires a multi-prime modulus. The PoC assumes a bypass that needs verification against the actual Blum ZKP implementation.
- **Submittability**: Conditional P1 if Blum ZKP bypass exists. Otherwise P3 (probe count valid but attack path blocked).

### Finding 4: Fiat-Shamir Hash Truncation (MTA Range Proof)
- **PoC**: `mta_range_zkp_hash_truncation_extended` (test in mta_test or similar)
- **Status**: ALL PASS (47/47)
- **What it proves**: Only 128/512 bytes (25%) of proof.A are included in the FS seed hash. Two different proof.A values with identical first 128 bytes produce identical FS challenges.
- **Submittability**: P3 standalone. Chained with the Paillier oracle, reduces key recovery probes. The strongest standalone crypto finding.

### Finding 5: Malicious Key + Blum ZKP Defense (CMP)
- **PoC**: `cmp_malicious_key_poc.cpp`
- **Status**: ALL PASS (37/37)
- **What it proves**: The Blum ZKP defense correctly verifies `gcd(λ,n)=1`, blocking multi-prime moduli in CMP. Combined with version downgrade proof of concept.
- **Submittability**: Complementary finding. No standalone value.

### Finding 6: Full Extraction (BAM)
- **PoC**: `bam_full_extraction_poc.cpp` — file exists but NOT registered in CMakeLists.txt
- **Status**: NOT BUILT — needs CMakeLists.txt registration

## Key Negative Findings (Important for Submission Accuracy)

1. **`MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION` is dead code.** Defined in `mpc_globals.h:13` but never checked anywhere in the source. Version downgrade to v=2 does NOT disable MTA encryption.

2. **`MPC_MIN_SUPPORTED_PROTOCOL_VERSION=2` is never enforced.** The version check at `cmp_ecdsa_online_signing_service.cpp:145-149` only blocks version > metadata.version (upgrade), not version < minimum (downgrade). Setup succeeds at any version.

3. **The bam_attack_poc's 1 failure is NOT a vulnerability.** The coprime check at Gate 1 correctly rejects ciphertexts that share a factor with n. This is the library working as designed.
