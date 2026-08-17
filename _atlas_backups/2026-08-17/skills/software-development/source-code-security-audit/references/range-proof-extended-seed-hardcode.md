# Adjacency-Hunting Finding: Hardcoded `use_extended_seed=0` in Range Proof ZKPs

## Finding

The range proof ZKP verification functions (`range_proof_exponent_zkpok_verify`, `range_proof_diffie_hellman_zkpok_verify`) receive a `use_extended_seed` parameter that is **hardcoded to 0** in 5 locations across the signing flow, even at protocol version >= 11.

## Root Cause

The MTA verifier (`mta.cpp:552-559`) correctly switches between simple/extended seed based on `_version`:
```cpp
if (version >= MPC_EXTENDED_MTA)
    generate_mta_range_zkp_extended_seed(response, proof, ...);
else
    generate_mta_range_zkp_seed(response, proof, ...);
```

But the range proof ZKPs (`range_proofs.c`) have their own seed computation (`genarate_zkpok_seed_internal`) with their own `use_extended_seed` parameter. The signing flow ignores version and hardcodes it to 0.

## Locations (5 total)

**GENERATE side** (mta.cpp in `create_mta_response`):
```
mta.cpp:658  — range_proof_diffie_hellman_zkpok_generate(..., /*use_extended_seed=*/0, NULL, 0, &len);
mta.cpp:661  — range_proof_diffie_hellman_zkpok_generate(..., /*use_extended_seed=*/0, proof.data(), proof.size(), &len);
mta.cpp:669  — range_proof_paillier_exponent_zkpok_generate(..., /*use_extended_seed=*/0, NULL, 0, &len);
mta.cpp:672  — range_proof_paillier_exponent_zkpok_generate(..., /*use_extended_seed=*/0, G_proof.data(), G_proof.size(), &len);
```

**VERIFY side** (signing service):
```
cmp_ecdsa_signing_service.cpp:176 — range_proof_exponent_zkpok_verify(..., strict_ciphertext_length, /*use_extended_seed=*/0);
cmp_ecdsa_offline_signing_service.cpp:143 — range_proof_diffie_hellman_zkpok_verify(..., strict_ciphertext_length, /*use_extended_seed=*/0);
```

**Contrast** with the setup service where it IS correctly computed:
```cpp
cmp_setup_service.cpp:231 — const uint8_t use_extended_seed = (temp_data.version >= MPC_EXTENDED_MTA) ? 1 : 0;
```

## Seed Difference

The `genarate_zkpok_seed_internal` function's extended mode (use_extended_seed=1) changes:
1. Ciphertext encoding: `BN_bn2binpad(ciphertext, n, 2*paillier_n_size)` — fixed width
2. Proof->S encoding: `BN_bn2binpad(proof->S, n, ring_pedersen_n_size)` — fixed width  
3. Proof->D encoding: `BN_bn2binpad(proof->D, n, 2*paillier_n_size)` — fixed width
4. Proof->T encoding: `BN_bn2binpad(proof->T, n, ring_pedersen_n_size)` — fixed width

Non-extended mode (use_extended_seed=0):
1. All BNs use `BN_bn2bin` — variable width (leading zeros stripped)

## Impact

Lower severity than the MTA finding because:
- The range proof "extended" mode only changes encoding fixedness, NOT key binding
- The algebraic verification equations still bind the proof to specific keys (modulus n is used in the verification math itself)
- The MTA extended seed additionally includes `ring_pedersen->n`, `prover_paillier_n`, `verifier_paillier_n` — actual key values

What the variable-length encoding enables: An attacker can produce two different BN representations that hash identically in the seed but differ in the actual mathematical value (by varying the number of leading zero bytes). This could enable proof malleability — the same proof being claimed for different mathematical values.

## How to Reproduce

Build the existing test suite:
```bash
cd ~/Dev/mpc-lib/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc) cosigner_test
```
Then trace the seed computation in `range_proofs.c` function `genarate_zkpok_seed_internal` to confirm the hardcoded 0 means variable-length encoding is always used.
