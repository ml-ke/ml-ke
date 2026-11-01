# mpc-lib CRT λ Extraction PoC Reference

## The Technique
Boneh-Joux-Nguyen CCA attack on Paillier, adapted for the BAM protocol's 5-gate oracle.

## Key Insight
3072-bit Paillier Commitment key has **λ of only ~256 bits**:
- The Paillier Commitment key generation produces unbalanced prime factors
- λ = lcm(p-1, q-1) is dominated by the smaller factor (~256 bits)
- This is NOT a vulnerability — the scheme doesn't require balanced factors
- BUT it means CRT extraction is much faster: 21 probes vs 192+ theoretical

## Oracle Probe Loop
```
For each 16-bit prime p_i:
  1. Start BAM signing session with fresh txid
  2. Craft Paillier ciphertext c_i = Enc(known_value, known_randomness)
  3. Submit as encrypted_partial_sig through verify_partial_signature_and_output_signature
  4. Observe which gate (1-5) rejects the probe
  5. Gate 5 (ECDSA verify) = oracle hit = decryption succeeded
  6. Response reveals λ mod p_i
```

## CRT Reconstruction
```
congruences = [(p_1, λ mod p_1), (p_2, λ mod p_2), ..., (p_k, λ mod p_k)]
M = p_1 × p_2 × ... × p_k
For each i:
  M_i = M / p_i
  inv_i = M_i⁻¹ mod p_i
  λ ≡ Σ(a_i × M_i × inv_i) (mod M)
```

## Build Notes
- Include `../../src/common/crypto/paillier_commitment/paillier_commitment_internal.h` to access priv->lambda
- Catch2 does not support `&&` in REQUIRE — split into separate statements
- BN_CTX_start/end frames are fragile — one end frees all BIGNUMs since start
- BN_mod(BIGNUM *rem, ..., BIGNUM *m, ...) — do NOT pass same BIGNUM as rem and m
- Blum ZKP proof size for 2048-bit key: ~33KB. Allocate 65536 bytes.
