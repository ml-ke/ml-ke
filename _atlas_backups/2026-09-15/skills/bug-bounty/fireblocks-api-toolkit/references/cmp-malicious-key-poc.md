# CMP Malicious Key PoC — Build & Results

## File Location
`scripts/cmp_malicious_key_poc.cpp` (copy to `test/cosigner/cmp_malicious_key_poc.cpp`)

## Build
```bash
# 1. Add to test/cosigner/CMakeLists.txt under cosigner_test sources:
#    cmp_malicious_key_poc.cpp

# 2. Build and run
cd ~/Dev/mpc-lib/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc) cosigner_test
./test/cosigner/cosigner_test "cmp_malicious_key_poc" -s
```

## Results (confirmed Jun 2026)

| Test | What It Proves | Result |
|------|---------------|--------|
| Balanced 2048-bit Blum ZKP | Blum ZKP gen + verify works | ✅ PASS |
| Unbalanced 256+1792-bit gcd check | gcd(λ,n)≠1 → Blum ZKP blocks it | ✅ Defence confirmed |
| is_coprime_fast timing | Variable iteration count | ✅ Timing confirmed |

## Key Lessons

1. **Blum ZKP buffer size**: ~33KB for 2048-bit key (64 iterations). NOT size-queryable via NULL+0 — the function writes `*proof_real_len` BEFORE checking `proof_len < needed_proof_len` at paillier_zkp.c:986-993, but still returns error 9 if buffer is too small. Use 65536 as initial estimate.

2. **gcd(λ,n) defense**: The `paillier_generate_key_pair` function (paillier.c:157-159) enforces `gcd(lambda, n) == 1`. When factors are unbalanced (256+1792), λ naturally shares factors with n, making the key invalid for Paillier operations. This is a BUILT-IN defense against small-factor attacks, not a ZKP-specific check.

3. **BN_CTX frames**: PoC lives at BN_CTX_start/free boundaries. Nesting BN_CTX frames across function calls corrupts the BIGNUM pool. Each sub-test needs its own BN_CTX.

4. **Real attack path**: NOT the CRT multi-prime attack (blocked by Blum ZKP + gcd check). Instead: version downgrade (7B) → weaker FS seed → faster Paillier oracle (004) key recovery.

## Error Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 9 | PAILLIER_ERROR_BUFFER_TOO_SHORT | Proof buffer too small |
| 6 | PAILLIER_ERROR_INVALID_KEY | Key structure invalid |
