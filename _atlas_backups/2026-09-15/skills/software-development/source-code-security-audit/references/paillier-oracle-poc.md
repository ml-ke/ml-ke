# BAM Paillier Oracle PoC — Reproduction & Results

**Repo:** `github.com/fireblocks/mpc-lib` (cloned to `~/Dev/mpc-lib/`)
**PoC:** appended to `test/cosigner/bam_test.cpp` as `TEST_CASE("bam_oracle_poc")` during session
**Also see:** `bug-bounty/fireblocks-api-toolkit/references/paillier-oracle-poc.md` for full details

## Quick Reproduction

```bash
cd ~/Dev/mpc-lib/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc) cosigner_test
./test/cosigner/cosigner_test "bam_attack_poc" -s
```

**Build pitfall:** The shared lib uses `-fvisibility=hidden`. See "Symbol Visibility — CRITICAL PoC Pitfall" in the parent skill for the static library workaround.

## Key Confirmed Results

| Observation | Result |
|-------------|--------|
| Paillier modulus leaked from proof header | 384 bytes = 3072 bits |
| Corrupted proof vs corrupted ciphertext | DISTINGUISHABLE errors |
| G1 (coprime) oracle works | CT=0 and CT=n correctly rejected |
| CCA oracle exists | 5 verification gates, each with distinct error |
| PoC buildable | Static library mirror workaround confirmed |

## Attack Chain

1. Extract modulus size from proof → know 3072-bit Paillier
2. Submit ~300-500 crafted ciphertexts with valid proofs
3. Oracle responses reveal Paillier private key λ
4. Decrypt `encrypted_server_share` → full ECDSA key → P1 ($100K-$250K)
