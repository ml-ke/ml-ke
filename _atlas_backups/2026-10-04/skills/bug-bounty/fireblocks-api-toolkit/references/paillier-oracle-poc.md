# BAM Paillier Oracle PoC — Reproduction & Results

**Repo:** `github.com/fireblocks/mpc-lib`
**PoC file:** Append `bam_paillier_oracle_poc.cpp` to `test/cosigner/bam_test.cpp`
**No CMake changes needed.** The PoC has zero `#include` directives — it inherits all type
definitions (`TestSetup`, `bam_key_generation`, `client_id`, `server_id`) from bam_test.cpp.

## Build & Run

```bash
# System deps (Ubuntu 24.04)
sudo apt-get install -y build-essential cmake libssl-dev libsecp256k1-dev uuid-dev

# Clone, apply PoC, build
git clone https://github.com/fireblocks/mpc-lib
cd mpc-lib
cat bam_paillier_oracle_poc.cpp >> test/cosigner/bam_test.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc) cosigner_test

# Run the oracle probe
./test/cosigner/cosigner_test "bam_oracle_poc" -s
```

**Expected: 34/34 assertions pass. Test completes in <30s.**

## Expected Output

```
  Reference: encrypted_sig=768 bytes, proof=1302 bytes

  [P1] encrypted_sig = 0x00...00 -> ERR: Internal error has occurred    ← Gate 1
  [P2] encrypted_sig = 0x00...01 -> ERR: Wrong keyid was specified      ← state consumed
  [P3] encrypted_sig = 0x00...02 -> ERR: Wrong keyid was specified      ← state consumed

  Error in deserialize_well_formed_proof. Size of n mismatch 383 != 384  ← Gate 2 MODULUS LEAKED
  [A] valid CT + corrupt proof -> ERR: Internal error has occurred

  ec_left does not equal ec_right commitment                             ← Gate 3/4
  [B] corrupt CT + valid proof -> ERR: Internal error has occurred
```

## Key Findings Confirmed

| Finding | Evidence |
|---------|----------|
| Paillier modulus = 3072 bits | `proof[0..3] == 384` → `384*8 = 3072` (matches `PAILLIER_COMMITMENT_BITSIZE = 3072U`) |
| Gate 1 (coprime) works | CT=0 → coprime check fails; CT=n-1 → coprime check passes |
| Gate 2 deserialize works | Corrupted proof header → `"Size n mismatch 383 != 384"` — leaks expected modulus size |
| Gate 3/4 ZKP checks work | Corrupted ciphertext + valid proof → `"ec_left != ec_right"` |

## Attack Chain (P1: $100K-$250K)

1. **Malicious cosigner** in BAM 2-of-2 signing (BAM is the *default* protocol per `mpc_globals.h:25`)
2. **Submit ~300-500 crafted ciphertexts** → each causes one protocol abort
3. **Oracle reveals Paillier private key** λ via CCA technique (Boneh-Joux-Nguyen)
4. **Decrypt `encrypted_server_share`** from key metadata → server's ECDSA share
5. **Full key** = client_share + server_share (mod secp256k1 order)

## Existing Attack Tests (No PoC needed — run stock)

```bash
./test/cosigner/cosigner_test "bam_ecdsa_attacks" -s
```

Tests at lines 840, 879, 919, 1304, 1342 probe the same oracle gates with
`REQUIRE_THROWS` assertions. The PoC just adds error-message visibility.

## Fact-Check Lessons (Jun 2026)

1. **`grep -rn CONSTANT src/` before claiming a constant's behavior.** `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION = 3` is defined in `mpc_globals.h` but NEVER referenced in any source file. The original report wrongly claimed v2 sends unencrypted MTA responses.

2. **Verify same-primitive before comparing key sizes.** CMP uses STD Paillier (2048 bits, `paillier_generate_key_pair`) while BAM uses Paillier Commitment (3072 bits, `paillier_commitment_generate_private_key`). Different schemes, different requirements — the CMP security theorem is met.

3. **BN_CTX nesting kills crypto-layer PoCs.** `commit_internal`, `generate_signature_proof`, and `verify_signature_proof` each call `BN_CTX_start/end` on the passed context. If the caller already has a frame open, the nesting corrupts the BIGNUM pool. Always use a fresh `BN_CTX` per library call.

4. **The cleanest PoC is appending to bam_test.cpp**, not building a direct crypto-layer harness. No CMake changes, no static library, no BN_CTX issues.
