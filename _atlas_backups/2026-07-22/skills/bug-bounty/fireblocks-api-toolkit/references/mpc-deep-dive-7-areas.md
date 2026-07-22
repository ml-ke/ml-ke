# Fireblocks MPC — 7 Unexplored Attack Surfaces (June 2026)

All findings from deep-dive source audit of `github.com/fireblocks/mpc-lib` at `~/Dev/mpc-lib/`.
Each area analyzed by reading all relevant source files and tracing protocol flows.

---

## 1. CMP Offline Refresh Service — Key Rotation

**Files:** `cmp_offline_refresh_service.cpp`, `cmp_key_persistency.h`, `prf.h`

### 1A — No Anti-Replay Nonce/Epoch Counter (P3)
Session identified solely by caller-supplied `request_id` (line 109). No persistent epoch counter in key metadata struct. Key shares have no binding to refresh epoch. Replay of old seeds possible if plaintext recoverable, though seeds transiently stored and deleted at line 200.

### 1B — No Sender Authentication on Encrypted Seeds (P3/4, actionable)
`encrypt_for_player` at line 72 uses recipient's public key for confidentiality only — no signature/MAC. No `request_id` binding in encryption AAD. Encrypted blobs replayable across sessions undetectably. Decryption at line 126 has no mechanism to verify sender identity.

**Contrast:** Full setup protocol uses commitments, decommitments, Schnorr proofs, Paillier Blum ZKP, Ring Pedersen ZKP — all bound to session. Refresh uses NONE of these.

### 1C — No ZK Proofs at All (P4)
Zero ZK proofs in entire refresh flow. No proof seeds honestly generated, no proof encrypted seed corresponds to claimed sender, no proof PRF output computed correctly. Coalition of malicious players could coordinate seed choices.

### 1D — Non-Atomic Two-Phase Commit With Race Windows (P4)
Window between `transform_preprocessed_data_and_store_temporary` (line 167) and `commit` (line 214) allows concurrent signing session to read inconsistent state. `cancel_refresh_key` (line 232) silently swallows all exceptions — partial failures leave system inconsistent.

---

## 2. MTA Batch Verifier — batch_response_verifier

**Files:** `mta.h:93-136`, `mta.cpp:886-1342`

### 2A — Count Validation External, Verifier Trusts Caller (P3/4)
Verifier tracks zero state about expected vs actual response count. Count check in calling code (`cmp_ecdsa_offline_signing_service.cpp:198-202`) is the only defense. If response array expands between check and loop, accumulators incorporate extra proofs.

### 2B — Weak 8-Bit Random Gamma in Paillier Batch (P5)
```cpp
uint8_t random[10];  // 10 bytes for 5 positions
BN_set_word(gamma, random[i * 2]);  // gamma = 0-255 (8 bits)
```
Each Paillier position: only 8 bits soundness (vs 40 bits for Ring-Pedersen). Total Paillier batch soundness: 2^(-40), below 2^(-112) for 3072-bit Paillier.

### 2C — No Accumulator Reset After verify() (P5)
Accumulators never reset after verify(). If same verifier reused (doesn't happen currently, nothing prevents it), state from first batch incorporated into second.

### 2D — 3 Probabilistic-Only Checks in Batch vs Single Verifier (Design)
| Check | Single | Batch |
|-------|--------|-------|
| Paillier C^z1×enc(z2,w)==A×D^e | ✓ Exact | ✗ Probabilistic (2⁻⁴⁰) |
| RP s^z1×t^z3==E×S^e | ✓ Exact | ✗ Probabilistic (2⁻²⁰⁰) |
| RP s^z2×t^z4==F×T^e | ✓ Exact | ✗ Probabilistic (2⁻²⁰⁰) |
| Schnorr g^z1==Bx×X^e | ✓ Exact | ✓ Exact (per-element) |

If verify() never called (crash after process() loop), Schnorr proof is only check that survives per-element.

---

## 3. is_coprime_fast — Gate 1 in BAM Oracle

**File:** `algebra_utils.c:325-382`

### 3A — Timing Side-Channel (Medium)
Classic Euclidean while-loop runs O(log(min(a,b))) iterations — data-dependent. Warning comment: "This function doesn't run in constant time". No BN_FLG_CONSTTIME usage. Amplifies oracle distinguishability.

### 3B — No Negative Input Validation (Low)
`BN_is_zero` checks for zero but no negative check. `BN_cmp(-5, 7)` swaps to positive, but `BN_mod` with negative divisor could behave unexpectedly. Callers pass unsigned byte arrays so unlikely in practice.

### 3C — Error Code Conflated With Legitimate Result (Low)
Returns -1 for zero input, but callers check `!= 1` — treating -1 same as 0 (not coprime). Works correctly (input rejected) but conflates error with legitimate mathematical result (gcd(0,n)=n≠1, so returning 0 is also correct).

### 3D — Gate 1 Trivially Bypassable by Design
Any value coprime to n passes: n+1, 2n-1, r^n mod n² all pass. Gate 1 is cheap pre-filter blocking trivially invalid inputs (0, n, multiples of n). Cryptographic security from Gates 2-4.

---

## 4. BAM Client-Side compute_partial_signature

**Files:** `bam_ecdsa_cosigner_client.cpp:542-747`, `bam_well_formed_proof.cpp:163-246`

### 4A — Well-Formed Proof Doesn't Bind to Message Hash (Design/Med)
`compute_e` function (bam_well_formed_proof.cpp:163-246) hashes signature_aad, EC/Paillier params, encrypted S, encrypted_share, r_server, U, V, D — but NOT message hash. `signature_aad` contains SALT + key_id + tx_id + server_id + client_id — no message.

Protocol relies on final ECDSA verification (server.cpp:915-919, LOG_FATAL on failure) as ultimate integrity check. A bug in ECDSA verification would render entire mechanism moot.

### 4B — Different Message Possible in ZK Proof (Caught by ECDSA Verify)
Client could compute u,v with msg_evil, generate valid well-formed proof for S=Enc(u)*Enc(share)^v. Server decrypts (encodes msg_evil), computes r from msg_protocol, fails ECDSA check. Prevented but concentrates security in one final check.

### 4C — Good Domain Separation ✅
Setup AAD = "BAM ECDSA Setup AAD" + setup_id. Signature AAD = "BAM ECDSA Signature AAD" + key_id + tx_id. Hash shift = SHA256(tx_id || public_key || client_R || common_R || message). Separate salts prevent cross-context binding.

---

## 5. Range Proofs — range_proofs.c Fiat-Shamir

**Files:** `range_proofs.c`, `cmp_setup_service.cpp`, `mta.cpp`

### 5A — Variable-Length BN Serialization in FS Hash (Legacy Mode, Medium)
**Two modes:**
- **Legacy (`use_extended_seed=0`):** `BN_bn2bin()` strips leading zeros → `0x00000042` hashes as `0x42` (1 byte)
- **Extended (`use_extended_seed=1`, v11+):** `BN_bn2binpad()` → fixed-length zero-padded

Same mathematical value produces different FS challenge if leading zeros differ. Adversary cannot freely choose leading zeros (commitments tied to random witnesses), so practical exploit low — but deviation from ideal fixed-length binding.

### 5B — Cross-Version Seed Inconsistency (Setup vs MTA, Low)
`use_extended_seed` from version only in `cmp_setup_service.cpp:231`. In `mta.cpp:658-672`, both exponent and DH proofs hardcoded with `use_extended_seed=0`. Signing services also hardcode 0. Only setup uses version-dependent seed. Cross-protocol FS collision possible if only one side updates.

### 5C — Range Bounds Use Byte-Count Not Bit-Precision (Low)
```cpp
if ((size_t)BN_num_bytes(zkpok.z1) > sizeof(...) + ZKPOK_EPSILON_SIZE)
```
Leading zero byte in MSB makes bound loose. Value just over 256 bits with zero MSB passes.

### 5D — Soundness: 2^(-256) EC, 2^(-112) Paillier ✅ Acceptable

### 5E — Fiat-Shamir Properly Binds All Proof Components ✅
```
Salt || AAD || ciphertext || X || S || D || Y || T
```
All four commitments (S, D, T, Y), ciphertext, and public point X in the hash. DH variant adds A, B, Y' points.

---

## 6. EdDSA Asymmetric Cosigner

**Files:** `asymmetric_eddsa_cosigner_client.cpp`, `asymmetric_eddsa_cosigner_server.cpp`, `ed25519_algebra.c`

### 6A — No ZK Proofs for Nonce Knowledge (Medium/Design)
Unlike BAM/CMP ECDSA (Paillier encryption, range proofs, Schnorr proofs, DF proofs), asymmetric EdDSA uses zero cryptographic proofs. Only validation: SHA256 commitment comparison + verify_client_s equation. No proof client knows DLOG of R.

**Mitigation:** Signature equation serves as implicit proof. Simpler protocol = smaller attack surface overall.

### 6B — Random Nonces, Not RFC 8032 Deterministic (Low)
Both client (line 75) and server (line 154) use `_ctx->rand()` — cryptographically random, not RFC 8032 deterministic SHA-512(prefix || message). Standard for threshold EdDSA. Security relies entirely on RNG quality.

### 6C — R Point Subgroup Validation Performed ✅
`ed25519_is_valid_point` (ed25519_algebra.c:102-113):
```c
EIGHT_INVERSE * P → P1; EIGHT * P1 → P2; memcmp(P, P2) == 0
```
Verifies point in prime-order subgroup (no torsion). Called by `ed25519_algebra_add_points` on both inputs. Likely added after previous audit.

### 6D — Version Parameter Ignored, Version Field Misleading (Low)
`broadcast_si` (server.cpp:304): `(void)version;` — accepted, discarded. Metadata hardcoded to MPC_BAM_ECDSA (13) even for EdDSA. Field unused in sig verification, stored for logging only.

### 6E — No Oracle-like Vulnerability ✅
No Paillier/encryption in EdDSA protocol. Pure EC operations. verify_client_s doesn't leak server key info.

---

## 7. Setup Service Key Validation

**Files:** `cmp_setup_service.cpp`, `paillier_zkp.c`, `ring_pedersen.c`

### 7A — No Cross-Party Paillier Modulus Uniqueness Check (P3/4, actionable)
`verify_and_load_setup_decommitments()` (line 760) deserializes each player's Paillier key independently. **No comparison across parties.** Malicious party can:
1. Eavesdrop on another party's commitment/decommitment
2. Submit same Paillier public key as their own
3. All ZKP checks pass (Blum ZKP, large factors proof, RP ZKP are self-referential)
4. If attacker later compromises original key's factorization, can decrypt that party's MTA ciphertexts

Same gap for Ring Pedersen parameters.

### 7B — No Minimum Version Check — Version Downgrade Weakens FS Seed (P3/4)
```cpp
if (version > MPC_PROTOCOL_VERSION)  // Only UPPER bound
```
No lower-bound check. Line 231: `use_extended_seed = (version >= MPC_EXTENDED_MTA)` — version < 11 uses weak legacy FS seed (variable-length BN, Finding 5A).

Attack chain: claim version 10 → use_extended_seed=0 → weak FS → forked proof → Paillier key recovery → MTA decryption → ECDSA key recovery.

### 7C — use_all_nth_roots=1 Weakens Small-Prime Check (P4)
Setup calls `paillier_verify_paillier_blum_zkp(i->second.paillier.get(), 1, ...)` with second param = 1. When 1: only checks n mod 3 != 0 (paillier_zkp.c:1530-1542). When 0: checks coprimality with product of ALL primes under 65536.

Large factors proof (create_secret line 304) partially mitigates via range bounds on z1/z2.

### 7D — Blum ZKP Uses 64 Iterations Not 80 (P5)
Verify loop uses `PAILLIER_BLUM_STATISTICAL_SECURITY_MINIMAL_REQUIRED` (64) not `PAILLIER_BLUM_STATISTICAL_SECURITY` (80). Named constant 80 defined but unused in verifier. Soundness ~2^(-64) when could be ~2^(-80).

### 7E — Ring Pedersen ZKP Misses Quadratic Residue Check (Low)
ZKP proves knowledge of λ where s = t^λ mod n. Does NOT verify s,t are quadratic residues mod n (would prove perfect hiding in Z_n^*). Minor weakening of hiding property.

### 7F — No Check n is Product of Exactly 2 Primes (Low)
`BN_is_prime_fasttest_ex` (paillier_zkp.c:1479) only verifies n is **not** prime. A composite with 3+ factors (p*q*r) or prime power (p²) could theoretically pass the Blum ZKP x^4 structure implicitly assumes n=p*q where p≡3 mod 8, q≡7 mod 8.

---

## Most Exploitable Primary Attack Paths

### Path A: Setup Version Downgrade → Weak FS → Key Recovery (P1-P2)
```
1. Malicious party claims version < 11 during setup → no lower-bound check (7B)
2. use_extended_seed=0 → FS challenge uses variable-length BN (5A)
3. Weakened proof binding in large factors proof
4. Combined with no cross-party key uniqueness (7A): attacker submits another's key
5. Forged proof → Paillier key recovery → decrypt server share → full ECDSA key
```
**Win condition:** < 1000 aborts = P1 ($100K-$250K)

### Path B: Refresh Seed Replay (P3-P4)
```
1. Obtain encrypted seed blobs from past refresh (persistency compromise)
2. Replay in new session → no sender auth, no request_id binding (1B)
3. Roll key back to known state → no epoch counter (1A)
4. Sign with previously-known key share
```
**Requires:** Persistency compromise, but replay itself is trivial once obtained

### Path C: Paillier Batch Weak Gamma → CCA (P4-P5)
```
1. batch_response_verifier uses 8-bit gamma (2B) → 2^-40 soundness
2. Requires auxiliary vulnerability for full CCA exploitation
```

### Defensive Observations ✅
- EdDSA R point subgroup validation (6C)
- is_coprime_fast mathematically correct for Paillier Z_n² space
- Domain separation between setup/signing AADs in BAM (4C)
- Range proofs: proper Fiat-Shamir binding of all proof components
- Soundness: 2^(-256) EC, 2^(-112) Paillier — sufficient
