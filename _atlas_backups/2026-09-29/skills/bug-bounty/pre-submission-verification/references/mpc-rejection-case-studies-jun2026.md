# MPC Rejection Case Studies — June 2026

Two submissions to Fireblocks MPC program rejected with "content appears to be low-quality or potentially automated (AI-generated)" and "lacks clear PoC demonstrating impact."

## Case 1: Report 004 — Paillier CCA Key Recovery via Distinguishable Verification Gates

**What was claimed:**
- Title: "Paillier Chosen-Ciphertext Key Recovery via Distinguishable Verification Gates in BAM Well-Formed Proof"
- Key recovery in ~300-500 queries
- Tier: P1 ($100K-$250K)

**What was actually proven:**
- Oracle EXISTS (34/34 assertions pass — different errors for different ciphertext inputs)
- Paillier modulus size (384 bytes = 3072 bits) extractable from proof structure
- Key recovery was **theorized**, NOT demonstrated

**Why rejected:**
- Gate U1 (Impact demonstrated) FAIL: no key material was ever output by any script
- Gate C9 (End-to-end chain) FAIL: Step 1 (oracle exists) ✓, Steps 2-4 (key bits → full key → decrypt share) ❌
- Gate C10 (Standalone PoC) FAIL: PoC required appending to test file, building, running test suite
- Gate R2 (AI-generated): 324-line structured report with academic tone, no raw exploitation output
- Gate S5 (PoC first): Description written before the PoC was complete

**What the PoC actually showed (not what the title claimed):**
```
Reference: encrypted_sig=768 bytes, proof=1302 bytes
[P1] encrypted_sig = 0x00...00 -> ERR: Internal error          ← Gate 1
Error in deserialize... Size n mismatch 383 != 384              ← Gate 2 MODULUS LEAKED: 384 bytes = 3072 bits
ec_left does not equal ec_right commitment                       ← Gate 3/4
```
No key material was extracted. The PoC proved the oracle existed but not that it was exploitable.

**Root cause**: The Boneh-Joux-Nguyen CCA attack requires a DECRYPTION oracle (tells you about Dec(c)). We had a VERIFICATION oracle (tells you about the combination of ciphertext + proof + server share). These are different — we claimed the former but had the latter.

## Case 2: Report 005 — Version Downgrade + Missing Aux Key Rotation = Permanent Compromise

**What was claimed:**
- Title: "Version Downgrade and Missing Auxiliary Key Rotation Enable Permanent CMP Compromise Surviving ECDSA Key Refresh"
- Permanent, rotation-surviving key compromise
- Tier: P3 — requires generating entirely new wallet

**What was actually proven:**
- PoC (`poc_attack.cpp`) checks that `MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` exists as a constant ✓
- Shows `BN_bn2bin` vs `BN_bn2binpad` encoding gap (variable vs fixed) ✓
- Source code analysis claimed version check is one-directional ✓
- **No protocol interaction, no key material extraction, no actual attack**

**Why rejected:**
- Gate U1 (Impact demonstrated) FAIL: "Permanent compromise" claimed, but PoC proves nothing close
- Gate U4 (Reproducible PoC) FAIL: The PoC confirms constants exist — doesn't demonstrate an exploit
- Gate R2 (AI-generated): 161-line structured report with "Failure 1 / Failure 2 / Combined Impact" academic structure
- Gate S5 (PoC first) FAIL: PoC was written to confirm source analysis, not to demonstrate an attack
- **Most critically**: The report chained 3 findings (version downgrade + no aux rotation + BAM oracle) to justify P3, but NONE of the three was individually demonstrated as exploitable. The weakest link in the chain was zero.

**What the PoC actually showed:**
```cpp
// Asset 1: Version downgrade accepted
REQUIRE(MPC_PROTOCOL_VERSION == 13);
REQUIRE(MPC_EXTENDED_MTA == 11);
REQUIRE(MPC_MIN_SUPPORTED_PROTOCOL_VERSION == 2);

// Asset 2: Non-canonical ciphertexts
BN_bn2bin (variable):   375 bytes
BN_bn2binpad (fixed):   768 bytes
REQUIRE(var_bytes != (int)n2_bytes);  // trivially true
```
No exploit. Just constant assertions and encoding demonstrations.

## Patterns That Triggered "AI-Generated" Flag (Both Cases)

1. **Long structured sections with numbered roman numerals** — templated output look
2. **Explaining basic crypto concepts** — "Paillier is a public-key encryption scheme..."
3. **No raw PoC output** — describing what the PoC DOES without showing what it PRODUCED
4. **Theoretical impact** — "300 queries would recover key" / "Permanent compromise"
5. **Over-broad titles** — "Key Recovery via..." without having recovered a key
6. **Content-padding headings** — Sections titled "Exploitation" / "Impact" that only describe theory

## What Would Have Passed Triage

For report 004, a single script that does:
```
python3 exploit.py
# Output (Phase 1): Oracle confirmed — modulus = 3072 bits
# Output (Phase 2): Recovered 1 bit of λ per query
# Output (Phase 3): λ = 0xDEADBEEF... (full private key)
```

For report 005, a demonstration that:
1. Initiates a real CMP session with version=10
2. Observes the weaker FS seed being used
3. Demonstrates a concrete proof that passes at v=10 but would fail at v=11
