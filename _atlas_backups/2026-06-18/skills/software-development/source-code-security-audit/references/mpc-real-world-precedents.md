# Real-World MPC Precedents (for MCP Crypto Audit)

## The BitGo Double-Exploit (2023)

**Source**: Fireblocks research team — `fireblocks-labs/bitgo-tss-exploit-poc`

BitGo implemented a threshold ECDSA scheme based on the GG18/GG20 protocol. The initial vulnerability (CVE-2023-33241 variant) was a **missing range proof** — malicious parties could submit Paillier ciphertexts containing values outside the expected range. Fireblocks disclosed it, BitGo patched by adding the range proof.

**The second exploit**: The range proof was only one of two required proofs. The **Paillier well-formedness proof** was still missing. This meant:
- A party could submit a malicious Paillier modulus n with small factors
- The range proof checked the size of encrypted values against n (passes)
- But without a Blum ZKP, nobody verified n was actually a valid biprime
- Result: attacker factors n → recovers other parties' key shares

**Lesson**: Two independent missing proofs = exploited twice. Single-fix patches are insufficient when the vulnerability has multiple root causes.

**Cross-reference**: CVE-2023-33241 (GG18/GG20 Paillier key vulnerability), `fireblocks-labs/safeheron-gg20-exploit-poc`

## The 6ix1een CRT Attack (2023)

**Source**: Makriyannis, Yomtov, Galansky — "Practical Key-Extraction Attacks in Leading MPC Wallets" (ACM CCS 2024)

Attack on GG18/GG20 implementations missing Blum ZKP:

1. Attacker constructs N = p₁×p₂×...×p₁₆×q (16 small primes of 16 bits each + 1 large prime)
2. During key setup, submits N as Paillier public key
3. During signing, submits crafted ciphertexts Cⱼ = Enc(N/pⱼ, r) for each small prime pⱼ
4. Each ciphertext exploits the fact that when e ≡ 0 (mod pⱼ), the verification equation is undetectable
5. After 16 signing sessions, attacker knows x mod pⱼ for all 16 primes
6. CRT reconstructs x mod (p₁×...×p₁₆) → since ∏pᵢ > 2²⁵⁶, this is the FULL SECRET KEY

**15+ wallet providers affected**: Binance Custody, ZenGo, BitGo, Coinbase WaaS, Safeheron, and others.

**Full PoC repos**: `fireblocks-labs/mpc-ecdsa-attacks-23`

## The Lindell17 Abort Oracle (CVE-2023-33242)

**Source**: Fireblocks research team — `fireblocks-labs/zengo-lindell17-exploit-poc`

Lindell17's 2-party ECDSA protocol leaks **1 bit per protocol abort**. The attacker:
1. Initiates signing → submits crafted values → protocol aborts
2. The abort reason distinguishes two cases, revealing 1 bit of the honest party's secret key
3. ~256 signing sessions → full key recovery

**Impact**: P1 (key recovery with <1000 aborts). 256 << 1000.

**Defense**: The CMP protocol (Canetti et al., eprint 2020/492) is UC-secure with identifiable abort — honest parties can identify the cheater and continue.

## TSSHOCK (2024)

**Source**: Duy Hieu Nguyen et al., Black Hat USA 2024

Combined attack on Bitcoin-native threshold signing schemes. Extended the 6ix1een technique to Schnorr-based threshold signing (FROST, ROAST). Demonstrated that the same missing-proof patterns apply across different signing algorithms.

## Version Downgrade as Attack Vector (Unique to mpc-lib)

**Source**: Fireblocks mpc-lib (github.com/fireblocks/mpc-lib), found Jun 2026

The Fiat-Shamir challenge in MTA range proofs uses a **version-dependent** seed computation:

- **v ≥ 11** (MPC_EXTENDED_MTA): `e = SHA256(seed || aad || ct || proof || ring_pedersen_n || prover_paillier_n || verifier_paillier_n)`
- **v < 11**: `e = SHA256(seed || aad || ct || proof)` — NO KEY BINDING

The version is a claimable uint32_t with **no lower-bound check** (`MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` is defined but NEVER referenced). Claiming v=10 instead of v=13 removes the key binding from all MTA range proofs.

**Unique to this codebase**: No other threshold ECDSA implementation gates ZKP strength on a claimable version number.
