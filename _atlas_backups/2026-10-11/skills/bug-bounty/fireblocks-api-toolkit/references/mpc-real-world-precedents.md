# Real-World MPC Wallet Exploit Precedents

A reference for understanding how cryptographic MPC vulnerabilities have been exploited in production
wallet systems, and what lessons apply to Fireblocks mpc-lib findings.

## Timeline of Major MPC Wallet Attacks

| Year | Name | CVE | Target Protocol | Attack Surface | Wallets Affected | PoC |
|------|------|-----|----------------|----------------|-------------------|-----|
| 2020 | "Multiple Bugs in Multi-Party Computation" | — | GG18, GG20, Lindell17 | Protocol abort oracle, malicious Paillier modulus, missing ZKP verification | Binance Custody, Coinbase, multiple others | Aumasson & Shlomovits, Black Hat USA 2020 |
| 2023 | **BitForge** | CVE-2023-33241 (GG18/GG20), CVE-2023-33242 (Lindell17) | GG18, GG20, Lindell17, BitGo TSS | 6ix1een CRT: **16 small primes × 1 large prime** in Paillier modulus; single-bit abort leak | 15+ providers incl. Binance Custody, ZenGo, BitGo, Coinbase WaaS | `fireblocks-labs/safeheron-gg20-exploit-poc` |
| 2023 | **Practical Key-Extraction Attacks** (CCS 2024) | CVE-2023-33241, CVE-2023-33242 | GG18, GG20, Lindell17, BitGo TSS | Unified taxonomy: (1) malicious modulus + CRT (2) abort oracle (3) missing Paillier well-formedness proof | Coinbase, Binance, ZenGo, BitGo, ING Bank | `fireblocks-labs/mpc-ecdsa-attacks-23` |
| 2024 | TSSHOCK | — | Bitcoin-native threshold schemes | Combined approach from BitForge family | Multiple Bitcoin custodians | Black Hat USA 2024 |

## The BitGo Case: Why Two Missing Proofs Matter

BitGo is the most instructive case for mpc-lib analysis because it demonstrates how **missing proofs compound**.

### Round 1 — Range proof missing (exploited)
- BitGo TSS had no range proof on Paillier ciphertexts
- Attacker submitted ciphertexts with values > 2^2000 (the range bound)
- Server decrypted and the overflow leaked key bits
- **BitGo patched**: added range proof

### Round 2 — Paillier well-formedness proof still missing (exploited AGAIN)
- The range proof fixed the overflow attack
- But BitGo had no Paillier well-formedness ZKP (proving n is biprime with no small factors)
- Attacker constructed N = p×q where p is small (16 bits), leaking the key via CRT
- **Exploited twice** for the same class of vulnerability

**Lesson**: Fixing one missing ZKP doesn't close the class — attackers pivot to the next missing check. mpc-lib has this same pattern: weakened range proofs (5A), weak Blum small-prime check (7C), absent refresh proofs (1C), and version-gated strength (7B) all provide separate attack paths.

## The 6ix1een CRT Technique (CVE-2023-33241)

The core attack that broke 15+ wallet providers:

```
1. Attacker constructs N = p₁ × p₂ × ... × p₁₆ × q
   where each pᵢ is a 16-bit prime (≈2^16, easily factored)
   and q is a large prime matching the expected bit-length (~2048 bits)

2. During signing, submit ciphertext Cⱼ = Enc_N(N/pⱼ)
   N/pⱼ is a massive number (≈2^2032), which would fail a range proof,
   but if range proof is missing → passes verification

3. When the challenge e ≡ 0 (mod pⱼ) in the ZKP verification:
   Cⱼ^e ≡ C₀^e (mod N²) — the ciphertext for N/pⱼ behaves like
   ciphertext for 0 mod pⱼ. This is undetectable by the verifier.

4. Each signing session leaks x mod pⱼ (one congruence per prime)
   
5. After 16 sessions: CRT reconstructs x mod (∏pᵢ) → full key

Only 16 protocol aborts needed. Dramatically under P1 threshold of <1000.
```

### Why This DOESN'T Work on mpc-lib CMP (critical)

The Blum ZKP in mpc-lib's CMP checks the **quartic residuosity** of n, which requires n to be a valid Blum integer (exactly 2 factors p ≡ 3 mod 8 and q ≡ 7 mod 8). A 17-factor modulus fails this check.

Additionally, when PoC-tested with 2-factor unbalanced key (256-bit + 1792-bit):
- `gcd(λ, n) = 1 → NO` (they share factors)
- This prevents Paillier key operations and Blum ZKP generation
- The gcd(λ,n) check in key generation naturally blocks severely unbalanced keys

**Conclusion for mpc-lib**: The CRT multi-prime attack is blocked. The real attack vector is **version downgrade amplifying the Paillier oracle** (Finding 7B + Finding 004).

## References

- Makriyannis, Yomtov, Galansky. "Practical Key-Extraction Attacks in Leading MPC Wallets." ACM CCS 2024. https://eprint.iacr.org/2023/1234
- Fireblocks. "GG18 and GG20 Paillier Key Vulnerability [CVE-2023-33241]: Technical Report." https://www.fireblocks.com/blog/gg18-and-gg20-paillier-key-vulnerability-technical-report
- Fireblocks. "Lindell17 Abort Vulnerability [CVE-2023-33242]: Technical Report." https://www.fireblocks.com/blog/lindell17-abort-vulnerability-technical-report
- Fireblocks. "BitGo Wallet Zero Proof Vulnerability: Technical Report." https://www.fireblocks.com/blog/bitgo-wallet-zero-proof-vulnerability
- Fireblocks PoC repos: https://github.com/fireblocks-labs/bitgo-tss-exploit-poc, https://github.com/fireblocks-labs/safeheron-gg20-exploit-poc, https://github.com/fireblocks-labs/zengo-lindell17-exploit-poc, https://github.com/fireblocks-labs/mpc-ecdsa-attacks-23
- Aumasson & Shlomovits. "Multiple Bugs in Multi-Party Computation." Black Hat USA 2020.
- SlowMist. "Common-Cryptographic-Risks-in-Blockchain-Applications." https://github.com/slowmist/Common-Cryptographic-Risks-in-Blockchain-Applications
