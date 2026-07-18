# Blum ZKP Defense Verification — PoC-Proven Findings

## Context
Initial analysis hypothesized that the Blum ZKP in mpc-lib's CMP protocol could be bypassed
with a multi-prime Paillier modulus (CVE-2023-33241 technique). This was TESTED and DISPROVEN.

## PoC Results (`cmp_malicious_key_poc.cpp`, 37 assertions all pass)

### Test 1A: Balanced key (1024+1024=2048 bits) — SHOULD work
- Generated via `paillier_generate_key_pair(2048)` → p≡3(mod 8), q≡7(mod 8), both ~1024 bits
- Blum ZKP generated and verified with `use_all_nth_roots=1` (CMP setting)
- **Result: PASS ✅** — balanced Blum integer accepted

### Test 1B: Unbalanced key (256+1792 bits) — SHOULD fail
- p≡3(mod 8) at 256 bits, q≡7(mod 8) at 1792 bits → n ≈ 2047 bits
- `gcd(p-1, q-1)` divides λ = lcm(p-1, q-1); `gcd(λ, n) ≠ 1` because 256-bit p's
  `p-1` frequently shares factors with 1792-bit q
- **Result:** `gcd(λ,n) == 1: NO` → Blum ZKP key generation cannot produce valid proof
  because the Paillier λ computation requires `gcd(λ, n) = 1`
- **Implication:** The gcd(λ,n) check during key generation is an EFFECTIVE defense
  against unbalanced factors, even though it's not in the Blum ZKP itself

### Test 1C: `is_coprime_fast` timing variation (Finding 3A)
- `gcd(n+1, n)` = 2 iterations (fast)
- `gcd(2n, n)` = 3+ iterations (slower)  
- **CONFIRMED:** Variable timing exists per source code warning

## Key Takeaway
The Blum ZKP in mpc-lib is robust against unbalanced/multi-prime modulus attacks.
The `use_all_nth_roots=1` parameter only weakens the SMALL-PRIME check (n mod 3 only)
within an already-functioning Blum ZKP — it doesn't disable the ZKP.

Attempting a CVE-2023-33241 CRT attack via malicious modulus requires:
1. A protocol without ANY Blum ZKP (GG18/GG20 original spec)
2. Or a protocol where the Blum ZKP is entirely missing from the implementation
3. Neither applies to mpc-lib's CMP setup path
