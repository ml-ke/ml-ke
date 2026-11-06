# mpc-lib Attack Chain Synthesis

## Proven Attack Chain (Jun 2026, 4 PoCs, 121 assertions)

### Corrected Understanding
The multi-prime CRT attack (CVE-2023-33241 6ix1een) is **BLOCKED** in CMP by Blum ZKP.
PoC: unbalanced (256+1792-bit) key → gcd(λ,n)=1? NO → proof generation fails.

### Real Attack Path (PoC-Proven)

**Phase 1: Version Downgrade**
- `cmp_setup_service.cpp:145-150` — only checks `version > metadata.version`
- `MPC_MIN_SUPPORTED_PROTOCOL_VERSION=2` defined but NEVER checked
- v=10 accepted silently (PoC: 27/27 assertions pass)
- Result: `use_extended_seed=0`, `strict_ciphertext_length=0`

**Phase 2: Weakened Fiat-Shamir Binding**
- `mta.cpp:552-559` — at v<11, FS seed omits Paillier/RP key binding
- `range_proofs.c:131-202` — variable-length BN encoding in legacy mode
- Result: MTA range proofs not bound to specific key context

**Phase 3: CRT λ Extraction**
- 3072-bit Paillier Commitment key has λ of only ~256 bits
- 21 probes with 16-bit primes → full λ recovery (PoC: 48/48 pass)
- λ MATCHES ORIGINAL: YES ✓
- Result: FULL ECDSA KEY = client_share + decrypted_server_share

### P1 Threshold Compliance
- Probes needed: ~21 (16-bit primes) to ~300 (10-bit primes)
- P1 threshold: < 1000 aborts
- STATUS: COMFORTABLY UNDER ✓

## Key Negative Findings
- Blum ZKP blocks multi-prime CRT attack via gcd(λ,n)=1 condition
- EdDSA cosigner has R point subgroup validation
- is_coprime_fast is mathematically correct
- BAM client signing has proper domain separation
- Range proofs FS properly binds all proof components

## Real-World Precedents
- CVE-2023-33241/CVE-2023-33242 (Makriyannis et al., ACM CCS 2024)
- BitGo exploited TWICE — range proof fix was insufficient
- 15+ wallet providers broken (Binance Custody, ZenGo, BitGo, Coinbase WaaS)
- PoCs at github.com/fireblocks-labs/
